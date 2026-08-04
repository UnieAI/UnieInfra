# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""UnieInfra's adaptive admission policy as an out-of-tree vLLM scheduler.

Loaded with ``--scheduler-cls unieinfra_scheduler.vllm_scheduler.AdaptiveScheduler``.
Nothing under ``third_party/vllm`` or the installed vLLM tree is modified; the
class is resolved by ``SchedulerConfig.get_scheduler_cls``
(``vllm/config/scheduler.py:180-201``) and constructed by ``EngineCore``
(``vllm/v1/engine/core.py:137,150``) *after* the KV cache config is finalised,
which is precisely what makes a KV-derived cap possible from here.

Why it subclasses ``AsyncScheduler`` and not ``Scheduler``: vLLM enables
asynchronous scheduling by default on this configuration
(``vllm/config/vllm.py:1040``), and ``get_scheduler_cls`` warns that a custom
class deriving from ``Scheduler`` silently loses it
(``vllm/config/scheduler.py:192-198``). Deriving from ``Scheduler`` would have
made every A/B in this work an async-vs-sync comparison wearing a scheduling
policy's clothes.

Two overrides, both additive:

* ``__init__`` recomputes ``self.max_num_running_reqs`` from the real block
  pool. It can only lower it — see :func:`~unieinfra_scheduler.policy.derive_admission_cap`.
* ``schedule`` decides ``throttle_prefills`` itself instead of accepting the
  engine's value. vLLM already owns the *mechanism* for deferring prefill
  compute (``defer_prefills``, ``vllm/v1/core/sched/scheduler.py:436-438,467,801``)
  but wires it only to a fixed DP-balancing cadence that is unreachable at
  ``dp_size == 1``: the base engine's ``_should_throttle_prefills`` returns
  ``False`` unconditionally (``vllm/v1/engine/core.py:474-477``) and the real
  implementation lives on ``DPEngineCoreProc`` (``:1916-1923``). We reuse the
  mechanism and replace the fixed cadence with an SLO- and load-driven gate.
"""

from __future__ import annotations

import os
import time

from unieinfra_scheduler.policy import (
    AdmissionCap,
    PrefillGate,
    ResourceCost,
    blocks_for_lookahead,
    derive_admission_cap,
    gate_config_for,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

# Name the logger under "vllm." so it inherits vLLM's configured handler;
# a top-level "unieinfra_scheduler" logger would be silent in the server log,
# and this policy's decisions have to be evidence, not assumption.
logger = init_logger("vllm.unieinfra_scheduler")

__all__ = ["AdaptiveScheduler"]

_ENV_OBJECTIVE = "UNIEINFRA_SCHED_OBJECTIVE"
_ENV_RESERVE = "UNIEINFRA_SCHED_RESERVE_DIVISOR"
_ENV_HOLD = "UNIEINFRA_SCHED_HOLD_STEPS"
_ENV_TTFT = "UNIEINFRA_SCHED_TTFT_MS"
_ENV_CAP = "UNIEINFRA_SCHED_CAP"
_ENV_LOG_EVERY = "UNIEINFRA_SCHED_LOG_EVERY"

# Requests to scan when totalling waiting-queue prompt tokens. The gate only
# needs to know whether a prefill burst can be filled, so a bounded scan keeps
# the per-step cost O(1) even with a deep queue.
_WAITING_SCAN_LIMIT = 64


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


class AdaptiveScheduler(AsyncScheduler):
    """vLLM scheduler with UnieInfra's SLO-aware, KV-derived admission policy.

    Configuration is read from the environment so that no vLLM CLI surface has
    to change:

    ``UNIEINFRA_SCHED_OBJECTIVE``
        ``interactive`` (**default**) | ``balanced`` | ``throughput`` |
        ``static``. ``interactive`` disables the prefill gate entirely, leaving
        admission timing byte-identical to stock vLLM; the KV-derived cap still
        applies. The gate is opt-in because E36 measured it as a TTFT/TPOT dial,
        not a throughput win. ``static`` is the no-signal ablation.
    ``UNIEINFRA_SCHED_RESERVE_DIVISOR``
        Steady-state occupancy divisor for context-proportional KV groups
        (default ``4``).
    ``UNIEINFRA_SCHED_HOLD_STEPS`` / ``UNIEINFRA_SCHED_TTFT_MS``
        Override the objective's hold bound and TTFT budget.
    ``UNIEINFRA_SCHED_CAP``
        ``off`` skips the cap derivation and keeps vLLM's ``max_num_seqs``.
    ``UNIEINFRA_SCHED_LOG_EVERY``
        Log gate counters every N steps (``0``, the default, logs only at
        shutdown).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Default OFF. E36 measured that the gate does not recover throughput on
        # this workload -- it is a TTFT/TPOT dial, so it has to be asked for.
        # `interactive` leaves admission timing identical to stock vLLM; only
        # the (strictly lowering, safety-side) KV-derived cap still applies.
        objective = os.environ.get(_ENV_OBJECTIVE, "interactive").strip().lower()
        self.objective = objective

        self.admission_cap: AdmissionCap | None = None
        if os.environ.get(_ENV_CAP, "on").strip().lower() != "off":
            self.admission_cap = self._derive_cap()
            if self.admission_cap.cap < self.max_num_running_reqs:
                logger.info(
                    "unieinfra_scheduler: admission cap %d -> %d (binding: %s, %s)",
                    self.max_num_running_reqs,
                    self.admission_cap.cap,
                    self.admission_cap.binding,
                    self.admission_cap.detail,
                )
                self.max_num_running_reqs = self.admission_cap.cap
            else:
                logger.info(
                    "unieinfra_scheduler: derived cap %d >= configured %d, "
                    "keeping configured (binding: %s, %s)",
                    self.admission_cap.cap,
                    self.max_num_running_reqs,
                    self.admission_cap.binding,
                    self.admission_cap.detail,
                )

        gate_cfg = gate_config_for(
            objective,  # type: ignore[arg-type]
            max_num_scheduled_tokens=self.max_num_scheduled_tokens,
            hold_steps=_env_int(_ENV_HOLD, None),
            ttft_budget_ms=_env_float(_ENV_TTFT, None),
        )
        self.gate = PrefillGate(cfg=gate_cfg)
        self._log_every = _env_int(_ENV_LOG_EVERY, 0) or 0
        self._gate_steps = 0

        logger.info(
            "unieinfra_scheduler: objective=%s gate=%s max_num_running_reqs=%d "
            "max_num_scheduled_tokens=%d num_lookahead_tokens=%d",
            objective,
            gate_cfg,
            self.max_num_running_reqs,
            self.max_num_scheduled_tokens,
            self.num_lookahead_tokens,
        )

    # ---------------------------------------------------------------- cap ---

    def _derive_cap(self) -> AdmissionCap:
        """Ask each KV cache group what one request costs, then take the min.

        The per-group cost comes from the group's own ``KVCacheSpec``
        (``vllm/v1/kv_cache_interface.py``), which is the object the allocator
        itself sizes the pool with. That is deliberate: it means a
        sliding-window group reports its window-bounded cost, a ``MambaSpec``
        group reports its fixed state-chunk cost including
        ``num_speculative_blocks``, and neither has to be special-cased here the
        way the old TokenSpeed policy had to be, three times.

        Returns:
            The derived :class:`~unieinfra_scheduler.policy.AdmissionCap`.
        """
        from vllm.v1.kv_cache_interface import MambaSpec

        groups = list(self.kv_cache_config.kv_cache_groups)
        costs: list[ResourceCost] = []
        for idx, group in enumerate(groups):
            spec = group.kv_cache_spec
            page = int(spec.page_size_bytes)
            if page <= 0:
                continue
            full_bytes = int(spec.max_memory_usage_bytes(self.vllm_config))
            blocks = max(1, -(-full_bytes // page))
            is_state_pool = isinstance(spec, MambaSpec)
            if not is_state_pool:
                # Speculative decoding reserves slots past the request's real
                # length; a context-proportional group pays for them in blocks.
                blocks += blocks_for_lookahead(
                    self.num_lookahead_tokens, int(spec.block_size)
                )
            costs.append(
                ResourceCost(
                    name=f"g{idx}:{type(spec).__name__}",
                    blocks_per_request=blocks,
                    grows_with_context=not is_state_pool,
                )
            )

        max_cg = self.vllm_config.compilation_config.max_cudagraph_capture_size
        return derive_admission_cap(
            num_blocks=int(self.kv_cache_config.num_blocks),
            costs=costs,
            configured_max_num_seqs=int(self.max_num_running_reqs),
            max_cudagraph_capture_size=int(max_cg) if max_cg else None,
            reserve_divisor=int(_env_int(_ENV_RESERVE, 4) or 4),
        )

    # --------------------------------------------------------------- gate ---

    def _waiting_signals(self) -> tuple[int, float, bool]:
        """Summarise the waiting queue for the gate.

        Returns:
            ``(waiting_prompt_tokens, oldest_wait_ms, preempted_pending)``.
            The token total is over at most ``_WAITING_SCAN_LIMIT`` requests,
            which is enough to answer "can we fill a prefill burst".
        """
        now = time.time()
        tokens = 0
        oldest_ms = 0.0
        preempted = False
        scanned = 0
        for req in self.waiting:
            if req.status == RequestStatus.PREEMPTED:
                preempted = True
            tokens += req.num_tokens - req.num_computed_tokens
            age = (now - req.arrival_time) * 1000.0
            if age > oldest_ms:
                oldest_ms = age
            scanned += 1
            if scanned >= _WAITING_SCAN_LIMIT:
                break
        return tokens, oldest_ms, preempted

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Schedule one step, choosing ``throttle_prefills`` from the policy.

        Args:
            throttle_prefills: The engine's own request to defer prefills (DP
                cadence balancing). It is honoured as a floor — if the engine
                says throttle, we throttle — and the gate may additionally
                throttle on a step the engine did not ask about.

        Returns:
            The ``SchedulerOutput`` produced by vLLM's scheduler.
        """
        if throttle_prefills or not self.gate.cfg.enabled:
            # Engine-driven throttle wins; and with the gate disabled we are
            # byte-identical to stock admission timing.
            return super().schedule(throttle_prefills)

        waiting_tokens, oldest_wait_ms, preempted = self._waiting_signals()
        decision = self.gate.decide(
            num_running=len(self.running),
            num_waiting=len(self.waiting) + len(self.skipped_waiting),
            waiting_tokens=waiting_tokens,
            oldest_wait_ms=oldest_wait_ms,
            kv_usage=float(self.kv_cache_manager.usage),
            preempted_pending=preempted,
            admission_cap=self.max_num_running_reqs,
        )

        self._gate_steps += 1
        if self._log_every and self._gate_steps % self._log_every == 0:
            logger.info(
                "unieinfra_scheduler: step=%d running=%d waiting=%d counters=%s",
                self._gate_steps,
                len(self.running),
                len(self.waiting),
                self.gate.counters,
            )
        return super().schedule(decision.throttle)

    def shutdown(self) -> None:
        """Log the gate's decision histogram, then shut down normally."""
        if self.gate.counters:
            logger.info(
                "unieinfra_scheduler: objective=%s steps=%d counters=%s",
                self.objective,
                self._gate_steps,
                self.gate.counters,
            )
        super().shutdown()

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Count preemptions so the gate's safety release can be audited.

        Args:
            request: The request being evicted from the running queue.
            timestamp: Monotonic time of the scheduling step.
        """
        self.gate.counters["preemptions"] = self.gate.counters.get("preemptions", 0) + 1
        super()._preempt_request(request, timestamp)
