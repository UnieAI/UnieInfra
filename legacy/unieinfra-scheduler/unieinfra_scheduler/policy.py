# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""The admission policy itself — pure Python, no vLLM import.

Two decisions live here, and they are deliberately separated because they
answer different questions:

* :func:`derive_admission_cap` — *how many* requests may be resident at once.
  A **minimum over every binding resource**, computed from per-request block
  costs that the KV cache manager already knows exactly. This is the corrected
  port of the old TokenSpeed ``derive_adaptive_max_batch``
  (``python/tokenspeed/runtime/engine/event_loop.py:113``), whose defect was
  structural: it estimated a physical quantity the allocator already owned
  (``context_len // reserve_divisor``) and therefore had to be patched three
  separate times — once for the recurrent-state pool (E28/E31), once for
  sliding-window budgets, and once for speculative decoding.

* :class:`PrefillGate` — *when* to spend a step on prefill. This is the
  SLO-aware half. Admitting a prefill into a step costs every decoding request
  in that step a slower token, so the trade between TTFT and TPOT/throughput is
  made here, driven by the objective and by observed load.

Keeping both free of vLLM imports is what makes them unit-testable without a
GPU; :mod:`unieinfra_scheduler.vllm_scheduler` is the only module that knows
what a ``VllmConfig`` is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "Objective",
    "ResourceCost",
    "AdmissionCap",
    "GateConfig",
    "GateDecision",
    "PrefillGate",
    "derive_admission_cap",
    "gate_config_for",
]

Objective = Literal["interactive", "balanced", "throughput", "static"]


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class ResourceCost:
    """Per-request block cost of one KV cache group.

    Args:
        name: Human-readable group label, used only in the decision log.
        blocks_per_request: Blocks this group needs for a request that runs to
            the full model context, as reported by the group's own
            ``KVCacheSpec`` (so a sliding-window group reports its
            *window-bounded* cost, not the full-context cost).
        grows_with_context: ``True`` when the cost above is proportional to how
            far the request has actually decoded (plain full attention), so a
            steady-state occupancy estimate may discount it. ``False`` for
            fixed-size-per-request pools — recurrent/Mamba state chunks — where
            discounting is exactly the over-admission bug that produced the
            E28/E31 retractions, C=64 hangs and the C=128 device-side assert.
    """

    name: str
    blocks_per_request: int
    grows_with_context: bool


@dataclass(frozen=True)
class AdmissionCap:
    """Result of :func:`derive_admission_cap`.

    Args:
        cap: The running-request cap to install.
        binding: Name of the resource that produced ``cap``.
        detail: Per-resource caps considered, for the startup log.
    """

    cap: int
    binding: str
    detail: dict[str, int]


def derive_admission_cap(
    *,
    num_blocks: int,
    costs: list[ResourceCost],
    configured_max_num_seqs: int,
    max_cudagraph_capture_size: int | None,
    reserve_divisor: int = 4,
    floor: int = 1,
) -> AdmissionCap:
    """Derive the running-request cap as the minimum over every binding resource.

    The pool is shared across KV cache groups, so the joint bound is
    ``num_blocks / sum(per-group blocks per request)`` — the sum in the
    denominator *is* the min-over-resources for a shared allocator. Groups whose
    cost is fixed per request (recurrent state) are **not** discounted by
    ``reserve_divisor``; only context-proportional groups are.

    Args:
        num_blocks: Total blocks in the shared KV block pool.
        costs: One :class:`ResourceCost` per KV cache group.
        configured_max_num_seqs: The engine's configured cap. The derived value
            never exceeds it — see the note below on why raising is useless.
        max_cudagraph_capture_size: Largest captured decode batch, or ``None``
            when graphs are disabled. Admitting past it buys batch size at the
            price of dropping off the graph ladder, so it is treated as a cap.
        reserve_divisor: Assume a context-proportional group holds ``1 /
            reserve_divisor`` of its full-context cost in steady state. ``1``
            disables the discount (admit only what fits at full context).
        floor: Never derive a cap below this.

    Returns:
        An :class:`AdmissionCap` naming the binding resource.

    Note:
        This function can only **lower** the engine's configured cap, never
        raise it. On vLLM that is deliberate: the CUDA-graph ladder is
        finalised inside ``VllmConfig.__post_init__``
        (``vllm/config/vllm.py:1330``) as ``min(max_num_seqs * 2, 512)``
        (``vllm/config/vllm.py:1725``), long before any scheduler exists, so a
        scheduler-side increase would admit batches the ladder cannot replay.
        The old TokenSpeed policy raised its cap because *its* configured
        default (160) sat below the KV-admissible batch; vLLM's default on a
        >=70 GiB device is already 1024 (``vllm/engine/arg_utils.py:2432``),
        which is above what this pool can actually hold.
    """
    if not costs:
        return AdmissionCap(
            cap=max(floor, configured_max_num_seqs),
            binding="no-kv-cache",
            detail={},
        )

    divisor = max(1, reserve_divisor)
    est_blocks = 0
    detail: dict[str, int] = {}
    for cost in costs:
        per_req = cost.blocks_per_request
        if cost.grows_with_context:
            per_req = max(1, per_req // divisor)
        est_blocks += per_req
        # Per-group caps if this group were the only claimant, for the log:
        # the steady-state estimate we actually admit on, and the hard
        # full-context bound below which no discount is ever safe.
        detail[f"kv:{cost.name}"] = num_blocks // max(1, per_req)
        detail[f"kv:{cost.name}:full"] = num_blocks // max(1, cost.blocks_per_request)

    kv_cap = num_blocks // max(1, est_blocks)
    detail["kv:joint"] = kv_cap

    cap = kv_cap
    binding = "kv-pool"

    if max_cudagraph_capture_size is not None and max_cudagraph_capture_size > 0:
        detail["cudagraph"] = max_cudagraph_capture_size
        if max_cudagraph_capture_size < cap:
            cap, binding = max_cudagraph_capture_size, "cudagraph-ladder"

    detail["configured"] = configured_max_num_seqs
    if configured_max_num_seqs < cap:
        cap, binding = configured_max_num_seqs, "configured"

    if cap < floor:
        cap, binding = floor, "floor"
    return AdmissionCap(cap=int(cap), binding=binding, detail=detail)


def blocks_for_lookahead(num_lookahead_tokens: int, block_size: int) -> int:
    """Extra blocks a context-proportional group needs for draft tokens.

    Speculative decoding appends ``num_lookahead_tokens`` unverified positions
    past the request's real length, and the allocator must have slots for them
    (``vllm/v1/core/kv_cache_manager.py:393-396``). This is the third signal the
    old policy was blind to.

    Args:
        num_lookahead_tokens: Draft tokens the scheduler reserves per request.
        block_size: Tokens per KV block for the group.

    Returns:
        Number of additional blocks, ``0`` when speculation is off.
    """
    if num_lookahead_tokens <= 0 or block_size <= 0:
        return 0
    return _cdiv(num_lookahead_tokens, block_size)


@dataclass(frozen=True)
class GateConfig:
    """Tuning of the prefill gate for one SLO objective.

    Args:
        enabled: ``False`` reproduces stock vLLM admission timing exactly.
        max_hold_steps: Hardest bound on consecutive throttled steps. Caps the
            worst-case TTFT the gate can add to roughly
            ``max_hold_steps * step_time``.
        ttft_budget_ms: Release as soon as the oldest waiting request has waited
            this long. This is the SLO knob: it bounds queueing delay directly
            rather than through a step count.
        burst_tokens: Release once the waiting queue holds at least this many
            prompt tokens, i.e. enough to fill a prefill step. Holding past this
            point buys no extra batching and only costs TTFT.
        min_decode_batch: Never throttle unless at least this many requests are
            already decoding. With a small running batch a throttled step wastes
            the GPU, and TTFT is all the user can perceive.
        kv_high_watermark: Never throttle above this KV pool utilisation; when
            the pool is nearly full the priority is draining, not batching.
        adaptive: ``True`` lets ``min_decode_batch`` scale with the observed
            running batch and lets the hold release early on a filled burst.
            ``False`` is the static-cadence ablation: throttle whenever
            ``step % (max_hold_steps + 1) != 0``, with no load signals at all.
    """

    enabled: bool
    max_hold_steps: int = 0
    ttft_budget_ms: float = 0.0
    burst_tokens: int = 0
    min_decode_batch: int = 32
    kv_high_watermark: float = 0.9
    adaptive: bool = True


def gate_config_for(
    objective: Objective,
    *,
    max_num_scheduled_tokens: int,
    hold_steps: int | None = None,
    ttft_budget_ms: float | None = None,
) -> GateConfig:
    """Build a :class:`GateConfig` for an SLO objective.

    Args:
        objective: ``"interactive"`` (never delay a prefill), ``"balanced"``,
            ``"throughput"`` (protect decode-step purity hardest), or
            ``"static"`` (fixed cadence, no load signals — the ablation).
        max_num_scheduled_tokens: The engine's per-step token budget; sets the
            default burst target.
        hold_steps: Override ``max_hold_steps``.
        ttft_budget_ms: Override ``ttft_budget_ms``.

    Returns:
        The gate configuration.
    """
    if objective == "interactive":
        cfg = GateConfig(enabled=False)
    elif objective == "balanced":
        cfg = GateConfig(
            enabled=True,
            max_hold_steps=4,
            ttft_budget_ms=750.0,
            burst_tokens=max(1, max_num_scheduled_tokens // 2),
            min_decode_batch=32,
        )
    elif objective == "throughput":
        cfg = GateConfig(
            enabled=True,
            max_hold_steps=16,
            ttft_budget_ms=4000.0,
            burst_tokens=max(1, max_num_scheduled_tokens),
            min_decode_batch=32,
        )
    elif objective == "static":
        cfg = GateConfig(
            enabled=True,
            max_hold_steps=4,
            ttft_budget_ms=math.inf,
            burst_tokens=1 << 30,
            min_decode_batch=1,
            kv_high_watermark=1.0,
            adaptive=False,
        )
    else:
        raise ValueError(f"unknown objective {objective!r}")

    if hold_steps is not None:
        cfg = GateConfig(**{**cfg.__dict__, "max_hold_steps": hold_steps})
    if ttft_budget_ms is not None:
        cfg = GateConfig(**{**cfg.__dict__, "ttft_budget_ms": ttft_budget_ms})
    return cfg


@dataclass
class GateDecision:
    """Why the gate decided what it did (one step)."""

    throttle: bool
    reason: str


@dataclass
class PrefillGate:
    """Decide, per step, whether to defer prefill compute.

    The gate is a *hold-then-burst* controller. While it holds, the engine runs
    pure decode steps, which is where a decode-only CUDA graph and an
    uncontended step time live. When it releases, the accumulated waiting
    requests are admitted together, so one expensive mixed step pays for several
    prefills instead of one.

    Every release condition is a safety property as well as a policy: a request
    that has waited past ``ttft_budget_ms`` is released regardless of batching
    opportunity, and preemption or KV pressure disables the gate outright.

    Args:
        cfg: The objective's :class:`GateConfig`.
    """

    cfg: GateConfig
    _held_steps: int = 0
    _step: int = 0
    counters: dict[str, int] = field(default_factory=dict)

    def _bump(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    def decide(
        self,
        *,
        num_running: int,
        num_waiting: int,
        waiting_tokens: int,
        oldest_wait_ms: float,
        kv_usage: float,
        preempted_pending: bool,
        admission_cap: int,
    ) -> GateDecision:
        """Return whether this step should defer prefill compute.

        Args:
            num_running: Requests currently in the running queue.
            num_waiting: Requests in the waiting queue.
            waiting_tokens: Total prompt tokens sitting in the waiting queue.
            oldest_wait_ms: Age of the oldest waiting request, milliseconds.
            kv_usage: KV block pool utilisation in ``[0, 1]``.
            preempted_pending: Whether any waiting request was preempted and is
                trying to resume. Deferring those would turn a preemption into a
                stall, so the gate always releases.
            admission_cap: The derived running-request cap, used to scale
                ``min_decode_batch`` with the machine rather than a constant.

        Returns:
            A :class:`GateDecision`.
        """
        self._step += 1

        if not self.cfg.enabled:
            return self._release("disabled")

        if not self.cfg.adaptive:
            # Static-cadence ablation: no load signals, no SLO term.
            period = self.cfg.max_hold_steps + 1
            if self._step % period != 0:
                self._bump("throttled")
                return GateDecision(True, "static-cadence")
            return self._release("static-cadence")

        if num_waiting == 0:
            return self._release("queue-empty")
        if preempted_pending:
            return self._release("preempted-pending")
        if kv_usage >= self.cfg.kv_high_watermark:
            return self._release("kv-pressure")

        # Load signal: only worth protecting a decode batch that exists. Scale
        # with the derived cap so this is not a hardware-specific constant.
        min_batch = self.cfg.min_decode_batch
        if admission_cap > 0:
            min_batch = max(
                min_batch, min(self.cfg.min_decode_batch, admission_cap // 8)
            )
        if num_running < min_batch:
            return self._release("underloaded")

        if oldest_wait_ms >= self.cfg.ttft_budget_ms:
            return self._release("ttft-budget")
        if waiting_tokens >= self.cfg.burst_tokens:
            return self._release("burst-full")
        if self._held_steps >= self.cfg.max_hold_steps:
            return self._release("hold-expired")

        self._held_steps += 1
        self._bump("throttled")
        return GateDecision(True, "holding")

    def _release(self, reason: str) -> GateDecision:
        self._held_steps = 0
        self._bump(f"release:{reason}")
        return GateDecision(False, reason)
