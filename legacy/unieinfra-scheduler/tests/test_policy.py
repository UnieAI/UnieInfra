# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Unit tests for the admission policy. No vLLM, no GPU."""

from __future__ import annotations

import math

import pytest
from unieinfra_scheduler.policy import (
    GateConfig,
    PrefillGate,
    ResourceCost,
    blocks_for_lookahead,
    derive_admission_cap,
    gate_config_for,
)

# Qwen3-8B on 1x RTX PRO 6000 at gpu-memory-utilization 0.85, max_model_len
# 4096, block_size 16 -- the configuration every benchmark in E35/E36 uses.
QWEN3_8B_BLOCKS = 29121  # 465,936 tokens / 16
FULL_ATTN_4K = ResourceCost("full", blocks_per_request=256, grows_with_context=True)


def test_dense_cap_matches_kv_pool_arithmetic():
    cap = derive_admission_cap(
        num_blocks=QWEN3_8B_BLOCKS,
        costs=[FULL_ATTN_4K],
        configured_max_num_seqs=1024,
        max_cudagraph_capture_size=512,
        reserve_divisor=4,
    )
    # 29121 blocks / (256 // 4) = 455 concurrent requests.
    assert cap.cap == 455
    assert cap.binding == "kv-pool"
    # vLLM's own full-context concurrency for this pool is 113.75x; that is the
    # hard bound the discount is taken against, and it must be reported too.
    assert cap.detail["kv:full:full"] == QWEN3_8B_BLOCKS // 256 == 113
    assert cap.detail["kv:full"] == 455


def test_reserve_divisor_one_is_the_full_context_bound():
    cap = derive_admission_cap(
        num_blocks=QWEN3_8B_BLOCKS,
        costs=[FULL_ATTN_4K],
        configured_max_num_seqs=1024,
        max_cudagraph_capture_size=None,
        reserve_divisor=1,
    )
    assert cap.cap == 113


def test_cudagraph_ladder_binds_before_kv():
    cap = derive_admission_cap(
        num_blocks=QWEN3_8B_BLOCKS,
        costs=[FULL_ATTN_4K],
        configured_max_num_seqs=1024,
        max_cudagraph_capture_size=256,
        reserve_divisor=4,
    )
    assert cap.cap == 256
    assert cap.binding == "cudagraph-ladder"


def test_never_raises_above_configured():
    cap = derive_admission_cap(
        num_blocks=QWEN3_8B_BLOCKS,
        costs=[FULL_ATTN_4K],
        configured_max_num_seqs=64,
        max_cudagraph_capture_size=512,
        reserve_divisor=4,
    )
    assert cap.cap == 64
    assert cap.binding == "configured"


def test_recurrent_state_pool_is_not_discounted():
    """The E28/E31 regression: a fixed-size state pool must bound admission.

    A GDN/Mamba group costs the same blocks per request no matter how far the
    request has decoded, so applying the steady-state divisor to it is exactly
    the over-admission that produced retractions, C=64 hangs and a C=128
    device-side assert. Here the state pool holds 114 requests; the KV group
    alone would allow far more.
    """
    state = ResourceCost("state", blocks_per_request=255, grows_with_context=False)
    cap = derive_admission_cap(
        num_blocks=QWEN3_8B_BLOCKS,
        costs=[FULL_ATTN_4K, state],
        configured_max_num_seqs=1024,
        max_cudagraph_capture_size=512,
        reserve_divisor=4,
    )
    # 64 (discounted attention) + 255 (undiscounted state) = 319 blocks/request.
    assert cap.cap == QWEN3_8B_BLOCKS // 319 == 91
    assert cap.cap < 114  # strictly below the state pool's own capacity
    assert cap.binding == "kv-pool"


def test_sliding_window_group_reports_its_window_bounded_cost():
    """A window-bounded group must not be charged the full-context cost.

    The old policy estimated live KV as ``context_len / reserve_divisor`` with
    no per-layer notion, which over-charged sliding layers by the ratio of
    context to window and under-admitted badly on Gemma-4-shaped models.
    """
    swa = ResourceCost("swa", blocks_per_request=32, grows_with_context=True)
    cap = derive_admission_cap(
        num_blocks=QWEN3_8B_BLOCKS,
        costs=[swa],
        configured_max_num_seqs=4096,
        max_cudagraph_capture_size=None,
        reserve_divisor=4,
    )
    assert cap.cap == QWEN3_8B_BLOCKS // 8


def test_lookahead_blocks():
    assert blocks_for_lookahead(0, 16) == 0
    assert blocks_for_lookahead(1, 16) == 1
    assert blocks_for_lookahead(16, 16) == 1
    assert blocks_for_lookahead(17, 16) == 2


def test_no_kv_cache_keeps_configured_cap():
    cap = derive_admission_cap(
        num_blocks=0,
        costs=[],
        configured_max_num_seqs=256,
        max_cudagraph_capture_size=512,
    )
    assert cap.cap == 256
    assert cap.binding == "no-kv-cache"


# --------------------------------------------------------------- the gate ---


def _gate(objective="throughput", **kw):
    cfg = gate_config_for(objective, max_num_scheduled_tokens=8192, **kw)
    return PrefillGate(cfg=cfg)


def _decide(gate, **kw):
    base = dict(
        num_running=128,
        num_waiting=1,
        waiting_tokens=1024,
        oldest_wait_ms=0.0,
        kv_usage=0.3,
        preempted_pending=False,
        admission_cap=455,
    )
    base.update(kw)
    return gate.decide(**base)


def test_interactive_never_throttles():
    gate = _gate("interactive")
    assert not gate.cfg.enabled
    for _ in range(10):
        d = _decide(gate)
        assert d.throttle is False
        assert d.reason == "disabled"


def test_throughput_holds_then_expires():
    gate = _gate("throughput", hold_steps=3)
    reasons = [_decide(gate).throttle for _ in range(5)]
    # Holds for max_hold_steps, then releases, then holds again.
    assert reasons == [True, True, True, False, True]


def test_underloaded_never_throttles():
    gate = _gate("throughput")
    for _ in range(5):
        d = _decide(gate, num_running=4)
        assert d.throttle is False
        assert d.reason == "underloaded"


def test_ttft_budget_releases_regardless_of_batching():
    gate = _gate("throughput", ttft_budget_ms=100.0)
    d = _decide(gate, oldest_wait_ms=250.0)
    assert d.throttle is False
    assert d.reason == "ttft-budget"


def test_burst_full_releases_early():
    gate = _gate("throughput")  # burst_tokens == 8192
    d = _decide(gate, waiting_tokens=8192)
    assert d.throttle is False
    assert d.reason == "burst-full"


def test_preemption_and_kv_pressure_release():
    gate = _gate("throughput")
    assert _decide(gate, preempted_pending=True).reason == "preempted-pending"
    assert _decide(gate, kv_usage=0.95).reason == "kv-pressure"


def test_empty_queue_releases():
    gate = _gate("throughput")
    assert _decide(gate, num_waiting=0).reason == "queue-empty"


def test_static_ablation_ignores_every_load_signal():
    """The static arm must behave identically under wildly different load."""
    a = _gate("static")
    b = _gate("static")
    pat_a = [_decide(a, num_running=128, num_waiting=8).throttle for _ in range(10)]
    pat_b = [
        _decide(b, num_running=1, num_waiting=1, oldest_wait_ms=60_000.0).throttle
        for _ in range(10)
    ]
    assert pat_a == pat_b
    assert pat_a == [True, True, True, True, False] * 2


def test_static_config_has_no_slo_term():
    cfg = gate_config_for("static", max_num_scheduled_tokens=8192)
    assert cfg.adaptive is False
    assert cfg.ttft_budget_ms == math.inf


def test_unknown_objective_rejected():
    with pytest.raises(ValueError):
        gate_config_for("nonsense", max_num_scheduled_tokens=8192)  # type: ignore[arg-type]


def test_gate_counters_record_release_reasons():
    gate = _gate("throughput", hold_steps=1)
    _decide(gate)
    _decide(gate)
    assert gate.counters["throttled"] == 1
    assert gate.counters["release:hold-expired"] == 1


def test_disabled_gate_config_is_inert():
    gate = PrefillGate(cfg=GateConfig(enabled=False))
    assert _decide(gate).throttle is False
