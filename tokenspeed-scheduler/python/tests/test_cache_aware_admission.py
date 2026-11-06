# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Cache-aware (longest-cached-prefix-first) admission:
TOKENSPEED_CACHE_AWARE_ADMISSION reorders the Submitted pass so a waiting
prompt with a Device-cached prefix is admitted before an earlier cold one,
and TOKENSPEED_CACHE_AWARE_MAX_OVERTAKES ages a passed-over prompt ahead."""

from __future__ import annotations

import pytest
from conftest import _advance, _finish, _spec

ts = pytest.importorskip("tokenspeed_scheduler")

PAGE = 8


def _config(max_batch_size: int = 1) -> "ts.SchedulerConfig":
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = PAGE
    cfg.num_device_pages = 256
    cfg.num_host_pages = 0
    cfg.num_snapshot_pages = 1  # never retracts
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = max_batch_size
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = False
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="full",
            block_granularity=PAGE,
            total_pages=cfg.num_device_pages,
            attn_kind=ts.AttnKind.Full,
        )
    ]
    return cfg


def _prefill_ops(plan) -> dict[str, int]:
    """request id -> extend_prefix_len of every forward row in the plan."""
    out: dict[str, int] = {}
    for op in plan.forward:
        for rid, prefix in zip(op.request_ids, op.extend_prefix_lens):
            out[rid] = prefix
    return out


def _warm(scheduler, request_id: str, tokens: list[int]) -> None:
    """Run one turn to completion so its prompt pages are published."""
    scheduler.submit_requests([_spec(request_id, tokens)])
    assert request_id in _prefill_ops(scheduler.next_execution_plan())
    _advance(scheduler, request_id, [9001])
    scheduler.next_execution_plan()  # first decode publishes the prefix pages
    _advance(scheduler, request_id, [9002])
    _finish(scheduler, request_id)
    scheduler.next_execution_plan()


SESSION_A = list(range(1, 33))  # 4 pages
SESSION_B = list(range(101, 133))


def _cold(base: int) -> list[int]:
    return list(range(base, base + 40))


def _admit_order(monkeypatch, enabled: bool, env: dict[str, str] | None = None):
    if enabled:
        monkeypatch.setenv("TOKENSPEED_CACHE_AWARE_ADMISSION", "1")
    else:
        monkeypatch.delenv("TOKENSPEED_CACHE_AWARE_ADMISSION", raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    return ts.Scheduler(_config())


@pytest.mark.parametrize("enabled", [False, True])
def test_cached_prompt_admitted_before_earlier_cold_one(monkeypatch, enabled):
    scheduler = _admit_order(monkeypatch, enabled)
    _warm(scheduler, "a1", SESSION_A)
    # The cold prompt is submitted first; the next turn of session A second.
    scheduler.submit_requests(
        [
            _spec("cold", _cold(5000)),
            _spec("a2", SESSION_A + [9001, 9002] + list(range(700, 706))),
        ]
    )
    first = _prefill_ops(scheduler.next_execution_plan())
    if enabled:
        # Longest cached prefix first: a2 hits A's cached turn (its 32 prompt
        # tokens, plus the published decode tail).
        assert set(first) == {"a2"} and first["a2"] >= 32
    else:
        assert first == {"cold": 0}  # FIFO (knob off: unchanged behavior)


def test_overtaken_prompt_ages_ahead_of_cached_ones(monkeypatch):
    scheduler = _admit_order(
        monkeypatch, True, {"TOKENSPEED_CACHE_AWARE_MAX_OVERTAKES": "1"}
    )
    _warm(scheduler, "a1", SESSION_A)
    _warm(scheduler, "b1", SESSION_B)
    scheduler.submit_requests(
        [
            _spec("cold", _cold(5000)),
            _spec("a2", SESSION_A + list(range(700, 708))),
            _spec("b2", SESSION_B + list(range(800, 808))),
        ]
    )
    # Round 1: a cached prompt overtakes the cold one (its first overtake).
    first = _prefill_ops(scheduler.next_execution_plan())
    assert set(first) == {"a2"} and first["a2"] > 0
    _advance(scheduler, "a2", [1])
    _finish(scheduler, "a2")
    # Round 2: the cold prompt has aged (1 overtake) and goes before b2,
    # though b2 still has a longer cached prefix.
    second = _prefill_ops(scheduler.next_execution_plan())
    assert second == {"cold": 0}


def test_without_aging_cached_prompts_keep_going_first(monkeypatch):
    scheduler = _admit_order(
        monkeypatch, True, {"TOKENSPEED_CACHE_AWARE_MAX_OVERTAKES": "100"}
    )
    _warm(scheduler, "a1", SESSION_A)
    _warm(scheduler, "b1", SESSION_B)
    scheduler.submit_requests(
        [
            _spec("cold", _cold(5000)),
            _spec("a2", SESSION_A + list(range(700, 708))),
            _spec("b2", SESSION_B + list(range(800, 808))),
        ]
    )
    first = _prefill_ops(scheduler.next_execution_plan())
    assert set(first) == {"a2"}
    _advance(scheduler, "a2", [1])
    _finish(scheduler, "a2")
    second = _prefill_ops(scheduler.next_execution_plan())
    assert set(second) == {"b2"} and second["b2"] > 0


def test_ranking_probe_does_not_change_admission_hits(monkeypatch):
    """Ranking a deep queue every round must not pin or evict: every
    queued prompt still gets its full cached prefix when admitted."""
    scheduler = _admit_order(
        monkeypatch, True, {"TOKENSPEED_CACHE_AWARE_REFRESH_ROUNDS": "1"}
    )
    _warm(scheduler, "a1", SESSION_A)
    scheduler.submit_requests(
        [_spec(f"cold{i}", _cold(5000 + 100 * i)) for i in range(3)]
        + [_spec("a2", SESSION_A + list(range(700, 708)))]
    )
    seen: dict[str, int] = {}
    for _ in range(40):
        ops = _prefill_ops(scheduler.next_execution_plan())
        for rid, prefix in ops.items():
            seen.setdefault(rid, prefix)
            _advance(scheduler, rid, [1])
            _finish(scheduler, rid)
        if len(seen) == 4:
            break
    assert seen["a2"] >= 32
    assert list(seen)[0] == "a2"
    assert sorted(seen) == ["a2", "cold0", "cold1", "cold2"]
