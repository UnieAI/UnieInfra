"""Tests for the chrome-trace summarizer used by the parity measurements.

The numbers ``trace_stats`` prints are the numerators and denominators of every
performance claim in ``unieinfra/xref/72-PARITY-STRATEGY.md``: ms/step,
kernels/step, us/kernel and device-busy. A summarizer that silently counts a
kernel twice, or attributes a kernel to the wrong step, would make a slow step
look fast, so these tests build traces whose answers are known by construction.
"""

from __future__ import annotations

import gzip
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import trace_stats  # noqa: E402


def _write(tmp_path, events, name="t.trace.json.gz"):
    path = tmp_path / name
    with gzip.open(path, "wt") as f:
        json.dump({"traceEvents": events}, f)
    return str(path)


def _step(ts, dur_us, kernels, cat="python_function", name="forward_step"):
    """One step event plus its kernels, given as (name, start_offset, dur)."""
    events = [
        {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur_us},
    ]
    for kernel, offset, kdur in kernels:
        events.append(
            {"ph": "X", "cat": "kernel", "name": kernel, "ts": ts + offset, "dur": kdur}
        )
    return events


def _contiguous(n, kdur, offset):
    return [("k", offset + i * kdur, kdur) for i in range(n)]


def test_per_step_totals_match_the_trace_construction(tmp_path):
    # 3 steps of 20 ms; each holds 100 contiguous 50 us kernels plus one 200 us
    # kernel. Kernel time per step is 5200 us and, being contiguous, so is the
    # device-busy interval.
    events = []
    ts = 1_000_000
    for _ in range(3):
        events += _step(
            ts, 20_000, _contiguous(100, 50, 100) + [("mhc_big_fuse", 5_100, 200)]
        )
        ts += 25_000
    summary = trace_stats.summarize(events, "forward_step", top=5, kernel_filter=None)

    assert summary["trace_steps"] == 3
    assert summary["step_ms"]["mean"] == pytest.approx(20.0)
    assert summary["kernels_per_step"]["mean"] == pytest.approx(101.0)
    assert summary["kernel_us_per_step"] == pytest.approx(5_200.0)
    assert summary["us_per_kernel"] == pytest.approx(5_200.0 / 101.0)
    assert summary["device_busy_ms_per_step"] == pytest.approx(5.2)
    assert summary["device_busy_fraction"] == pytest.approx(5.2 / 20.0)


def test_busy_interval_collapses_overlapping_kernels(tmp_path):
    # Two kernels on different streams running at the same time: the device is
    # busy for one kernel's duration, not two. Summing durations instead of
    # unioning intervals is what would fabricate a device-bound step.
    kernels = [("a", 0, 400), ("b", 0, 400)]
    events = _step(0, 10_000, kernels)
    summary = trace_stats.summarize(events, "forward_step", top=5, kernel_filter=None)

    assert summary["kernel_us_per_step"] == pytest.approx(800.0)
    assert summary["device_busy_ms_per_step"] == pytest.approx(0.4)


def test_a_kernel_started_before_the_step_is_not_attributed_to_it(tmp_path):
    # Host launches step N while step N-1 still runs on the device, so a trace
    # contains kernels that began before the step boundary. Attributing them to
    # the step that merely overlaps them would double count.
    events = [
        {"ph": "X", "cat": "kernel", "name": "early", "ts": 0, "dur": 1_000},
        *_step(500, 10_000, [("mine", 100, 300)]),
    ]
    summary = trace_stats.summarize(events, "forward_step", top=5, kernel_filter=None)

    assert summary["kernels_per_step"]["mean"] == pytest.approx(1.0)
    assert [name for name, _ in summary["top_by_time"]] == ["mine"]


def test_kernel_filter_reports_selection_separately(tmp_path):
    events = _step(0, 5_000, [("mhc_big_fuse", 10, 200), ("other", 300, 100)])

    found = trace_stats.summarize(
        events, "forward_step", top=5, kernel_filter="mhc_big_fuse"
    )
    assert set(found["matches"]) == {"mhc_big_fuse"}
    assert found["matches"]["mhc_big_fuse"] == [1, 200]

    # A registered-but-unselected fast path shows up as an empty match, which is
    # the whole point of asking: "registered" is not "used".
    missing = trace_stats.summarize(
        events, "forward_step", top=5, kernel_filter="mhc_big_fuse_v2"
    )
    assert missing["matches"] == {}


def test_nested_step_instances_are_counted_once(tmp_path):
    # A recursive or nested call with the same name would otherwise turn one
    # 20 ms step into two 10 ms steps and halve every per-step number.
    events = [
        {
            "ph": "X",
            "cat": "python_function",
            "name": "forward_step",
            "ts": 0,
            "dur": 20_000,
        },
        {
            "ph": "X",
            "cat": "python_function",
            "name": "forward_step",
            "ts": 5_000,
            "dur": 10_000,
        },
        {"ph": "X", "cat": "kernel", "name": "k", "ts": 1_000, "dur": 100},
    ]
    summary = trace_stats.summarize(events, "forward_step", top=5, kernel_filter=None)

    assert summary["trace_steps"] == 1
    assert summary["step_ms"]["mean"] == pytest.approx(20.0)


def test_step_function_is_auto_detected_from_a_real_trace_file(tmp_path):
    events = _step(0, 1_000, [("k", 10, 100)])
    path = _write(tmp_path, events)

    loaded = trace_stats.load_events(path)
    assert trace_stats.pick_step_fn(loaded, None) == "forward_step"
    with pytest.raises(SystemExit):
        trace_stats.pick_step_fn(loaded, "not_in_trace")
