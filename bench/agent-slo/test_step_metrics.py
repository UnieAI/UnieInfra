"""Tests for the scheduler-counter accounting the parity measurements quote.

`step_metrics.py` scrapes the engine's exposition and `step_report.py` turns the
series into iterations per second, tokens per iteration and the split of steps by
tokens carried. Those numbers decide whether a cell is short of work or short of
overlap, so a parser that dropped a label set, or a report that counted the
warmup as measured, would point the investigation at the wrong subsystem. These
tests build a scrape and a result window whose answers are known by construction.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import step_metrics  # noqa: E402
import step_report  # noqa: E402

KEYS = list(step_metrics.KEYS)


def _body(iteration_buckets, extra=""):
    """An exposition body in the shape the server publishes."""
    lines = []
    for bound, count in iteration_buckets:
        lines.append(f'tokenspeed:iteration_tokens_total_bucket{{le="{bound}"}} {count}')
    lines += [
        "tokenspeed:iteration_tokens_total_sum 9000.0",
        "tokenspeed:iteration_tokens_total_count 12.0",
        # A second label set must be added to the first, not dropped.
        'tokenspeed:iteration_tokens_total_count{engine="0"} 3.0',
        "tokenspeed:num_requests_running 64.0",
        "tokenspeed:num_requests_waiting 38.0",
        "tokenspeed:kv_cache_usage_perc 0.4295",
        # Buckets of histograms that are not kept must not become columns.
        'tokenspeed:time_to_first_token_seconds_bucket{le="1.0"} 3.0',
        "tokenspeed:time_to_first_token_seconds_sum 12.0",
        "tokenspeed:time_to_first_token_seconds_count 4.0",
    ]
    return "\n".join(lines) + "\n" + extra


def test_buckets_are_kept_only_for_the_step_size_histogram():
    body = _body([("0.0", 0), ("64.0", 10), ("8192.0", 12), ("+Inf", 12)])
    assert step_metrics.discover(body) == [
        "iteration_tokens_total_le+Inf",
        "iteration_tokens_total_le0.0",
        "iteration_tokens_total_le64.0",
        "iteration_tokens_total_le8192.0",
    ]


def test_scrape_is_summed_across_label_sets():
    body = _body([("64.0", 10), ("+Inf", 12)])
    keys = KEYS + step_metrics.discover(body)
    sample = step_metrics.parse(body, keys)
    assert sample["num_requests_running"] == 64.0
    assert sample["kv_cache_usage_perc"] == pytest.approx(0.4295)
    assert sample["iteration_tokens_total_count"] == 15.0
    assert sample["iteration_tokens_total_sum"] == 9000.0
    assert sample["iteration_tokens_total_le64.0"] == 10.0
    assert not any(k.startswith("time_to_first_token_seconds_le") for k in sample)


def test_unrequested_series_are_ignored():
    body = _body([("+Inf", 12)], extra="some_other_metric 5.0\n")
    sample = step_metrics.parse(body, list(KEYS))
    assert "some_other_metric" not in sample
    assert set(sample) == set(KEYS)


def _write_csv(tmp_path, rows):
    columns = [
        "t",
        "wall",
        "num_requests_running",
        "num_requests_waiting",
        "kv_cache_usage_perc",
        "iteration_tokens_total_sum",
        "iteration_tokens_total_count",
        "iteration_tokens_total_le0.0",
        "iteration_tokens_total_le64.0",
        "iteration_tokens_total_le8192.0",
        "iteration_tokens_total_le+Inf",
    ]
    path = tmp_path / "metrics.csv"
    path.write_text(
        ",".join(columns) + "\n" + "".join(",".join(str(v) for v in row) + "\n" for row in rows)
    )
    return str(path)


def _row(t, wall, running, iters, tokens, decode_steps, prefill_steps):
    return [
        t,
        wall,
        running,
        0,
        0.4,
        tokens,
        iters,
        0,
        decode_steps,
        decode_steps + prefill_steps,
        decode_steps + prefill_steps,
    ]


def test_report_slices_to_the_measured_window(tmp_path, capsys):
    # The sampler starts before the warmup, so the first two samples are warmup:
    # they must not enter the accounting the measured phase is judged by.
    rows = [
        _row(0.0, 1000.0, 0, 0, 0, 0, 0),
        _row(1.0, 1001.0, 64, 100, 900000, 90, 10),
        _row(2.0, 1002.0, 64, 200, 1800000, 180, 20),
        _row(3.0, 1010.0, 64, 300, 1900000, 270, 30),
        _row(5.0, 1012.0, 64, 400, 2024000, 360, 40),
    ]
    csv_path = _write_csv(tmp_path, rows)
    result = tmp_path / "measured.json"
    result.write_text(json.dumps({"started_at": 1009.0, "finished_at": 1013.0, "rows": []}))

    step_report.main(["step_report.py", csv_path, "--json", str(result), "--label", "cell"])

    out = capsys.readouterr().out
    # Measured window covers t=3..5: 100 iterations over 2 s, 124000 tokens,
    # 90 of 100 steps carrying decode rows alone and 10 carrying a prefill chunk.
    assert "2.0s  100 iterations  50.00 iter/s  20.0 ms/iter" in out
    assert "1240 tokens/iter" in out
    assert "(0, 64]" in out and "90  " in out
    assert "(64, 8192]" in out and "10  " in out
    assert "measured phase" in out


def test_report_without_a_window_counts_the_whole_file(tmp_path, capsys):
    rows = [
        _row(0.0, 1000.0, 64, 50, 100000, 40, 10),
        _row(2.0, 1002.0, 64, 150, 200000, 130, 20),
    ]
    csv_path = _write_csv(tmp_path, rows)

    step_report.main(["step_report.py", csv_path])

    out = capsys.readouterr().out
    assert "whole sample file" in out
    assert "50.00 iter/s" in out


def test_report_refuses_a_result_without_a_window(tmp_path):
    csv_path = _write_csv(tmp_path, [_row(0.0, 1000.0, 64, 10, 100, 9, 1)])
    result = tmp_path / "old.json"
    result.write_text(json.dumps({"duration_s": 1.0, "rows": []}))

    with pytest.raises(SystemExit):
        step_report.main(["step_report.py", csv_path, "--json", str(result)])