#!/usr/bin/env python3
"""Report the step accounting a step_metrics.py CSV supports.

The throughput number from `drive.py` says how fast a cell ran; it does not say
why. The scheduler exports enough to answer that without a trace, from counters
it already increments once per iteration:

  iterations per second and ms/iteration   the unit of scheduling, directly
  tokens per iteration                     how much of the token budget a step uses
  steps by tokens carried                  a step carrying decode rows alone lands
                                           in the low buckets, a step carrying a
                                           prefill chunk in the high ones -- so the
                                           split says whether prefill and decode
                                           share a round or take turns
  running / waiting / cache gauges         whether admission or memory binds

Give it the result JSON too and the accounting covers the measured phase only,
not the warmup the sampler necessarily also saw:

    python3 bench/agent-slo/step_report.py METRICS_CSV [--json RESULT.json]
        [--label TEXT]
"""

import csv
import json
import statistics
import sys


def bucket_columns(row):
    return sorted(
        (k for k in row if k.startswith("iteration_tokens_total_le")),
        key=lambda k: float(
            k[len("iteration_tokens_total_le") :].replace("+Inf", "inf")
        ),
    )


def measured_window(result):
    data = json.load(open(result))
    if "started_at" not in data or "finished_at" not in data:
        raise SystemExit(f"{result} does not record when its measured phase ran")
    return data["started_at"], data["finished_at"]


def main(argv):
    path = argv[1]
    label = argv[argv.index("--label") + 1] if "--label" in argv else path
    rows = list(csv.DictReader(open(path)))
    window = (
        measured_window(argv[argv.index("--json") + 1]) if "--json" in argv else None
    )
    if window is not None:
        lo, hi = window
        rows = [r for r in rows if "wall" in r and lo <= float(r["wall"]) <= hi]
    live = [r for r in rows if float(r["num_requests_running"]) > 0]
    where = "measured phase" if window else "whole sample file"
    print(
        f"== step accounting: {label} ({len(rows)} samples in the {where}, {len(live)} with a non-empty batch)"
    )
    if len(live) < 2:
        print("   no non-empty batch sampled; nothing to account for")
        return
    first, last = live[0], live[-1]
    seconds = float(last["t"]) - float(first["t"])
    iterations = float(last["iteration_tokens_total_count"]) - float(
        first["iteration_tokens_total_count"]
    )
    tokens = float(last["iteration_tokens_total_sum"]) - float(
        first["iteration_tokens_total_sum"]
    )
    if iterations <= 0:
        print("   no iterations sampled")
        return
    running = [float(r["num_requests_running"]) for r in live]
    waiting = [float(r["num_requests_waiting"]) for r in live]
    cache = [float(r["kv_cache_usage_perc"]) for r in live]
    print(
        f"   {seconds:.1f}s  {iterations:.0f} iterations  "
        f"{iterations / seconds:.2f} iter/s  {1000 * seconds / iterations:.1f} ms/iter"
    )
    print(f"   {tokens / iterations:.0f} tokens/iter")
    print(
        f"   running p50 {statistics.median(running):.0f} max {max(running):.0f}"
        f" | waiting p50 {statistics.median(waiting):.0f} max {max(waiting):.0f}"
        f" | cache p50 {statistics.median(cache):.2f} max {max(cache):.2f}"
    )
    buckets = bucket_columns(rows[0])
    if not buckets:
        print("   (this CSV predates the bucket columns; no step-size split)")
        return
    total = float(last[buckets[-1]]) - float(first[buckets[-1]])
    print(f"   steps by tokens carried ({total:.0f} steps):")
    # An exposition's bucket counters are cumulative, so each row of the split is
    # the difference from the previous bound, not the counter itself.
    low, below = 0.0, 0.0
    for column in buckets:
        bound = float(column[len("iteration_tokens_total_le") :].replace("+Inf", "inf"))
        cumulative = float(last[column]) - float(first[column])
        count = cumulative - below
        span = (
            f"({low:.0f}, inf)"
            if bound == float("inf")
            else f"({low:.0f}, {bound:.0f}]"
        )
        print(
            f"     {span:>16} {count:7.0f}  {100 * count / total if total else 0:5.1f}%"
        )
        low, below = bound, cumulative


if __name__ == "__main__":
    main(sys.argv)
