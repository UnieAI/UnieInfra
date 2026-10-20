#!/usr/bin/env python3
"""Summarize a TokenSpeed chrome trace the way the parity method needs it.

The engine already renders the numbers a decode investigation asks for: POST
``/start_profile`` with a body of ``{"output_dir": ..., "num_steps": N,
"activities": ["CPU", "GPU"]}`` and it writes
``<output_dir>/<profile_id>-<rank><stage>.trace.json.gz`` when profiling stops.
This script turns that file into per-decode-step quantities instead of a
hierarchy of averages:

  ms/step        wall time of one step, from the host function that brackets it
  kernels/step   how many kernel launches that step issued
  us/kernel      mean kernel duration -- the fingerprint that separates a
                 launch-bound step (many short kernels) from a compute-bound one
  device_busy    union of the kernel intervals, i.e. how much of ms/step the
                 device actually spent executing kernels

A step with ``us/kernel`` near the empty-launch floor (7.7-13.9 us on H200) and
``device_busy`` well under its wall time is not compute bound; adding FLOPs to
that step buys nothing. A step whose ``device_busy`` is close to ms/step is at
the hardware floor and only fewer ops or less traffic will move it.

Usage:
    python3 bench/agent-slo/trace_stats.py TRACE.json.gz [--step-fn NAME]
        [--kernel NAME_SUBSTRING] [--top 15] [--json OUT]

``--step-fn`` defaults to auto-detection: the host function whose instances
each contain kernel launches becomes the step boundary, and the candidates are
printed so a wrong pick is visible rather than silent.
"""

import argparse
import bisect
import gzip
import json
import statistics
import sys

# Host functions that bracket a decode step, most specific first. The engine's
# event loop calls these once per forward; a trace names whichever one the
# merged tree happens to use.
STEP_FN_CANDIDATES = (
    "forward_step",
    "execute_forward",
    "run_once",
    "_run_once",
    "step",
    "event_loop",
)

HOST_CATS = ("python_function", "cpu_op")


def load_events(path):
    """Complete events from a chrome trace, decoded as the file streams.

    A serving-scale trace is hundreds of megabytes before decompression, and
    ``json.load`` on one expands every annotation it carries: the first attempt
    at the c64 Qwen3.8 profile reached 11.8 GiB of resident memory and had to be
    killed. The events are decoded one at a time instead, and only the fields
    the summary reads are kept.
    """
    opener = gzip.open if path.endswith(".gz") else open
    decoder = json.JSONDecoder()
    events = []
    buf = ""
    with opener(path, "rt") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            buf += chunk
            pos = 0
            while True:
                # Skip separators; a partial value at the end of the buffer is
                # completed by the next chunk.
                while pos < len(buf) and buf[pos] in " \t\r\n,[]":
                    pos += 1
                if pos >= len(buf):
                    break
                try:
                    value, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break
                pos = end
                if isinstance(value, dict) and value.get("ph") == "X":
                    if "ts" in value and "dur" in value:
                        events.append(
                            {
                                "name": value.get("name"),
                                "cat": value.get("cat"),
                                "ts": value["ts"],
                                "dur": value["dur"],
                            }
                        )
            buf = buf[pos:]
    return events


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def union_us(intervals):
    """Total time covered by a set of (start, end) intervals."""
    if not intervals:
        return 0.0
    total = 0.0
    start, end = sorted(intervals)[0]
    for s, e in sorted(intervals)[1:]:
        if s > end:
            total += end - start
            start, end = s, e
        else:
            end = max(end, e)
    return total + (end - start)


def host_candidates(events):
    counts = {}
    for e in events:
        if e.get("cat") in HOST_CATS:
            counts[e["name"]] = counts.get(e["name"], 0) + 1
    return counts


def pick_step_fn(events, requested):
    counts = host_candidates(events)
    if requested:
        if requested not in counts:
            raise SystemExit(
                f"--step-fn {requested!r} does not appear as a host event; "
                f"most frequent host names: {sorted(counts, key=counts.get, reverse=True)[:10]}"
            )
        return requested
    for name in STEP_FN_CANDIDATES:
        if counts.get(name, 0) >= 1:
            return name
    best = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    raise SystemExit(
        "could not auto-detect the step function; pass --step-fn. "
        f"Most frequent host names and counts: {best}"
    )


def steps_from(events, step_fn):
    """Group kernel events under the innermost host calls that contain them."""
    steps = sorted(
        (e for e in events if e.get("cat") in HOST_CATS and e["name"] == step_fn),
        key=lambda e: e["ts"],
    )
    # Overlapping instances of the same name mean the trace caught a nested or
    # recursive call; keep the outermost so a step is counted once.
    outer = []
    for e in steps:
        if outer and e["ts"] < outer[-1]["ts"] + outer[-1]["dur"]:
            continue
        outer.append(e)
    return outer


def summarize(events, step_fn, top, kernel_filter):
    steps = steps_from(events, step_fn)
    if not steps:
        raise SystemExit(
            f"no host instances of {step_fn!r} found; nothing to summarize"
        )
    kernels = [e for e in events if e.get("cat") == "kernel"]
    # Steps and kernels are both in trace-time order, so each step's kernels are
    # a contiguous range: bisect into it instead of scanning every kernel once
    # per step. The scan was steps x kernels -- minutes of CPU on a serving-scale
    # trace that holds millions of launches.
    kernels.sort(key=lambda e: e["ts"])
    kernel_ts = [k["ts"] for k in kernels]

    per_step = []
    attributed = []
    for step in steps:
        start, end = step["ts"], step["ts"] + step["dur"]
        lo = bisect.bisect_left(kernel_ts, start)
        hi = bisect.bisect_left(kernel_ts, end)
        inside = kernels[lo:hi]
        if not inside:
            continue
        attributed.extend(inside)
        busy = union_us([(k["ts"], k["ts"] + k["dur"]) for k in inside])
        per_step.append(
            {
                "ms": step["dur"] / 1000.0,
                "kernels": len(inside),
                "kernel_us": sum(k["dur"] for k in inside),
                "busy_us": busy,
            }
        )
    if not per_step:
        raise SystemExit(
            f"{len(steps)} host instances of {step_fn!r} contain no kernel launches; "
            "pick a function that wraps the forward"
        )

    # Only kernels inside a summarized step are attributed. A trace also holds
    # startup captures, warmup and any other work that happened while profiling,
    # and ranking those as "top GPU time" would point an investigation at code
    # the step never runs.
    by_name = {}
    by_count = {}
    for k in attributed:
        name = k["name"]
        agg = by_name.setdefault(name, [0, 0.0])
        agg[0] += 1
        agg[1] += k["dur"]
        by_count[name] = by_count.get(name, 0) + 1

    ms = [s["ms"] for s in per_step]
    kn = [s["kernels"] for s in per_step]
    kus = [s["kernel_us"] for s in per_step]
    busy = [s["busy_us"] for s in per_step]
    total_kernels = sum(kn)
    total_kernel_us = sum(kus)

    summary = {
        "trace_steps": len(per_step),
        "step_ms": {
            "mean": statistics.mean(ms),
            "p50": pct(ms, 0.50),
            "p90": pct(ms, 0.90),
            "p99": pct(ms, 0.99),
        },
        "kernels_per_step": {
            "mean": statistics.mean(kn),
            "p50": pct(kn, 0.50),
            "p90": pct(kn, 0.90),
            "p99": pct(kn, 0.99),
        },
        "us_per_kernel": total_kernel_us / total_kernels if total_kernels else None,
        "kernel_us_per_step": statistics.mean(kus),
        "device_busy_ms_per_step": statistics.mean(busy) / 1000.0,
        "device_busy_fraction": statistics.mean(busy) / (statistics.mean(ms) * 1000.0),
        "top_by_time": sorted(by_name.items(), key=lambda kv: kv[1][1], reverse=True)[
            :top
        ],
        "top_by_count": sorted(by_count.items(), key=lambda kv: kv[1], reverse=True)[
            :top
        ],
    }
    if kernel_filter:
        summary["matches"] = {
            name: agg
            for name, agg in by_name.items()
            if kernel_filter.lower() in name.lower()
        }
    return summary


def report(s, step_fn, kernel_filter):
    print(f"step function        {step_fn}")
    print(f"steps in trace       {s['trace_steps']}")
    print(
        "ms/step              mean {:.2f}  p50 {:.2f}  p90 {:.2f}  p99 {:.2f}".format(
            s["step_ms"]["mean"],
            s["step_ms"]["p50"],
            s["step_ms"]["p90"],
            s["step_ms"]["p99"],
        )
    )
    print(
        "kernels/step         mean {:.1f}  p50 {:.0f}  p90 {:.0f}  p99 {:.0f}".format(
            s["kernels_per_step"]["mean"],
            s["kernels_per_step"]["p50"],
            s["kernels_per_step"]["p90"],
            s["kernels_per_step"]["p99"],
        )
    )
    print(f"us/kernel            {s['us_per_kernel']:.1f}")
    print(
        "device busy          {:.2f} ms/step ({:.0%} of step wall)".format(
            s["device_busy_ms_per_step"], s["device_busy_fraction"]
        )
    )
    # Host time is not kernel time and not idle either; name it so the three
    # add up instead of leaving an unexplained remainder.
    host = s["step_ms"]["mean"] - s["device_busy_ms_per_step"]
    print(f"host/other           {host:.2f} ms/step")
    print("\ntop by total GPU time:")
    for name, (count, total_us) in s["top_by_time"]:
        print(
            f"  {total_us / 1000.0:8.3f} ms  {count:6d}x  {total_us / count:7.1f} us  {name}"
        )
    if kernel_filter:
        print(f"\nkernels matching {kernel_filter!r}:")
        if not s["matches"]:
            print(
                "  NONE -- the fast path is registered but not selected, or not captured"
            )
        for name, (count, total_us) in s["matches"].items():
            print(
                f"  {total_us / 1000.0:8.3f} ms  {count:6d}x  {total_us / count:7.1f} us  {name}"
            )


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--step-fn", default=None)
    ap.add_argument("--kernel", default=None, help="substring to report separately")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    events = load_events(args.trace)
    step_fn = pick_step_fn(events, args.step_fn)
    summary = summarize(events, step_fn, args.top, args.kernel)
    report(summary, step_fn, args.kernel)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"step_fn": step_fn, **summary}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
