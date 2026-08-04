#!/usr/bin/env python3
"""Repeat-aware comparison: median + min/max spread per (engine, concurrency).

`trajectory.py` shows one row per *run*, which cannot answer "is this delta
bigger than run-to-run noise?". This script collapses N repeat runs that share
a label prefix (`<prefix>_<engine>_r1`, `_r2`, ...) into one median-with-spread
cell per (engine, concurrency, metric), then issues a WIN/TIE/LOSE verdict for
UnieInfra against each competitor.

Verdict rule (stated up front, deliberately conservative):
  * TIE  if the two engines' [min, max] observed ranges OVERLAP at that
    concurrency -- the delta is not separable from run-to-run noise.
  * WIN/LOSE otherwise (ranges disjoint), direction taken from the medians.

Two layouts are supported, because two kinds of comparison matter:

  * **engine** (default) -- `results/<engine>/<prefix>_<engine>_r*/summary.csv`,
    one column per engine, verdicts for UnieInfra against each competitor.
  * **arm** -- `results/<engine>/<prefix>_<arm>_r*/summary.csv`, one column per
    *policy arm* inside a single engine. This is the layout a scheduling A/B
    needs: every arm is the same binary with the same compiled graphs, so the
    only thing the columns can differ by is the policy.

Usage:
    benchmark/recut_aggregate.py [label-prefix]     # default: recut_a_bf16
    benchmark/recut_aggregate.py e36 --arms stock,thr,inter,static \
        --engine vllm --baseline stock

Stdlib only.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

RESULTS = Path("/workspace/roy/UnieInfra/benchmark/results")
CONCS = [1, 16, 32, 64, 128, 256]
ENGINES = ["unieinfra", "vllm", "sglang"]

METRICS = [
    ("output_throughput", "out tok/s", "hi"),
    ("request_throughput", "req/s", "hi"),
    ("mean_ttft_ms", "TTFT mean ms", "lo"),
    ("p99_ttft_ms", "TTFT p99 ms", "lo"),
    ("mean_tpot_ms", "TPOT mean ms", "lo"),
    ("p99_tpot_ms", "TPOT p99 ms", "lo"),
    ("tok_s_per_user", "tok/s/user", "hi"),
]


def load(prefix: str, cols: list[str] | None = None, engine: str | None = None):
    """Collapse repeat runs into per-column value lists.

    Args:
        prefix: Label prefix shared by every run directory.
        cols: Column names to load. Engine names in the default layout, arm
            names when ``engine`` is given.
        engine: When set, every column is an *arm* of this one engine and runs
            are read from ``results/<engine>/<prefix>_<arm>_r*``.

    Returns:
        ``(data, runs)`` where ``data[col][conc][metric]`` is the list of values
        observed across repeats and ``runs[col]`` names the directories used.
    """
    cols = list(cols or ENGINES)
    data = {e: {c: {} for c in CONCS} for e in cols}
    runs = {e: [] for e in cols}
    for eng in cols:
        root = RESULTS / (engine or eng)
        for d in sorted(root.glob(f"{prefix}_{eng}_r*")):
            f = d / "summary.csv"
            if not f.exists():
                continue
            runs[eng].append(d.name)
            for row in csv.DictReader(f.open()):
                c = int(row["max_concurrency"])
                if c not in CONCS:
                    continue
                for m, _, _ in METRICS:
                    if m == "tok_s_per_user":
                        tp = float(row["mean_tpot_ms"])
                        v = 1000.0 / tp if tp else 0.0
                    else:
                        v = float(row[m])
                    data[eng][c].setdefault(m, []).append(v)
    return data, runs


def fmt(vals, nd=1):
    if not vals:
        return "-"
    med = statistics.median(vals)
    return f"{med:.{nd}f} [{min(vals):.{nd}f}-{max(vals):.{nd}f}]"


def _argval(flag: str, default: str | None = None) -> str | None:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    prefix = (
        sys.argv[1]
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
        else "recut_a_bf16"
    )
    arms = _argval("--arms")
    engine = _argval("--engine")
    cols = arms.split(",") if arms else list(ENGINES)
    baseline = _argval("--baseline", "unieinfra" if not arms else cols[0])
    data, runs = load(prefix, cols, engine if arms else None)
    print(f"# Re-cut aggregate: {prefix}\n")
    for eng in cols:
        print(f"  {eng}: {len(runs[eng])} repeats -> {runs[eng]}")
    print()

    for m, label, direction in METRICS:
        nd = 1 if "ms" in label or "tok" in label else 3
        print(
            f"\n## {label}  (median [min-max], {'higher' if direction=='hi' else 'lower'} is better)\n"
        )
        print("| C | " + " | ".join(cols) + " |")
        print("|---|" + "---|" * len(cols))
        for c in CONCS:
            cells = [fmt(data[e][c].get(m, []), nd) for e in cols]
            print(f"| {c} | " + " | ".join(cells) + " |")

    # ---- verdict on output_throughput and on tok/s/user ----
    for m, label, direction in [
        ("output_throughput", "output throughput", "hi"),
        ("mean_tpot_ms", "TPOT mean", "lo"),
        ("mean_ttft_ms", "TTFT mean", "lo"),
    ]:
        others = [e for e in cols if e != baseline]
        print(f"\n\n## VERDICT on {label} ({baseline} vs each)\n")
        print("| C | " + " | ".join(f"vs {o}" for o in others) + " |")
        print("|---|" + "---|" * len(others))
        for c in CONCS:
            u = data[baseline][c].get(m, [])
            out = []
            for opp in others:
                o = data[opp][c].get(m, [])
                if not u or not o:
                    out.append("-")
                    continue
                mu, mo = statistics.median(u), statistics.median(o)
                overlap = not (max(u) < min(o) or max(o) < min(u))
                delta = (mu - mo) / mo * 100.0
                if direction == "lo":
                    better = mu < mo
                else:
                    better = mu > mo
                pct = f"{delta:+.1f}%"
                if overlap:
                    out.append(f"TIE ({pct}, ranges overlap)")
                else:
                    out.append(("WIN " if better else "LOSE ") + pct)
            print(f"| {c} | " + " | ".join(out) + " |")


if __name__ == "__main__":
    main()
