#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Aggregate the direct-seam quantized A/B into one table.

Usage::

    summarize_direct_quant.py <results-dir> [fp8|nvfp4]

Reads ``<results-dir>/raw/<mode>_<arm>_r<n>.json`` written by
``bench_quant_arm.py`` and prints, per concurrency level, each arm's median
output throughput across rounds, its within-arm spread, and its delta against
the ``vllm`` arm. Also folds in the per-process selection dumps so the table
carries the *proof* of which kernel ran, not just the timing.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

ARM_ORDER = ["vllm", "ours", "opaque", "ctrl", "tax", "taxh"]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    mode = sys.argv[2] if len(sys.argv) > 2 else "fp8"

    per_arm: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    sanity: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(outdir, "raw", f"{mode}_*_r*.json"))):
        label = os.path.basename(path)[: -len(".json")]
        arm = label[len(mode) + 1 :].rsplit("_r", 1)[0]
        data = json.load(open(path))
        for conc, run in data["runs"].items():
            # Every rep, not just the round median: the spread is the evidence
            # that a delta is outside noise.
            per_arm[arm][conc].extend(s["out_tok_per_s"] for s in run["samples"])
        sanity.setdefault(arm, data.get("sanity", ""))

    if not per_arm:
        print(f"no results for mode={mode} in {outdir}")
        return 1

    concs = sorted({c for arm in per_arm.values() for c in arm}, key=int)
    arms = [a for a in ARM_ORDER if a in per_arm] + [
        a for a in sorted(per_arm) if a not in ARM_ORDER
    ]

    print(f"\n== {mode}: output tok/s, median of all reps (spread = min..max) ==")
    header = f"{'arm':<8}" + "".join(f"{'C=' + c:>28}" for c in concs)
    print(header)
    base = {c: _median(per_arm["vllm"][c]) for c in concs if per_arm.get("vllm")}
    for arm in arms:
        cells = []
        for c in concs:
            vals = per_arm[arm].get(c)
            if not vals:
                cells.append(f"{'-':>28}")
                continue
            med = _median(vals)
            delta = f"{100 * (med - base[c]) / base[c]:+.2f}%" if base else "n/a"
            cells.append(f"{med:>10.1f} [{min(vals):.0f}-{max(vals):.0f}] {delta:>7}")
        print(f"{arm:<8}" + "".join(cells))

    print("\n== kernel selection (per arm, from the worker selection dumps) ==")
    for arm in arms:
        counts: collections.Counter = collections.Counter()
        fast = 0
        for path in sorted(
            glob.glob(os.path.join(outdir, "selection", f"{mode}_{arm}_r*.*.json"))
        ):
            data = json.load(open(path))
            for entry in data.get("selection", []):
                if entry.get("quant"):
                    counts[entry["kernel"]] += 1
            for stats in data.get("counters", {}).values():
                fast += stats.get("fast", 0)
        print(f"{arm:<8} fast={fast:<6} {dict(counts) or '{}'}")

    print("\n== greedy sanity continuation ==")
    for arm in arms:
        print(f"{arm:<8} {sanity.get(arm, '')[:90]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
