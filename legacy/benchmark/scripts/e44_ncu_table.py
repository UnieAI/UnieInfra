#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""E44 step 2 reader: turn an `ncu --page details --csv` export into the
per-kernel roofline table used to rank throughput levers.

Groups launches by (kernel, grid, block) -- one row per distinct launch shape,
median over the profiled launches -- and prints the counters E26/E33 used so
the vLLM numbers sit directly alongside the retiring engine's.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict

WANT = [
    "gpu__time_duration.sum",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__bytes.sum",
    "launch__registers_per_thread",
    "launch__waves_per_multiprocessor",
    "Duration",
    "Achieved Occupancy",
    "Memory Throughput",
    "Compute (SM) Throughput",
    "DRAM Throughput",
    "L2 Hit Rate",
    "Registers Per Thread",
    "Waves Per SM",
    "Theoretical Occupancy",
    "Block Limit Registers",
    "Block Limit Shared Mem",
]


def label(name: str) -> str:
    m = re.search(r"cutlass_\w*?gemm\w*?_(\d+x\d+_\d+x\d+)", name)
    if m:
        return f"cutlass_{m.group(1)}"
    for pat in (r"flash::(flash_fwd_\w+)", r"cublasLt::(\w+)", r"(triton_\w+)"):
        mm = re.search(pat, name)
        if mm:
            return mm.group(1)
    return name[:50]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--map", default=None, help="json mapping 'label|grid' -> role, e.g. qkv_proj"
    )
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv_path)))
    # key: (kernel label, grid, block) -> metric -> [values]
    agg: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    rules: dict[tuple, set] = defaultdict(set)
    launches: dict[tuple, set] = defaultdict(set)
    for r in rows:
        k = (label(r["Kernel Name"]), r["Grid Size"], r["Block Size"])
        launches[k].add(r["ID"])
        mn, mv, mu = r["Metric Name"], r["Metric Value"], r.get("Metric Unit", "")
        if mn in WANT and mv not in ("", "n/a", None):
            try:
                agg[k][mn].append((float(mv.replace(",", "")), mu))
            except ValueError:
                pass
        rd = r.get("Rule Description", "")
        if rd and r.get("Rule Name", "").lower().startswith(
            ("sol", "bottleneck", "roofline", "memory", "occupancy")
        ):
            rules[k].add(f"[{r['Rule Name']}] {rd[:220]}")

    def med(k, m, default=None):
        v = agg[k].get(m)
        if not v:
            return default
        return statistics.median(x[0] for x in v)

    def unit(k, m):
        v = agg[k].get(m)
        return v[0][1] if v else ""

    out = []
    hdr = (
        f"{'kernel':<28} {'grid':>14} {'blk':>10} {'n':>3} {'dur us':>9} "
        f"{'DRAM%':>7} {'SM%':>7} {'TC%':>7} {'L2hit%':>7} {'warps%':>7} "
        f"{'regs':>5} {'waves':>7} {'DRAM MB':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(agg, key=lambda x: -(med(x, "Duration", 0) or 0)):
        lab, grid, blk = k
        dur = med(k, "Duration")
        du = unit(k, "Duration")
        if du == "msecond":
            dur = dur * 1000 if dur else dur
        elif du == "nsecond":
            dur = dur / 1000 if dur else dur
        mb = med(k, "dram__bytes.sum")
        if unit(k, "dram__bytes.sum") == "Gbyte":
            mb = mb * 1000 if mb else mb
        row = {
            "kernel": lab,
            "grid": grid,
            "block": blk,
            "n_launches": len(launches[k]),
            "duration_us": dur,
            "dram_pct_peak": med(k, "DRAM Throughput")
            or med(k, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
            "sm_pct_peak": med(k, "Compute (SM) Throughput")
            or med(k, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            "tensor_pipe_pct": med(
                k, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
            ),
            "l2_hit_pct": med(k, "L2 Hit Rate") or med(k, "lts__t_sector_hit_rate.pct"),
            "warps_active_pct": med(k, "Achieved Occupancy")
            or med(k, "sm__warps_active.avg.pct_of_peak_sustained_active"),
            "theoretical_occ_pct": med(k, "Theoretical Occupancy"),
            "regs_per_thread": med(k, "Registers Per Thread")
            or med(k, "launch__registers_per_thread"),
            "waves_per_sm": med(k, "Waves Per SM")
            or med(k, "launch__waves_per_multiprocessor"),
            "dram_mbytes": mb,
            "rules": sorted(rules[k]),
        }
        out.append(row)
        f = lambda v, w=7, p=2: (
            f"{v:>{w}.{p}f}" if v is not None else " " * (w - 1) + "-"
        )
        print(
            f"{lab:<28} {grid:>14} {blk:>10} {row['n_launches']:>3} "
            f"{f(dur,9,2)} {f(row['dram_pct_peak'])} {f(row['sm_pct_peak'])} "
            f"{f(row['tensor_pipe_pct'])} {f(row['l2_hit_pct'])} "
            f"{f(row['warps_active_pct'])} {f(row['regs_per_thread'],5,0)} "
            f"{f(row['waves_per_sm'])} {f(mb,9,1)}"
        )

    print()
    for row in out:
        if row["rules"]:
            print(f"== {row['kernel']} {row['grid']}")
            for r in row["rules"]:
                print(f"   {r}")

    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
