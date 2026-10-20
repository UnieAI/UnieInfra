#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""E45 reader: incumbent-vs-ours launch geometry from an `ncu --csv` export.

The point of the table is the *mechanism*, not the clock. E44 attributed the
incumbent's loss to wave quantization; the counters that carry that claim are
the launch statistics -- grid size, CTAs resident per SM, waves per SM, achieved
occupancy -- because they are properties of the launch configuration and do not
move when another job shares the card. Timing belongs in
``bench_decode_batch_gemm.py --mode compare``, which A/Bs interleaved.

Usage::

    python benchmark/scripts/e45_ncu_table.py benchmark/profiles/e45/*.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re

WANT = {
    "gpu__time_duration.sum": "us",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram%",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm%",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "tensor%",
    "launch__grid_size": "grid",
    "launch__block_size": "block",
    "launch__waves_per_multiprocessor": "waves",
    "launch__occupancy_limit_shared_mem": "cta/sm(smem)",
    "launch__occupancy_limit_registers": "cta/sm(reg)",
    "sm__maximum_warps_per_active_cycle_pct": "occ%",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "warps%",
    "launch__registers_per_thread": "regs",
    "dram__bytes.sum": "dram_bytes",
    "lts__t_sector_hit_rate.pct": "l2hit%",
}


def label(name: str) -> str:
    """Short, stable name for a kernel: the CUTLASS tile, or ours."""
    m = re.search(r"cutlass_\w*?gemm\w*?_(\d+x\d+_\d+x\d+)", name)
    if m:
        return f"cutlass_{m.group(1)}"
    if "_decode_batch_gemm_kernel" in name:
        return "unieinfra_batch_gemm"
    if "_split_k_reduce_kernel" in name:
        return "unieinfra_split_k_reduce"
    m = re.search(r"cublasLt::(\w+)", name)
    if m:
        return f"cublasLt::{m.group(1)}"
    return name[:44]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_files", nargs="+")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    # One row per *launch*, keyed on the ncu launch ID. Aggregating by kernel
    # name -- or even by name plus grid -- silently averages unrelated
    # projections: ``down_proj`` and ``o_proj`` are served by the same CUTLASS
    # tile at the same grid, and their medians differ by 3x in DRAM bytes. The
    # harness issues the contenders in a fixed shape order, so the launch ID is
    # what attributes a row to a projection.
    launches: dict[int, dict[str, object]] = {}
    for path in args.csv_files:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    launch_id = int(row["ID"])
                except (KeyError, ValueError):
                    continue
                rec = launches.setdefault(
                    launch_id,
                    {"kernel": label(row.get("Kernel Name", "")), "metrics": {}},
                )
                metric = row.get("Metric Name", "")
                if metric not in WANT:
                    continue
                try:
                    rec["metrics"][metric] = float(row["Metric Value"].replace(",", ""))
                except (KeyError, ValueError):
                    continue

    cols = list(WANT)
    header = f"{'#':>3s} {'kernel':26s} " + " ".join(f"{WANT[c]:>11s}" for c in cols)
    print(header)
    print("-" * len(header))
    rows = []
    for launch_id in sorted(launches):
        rec = launches[launch_id]
        metrics = rec["metrics"]
        vals = {c: metrics.get(c, float("nan")) for c in cols}
        print(
            f"{launch_id:3d} {rec['kernel']:26s} "
            + " ".join(f"{vals[c]:11.2f}" for c in cols)
        )
        rows.append(
            {
                "launch_id": launch_id,
                "kernel": rec["kernel"],
                **{WANT[c]: vals[c] for c in cols},
            }
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
