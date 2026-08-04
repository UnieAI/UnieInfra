#!/usr/bin/env python3
"""E39: repeat-aware base-vs-MTP comparison, split by DATASET.

`recut_aggregate.py` collapses repeats per (engine, concurrency). E39 adds a
third axis -- the *dataset* -- because the whole question is whether the MTP
gain depends on what the workload looks like. So this script pivots

    results/unieinfra/e39_<arm>_<dataset>_r<rep>/c<C>.json

into one median-with-[min,max] cell per (dataset, concurrency, arm, metric),
and puts the agent-shaped datasets next to the RandomDataset control that
reproduces our E1-E38 history.

Verdict rule (same conservative rule as recut_aggregate.py):
  * TIE if the two arms' observed [min, max] ranges OVERLAP -- the delta is not
    separable from run-to-run noise.
  * WIN/LOSE otherwise, direction from the medians.

Usage:
    benchmark/e39_aggregate.py [--prefix e39] [--out results/E39_AGGREGATE.md]

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "unieinfra"

DATASETS = [
    ("specbench", "SpecBench (agent/assistant mix)"),
    ("gsm8k", "GSM8K (reasoning)"),
    ("random", "RandomDataset ILEN1024/OLEN256 (E1-E38 CONTROL)"),
]
ARMS = ["base", "mtp"]
METRICS = [
    ("output_throughput", "out tok/s", "hi"),
    ("mean_ttft_ms", "TTFT mean ms", "lo"),
    ("mean_tpot_ms", "TPOT mean ms", "lo"),
]
SPEC_METRICS = [
    ("spec_decode_acceptance_length", "accept len (/4)"),
    ("spec_decode_acceptance_rate", "accept rate %"),
]


def collect(prefix: str) -> dict:
    """(dataset, conc, arm, metric) -> list of values, one per repeat."""
    out: dict = {}
    for d in sorted(RESULTS.glob(f"{prefix}_*_r*")):
        parts = d.name.split("_")
        # e39_<arm>_<dataset>_r<rep>
        if len(parts) < 4 or not parts[-1].startswith("r"):
            continue
        arm, dataset = parts[1], "_".join(parts[2:-1])
        for jf in sorted(d.glob("c*.json")):
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            if not data.get("completed"):
                continue  # a failed run contributes nothing, it does not zero the median
            conc = data.get("max_concurrency")
            for key, *_ in METRICS + SPEC_METRICS:
                if key in data and isinstance(data[key], (int, float)):
                    out.setdefault((dataset, conc, arm, key), []).append(data[key])
    return out


def cell(vals: list[float]) -> str:
    if not vals:
        return "-"
    m = statistics.median(vals)
    return f"{m:.1f} [{min(vals):.1f}-{max(vals):.1f}]" if len(vals) > 1 else f"{m:.1f}"


def verdict(base: list[float], mtp: list[float], direction: str) -> str:
    """WIN = MTP is better than base. TIE when observed ranges overlap."""
    if not base or not mtp:
        return "-"
    bmed, mmed = statistics.median(base), statistics.median(mtp)
    delta = (mmed - bmed) / bmed * 100 if bmed else 0.0
    if direction == "lo":
        delta = -delta
    overlap = min(max(base), max(mtp)) >= max(min(base), min(mtp))
    if overlap and len(base) > 1 and len(mtp) > 1:
        return f"TIE ({delta:+.1f}%)"
    return f"{'WIN' if delta > 0 else 'LOSE'} ({delta:+.1f}%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="e39")
    ap.add_argument("--out", default=str(RESULTS.parent / "E39_AGGREGATE.md"))
    args = ap.parse_args()

    data = collect(args.prefix)
    concs = sorted({k[1] for k in data if k[1] is not None})
    lines = [
        "# E39 — MTP on agent-shaped workloads vs the RandomDataset control",
        "",
        "Qwen3.5-9B bf16, sm_120, GPU 2 only, one engine at a time. Arms differ ONLY",
        "by `--speculative-config` (MTP k=3, i.e. 4 positions). Greedy (temperature 0),",
        "so MTP is output-lossless and the arms emit the same tokens.",
        "Cells are median [min-max] over repeats. TIE = observed ranges overlap.",
        "",
    ]

    for ds, dstitle in DATASETS:
        if not any(k[0] == ds for k in data):
            continue
        lines += [f"## {dstitle}", ""]
        header = "| metric | C | base | +MTP | verdict |"
        lines += [header, "|---|---|---|---|---|"]
        for key, label, direction in METRICS:
            for c in concs:
                b = data.get((ds, c, "base", key), [])
                m = data.get((ds, c, "mtp", key), [])
                if not b and not m:
                    continue
                lines.append(
                    f"| {label} | {c} | {cell(b)} | {cell(m)} | {verdict(b, m, direction)} |"
                )
        lines.append("")
        # acceptance is MTP-only -- it is the MECHANISM behind the rows above
        lines += [
            "Acceptance (MTP arm only):",
            "",
            "| metric | C | value |",
            "|---|---|---|",
        ]
        for key, label in SPEC_METRICS:
            for c in concs:
                v = data.get((ds, c, "mtp", key), [])
                if v:
                    lines.append(f"| {label} | {c} | {cell(v)} |")
        lines.append("")

    # The headline table: MTP throughput gain, agent-shaped vs control, side by side.
    lines += [
        "## THE COMPARISON THAT ANSWERS THE QUESTION",
        "",
        "MTP output-throughput gain over baseline, by dataset:",
        "",
        "| C | " + " | ".join(t for _, t in DATASETS) + " |",
        "|---|" + "---|" * len(DATASETS),
    ]
    for c in concs:
        row = [str(c)]
        for ds, _ in DATASETS:
            b = data.get((ds, c, "base", "output_throughput"), [])
            m = data.get((ds, c, "mtp", "output_throughput"), [])
            if not b or not m:
                row.append("-")
                continue
            bmed, mmed = statistics.median(b), statistics.median(m)
            al = data.get((ds, c, "mtp", "spec_decode_acceptance_length"), [])
            altxt = f", AL {statistics.median(al):.2f}" if al else ""
            row.append(f"{(mmed - bmed) / bmed * 100:+.1f}%{altxt}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    out = Path(args.out)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
