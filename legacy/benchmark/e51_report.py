#!/usr/bin/env python3
"""Turn E51's per-arm result files into the comparison table.

Reads every ``<arm>_c<N>.json`` written by ``vllm bench serve`` and reports each
arm against the ``stock`` baseline at matching concurrency. Deltas are always
signed the way a reader expects: higher is better for throughput, lower is better
for latency, and the sign in the table already accounts for that.

Usage:
    python3 benchmark/e51_report.py [result_dir]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BASELINE = "stock"
ARM_ORDER = ["stock", "mtp", "fp8kv", "unieinfra"]

# (json key, column header, whether a larger value is better)
METRICS = [
    ("output_throughput", "out tok/s", True),
    ("request_throughput", "req/s", True),
    ("mean_ttft_ms", "TTFT mean", False),
    ("p99_ttft_ms", "TTFT p99", False),
    ("mean_tpot_ms", "TPOT mean", False),
    ("p99_tpot_ms", "TPOT p99", False),
    ("mean_itl_ms", "ITL mean", False),
    ("mean_e2el_ms", "E2E mean", False),
]


def load(result_dir: Path) -> dict[tuple[str, int], dict]:
    """Map ``(arm, concurrency)`` to the parsed result for that run."""
    out: dict[tuple[str, int], dict] = {}
    for path in sorted(result_dir.glob("*_c*.json")):
        m = re.fullmatch(r"(.+)_c(\d+)\.json", path.name)
        if not m:
            continue
        try:
            out[(m.group(1), int(m.group(2)))] = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  (skipping unreadable {path.name})", file=sys.stderr)
    return out


def fmt(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.1f}" if abs(value) >= 100 else f"{value:.2f}"


def delta(arm_value: float | None, base: float | None, higher_better: bool) -> str:
    """Percentage change against the baseline, signed so + always means better."""
    if arm_value is None or not base:
        return "--"
    pct = (arm_value - base) / base * 100.0
    if not higher_better:
        pct = -pct
    return f"{pct:+.1f}%"


def main(result_dir: Path) -> int:
    runs = load(result_dir)
    if not runs:
        print(f"no results in {result_dir}", file=sys.stderr)
        return 1

    arms = [a for a in ARM_ORDER if any(k[0] == a for k in runs)]
    arms += sorted({k[0] for k in runs} - set(arms))
    concurrencies = sorted({k[1] for k in runs})

    print(f"E51 — stock vLLM vs UnieInfra   ({result_dir})")
    print(f"arms: {', '.join(arms)}   concurrency: {concurrencies}\n")

    for key, header, higher_better in METRICS:
        print(f"## {header}" + ("   (higher is better)" if higher_better
                                else "   (lower is better)"))
        width = max(len(a) for a in arms) + 2
        print("  " + "arm".ljust(width) + "".join(f"{'C=' + str(c):>20}"
                                                  for c in concurrencies))
        for arm in arms:
            row = "  " + arm.ljust(width)
            for c in concurrencies:
                run = runs.get((arm, c))
                value = run.get(key) if run else None
                cell = fmt(value)
                if arm != BASELINE:
                    base_run = runs.get((BASELINE, c))
                    base = base_run.get(key) if base_run else None
                    cell += f" ({delta(value, base, higher_better)})"
                row += f"{cell:>20}"
            print(row)
        print()

    # Acceptance rate is the number that explains whether MTP paid off, and it is
    # absent from arms that ran no speculation -- so report it separately rather
    # than as a column of dashes.
    spec_rows = [
        (arm, c, runs[(arm, c)])
        for arm in arms
        for c in concurrencies
        if (arm, c) in runs and runs[(arm, c)].get("accept_rate") is not None
    ]
    if spec_rows:
        print("## speculative acceptance")
        for arm, c, run in spec_rows:
            print(f"  {arm:12s} C={c:<5d} accept_rate={run['accept_rate']}")
        print()

    missing = [(a, c) for a in arms for c in concurrencies if (a, c) not in runs]
    if missing:
        # Say what is absent rather than letting a gap read as a complete sweep.
        print("## missing runs (arm, concurrency)")
        print("  " + ", ".join(f"{a}@C={c}" for a, c in missing))
    return 0


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1
                else Path(__file__).parent / "results" / "e51")
    raise SystemExit(main(root))
