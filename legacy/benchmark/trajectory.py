#!/usr/bin/env python3
"""Trajectory view: scan every saved sweep and show "did the change help?" at a glance.

Scans `results/<engine>/<label>/summary.csv` (as written by run_sweep.sh /
summarize.py — see benchmark/README.md) for ALL engines under `results/`
(unieinfra, vllm, sglang, hc, ...) and pivots them into one table:

    rows    = run (engine/label), ordered chronologically (by the `date` column)
    columns = concurrency (1/16/32/64/128/256 by default)
    cells   = a chosen metric (output_throughput by default; mean_ttft_ms /
              mean_tpot_ms selectable via --metric)

On top of the raw pivot it adds:
  * a delta-vs-baseline table (default baseline = unieinfra/clean) so a
    regression or win jumps out without doing mental math.
  * a side-by-side UnieInfra vs vLLM vs SGLang comparison at each concurrency
    (using each engine's "clean" run) with a win/lose marker, because the
    product bar is "beat vLLM/SGLang", not just "improve on yesterday".

Ragged runs (a label that only swept a subset of concurrencies, e.g.
fp8w_fixed only has C=1/32/128) show "-" for the missing cells -- this script
never crashes on missing data, it just reports what it has.

Output: BOTH a machine-readable CSV (the raw pivot) and a human-readable
Markdown report (pivot + delta + comparison) written to
`results/TRAJECTORY.md` by default.

Usage:
    trajectory.py                                   # metric=output_throughput, baseline=clean
    trajectory.py --metric mean_ttft_ms
    trajectory.py --baseline before_baseline
    trajectory.py --focus fp8w_fixed                # also prints a standalone
                                                      # delta report for one label
                                                      # (used by bench_and_profile.sh)
    trajectory.py --dry-run                          # same as default; nothing here
                                                      # touches a GPU or server, so
                                                      # --dry-run is accepted as a no-op
                                                      # for symmetry with bench_and_profile.sh

Python stdlib only (csv, argparse, pathlib) -- no pandas/matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_CONCURRENCIES = [1, 16, 32, 64, 128, 256]

# Metrics where a SMALLER value is better (everything else -> bigger is better).
LOWER_IS_BETTER = {
    "mean_ttft_ms",
    "p99_ttft_ms",
    "median_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "median_tpot_ms",
    "mean_itl_ms",
    "mean_e2el_ms",
    "p99_e2el_ms",
}

MISSING = "-"


def eprint(*a, **kw):
    print(*a, file=sys.stderr, **kw)


class Run:
    """One results/<engine>/<label>/summary.csv, pivoted to {concurrency: value}."""

    def __init__(self, engine: str, label: str, csv_path: Path):
        self.engine = engine
        self.label = label
        self.path = csv_path
        self.by_conc: dict[int, dict] = {}
        self.date = ""

    @property
    def key(self) -> str:
        return f"{self.engine}/{self.label}"

    def load(self, metric: str):
        try:
            with open(self.path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        c = int(float(row.get("max_concurrency", "")))
                    except (TypeError, ValueError):
                        continue
                    self.by_conc[c] = row
                    d = row.get("date") or ""
                    if d and (not self.date or d < self.date):
                        self.date = d
        except (OSError, csv.Error) as e:
            eprint(f"[trajectory] WARN: could not read {self.path}: {e}")

    def value(self, conc: int, metric: str):
        row = self.by_conc.get(conc)
        if not row:
            return None
        raw = row.get(metric)
        if raw in (None, ""):
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def fmt(self, conc: int, metric: str) -> str:
        v = self.value(conc, metric)
        return MISSING if v is None else f"{v:g}"


def discover_runs(results_dir: Path) -> list[Run]:
    runs = []
    for csv_path in sorted(results_dir.glob("**/summary.csv")):
        rel = csv_path.relative_to(results_dir)
        parts = rel.parts[:-1]  # drop "summary.csv"
        if len(parts) < 2:
            eprint(
                f"[trajectory] WARN: skipping {csv_path} (expected <engine>/<label>/summary.csv)"
            )
            continue
        engine, label = parts[0], "/".join(parts[1:])
        runs.append(Run(engine, label, csv_path))
    return runs


def better(a, b, metric: str):
    """Return True if value `a` beats value `b` for this metric. None never wins."""
    if a is None:
        return False
    if b is None:
        return True
    if metric in LOWER_IS_BETTER:
        return a < b
    return a > b


def pct_delta(new, base, metric: str):
    if new is None or base is None or base == 0:
        return None
    d = (new - base) / abs(base) * 100.0
    return d


def render_pivot_md(runs: list[Run], concurrencies: list[int], metric: str) -> str:
    header = ["run"] + [f"C{c}" for c in concurrencies]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in runs:
        cells = [r.key] + [r.fmt(c, metric) for c in concurrencies]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_delta_md(
    runs: list[Run], concurrencies: list[int], metric: str, baseline: Run
) -> str:
    header = ["run"] + [f"C{c}" for c in concurrencies]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in runs:
        cells = [r.key]
        for c in concurrencies:
            if r is baseline:
                cells.append("baseline")
                continue
            d = pct_delta(r.value(c, metric), baseline.value(c, metric), metric)
            if d is None:
                cells.append(MISSING)
            else:
                sign = "+" if d >= 0 else ""
                good = (d < 0) if metric in LOWER_IS_BETTER else (d > 0)
                marker = "better" if good else ("worse" if d != 0 else "flat")
                cells.append(f"{sign}{d:.1f}% ({marker})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_comparison_md(
    by_key: dict[str, Run], concurrencies: list[int], metric: str, unie_label: str
) -> str:
    unie = by_key.get(f"unieinfra/{unie_label}")
    vllm = by_key.get("vllm/clean")
    sglang = by_key.get("sglang/clean")
    if unie is None and vllm is None and sglang is None:
        return "_(no unieinfra/vllm/sglang runs found to compare)_"
    header = [
        "concurrency",
        f"unieinfra/{unie_label}",
        "vllm/clean",
        "sglang/clean",
        "winner",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for c in concurrencies:
        uv = unie.value(c, metric) if unie else None
        vv = vllm.value(c, metric) if vllm else None
        sv = sglang.value(c, metric) if sglang else None
        candidates = {"unieinfra": uv, "vllm": vv, "sglang": sv}
        present = {k: v for k, v in candidates.items() if v is not None}
        if not present:
            winner = MISSING
        else:
            if metric in LOWER_IS_BETTER:
                winner = min(present, key=present.get)
            else:
                winner = max(present, key=present.get)
        marker = winner
        if winner == "unieinfra":
            marker = "unieinfra WINS"
        elif winner != MISSING:
            marker = f"{winner} beats unieinfra"
        row = [
            f"C{c}",
            unie.fmt(c, metric) if unie else MISSING,
            vllm.fmt(c, metric) if vllm else MISSING,
            sglang.fmt(c, metric) if sglang else MISSING,
            marker,
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_csv(runs: list[Run], concurrencies: list[int], metric: str, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["engine", "label", "date", "metric"] + [f"C{c}" for c in concurrencies]
        )
        for r in runs:
            w.writerow(
                [r.engine, r.label, r.date, metric]
                + [r.fmt(c, metric) for c in concurrencies]
            )


def print_focus_report(
    runs_by_key: dict[str, Run],
    baseline: Run,
    focus_key: str,
    concurrencies: list[int],
    metric: str,
):
    r = runs_by_key.get(focus_key)
    if r is None:
        eprint(
            f"[trajectory] --focus {focus_key}: no such run found under results/**/summary.csv"
        )
        return
    print(f"[trajectory] {r.key} vs baseline={baseline.key} (metric={metric}):")
    any_regression = False
    for c in concurrencies:
        newv = r.value(c, metric)
        basev = baseline.value(c, metric)
        d = pct_delta(newv, basev, metric)
        if newv is None:
            print(f"  C{c:<4}: (no data)")
            continue
        if d is None:
            print(f"  C{c:<4}: {newv:g} (no baseline to compare)")
            continue
        good = (d < 0) if metric in LOWER_IS_BETTER else (d > 0)
        sign = "+" if d >= 0 else ""
        if not good and abs(d) > 1.0:  # >1% adverse move = flag it
            any_regression = True
            print(
                f"  C{c:<4}: {basev:g} -> {newv:g}  ({sign}{d:.1f}%)  REGRESSION at C{c}: {sign}{d:.1f}% vs baseline"
            )
        else:
            print(f"  C{c:<4}: {basev:g} -> {newv:g}  ({sign}{d:.1f}%)")
    if any_regression:
        print(
            f"[trajectory] REGRESSION FLAG: {r.key} regressed vs {baseline.key} on at least one concurrency"
        )
    else:
        print(f"[trajectory] no regression vs {baseline.key}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    default_results = Path(__file__).resolve().parent / "results"
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help=f"root to scan for **/summary.csv (default: {default_results})",
    )
    ap.add_argument(
        "--metric",
        default="output_throughput",
        choices=["output_throughput", "mean_ttft_ms", "mean_tpot_ms"],
        help="which summary.csv column to pivot on (default: output_throughput)",
    )
    ap.add_argument(
        "--baseline",
        default="clean",
        help="unieinfra label to diff every unieinfra run against (default: clean)",
    )
    ap.add_argument(
        "--concurrencies",
        default=",".join(str(c) for c in DEFAULT_CONCURRENCIES),
        help="comma-separated concurrency columns (default: 1,16,32,64,128,256)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="default: <results-dir>/trajectory.csv",
    )
    ap.add_argument(
        "--out-md", type=Path, default=None, help="default: <results-dir>/TRAJECTORY.md"
    )
    ap.add_argument(
        "--focus",
        default=None,
        help="engine/label (or bare label, assumed unieinfra/<label>) to print a "
        "standalone delta+regression report for on stdout",
    )
    ap.add_argument(
        "--no-write",
        action="store_true",
        help="skip writing the CSV/Markdown files; only print (used with --focus for "
        "a quick one-off diff, e.g. bench_and_profile.sh's vs-previous-label check)",
    )
    ap.add_argument(
        "--list-runs",
        action="store_true",
        help="print 'engine/label' for every discovered run, one per line, in the same "
        "chronological order as the table rows, then exit (no files written; used "
        "by bench_and_profile.sh to find the previous label for a given engine)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="no-op flag accepted for symmetry with bench_and_profile.sh --dry-run "
        "(this script never touches a GPU or server)",
    )
    args = ap.parse_args()

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    out_csv = args.out_csv or (args.results_dir / "trajectory.csv")
    out_md = args.out_md or (args.results_dir / "TRAJECTORY.md")

    if not args.results_dir.exists():
        eprint(f"[trajectory] results dir not found: {args.results_dir}")
        sys.exit(1)

    runs = discover_runs(args.results_dir)
    if not runs:
        eprint(f"[trajectory] no results/**/summary.csv found under {args.results_dir}")
        sys.exit(1)
    for r in runs:
        r.load(args.metric)

    if args.list_runs:
        runs.sort(key=lambda r: (r.date, r.key))
        for r in runs:
            print(r.key)
        return

    # Chronological order (by earliest `date` seen in the run); runs with no
    # date (older/hand-made dirs) sort first via empty-string < any timestamp,
    # then fall back to key for stability.
    runs.sort(key=lambda r: (r.date, r.key))

    by_key = {r.key: r for r in runs}

    baseline_key = (
        args.baseline if "/" in args.baseline else f"unieinfra/{args.baseline}"
    )
    baseline = by_key.get(baseline_key)

    if args.no_write:
        if args.focus:
            focus_key = args.focus if "/" in args.focus else f"unieinfra/{args.focus}"
            if baseline is None:
                eprint(
                    f"[trajectory] --focus requested but baseline {baseline_key} not found; cannot compute delta"
                )
            else:
                print_focus_report(
                    by_key, baseline, focus_key, concurrencies, args.metric
                )
        return

    write_csv(runs, concurrencies, args.metric, out_csv)

    md = []
    md.append(
        f"<!-- Generated by benchmark/trajectory.py. Do not hand-edit; re-run the script instead. -->"
    )
    md.append(f"# UnieInfra Benchmark Trajectory\n")
    md.append(
        f"Metric: `{args.metric}` "
        f"({'lower is better' if args.metric in LOWER_IS_BETTER else 'higher is better'}). "
        f"Baseline for deltas: `{baseline_key}`{'' if baseline else ' (NOT FOUND -- delta table skipped)'}.\n"
    )
    md.append(f"Scanned `{len(runs)}` run(s) under `{args.results_dir}`.\n")

    md.append("## 1. Trajectory (all runs, all engines)\n")
    md.append(render_pivot_md(runs, concurrencies, args.metric))
    md.append("")

    if baseline is not None:
        unie_runs = [r for r in runs if r.engine == "unieinfra" or r is baseline]
        md.append(f"\n## 2. Delta vs baseline (`{baseline_key}`)\n")
        md.append(render_delta_md(unie_runs, concurrencies, args.metric, baseline))
        md.append("")
    else:
        md.append(
            f"\n## 2. Delta vs baseline\n\n_(baseline `{baseline_key}` not found among scanned runs -- skipped)_\n"
        )

    unie_label_for_compare = (
        args.baseline if "/" not in args.baseline else args.baseline.split("/", 1)[1]
    )
    md.append(
        f"\n## 3. UnieInfra vs vLLM vs SGLang (using unieinfra/{unie_label_for_compare}, vllm/clean, sglang/clean)\n"
    )
    md.append(
        render_comparison_md(by_key, concurrencies, args.metric, unie_label_for_compare)
    )
    md.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n")

    print(f"[trajectory] wrote {out_csv}")
    print(f"[trajectory] wrote {out_md}")

    if args.focus:
        focus_key = args.focus if "/" in args.focus else f"unieinfra/{args.focus}"
        if baseline is None:
            eprint(
                f"[trajectory] --focus requested but baseline {baseline_key} not found; cannot compute delta"
            )
        else:
            print()
            print_focus_report(by_key, baseline, focus_key, concurrencies, args.metric)


if __name__ == "__main__":
    main()
