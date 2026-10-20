# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the UnieInfra project
"""Summarize an E49 product sweep into the two tables the comparison needs.

Table 1 (product):     arm A (UnieInfra defaults) vs the stock-defaults arm.
Table 2 (attribution): A vs C, i.e. our defaults vs the same features switched
                       on by hand, so a win can be attributed to the right
                       cause.

A delta whose magnitude is inside the combined run-to-run spread of the two
arms is reported as **TIE**, not as a number.  Quoting a 1% win drawn from
arms that each wander 2% between repeats is how benchmarks lie.

Usage:
    python e49_summarize.py <results-dir> [--baseline B]
"""

import argparse
import json
import statistics
from pathlib import Path

ARM_LABELS = {
    "A": "UnieInfra (our defaults)",
    "B": "vLLM defaults (same binary, seams off)",
    "C": "vLLM + features enabled by hand",
    "D": "stock vLLM install (out of the box)",
}

METRICS = [
    ("output_throughput", "out tok/s", True),
    ("median_ttft_ms", "TTFT ms", False),
    ("median_tpot_ms", "TPOT ms", False),
    ("request_throughput", "req/s", True),
]


def load_runs(results_dir: Path) -> dict:
    """Collect {(arm, concurrency): [per-rep metric dicts]} from result JSONs."""
    runs: dict[tuple[str, int], list[dict]] = {}
    for path in sorted(results_dir.glob("*_r*_c*.json")):
        if "metrics" in path.name:
            continue
        stem = path.stem
        try:
            arm, rep, conc = stem.split("_")
            concurrency = int(conc.lstrip("c"))
        except ValueError:
            continue
        try:
            with path.open() as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        runs.setdefault((arm, concurrency), []).append(data)
    return runs


def spread_pct(values: list[float]) -> float:
    """Half the peak-to-peak range as a percentage of the median."""
    if len(values) < 2:
        return 0.0
    median = statistics.median(values)
    if median == 0:
        return 0.0
    return 100.0 * (max(values) - min(values)) / 2.0 / abs(median)


def summarize_metric(runs, arm, concurrency, key):
    entries = runs.get((arm, concurrency), [])
    values = [e[key] for e in entries if isinstance(e.get(key), (int, float))]
    if not values:
        return None
    return statistics.median(values), spread_pct(values), len(values)


def compare(runs, treatment, baseline, concurrencies, key, higher_is_better):
    """Rows of (C, baseline, treatment, delta-or-TIE)."""
    rows = []
    for concurrency in concurrencies:
        base = summarize_metric(runs, baseline, concurrency, key)
        treat = summarize_metric(runs, treatment, concurrency, key)
        if base is None or treat is None:
            rows.append((concurrency, base, treat, "n/a"))
            continue
        base_median, base_spread, _ = base
        treat_median, treat_spread, _ = treat
        if base_median == 0:
            rows.append((concurrency, base, treat, "n/a"))
            continue
        delta = 100.0 * (treat_median - base_median) / base_median
        if not higher_is_better:
            delta = -delta
        if abs(delta) <= (base_spread + treat_spread):
            verdict = f"TIE (±{base_spread + treat_spread:.1f}%)"
        else:
            verdict = f"{delta:+.1f}%"
        rows.append((concurrency, base, treat, verdict))
    return rows


def render_table(title, rows, unit):
    lines = [
        f"\n### {title}",
        "",
        f"| C | baseline ({unit}) | arm ({unit}) | delta |",
        "|---|---|---|---|",
    ]
    for concurrency, base, treat, verdict in rows:
        base_str = f"{base[0]:.1f} ±{base[1]:.1f}% (n={base[2]})" if base else "—"
        treat_str = f"{treat[0]:.1f} ±{treat[1]:.1f}% (n={treat[2]})" if treat else "—"
        lines.append(f"| {concurrency} | {base_str} | {treat_str} | {verdict} |")
    return "\n".join(lines)


def spec_engagement(results_dir: Path) -> str:
    """Prove MTP actually engaged rather than silently no-op'ing."""
    lines = [
        "\n### Speculative decoding engagement (draft/accepted tokens)",
        "",
        "| arm | C | drafted | accepted | acceptance |",
        "|---|---|---|---|---|",
    ]
    found = False
    for after in sorted(results_dir.glob("*_metrics_after.json")):
        before = after.with_name(after.name.replace("_after", "_before"))
        if not before.exists():
            continue
        try:
            a = json.loads(after.read_text())
            b = json.loads(before.read_text())
        except (OSError, ValueError):
            continue
        drafted = a["draft_tokens"] - b["draft_tokens"]
        accepted = a["accepted_tokens"] - b["accepted_tokens"]
        if drafted <= 0:
            continue
        found = True
        parts = after.stem.split("_")
        lines.append(
            f"| {parts[0]} | {parts[2].lstrip('c')} | {drafted} | {accepted} | "
            f"{100.0 * accepted / drafted:.1f}% |"
        )
    if not found:
        lines.append("| — | — | 0 | 0 | **no speculative decoding observed** |")
    return "\n".join(lines)


def verify_arm_identity(results_dir: Path) -> tuple[str, bool]:
    """Check each arm's server log actually ran the arm it claims to be.

    A filename is not evidence. Twice during E49 a stray client wrote files
    labelled for one arm while driving another arm's server -- caught only
    because the numbers carried a speculative-decoding fingerprint that the
    labelled arm cannot produce. So before any table is built, confirm from the
    server logs that arm A enabled the UnieInfra defaults and arm B did not.

    Returns:
        ``(report_text, ok)`` where ``ok`` is False if any arm looks mislabeled.
    """
    lines = [
        "\n### Arm identity verification (filenames are not evidence)",
        "",
        "| arm | server log | UnieInfra defaults banner | MTP wired | verdict |",
        "|---|---|---|---|---|",
    ]
    ok = True
    srvlogs = results_dir / "_srvlogs"
    for log in sorted(srvlogs.glob("*.log")) if srvlogs.is_dir() else []:
        arm = log.stem.split("_")[0]
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        has_banner = "UnieInfra defaults:" in text
        mtp = text.count("Detected MTP model")
        # Arm A must show the banner; every other arm must not.
        expected = arm == "A"
        good = has_banner == expected
        ok = ok and good
        lines.append(
            f"| {arm} | `{log.name}` | {'yes' if has_banner else 'no'} | "
            f"{mtp} | {'OK' if good else '**MISLABELED**'} |"
        )
    if len(lines) == 4:
        lines.append("| — | no server logs found | — | — | **cannot verify** |")
        ok = False
    return "\n".join(lines), ok


def check_spec_fingerprint(runs: dict) -> str:
    """Flag any arm whose cells carry speculative decoding it should not have."""
    problems = []
    for (arm, concurrency), entries in sorted(runs.items()):
        for entry in entries:
            length = entry.get("spec_decode_acceptance_length")
            if arm != "A" and isinstance(length, (int, float)) and length > 1.05:
                problems.append(
                    f"arm {arm} C={concurrency} has acceptance_length={length:.2f} "
                    "but only arm A runs a drafter"
                )
    if not problems:
        return "\nSpeculative-decoding fingerprint: consistent with arm design.\n"
    return (
        "\n**FINGERPRINT MISMATCH — DO NOT TRUST THESE RESULTS:**\n"
        + "\n".join(f"- {p}" for p in problems)
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--baseline", default="B", help="arm to treat as the stock-defaults baseline"
    )
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        print(f"no result JSONs found under {args.results_dir}")
        return 1

    arms = sorted({arm for arm, _ in runs})
    concurrencies = sorted({c for _, c in runs})

    out = [
        f"# E49 product sweep — {args.results_dir.name}",
        "",
        f"Arms present: {', '.join(f'{a} = {ARM_LABELS.get(a, a)}' for a in arms)}",
        f"Concurrencies: {concurrencies}",
    ]

    identity, identity_ok = verify_arm_identity(args.results_dir)
    out.append(identity)
    out.append(check_spec_fingerprint(runs))
    if not identity_ok:
        out.append(
            "\n> **Arm identity could not be confirmed. The tables below are "
            "NOT trustworthy until this is resolved.**\n"
        )

    for key, unit, higher_better in METRICS:
        if args.baseline in arms and "A" in arms:
            out.append(
                render_table(
                    f"Product: A vs {args.baseline} — {unit}",
                    compare(
                        runs, "A", args.baseline, concurrencies, key, higher_better
                    ),
                    unit,
                )
            )
        if "C" in arms and "A" in arms:
            out.append(
                render_table(
                    f"Attribution: A vs C — {unit}",
                    compare(runs, "A", "C", concurrencies, key, higher_better),
                    unit,
                )
            )

    out.append(spec_engagement(args.results_dir))
    report = "\n".join(out)
    print(report)
    (args.results_dir / "SUMMARY.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
