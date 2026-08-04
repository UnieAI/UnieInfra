#!/usr/bin/env python3
"""E42: does masking DRAFT logits with the request's grammar buy anything?

Pivots ``results/unieinfra/e42_<arm>_k<k>_r<rep>/c<C>.json`` into one
median-with-[min,max] cell per ``(k, concurrency, arm, metric)``.

Two reporting rules this script enforces, both learned the hard way:

* **Accepted LENGTH, not acceptance RATE.** ``spec_decode_acceptance_rate`` is
  ``accepted / submitted``, and prefix truncation shrinks *submitted*. Turning a
  grammar on therefore raises the rate without anything improving (E40 saw
  44.4% -> 72.9% from truncation alone). ``spec_decode_acceptance_length`` --
  ``1 + accepted/num_drafts`` -- has a denominator that does not move, so it is
  the statistic that is comparable across arms. The rate is printed only in a
  clearly-labelled diagnostic row.
* **Verdict by overlapping ranges.** TIE if the two arms' observed [min, max]
  ranges overlap; otherwise WIN/LOSE from the medians. Same conservative rule as
  ``e39_aggregate.py``.

It also prints the mechanism row that motivated the whole experiment: drafts
*submitted* per step (``spec_decode_draft_tokens / spec_decode_num_drafts``).
Draft masking is supposed to move that number up towards k; whether accepted
length follows is the actual question.

Usage:
    benchmark/e42_aggregate.py [--prefix e42] [--out results/E42_AGGREGATE.md]

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "unieinfra"

ARMS = ["maskoff", "maskon"]
ARM_LABEL = {
    "maskoff": "draft mask OFF (upstream)",
    "maskon": "draft mask ON (UnieInfra)",
}
METRICS = [
    ("output_throughput", "out tok/s", "hi"),
    ("mean_tpot_ms", "TPOT mean ms", "lo"),
    ("mean_ttft_ms", "TTFT mean ms", "lo"),
    ("spec_decode_acceptance_length", "accepted length (1+acc/drafts)", "hi"),
]
DIAGNOSTIC = [
    ("submitted_per_step", "drafts SUBMITTED per step (of k)"),
    ("spec_decode_acceptance_rate", "acceptance RATE % (NOT comparable)"),
    ("total_output_tokens", "total output tokens"),
]


def collect(prefix: str) -> tuple[dict, set]:
    """(k, conc, arm, metric) -> list of values, one per repeat."""
    out: dict = {}
    ks: set = set()
    for d in sorted(RESULTS.glob(f"{prefix}_*_k*_r*")):
        parts = d.name.split("_")
        # e42_<arm>_k<k>_r<rep>
        if len(parts) != 4 or not parts[3].startswith("r"):
            continue
        arm, kstr = parts[1], parts[2]
        if arm not in ARMS or not kstr.startswith("k"):
            continue
        k = int(kstr[1:])
        ks.add(k)
        for jf in sorted(d.glob("c*.json")):
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            if not data.get("completed"):
                continue
            conc = data.get("max_concurrency")
            num_drafts = data.get("spec_decode_num_drafts") or 0
            if num_drafts:
                data["submitted_per_step"] = (
                    data.get("spec_decode_draft_tokens", 0) / num_drafts
                )
            for key, *_ in METRICS + DIAGNOSTIC:
                v = data.get(key)
                if isinstance(v, (int, float)):
                    out.setdefault((k, conc, arm, key), []).append(v)
    return out, ks


def cell(vals: list[float]) -> str:
    if not vals:
        return "-"
    m = statistics.median(vals)
    if len(vals) == 1:
        return f"{m:.2f}"
    return f"{m:.2f} [{min(vals):.2f}-{max(vals):.2f}]"


def verdict(off: list[float], on: list[float], better: str) -> str:
    if not off or not on:
        return "-"
    if min(off) <= max(on) and min(on) <= max(off):
        return "TIE"
    mo, mn = statistics.median(off), statistics.median(on)
    delta = 100.0 * (mn - mo) / mo if mo else 0.0
    good = (mn > mo) if better == "hi" else (mn < mo)
    return f"{'WIN' if good else 'LOSE'} {delta:+.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="e42")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data, ks = collect(args.prefix)
    if not data:
        print("no e42 results found")
        return
    concs = sorted({c for (_k, c, _a, _m) in data})

    lines: list[str] = []
    lines.append("# E42 — draft-side grammar mask (structured output + MTP)\n")
    lines.append(
        "Both arms serve the same JSON schema with MTP on and differ only in "
        "`UNIEINFRA_DRAFT_GRAMMAR_MASK`. Cells are median [min-max] over "
        "repeats. Verdict is TIE whenever the two arms' observed ranges "
        "overlap.\n"
    )
    for k in sorted(ks):
        lines.append(f"\n## k = {k} draft tokens\n")
        lines.append(
            "| C | metric | " + " | ".join(ARM_LABEL[a] for a in ARMS) + " | verdict |"
        )
        lines.append("|---|---|---|---|---|")
        for c in concs:
            for key, label, better in METRICS:
                vals = {a: data.get((k, c, a, key), []) for a in ARMS}
                lines.append(
                    f"| {c} | {label} | "
                    + " | ".join(cell(vals[a]) for a in ARMS)
                    + f" | {verdict(vals['maskoff'], vals['maskon'], better)} |"
                )
        lines.append("")
        lines.append("Diagnostics (mechanism, not verdict):\n")
        lines.append("| C | metric | " + " | ".join(ARM_LABEL[a] for a in ARMS) + " |")
        lines.append("|---|---|---|---|")
        for c in concs:
            for key, label in DIAGNOSTIC:
                vals = {a: data.get((k, c, a, key), []) for a in ARMS}
                lines.append(
                    f"| {c} | {label} | "
                    + " | ".join(cell(vals[a]) for a in ARMS)
                    + " |"
                )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = Path(__file__).resolve().parent / out
        out.write_text(text)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
