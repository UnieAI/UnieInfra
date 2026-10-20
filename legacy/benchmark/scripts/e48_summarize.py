# SPDX-License-Identifier: Apache-2.0
"""E48 — aggregate the fp8-KV sweep into the four tables the decision needs.

Reporting rules (E35): median across repeats with the [min-max] spread shown, and
a between-arm delta that falls inside the arms' own spread is called a TIE rather
than a win.  Accuracy is compared BOTH in aggregate and paired per question --
1319 GSM8K items carry a ~1.0pp standard error, too coarse to resolve a small
regression on its own, whereas the paired flip counts resolve it directly.

Usage: python e48_summarize.py <results-dir>
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

ARM_ORDER = ["bf16fa", "bf16tri", "bf16fi", "e4m3tri", "e5m2tri", "e4m3auto"]
ARM_DESC = {
    "bf16fa": "bf16 KV, FLASH_ATTN (production default)",
    "bf16tri": "bf16 KV, TRITON_ATTN (backend control for e4m3tri)",
    "bf16fi": "bf16 KV, FLASHINFER (backend control for e4m3auto)",
    "e4m3tri": "fp8_e4m3 KV, TRITON_ATTN",
    "e5m2tri": "fp8_e5m2 KV, TRITON_ATTN",
    "e4m3auto": "fp8_e4m3 KV, auto backend (-> FLASHINFER on sm_120)",
}


def load_tput(root: str, tag: str):
    rows: dict[tuple[int, str], dict[str, list[float]]] = {}
    for path in glob.glob(os.path.join(root, f"{tag}_*_r*", "c*.json")):
        base = os.path.basename(os.path.dirname(path))
        arm = base[len(tag) + 1 :].rsplit("_r", 1)[0]
        c = int(os.path.basename(path)[1:-5])
        d = json.load(open(path))
        slot = rows.setdefault((c, arm), {})
        for k in (
            "output_throughput",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "median_ttft_ms",
            "median_tpot_ms",
            "request_throughput",
        ):
            if k in d:
                slot.setdefault(k, []).append(d[k])
    return rows


def fmt(vals: list[float]) -> str:
    return f"{statistics.median(vals):.2f} [{min(vals):.2f}-{max(vals):.2f}]"


def spread_frac(vals: list[float]) -> float:
    m = statistics.median(vals)
    return (max(vals) - min(vals)) / m if m else 0.0


def verdict(base: list[float], arm: list[float]) -> str:
    mb, ma = statistics.median(base), statistics.median(arm)
    delta = (ma - mb) / mb * 100
    tol = 100 * max(spread_frac(base), spread_frac(arm))
    if abs(delta) <= tol:
        return f"{delta:+.2f}% TIE (spread +/-{tol:.2f}%)"
    return f"{delta:+.2f}%"


def table_tput(root: str, tag: str, title: str) -> None:
    rows = load_tput(root, tag)
    if not rows:
        return
    arms = [a for a in ARM_ORDER if any(a == k[1] for k in rows)]
    arms += sorted({k[1] for k in rows} - set(arms))
    print(f"\n### {title} — output tok/s, median [min-max] over repeats")
    print(f"{'C':>5}  " + "  ".join(f"{a:>26}" for a in arms))
    for c in sorted({k[0] for k in rows}):
        cells = []
        for a in arms:
            v = rows.get((c, a), {}).get("output_throughput")
            cells.append(fmt(v) if v else "—")
        print(f"{c:>5}  " + "  ".join(f"{x:>26}" for x in cells))

    for baseline in ("bf16fa", "bf16tri"):
        if not any(k[1] == baseline for k in rows):
            continue
        others = [a for a in arms if a != baseline]
        print(f"\n  delta vs {baseline} ({ARM_DESC.get(baseline, '')})")
        print(f"  {'C':>5}  " + "  ".join(f"{a:>24}" for a in others))
        for c in sorted({k[0] for k in rows}):
            b = rows.get((c, baseline), {}).get("output_throughput")
            if not b:
                continue
            cells = []
            for a in others:
                v = rows.get((c, a), {}).get("output_throughput")
                cells.append(verdict(b, v) if v else "—")
            print(f"  {c:>5}  " + "  ".join(f"{x:>24}" for x in cells))

    for metric, label in (
        ("mean_ttft_ms", "mean TTFT (ms)"),
        ("mean_tpot_ms", "mean TPOT (ms)"),
    ):
        print(f"\n  {label} — median [min-max]")
        print(f"  {'C':>5}  " + "  ".join(f"{a:>24}" for a in arms))
        for c in sorted({k[0] for k in rows}):
            cells = []
            for a in arms:
                v = rows.get((c, a), {}).get(metric)
                cells.append(fmt(v) if v else "—")
            print(f"  {c:>5}  " + "  ".join(f"{x:>24}" for x in cells))


def table_capacity(root: str) -> None:
    print("\n### KV capacity and the backend actually selected")
    for arm in ARM_ORDER:
        p = os.path.join(root, f"config_{arm}.txt")
        if not os.path.exists(p):
            continue
        print(f"\n-- {arm}: {ARM_DESC.get(arm, '')}")
        print(open(p).read().rstrip())


def table_accuracy(root: str) -> None:
    accs = {}
    for arm in ARM_ORDER:
        p = os.path.join(root, f"gsm8k_{arm}.json")
        if os.path.exists(p):
            accs[arm] = json.load(open(p))
    if not accs:
        return
    print("\n### GSM8K accuracy (5-shot, greedy, paired prompts)")
    print(f"{'arm':>10}  {'n':>6}  {'strict':>9}  {'flexible':>9}")
    for arm, d in accs.items():
        print(
            f"{arm:>10}  {d['n']:>6}  {100 * d['strict_acc']:>8.2f}%  "
            f"{100 * d['flexible_acc']:>8.2f}%"
        )

    base = accs.get("bf16fa") or accs.get("bf16tri")
    if base is None:
        return
    bname = "bf16fa" if "bf16fa" in accs else "bf16tri"
    print(
        f"\n  paired flips vs {bname} (same questions, greedy) — "
        f"the sensitive view; aggregate accuracy on n=1319 has ~1.0pp SE"
    )
    print(
        f"  {'arm':>10}  {'right->wrong':>13}  {'wrong->right':>13}  "
        f"{'net':>7}  {'identical text':>15}"
    )
    for arm, d in accs.items():
        if arm == bname:
            continue
        r2w = w2r = same_text = 0
        for a, b in zip(base["per_question"], d["per_question"]):
            if a["ok_flex"] and not b["ok_flex"]:
                r2w += 1
            elif not a["ok_flex"] and b["ok_flex"]:
                w2r += 1
            if a["text"] == b["text"]:
                same_text += 1
        n = len(base["per_question"])
        print(
            f"  {arm:>10}  {r2w:>13}  {w2r:>13}  {w2r - r2w:>+7}  "
            f"{same_text:>8}/{n:<6}"
        )


def main() -> int:
    root = sys.argv[1]
    table_capacity(root)
    table_tput(root, "tput", "Throughput, ILEN 1024 / OLEN 256")
    table_tput(root, "long8k", "Long context, ILEN 8192 / OLEN 256")
    table_tput(root, "long32k", "Long context, ILEN 32768 / OLEN 256")
    table_accuracy(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
