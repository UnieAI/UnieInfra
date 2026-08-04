#!/usr/bin/env python3
"""E47: the stock-vLLM-vs-UnieInfra table, one per precision arm.

Reads ``results/e47/e47_<precision>_<arm>_r<rep>/c<C>.json`` (the raw
``vllm bench serve`` output) and collapses the repeats into one
median-with-``[min-max]`` cell per (precision, concurrency, arm, metric).

Three arms per precision, and the middle one is the point:

  stock     vLLM 0.25.1 from PyPI, its own best working defaults.
  forkbase  our vendored fork with every ``UNIEINFRA_*`` seam OFF.
  unie      the same fork with everything of ours ON.

``stock -> forkbase`` is therefore *upstream drift* between two engine
snapshots and ``forkbase -> unie`` is *our work*. Quoting only
``stock -> unie`` -- which is what a naive two-arm run would produce --
attributes the sum of both to us, in whichever direction it happens to point.

Verdict rule, deliberately conservative and identical to `recut_aggregate.py`:
a delta whose ``[min, max]`` ranges OVERLAP is a **TIE**, not a win and not a
loss. It is not separable from run-to-run noise and must not be quoted as if it
were.

Also emits, per cell, whether our kernel was actually selected -- read from the
``UNIEINFRA_KERNEL_SELECTION_DUMP`` snapshots copied next to each cell. A cell
where our kernel never fired is not evidence about our kernel.

Usage:
    benchmark/e47_aggregate.py [--out results/E47_AGGREGATE.md]

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "e47"
CONCS = [1, 16, 32, 64, 128, 256]
PRECISIONS = [
    ("bf16", "bf16 — the headline apples-to-apples comparison"),
    ("fp8", "fp8 — ours vs vLLM's own fp8 path"),
    ("nvfp4", "nvfp4 — ours vs vLLM's own `modelopt_fp4`"),
]
ARMS = ["stock", "forkbase", "unie"]
METRICS = [
    ("output_throughput", "out tok/s", "hi"),
    ("mean_ttft_ms", "TTFT mean ms", "lo"),
    ("p99_ttft_ms", "TTFT p99 ms", "lo"),
    ("mean_tpot_ms", "TPOT mean ms", "lo"),
    ("p99_tpot_ms", "TPOT p99 ms", "lo"),
    ("tok_s_per_user", "tok/s per user", "hi"),
]
OURS = (
    "unieinfra_decode_gemv_triton",
    "unieinfra_decode_gemv_fp8_triton",
    "unieinfra_decode_gemv_nvfp4_triton",
)


def collect() -> tuple[dict, dict]:
    """Load every cell.

    Returns:
        ``(data, runs)`` where ``data[(prec, arm, C, metric)]`` is the list of
        values across repeats and ``runs[(prec, arm)]`` names the directories.
    """
    data: dict = {}
    runs: dict = {}
    for d in sorted(RESULTS.glob("e4[17]_*_r*")):
        parts = d.name.split("_")
        if len(parts) < 4:
            continue
        prec, arm, rep = parts[1], "_".join(parts[2:-1]), parts[-1]
        runs.setdefault((prec, arm), []).append(rep)
        for jf in sorted(d.glob("c*.json")):
            try:
                j = json.load(jf.open())
            except Exception:
                continue
            c = int(j.get("max_concurrency") or 0)
            for m, _, _ in METRICS:
                if m == "tok_s_per_user":
                    tp = j.get("mean_tpot_ms") or 0.0
                    v = 1000.0 / tp if tp else 0.0
                else:
                    v = j.get(m)
                if v is None:
                    continue
                data.setdefault((prec, arm, c, m), []).append(float(v))
    return data, runs


def selection_evidence() -> dict:
    """(prec, arm, C) -> a short human string proving whether our kernel fired.

    The counters are cumulative within one server process and the dense GEMV is
    resolved once per shape at CUDA-graph *capture* time, so the honest evidence
    is the shape table (which ``M`` resolved to one of our kernels) plus the
    ``fast``/``delegated`` totals -- not a per-request count, which no captured
    graph can produce.
    """
    out: dict = {}
    for d in sorted(RESULTS.glob("e4[17]_*_r*")):
        parts = d.name.split("_")
        if len(parts) < 4:
            continue
        prec, arm = parts[1], "_".join(parts[2:-1])
        for sf in d.glob("sel_c*.json"):
            c = int(sf.name.split("sel_c")[1].split(".")[0])
            try:
                j = json.load(sf.open())
            except Exception:
                continue
            ms = sorted({e["M"] for e in j.get("selection", []) if e["kernel"] in OURS})
            fast = sum(v.get("fast", 0) for v in j.get("counters", {}).values())
            deleg = sum(v.get("delegated", 0) for v in j.get("counters", {}).values())
            if not ms:
                continue
            prev = out.get((prec, arm, c))
            cell = (ms, fast, deleg)
            if prev is None or fast > prev[1]:
                out[(prec, arm, c)] = cell
    return {
        k: f"M<={max(v[0])} (fast={v[1]}, delegated={v[2]})" for k, v in out.items()
    }


def fmt(vals: list[float], nd: int = 1) -> str:
    if not vals:
        return "-"
    return f"{statistics.median(vals):.{nd}f} [{min(vals):.{nd}f}-{max(vals):.{nd}f}]"


def verdict(a: list[float], b: list[float], direction: str) -> str:
    """``a`` (ours) against ``b`` (reference). TIE when the ranges overlap."""
    if not a or not b:
        return "-"
    ma, mb = statistics.median(a), statistics.median(b)
    delta = (ma - mb) / mb * 100.0 if mb else 0.0
    overlap = not (max(a) < min(b) or max(b) < min(a))
    if overlap:
        return f"TIE ({delta:+.1f}%)"
    better = (ma > mb) if direction == "hi" else (ma < mb)
    return ("WIN " if better else "LOSE ") + f"{delta:+.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS.parent / "E47_AGGREGATE.md"))
    args = ap.parse_args()

    data, runs = collect()
    sel = selection_evidence()
    L: list[str] = []
    w = L.append

    w("# E47 — stock vLLM vs UnieInfra, everything of ours ON\n")
    w("Qwen3-8B, sm_120 (1x RTX PRO 6000), GPU 3, RandomDataset ILEN 1024 /")
    w("OLEN 256 `--ignore-eos`, greedy, 8 prompts per concurrent slot (capped at")
    w("512), warmups 8, seed 0. One client for both arms: the stock")
    w("`vllm bench serve` on the host. MTP off everywhere; prefix caching left at")
    w("each engine's default and never flushed. N=3, arms rotated back-to-back")
    w("within each round. **A delta whose [min-max] ranges overlap is a TIE.**\n")

    for prec, title in PRECISIONS:
        present = [a for a in ARMS if (prec, a) in runs]
        if not present:
            continue
        w(f"\n## {title}\n")
        w(
            "Repeats: "
            + ", ".join(f"{a}={len(runs[(prec, a)])}" for a in present)
            + "\n"
        )
        for m, label, direction in METRICS:
            nd = 1
            w(
                f"\n### {label} (median [min-max], "
                f"{'higher' if direction == 'hi' else 'lower'} is better)\n"
            )
            w("| C | " + " | ".join(present) + " | unie vs stock | unie vs forkbase |")
            w("|---|" + "---|" * (len(present) + 2))
            for c in CONCS:
                cells = [fmt(data.get((prec, a, c, m), []), nd) for a in present]
                v1 = verdict(
                    data.get((prec, "unie", c, m), []),
                    data.get((prec, "stock", c, m), []),
                    direction,
                )
                v2 = verdict(
                    data.get((prec, "unie", c, m), []),
                    data.get((prec, "forkbase", c, m), []),
                    direction,
                )
                w(f"| {c} | " + " | ".join(cells) + f" | {v1} | {v2} |")

        w("\n### Was our kernel actually selected?\n")
        w("| C | unie | forkbase |")
        w("|---|---|---|")
        for c in CONCS:
            u = sel.get((prec, "unie", c), "no — our kernel never fired")
            f = sel.get((prec, "forkbase", c), "n/a (seams off)")
            w(f"| {c} | {u} | {f} |")

    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
