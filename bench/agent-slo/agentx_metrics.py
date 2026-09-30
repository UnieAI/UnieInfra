#!/usr/bin/env python3
"""Turn a drive.py result dump into the AgentX SLO axes.

Reads the JSON files written by ``drive.py`` (one per engine/cell) and reports:
  - dur / rpm / tps (server-side) and tps/req (per-user) throughput,
  - TTFT + TPOT p50 / p90 / p99 (interactivity axes),
  - completion count + mean output chunks (nsplit after chunk-versus-token
    caveat),
  - naive token-agreement between the two engines (LCP of the generated text,
    both greedy), to catch an engine that corrupts output under load.
No network, no servers: everything is derived from the saved text.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def _pct(vals, p):
    vals = sorted(vals)
    if not vals:
        return None
    return vals[min(int(p * (len(vals) - 1)), len(vals) - 1)]


def load(path):
    d = json.loads(Path(path).read_text())
    rows = [r for r in d["rows"] if r.get("ttft") is not None and r.get("text")]
    return d, rows


def row_metrics(d, rows):
    dur = d["duration_s"]
    nok = len(rows)
    ntok = sum(r.get("ntok") or 0 for r in rows)
    e2e = [r["ms"] for r in rows]
    ttft = [r["ttft"] for r in rows]
    tpot = [r.get("tpot") for r in rows if r.get("tpot") is not None]
    return {
        "requests": d["n"],
        "ok": nok,
        "duration_s": dur,
        "rpm": 60 * nok / dur if dur else None,
        "tps": ntok / dur if dur else None,
        "tps_per_req": statistics.mean(
            [r.get("ntok", 0) / max(1e-9, r["ms"] / 1000) for r in rows]
        ),
        "mean_chunks": statistics.mean([r.get("ntok") or 0 for r in rows]),
        "ttft_p50": _pct(ttft, 0.50),
        "ttft_p90": _pct(ttft, 0.90),
        "ttft_p99": _pct(ttft, 0.99),
        "tpot_p50": _pct(tpot, 0.50),
        "tpot_p90": _pct(tpot, 0.90),
        "tpot_p99": _pct(tpot, 0.99),
        "e2e_p50": _pct(e2e, 0.50),
        "e2e_p99": _pct(e2e, 0.99),
    }


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def agree(a_rows, b_rows):
    n = min(len(a_rows), len(b_rows))
    matched = sum(
        1
        for i in range(n)
        if lcp(a_rows[i]["text"], b_rows[i]["text"]) / max(1, len(a_rows[i]["text"]))
        > 0.5
    )
    return matched, n


def main(argv):
    if len(argv) < 2:
        print(
            "usage: agentx_metrics.py <engine.json> [<vllm.json> ...]", file=sys.stderr
        )
        return 2
    names = argv[1:]
    loaded = {p: load(p) for p in names}
    for p, (d, rows) in loaded.items():
        m = row_metrics(d, rows)
        print(f"\n== {Path(p).name} ==")
        for k in (
            "requests",
            "ok",
            "duration_s",
            "rpm",
            "tps",
            "tps_per_req",
            "mean_chunks",
            "ttft_p50",
            "ttft_p90",
            "ttft_p99",
            "tpot_p50",
            "tpot_p90",
            "tpot_p99",
            "e2e_p50",
            "e2e_p99",
        ):
            print(f"  {k:12s} {m[k]}")
    if len(names) == 2:
        a = loaded[names[0]][1]
        b = loaded[names[1]][1]
        m, n = agree(a, b)
        print(f"\ntext agreement (>50% LCP): {m}/{n} ({100 * m / max(1, n):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
