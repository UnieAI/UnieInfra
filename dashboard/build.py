#!/usr/bin/env python3
"""Build the single-file AgentX dashboard page.

Reads dash_data/agentx_fresh.json (one row per engine x concurrency, produced by
scripts/export_dashboard.py from results/agentx), computes the headline prose
from that same data, and writes dashboard/index.html from src/template.html.

Usage: python3 dashboard/build.py [ROWS.json]
"""

import datetime as dt
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "src" / "template.html"
OUT = HERE / "index.html"

CONCS = (1, 4, 16, 32, 64, 128)


def n(x, dp=0):
    return f"{x:,.{dp}f}"


def ratio(x):
    return f"{x:.2f}×" if x < 10 else f"{x:.0f}×"


def main() -> None:
    rows_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dash_data" / "agentx_fresh.json"
    rows = json.loads(rows_path.read_text())

    def row(engine, conc):
        for r in rows:
            if r["engine"] == engine and r["concurrency"] == conc:
                return r
        raise KeyError((engine, conc))

    def best_vllm(conc, kind="any"):
        pool = [r for r in rows if r["engine"].startswith("vLLM") and r["concurrency"] == conc]
        if kind == "official":
            pool = [r for r in pool if "(stress" not in r["engine"]]
        elif kind == "stress":
            pool = [r for r in pool if "(stress" in r["engine"]]
        return max(pool, key=lambda r: r["output_tps"]) if pool else None

    ui16 = row("UnieInfra optimized", 16)
    vl16 = row("vLLM", 16)
    sg16 = row("SGLang", 16)
    r_vl = ui16["output_tps"] / vl16["output_tps"]
    r_sg = ui16["output_tps"] / sg16["output_tps"]

    lead, behind = [], []
    for c in CONCS:
        try:
            u, v = row("UnieInfra optimized", c), row("vLLM", c)
        except KeyError:
            continue
        (lead if u["output_tps"] > v["output_tps"] else behind).append(c)
    c64u, c64v = row("UnieInfra optimized", 64), row("vLLM", 64)
    c128u, c128v = row("UnieInfra optimized", 128), row("vLLM", 128)

    cd = {c: row("UnieInfra coupled draft, sampled (stress)", c) for c in CONCS
          if any(r["engine"] == "UnieInfra coupled draft, sampled (stress)" and r["concurrency"] == c for r in rows)}
    cd_ratio = {}
    for c, r in cd.items():
        b = best_vllm(c, "stress")
        if b:
            cd_ratio[c] = (r["output_tps"] / b["output_tps"], r["output_tps"], b["output_tps"], b["engine"])

    stamps = sorted({m.group(1) for r in rows if (m := re.search(r"(20\d{6})T", r["run"]))})
    window = f"{stamps[0][:4]}-{stamps[0][4:6]}-{stamps[0][6:]} → {stamps[-1][:4]}-{stamps[-1][4:6]}-{stamps[-1][6:]}"
    engines = len({r["engine"] for r in rows})

    lead_bits = " · ".join(f"{ratio(r2)} at c{c}" for c in lead
                            for r2 in [row('UnieInfra optimized', c)['output_tps'] / row('vLLM', c)['output_tps']])
    lede = (
        f"At c16, optimized UnieInfra reaches <strong>{n(ui16['output_tps'], 0)} output tok/s</strong> — "
        f"{ratio(r_vl)} vLLM's {n(vl16['output_tps'])} and {ratio(r_sg)} SGLang's {n(sg16['output_tps'])} — "
        f"with a mean TTFT of <strong>{n(ui16['ttft_mean_ms'])} ms</strong> against {n(vl16['ttft_mean_ms'])} ms "
        f"and {n(sg16['ttft_mean_ms'])} ms. Across the ladder it stays ahead of vLLM's official recipe at "
        f"c{', c'.join(map(str, lead))} ({lead_bits}), loses at c64 "
        f"({n(c64u['output_tps'])} vs {n(c64v['output_tps'])} tok/s) and falls to "
        f"{n(c128u['output_tps'], 1)} tok/s at c128, where vLLM still holds {n(c128v['output_tps'])}."
    )

    cd_bits = " · ".join(f"{ratio(v[0])} at c{c}" for c, v in sorted(cd_ratio.items()))
    sub = (
        f"The stress series drop think time and carry the merged fixes (larger KV cache, 4K prefill chunks, "
        f"INT8 prefill all-reduce). Among them <em>coupled draft, sampled</em> keeps T=1.0 sampling as vLLM does; "
        f"against the best same-kind vLLM stress run it lands at {cd_bits}."
    )

    kpis = f"""
    <div class="kpi"><div class="kpi-k">Output · c16</div>
      <div class="kpi-v">{n(ui16['output_tps'])}<span class="kpi-u">tok/s</span></div>
      <div class="kpi-f">vLLM {n(vl16['output_tps'])} · SGLang {n(sg16['output_tps'])}</div></div>
    <div class="kpi"><div class="kpi-k">Mean TTFT · c16</div>
      <div class="kpi-v">{n(ui16['ttft_mean_ms'])}<span class="kpi-u">ms</span></div>
      <div class="kpi-f">vLLM {n(vl16['ttft_mean_ms'])} · SGLang {n(sg16['ttft_mean_ms'])}</div></div>
    <div class="kpi"><div class="kpi-k">Peak official</div>
      <div class="kpi-v">{n(row('UnieInfra optimized', 32)['output_tps'])}<span class="kpi-u">tok/s</span></div>
      <div class="kpi-f">c32 · {ratio(row('UnieInfra optimized', 32)['output_tps'] / row('vLLM', 32)['output_tps'])} vLLM</div></div>
    <div class="kpi kpi-warn"><div class="kpi-k">At c128</div>
      <div class="kpi-v">{n(c128u['output_tps'], 1)}<span class="kpi-u">tok/s</span></div>
      <div class="kpi-f">vLLM {n(c128v['output_tps'])} · collapse</div></div>
    """

    meta = {
        "cells": len(rows),
        "engines": engines,
        "window": window,
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": str(rows_path.relative_to(ROOT)),
    }

    html = SRC.read_text()
    for token, value in {
        "__ROWS__": json.dumps(rows, separators=(",", ":"), ensure_ascii=False),
        "__META__": json.dumps(meta, ensure_ascii=False),
        "__LEDE__": lede,
        "__SUB__": sub,
        "__KPIS__": kpis,
    }.items():
        if token not in html:
            raise SystemExit(f"template is missing {token}")
        html = html.replace(token, value)
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html)/1024:.1f} KB) cells={meta['cells']} engines={engines} window={window}")


if __name__ == "__main__":
    main()
