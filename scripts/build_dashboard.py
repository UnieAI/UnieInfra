#!/usr/bin/env python3
"""Generate the self-contained dashboard page from results/ and experiments.json.

Output: .unieai/pages/unieinfra-dashboard/index.html (published with publish_artifact).
Everything is inlined: no network requests, no chart library.
"""
import html
import json
import math
import os
import sys

sys.path.insert(0, "/home/smc/workspace/roy/scripts")
from fair import floor, geomean, matched, ours, reference  # noqa: E402
from offtable import cells  # noqa: E402

ROOT = "/home/smc/workspace/roy"
OUT = os.path.join(ROOT, ".unieai/pages/unieinfra-dashboard/index.html")

GREEN = "#15803d"
RED = "#b91c1c"
GREY = "#64748b"
BLUE = "#2563eb"
AMBER = "#b45309"


def _window(dur):
    """Which benchmark window produced a cell: the agentx preset's one hour is the
    documented recipe, 900 s is the screening floor used while iterating."""
    return "官方 3600 s" if dur == 3600 else ("篩選 900 s" if dur == 900 else f"{dur} s")


def esc(s):
    return html.escape(str(s))


# ---------------------------------------------------------------- chart geometry
W, H = 760, 430
ML, MR, MT, MB = 62, 18, 18, 46
XMIN, XMAX = 8.0, 220.0
YMIN, YMAX = 4.0, 130.0


def sx(v):
    f = (math.log10(v) - math.log10(XMIN)) / (math.log10(XMAX) - math.log10(XMIN))
    return ML + f * (W - ML - MR)


def sy(v):
    f = (math.log10(v) - math.log10(YMIN)) / (math.log10(YMAX) - math.log10(YMIN))
    return H - MB - f * (H - MT - MB)


def chart():
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="RPM versus interactivity">']
    # gridlines
    for xt in (10, 20, 50, 100, 200):
        x = sx(xt)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{H - MB}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{H - MB + 14}" text-anchor="middle">{xt}</text>')
    for yt in (5, 10, 20, 50, 100):
        y = sy(yt)
        parts.append(f'<line class="grid" x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{ML - 6}" y="{y + 3.5:.1f}" text-anchor="end">{yt}</text>')
    parts.append(f'<text class="axis" x="{(ML + W - MR) / 2:.0f}" y="{H - 6}" text-anchor="middle">互動性 1000÷p90 ITL（tok/s/user，對數）</text>')
    parts.append(f'<text class="axis" transform="translate(14 {(MT + H - MB) / 2:.0f}) rotate(-90)" text-anchor="middle">RPM（對數）</text>')

    ref = reference()
    mine = ours("any")
    # vLLM reference polyline (sorted by interactivity)
    pts = sorted(ref.values(), key=lambda c: c["inter"])
    if len(pts) > 1:
        d = " ".join(f"{sx(c['inter']):.1f},{sy(c['rpm']):.1f}" for c in pts)
        parts.append(f'<polyline class="ref-line" points="{d}"/>')
    for c in pts:
        x, y = sx(c["inter"]), sy(c["rpm"])
        parts.append(f'<rect class="ref" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1"/>')
        parts.append(f'<text class="pt" x="{x + 7:.1f}" y="{y + 3:.1f}">vLLM c{c["conc"]}</text>')

    # our cells
    for conc, cells in sorted(mine.items()):
        cells = sorted(cells, key=lambda c: c["inter"])
        if len(cells) > 1:
            d = " ".join(f"{sx(c['inter']):.1f},{sy(c['rpm']):.1f}" for c in cells)
            parts.append(f'<polyline class="our-line" points="{d}"/>')
        for c in cells:
            x, y = sx(c["inter"]), sy(c["rpm"])
            cls = "our ep" if c.get("ep") else "our"
            parts.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/>')
            lab = f"c{conc} cap{c['cap']}" + (" EP" if c.get("ep") else "")
            parts.append(f'<text class="pt" x="{x + 7:.1f}" y="{y - 6:.1f}">{lab}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def matched_table(rows):
    out = [
        '<div class="scroll"><table><thead><tr>'
        "<th>並行度</th><th>vLLM RPM</th><th>UnieInfra RPM</th><th>對齊互動性</th>"
        "<th>比值</th><th>2× 需再快</th><th>狀態</th></tr></thead><tbody>"
    ]
    for r in rows:
        ratio = r["ratio"]
        b = None if ratio else floor(r)
        if ratio:
            cls = "good" if ratio >= 1 else "bad"
            rt = f'<span class="{cls}">{ratio:.2f}×</span>'
            need = f'<span class="bad">{2 / ratio:.2f}×</span>'
        elif b:
            need = f'<span class="muted">≤{2 / b["ratio"]:.2f}×</span>'
            # No matched point exists, but our biggest-cap cell already has both
            # more throughput and better per-user latency than the reference's own
            # best cell, so the ratio can only be a floor, never a headline.
            rt = (
                f'<span class="muted">≥{b["ratio"]:.2f}×</span>'
                f'<div class="muted" style="font-size:11px">下界：{b["our_rpm"]:.1f} @ {b["our_inter"]:.0f}</div>'
            )
        else:
            rt = '<span class="muted">—</span>'
            need = '<span class="muted">—</span>'
        ours_txt = f"{r['rpm']:.1f}" if r["rpm"] else '<span class="muted">—</span>'
        out.append(
            f"<tr><td>c{r['conc']}</td><td>{r['ref_rpm']:.1f}</td>"
            f"<td>{ours_txt}</td>"
            f"<td>{r['ref_inter']:.0f}</td><td>{rt}</td><td>{need}</td>"
            f"<td class=\"muted\">{r['status']}</td></tr>"
        )
    out.append("</tbody></table></div>")
    return "\n".join(out)


# serve_vllm.sh pins max-num-seqs per concurrency point (1|2 -> 1, 4|8 -> 2*CONC,
# 16 -> 4, 32 -> 8, 64|128|256 -> CONC/4), so the reference's per-point cap is
# known from the tag alone and the official rows can be compared cap for cap.
VLLM_RECIPE_CAP = {1: 1, 2: 1, 4: 8, 8: 16, 16: 4, 32: 8, 64: 16, 128: 32, 256: 64}


def official_table():
    """Official-protocol rows, both engines, cap against cap.

    The matched-interactivity table above cannot show these yet: at the same cap
    we are behind on both axes, so no interpolation point exists and the row
    reads "need-smaller-cap". This table is the unfiltered truth for the points
    that have run, and it grows by itself as the sweep lands.
    """
    by = {}
    for c in cells():
        if not c.get("official"):
            continue
        slot = by.setdefault(c["conc"], {})
        key = c["engine"]
        # The reference ran the recipe's mxfp4 sparse indexer. When the unquantized
        # control exists, it is the fair row; the compressed one stays as a note.
        if key == "vllm" and slot.get("vllm-fair"):
            slot["vllm-compressed"] = c
            continue
        if key == "vllm" and c.get("tag") == "noiqref":
            slot["vllm-fair"] = c
            continue
        slot[key] = c
    for conc, slot in by.items():
        if slot.get("vllm-fair") and "vllm" in slot:
            slot["vllm-compressed"] = slot.pop("vllm")
        if slot.get("vllm-fair"):
            slot["vllm"] = slot.pop("vllm-fair")
    out = [
        '<div class="scroll"><table><thead><tr>'
        "<th>並行度</th><th>cap</th><th>vLLM RPM</th><th>vLLM 互動性</th>"
        "<th>UnieInfra RPM</th><th>UnieInfra 互動性</th><th>RPM 比</th>"
        "<th>TTFT p90 (s)</th><th>狀態</th></tr></thead><tbody>"
    ]
    for conc in sorted(by):
        v, o = by[conc].get("vllm"), by[conc].get("unieinfra")
        cap_v = VLLM_RECIPE_CAP.get(conc)
        if v and o and cap_v == o["cap"]:
            ratio = o["rpm"] / v["rpm"]
            cls = "good" if ratio >= 1 else "bad"
            rtxt = f'<span class="{cls}">{ratio:.2f}×</span>'
            state = f'<span class="muted">{2 / ratio:.2f}× 才到 2×</span>'
        elif v and o:
            rtxt = '<span class="muted">—</span>'
            state = f'<span class="muted">cap 不同（{cap_v} vs {o["cap"]}）</span>'
        else:
            rtxt = '<span class="muted">—</span>'
            state = '<span class="muted">等另一邊</span>'
        vr = f"{v['rpm']:.1f}" if v else "—"
        vi = f"{v['inter']:.0f}" if v else "—"
        orpm = f"{o['rpm']:.1f}" if o else "—"
        oi = f"{o['inter']:.0f}" if o else "—"
        ttft = f"{v['ttft90']:.0f} / {o['ttft90']:.0f}" if v and o else "—"
        comp = by[conc].get("vllm-compressed")
        if comp:
            state += (
                f'<div class="muted" style="font-size:11px">'
                f'壓縮 indexer 版 {comp["rpm"]:.1f} RPM @ {comp["inter"]:.0f}</div>'
            )
        out.append(
            f"<tr><td>c{conc}</td><td>{cap_v if cap_v else '—'}</td>"
            f"<td>{vr}</td><td>{vi}</td><td>{orpm}</td><td>{oi}</td>"
            f"<td>{rtxt}</td><td>{ttft}</td><td>{state}</td></tr>"
        )
    if len(out) == 1:  # header only
        out.append('<tr><td colspan="9" class="muted">官方格子還沒跑（佇列進行中）</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


def cells_table():
    # Every measured cell, not just the ones the comparison currently admits: the
    # window column keeps the 900 s screening runs and the official 3600 s ones
    # apart, and the raw list is what makes a cell visible before its counterpart
    # exists (a half-finished official sweep still shows our side).
    mine = {}
    for c in cells():
        if c["engine"] != "unieinfra" or not c["valid"] or c["noisy"] or c["quant"] or not c["cap"]:
            continue
        mine.setdefault(c["conc"], []).append(c)
    rows = []
    for conc in sorted(mine):
        for c in sorted(mine[conc], key=lambda c: (c["dur"], c["cap"], bool(c.get("ep")))):
            layout = "EP" if c.get("ep") else "TP-MoE"
            rows.append(
                f"<tr><td>c{conc}</td><td>{c['cap']}</td><td>{layout}</td>"
                f"<td>{_window(c['dur'])}</td>"
                f"<td>{c['rpm']:.1f}</td><td>{c['inter']:.0f}</td>"
                f"<td>{(c['itl50'] or 0):.1f} / {(c['itl90'] or 0):.1f}</td>"
                f"<td>{c['ttft90']:.1f}</td><td>{(c['outs'] or 0):.0f}</td>"
                f"<td>{(c['reqs'] or 0):.0f}</td><td>{100 * (c['err'] or 0):.1f}%</td></tr>"
            )
    return (
        '<div class="scroll"><table><thead><tr><th>並行度</th><th>上限</th><th>MoE 佈局</th><th>窗口</th>'
        "<th>RPM</th><th>互動性</th><th>ITL p50/p90 (ms)</th><th>TTFT p90 (s)</th>"
        "<th>輸出 tok/s</th><th>請求數</th><th>錯誤</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div>"
    )


def iterations(n=8):
    d = json.load(open(os.path.join(ROOT, "experiments.json")))
    out = []
    for e in d[-n:][::-1]:
        out.append(
            f'<details><summary>#{e["id"]} {esc(e.get("area", ""))} — '
            f'{esc(e.get("experiment", "")[:110])}</summary>'
            f'<p>{esc(e.get("result", ""))}</p><p class="muted">{esc(e.get("decision", ""))}</p></details>'
        )
    return "\n".join(out)


def main():
    rows = matched("any")
    g = geomean(rows)
    n = sum(1 for r in rows if r["ratio"])
    best = max((r for r in rows if r["ratio"]), key=lambda r: r["ratio"], default=None)
    bounds = [b for b in (floor(r) for r in rows) if b]
    allr = [r["ratio"] for r in rows if r["ratio"]] + [b["ratio"] for b in bounds]
    gg = math.exp(sum(math.log(x) for x in allr) / len(allr)) if allr else None
    soft = [r for r in rows if r["status"] == "near" and r["cells"]]
    soft_n = sum(1 for r in soft if min(c["inter"] for c in r["cells"]) < r["ref_inter"])
    page = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UnieInfra 公平比較</title>
<style>
  :root {{ --bg:#fff; --fg:#1d1f23; --muted:#5f6672; --card:#f4f5f7; --border:#e3e5e8;
          --accent:{BLUE}; --good:{GREEN}; --bad:{RED}; --grid:#e8eaed; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#15171a; --fg:#e8eaed; --muted:#9aa0a8; --card:#1f2226; --border:#2e3237;
            --accent:#6ea8fe; --good:#4ade80; --bad:#f87171; --grid:#272b30; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  main {{ max-width:880px; margin:0 auto; padding:24px 16px 56px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  h2 {{ font-size:16px; margin:28px 0 8px; }}
  .muted {{ color:var(--muted); }}
  .warn {{ color:var(--muted); border-left:3px solid #d97706; padding-left:10px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px; margin:14px 0; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:10px; }}
  .kpi {{ flex:1 1 150px; background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px; }}
  .kpi .v {{ font-size:22px; font-weight:600; }}
  .kpi .l {{ font-size:12px; color:var(--muted); }}
  .scroll {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ text-align:right; padding:6px 8px; border-bottom:1px solid var(--border); white-space:nowrap; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ color:var(--muted); font-weight:600; }}
  .good {{ color:var(--good); font-weight:600; }}
  .bad {{ color:var(--bad); font-weight:600; }}
  svg {{ width:100%; height:auto; display:block; }}
  .grid {{ stroke:var(--grid); stroke-width:1; }}
  .tick,.axis,.pt {{ fill:var(--muted); font-size:10px; }}
  .ref {{ fill:{GREY}; }}
  .ref-line {{ fill:none; stroke:{GREY}; stroke-width:1.5; stroke-dasharray:4 3; }}
  .our {{ fill:var(--accent); }}
  .our.ep {{ fill:var(--good); }}
  .our-line {{ fill:none; stroke:var(--accent); stroke-width:1.5; }}
  details {{ border-bottom:1px solid var(--border); padding:8px 0; }}
  summary {{ cursor:pointer; }}
  p {{ margin:6px 0; }}
</style>
</head>
<body>
<main>
  <h1>UnieInfra 對 vLLM：公平比較（無量化）</h1>
  <p class="muted">AgentX 官方 recipe（3600 s、每 lane 10 個 warmup、seed 42、保留 trace 延遲）為準；
  早期 900 s/2 warmup 的格子降級為「篩選」，只在雙方同窗口時才拿來算比值。
  同 checkpoint、bf16、TP4、無量化旗標。比值在「相同互動性」下比較（1000÷p90 ITL），
  資料點為對數內插，容差 ±25% 標記為 near。</p>

  <div class="kpis">
    <div class="kpi"><div class="v">{f"{g:.2f}×" if g else "—"}</div><div class="l">幾何平均比值（已對齊 {n}/6）</div></div>
    <div class="kpi"><div class="v">{f"{best['ratio']:.2f}×" if best else "—"}</div><div class="l">最佳：{"c%d" % best["conc"] if best else "—"}</div></div>
    <div class="kpi"><div class="v">{f"{gg:.2f}×" if gg else "—"}</div><div class="l">含下界（{len(allr)} 行，非對等）</div></div>
    <div class="kpi"><div class="v">{soft_n}/{n}</div><div class="l">對齊行偏寬（我方格較不互動）</div></div>
    <div class="kpi"><div class="v">≥2.00×</div><div class="l">目標</div></div>
  </div>

  <p class="muted">「偏寬」= 我方最近的格互動性低於參考值，等於用比較慢的格去代表對齊點，比值偏樂觀。</p>

  <p class="warn"><b>量測紀律（c128）。</b>參考引擎在 c128 是<b>失效格</b>，不是靶：vLLM 的 slot 預算固定
  CONC/4 = 32，128 條 lane 灌進來後 effective_prefill_concurrency 171 對 32 個 decode slot，
  它從 c64 的 66.6 RPM 崩到 20.2 RPM、互動性 8.8。因此 c128 不以「贏過 20.2」為驗收，
  驗收標準是<b>我方自身曲線</b>：cap32 應回到篩選水準 ≳55 RPM（互動性 ~19），cap128 篩選為 127.9 RPM（互動性 14）。
  打贏一個壞掉的參考不算證據，只記在表上並標記 not-a-yardstick。</p>

  <p class="warn"><b>c128 官方格跑完了，自身曲線也沒守住。</b>我方 cap32 跑滿 3600 s 只有
  19.0 RPM、互動性 6.0、TTFT p90 977 s（#154）——不是 55.3 RPM。原因已查清：900 s 篩選時長文還沒灌滿，
  3600 s 後 isl p99 到 668K，158 個請求卡在 prefill、decode 只剩 27 槽，兩邊都被佇列壓住（vLLM 同樣 171/32）。
  所以 c128 這格<b>只能讀「與 vLLM 的比值」</b>（RPM 0.94×、每 token 1.47×），不能拿來驗收 1.5×；
  「cap32 應回到 55 RPM」是我先前的錯判，已撤回。</p>

  <h2>RPM 對互動性</h2>
  <section class="card">{chart()}</section>

  <h2>同互動性比值</h2>
  {matched_table(rows)}

  <h2>官方協議：同 cap 對照</h2>
  <p class="muted">官方 recipe 下兩邊 cap 由 <code>serve_vllm.sh</code> 的 max-num-seqs 與我方的
  MAX_NUM_SEQS 各自決定；c64 兩邊都是 16 個 sequence（vLLM = CONC/4），所以這一列是
  apples-to-apples：同 cap、同 checkpoint、同 TP4、同 dataset+seed、無量化。
  這個協議下我方是落後兩軸（RPM 與互動性同時較差），不是取捨。</p>
  {official_table()}

  <h2>解碼步成本：同一批大小的差距</h2>
  <p class="muted">c64 那一列兩邊都是 16 個 sequence、同 checkpoint、同 TP4，差別只剩引擎本身。
  把 ITL 乘上各自的 spec 接受長度就得到「每個 decode step 的牆上時間」，這是唯一可直接比較的量。</p>
  <section class="card">
  <table class="t">
    <tr><th>項目</th><th>UnieInfra c64 cap16</th><th>vLLM c64 cap16</th><th>比值</th></tr>
    <tr><td>ITL p50 / p90 (ms)</td><td>44.66 / 62.10</td><td>12.53 / 17.97</td><td>3.5×</td></tr>
    <tr><td>spec 接受長度 (tok/step)</td><td>3.69–3.71</td><td>3.48–3.62</td><td>1.02×</td></tr>
    <tr><td>推得每步時間 (ms)</td><td>~229</td><td>~63</td><td><b>3.6×</b></td></tr>
    <tr><td>RPM / 互動性</td><td>25.9 / 16.1</td><td>66.6 / 55.6</td><td>0.39× / 0.29×</td></tr>
  </table>
  <p>接受長度一樣，所以 3.5× 的 ITL 差距不是投機解碼被稀釋，而是每一步真的慢 3.6×。
  排程側的槓桿（EP、DP attention、all-reduce overlap、Iris、spec 寬度、chunk 2048、mixed batch）
  已全部關閉，剩下的只能在 kernel：待驗證的嫌疑是 MoE grouped GEMM 的 M 太小（aiter 日誌見 <code>M:8</code>，
  權重按 request 重讀）與 top-512 gather／indexer scoring 走手寫 Triton+Gluon 而非 aiter/DeepGEMM 路徑。
  <code>V41_CACHE_FORMATS["v4"]</code> 那條路已排除：<code>configs/deepseek_v41.py:176</code> 寫死
  <code>"v4" if is_hopper else "v41"</code>，gfx950 沒有對應 kernel。</p>
  <p class="warn">已被推翻的說法：(1) 「vLLM 在 c128 約 100 RPM、互動性 15」是我用 profile 前 5 分鐘
  （300 s、502 請求）外推的，整段 3624 s 實際只有 <b>20.2 RPM、互動性 8.8</b>，那段是 warmup 剛結束、
  佇列還沒堆起來的暫態。(2) 「整體落後 vLLM 約 2.6×」只對 c64 cap16 這一列成立，不能推廣。</p>
  </section>

  <h2>UnieInfra 公平格</h2>
  {cells_table()}

  <h2>迭代記錄</h2>
  <section class="card">{iterations()}</section>
</main>
</body>
</html>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(page)
    print(OUT, len(page), "bytes")


if __name__ == "__main__":
    main()
