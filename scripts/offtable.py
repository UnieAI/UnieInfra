#!/usr/bin/env python3
"""Official-mode AgentX cells as RPM / interactivity / TTFT p90 (matched-interactivity view).

Scans results/agentx/<engine>-<tag>-c<conc>-d<dur>-<stamp>/aiperf_artifacts/profile_export_aiperf.json
and prints one row per cell. `--vs <conc>` prints, for each cell, the vLLM recipe point
(cap = conc/4) at the same concurrency so the matched-interactivity comparison is explicit.
"""
import argparse
import glob
import json
import os
import re

ROOT = "/home/smc/workspace/roy/results/agentx"
# -dofficial marks a run that left the profile length to the agentx preset (one
# hour, ten warmup requests per lane); it is the documented recipe and is priced
# as 3600 s so it sorts with the fixed-duration cells.
OFFICIAL_SECONDS = 3600
PAT = re.compile(
    r"^(?P<engine>[a-z]+)(?:-(?P<tag>[A-Za-z0-9-]+?))?-c(?P<conc>\d+)-d(?P<dur>\d+|official)-(?P<stamp>\d{8}T\d{6}Z)$"
)


def load(d):
    f = os.path.join(d, "aiperf_artifacts", "profile_export_aiperf.json")
    if not os.path.exists(f):
        return None
    try:
        with open(f) as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        # A cell that is still running (or died early) can leave an empty or
        # truncated export; treat it as "no data yet" instead of crashing.
        return None

    def p(k, q="avg"):
        v = m.get(k)
        return v.get(q) if isinstance(v, dict) else None

    itl90 = p("inter_token_latency", "p90")
    bench = m.get("benchmark_duration")
    bench = bench.get("avg") if isinstance(bench, dict) else bench
    return {
        "dir": d,
        "engine": None,
        "tag": None,
        "conc": None,
        "dur": None,
        "stamp": None,
        "rpm": (p("request_throughput") or 0) * 60,
        "inter": 1000.0 / itl90 if itl90 else 0.0,
        "itl50": p("inter_token_latency", "p50"),
        "itl90": itl90,
        "ttft90": (p("time_to_first_token", "p90") or 0) / 1000.0,
        "outs": p("output_token_throughput"),
        "err": p("request_error_rate"),
        "reqs": p("request_count"),
        "bench": bench or 0.0,
        "cancelled": bool(m.get("was_cancelled")),
    }


def cells():
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
        c = load(d)
        if c is None:
            continue
        m = PAT.match(os.path.basename(d))
        if m:
            c.update(m.groupdict())
            c["conc"] = int(c["conc"])
            c["official"] = c["dur"] == "official"
            c["dur"] = OFFICIAL_SECONDS if c["official"] else int(c["dur"])
            tag = c["tag"] or ""
            # A variant prefix marks a cell that differs from the production recipe
            # at the same cap, so the two stay distinguishable: fair.py builds the
            # frontier from the plain production cells only.
            #   mix    --enable-mixed-batch
            #   chunk2/chunk8  chunked prefill 2048/8192
            #   dec    a decode round between prefill rounds
            #          (TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS)
            #   declog --decode-log-interval 1, so every decode round is logged
            #          and prefill/decode time splits without censoring
            #   nopfg  --disable-prefill-graph
            m_cap = re.match(r"^(mix|chunk2|chunk8|dec|declog|nopfg)?(?:parS|off)(\d+)(ep)?$", tag)
            c["cap"] = int(m_cap.group(2)) if m_cap else None
            c["ep"] = bool(m_cap and m_cap.group(3))
            c["variant"] = m_cap.group(1) if m_cap else None
            c["mixed"] = c["variant"] == "mix"
            # A cell counts only if it ran the full window and was not interrupted;
            # the Q8-on sweep (before the 2026-11-10T00:08:25Z restart) is quantized
            # and therefore inadmissible for fair comparisons.
            c["valid"] = (not c["cancelled"]) and c["bench"] >= 0.85 * c["dur"]
            # Above ~2% the run lost requests to the frontend (see id 113:
            # "transport error"/"tcp connect error" from the request router),
            # so its RPM understates the engine.
            c["noisy"] = (c["err"] or 0) > 0.02
            c["quant"] = tag.startswith("off") and c["stamp"] < "20261110T000825"
            if tag.startswith("parS"):
                c["engine"] = "unieinfra-parS"
        else:
            # Directory names that miss the full pattern (FAILED-..., rerun copies)
            # still deserve a usable engine label instead of None; they have no
            # cap tag, so every cap-filtered analysis ignores them anyway.
            c["engine"] = os.path.basename(d).split("-")[0]
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vs", type=int, help="print the vLLM recipe point for this concurrency alongside")
    ap.add_argument("--engine", default=None)
    a = ap.parse_args()
    rows = [c for c in cells() if c["conc"] and (a.engine is None or c["engine"].startswith(a.engine))]
    rows.sort(key=lambda c: (c["engine"], c["conc"], c["cap"] or 0, c["dur"]))
    print(f"{'engine':16s} {'conc':>4s} {'cap':>4s} {'dur':>4s} {'RPM':>6s} {'inter':>6s} {'ITL50':>6s} {'ITL90':>6s} {'TTFT90':>7s} {'out/s':>6s} {'err%':>5s}")
    for c in rows:
        if c.get("valid") is False:
            continue
        name = c["engine"][:16] + ("(EP)" if c.get("ep") else "") + ("(Q8)" if c.get("quant") else "")
        print(
            f"{name:16s} {c['conc']:4d} {c['cap'] or 0:4d} {c['dur']:4d} {c['rpm']:6.1f} "
            f"{c['inter']:6.0f} {c['itl50'] or 0:6.1f} {c['itl90'] or 0:6.1f} {c['ttft90']:7.1f} "
            f"{c['outs'] or 0:6.0f} {100 * (c['err'] or 0):5.1f}"
        )
    if a.vs:
        ref = [c for c in rows if c["conc"] == a.vs and c["engine"].startswith("vllm") and c["cap"] == max(1, a.vs // 4)]
        if ref:
            r = ref[-1]
            print(f"\nreference: vLLM recipe c{a.vs} cap{max(1, a.vs // 4)} -> {r['rpm']:.1f} RPM at interactivity {r['inter']:.0f} (ITL p90 {r['itl90']:.1f} ms)")


if __name__ == "__main__":
    main()
