"""Throughput vs interactivity table from AiPerf exports (official-mode runs).

Interactivity is 1000 / p90 inter-token latency (ms): the per-client generation
speed InferenceX derives from p90 ITL. Usage: python3 scripts/pareto_table.py
"""
import glob, json, os, re

ROOT = "/home/smc/workspace/roy/results/agentx"


def load(run):
    f = f"{run}/aiperf_artifacts/profile_export_aiperf.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    g = lambda k, s: d.get(k, {}).get(s)
    itl90 = g("inter_token_latency", "p90")
    return {
        "rpm": (g("request_throughput", "avg") or 0) * 60,
        "out_tps": g("output_token_throughput", "avg"),
        "itl50": g("inter_token_latency", "p50"),
        "itl90": itl90,
        "inter": 1000 / itl90 if itl90 else None,
        "ttft50": g("time_to_first_token", "p50"),
        "ttft90": g("time_to_first_token", "p90"),
        "n": d.get("request_count", {}).get("avg") if isinstance(d.get("request_count"), dict) else d.get("request_count"),
    }


rows = []
for run in sorted(glob.glob(f"{ROOT}/*")):
    name = os.path.basename(run)
    if name.startswith("FAILED"):
        continue
    m = re.match(r"unieinfra-parS(\d+)-c(\d+)-d(\d+)-", name)
    if m:
        cap, conc, dur = map(int, m.groups())
        r = load(run)
        if r:
            rows.append(("UnieInfra (coupled, sampled)", conc, f"cap {cap}", dur, r))
        continue
    m = re.match(r"vllm-(baseline|baseline-rerun|highconc-rerun|final)-c(\d+)-d900", name)
    if m:
        r = load(run)
        if r:
            rows.append(("vLLM official recipe", int(m.group(2)), "recipe", 900, r))
rows.sort(key=lambda x: (x[1], x[0], x[2]))
print(f"{'engine':30s} {'conc':>4s} {'cap':>8s} {'dur':>4s} {'RPM':>6s} {'out tok/s':>9s} {'ITL p50':>7s} {'ITL p90':>7s} {'inter.':>7s} {'TTFT p50':>8s} {'TTFT p90':>8s}")
for eng, conc, cap, dur, r in rows:
    f = lambda v, p=1: f"{v:.{p}f}" if v is not None else "-"
    print(f"{eng:30s} {conc:4d} {cap:>8s} {dur:4d} {f(r['rpm']):>6s} {f(r['out_tps'],0):>9s} {f(r['itl50']):>7s} {f(r['itl90']):>7s} {f(r['inter'],0):>7s} {f(r['ttft50'],0):>8s} {f(r['ttft90'],0):>8s}")
