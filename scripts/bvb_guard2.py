"""Early-stop guard for scripts/bvb_sweep.sh: after the four cells of a concurrency are done, compare each engine's best
cell (by RPM) and print the three target metrics; stop when UnieInfra's best is below 0.85x of vLLM's best on both RPM and
output tok/s."""
import glob, json, os, subprocess, sys, time
ROOT = "/home/smc/workspace/roy"
done = set()

def m(conc, kind, eng):
    tag = {"vllm": f"vllm-BV{conc}{kind}v", "unieinfra": f"unieinfra-BV{conc}{kind}u"}[eng]
    fs = sorted(glob.glob(f"{ROOT}/results/agentx/{tag}-c{conc}-d900-*/aiperf_artifacts/profile_export_aiperf.json"))
    if not fs: return None
    d = json.load(open(fs[-1])); g = lambda k, s: d[k][s]
    return dict(rpm=g("request_throughput", "avg") * 60, out=g("output_token_throughput", "avg"), ttft=g("time_to_first_token", "avg"),
                ttft50=g("time_to_first_token", "p50"), ttft90=g("time_to_first_token", "p90"), itl90=g("inter_token_latency", "p90"), n=d["request_count"]["avg"])

while True:
    for conc in (128, 64, 32, 4, 1):
        if conc in done: continue
        ks = "r" if conc == 4 else "rx"
        cells = {(e, k): m(conc, k, e) for e in ("vllm", "unieinfra") for k in ks}
        if any(v is None for v in cells.values()): continue
        done.add(conc)
        best = {e: max((cells[(e, k)] | {"kind": k} for k in ks), key=lambda c: c["rpm"]) for e in ("vllm", "unieinfra")}
        for e in ("vllm", "unieinfra"):
            for k in ks:
                c = cells[(e, k)]
                print(f"c{conc} {e:9s} cap-{k}: RPM {c['rpm']:6.1f} out {c['out']:5.0f} tok/s TTFT avg {c['ttft']:6.0f} p50 {c['ttft50']:6.0f} p90 {c['ttft90']:7.0f} ITL p90 {c['itl90']:5.1f} n={c['n']:.0f}", flush=True)
        u, v = best["unieinfra"], best["vllm"]
        print(f"c{conc} BEST-VS-BEST: RPM {u['rpm']/v['rpm']:.2f}x (UnieInfra cap-{u['kind']} vs vLLM cap-{v['kind']}) | out tok/s {u['out']/v['out']:.2f}x | TTFT avg vLLM/UnieInfra {v['ttft']/u['ttft']:.2f}x", flush=True)
        if u["rpm"] / v["rpm"] < 0.85 and u["out"] / v["out"] < 0.85:
            print(f"STOP: UnieInfra's best cell is below 0.85x of vLLM's best on RPM and output tok/s at c{conc}", flush=True)
            for pat in ("bvb_sweep2.sh",): subprocess.run(["pkill", "-f", pat])
            time.sleep(2); subprocess.run(["pkill", "-f", "run_round.sh BV"]); subprocess.run([f"{ROOT}/scripts/stop_servers.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            sys.exit(0)
    if len(done) == 5: print("GUARD DONE", flush=True); break
    time.sleep(20)
