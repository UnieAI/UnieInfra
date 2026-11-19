"""Early-stop guard for the fair sweep (scripts/aligned_sweep2.sh).

After each finished cell, when both the vLLM (FS<c>v) and the UnieInfra (FS<c>u) result of a concurrency exist, print
the ratios UnieInfra / vLLM for RPM, output tok/s, request count and TTFT p50 (vLLM / UnieInfra, so >1 is better), and
stop the sweep when, at c >= 4, RPM or output tok/s is below 0.95x of vLLM, TTFT p50 is more than 5% worse, or fewer
than 70% of vLLM's requests completed. c1 is demand-limited and is only reported.
"""
import glob, json, os, re, subprocess, sys, time

ROOT = "/home/smc/workspace/roy"
LOG = f"{ROOT}/logs/aligned_sweep2.log"
seen = set()


def metrics(conc, engine):
    tag = {"vllm": f"vllm-FS{conc}v", "unieinfra": f"unieinfra-FS{conc}u"}[engine]
    fs = sorted(glob.glob(f"{ROOT}/results/agentx/{tag}-c{conc}-d900-*/aiperf_artifacts/profile_export_aiperf.json"))
    if not fs:
        return None
    d = json.load(open(fs[-1]))
    g = lambda k, s: d.get(k, {}).get(s)
    return {"rpm": g("request_throughput", "avg") * 60, "out": g("output_token_throughput", "avg"),
            "ttft50": g("time_to_first_token", "p50"), "ttft90": g("time_to_first_token", "p90"),
            "itl90": g("inter_token_latency", "p90"), "n": d["request_count"]["avg"]}


def stop(why):
    print(f"STOP: {why}", flush=True)
    for pat in ("aligned_sweep2.sh", "aligned_sweep2_fix.sh"):
        subprocess.run(["pkill", "-f", pat])
    time.sleep(2)
    subprocess.run(["pkill", "-f", "run_round.sh FS"])
    subprocess.run([f"{ROOT}/scripts/stop_servers.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sys.exit(0)


while True:
    text = open(LOG).read() if os.path.exists(LOG) else ""
    for conc in (1, 4, 16, 32, 64, 128):
        if conc in seen:
            continue
        v, u = metrics(conc, "vllm"), metrics(conc, "unieinfra")
        if not (v and u):
            continue
        seen.add(conc)
        r = lambda a, b: a / b if b else float("nan")
        rpm, out, nreq = r(u["rpm"], v["rpm"]), r(u["out"], v["out"]), r(u["n"], v["n"])
        ttft = r(v["ttft50"], u["ttft50"])
        print(f"c{conc}: RPM {u['rpm']:.1f} vs {v['rpm']:.1f} = {rpm:.2f}x | out tok/s {u['out']:.0f} vs {v['out']:.0f} = {out:.2f}x | "
              f"TTFT p50 {u['ttft50']:.0f} vs {v['ttft50']:.0f} ms (vLLM/UnieInfra {ttft:.2f}x) | TTFT p90 {u['ttft90']:.0f} vs {v['ttft90']:.0f} | "
              f"ITL p90 {u['itl90']:.1f} vs {v['itl90']:.1f} | requests {u['n']:.0f} vs {v['n']:.0f}", flush=True)
        # Target: more requests per minute, lower TTFT, higher output throughput than vLLM (5% noise margin each).
        if conc >= 4 and (rpm < 0.95 or out < 0.95 or ttft < 1 / 1.05 or nreq < 0.7):
            stop(f"c{conc}: UnieInfra misses the target vs vLLM: RPM {rpm:.2f}x, output tok/s {out:.2f}x, "
                 f"TTFT p50 vLLM/UnieInfra {ttft:.2f}x, requests {nreq:.2f}x (needs RPM>=0.95, out>=0.95, TTFT ratio>=0.95, requests>=0.7)")
    if "SWEEP DONE" in text and "FIX DONE" in open(f"{ROOT}/logs/aligned_sweep2_fix.log").read() if os.path.exists(f"{ROOT}/logs/aligned_sweep2_fix.log") else False:
        print("GUARD DONE", flush=True)
        break
    time.sleep(20)
