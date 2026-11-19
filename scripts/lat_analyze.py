"""Per-request latency breakdown of the UnieInfra c4 run with --enable-log-request-stats (LAT4u) against vLLM's FS4v."""
import glob, json, re, statistics as st

ROOT = "/home/smc/workspace/roy"
log = sorted(glob.glob(f"{ROOT}/logs/unieinfra-server-c4-*.log"))[-1]
txt = re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="ignore").read())
rows = []
for m in re.finditer(r"Req: (\S+) Finish! RequestStats\(status='(\w+)', reason='(\w+)', prompt_tokens=(\d+), cache_tokens=(\d+), output_tokens=(\d+), cache_hit_rate=[\d.]+, queue_ms=([\d.]+), prefill_ms=([\d.]+), ttft_ms=([\d.]+), total_ms=([\d.]+), preempt_ms=([\d.]+).*?decode_tps=([\d.]+), acc_len=([\d.None]+)", txt):
    if m[1].startswith(("WARMUP", "HEALTH")):
        continue
    rows.append(dict(prompt=int(m[4]), cache=int(m[5]), out=int(m[6]), queue=float(m[7]), prefill=float(m[8]), ttft=float(m[9]), total=float(m[10]), tps=float(m[12])))
print(log.split("/")[-1], len(rows), "requests")
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
f = lambda xs: f"avg {st.mean(xs):7.1f}  p50 {q(xs, .5):7.1f}  p90 {q(xs, .9):7.1f}  p99 {q(xs, .99):7.1f}"
new = [r["prompt"] - r["cache"] for r in rows]
print("new (uncached) tokens/request   ", f(new))
print("queue_ms                         ", f([r["queue"] for r in rows]))
print("prefill_ms                       ", f([r["prefill"] for r in rows]))
print("ttft_ms                          ", f([r["ttft"] for r in rows]))
print("decode tok/s per request         ", f([r["tps"] for r in rows if r["tps"] > 0]))
for lo, hi, name in ((0, 1000, "new < 1K"), (1000, 4000, "new 1-4K"), (4000, 10**9, "new >= 4K")):
    sel = [r for r in rows if lo <= r["prompt"] - r["cache"] < hi]
    if sel:
        print(f"{name:10s} n={len(sel):3d}  ttft avg {st.mean(r['ttft'] for r in sel):6.0f} ms, prefill avg {st.mean(r['prefill'] for r in sel):6.0f} ms, queue avg {st.mean(r['queue'] for r in sel):5.0f} ms, new tokens avg {st.mean(r['prompt']-r['cache'] for r in sel):6.0f}")
fv = glob.glob(f"{ROOT}/results/agentx/vllm-FS4v-c4-d900-*/aiperf_artifacts/server_metrics_export.json")[0]
d = json.load(open(fv))["metrics"]
avg = lambda n: d[n]["series"][0]["stats"]["avg"]
print(f"vLLM FS4v: queue {avg('vllm:request_queue_time_seconds')*1e3:.0f} ms, prefill {avg('vllm:request_prefill_time_seconds')*1e3:.0f} ms, "
      f"ttft {avg('vllm:time_to_first_token_seconds')*1e3:.0f} ms (avg), new tokens {avg('vllm:request_prefill_kv_computed_tokens'):.0f} per request, "
      f"per-iteration inter-token {avg('vllm:inter_token_latency_seconds')*1e3:.1f} ms")
