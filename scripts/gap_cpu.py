"""For GPU idle gaps in a CPU+GPU torch trace, report which host ops were running and which runtime call ended the gap."""
import gzip, json, sys, collections, re
tr = json.load(gzip.open(sys.argv[1])); min_gap = float(sys.argv[2]) if len(sys.argv) > 2 else 300
E_ = tr["traceEvents"]
gpu = sorted((e for e in E_ if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
cpu = [e for e in E_ if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "python_function", "cuda_runtime")]
corr = {e["args"].get("correlation"): e for e in E_ if e.get("cat") == "cuda_runtime" and "args" in e}
gaps = []; end = gpu[0]["ts"] + gpu[0]["dur"]
for p, e in zip(gpu, gpu[1:]):
    end = max(end, p["ts"] + p["dur"]); g = e["ts"] - end
    if g > min_gap: gaps.append((end, e["ts"], e))
print(f"{len(gaps)} gaps > {min_gap}us, {sum(b-a for a,b,_ in gaps)/1e3:.1f} ms")
during = collections.Counter(); enders = collections.Counter(); launch_lag = []
for a, b, e in gaps:
    rt = corr.get(e.get("args", {}).get("correlation"))
    if rt:
        enders[re.sub(r"\d+", "#", rt["name"])[:40]] += 1
        launch_lag.append(b - rt["ts"])  # GPU start minus host launch time
    # innermost annotations overlapping the gap (by duration < 50 ms)
    act = [c for c in cpu if c["ts"] < b and c["ts"] + c["dur"] > a and c["dur"] < 5e4 and c.get("cat") in ("user_annotation", "cpu_op")]
    for c in act: during[c["name"][:70]] += min(b, c["ts"] + c["dur"]) - max(a, c["ts"])
print("gap ended by runtime call:", enders.most_common(6))
if launch_lag: print(f"GPU start - host enqueue at gap end: median {sorted(launch_lag)[len(launch_lag)//2]:.0f} us")
print("host ops overlapping gaps (us total):")
for n, us in during.most_common(25): print(f"  {us/1e3:7.1f} ms  {n}")
