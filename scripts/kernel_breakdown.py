"""Summarize GPU kernel time from a torch profiler chrome trace (.json.gz)."""
import collections, gzip, json, re, sys

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
with gzip.open(path) as f:
    tr = json.load(f)
ev = [e for e in tr["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
tot = collections.Counter(); cnt = collections.Counter()
for e in ev:
    name = re.sub(r"_\d+$", "", e["name"])[:90]
    tot[name] += e["dur"]; cnt[name] += 1
start = min(e["ts"] for e in ev); end = max(e["ts"] + e["dur"] for e in ev)
busy = sum(tot.values())
print(f"wall={ (end-start)/1e3:.1f} ms  gpu_busy={busy/1e3:.1f} ms ({100*busy/(end-start):.0f}%)  kernels={len(ev)}")
for name, us in tot.most_common(top):
    print(f"{us/1e3:9.2f} ms {100*us/busy:5.1f}%  n={cnt[name]:6d}  {name}")
