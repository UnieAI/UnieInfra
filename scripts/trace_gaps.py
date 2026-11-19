"""Find GPU idle gaps in a torch profiler chrome trace (.json.gz)."""
import collections, gzip, json, re, sys

tr = json.load(gzip.open(sys.argv[1]))
min_gap = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
ev = sorted((e for e in tr["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
end = ev[0]["ts"] + ev[0]["dur"]
gaps = []
for prev, e in zip(ev, ev[1:]):
    end = max(end, prev["ts"] + prev["dur"])
    gap = e["ts"] - end
    if gap > min_gap:
        gaps.append((gap, re.sub(r"_\d+$", "", prev["name"])[:60], re.sub(r"_\d+$", "", e["name"])[:60]))
total = ev[-1]["ts"] + ev[-1]["dur"] - ev[0]["ts"]
print(f"wall {total/1e3:.1f} ms, gaps>{min_gap:.0f}us: {len(gaps)} totaling {sum(g for g,_,_ in gaps)/1e3:.1f} ms")
by = collections.defaultdict(lambda: [0, 0.0])
for g, a, b in gaps:
    by[(a, b)][0] += 1; by[(a, b)][1] += g
for (a, b), (n, s) in sorted(by.items(), key=lambda kv: -kv[1][1])[:12]:
    print(f"{s/1e3:7.1f} ms n={n:4d} avg {s/n:7.0f} us  after [{a}] before [{b}]")
hist = collections.Counter(min(int(g // 100) * 100, 2000) for g, _, _ in gaps)
print("gap histogram (us):", dict(sorted(hist.items())))
