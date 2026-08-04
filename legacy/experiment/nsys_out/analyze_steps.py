import csv
import statistics as st
import sys
from collections import defaultdict

path = sys.argv[1]
rows = []
with open(path) as f:
    r = csv.DictReader(f)
    for d in r:
        try:
            start = int(d["Start (ns)"])
            dur = int(d["Duration (ns)"])
        except:
            continue
        rows.append((start, dur, d.get("GrdX", ""), d["Name"]))
rows.sort()
print(f"total kernels in trace: {len(rows)}")


def cat(name):
    if "gemvx" in name:
        return "gemv"
    if "BatchPrefillWithPagedKV" in name or "BatchDecodeWithPagedKV" in name:
        return "attention"
    if "RMSNorm" in name.lower() or "rmsnorm" in name.lower() or "RMSNorm" in name:
        return "rmsnorm"
    if "Rotary" in name or "rotary" in name or "rope" in name:
        return "rope"
    if "act_and_mul" in name or "silu" in name:
        return "silu"
    if "store_kv" in name or "kv_cache" in name:
        return "kv_store"
    if "cutlass" in name and "wmma" in name:
        return "cutlass_wmma"
    return "other_glue"


# lm_head = gemv with large duration (reads ~1.24GB weights). Use >250us to segment.
lm = [i for i, (s, du, g, n) in enumerate(rows) if "gemvx" in n and du > 250000]
print(f"lm_head-class gemv count (dur>250us): {len(lm)}")
if len(lm) >= 3:
    durs = [rows[i][1] for i in lm]
    print(
        f"  lm_head dur: min={min(durs)} med={int(st.median(durs))} max={max(durs)} ns"
    )

# segment steps by consecutive lm_head starts
steps = []
for k in range(len(lm) - 1):
    a = rows[lm[k]][0]
    b = rows[lm[k + 1]][0]
    seg = [r for r in rows if a <= r[0] < b]
    steps.append((a, b, seg))
print(f"num full steps segmented: {len(steps)}")

# pick steady-state steps: middle third
n = len(steps)
lo = n // 3
hi = 2 * n // 3 if 2 * n // 3 > lo else n
sel = steps[lo:hi] if hi > lo else steps
print(f"analyzing steady-state steps [{lo}:{hi}) = {len(sel)} steps")

periods = []
busys = []
kcounts = []
cat_dur = defaultdict(float)
cat_cnt = defaultdict(int)
gemv_durs = []
for a, b, seg in sel:
    period = b - a
    busy = sum(du for (s, du, g, nn) in seg)
    periods.append(period)
    busys.append(busy)
    kcounts.append(len(seg))
    for s, du, g, nn in seg:
        c = cat(nn)
        cat_dur[c] += du
        cat_cnt[c] += 1
        if c == "gemv":
            gemv_durs.append((du, g))

ns = len(sel)
mp = st.median(periods)
mb = st.median(busys)
print(f"\n=== STEADY-STATE PER-TOKEN (median over {ns} steps) ===")
print(
    f"wall period/token : {mp/1000:.1f} us  -> {1e9/mp:.1f} tok/s (this run, under nsys)"
)
print(
    f"GPU busy/token    : {mb/1000:.1f} us  -> {1e9/mb:.1f} tok/s (back-to-back floor)"
)
print(f"idle gap/token    : {(mp-mb)/1000:.1f} us  = {100*(mp-mb)/mp:.1f}% of wall")
print(f"kernels/token     : {int(st.median(kcounts))}")

print(f"\n=== GPU-BUSY BREAKDOWN BY CATEGORY (avg per token) ===")
tot = sum(cat_dur.values())
for c in sorted(cat_dur, key=lambda x: -cat_dur[x]):
    print(
        f"  {c:14s} {cat_dur[c]/ns/1000:8.2f} us/tok  {100*cat_dur[c]/tot:5.1f}%  ({cat_cnt[c]/ns:.0f} launches/tok)"
    )

# cluster gemv durations -> map to projections by known bytes
print(f"\n=== GEMV DURATION CLUSTERS (per token) ===")
# bucket by rounded duration
from collections import Counter

buck = defaultdict(list)
for du, g in gemv_durs:
    buck[round(du, -3)].append((du, g))  # round to nearest 1000ns
# aggregate into clusters by proximity
alld = sorted(du for du, g in gemv_durs)
# print histogram of gemv durations
import math

clusters = []
cur = [alld[0]]
for x in alld[1:]:
    if x - cur[-1] <= max(2000, 0.15 * cur[-1]):
        cur.append(x)
    else:
        clusters.append(cur)
        cur = [x]
clusters.append(cur)
PEAK = 1.60e12
sizes = {  # bytes
    "qkv_fused(50.3MB)": 4096 * 6144 * 2,
    "q(33.6MB)": 4096 * 4096 * 2,
    "kv(8.4MB)": 4096 * 1024 * 2,
    "o(33.6MB)": 4096 * 4096 * 2,
    "mlp(100.7MB)": 4096 * 12288 * 2,
    "lmhead(1244.7MB)": 4096 * 151936 * 2,
}
for cl in clusters:
    md = st.median(cl)
    cnt = len(cl) / ns
    # best matching known size by bandwidth plausibility
    print(f"  dur~{md/1000:8.2f}us  count/tok={cnt:4.1f}  n={len(cl)}")
print(f"\n  (peak DRAM {PEAK/1e12:.2f} TB/s; achieved BW = weight_bytes / dur)")
print(
    "  reference weight bytes: qkv_fused=50.3MB o=33.6MB mlp_each=100.7MB lmhead=1244.7MB"
)
for label, byt in sizes.items():
    print(f"    {label:20s}: {byt/1e6:7.1f} MB -> at peak needs {byt/PEAK*1e6:7.1f} us")
