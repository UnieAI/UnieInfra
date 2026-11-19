"""Per cold prompt: GPU forward time, host launch time and GPU idle from gpu_step_timing (rank 0).
Usage: prefill_timing_split.py logs/rocm_prefill_t.log"""
import re, sys
from datetime import datetime
M = sys.argv[1]
cur = None; groups = []  # (rt, log, [(label, t_start, latency)])
for l in open(M):
    m = re.match(r"(\S+) log (\S+)", l)
    if m: groups.append((m.group(1), m.group(2), [])); continue
    m = re.match(r"(\S+) mark (\S+) (\d\d:\d\d:\d\d)", l)
    if m and groups: groups[-1][2].append([m.group(2), m.group(3), None]); continue
    m = re.match(r"\S+ (\S+) round 0: .* ok ([\d.]+)s", l)
    if m and groups and groups[-1][2]: groups[-1][2][-1][2] = float(m.group(2))
pat = re.compile(r"(\d\d:\d\d:\d\d),\d+\s+ATTN TP RANK 0\].*gpu_step_timing span_ms=(\d+) gap_ms=([\d.]+)(.*)")
def field(rest, key):
    m = re.search(rf"\| {re.escape(key)}: n=(\d+) sum_ms=(\d+) p50_ms=([\d.]+) mean_ms=([\d.]+)", rest)
    return (int(m.group(1)), int(m.group(2)), float(m.group(4))) if m else (0, 0, 0.0)
print(f"{'rt':5s} {'prompt':8s} {'lat_s':>6s} {'ext_n':>5s} {'ext_gpu_s':>9s} {'ext_host_s':>10s} {'gap_s':>6s} {'max_win_ext_mean_ms':>19s}")
for rt, log, marks in groups:
    lines = [pat.search(l) for l in open(log.replace("/home/smc/workspace/roy/logs", "/home/smc/workspace/roy/logs"), errors="ignore")]
    lines = [m for m in lines if m]
    for i, (label, t0, lat) in enumerate(marks):
        t1 = marks[i + 1][1] if i + 1 < len(marks) else "99:99:99"
        win = [m for m in lines if t0 <= m.group(1) < t1]
        n = g = h = 0; gap = 0.0; mx = 0.0
        for m in win:
            en, es, emean = field(m.group(4), "extend"); hn, hs, _ = field(m.group(4), "extend.step_host")
            n += en; g += es; h += hs; gap += float(m.group(3)); mx = max(mx, emean)
        print(f"{rt:5s} {label:8s} {lat or 0:6.1f} {n:5d} {g/1000:9.2f} {h/1000:10.2f} {gap/1000:6.2f} {mx:19.1f}")
