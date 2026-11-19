"""Split server wall time into prefill rounds vs decode rounds from 'Prefill batch.'/'Decode batch.' lines (UnieInfra)."""
import re, sys, datetime as dt, collections
L = sys.argv[1]
ev = []
for line in open(L, errors="ignore"):
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    m = re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)\s+ATTN TP RANK 0\] - INFO - (Prefill|Decode) batch\.(.*)", line)
    if not m: continue
    t = dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S,%f").timestamp()
    nt = re.search(r"#new-token: (\d+)", m[3]); run = re.search(r"#running-req: (\d+)", m[3])
    ev.append((t, m[2], int(nt[1]) if nt else 0, int(run[1]) if run else 0))
t0, t1 = float(sys.argv[2]) if len(sys.argv) > 2 else ev[0][0], ev[-1][0]
pre = dec = 0; ptok = 0; bands = collections.defaultdict(lambda: [0, 0.0])
for a, b in zip(ev, ev[1:]):
    d = b[0] - a[0]
    if d > 5: continue  # idle
    if a[1] == "Prefill":
        pre += d; ptok += a[2]; k = min(a[2] // 2048, 4); bands[k][0] += 1; bands[k][1] += d
    else: dec += d
tot = pre + dec
print(f"busy {tot:.0f} s: prefill rounds {pre:.0f} s ({100*pre/tot:.0f}%), decode {dec:.0f} s; prefill tokens {ptok/1e6:.2f}M -> {ptok/pre/1e3:.1f}K tok/s")
for k in sorted(bands): print(f"  new-token band {k*2048}-{(k+1)*2048 if k<4 else 'max'}: n={bands[k][0]} avg {1e3*bands[k][1]/bands[k][0]:.0f} ms")
