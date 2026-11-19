"""Summarize TOKENSPEED_GPU_STEP_TIMING lines (TP rank 0) of a server log, skipping the first SKIP windows."""
import re,sys,statistics as S
f=sys.argv[1]; skip=int(sys.argv[2]) if len(sys.argv)>2 else 40
rows=[]
for l in open(f,errors='ignore'):
    if 'gpu_step_timing' not in l or 'RANK 0]' not in l: continue
    m=re.search(r'span_ms=(\d+) gap_ms=([\d.]+)',l); r={'span':float(m[1]),'gap':float(m[2])}
    for k in ("decode","extend","decode.step","decode.step_host","extend.step","extend.step_host"):
        mm=re.search(r'(?:^|\| )'+re.escape(k)+r': n=(\d+) sum_ms=(\d+) p50_ms=([\d.]+) mean_ms=([\d.]+)',l)
        r[k]=(int(mm[1]),float(mm[2]),float(mm[3]),float(mm[4])) if mm else (0,0,0,0)
    rows.append(r)
rows=rows[skip:]
if not rows: sys.exit('no rows')
span=sum(r['span'] for r in rows)
dn=sum(r['decode'][0] for r in rows); ds=sum(r['decode'][1] for r in rows)
en=sum(r['extend'][0] for r in rows); es=sum(r['extend'][1] for r in rows)
gap=sum(r['gap'] for r in rows)
p50=S.median([r['decode'][2] for r in rows if r['decode'][0]])
ex=" | ".join(f"{k} mean={sum(r[k][1] for r in rows)/max(sum(r[k][0] for r in rows),1):.1f}ms" for k in ("decode.step","decode.step_host","extend.step","extend.step_host"))
print(ex)
print(f"{f.split('/')[-1]}: windows={len(rows)} span={span/1e3:.0f}s | decode steps/6s={dn/span*6000:.1f} mean_ms={ds/max(dn,1):.1f} p50(of window p50)={p50:.1f} | extend n/6s={en/span*6000:.1f} mean_ms={es/max(en,1):.1f} | gap between forwards {gap/span*100:.2f}% of span | GPU-bracketed share decode {ds/span*100:.0f}% extend {es/span*100:.0f}%")
