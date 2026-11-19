import re,sys,statistics as st,collections
pat=re.compile(r"(\d\d:\d\d:\d\d),\d+.*?Decode batch.*?#running-req: (\d+), avg_seq_len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+), avg_accept_len: ([\d.]+), accept_rate: ([\d.-]+)")
for L in sys.argv[1:]:
    rows=[]
    for line in open(L,errors='ignore'):
        m=pat.search(line)
        if m: rows.append(m.groups())
    if not rows: print(L,'no rows'); continue
    print('==',L,len(rows),'lines')
    b=collections.defaultdict(list)
    for t,bs,sl,gt,acc,ar in rows:
        mm=int(t[3:5])*60+int(t[6:8])
        b[mm//30].append((int(bs),float(sl),float(gt),float(acc),float(ar)))
    for k in sorted(b):
        v=b[k]
        print(f"  t={k*0.5:5.1f}m n={len(v):3d} bs={st.median([x[0] for x in v]):4.0f} seqlen={st.median([x[1] for x in v])/1000:6.1f}K acc={st.median([x[3] for x in v]):4.2f} rate={st.median([x[4] for x in v]):5.2f} gt={st.median([x[2] for x in v]):6.0f}")
