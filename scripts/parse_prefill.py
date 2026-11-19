import re,sys,collections
pat=re.compile(r"(\d\d):(\d\d):(\d\d),\d+.*?(Prefill|Mix) batch.*?#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), #running-req: (\d+)")
dec=re.compile(r"(\d\d):(\d\d):(\d\d),\d+.*?Decode batch.*?#running-req: (\d+).*?gen throughput \(token/s\): ([\d.]+), avg_accept_len: ([\d.]+)")
for L in sys.argv[1:]:
    print('==',L)
    b=collections.defaultdict(lambda:[0,0,0,0])  # extends, newtok, mix, maxtok
    d=collections.defaultdict(list)
    for line in open(L,errors='ignore'):
        m=pat.search(line)
        if m:
            k=(int(m.group(1)),int(m.group(2))//1)
            e=b[k]; e[0]+=1; e[1]+=int(m.group(6)); e[2]+= (m.group(4)=='Mix'); e[3]=max(e[3],int(m.group(6)))
        m2=dec.search(line)
        if m2:
            d[(int(m2.group(1)),int(m2.group(2)))].append((int(m2.group(4)),float(m2.group(5)),float(m2.group(6))))
    for k in sorted(b):
        e=b[k]
        ds=d.get(k,[])
        s=f"  {k[0]:02d}:{k[1]:02d} extends={e[0]:3d} newtok={e[1]:7d} mix={e[2]:3d} maxtok={e[3]:5d}"
        if ds: s+=f" | dec bs={sum(x[0] for x in ds)/len(ds):4.1f} gt={sum(x[1] for x in ds)/len(ds):6.0f} acc={sum(x[2] for x in ds)/len(ds):4.2f}"
        print(s)
