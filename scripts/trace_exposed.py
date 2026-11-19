"""GPU-only trace: per-category kernel time, by stream, and how much of it is exposed (no other-stream kernel running)."""
import gzip,json,glob,collections,sys,re
f=glob.glob(sys.argv[1]+'/*TP0*')[0]
K=[e for e in json.load(gzip.open(f))['traceEvents'] if e.get('ph')=='X' and e.get('cat') in('kernel','gpu_memcpy','gpu_memset')]
def cat(n):
    for c,p in [('AR-iris','iris_'),('RCCL','nccl'),('MoE','moe|_gather_package|_mxfp4_quantize|sqrt_softplus_topk|silu_and_mul'),('attn','attention|dsv4_prefill|rope|swa_'),
                ('indexer','index_topk'),('mHC','mhc'),('GEMM','mm_mxfp8|Cijk|quantize_fp8|rmsnorm_fp8'),('copy/misc','copy|elementwise|Fill|pack|gather|index|SoftMax|memcpy|memset')]:
        if re.search(p,n): return c
    return 'other'
K.sort(key=lambda k:k['ts']); t0=K[0]['ts']; t1=max(k['ts']+k['dur'] for k in K)
# union busy
ev=sorted([(k['ts'],1) for k in K]+[(k['ts']+k['dur'],-1) for k in K]); busy=0;d=0;last=t0
for t,s in ev:
    if d>0: busy+=t-last
    d+=s; last=t
print(f"window {(t1-t0)/1e3:.0f} ms  union busy {busy/1e3:.0f} ms ({busy/(t1-t0)*100:.0f}%)  idle {(t1-t0-busy)/1e3:.0f} ms")
# exposure: for each kernel, time not overlapped by any kernel on another stream
by_s=collections.defaultdict(list)
for k in K: by_s[k['args'].get('stream')].append((k['ts'],k['ts']+k['dur']))
def merged(iv):
    out=[]
    for a,b in sorted(iv):
        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out
other={s:merged([x for t,v in by_s.items() if t!=s for x in v]) for s in by_s}
import bisect
starts={s:[a for a,b in other[s]] for s in other}
def overlap(s,a,b):
    iv=other[s]; i=max(bisect.bisect_right(starts[s],a)-1,0); o=0
    while i<len(iv) and iv[i][0]<b:
        o+=max(0,min(b,iv[i][1])-max(a,iv[i][0])); i+=1
    return o
tot=collections.Counter(); exp=collections.Counter(); cnt=collections.Counter()
for k in K:
    c=cat(k['name']); s=k['args'].get('stream'); key=(c,s)
    big = c=='AR-iris' and k['dur']>500
    if c=='AR-iris': key=('AR-iris '+('prefill' if big else 'decode'),s)
    tot[key]+=k['dur']; exp[key]+=k['dur']-overlap(s,k['ts'],k['ts']+k['dur']); cnt[key]+=1
W=(t1-t0)/100
for key in sorted(tot,key=lambda x:-exp[x]):
    print(f"{key[0]:20s} stream {key[1]}  n {cnt[key]:6d}  total {tot[key]/1e3:7.1f} ms  exposed {exp[key]/1e3:7.1f} ms ({exp[key]/W:4.1f}% wall)")
