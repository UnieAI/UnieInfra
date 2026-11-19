"""Split an AgentX-load trace into forward steps by mode (tokenspeed.forward.v1 annotations, rank given)
and attribute GPU kernel time between consecutive step starts to the earlier step."""
import gzip,json,glob,sys,collections,re,bisect
f=glob.glob(sys.argv[1]+'/*TP0*')[0]
E=json.load(gzip.open(f))['traceEvents']
ann=sorted([e for e in E if e.get('ph')=='X' and str(e.get('name','')).startswith('tokenspeed.forward.v1')],key=lambda e:e['ts'])
ker=sorted([e for e in E if e.get('ph')=='X' and e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')],key=lambda e:e['ts'])
def mode(e):
    m=re.search(r"'mode':\s*'(\w+)'",e['name']); n=re.search(r"'num_extends':\s*(\d+)",e['name'])
    md=m.group(1) if m else '?'; ne=int(n.group(1)) if n else 0
    return md if md!='DECODE' or ne==0 else 'DECODE+EXT'
t0=ker[0]['ts']; t1=max(k['ts']+k['dur'] for k in ker); wall=(t1-t0)/1e3
busy=collections.Counter(); span=collections.Counter(); cnt=collections.Counter()
starts=[a['ts'] for a in ann]
# GPU kernels are launched after the CPU annotation; assign each kernel to the last annotation that started before it
for k in ker:
    i=bisect.bisect_right(starts,k['ts'])-1
    md=mode(ann[i]) if i>=0 else 'pre'
    busy[md]+=k['dur']/1e3
for a,b in zip(ann,ann[1:]+[None]):
    md=mode(a); cnt[md]+=1; span[md]+=((b['ts'] if b else t1)-a['ts'])/1e3
tot_busy=sum(busy.values())
print(f"window {wall:.0f} ms, GPU busy {tot_busy:.0f} ms ({tot_busy/wall*100:.0f}%), idle {wall-tot_busy:.0f} ms")
for md in sorted(span,key=lambda m:-span[m]):
    print(f"  {md:11s} steps {cnt[md]:4d}  wall {span[md]:7.0f} ms ({span[md]/wall*100:4.1f}%)  GPU busy {busy[md]:7.0f} ms  per step {span[md]/cnt[md]:6.1f} ms")
