"""Per-slot economics of an AgentX cell: is the RPM gap slots or per-slot speed?

For each run: completed requests, the wall clock they span, median prompt tokens
that were actually computed (cache miss), median output tokens, and RPM both
overall and per configured slot. Comparing engines on "RPM per slot" separates
"we schedule fewer requests" from "each request is slower".
"""
import json,sys,statistics as st
def g(r,k):
    v=r['metrics'].get(k)
    return v.get('value') if isinstance(v,dict) else v
def load(d):
    rows=[json.loads(l) for l in open(f'{d}/aiperf_artifacts/profile_export.jsonl')]
    return [r for r in rows if r['metadata'].get('benchmark_phase')!='warmup']
for spec in sys.argv[1:]:
    d,slots=spec.split(',')
    rows=load(d)
    t0=min(r['metadata']['request_start_ns'] for r in rows)
    t1=max(r['metadata']['request_end_ns'] for r in rows)
    wall=(t1-t0)/1e9
    newt=[g(r,'usage_prompt_tokens')-g(r,'usage_prompt_cache_read_tokens') for r in rows]
    isl=[g(r,'input_sequence_length') for r in rows]
    osl=[g(r,'output_sequence_length') for r in rows]
    lat=[g(r,'request_latency')/1e3 for r in rows]
    rpm=len(rows)/wall*60
    slots=int(slots)
    print(f"{d.split('/')[-1][:46]:46s} reqs={len(rows):5d} wall={wall:6.0f}s "
          f"rpm={rpm:5.1f} slots={slots:3d} rpm/slot={rpm/slots:5.2f} "
          f"isl50={st.median(isl)/1e3:6.1f}K new50={st.median(newt)/1e3:5.1f}K "
          f"osl50={st.median(osl):5.0f} lat50={st.median(lat):6.1f}s "
          f"out/s={sum(osl)/wall:6.0f} newtok/s={sum(newt)/wall/1e3:5.1f}K")
