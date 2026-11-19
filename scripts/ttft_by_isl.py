"""TTFT (prefill latency) as a function of input length, per run.

Why: the c64 cliff is a long-context extend cost, so the fair question is how
each engine's time-to-first-token scales with the prompt it must prefill.
"""
import json,sys,statistics as st
B=[(0,8e3),(8e3,32e3),(32e3,64e3),(64e3,128e3),(128e3,192e3),(192e3,256e3),(256e3,1e9)]
def g(r,k):
    v=r['metrics'].get(k)
    return v.get('value') if isinstance(v,dict) else v
def med(v): return st.median(v) if v else float('nan')
for d in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f'{d}/aiperf_artifacts/profile_export.jsonl')]
    rows=[r for r in rows if r['metadata'].get('benchmark_phase')!='warmup' and g(r,'time_to_first_token') is not None]
    print('==',d.split('/')[-1], len(rows),'reqs')
    print('  isl_bucket            n   ttft50    ttft90   newtok50   new50  hit%   pfx_K/s  osl50')
    for lo,hi in B:
        s=[r for r in rows if lo<=g(r,'input_sequence_length')<hi]
        if not s: continue
        ttft=[g(r,'time_to_first_token')/1e3 for r in s]
        newt=[g(r,'usage_prompt_tokens')-g(r,'usage_prompt_cache_read_tokens') for r in s]
        hit=[100*g(r,'usage_prompt_cache_read_tokens')/max(1,g(r,'usage_prompt_tokens')) for r in s]
        osl=[g(r,'output_sequence_length') for r in s]
        # prefill speed over the newly computed tokens only
        rate=[n/(t) for n,t in zip(newt,ttft) if t>0]
        lab=f"{lo/1000:5.0f}-{min(hi,1e12)/1000:5.0f}K"
        print(f"  {lab:18s} {len(s):5d} {med(ttft):8.2f} {st.quantiles(ttft,n=10)[8] if len(ttft)>1 else ttft[0]:8.2f} {med(newt):9.0f} {med(newt)/1e3:7.0f} {med(hit):6.1f} {med(rate)/1e3:8.1f} {med(osl):7.0f}")
