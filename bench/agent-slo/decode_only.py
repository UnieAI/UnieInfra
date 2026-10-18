#!/usr/bin/env python3
"""Decode-only: short prompts (negligible prefill), max_tokens=128, conc=N. Isolates decode ceiling."""
import asyncio,json,sys,time,statistics
import aiohttp
base,model,data,conc,out=sys.argv[1:6]; conc=int(conc)
prompts=[json.loads(l)["prompt"] for l in open(data)][:64]
async def one(sess,sem,i,p,res):
    async with sem:
        seg=[]
        t0=time.perf_counter(); tt=None
        async with sess.post(f"{base}/v1/chat/completions",json={"model":model,"messages":[{"role":"user","content":p}],"max_tokens":128,"temperature":0,"stream":True,"ignore_eos":True},timeout=aiohttp.ClientTimeout(total=1800)) as r:
            async for ln in r.content:
                ln=ln.decode('utf-8','replace').strip()
                if not ln.startswith('data:'): continue
                pl=ln[5:].strip()
                if pl=='[DONE]': break
                j=json.loads(pl); dt=(j.get('choices') or [{}])[0].get('delta',{}).get('content')
                if dt:
                    if tt is None: tt=(time.perf_counter()-t0)*1000
                    seg.append(dt)
        # Keep the text: a throughput A/B is only admissible if the two arms
        # agree token-for-token at temperature 0, and the corrupt arm can look
        # faster. `ttft` stays None when a request streamed no content at all;
        # the row is kept so a failed request shows up as ntok=0 and the caller
        # refuses the comparison instead of silently averaging it away.
        end=(time.perf_counter()-t0)*1000
        res[i]={"ttft":tt,"ntok":len(seg),"tpot":(end-tt) if tt is not None else None,"text":"".join(seg)}
async def main():
    sem=asyncio.Semaphore(conc);res={}
    async with aiohttp.ClientSession() as cs:
        t0=time.perf_counter()
        await asyncio.gather(*[one(cs,sem,i,p,res) for i,p in enumerate(prompts)])
        dur=time.perf_counter()-t0
    rows=[res[i] for i in sorted(res)]; ntok=sum(r['ntok'] for r in rows)
    json.dump({"duration_s":round(dur,3),"n":len(rows),"rows":rows},open(out,"w"))
    tp=[r['tpot']/r['ntok'] for r in rows if r['ntok']]
    empty=sum(1 for r in rows if not r['ntok'])
    print(f"{out}: decode-only conc={conc} n={len(rows)} ntok={ntok} dur={dur:.2f}s tps={ntok/dur:.1f} tpot_p50={statistics.median(tp) if tp else float('nan'):.1f}ms empty={empty}")
asyncio.run(main())
