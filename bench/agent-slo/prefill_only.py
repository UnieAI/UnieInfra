#!/usr/bin/env python3
"""Isolate prefill: max_tokens=1, all 64 prompts, conc=8. Measures prefill tok/s."""
import asyncio,json,sys,time
import aiohttp
base,model,data,conc,out=sys.argv[1:6]
conc=int(conc)
prompts=[json.loads(l)["prompt"] for l in open(data)][:64]
async def one(sess,sem,i,p,res):
    async with sem:
        t0=time.perf_counter()
        async with sess.post(f"{base}/v1/chat/completions",json={
            "model":model,"messages":[{"role":"user","content":p}],"max_tokens":1,
            "temperature":0,"stream":False,"ignore_eos":True},timeout=aiohttp.ClientTimeout(total=1800)) as r:
            j=json.loads(await r.read())
            res[i]={"ttft":round((time.perf_counter()-t0)*1000),"ntok":j["usage"].get("completion_tokens",1)}
async def main():
    sem=asyncio.Semaphore(conc);res={}
    async with aiohttp.ClientSession() as s:
        t0=time.perf_counter()
        await asyncio.gather(*[one(s,sem,i,p,res) for i,p in enumerate(prompts)])
        dur=time.perf_counter()-t0
    json.dump({"duration_s":round(dur,2),"n":len(res),"rows":[res[i] for i in sorted(res)]},open(out,"w"))
    tt=[r["ttft"] for r in res.values()]
    import statistics
    print(f"{out}: prefill-only dur={dur:.1f}s ttftp50={statistics.median(tt):.0f}ms p90={sorted(tt)[int(len(tt)*.9)]:.0f}ms")
asyncio.run(main())
