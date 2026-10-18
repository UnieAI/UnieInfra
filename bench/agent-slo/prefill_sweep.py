#!/usr/bin/env python3
"""Prefill-only throughput vs request count — isolates serialization vs compute."""
import asyncio,json,sys,time,statistics
import aiohttp
base,model,data,conc,sizes,out=sys.argv[1:7]
conc=int(conc); sizes=[int(x) for x in sizes.split(',')]
prompts=[json.loads(l)["prompt"] for l in open(data)][:max(sizes)]
async def one(sess,sem,i,p,res):
    async with sem:
        t0=time.perf_counter()
        async with sess.post(f"{base}/v1/chat/completions",json={"model":model,"messages":[{"role":"user","content":p}],"max_tokens":1,"temperature":0,"stream":False,"ignore_eos":True},timeout=aiohttp.ClientTimeout(total=1800)) as r:
            j=json.loads(await r.read()); res[i]=round((time.perf_counter()-t0)*1000)
async def bench(s,n):
    sem=asyncio.Semaphore(conc); res={}
    async with aiohttp.ClientSession() as cs:
        t0=time.perf_counter()
        await asyncio.gather(*[one(cs,sem,i,p,res) for i,p in enumerate(prompts[:n])])
        return time.perf_counter()-t0, [res[i] for i in sorted(res)]
async def main():
    s=aiohttp.ClientSession(); out={}
    for n in sizes:
        dur,tt=await bench(s,n)
        out[n]=round(dur,2)
        print(f"n={n:3d} dur={dur:6.2f}s ttft_p50={statistics.median(tt):6.0f}ms p90={sorted(tt)[int(len(tt)*.9)]:6.0f}ms")
    await s.close(); json.dump(out,open(out_f:="prefill_sweep.json","w"))
asyncio.run(main())
