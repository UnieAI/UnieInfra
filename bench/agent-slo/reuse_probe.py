"""重用路徑的隔離量測：同一個長 prompt，冷送一次、熱送三次。"""
import asyncio, aiohttp, json, sys, time, statistics
PORT=int(sys.argv[1]); TAG=sys.argv[2]; NTOK=int(sys.argv[3]) if len(sys.argv)>3 else 6000
GAP=float(sys.argv[4]) if len(sys.argv)>4 else 0.0
URL=f"http://127.0.0.1:{PORT}/v1/chat/completions"
async def one(s,prompt):
    body={"model":"q27","messages":[{"role":"user","content":prompt}],
          "max_tokens":1,"temperature":0,"stream":True,"ignore_eos":True}
    t0=time.perf_counter()
    async with s.post(URL,json=body,timeout=aiohttp.ClientTimeout(total=900)) as r:
        async for raw in r.content:
            l=raw.decode('utf-8','replace').strip()
            if l.startswith('data:') and l[5:].strip() not in ('[DONE]',''):
                j=json.loads(l[5:])
                if (j["choices"][0].get("delta") or {}).get("content"):
                    return time.perf_counter()-t0
    return None
async def main():
    pad=" ".join(f"tok{j}" for j in range(NTOK))
    out={}
    async with aiohttp.ClientSession() as s:
        await one(s,"warm")
        for trial in range(3):
            p=f"trial{trial}_{TAG} {pad} end"
            cold=await one(s,p)
            if GAP: await asyncio.sleep(GAP)
            h1=await one(s,p); h2=await one(s,p); h3=await one(s,p)
            out.setdefault('cold',[]).append(cold*1000)
            out.setdefault('h1',[]).append(h1*1000)
            out.setdefault('h2',[]).append(h2*1000)
            out.setdefault('h3',[]).append(h3*1000)
    m=lambda k: round(statistics.median(out[k]),1)
    c=m('cold')
    print(json.dumps({"tag":TAG,"gap":GAP,"cold_ms":c,"hot1_ms":m('h1'),"hot2_ms":m('h2'),
                      "hot3_ms":m('h3'),
                      "saved1_pct":round(100*(1-m('h1')/c),1),
                      "saved3_pct":round(100*(1-m('h3')/c),1)}))
asyncio.run(main())
