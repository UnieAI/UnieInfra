#!/usr/bin/env python3
"""Measure embedding-model request throughput/latency over /v1/embeddings.

Usage: emb_bench.py BASE MODEL DATA CONC N OUT
DATA = ndjson with {"prompt": ...} lines; conc is the concurrency, N the count.
Reports req/s, p50/p90 per-request latency, and total tokens.
"""
import asyncio, json, statistics, sys, time
import aiohttp

base, model, data, conc, n, out = sys.argv[1:7]
conc, n = int(conc), int(n)
prompts = [json.loads(l)["prompt"] for l in open(data)][:n]


async def one(sess, sem, i, p, res):
    async with sem:
        t0 = time.perf_counter()
        async with sess.post(
            f"{base}/v1/embeddings",
            json={"model": model, "input": p},
            timeout=aiohttp.ClientTimeout(total=1800),
        ) as r:
            j = json.loads(await r.read())
            res[i] = {"err": None if r.status == 200 else r.status,
                      "lat": round((time.perf_counter() - t0) * 1000),
                      "ndim": len(j["data"][0]["embedding"]) if r.status == 200 else 0}


async def main():
    sem = asyncio.Semaphore(conc)
    res = {}
    async with aiohttp.ClientSession() as s:
        t0 = time.perf_counter()
        await asyncio.gather(*[one(s, sem, i, p, res) for i, p in enumerate(prompts)])
        dur = time.perf_counter() - t0
    lat = sorted(r["lat"] for r in res.values() if r["err"] is None)
    ok = len(lat)
    json.dump({"conc": conc, "duration_s": round(dur, 3), "n": ok,
               "rows": [res[i] for i in sorted(res)]}, open(out, "w"))
    print(f"{out}: conc={conc} n={ok}/{n} dur={dur:.3f}s req/s={round(ok/dur,1)} "
          f"lat_p50={statistics.median(lat):.0f}ms p90={lat[int(len(lat)*.9)]:.0f}ms")


asyncio.run(main())