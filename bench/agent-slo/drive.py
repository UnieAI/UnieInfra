#!/usr/bin/env python3
"""Concurrent driver that SAVES THE GENERATED TEXT.

`vllm bench serve` reports latency only, and roy-ad measured that gemma-4 on this
engine corrupts output when the scheduler fuses a decode row into a prefill
forward -- 4 bad answers in 16 with mixed batching on, 0 with it off, while the
latency numbers look healthy either way. So a run that only records latency
cannot tell a correct engine from a broken one. Greedy (temperature 0) so two
engines must agree token for token.
"""

import asyncio
import json
import sys
import time

import aiohttp

base, model, data, conc, nreq, out = sys.argv[1:7]
conc, nreq = int(conc), int(nreq)
prompts = [json.loads(l)["prompt"] for l in open(data)][:nreq]
# Engine warmup on a DIFFERENT corpus. Repeating the measured corpus warms the
# prefix cache with the exact prompts about to be timed, which is replay, not
# warmup: hit rate climbs to 95% and the "warm" number stops describing serving.
# This warms autotune / graph capture / first-touch allocation only.
warm_src = sys.argv[7] if len(sys.argv) > 7 else None
warm_prompts = (
    [json.loads(l)["prompt"] for l in open(warm_src)][: 2 * conc] if warm_src else []
)


async def one(sess, sem, i, p, res):
    async with sem:
        t0 = time.perf_counter()
        # STREAM, so first-token time is separable from the rest. A non-streaming
        # request only yields end-to-end, which folds TTFT and TPOT into one
        # number -- and those two axes fail for different reasons and have
        # different fixes, so a combined number cannot drive either.
        ttft = None
        chunks = []
        err = None
        async with sess.post(
            f"{base}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": p}],
                "max_tokens": 128,
                "temperature": 0,
                "stream": True,
                # Match `vllm bench serve --ignore-eos`: without it a reply that
                # stops early generates fewer tokens, which shortens the run and
                # inflates req/min against the official tool's fixed 128. Measured
                # 33.4 vs 26.4 rpm on the same cell before this was matched.
                "ignore_eos": True,
            },
            timeout=aiohttp.ClientTimeout(total=1800),
        ) as r:
            async for raw in r.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except Exception:
                    continue
                if "error" in j:
                    err = j["error"]
                    break
                delta = (j.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = round((time.perf_counter() - t0) * 1000)
                    chunks.append(delta)
        e2e = round((time.perf_counter() - t0) * 1000)
        ntok = len(chunks)
        res[i] = {
            "i": i,
            "ms": e2e,
            "ttft": ttft,
            "ntok": ntok,
            "tpot": (
                round((e2e - ttft) / max(1, ntok - 1)) if ttft and ntok > 1 else None
            ),
            "text": "".join(chunks),
            "err": err,
        }


async def warmup(sess):
    if not warm_prompts:
        return

    async def one_warm(p):
        async with sess.post(
            f"{base}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": p}],
                "max_tokens": 16,
                "temperature": 0,
                "stream": False,
            },
            timeout=aiohttp.ClientTimeout(total=1800),
        ) as r:
            await r.read()

    await asyncio.gather(*[one_warm(p) for p in warm_prompts])


async def main():
    sem = asyncio.Semaphore(conc)
    res = {}
    async with aiohttp.ClientSession() as s:
        await warmup(s)
        t0 = time.perf_counter()
        await asyncio.gather(*[one(s, sem, i, p, res) for i, p in enumerate(prompts)])
        dur = time.perf_counter() - t0
    rows = [res[i] for i in sorted(res)]
    json.dump(
        {"duration_s": round(dur, 2), "n": len(rows), "rows": rows}, open(out, "w")
    )
    ok = sum(1 for r in rows if r["text"])
    print(f"{out}: n={len(rows)} ok={ok} duration={dur:.1f}s")


asyncio.run(main())
