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
import os
import sys
import time

import aiohttp

base, model, data, conc, nreq, out = sys.argv[1:7]
conc, nreq = int(conc), int(nreq)
# Which endpoint carries the work. This is not cosmetic: `chat` makes each
# engine render our single user turn into its own prompt format, and the
# engines do not agree on that format (sglang/vLLM render DeepSeek-V4.1 through
# a Python encoder; a gateway needs a chat template the checkpoint does not
# ship). Two engines that prefill different token counts cannot be compared on
# TTFT, so the comparable arm sends the corpus text verbatim to
# `/v1/completions`: same prompt, same tokens, same work by construction.
# Required, because defaulting it would silently pick one of those two
# measurements.
ENDPOINT = os.environ["ENDPOINT"]
if ENDPOINT not in ("chat", "completions"):
    raise SystemExit(f"ENDPOINT must be chat or completions, got {ENDPOINT!r}")
# Fallback token counter for servers that do not send a usage chunk. Optional:
# `usage.completion_tokens` is authoritative when present. Point TOKENIZER at
# the checkpoint so counts are the model's own tokens, not chunks of text.
TOKENIZER = None
if os.environ.get("TOKENIZER"):
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained(
        os.environ["TOKENIZER"], trust_remote_code=True
    )
# Some engines terminate on EOS/stop tokens that `ignore_eos` does not cover
# (sglang's chat template contributes its own stop_token_ids, which
# SamplingParams.ignore_eos does not suppress). A cell that stops at token 41
# while another runs to 128 is not doing the same work, so its throughput and
# its prefill amortisation are not comparable. Setting MIN_TOKENS forces a floor
# on the generated length, which makes the arms equal-work by construction.
# Env-gated so the default protocol is unchanged.
MIN_TOKENS = os.environ.get("MIN_TOKENS") or None
prompts = [json.loads(l)["prompt"] for l in open(data)][:nreq]
# Engine warmup on a DIFFERENT corpus. Repeating the measured corpus warms the
# prefix cache with the exact prompts about to be timed, which is replay, not
# warmup: hit rate climbs to 95% and the "warm" number stops describing serving.
# This warms autotune / graph capture / first-touch allocation only.
warm_src = sys.argv[7] if len(sys.argv) > 7 else None
warm_prompts = (
    [json.loads(l)["prompt"] for l in open(warm_src)][: 2 * conc] if warm_src else []
)


def url(path: str) -> str:
    return f"{base}/v1/{path}"


def request_body(p: str, max_tokens: int) -> dict:
    """Request body for the selected endpoint, at a fixed generated length."""
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        # Ask for the authoritative generated length (see the usage handling in
        # `one()`); engines that ignore the field simply omit the usage chunk.
        "stream_options": {"include_usage": True},
    }
    if ENDPOINT == "chat":
        body["messages"] = [{"role": "user", "content": p}]
    else:
        body["prompt"] = p
    # Match `vllm bench serve --ignore-eos`: without it a reply that stops early
    # generates fewer tokens, which shortens the run and inflates req/min
    # against the official tool's fixed 128. Measured 33.4 vs 26.4 rpm on the
    # same cell before this was matched.
    body["ignore_eos"] = True
    if MIN_TOKENS:
        body["min_tokens"] = int(MIN_TOKENS)
    return body


def emitted(j: dict) -> str | None:
    """The text this stream item added, whichever endpoint shapes it."""
    choice = (j.get("choices") or [{}])[0]
    if ENDPOINT == "chat":
        delta = choice.get("delta") or {}
        # Reasoning model: output is streamed in `reasoning_content` until the
        # final answer, which lands in `content`. Counting only `content`
        # records ~80% of DeepSeek-V4 requests as empty and zeroes their
        # throughput. Count and time the whole stream.
        return delta.get("content") or delta.get("reasoning_content")
    return choice.get("text")


async def one(sess, sem, i, p, res):
    async with sem:
        t0 = time.perf_counter()
        # STREAM, so first-token time is separable from the rest. A non-streaming
        # request only yields end-to-end, which folds TTFT and TPOT into one
        # number -- and those two axes fail for different reasons and have
        # different fixes, so a combined number cannot drive either.
        ttft = None
        chunks = []
        reported = None
        err = None
        async with sess.post(
            url("chat/completions" if ENDPOINT == "chat" else "completions"),
            json=request_body(p, 128),
            timeout=aiohttp.ClientTimeout(total=1800),
        ) as r:
            status = r.status
            if status != 200:
                # Never swallow an HTTP rejection: a non-200 body is not an SSE
                # stream, so the data: loop below would silently record empty text.
                # Capture the status + error body so the caller can see it.
                try:
                    body = (await r.read()).decode("utf-8", "replace")[:300]
                except Exception:
                    body = "<unreadable>"
                err = f"HTTP {status}: {body}"
                res[i] = {
                    "i": i,
                    "ms": round((time.perf_counter() - t0) * 1000),
                    "ttft": None,
                    "ntok": 0,
                    "tpot": None,
                    "text": "",
                    "err": err,
                }
                return
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
                # Servers that support stream_options include a final chunk
                # whose usage.completion_tokens is the generated length. That
                # number, not the chunk count, is what throughput is made of:
                # one SSE chunk carries one token on some servers and several
                # on others (measured 122 vs 57 chunks for the same 128-token
                # request on two engines), so counting chunks understates a
                # batching engine by up to 2x.
                usage = j.get("usage") or {}
                if usage.get("completion_tokens"):
                    reported = usage["completion_tokens"]
                tok = emitted(j)
                if tok:
                    if ttft is None:
                        ttft = round((time.perf_counter() - t0) * 1000)
                    chunks.append(tok)
        e2e = round((time.perf_counter() - t0) * 1000)
        text = "".join(chunks)
        # Without a usage chunk, count real tokens from the model's tokenizer.
        ntok = reported or (len(TOKENIZER.encode(text)) if TOKENIZER else len(chunks))
        res[i] = {
            "i": i,
            "ms": e2e,
            "ttft": ttft,
            "ntok": ntok,
            "tpot": (
                round((e2e - ttft) / max(1, ntok - 1)) if ttft and ntok > 1 else None
            ),
            "nchunk": len(chunks),
            "text": text,
            "err": err,
        }


async def warmup(sess):
    if not warm_prompts:
        return

    async def one_warm(p):
        async with sess.post(
            url("chat/completions" if ENDPOINT == "chat" else "completions"),
            json={**request_body(p, 16), "stream": False},
            timeout=aiohttp.ClientTimeout(total=1800),
        ) as r:
            await r.read()

    await asyncio.gather(*[one_warm(p) for p in warm_prompts])


async def main():
    sem = asyncio.Semaphore(conc)
    res = {}
    async with aiohttp.ClientSession() as s:
        await warmup(s)
        started_at = time.time()
        t0 = time.perf_counter()
        await asyncio.gather(*[one(s, sem, i, p, res) for i, p in enumerate(prompts)])
        dur = time.perf_counter() - t0
        finished_at = time.time()
    rows = [res[i] for i in sorted(res)]
    json.dump(
        {
            "duration_s": round(dur, 2),
            "started_at": round(started_at, 3),
            "finished_at": round(finished_at, 3),
            "n": len(rows),
            "rows": rows,
        },
        open(out, "w"),
    )
    ok = sum(1 for r in rows if r["text"])
    print(f"{out}: n={len(rows)} ok={ok} duration={dur:.1f}s")


asyncio.run(main())
