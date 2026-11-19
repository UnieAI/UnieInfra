"""Measure per-user decode speed with N concurrent streaming requests.

Each request has a unique prompt of about --tokens tokens and decodes
--max-tokens tokens with ignore_eos. Decode speed per user is
(completion_tokens - 1) / (t_last_chunk - t_first_token); DSpark delivers
tokens in bursts, so this averages over the whole decode.

Usage: python3 bench_decode.py --n 16 --tokens 2000 --max-tokens 512
"""

import argparse
import concurrent.futures as cf
import json
import random
import statistics
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey"
).split()


def make_prompt(seed: int, tokens: int) -> str:
    rng = random.Random(seed)
    words = int(tokens / 1.3)
    lines = [
        f"{seed}-{i}: " + " ".join(rng.choice(WORDS) for _ in range(20))
        for i in range(max(1, words // 20))
    ]
    return "\n".join(lines) + "\nWrite a very long story about these words."


def one(seed: int, tokens: int, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": make_prompt(seed, tokens)}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("choices"):
                now = time.time()
                first = first or now
                last = now
    n_out = usage["completion_tokens"] if usage else 0
    return {
        "ttft": first - t0,
        "decode_tps": (n_out - 1) / (last - first) if last > first else 0.0,
        "out": n_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    seed = int(time.time())
    with cf.ThreadPoolExecutor(args.n) as ex:
        res = list(ex.map(lambda k: one(seed + k, args.tokens, args.max_tokens), range(args.n)))
    tps = [r["decode_tps"] for r in res]
    print(
        f"n={args.n} prompt~{args.tokens} out={res[0]['out']} "
        f"decode tok/s/user median={statistics.median(tps):.1f} "
        f"min={min(tps):.1f} max={max(tps):.1f} "
        f"(~{1000 / statistics.median(tps):.2f} ms/token) "
        f"ttft median={statistics.median(r['ttft'] for r in res):.1f}s"
    )


if __name__ == "__main__":
    main()
