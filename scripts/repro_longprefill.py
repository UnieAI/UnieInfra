"""Reproduce the unieinfra multi-request long-prefill crash quickly.

Sends N concurrent chat requests, each with a unique ~TOKENS-token prompt
(unique so prefix caching cannot shortcut the prefill), for ROUNDS rounds.
Prints per-request latency; a dropped connection or 5xx means the engine died.

Usage: python3 repro_longprefill.py [--n 3] [--tokens 170000] [--rounds 3]
"""

import argparse
import concurrent.futures as cf
import json
import random
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu compile kernel tensor matrix vector scheduler cache page"
).split()


def make_prompt(seed: int, tokens: int) -> str:
    rng = random.Random(seed)
    # ~1.3 tokens per short English word on this tokenizer; a numeric tag per
    # line keeps every prompt unique from its first tokens.
    lines = []
    words = int(tokens / 1.3)
    per_line = 20
    for i in range(words // per_line):
        lines.append(f"{seed}-{i}: " + " ".join(rng.choice(WORDS) for _ in range(per_line)))
    return "\n".join(lines) + "\nSummarize the above in one word."


def one(seed: int, tokens: int) -> str:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": make_prompt(seed, tokens)}],
            "max_tokens": 16,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.load(r)
        return f"seed={seed} ok {time.time() - t0:.1f}s prompt_tokens={out['usage']['prompt_tokens']}"
    except Exception as e:  # noqa: BLE001 - report any failure mode
        return f"seed={seed} FAIL {time.time() - t0:.1f}s {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=170000)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    seed = int(time.time())
    for rnd in range(args.rounds):
        with cf.ThreadPoolExecutor(args.n) as ex:
            futs = [ex.submit(one, seed + rnd * 100 + k, args.tokens) for k in range(args.n)]
            for f in cf.as_completed(futs):
                print(f"round {rnd}: {f.result()}", flush=True)


if __name__ == "__main__":
    main()
