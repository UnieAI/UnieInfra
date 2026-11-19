"""Capture a torch trace of steady-state decode at a chosen batch and context.

Warm round primes N prompts of ~TOKENS tokens so the long context sits in the
prefix cache; the profiled round re-sends the *same* prompts, so its prefill is
a cache hit and the trace is pure decode at batch N.

Usage: decode_profile.py [--n 16] [--tokens 170000] [--prof-tokens 400]
                         [--out /logs/prof-dec-c64]
"""

import argparse
import concurrent.futures as cf
import json
import random
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu compile kernel tensor matrix vector scheduler cache page"
).split()


def make_prompt(seed: int, tokens: int) -> str:
    # Token/word measured with the model's own tokenizer on this generator:
    # 1.5 tokens per word plus ~4.5 per line prefix ("<seed>-<i>: ").
    rng = random.Random(seed)
    words = int(tokens / (1.5 + 4.5 / 20))
    lines = []
    per_line = 20
    for i in range(words // per_line):
        lines.append(f"{seed}-{i}: " + " ".join(rng.choice(WORDS) for _ in range(per_line)))
    return "\n".join(lines) + "\nSummarize the above in one word."


def one(prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    usage = out["usage"]
    details = usage.get("prompt_tokens_details") or {}
    return {
        "lat": time.time() - t0,
        "completion": usage["completion_tokens"],
        "prompt": usage["prompt_tokens"],
        "cached": details.get("cached_tokens", 0),
    }


def post(path: str, payload: dict | None = None) -> str:
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(
        BASE + path, data, {"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read().decode(errors="replace")[:400]
    except urllib.error.HTTPError as exc:
        # The engine rejects a profile request with the reason in the body;
        # a bare traceback hides it.
        raise SystemExit(
            f"{path} -> HTTP {exc.code}: "
            f"{exc.read().decode(errors='replace')[:400]}"
        ) from None


def round_of(prompts: list[str], max_tokens: int) -> list[dict]:
    with cf.ThreadPoolExecutor(len(prompts)) as ex:
        futs = [ex.submit(one, p, max_tokens) for p in prompts]
        return [f.result() for f in futs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=170000)
    ap.add_argument("--warm-tokens", type=int, default=8)
    ap.add_argument("--prof-tokens", type=int, default=400)
    ap.add_argument("--out", default="/logs/prof-dec-c64")
    args = ap.parse_args()

    seed = 700000
    prompts = [make_prompt(seed + k, args.tokens) for k in range(args.n)]
    print(f"warming {args.n} prompts of ~{args.tokens} tokens", flush=True)
    warm = round_of(prompts, args.warm_tokens)
    print(
        f"warm done: max {max(w['lat'] for w in warm):.1f}s "
        f"prompt_tokens {min(w['prompt'] for w in warm)}..{max(w['prompt'] for w in warm)}",
        flush=True,
    )

    print("start_profile:", post("/start_profile", {"output_dir": args.out, "activities": ["GPU"]}), flush=True)
    t0 = time.time()
    prof = round_of(prompts, args.prof_tokens)
    span = time.time() - t0
    print("stop_profile:", post("/stop_profile"), flush=True)

    lat = sorted(w["lat"] for w in prof)
    tok = sorted(w["completion"] for w in prof)
    hit = sorted(w["cached"] for w in prof)
    print(
        f"profiled {len(prof)} reqs: span {span:.1f}s latency p50 {lat[len(lat) // 2]:.1f}s "
        f"max {lat[-1]:.1f}s completion p50 {tok[len(tok) // 2]} total {sum(tok)} "
        f"cached p50 {hit[len(hit) // 2]}",
        flush=True,
    )
    print(f"trace dir {args.out}", flush=True)


if __name__ == "__main__":
    main()
