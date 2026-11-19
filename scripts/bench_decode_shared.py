"""Long-context decode speed without long prefills.

All N requests share one ~TOKENS-token document (prefilled once by a warm-up
request, then served from the prefix cache) plus a unique short question, and
decode MAX tokens with ignore_eos. Per-user decode speed is
(completion_tokens - 1) / (t_last - t_first).
Usage: python3 bench_decode_shared.py --n 64 --tokens 130000 --max-tokens 512
"""
import argparse, concurrent.futures as cf, json, random, statistics, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
WORDS = "river stone cloud lantern harbor meadow copper violet signal orchard canyon ember".split()


def run(doc, q, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": doc + "\n\n" + q}],
                       "max_tokens": max_tokens, "ignore_eos": True, "min_tokens": max_tokens,
                       "temperature": TEMP, "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
    n = usage.get("completion_tokens", 0)
    return (n - 1) / max(1e-9, last - first), first - t0, usage.get("prompt_tokens_details", {})


TEMP = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=130000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=None,
                    help="document seed; a fixed seed reuses the same document across runs")
    ap.add_argument("--warm", type=int, default=1,
                    help="concurrent warm-up requests; with data-parallel ranks use at least the DP size "
                         "so every rank caches the document")
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    global TEMP
    TEMP = a.temperature
    rng = random.Random(int(time.time()) if a.seed is None else a.seed)
    doc = "\n".join(" ".join(rng.choice(WORDS) for _ in range(16)) + "." for _ in range(a.tokens // 18))
    with cf.ThreadPoolExecutor(a.warm) as ex:  # cache the shared document
        list(ex.map(lambda i: run(doc, f"Say OK {i}.", 8), range(a.warm)))
    rng = random.Random(int(time.time()))
    qs = [f"Question {i}: write a long story about item {rng.randint(0, 10**9)}." for i in range(a.n)]
    t = time.perf_counter()
    with cf.ThreadPoolExecutor(a.n) as ex:
        res = list(ex.map(lambda q: run(doc, q, a.max_tokens), qs))
    speeds = sorted(r[0] for r in res)
    total = a.n * a.max_tokens / (time.perf_counter() - t)
    print(f"n={a.n} ctx~{a.tokens} decode tok/s/user median={statistics.median(speeds):.1f} "
          f"(~{1000 / statistics.median(speeds):.1f} ms/token) min={speeds[0]:.1f} max={speeds[-1]:.1f} "
          f"ttft median={statistics.median(r[1] for r in res):.1f}s total out tok/s={total:.0f}", flush=True)


if __name__ == "__main__":
    main()
