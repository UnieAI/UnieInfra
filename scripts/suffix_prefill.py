"""TTFT of a short new suffix on a long cached prefix (the agentic turn shape).

Caches a unique ~CTX-token document, then for each suffix size sends
document + a fresh suffix of that many tokens (3 times) and reports the
median TTFT, with the server otherwise idle.
Usage: python3 suffix_prefill.py [--ctx 100000] [--sizes 256,1024,2048,4096,8192]
"""
import argparse, json, random, statistics, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
WORDS = "river stone cloud lantern harbor meadow copper violet signal orchard canyon ember".split()


def text(rng, tokens):
    return "\n".join(" ".join(rng.choice(WORDS) for _ in range(16)) + "." for _ in range(max(1, tokens // 18)))


def ttft(content):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 4,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    first = time.perf_counter() - t0
    return first, usage.get("prompt_tokens"), (usage.get("prompt_tokens_details") or {}).get("cached_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=100000)
    ap.add_argument("--sizes", default="256,1024,2048,4096,8192")
    ap.add_argument("--profile-dir", default="", help="GPU-profile one extra suffix of the last size into this dir")
    a = ap.parse_args()
    rng = random.Random(int(time.time()))
    doc = text(rng, a.ctx)
    t, p, c = ttft(doc + "\n\nSay OK.")
    print(f"cold ctx {p} tok: ttft {t:.2f}s", flush=True)
    for size in map(int, a.sizes.split(",")):
        res = [ttft(doc + "\n\n" + text(rng, size) + "\nSay OK.") for _ in range(3)]
        print(f"suffix ~{size}: ttft median {statistics.median(r[0] for r in res) * 1000:.0f} ms "
              f"(prompt {res[0][1]}, cached {res[0][2]})", flush=True)
    if a.profile_dir:
        post = lambda path, body: urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8000/" + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600).read()
        post("start_profile", {"output_dir": a.profile_dir, "activities": ["GPU"]})
        r = ttft(doc + "\n\n" + text(rng, size) + "\nSay OK.")
        post("stop_profile", {})
        print(f"profiled suffix ~{size}: ttft {r[0] * 1000:.0f} ms (prompt {r[1]}, cached {r[2]})", flush=True)


if __name__ == "__main__":
    main()
