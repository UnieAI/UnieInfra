"""How much of a multi-turn conversation does the prefix cache reuse?

Turn 1: unique ~TOKENS-token document + a request for a long answer (reasoning
on, up to OUT tokens). Turn 2: the same messages + the assistant's turn-1 reply
(content and reasoning as the server returned them) + a short new question.
Ideal reuse for turn 2 is about prompt_1 + reply tokens. Prints prompt tokens,
cached tokens and TTFT for each turn.
Usage: python3 multiturn_cache.py [--cases 3] [--tokens 60000] [--out 1500]
"""
import argparse, json, random, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
WORDS = "river stone cloud lantern harbor meadow copper violet signal orchard canyon ember".split()


def chat(messages, max_tokens):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; content = ""; reasoning = ""; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if (d.get("content") or d.get("reasoning_content") or d.get("reasoning")) and ttft is None:
                    ttft = time.perf_counter() - t0
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or d.get("reasoning") or ""
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return content, reasoning, usage.get("prompt_tokens"), usage.get("completion_tokens"), cached, ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=60000)
    ap.add_argument("--out", type=int, default=1500)
    a = ap.parse_args()
    for i in range(a.cases):
        rng = random.Random(int(time.time() * 1000) + i)
        doc = "\n".join(" ".join(rng.choice(WORDS) for _ in range(16)) + "." for _ in range(a.tokens // 18))
        m1 = [{"role": "user", "content": doc + "\n\nWrite a detailed 800-word story using the words above."}]
        c1, r1, p1, o1, k1, t1 = chat(m1, a.out)
        m2 = m1 + [{"role": "assistant", "content": c1, "reasoning_content": r1},
                   {"role": "user", "content": "Now summarize your story in one sentence."}]
        c2, r2, p2, o2, k2, t2 = chat(m2, 200)
        print(f"case {i}: turn1 prompt {p1} out {o1} cached {k1} ttft {t1:.2f}s | turn2 prompt {p2} cached {k2} "
              f"= {(k2 or 0) / max(1, p2):.1%} of prompt, {(k2 or 0) / max(1, p1 + (o1 or 0)):.1%} of turn1 prompt+reply, "
              f"new {p2 - (k2 or 0)} | ttft {t2:.2f}s", flush=True)


if __name__ == "__main__":
    main()
