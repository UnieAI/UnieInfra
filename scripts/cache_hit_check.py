"""Is a prefix-cache hit real? Cold vs warm TTFT, reported cached tokens, and answers.

For each case: build a unique ~TOKENS-token document with a hidden code.
  1. cold: [doc + "Say OK."]                  -> TTFT_cold, cached_tokens (expect ~0)
  2. warm: [doc + "Say OK.", "OK", question]  -> TTFT_warm, cached_tokens (expect ~doc)
     and the answer must contain the code (the cached KV is actually right).
  3. repeat 1 exactly at temperature 0         -> same first tokens as 1 (cache vs fresh)
Usage: python3 cache_hit_check.py [--cases 3] [--tokens 100000]
"""
import argparse, json, random, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
WORDS = "river stone cloud lantern harbor meadow copper violet signal orchard canyon ember".split()


def stream(messages, max_tokens):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; text = ""; usage = None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
                text += d.get("content") or ""
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return ttft, text, (usage or {}).get("prompt_tokens"), cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=100000)
    a = ap.parse_args()
    ok_all = True
    for i in range(a.cases):
        rng = random.Random(7000 + i + int(time.time()))
        code = f"{rng.randint(100000, 999999)}-{rng.choice(WORDS).upper()}"
        lines = [" ".join(rng.choice(WORDS) for _ in range(16)) + "." for _ in range(a.tokens // 18)]
        lines.insert(len(lines) // 3, f"IMPORTANT: the vault access code is {code}. Remember it.")
        doc = "\n".join(lines)
        m1 = [{"role": "user", "content": doc + "\n\nRead the text above. Reply with OK only."}]
        t_cold, out1, p1, c1 = stream(m1, 64)
        m2 = m1 + [{"role": "assistant", "content": out1 or "OK"},
                   {"role": "user", "content": "What is the vault access code in the text? Reply with the code only."}]
        t_warm, out2, p2, c2 = stream(m2, 2048)
        t_rep, out3, p3, c3 = stream(m1, 64)
        correct = code in out2
        same = out1[:40] == out3[:40]
        ok_all &= correct
        print(f"case {i}: prompt {p1} tok | cold TTFT {t_cold:.2f}s cached={c1} | "
              f"warm TTFT {t_warm:.2f}s prompt {p2} cached={c2} ({(c2 or 0) / max(1, p2):.0%}) answer_ok={correct} | "
              f"repeat TTFT {t_rep:.2f}s cached={c3} same_output={same}", flush=True)
    print("CACHE CHECK", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
