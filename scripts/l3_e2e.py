"""L3 e2e: send N long unique prompts (seeded), record TTFT and greedy output.
Usage: l3_e2e.py <phase> <out.jsonl> [N] [ISL_words]"""
import json, random, sys, time, urllib.request
phase, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
W = int(sys.argv[4]) if len(sys.argv) > 4 else 24000
URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
words = "alpha beta gamma delta river stone cloud metal paper green orange silver window garden market engine signal winter summer bridge".split()
with open(out, "a") as f:
    for i in range(N):
        rng = random.Random(1000 + i)
        if phase.startswith("needle"):
            code = "%06d" % rng.randrange(10**6)
            filler = [rng.choice(words) for _ in range(W)]
            filler.insert(rng.randrange(W), f". The secret code is {code}. ")
            prompt = f"Needle doc {i}: " + " ".join(filler) + "\nQuestion: what is the secret code? Answer with the six digits only.\nAnswer:"
        else:
            code = None
            prompt = f"Doc {i}: " + " ".join(rng.choice(words) for _ in range(W)) + "\nSummarize in one line:"
        body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 16, "temperature": 0, "stream": True}).encode()
        req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
        t0 = time.time(); ttft = None; text = ""; usage = None
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                d = json.loads(line[5:])
                for c in d.get("choices", []):
                    if c.get("text") and ttft is None: ttft = time.time() - t0
                    text += c.get("text", "")
        rec = {"phase": phase, "i": i, "ttft_s": round(ttft or -1, 3), "e2e_s": round(time.time() - t0, 3), "text": text, "code": code, "hit": (code in text) if code else None}
        f.write(json.dumps(rec) + "\n"); f.flush(); print(rec, flush=True)
