"""DSpark acceptance on long outputs: 8 prompts = distinct repo docs (~20k tokens) + a
summary request, max_tokens 256, greedy. Usage: l3_accept.py <phase> <out.jsonl>"""
import glob, json, sys, time, urllib.request
phase, out = sys.argv[1], sys.argv[2]
docs = sorted(glob.glob("/home/smc/workspace/roy/unieinfra/.wt-l3/docs/**/*.md", recursive=True))
text = "\n\n".join(open(p).read() for p in docs)
URL = "http://127.0.0.1:8000/v1/completions"; MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
with open(out, "a") as f:
    for i in range(8):
        chunk = text[i * 60000:(i + 1) * 60000]
        prompt = f"Document {i}:\n{chunk}\n\nWrite a detailed technical summary of the document above:\n"
        body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 256, "temperature": 0, "ignore_eos": True}).encode()
        t0 = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600))
        rec = {"phase": phase, "i": i, "e2e_s": round(time.time() - t0, 3), "prompt_tokens": r["usage"]["prompt_tokens"], "text": r["choices"][0]["text"]}
        f.write(json.dumps(rec) + "\n"); f.flush()
