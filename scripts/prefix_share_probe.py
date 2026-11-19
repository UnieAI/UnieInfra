"""Probe cross-request prefix sharing on a live server.

Sends chat requests that share a long token-heavy prefix and prints the server's
cached-token count for each, read from the server log by the caller.
Usage: prefix_share_probe.py PORT
"""
import json, sys, time, threading, urllib.request

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/chat/completions"
model = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
# ~30K tokens of deterministic, tokenizer-friendly filler that differs from other runs.
shared = " ".join(f"item{i} alpha{i % 97} beta{i % 31} gamma{i % 13}." for i in range(5200))

def ask(tag, suffix):
    body = {"model": model, "max_tokens": 1, "temperature": 0,
            "messages": [{"role": "system", "content": shared},
                         {"role": "user", "content": f"{suffix}"}]}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    print(f"MARK {tag} prompt_tokens={r['usage']['prompt_tokens']} t={time.time()-t:.2f}s", flush=True)

ask("A_first", "Question one: say hi.")
time.sleep(2)
ask("B_sequential_same_prefix", "Question two: say bye.")
time.sleep(2)
ts = [threading.Thread(target=ask, args=(f"{n}_concurrent", f"Question {n}: say ok.")) for n in ("C", "D", "E")]
[t.start() for t in ts]; [t.join() for t in ts]
