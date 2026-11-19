"""Keep long cold prefills flowing (max_tokens 1) until killed, so concurrent decodes share mixed rounds."""
import argparse, itertools, json, time, urllib.request
from repro_longprefill import make_prompt, URL, MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=40000)
ap.add_argument("--gap", type=float, default=0.5)
a = ap.parse_args()
for i in itertools.count():
    body = json.dumps({"model": MODEL, "max_tokens": 1, "messages": [{"role": "user", "content": make_prompt(10_000 + i, a.tokens)}]}).encode()
    t = time.time()
    try:
        urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600).read()
        print(f"{i} {time.time() - t:.1f}s", flush=True)
    except Exception as e:
        print(f"{i} ERR {e}", flush=True)
    time.sleep(a.gap)
