"""Cold long prefill beside running long-context decodes.

Starts D decoders (unique ~CTX-token prompts, max_tokens DEC_TOK), waits WAIT s so they
are decoding, then sends one unique cold prompt of ~COLD tokens with max_tokens 1 and
reports its latency (= prefill time beside decodes). Compare with the isolated cold time.
Usage: prefill_interfere.py --decoders 6 --ctx 150000 --dec-tokens 4000 --cold 450000 --wait 40
"""
import argparse, concurrent.futures as cf, json, sys, time, urllib.request
sys.path.insert(0, "/home/smc/workspace/roy/scripts")
from repro_longprefill import make_prompt, URL, MODEL

def req(seed, tokens, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": make_prompt(seed, tokens)}],
                       "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1800) as r:
        out = json.load(r)
    return time.time() - t0, out["usage"]

ap = argparse.ArgumentParser()
ap.add_argument("--decoders", type=int, default=6); ap.add_argument("--ctx", type=int, default=150000)
ap.add_argument("--dec-tokens", type=int, default=4000); ap.add_argument("--cold", type=int, default=450000)
ap.add_argument("--wait", type=float, default=40)
a = ap.parse_args(); seed = int(time.time())
with cf.ThreadPoolExecutor(a.decoders + 1) as ex:
    decs = [ex.submit(req, seed + k, a.ctx, a.dec_tokens) for k in range(a.decoders)]
    time.sleep(a.wait)
    t, u = ex.submit(req, seed + 999, a.cold, 1).result()
    alive = sum(not f.done() for f in decs)
    print(f"cold {u['prompt_tokens']} tok latency {t:.1f}s ({u['prompt_tokens']/t:.0f} tok/s) beside {alive}/{a.decoders} decoders still running", flush=True)
    for f in decs:
        dt, du = f.result(); print(f"decoder {du['prompt_tokens']} in / {du['completion_tokens']} out {dt:.1f}s", flush=True)
