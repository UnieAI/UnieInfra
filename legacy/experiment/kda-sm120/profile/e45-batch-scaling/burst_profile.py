#!/usr/bin/env python3
"""Pure-decode profiling at a controlled batch size.

E36's method, re-run at a chosen batch. Fire B identical requests at once with
min_tokens == max_tokens so nothing finishes early, wait until the engine's own
`Decode batch. #running-req:` line reports B for several consecutive samples
(i.e. every prefill is done and the window is pure decode), then drive
/start_profile with num_steps so the profiler stops itself after exactly N
decode steps.

Nothing here trusts a client-side token count (E23/E44): batch size comes from
the engine log, step time comes from the trace, and the engine's own
`gen throughput (token/s)` is kept as an independent cross-check on the trace's
step wall (bs / gen_tok_s == step seconds).
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8411")
MODEL = os.environ.get("MODEL_NAME", "bench-model")
CONTAINER = os.environ.get("CONTAINER", "prof128-b")
BATCH = int(os.environ.get("BATCH", "126"))
ILEN = int(os.environ.get("ILEN", "1024"))
OLEN = int(os.environ.get("OLEN", "560"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "40"))
TAG = os.environ.get("TAG", f"b{BATCH}")
OUTDIR = os.environ.get("OUTDIR", "/prof/" + TAG)
HOSTDIR = os.environ.get(
    "HOSTDIR", "/workspace/roy/UnieInfra/experiment/kda-sm120/profile/"
    "e45-batch-scaling/" + TAG)
STABLE_SAMPLES = int(os.environ.get("STABLE_SAMPLES", "2"))
WAIT_S = float(os.environ.get("WAIT_S", "600"))
# extra decode-log lines to let elapse after the batch stabilises, so the mean
# context length at profile time can be matched across arms (each line = 40 steps)
EXTRA_LINES = int(os.environ.get("EXTRA_LINES", "0"))

_fail = [0]
_ok = [0]
_ptok = []


def post(path, payload, timeout=1800):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ntok(text):
    r = post("/v1/completions", {"model": MODEL, "prompt": text,
                                 "max_tokens": 1, "temperature": 0.0})
    return int(r["usage"]["prompt_tokens"])


def calibrate():
    """Match ILEN on the sample, the way steady_bench does."""
    n = int(ILEN * 0.75)
    for _ in range(8):
        p = "benchmark " * n
        got = ntok("[cal] " + p)
        if abs(got - ILEN) <= 8:
            return p, got
        n = max(1, int(n * ILEN / max(got, 1)))
    return "benchmark " * n, got


def worker(i, prompt):
    try:
        r = post("/v1/completions", {
            "model": MODEL, "prompt": f"[prof{TAG}w{i}] " + prompt,
            "max_tokens": OLEN, "min_tokens": OLEN,
            "temperature": 0.0, "ignore_eos": True, "stream": False,
        })
        _ok[0] += 1
        _ptok.append(r["usage"]["prompt_tokens"])
    except Exception as e:  # noqa: BLE001
        _fail[0] += 1
        sys.stderr.write(f"w{i} failed: {e}\n")


DECODE_RE = re.compile(
    r"Decode batch\. #running-req: (\d+),.*?page ratio: ([\d.]+), "
    r"gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")


def logs(since):
    o = subprocess.run(["docker", "logs", "--since", since, CONTAINER],
                       capture_output=True, text=True, timeout=120)
    return o.stdout + o.stderr


def main():
    os.makedirs(HOSTDIR, exist_ok=True)
    prompt, ptok = calibrate()
    print(f"[setup] prompt={ptok} tok (target {ILEN}) OLEN={OLEN} BATCH={BATCH} "
          f"maxlen={ptok+OLEN}", flush=True)
    # flush so the calibration probe cannot be reused as a prefix
    try:
        post("/flush_cache", {}, timeout=60)
    except Exception:
        pass

    t_launch = time.strftime("%Y-%m-%dT%H:%M:%S")
    time.sleep(1)
    threads = [threading.Thread(target=worker, args=(i, prompt), daemon=True)
               for i in range(BATCH)]
    t0 = time.time()
    for t in threads:
        t.start()

    seen, stable, profiled, extra = [], 0, False, 0
    t_prof_start = None
    while time.time() - t0 < WAIT_S:
        time.sleep(6)
        rows = DECODE_RE.findall(logs(t_launch))
        new = rows[len(seen):]
        seen = rows
        for bs, pr, gt, q in new:
            print(f"  [{time.time()-t0:6.1f}s] running={bs} pg={pr} "
                  f"gen_tok_s={gt} q={q}", flush=True)
        if not new:
            continue
        stable = stable + 1 if all(int(r[0]) == BATCH for r in new) else 0
        if stable >= STABLE_SAMPLES:
            if extra < EXTRA_LINES:
                extra += len(new)
                continue
            print(f"  --> stable at {BATCH}; profiling {NUM_STEPS} steps",
                  flush=True)
            t_prof_start = time.time()
            print("  start_profile:", post("/start_profile", {
                "output_dir": OUTDIR, "num_steps": NUM_STEPS,
                "activities": ["CPU", "GPU"], "with_stack": False,
                "record_shapes": True}, timeout=180), flush=True)
            profiled = True
            break

    if not profiled:
        print(f"!! never reached stable batch {BATCH}; last: {seen[-3:]}",
              flush=True)

    tp = time.time()
    while time.time() - tp < 240:
        time.sleep(10)
        if "Profiling done" in logs(t_launch):
            print("  profiling done", flush=True)
            break

    txt = logs(t_launch)
    rows = DECODE_RE.findall(txt)
    window = rows[len(seen):]
    print("  rows in/after window:", window[:6], flush=True)
    with open(os.path.join(HOSTDIR, "engine.log"), "w") as f:
        f.write(txt)

    for t in threads:
        t.join(timeout=1800)
    meta = {
        "tag": TAG, "batch": BATCH, "ilen": ILEN, "prompt_tokens": ptok,
        "olen": OLEN, "num_steps": NUM_STEPS, "t_launch": t_launch,
        "t_profile_rel_s": None if t_prof_start is None else round(t_prof_start - t0, 1),
        "rows_before_profile": seen[-6:], "rows_in_window": window[:6],
        "ok": _ok[0], "fail": _fail[0], "profiled": profiled,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(HOSTDIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
