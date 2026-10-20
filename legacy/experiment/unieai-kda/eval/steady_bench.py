#!/usr/bin/env python3
"""Windowed steady-state throughput + GPU utilisation probe.

v2. The first version counted whole requests, which aliased badly: 128 workers
started together, every request had identical fixed length, so they stayed in
lockstep and completions quantised to exactly one cohort per window. Three
identical windows then satisfied the convergence test for the wrong reason.

Two fixes:
  * workers start staggered across STAGGER_S, breaking the cohort
  * throughput is counted per streamed token, not per finished request, so the
    window metric is continuous and cannot quantise

Each request should emit exactly OLEN chunks (min_tokens == max_tokens,
ignore_eos). Any request that does not is counted in `token_mismatch`, so the
instrument reports when its own assumption breaks.

stdlib only.
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8001")
MODEL = os.environ.get("MODEL_NAME", "bench-model")
GPU_IDX = os.environ.get("GPU_IDX", "1")
CONC = int(os.environ.get("CONC", "128"))
ILEN = int(os.environ.get("ILEN", "1024"))
OLEN = int(os.environ.get("OLEN", "256"))
WINDOW = float(os.environ.get("WINDOW", "15"))
TOL = float(os.environ.get("TOL", "0.01"))
STABLE_NEEDED = int(os.environ.get("STABLE_NEEDED", "3"))
STAGGER_S = float(os.environ.get("STAGGER_S", "20"))
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "900"))
OUT = os.environ.get("OUT", "steady_result.json")
# vLLM serves prometheus on the API port; TokenSpeed's gateway puts it on a
# separate one (`--prometheus-port`, otherwise a random free port). Override
# this when the two differ, or the counter_agreement gate reads nothing.
METRICS_URL = os.environ.get("METRICS_URL") or (BASE + "/metrics")
# "repeat" (default, unchanged) fills the prompt with one repeated word. That is
# fine for measuring attention/GEMM throughput, where only token count matters,
# and every result before 2026-07-31 used it.
#
# It is actively misleading for speculative decoding. ngram and suffix drafting
# predict from the text itself, so a prompt of "benchmark " x768 hands them a
# near-perfect hit rate that no real workload would produce -- the measurement
# would report a speedup that evaporates in production. "prose" draws each
# request a distinct slice of real English (the vLLM docs tree) so acceptance
# rates reflect ordinary text.
PROMPT_MODE = os.environ.get("PROMPT_MODE", "repeat")
CORPUS_DIR = os.environ.get(
    "CORPUS_DIR", "/workspace/roy/UnieInfra/unieinfra-vllm/docs")

_done = threading.Event()
_tokens = [0] * CONC          # per-worker, no lock needed (each owns its slot)
_reqs = [0] * CONC
_fail = [0] * CONC
_mismatch = [0] * CONC
# Server-reported completion_tokens, summed from each request's final usage
# chunk. This is the fallback authority for counter_agreement on an engine
# whose prometheus counters are unreachable — TokenSpeed's `serve` puts the
# engine's `tokenspeed:*` metrics behind an SMG gateway that exports only its
# own `smg_*`, so scraping gets HTTP-level counters and no token counts.
# Unlike a prometheus counter this only advances when a request finishes, so
# it is bursty inside one window and should be read over periods, not windows.
_srv_tokens = [0] * CONC
_sm_lock = threading.Lock()
_sm_samples = []


def post_json(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# Both engines export the same two quantities under their own namespace, and
# prometheus appends `_total` to counters, so accept the name with or without
# it. An arm whose prefix is missing here reports gen=None, which reads as "no
# server counter" and silently disables the counter_agreement gate — the gate
# that caught the speculative-decoding retraction. Keep this list in step with
# whatever engines are under comparison.
_ENGINE_PREFIXES = ("vllm", "tokenspeed")
_GEN_NAMES = frozenset(
    f"{e}:generation_tokens{suffix}"
    for e in _ENGINE_PREFIXES
    for suffix in ("", "_total")
)
_RUNNING_NAMES = frozenset(f"{e}:num_requests_running" for e in _ENGINE_PREFIXES)


def server_counters():
    """Authoritative server-side counters, to cross-check the client.

    A pure-Python client parsing SSE per token can itself become the limiter;
    if that happens the client rate reads low while the server is still
    producing. Comparing against the server's own generation counter is what
    tells the two apart.

    Returns:
        (generation_tokens, requests_running), either of which is None when the
        server does not export it. None is not zero — a caller that treats it as
        zero turns a missing instrument into a passing gate.
    """
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=10) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return None, None
    gen, running = None, None
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except Exception:
            continue
        if name in _GEN_NAMES:
            gen = (gen or 0.0) + value
        elif name in _RUNNING_NAMES:
            running = (running or 0.0) + value
    return gen, running


def count_tokens(text):
    """Prompt length in tokens, without assuming a /tokenize endpoint.

    vLLM exposes /tokenize; TokenSpeed does not. `usage.prompt_tokens` from a
    1-token completion is the tokenizer both engines already agree on, so
    calibration is identical across arms rather than engine-specific.
    """
    try:
        return len(post_json("/tokenize", {"model": MODEL, "prompt": text})["tokens"])
    except Exception:
        r = post_json("/v1/completions", {
            "model": MODEL, "prompt": text, "max_tokens": 1,
            "temperature": 0.0, "stream": False}, timeout=120)
        return int(r["usage"]["prompt_tokens"])


def _load_corpus():
    """Concatenate real prose from the docs tree, code fences removed.

    Fenced blocks are dropped because they are far more self-similar than prose
    and would drift this back toward the repeated-word case the mode exists to
    avoid. Returns one long string; callers slice distinct windows out of it.
    """
    import glob

    parts = []
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "**", "*.md"),
                                 recursive=True)):
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        keep, fenced = [], False
        for line in text.splitlines():
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            if not fenced and line.strip():
                keep.append(line.strip())
        parts.append(" ".join(keep))
    return " ".join(parts)


_CORPUS = None


def prose_slice(wid, i, n_words):
    """A distinct, deterministic window of the corpus per (worker, request)."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = _load_corpus()
    words = _CORPUS.split()
    if not words:
        return "benchmark " * n_words
    # Stride by a prime so successive requests do not overlap heavily.
    start = ((wid * 7919) + (i * 1031)) % max(1, len(words) - n_words)
    return " ".join(words[start:start + n_words])


def calibrate_prompt():
    """Return (prompt_or_template, measured_tokens).

    In prose mode the returned text is only a length calibration sample; each
    worker builds its own slice at request time, so the token count is
    approximate rather than exact. ILEN is matched on the sample.
    """
    if PROMPT_MODE == "prose":
        n = int(ILEN * 0.75)
        got = 0
        for _ in range(12):
            text = prose_slice(0, 0, n)
            got = count_tokens(text)
            if abs(got - ILEN) <= 8:
                return n, got
            n = max(1, int(n * ILEN / max(got, 1)))
        return n, got

    word = "benchmark "
    n = int(ILEN * 0.75)
    text = word * n
    got = 0
    for _ in range(12):
        text = word * n
        got = count_tokens(text)
        if abs(got - ILEN) <= 8:
            return text, got
        n = max(1, int(n * ILEN / max(got, 1)))
    return text, got


def worker(prompt, wid):
    # Stagger so the cohort does not stay synchronised for the whole run.
    time.sleep(STAGGER_S * wid / CONC)
    i = 0
    while not _done.is_set():
        i += 1
        text = prose_slice(wid, i, prompt) if PROMPT_MODE == "prose" else prompt
        body = {
            "model": MODEL,
            "prompt": f"[w{wid}r{i}] " + text,   # unique -> no prefix-cache inflation
            "max_tokens": OLEN, "min_tokens": OLEN,
            "ignore_eos": True, "temperature": 0.0, "stream": True,
            # Final chunk carries the server's own token count. Both engines
            # send it with `choices: []`, which the token loop below already
            # skips, so this cannot inflate the client count.
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            BASE + "/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        n_chunk = 0
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                for raw in r:
                    if _done.is_set():
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    body_s = line[5:].strip()
                    if body_s == "[DONE]":
                        break
                    try:
                        obj = json.loads(body_s)
                    except Exception:
                        continue
                    usage = obj.get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        _srv_tokens[wid] += int(usage["completion_tokens"])
                    ch = obj.get("choices") or []
                    if ch and ch[0].get("text"):
                        n_chunk += 1
                        # One SSE chunk is NOT one token. Under speculative
                        # decoding the engine emits every token accepted in a
                        # step together, so a chunk can carry k+1 of them. This
                        # line used to do `_tokens[wid] += 1`, which undercounted
                        # by the acceptance factor and turned a +45% MTP gain
                        # into a reported -42% loss. Prefer the token ids the
                        # server sends; fall back to the chunk count only when
                        # they are absent, and let client_captures_frac expose
                        # the discrepancy when that fallback is wrong.
                        tid = ch[0].get("token_ids")
                        _tokens[wid] += len(tid) if tid else 1
            _reqs[wid] += 1
            if n_chunk != OLEN and not _done.is_set():
                _mismatch[wid] += 1
        except Exception:
            _fail[wid] += 1
            if _done.is_set():
                return
            time.sleep(0.2)


def sm_sampler():
    p = subprocess.Popen(
        ["nvidia-smi", "dmon", "-i", GPU_IDX, "-s", "u", "-d", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        for line in p.stdout:
            if _done.is_set():
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                with _sm_lock:
                    _sm_samples.append((time.time(), int(parts[1])))
    finally:
        p.terminate()


def main():
    prompt, tok = calibrate_prompt()
    print(f"[setup] prompt={tok} tok (target {ILEN}), OLEN={OLEN}, conc={CONC}, "
          f"window={WINDOW}s, tol={TOL:.1%}, stable_needed={STABLE_NEEDED}, "
          f"stagger={STAGGER_S}s, gpu={GPU_IDX}", flush=True)

    threading.Thread(target=sm_sampler, daemon=True).start()
    for w in range(CONC):
        threading.Thread(target=worker, args=(prompt, w), daemon=True).start()

    t0 = time.time()
    prev_tok, prev_t = 0, t0
    prev_srv, _ = server_counters()
    windows, stable, converged = [], 0, False

    while time.time() - t0 < MAX_SECONDS:
        time.sleep(WINDOW)
        now = time.time()
        tok_now = sum(_tokens)
        srv_now, running = server_counters()
        srv_source = "prometheus"
        if srv_now is None:
            # No scrapeable counter on this engine; fall back to the sum of
            # per-request usage. Never leave it None — a missing server count
            # makes counter_agreement vacuously pass.
            srv_now = float(sum(_srv_tokens))
            srv_source = "stream_usage"
        with _sm_lock:
            sm = [v for (ts, v) in _sm_samples if prev_t <= ts <= now]
        dt = now - prev_t
        tps = (tok_now - prev_tok) / dt
        srv_tps = ((srv_now - prev_srv) / dt
                   if srv_now is not None and prev_srv is not None else None)
        w = {
            "window": len(windows) + 1,
            "t_s": round(now - t0, 1),
            "output_tok_s": round(tps, 1),
            "server_tok_s": round(srv_tps, 1) if srv_tps is not None else None,
            "server_tok_source": srv_source,
            "client_vs_server": (round(tps / srv_tps, 4)
                                 if srv_tps else None),
            "batch_running": running,
            "reqs_done": sum(_reqs),
            "failed": sum(_fail),
            "token_mismatch": sum(_mismatch),
            "sm_mean": round(sum(sm) / len(sm), 1) if sm else None,
            "sm_min": min(sm) if sm else None,
        }
        prev_srv = srv_now if srv_now is not None else prev_srv
        windows.append(w)
        print(json.dumps(w), flush=True)

        # Only start testing convergence once every worker has launched.
        if now - t0 > STAGGER_S + WINDOW and len(windows) >= 2:
            a, b = windows[-2]["output_tok_s"], windows[-1]["output_tok_s"]
            if a > 0 and abs(b - a) / a < TOL:
                stable += 1
                if stable >= STABLE_NEEDED:
                    converged = True
                    prev_tok, prev_t = tok_now, now
                    break
            else:
                stable = 0
        prev_tok, prev_t = tok_now, now

    _done.set()
    time.sleep(0.5)

    tail = windows[-STABLE_NEEDED:] if converged else windows[-3:]
    steady = sum(x["output_tok_s"] for x in tail) / len(tail) if tail else 0.0
    srv_tail = [x["server_tok_s"] for x in tail if x.get("server_tok_s")]
    steady_srv = sum(srv_tail) / len(srv_tail) if srv_tail else None
    sms = [x["sm_mean"] for x in tail if x["sm_mean"] is not None]
    spread = (max(x["output_tok_s"] for x in tail) -
              min(x["output_tok_s"] for x in tail)) / steady if tail and steady else None
    result = {
        "converged": converged,
        "windows_run": len(windows),
        "time_to_converge_s": windows[-1]["t_s"] if converged else None,
        "steady_output_tok_s": round(steady, 1),
        "steady_server_tok_s": round(steady_srv, 1) if steady_srv else None,
        "client_captures_frac": (round(steady / steady_srv, 4)
                                 if steady_srv else None),
        "steady_batch_running": tail[-1].get("batch_running") if tail else None,
        "steady_window_spread": round(spread, 4) if spread is not None else None,
        "steady_sm_mean": round(sum(sms) / len(sms), 1) if sms else None,
        "steady_sm_min": min(x["sm_min"] for x in tail if x["sm_min"] is not None) if tail else None,
        "total_failed": sum(_fail),
        "total_token_mismatch": sum(_mismatch),
        "config": {"conc": CONC, "ilen": tok, "olen": OLEN, "window_s": WINDOW,
                   "tol": TOL, "stagger_s": STAGGER_S, "gpu": GPU_IDX},
        "windows": windows,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== RESULT ===", flush=True)
    print(f"converged        : {converged}", flush=True)
    print(f"converged at     : {result['time_to_converge_s']}s", flush=True)
    print(f"steady tok/s     : {result['steady_output_tok_s']}  "
          f"(tail spread {spread:.2%})" if spread is not None else "", flush=True)
    print(f"steady SM        : mean {result['steady_sm_mean']}%  "
          f"min {result['steady_sm_min']}%", flush=True)
    print(f"failed / mismatch: {result['total_failed']} / {result['total_token_mismatch']}", flush=True)
    print(f"\nwritten to {OUT}", flush=True)
    return 0 if converged else 2


if __name__ == "__main__":
    sys.exit(main())
