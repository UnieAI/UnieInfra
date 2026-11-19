#!/usr/bin/env python3
"""Long-context decode load for kernel tracing, and a rocprofv3 CSV summariser.

Two modes:
  run <conc> <prefix_tokens> <out_tokens>   drive CONC concurrent streams and
                                            report ITL / TTFT per request
  top <kernel_trace.csv> [n]                aggregate kernel time, top n

The load is deliberately shape-only (same prompt length, same concurrency, same
decode length) so a trace taken under it is comparable between runs; the real
benchmark remains AgentX.
"""
import json
import sys
import threading
import time
import urllib.request
from statistics import mean

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"

# Neutral filler; ~3.5 characters per token for English prose, refined by the
# prompt_tokens the server reports back.
FILLER = (
    "The scheduler admits a request only when the block table has room for the "
    "prompt and the decode reserve. A batch that mixes prefill and decode pays "
    "the larger of the two costs, so the engine keeps separate rings and drains "
    "them in proportion to the queued work. "
)


def post(payload):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=3600)


def one(idx, prefix_tokens, out_tokens, result):
    text = ("Request %d. " % idx) + FILLER * max(1, int(prefix_tokens * 3.5 / len(FILLER)))
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text + "\nSummarise the above in detail."}],
        "max_tokens": out_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    itls, first, last, ntok, prompt_tokens = [], None, None, 0, 0
    t0 = time.time()
    with post(payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if not delta.get("content"):
                    continue
                now = time.time()
                ntok += 1
                if first is None:
                    first = now
                else:
                    itls.append((now - last) * 1000.0)
                last = now
    result[idx] = {
        "prompt_tokens": prompt_tokens,
        "out_tokens": ntok,
        "ttft_s": (first - t0) if first else float("nan"),
        "itl_ms": itls,
        "wall_s": last - t0 if last else float("nan"),
    }


def run(conc, prefix_tokens, out_tokens):
    result = {}
    threads = [
        threading.Thread(target=one, args=(i, prefix_tokens, out_tokens, result))
        for i in range(conc)
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    itls = [v for r in result.values() for v in r["itl_ms"]]
    if not itls:
        print("no decode tokens returned")
        return 1
    itls.sort()

    def pct(p):
        return itls[min(len(itls) - 1, int(len(itls) * p))]

    out = sum(r["out_tokens"] for r in result.values())
    print(
        "conc %d  prefix %s tok (server-reported)  out %d tok  wall %.1fs"
        % (
            conc,
            sorted({r["prompt_tokens"] for r in result.values()}),
            out,
            wall,
        )
    )
    print(
        "ITL ms n=%d p50 %.1f p90 %.1f p99 %.1f max %.1f | decode-only %.0f tok/s | TTFT p50 %.1fs"
        % (len(itls), pct(0.50), pct(0.90), pct(0.99), itls[-1],
           out / wall, sorted(r["ttft_s"] for r in result.values())[len(result) // 2])
    )
    return 0


def top(path, n=20):
    import csv
    import os
    from collections import defaultdict

    if not os.path.exists(path):
        print("no such file: %s" % path)
        return 1
    total = defaultdict(float)
    count = defaultdict(int)
    span = [None, None]
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Kernel_Name") or row.get("Name") or "?"
            dur = row.get("Duration") or row.get("DurationNs") or row.get("Duration_Ns")
            if dur is None:
                continue
            try:
                # rocprofv3 writes a value like "1234" or "1.234 us".
                if any(c.isalpha() for c in dur):
                    value, unit = dur.split()
                    scale = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}[unit.strip()]
                    ns = float(value) * scale
                else:
                    ns = float(dur)
            except (ValueError, KeyError):
                continue
            total[name] += ns
            count[name] += 1
            start = row.get("Start_Timestamp") or row.get("Start_TimestampNs")
            end = row.get("End_Timestamp") or row.get("End_TimestampNs")
            try:
                if start is not None:
                    span[0] = float(start) if span[0] is None else min(span[0], float(start))
                    span[1] = float(end) if span[1] is None else max(span[1], float(end))
            except ValueError:
                pass
    if not total:
        print("no kernel rows in %s" % path)
        return 1
    grand = sum(total.values())
    print("kernels %d events, %.1f ms of kernel time total" % (sum(count.values()), grand / 1e6))
    print("%-10s %8s %10s %9s  %s" % ("share", "count", "total_ms", "mean_us", "kernel"))
    for name, ns in sorted(total.items(), key=lambda kv: -kv[1])[:n]:
        print(
            "%-9.1f%% %8d %10.1f %9.1f  %s"
            % (100.0 * ns / grand, count[name], ns / 1e6, ns / 1e3 / count[name], name[:95])
        )
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "run":
        sys.exit(run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])))
    if sys.argv[1] == "top":
        sys.exit(top(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20))
    print(__doc__)
    sys.exit(2)
