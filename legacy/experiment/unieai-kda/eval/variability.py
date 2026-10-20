#!/usr/bin/env python3
"""Compare throughput *distributions*, not just their means.

The 900s bf16 C=128 rerun fixed the drift (-11.5% -> +2.1%) but still failed
gate_stationary on envelope, because the windows range 265 to 1597 tok/s -- a 6x
spread around a 963.7 mean. That is not measurement error to be averaged away;
it is what running at the KV ceiling does, and `batch_running` sat at 88.9/128
throughout. fp8 at the same offered load used 130,458 of 184,064 tokens and was
never at its ceiling.

So the interesting comparison may not be "faster" but "steadier". A service
whose 15s throughput swings 6x has tail latencies a mean cannot show. This
reports the shape: percentiles, coefficient of variation, and how often a window
falls badly below the median.

Means are still printed, flagged when gate_stationary rejects them, because a
mean from a non-stationary series is not a point estimate.
"""

import json
import os
import statistics as st
import sys

sys.path.insert(0, "/home/ubuntu/workspace/roy/UnieInfra/experiment/unieai-kda")
from kda.gates import gate_stationary, period_blocked_mean  # noqa: E402

SP = sys.argv[1] if len(sys.argv) > 1 else "."


def pct(xs, q):
    xs = sorted(xs)
    i = min(int(q / 100 * len(xs)), len(xs) - 1)
    return xs[i]


def report(label, path, cap=None):
    if not os.path.exists(path):
        print(f"{label:22s} (not written yet)")
        return
    d = json.load(open(path))
    w = [x for x in d["windows"] if x.get("server_tok_s")]
    tail = w[len(w) // 2:]
    s = [x["server_tok_s"] for x in tail]
    br = [x["batch_running"] for x in tail if x.get("batch_running")]
    if len(s) < 6:
        print(f"{label:22s} only {len(s)} windows")
        return
    m, sem, _ = period_blocked_mean(s, period=2)
    g = gate_stationary(s)
    cv = st.pstdev(s) / st.mean(s)
    med = st.median(s)
    bad = sum(1 for x in s if x < med * 0.5)      # windows at under half the median
    occ = ""
    if cap and br:
        kv = sum(br) / len(br) * 1024
        occ = f"  KV {kv / cap:.0%} of cap"
    print(f"{label:22s} mean {m:7.1f}{'' if g.ok else ' (NOT STATIONARY)'}")
    print(f"{'':22s} p5 {pct(s,5):6.0f}  p50 {med:6.0f}  p95 {pct(s,95):6.0f}"
          f"   CV {cv:5.1%}   windows under half-median: {bad}/{len(s)}")
    if br:
        print(f"{'':22s} batch_running {sum(br)/len(br):6.1f}{occ}")


print("=== throughput distribution at the KV ceiling ===\n")
report("bf16 C=128 (900s)", f"{SP}/hi_bf16_c128.json", 92032)
report("fp8  C=128 (900s)", f"{SP}/hi_fp8_c128.json", 184064)
print()
report("bf16 C=256 (900s)", f"{SP}/hi_bf16_c256.json", 92032)
report("fp8  C=256 (900s)", f"{SP}/hi_fp8_c256.json", 184064)
print("\nA lower CV and fewer sub-half-median windows at the same offered load")
print("means capacity bought predictability, which no mean can show.")
