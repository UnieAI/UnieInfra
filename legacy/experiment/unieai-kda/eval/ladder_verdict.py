#!/usr/bin/env python3
"""Does fp8 KV's doubled capacity pay, and where?

The +5.28% measured at C=32 was bandwidth alone: batch_running held at 32.0
with zero preemptions, so bf16's 92,032-token pool (54% used) was never the
binding constraint and having twice as much could not help. This reads the
sweep to find where it starts to.

The prediction under test is specific enough to be wrong: bf16 exhausts its pool
at 92,032/1536 = 60 concurrent requests, so the fp8 advantage should be flat and
small at C=32 and grow from C=64. A flat curve across the whole sweep falsifies
the capacity story, and the right response is to drop it.

Period-blocked throughout, with gate_stationary on the raw windows -- these arms
enter the period-2 limit cycle and a plain mean would compare two arbitrary
phases of two oscillations.
"""

import json
import os
import sys

sys.path.insert(0, "/home/ubuntu/workspace/roy/UnieInfra/experiment/unieai-kda")
from kda.gates import gate_stationary, period_blocked_mean  # noqa: E402

SP = sys.argv[1] if len(sys.argv) > 1 else "."
CONCS = [32, 64, 128, 256]
NEED = 512 + 1024          # KV tokens per request at ILEN=512 / OLEN=1024


def load(name, c):
    p = os.path.join(SP, f"lad_{name}_c{c}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    s = [w["server_tok_s"] for w in d.get("windows", [])
         if isinstance(w, dict) and w.get("server_tok_s")]
    return d, s


def stat(series):
    tail = series[len(series) // 2:]
    if len(tail) < 4:
        return None
    m, sem, _ = period_blocked_mean(tail, period=2)
    return m, sem, gate_stationary(tail).ok


print("=== fp8 KV across concurrency ===\n")
print(f"{'C':>5}  {'KV needed':>10}  {'bf16 tok/s':>16}  {'fp8 tok/s':>16}  "
      f"{'delta':>9}  {'sigma':>6}")
print("-" * 78)
rows = []
for c in CONCS:
    a, b = load("bf16", c), load("fp8", c)
    if not (a and b):
        print(f"{c:>5}  {'':>10}  {'(missing arm)':>16}")
        continue
    sa, sb = stat(a[1]), stat(b[1])
    if not (sa and sb):
        print(f"{c:>5}  too few windows")
        continue
    (ma, ea, oka), (mb, eb, okb) = sa, sb
    delta = (mb - ma) / ma * 100
    sig = (ea ** 2 + eb ** 2) ** 0.5
    ns = abs(mb - ma) / sig if sig else float("inf")
    flag = "" if (oka and okb) else "  <-- NOT STATIONARY"
    print(f"{c:>5}  {c * NEED:>10,}  {ma:>9.1f}+/-{ea:<5.1f}  {mb:>9.1f}+/-{eb:<5.1f}  "
          f"{delta:>+8.2f}%  {ns:>5.1f}{flag}")
    rows.append((c, delta, ns, oka and okb))

print()
usable = [r for r in rows if r[3]]
if len(usable) < 2:
    print("not enough stationary points to read a trend")
    raise SystemExit(0)
# The split must be at the WALL, not at an arbitrary concurrency. bf16 admits
# ~91 concurrent requests (92,032 KV tokens / ~1024 mean sequence length), and
# batch_running confirms it: 90.8 at C=128, 92.7 at C=256, while C=32 and C=64
# both run FULL with zero preemptions. So C=32 and C=64 are BOTH below the wall
# and neither is capacity-limited -- any difference there is bandwidth, and a
# growing advantage across them is bandwidth scaling with batch size (a larger
# batch reads more KV per decode step, so halving KV reads matters more), not
# capacity. Splitting at C=64 would read that growth as a capacity effect.
below = [d for c, d, _, _ in usable if c <= 64]
above = [d for c, d, _, _ in usable if c >= 128]
if below:
    print(f"below the wall (C<=64, neither arm throttled): "
          f"{sum(below)/len(below):+.2f}% mean -- bandwidth")
if not above:
    print("above the wall (C>=128): no stationary data yet -- the capacity")
    print("  question is still open. Do not read the trend below the wall as")
    print("  evidence for it.")
elif sum(above) / len(above) > sum(below) / len(below) + 2:
    print(f"above the wall (C>=128): {sum(above)/len(above):+.2f}% mean")
    print("-> the advantage grows once bf16 is throttled: capacity pays in tok/s")
else:
    print(f"above the wall (C>=128): {sum(above)/len(above):+.2f}% mean")
    print("-> no extra gain once bf16 is throttled: the GPU saturates before")
    print("   the KV ceiling, so capacity buys queueing latency, not tok/s.")
