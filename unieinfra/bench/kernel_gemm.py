# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Dense bf16 GEMM at Qwen3-8B prefill shapes: tokenspeed_kernel.mm vs torch.

Prefill of ~1016 tokens costs 151 ms here against vLLM's 128. Attention is only
~2.5 ms of that (36 layers x 70 us), so the rest is the four linear layers per
block -- a kernel family with no row on the scoreboard at all. vLLM's
unquantized linear is plain F.linear (cuBLAS), so that is the reference.
"""

import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

BACKEND = os.environ["BACKEND"]
H, I, NQ, NKV, D = 4096, 12288, 32, 8, 128  # Qwen3-8B
QKV_OUT = (NQ + 2 * NKV) * D

# (name, K, N) for one decoder block
SHAPES = [
    ("qkv_proj", H, QKV_OUT),
    ("o_proj", NQ * D, H),
    ("gate_up_proj", H, 2 * I),
    ("down_proj", I, H),
]
TOKENS = [int(t) for t in os.environ.get("GEMM_TOKENS", "1,128,1024").split(",")]

if BACKEND == "ts":
    import tokenspeed_kernel

    mm = lambda x, w: tokenspeed_kernel.mm(x, w, bias=None)
else:
    mm = lambda x, w: torch.nn.functional.linear(x, w)


def host_us(fn, n=500):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) * 1e6 / n
    torch.cuda.synchronize()
    return dt


def device_us(fn, reps=50):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    total = launches = 0
    for ev in prof.key_averages():
        if ev.device_time_total > 0 and ev.count:
            total += ev.device_time_total
            launches += ev.count
    return total / reps, launches / reps


print(f"backend={BACKEND}")
for m in TOKENS:
    block_dev = 0.0
    print(f"  tokens={m}")
    print(f"    {'layer':>14} {'K':>6} {'N':>6} {'device us':>10} {'launch':>7} "
          f"{'host us':>9} {'TFLOP/s':>9}")
    for name, k, n in SHAPES:
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        fn = lambda: mm(x, w)
        d, launches = device_us(fn)
        h = host_us(fn)
        tflops = 2 * m * k * n / (d / 1e6) / 1e12
        block_dev += d
        print(f"    {name:>14} {k:6d} {n:6d} {d:10.2f} {launches:7.1f} "
              f"{h:9.2f} {tflops:9.1f}")
    print(f"    {'block total':>14} {'':6} {'':6} {block_dev:10.2f}"
          f"   -> 36 layers = {block_dev * 36 / 1000:.2f} ms")
