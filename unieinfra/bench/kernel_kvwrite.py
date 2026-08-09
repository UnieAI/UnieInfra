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

"""KV cache write scoreboard row, same two-instrument method as rope_final.py.

Each engine writes into its own native cache layout -- tokenspeed a flat
[slots, heads, dim] pool, vLLM a paged [blocks, block_size, heads, dim] one --
because that is what each actually does in the serving path. Same token counts,
same bytes moved.
"""

import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

KV_HEADS, HEAD_DIM = 8, 128  # Qwen3-8B
SLOTS = 65536
TOKENS = [1, 32, 1024, 4096]
BACKEND = os.environ["BACKEND"]


def host_us(fn, n=3000):
    for _ in range(100):
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


if BACKEND == "vllm":
    from vllm import _custom_ops as ops

    BLOCK = 16
    k_cache = torch.zeros(
        SLOTS // BLOCK, BLOCK, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    v_cache = torch.zeros_like(k_cache)
    one = torch.tensor(1.0, device="cuda")

    def make(n, k, v, loc):
        return lambda: ops.reshape_and_cache_flash(
            k, v, k_cache, v_cache, loc, "auto", one, one
        )

else:
    import tokenspeed_kernel.ops  # noqa: F401
    from tokenspeed_kernel.ops.kvcache.triton import store_kv_cache

    k_cache = torch.zeros(
        SLOTS, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    v_cache = torch.zeros_like(k_cache)

    def make(n, k, v, loc):
        return lambda: store_kv_cache(k, v, k_cache, v_cache, loc)


print(f"backend={BACKEND}")
print(f"  {'tokens':>7} {'device us':>10} {'launches':>9} {'host us':>9} {'GB/s':>9}")
for n in TOKENS:
    k = torch.randn(n, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    loc = torch.randperm(SLOTS, device="cuda")[:n].to(torch.int64)
    fn = make(n, k, v, loc)
    d, launches = device_us(fn)
    h = host_us(fn)
    gb = k.numel() * 2 * 2 * 2 / 1e9  # k+v, read+write
    print(f"  {n:7d} {d:10.2f} {launches:9.1f} {h:9.2f} {gb / (d / 1e6):9.1f}")
