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

"""RoPE scoreboard row: device kernel time + host enqueue cost, both engines.

Two separate instruments on purpose. CUDA-event deltas are useless here: the
host takes longer to enqueue than the device takes to run, so the GPU idles
between the events and the "device" number reads as host stall. Device time
comes from the profiler (its kernel names and counts are exact even though its
host times are inflated ~3.6x on this box); host time comes from an enqueue
loop with no sync and no events in it.
"""

import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

H, NQ, NKV, MP = 128, 32, 8, 8192
TOKENS = [1, 32, 1024, 4096]
BACKEND = os.environ["BACKEND"]

inv = 1.0 / (1e6 ** (torch.arange(0, H, 2, dtype=torch.float32) / H))
f = torch.einsum("i,j->ij", torch.arange(MP, dtype=torch.float32), inv)
cache = torch.cat((f.cos(), f.sin()), dim=-1).cuda()

if BACKEND == "vllm":
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(H, MP, True, {"rope_theta": 1e6, "rope_type": "default"},
                        dtype=torch.bfloat16).cuda()
    make = lambda pos, q, k: (lambda: rope.forward_cuda(pos, q, k))
else:
    from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels

    load_builtin_kernels()
    impl = KernelRegistry.get().get_impl("cuda_embedding_rope")
    make = lambda pos, q, k: (lambda: impl(positions=pos, q=q, k=k, head_size=H,
                                           cos_sin_cache=cache, is_neox=True))


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
    total = n_launch = 0
    for ev in prof.key_averages():
        if ev.device_time_total > 0 and ev.count:
            total += ev.device_time_total
            n_launch += ev.count
    return total / reps, n_launch / reps


print(f"backend={BACKEND}")
print(f"  {'tokens':>7} {'device us':>10} {'launches':>9} {'host us':>9} {'GB/s':>9}")
for n in TOKENS:
    pos = torch.arange(n, dtype=torch.int64, device="cuda")
    q = torch.randn(n, NQ * H, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(n, NKV * H, dtype=torch.bfloat16, device="cuda")
    fn = make(pos, q, k)
    d, launches = device_us(fn)
    h = host_us(fn)
    gb = (q.numel() + k.numel()) * 2 * 2 / 1e9
    print(f"  {n:7d} {d:10.2f} {launches:9.1f} {h:9.2f} {gb / (d / 1e6):9.1f}")
