# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
        rope = get_rope(
            H,
            MP,
            True,
            {"rope_theta": 1e6, "rope_type": "default"},
            dtype=torch.bfloat16,
        ).cuda()
    make = lambda pos, q, k: (lambda: rope.forward_cuda(pos, q, k))
else:
    from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels

    load_builtin_kernels()
    impl = KernelRegistry.get().get_impl("cuda_embedding_rope")
    make = lambda pos, q, k: (
        lambda: impl(
            positions=pos, q=q, k=k, head_size=H, cos_sin_cache=cache, is_neox=True
        )
    )


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
