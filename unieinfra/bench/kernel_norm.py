# Copyright (c) 2026 UnieAI
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

"""RMSNorm scoreboard row, same two-instrument method as rope_final.py.

Measures the layer the runtime actually calls (tokenspeed: RMSNorm.forward ->
ops/layernorm/flashinfer.py; vLLM: RMSNorm.forward_cuda), for both the plain
and the fused-add-residual form.
"""

import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

HIDDEN = 4096
ROWS = [1, 32, 1024, 4096]
BACKEND = os.environ["BACKEND"]

if BACKEND == "vllm":
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationMode
    from vllm.model_executor.layers.layernorm import RMSNorm

    # A bare VllmConfig() enables inductor, and CUDA's default op priority is
    # ["native"] when compiling -- i.e. the eager PyTorch decomposition, which
    # inductor would then fuse. Measuring that eagerly understates vLLM. With
    # compilation off the priority becomes ["vllm_c", "native"], which is the
    # hand-written CUDA kernel this comparison is actually about.
    cfg = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    with set_current_vllm_config(cfg):
        cfg.kernel_config.set_platform_defaults(cfg)
        import vllm.kernels  # noqa: F401  (registers the vllm_c impls)
        from vllm import ir

        ir.ops.rms_norm.set_default(cfg.kernel_config.ir_op_priority.rms_norm)
        ir.ops.fused_add_rms_norm.set_default(
            cfg.kernel_config.ir_op_priority.fused_add_rms_norm
        )
        norm = RMSNorm(HIDDEN, eps=1e-6).cuda().to(torch.bfloat16)
    fwd = norm.forward_cuda
else:
    import tokenspeed_kernel.ops  # noqa: F401

    from tokenspeed.runtime.layers.layernorm import RMSNorm

    norm = RMSNorm(HIDDEN, eps=1e-6).cuda().to(torch.bfloat16)
    fwd = norm.forward


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


print(f"backend={BACKEND}")
for label, use_res in (("rmsnorm", False), ("fused_add_rmsnorm", True)):
    print(f"  {label}")
    print(
        f"    {'rows':>6} {'device us':>10} {'launches':>9} {'host us':>9} {'GB/s':>9}"
    )
    for n in ROWS:
        x = torch.randn(n, HIDDEN, dtype=torch.bfloat16, device="cuda")
        res = torch.randn_like(x) if use_res else None
        fn = (lambda: fwd(x, res)) if use_res else (lambda: fwd(x))
        d, launches = device_us(fn)
        h = host_us(fn)
        gb = x.numel() * 2 * (3 if use_res else 2) / 1e9
        print(f"    {n:6d} {d:10.2f} {launches:9.1f} {h:9.2f} {gb / (d / 1e6):9.1f}")
