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

"""How much host time does the mha_extend dispatch wrapper add per call?

Prefill on Qwen3-8B is 36 of these per forward, and end-to-end the marginal
prefill of ~1016 tokens costs 151 ms here against vLLM's 128 -- so per-call
Python on this path is worth attributing before touching any kernel.

Compares, in host time with no events and no sync:
  1. mha_extend_with_kvcache      -- what the runtime calls
  2. the selected kernel directly -- same work, no wrapper
  3. the wrapper's bookkeeping alone, with no kernel launched
"""

import time

import torch

from tokenspeed_kernel.ops.attention import mha_extend_with_kvcache
from tokenspeed_kernel.ops.attention import _blockscaled_signature_and_scales
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel

NQ, NKV, D, PAGE = 32, 8, 128, 64
BATCH = 1


def host_us(fn, n=2000):
    for _ in range(100):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) * 1e6 / n
    torch.cuda.synchronize()
    return dt


def build(total_q):
    pages = (total_q + PAGE - 1) // PAGE + 2
    q = torch.randn(total_q, NQ, D, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(pages, PAGE, NKV, D, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)
    cu_q = torch.tensor([0, total_q], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, total_q], dtype=torch.int32, device="cuda")
    seqlens = torch.tensor([total_q], dtype=torch.int32, device="cuda")
    return dict(q=q, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv, k_cache=k_cache,
                v_cache=v_cache, page_table=page_table, cache_seqlens=seqlens,
                max_seqlen_q=total_q, max_seqlen_k=total_q, is_causal=True)


print(f"  {'total_q':>8} {'wrapper us':>11} {'raw kernel us':>14} {'overhead us':>12}")
for total_q in (128, 512, 1024, 4096):
    a = build(total_q)
    full = lambda: mha_extend_with_kvcache(**a)
    try:
        full()
    except Exception as exc:
        print(f"  {total_q:8d}  FAIL {type(exc).__name__}: {str(exc)[:70]}")
        continue

    sig, scale_kwargs = _blockscaled_signature_and_scales(
        a["q"], a["k_cache"], a["v_cache"], None, None, None
    )
    kernel = select_kernel(
        "attention", "mha_extend_with_kvcache", sig,
        traits={"head_dim": D, "page_size": PAGE, "is_causal": True,
                "sliding_window": False, "support_logit_cap": False,
                "support_sinks": False, "return_lse": False},
        solution=None, override=None,
    )
    raw = lambda: kernel(
        q=a["q"], cu_seqlens_q=a["cu_seqlens_q"], cu_seqlens_kv=a["cu_seqlens_kv"],
        k_cache=a["k_cache"], v_cache=a["v_cache"], page_table=a["page_table"],
        cache_seqlens=a["cache_seqlens"], max_seqlen_q=a["max_seqlen_q"],
        max_seqlen_k=a["max_seqlen_k"], is_causal=True, window_left=-1,
        logit_cap=0.0, sinks=None, return_lse=False, softmax_scale=None,
        **scale_kwargs,
    )
    try:
        raw()
    except TypeError as exc:
        print(f"  {total_q:8d}  raw call signature mismatch: {str(exc)[:70]}")
        continue

    w, r = host_us(full), host_us(raw)
    print(f"  {total_q:8d} {w:11.2f} {r:14.2f} {w - r:12.2f}")

# What the bookkeeping alone costs, no kernel at all.
a = build(1024)
q, k_cache, v_cache = a["q"], a["k_cache"], a["v_cache"]
page_table, cache_seqlens = a["page_table"], a["cache_seqlens"]


def bookkeeping():
    sig, _ = _blockscaled_signature_and_scales(q, k_cache, v_cache, None, None, None)
    traits = {"head_dim": q.shape[-1], "page_size": k_cache.shape[1],
              "is_causal": True, "sliding_window": False,
              "support_logit_cap": False, "support_sinks": False,
              "return_lse": False}
    kern = select_kernel("attention", "mha_extend_with_kvcache", sig,
                         traits=traits, solution=None, override=None)
    shape_params = {
        "batch_size": cache_seqlens.shape[0], "total_q": q.shape[0],
        "num_pages": k_cache.shape[0], "page_size": k_cache.shape[1],
        "max_pages_per_seq": page_table.shape[1], "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[2], "head_dim": q.shape[-1],
        "max_seqlen_q": 1024, "max_seqlen_k": 1024,
    }
    ShapeCapture.get().record("attention", "mha_extend_with_kvcache",
                              kern.name, q.dtype, shape_params)
    with kernel_scope("attention", "mha_extend_with_kvcache", q.dtype,
                      kernel_name=kern.name, **shape_params):
        pass


print(f"\n  bookkeeping only (no kernel launched): {host_us(bookkeeping):.2f} us")
