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

"""Extend attention at RAG shapes: short query against a long cached KV.

The square shape (q == kv) that the earlier sweep used is what a cold prefill
looks like. A partial prefix hit is the opposite: a few hundred new tokens
attending to thousands of cached ones, and that is the shape both remaining
gaps live in (rag TTFT 20% behind, shared throughput 12% behind).
"""

import os
import time

import torch
from tokenspeed_kernel.ops.attention import mha_extend_with_kvcache
from torch.profiler import ProfilerActivity, profile

NQ, NKV, D, PAGE = 32, 8, 128, 64
# (new query tokens, total KV length)
SHAPES = [
    (1, 4096),  # decode-like tail of a chunked prefill
    (32, 4096),  # shared-throughput shape
    (360, 4200),  # the measured rag shape
    (1024, 5120),
    (4096, 4096),  # square, for reference
]


def build(q_len, kv_len, device="cuda"):
    pages = (kv_len + PAGE - 1) // PAGE
    q = torch.randn(q_len, NQ, D, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(pages + 2, PAGE, NKV, D, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(pages, dtype=torch.int32, device=device).view(1, -1)
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    cu_kv = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
    seqlens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    return dict(
        q=q,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=seqlens,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        is_causal=True,
    )


def device_us(fn, reps=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (
        sum(
            ev.device_time_total
            for ev in prof.key_averages()
            if ev.device_time_total > 0 and ev.count
        )
        / reps
    )


def host_us(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) * 1e6 / n
    torch.cuda.synchronize()
    return dt


def unwrap(r):
    return r[0] if isinstance(r, tuple) else r


print(
    f"  {'q':>6} {'kv':>6} {'triton us':>10} {'fa2 us':>9} {'ratio':>7} "
    f"{'max|d|':>8} {'tri host':>9} {'fa2 host':>9}  36 層合計"
)
for q_len, kv_len in SHAPES:
    a = build(q_len, kv_len)
    tri = lambda: unwrap(mha_extend_with_kvcache(**a, solution="triton"))
    fa2 = lambda: unwrap(mha_extend_with_kvcache(**a, solution="flashinfer_fa2"))
    try:
        o_t, o_f = tri(), fa2()
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"  {q_len:6d} {kv_len:6d}  FAIL {type(exc).__name__}: {str(exc)[:60]}")
        continue
    diff = (o_t.float() - o_f.float()).abs().max().item()
    d_t, d_f = device_us(tri), device_us(fa2)
    h_t, h_f = host_us(tri), host_us(fa2)
    per_fwd_t, per_fwd_f = max(d_t, h_t) * 36 / 1000, max(d_f, h_f) * 36 / 1000
    print(
        f"  {q_len:6d} {kv_len:6d} {d_t:10.1f} {d_f:9.1f} {d_t / d_f:6.2f}x "
        f"{diff:8.4f} {h_t:9.1f} {h_f:9.1f}   "
        f"tri {per_fwd_t:6.2f} ms / fa2 {per_fwd_f:6.2f} ms"
    )
