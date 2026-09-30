"""Prefill attention at gemma-4-31B's two geometries, ours vs vLLM's kernel.

Why this exists: at c=1, with no queueing and no scheduling, our TTFT is 1.24x
to 2.00x vLLM's on the same prompts. That is a per-request cost and the only
thing in it is prefill. But TTFT also contains tokenization, the HTTP hop and
the first decode step, so it cannot say whether the kernel is slow or the
engine is slow around it. This isolates the kernel.

gemma-4-31B is 60 layers: 10 full-attention (32 q x 4 kv x 512) and 50 sliding
(32 q x 16 kv x 256, window 1024). Both are wide heads -- Qwen is 128 -- and
wide heads on sm_120's 100 KB of shared memory are exactly where the tiling
heuristics in mha_prefill.py:424-465 and vLLM's triton_unified_attention
disagree:

    ours   sliding (Lq 256, window>=0)  -> BLOCK_M 32, BLOCK_N 64, 8 warps
    vLLM   head_size 256, sm_120        -> BLOCK_M 64,             8 warps

Each was measured better than its own alternative. Neither was measured
against the other.
"""

import time

import torch
from tokenspeed_kernel.ops.attention import mha_extend_with_kvcache
from torch.profiler import ProfilerActivity, profile

PAGE = 64
GEOM = {  # name: (num_q_heads, num_kv_heads, head_dim, sliding_window)
    "sliding": (32, 16, 256, 1024),
    "full": (32, 4, 512, -1),
}
# M is the chunk the scheduler hands the kernel; 1024 is what we run in
# production (--chunked-prefill-size 1024) and 8192 is the engine default.
MS = [256, 1024, 2048, 8192]


def build(nq, nkv, d, m, kv_len, device="cuda"):
    pages = (kv_len + PAGE - 1) // PAGE
    q = torch.randn(m, nq, d, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(pages + 2, PAGE, nkv, d, dtype=torch.bfloat16, device=device)
    return dict(
        q=q,
        cu_seqlens_q=torch.tensor([0, m], dtype=torch.int32, device=device),
        cu_seqlens_kv=torch.tensor([0, kv_len], dtype=torch.int32, device=device),
        k_cache=k_cache,
        v_cache=torch.randn_like(k_cache),
        page_table=torch.arange(pages, dtype=torch.int32, device=device).view(1, -1),
        cache_seqlens=torch.tensor([kv_len], dtype=torch.int32, device=device),
        max_seqlen_q=m,
        max_seqlen_k=kv_len,
        is_causal=True,
    )


def device_us(fn, reps=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    return (
        sum(
            e.device_time_total
            for e in prof.key_averages()
            if e.device_time_total > 0 and e.count
        )
        / reps
    )


def unwrap(r):
    return r[0] if isinstance(r, tuple) else r


print(
    f"{'geom':>8} {'M':>6} {'kv':>7} {'triton us':>10} {'fi us':>9} "
    f"{'tri/fi':>7} {'TFLOP/s':>9} {'max|d|':>8}"
)
for name, (nq, nkv, d, win) in GEOM.items():
    for m in MS:
        kv_len = m if win < 0 else min(m, win)
        a = build(nq, nkv, d, m, m)
        tri = lambda: unwrap(mha_extend_with_kvcache(**a, solution="triton"))
        try:
            fi = lambda: unwrap(mha_extend_with_kvcache(**a, solution="flashinfer_fa2"))
            o_t, o_f = tri(), fi()
            torch.cuda.synchronize()
            diff = (o_t.float() - o_f.float()).abs().max().item()
            d_f = device_us(fi)
        except Exception as exc:
            diff, d_f = float("nan"), float("nan")
            print(
                f"  (flashinfer unavailable at {name} M={m}: "
                f"{type(exc).__name__}: {str(exc)[:50]})"
            )
        d_t = device_us(tri)
        # causal attention over the effective KV window
        flop = 2 * 2 * m * kv_len * nq * d / 2
        print(
            f"{name:>8} {m:6d} {kv_len:7d} {d_t:10.1f} {d_f:9.1f} "
            f"{d_t / d_f if d_f == d_f else float('nan'):6.2f}x "
            f"{flop / (d_t * 1e-6) / 1e12:8.1f} {diff:8.2e}"
        )
