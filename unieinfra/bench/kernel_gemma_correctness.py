"""Are gemma-4's attention kernels numerically right, in isolation?

The engine answers gemma-4 with garbage ('額LLLLLLL額額額額...'). Two candidate
causes were found by reading and one was fixed -- mha.py never passed
softmax_scale, so every kernel used 1/sqrt(head_dim) where gemma-4 wants 1.0 --
and the output changed but stayed wrong. The rope was checked next and is
right: build_rope_inv_freq reproduces HF's _compute_proportional_rope_parameters
including the zero-frequency tail.

So take the model out of it. Build gemma-4's exact attention geometries, run
every kernel the registry offers for them, and compare against
scaled_dot_product_attention on the same tensors. A kernel that disagrees with
SDPA is the bug; if they all agree, the bug is upstream of attention and the
next probe is the model's own hidden states.

    sliding  32 q x 16 kv x 256, window 1024
    global   32 q x  4 kv x 512, no window

Both are run with softmax_scale=1.0, which is gemma-4's actual scale
(HF modeling_gemma4.py:1194), not the 1/sqrt(d) a kernel defaults to -- a
kernel that silently ignores the argument shows up here as a large diff.

    python3 unieinfra/bench/kernel_gemma_correctness.py
"""

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
)

PAGE = 64
SCALE = 1.0
GEOM = {  # name: (num_q_heads, num_kv_heads, head_dim, window)
    "sliding": (32, 16, 256, 1024),
    "global": (32, 4, 512, -1),
}
DECODE_SOLUTIONS = ["triton", "flashinfer_trtllm", None]
EXTEND_SOLUTIONS = ["triton", "flashinfer_paged", None]


def gather(cache, page_table, seqlen):
    """[pages, PAGE, H, D] + [1, npages] -> [seqlen, H, D]"""
    pages = page_table[0].tolist()
    rows = torch.cat([cache[p] for p in pages], dim=0)
    return rows[:seqlen]


def ref_decode(q, k, v, window):
    """q [1, Hq, D]; k/v [S, Hkv, D] -> [1, Hq, D], SDPA with scale=1.0."""
    s, hkv, d = k.shape
    hq = q.shape[1]
    rep = hq // hkv
    kk = k.repeat_interleave(rep, dim=1).permute(1, 0, 2).unsqueeze(0).float()
    vv = v.repeat_interleave(rep, dim=1).permute(1, 0, 2).unsqueeze(0).float()
    qq = q.permute(1, 0, 2).unsqueeze(0).float()
    mask = None
    if window >= 0 and s > window + 1:
        mask = torch.zeros(1, 1, 1, s, device=q.device, dtype=torch.float32)
        mask[..., : s - (window + 1)] = float("-inf")
    out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=SCALE)
    return out.squeeze(0).permute(1, 0, 2)


def ref_extend(q, k, v, window):
    """q [M, Hq, D]; k/v [S, Hkv, D] with the last M rows being q's own."""
    s, hkv, d = k.shape
    m, hq, _ = q.shape
    rep = hq // hkv
    kk = k.repeat_interleave(rep, dim=1).permute(1, 0, 2).unsqueeze(0).float()
    vv = v.repeat_interleave(rep, dim=1).permute(1, 0, 2).unsqueeze(0).float()
    qq = q.permute(1, 0, 2).unsqueeze(0).float()
    qpos = torch.arange(s - m, s, device=q.device).view(m, 1)
    kpos = torch.arange(s, device=q.device).view(1, s)
    allow = kpos <= qpos
    if window >= 0:
        allow &= kpos > qpos - (window + 1)
    mask = torch.where(allow, 0.0, float("-inf")).view(1, 1, m, s)
    out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=SCALE)
    return out.squeeze(0).permute(1, 0, 2)


def unwrap(r):
    return r[0] if isinstance(r, tuple) else r


def run(name, nq, nkv, d, window, ctx, m):
    dev = "cuda"
    npages = (ctx + PAGE - 1) // PAGE
    k_cache = torch.randn(npages + 2, PAGE, nkv, d, dtype=torch.bfloat16, device=dev)
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(npages, dtype=torch.int32, device=dev).view(1, -1)
    k = gather(k_cache, page_table, ctx)
    v = gather(v_cache, page_table, ctx)

    print(f"\n## {name}: {nq}q x {nkv}kv x {d}, window {window}, ctx {ctx}")

    q1 = torch.randn(1, nq, d, dtype=torch.bfloat16, device=dev)
    r = ref_decode(q1, k, v, window)
    for sol in DECODE_SOLUTIONS:
        try:
            o = unwrap(
                mha_decode_with_kvcache(
                    q=q1, k_cache=k_cache, v_cache=v_cache, page_table=page_table,
                    cache_seqlens=torch.tensor([ctx], dtype=torch.int32, device=dev),
                    max_seqlen_k=ctx, max_seqlen_q=1, window_left=window,
                    softmax_scale=SCALE, solution=sol,
                )
            ).float().view(1, nq, d)
            diff = (o - r).abs().max().item()
            rel = diff / r.abs().max().item()
            print(f"  decode  solution={str(sol):18s} max|d|={diff:9.3e}  rel={rel:8.2e}"
                  f"  {'OK' if rel < 5e-2 else '<-- WRONG'}")
        except Exception as exc:  # noqa: BLE001
            print(f"  decode  solution={str(sol):18s} {type(exc).__name__}: {str(exc)[:60]}")

    qm = torch.randn(m, nq, d, dtype=torch.bfloat16, device=dev)
    rm = ref_extend(qm, k, v, window)
    for sol in EXTEND_SOLUTIONS:
        try:
            o = unwrap(
                mha_extend_with_kvcache(
                    q=qm,
                    cu_seqlens_q=torch.tensor([0, m], dtype=torch.int32, device=dev),
                    cu_seqlens_kv=torch.tensor([0, ctx], dtype=torch.int32, device=dev),
                    k_cache=k_cache, v_cache=v_cache, page_table=page_table,
                    cache_seqlens=torch.tensor([ctx], dtype=torch.int32, device=dev),
                    max_seqlen_q=m, max_seqlen_k=ctx, is_causal=True,
                    window_left=window, softmax_scale=SCALE, solution=sol,
                )
            ).float().view(m, nq, d)
            diff = (o - rm).abs().max().item()
            rel = diff / rm.abs().max().item()
            print(f"  extend  solution={str(sol):18s} max|d|={diff:9.3e}  rel={rel:8.2e}"
                  f"  {'OK' if rel < 5e-2 else '<-- WRONG'}")
        except Exception as exc:  # noqa: BLE001
            print(f"  extend  solution={str(sol):18s} {type(exc).__name__}: {str(exc)[:60]}")


def main():
    torch.manual_seed(0)
    for name, (nq, nkv, d, win) in GEOM.items():
        for ctx in (512, 2048):
            run(name, nq, nkv, d, win, ctx, m=min(256, ctx))


if __name__ == "__main__":
    main()
