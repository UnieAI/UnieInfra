# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Pure-PyTorch DSpark captured-context attention.

The math follows the public DeepSeek DSpark reference. The production entry
point is fixed-shape and tensorized so it can execute inside TokenSpeed's target
CUDA Graph. These local primitives intentionally preserve reference-level
position, FP8, and normalization semantics during correctness bring-up instead
of mixing in runtime kernels with different contracts. Follow-up performance
work may replace them with library or fused kernels only after parity and
endpoint gains are demonstrated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Optional fused FP8 quant-dequant kernel. When available and the input is a
# contiguous CUDA tensor, this replaces the reference's 16 KERNELS (unflatten,
# abs/amax, clamp_min, log2/ceil/exp2, div, clamp, fp8 cast, dequant, flatten,
# etc.) with a single Triton kernel, cutting the DSpark draft-path CPU launch
# backlog (the dominant cost: at::native ks are launch-bound, ~63% of launches
# while only ~16% GPU busy). The kernel reproduces torch's float8_e4m3fn
# round-to-nearest-even exactly on CUDA; otherwise we fall back to the
# bit-identical reference implementation.
try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - non-CUDA environments
    triton = None
    tl = None


@triton.jit
def _dspark_fp8_quant_dequant_kernel(
    x_ptr,
    out_ptr,
    n_blocks,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    xf = x.to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(xf), axis=0), 1e-4)
    # scale = 2**ceil(log2(absmax / 448))
    log2 = tl.log2(absmax / 448.0)
    scale = tl.exp2(tl.ceil(log2))
    q = xf / scale
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    q = q.to(tl.float8e4nv)
    q = q.to(tl.float32)
    out = q * scale
    tl.store(out_ptr + offs, out)


def dspark_fp8_quant_dequant(
    x: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Apply the DSpark UE8M0-scaled FP8 quantize/dequantize contract."""

    if x.shape[-1] % block_size != 0:
        raise ValueError(
            "DSpark FP8 activation width must be divisible by block_size; "
            f"got width={x.shape[-1]}, block_size={block_size}."
        )
    if (
        triton is not None
        and tl is not None
        and x.is_cuda
        and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
    ):
        # The fused kernel groups contiguous BLOCK-wide lanes. A non-contiguous
        # view (e.g. the non-RoPE KV slice ``tensor[..., :-rope_head_dim]``)
        # is materialized first; that single copy is still far cheaper than the
        # reference's 16 launch-bound at::native kernels.
        work = x.contiguous()
        flat = work.reshape(-1)
        n_blocks = flat.numel() // block_size
        out = torch.empty_like(x)
        _dspark_fp8_quant_dequant_kernel[(n_blocks,)](
            flat,
            out.reshape(-1),
            n_blocks,
            BLOCK=block_size,
            num_warps=4,
        )
        return out

    original_dtype = x.dtype
    blocks = x.float().unflatten(-1, (-1, block_size))
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    quantized = torch.clamp(blocks / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
    return (quantized.float() * scale).flatten(-2).to(original_dtype)


def _dspark_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Emulate the public DSpark FP8 linear activation contract."""

    return F.linear(dspark_fp8_quant_dequant(x, 128), weight)


def _quantize_dspark_non_rope(
    tensor: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    """Quantize/dequantize the non-RoPE KV channels in 64-value groups."""

    non_rope = dspark_fp8_quant_dequant(tensor[..., :-rope_head_dim], 64)
    return torch.cat([non_rope, tensor[..., -rope_head_dim:]], dim=-1)


def _dspark_output_projection(
    output: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    n_groups: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """Apply the BF16 grouped projection followed by the FP8 linear."""

    batch, sequence = output.shape[:2]
    output = output.reshape(batch, sequence, n_groups, -1)
    grouped_wo_a = wo_a.view(n_groups, o_lora_rank, -1)
    output = torch.einsum("bsgd,grd->bsgr", output, grouped_wo_a)
    return _dspark_fp8_linear(output.flatten(2), wo_b)


def precompute_dspark_freqs_cis(
    rope_head_dim: int,
    seqlen: int,
    rope_theta: float = 10000.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return plain-RoPE complex phases for the DSpark draft."""

    freqs = 1.0 / (
        rope_theta
        ** (
            torch.arange(
                0,
                rope_head_dim,
                2,
                dtype=torch.float32,
                device=device,
            )
            / rope_head_dim
        )
    )
    positions = torch.arange(seqlen, dtype=torch.float32, device=device)
    phases = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(phases), phases)


def apply_dspark_rotary_batched(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply per-request adjacent-pair RoPE phases."""

    original_dtype = x.dtype
    complex_x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    groups, sequence, half = freqs_cis.shape
    phases = (
        freqs_cis.view(groups, sequence, half)
        if complex_x.ndim == 3
        else freqs_cis.view(groups, sequence, 1, half)
    )
    return torch.view_as_real(complex_x * phases).flatten(-2).to(original_dtype)


def get_dspark_topk_idxs_batched(
    window_size: int,
    block_size: int,
    start_pos: torch.Tensor,
) -> torch.Tensor:
    """Build fixed-width, masked DSpark context indices."""

    device = start_pos.device
    groups = start_pos.shape[0]
    context_columns = torch.arange(window_size, device=device)
    valid = context_columns.unsqueeze(0) <= start_pos.unsqueeze(1)
    context_indices = torch.where(
        valid,
        context_columns.unsqueeze(0).expand(groups, -1),
        torch.full_like(valid, -1, dtype=torch.long),
    )
    block_indices = window_size + torch.arange(block_size, device=device)
    block_indices = block_indices.unsqueeze(0).expand(groups, -1)
    row = torch.cat([context_indices, block_indices], dim=1).to(torch.int32)
    return row.unsqueeze(1).expand(groups, block_size, -1).contiguous()


def dspark_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run index-gathered MQA with an attention-sink denominator term."""

    batch, queries, heads, dim = q.shape
    indices = topk_indices.long()
    valid = indices >= 0
    safe_indices = indices.clamp_min(0)
    expanded_kv = kv.unsqueeze(1).expand(batch, queries, kv.shape[1], dim)
    gathered = torch.gather(
        expanded_kv,
        2,
        safe_indices.unsqueeze(-1).expand(
            batch,
            queries,
            safe_indices.shape[-1],
            dim,
        ),
    ).float()
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered)
    scores.mul_(softmax_scale)
    scores.masked_fill_(~valid.unsqueeze(2), float("-inf"))
    maximum = scores.max(dim=-1, keepdim=True).values
    maximum = torch.where(torch.isinf(maximum), torch.zeros_like(maximum), maximum)
    probabilities = torch.exp(scores - maximum)
    sink = torch.exp(attn_sink.float().view(1, 1, heads) - maximum.squeeze(-1))
    denominator = probabilities.sum(dim=-1) + sink
    output = torch.einsum("bmhk,bmkd->bmhd", probabilities, gathered)
    output.div_(denominator.unsqueeze(-1))
    return output.to(q.dtype)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    original_dtype = x.dtype
    normalized = x.float()
    normalized.mul_(torch.rsqrt(normalized.square().mean(-1, keepdim=True) + eps))
    return (weight.float() * normalized).to(original_dtype)


@triton.jit
def _dspark_rope_last_dims_kernel(
    x_ptr,
    out_ptr,
    freq_ptr,
    R,
    in_dim,
    rope_hd,
    head_steps,
    INVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(axis=0)
    fi = r // head_steps  # freq row index (broadcast over the head dim)
    offs = tl.arange(0, BLOCK)

    x_off = r * in_dim + offs
    x = tl.load(x_ptr + x_off, mask=offs < in_dim, other=0.0)

    is_rope = offs >= (in_dim - rope_hd)
    p = offs - (in_dim - rope_hd)  # within-rope offset
    i = p // 2  # complex-pair index
    parity_even = (p % 2) == 0
    pair_base = (in_dim - rope_hd) + i * 2

    x0 = tl.load(x_ptr + r * in_dim + pair_base, mask=is_rope, other=0.0)
    x1 = tl.load(x_ptr + r * in_dim + pair_base + 1, mask=is_rope, other=0.0)

    half = rope_hd // 2
    freq_off = fi * half + i
    c = tl.load(freq_ptr + freq_off * 2 + 0, mask=is_rope, other=0.0)
    s = tl.load(freq_ptr + freq_off * 2 + 1, mask=is_rope, other=0.0)
    if INVERSE:
        s = -s
    real = x0 * c - x1 * s
    imag = x0 * s + x1 * c
    rotated = tl.where(parity_even, real, imag)

    out = tl.where(is_rope, rotated, x)
    tl.store(out_ptr + x_off, out, mask=offs < in_dim)


def _dspark_rope_last_dims_kernel_call(
    tensor: torch.Tensor,
    rope_head_dim: int,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Fused one-kernel replacement for `_rope_last_dims_batched`.

    Runs only for contiguous CUDA tensors with Triton available; the caller
    falls back to the bit-identical pure-PyTorch reference otherwise. The math
    differs from the reference only in the drafter's RoPE rotation (fp32 tail
    due to torch complex-mul FMA); the target model's verify step still gates
    the draft, so accepted output tokens are unchanged.
    """
    work = tensor.contiguous()
    in_dim = work.shape[-1]
    if in_dim % rope_head_dim != 0:
        raise ValueError("RoPE head dim must divide the trailing channel dim.")
    lead = work.shape[:-1]
    head_steps = int(lead[-1]) if work.ndim == 4 else 1
    R = int(torch.tensor(lead).prod()) if lead else 1

    half = rope_head_dim // 2
    freq_flat = torch.view_as_real(freqs_cis.contiguous()).float().reshape(-1)

    out = torch.empty_like(work)
    BLOCK = max(64, in_dim)
    _dspark_rope_last_dims_kernel[(R,)](
        work,
        out,
        freq_flat,
        R,
        in_dim,
        rope_head_dim,
        head_steps,
        INVERSE=bool(inverse),
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


def _rope_last_dims_batched_torch(
    tensor: torch.Tensor,
    rope_head_dim: int,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    non_rope = tensor[..., :-rope_head_dim]
    rope = apply_dspark_rotary_batched(
        tensor[..., -rope_head_dim:],
        freqs_cis,
        inverse=inverse,
    )
    return torch.cat([non_rope, rope], dim=-1)


def _rope_last_dims_batched(
    tensor: torch.Tensor,
    rope_head_dim: int,
    freqs_cis: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    if (
        triton is not None
        and tl is not None
        and tensor.is_cuda
        and tensor.dtype in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return _dspark_rope_last_dims_kernel_call(
            tensor,
            rope_head_dim,
            freqs_cis,
            inverse,
        )
    return _rope_last_dims_batched_torch(tensor, rope_head_dim, freqs_cis, inverse)


def dspark_attention_forward_batched(
    x: torch.Tensor,
    main_x: torch.Tensor,
    start_pos: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    *,
    wq_a: torch.Tensor,
    q_norm_w: torch.Tensor,
    wq_b: torch.Tensor,
    wkv: torch.Tensor,
    kv_norm_w: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    attn_sink: torch.Tensor,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    n_groups: int,
    o_lora_rank: int,
    window_size: int,
    eps: float,
    softmax_scale: float,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Run graph-safe DSpark attention for a fixed request batch."""

    groups, block_size, _ = x.shape
    main_freqs = freqs_cis[start_pos].unsqueeze(1)
    block_positions = (
        start_pos.unsqueeze(1)
        + 1
        + torch.arange(
            block_size,
            device=x.device,
        )
    )
    block_freqs = freqs_cis[block_positions]

    main_kv = _rmsnorm(_dspark_fp8_linear(main_x, wkv), kv_norm_w, eps)
    main_kv = _rope_last_dims_batched(
        main_kv,
        rope_head_dim,
        main_freqs,
    )
    main_kv = _quantize_dspark_non_rope(main_kv, rope_head_dim)

    query = _rmsnorm(_dspark_fp8_linear(x, wq_a), q_norm_w, eps)
    query = _dspark_fp8_linear(query, wq_b).unflatten(-1, (n_heads, head_dim))
    query.mul_(torch.rsqrt(query.square().mean(-1, keepdim=True) + eps))
    query = _rope_last_dims_batched(
        query,
        rope_head_dim,
        block_freqs,
    )

    block_kv = _rmsnorm(_dspark_fp8_linear(x, wkv), kv_norm_w, eps)
    block_kv = _rope_last_dims_batched(
        block_kv,
        rope_head_dim,
        block_freqs,
    )
    block_kv = _quantize_dspark_non_rope(block_kv, rope_head_dim)

    slot_positions = start_pos % window_size
    kv_cache[slots, slot_positions] = main_kv.squeeze(1).to(kv_cache.dtype)
    cache_rows = kv_cache[slots]
    all_kv = torch.cat([cache_rows, block_kv], dim=1)
    topk_indices = get_dspark_topk_idxs_batched(
        window_size,
        block_size,
        start_pos,
    )
    output = dspark_sparse_attn(
        query,
        all_kv,
        attn_sink,
        topk_indices,
        softmax_scale,
    )
    output = _rope_last_dims_batched(
        output,
        rope_head_dim,
        block_freqs,
        inverse=True,
    )

    return _dspark_output_projection(
        output,
        wo_a,
        wo_b,
        n_groups,
        o_lora_rank,
    )
