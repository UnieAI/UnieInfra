# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#
# DeepSeek V4 attention helpers keep runtime validation here; production Triton
# kernels live under tokenspeed-kernel ops.

"""DeepSeek V4 attention kernel boundaries.

Keep the model layer independent from the CUDA extension import details. The
runtime requires TokenSpeed's own built DeepSeek V4 attention op.
"""

from __future__ import annotations

import math
import struct

import torch
from tokenspeed_kernel.ops.attention.cuda.deepseek_v4 import (
    fused_qnorm_rope_kv_insert as _cuda_fused_qnorm_rope_kv_insert,
)
from tokenspeed_kernel.ops.attention.cuda.deepseek_v4 import (
    has_fused_qnorm_rope_kv_insert as _cuda_has_fused_qnorm_rope_kv_insert,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_build_dense_prefill_local_compressed_indices,
    deepseek_v4_combine_dense_swa_indices,
    deepseek_v4_combine_topk_swa_indices,
    deepseek_v4_compressed_slot_mapping,
    deepseek_v4_compute_global_topk_indices_and_lens,
    deepseek_v4_decode_swa_indices_and_lens,
    deepseek_v4_dequantize_and_gather_k_cache,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_fused_csa_indexer_mxfp4_cache_insert as _triton_fused_csa_indexer_mxfp4_cache_insert,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_fused_csa_indexer_fp8_cache_insert as _triton_fused_csa_indexer_fp8_cache_insert,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_fused_indexer_q_rope_hadamard_mxfp4 as _triton_fused_indexer_q_rope_hadamard_mxfp4,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_fused_inv_rope_fp8_quant,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_fused_sparse_compress_cache_insert as _triton_fused_sparse_compress_cache_insert,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_save_compressor_state as _triton_save_compressor_state,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    write_deepseek_v4_indexer_mxfp4_cache_cuda as _triton_write_indexer_mxfp4_cache_cuda,
)
from tokenspeed_kernel.ops.transform import hadamard_transform

from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4_cache_spec import (
    DEEPSEEK_V4_FP8_MAX,
    DEEPSEEK_V4_FP8_QUANT_BLOCK,
    DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
    deepseek_v4_indexer_fp8_layout_from_row_bytes,
    deepseek_v4_indexer_fp8_row_bytes,
    deepseek_v4_indexer_fp8_scale_bytes,
    deepseek_v4_indexer_mxfp4_layout_from_row_bytes,
    deepseek_v4_indexer_mxfp4_row_bytes,
    deepseek_v4_nope_dim,
    deepseek_v4_swa_row_bytes,
    deepseek_v4_swa_scale_dim,
    deepseek_v4_swa_token_stride,
)

__all__ = (
    "deepseek_v4_build_dense_prefill_local_compressed_indices",
    "deepseek_v4_combine_dense_swa_indices",
    "deepseek_v4_combine_topk_swa_indices",
    "deepseek_v4_compressed_slot_mapping",
    "deepseek_v4_compute_global_topk_indices_and_lens",
    "deepseek_v4_decode_swa_indices_and_lens",
    "deepseek_v4_dequantize_and_gather_k_cache",
    "deepseek_v4_fused_inv_rope_fp8_quant",
    "deepseek_v4_csa_compress_kv_cache_insert",
    "deepseek_v4_csa_indexer_cache_insert",
    "deepseek_v4_hca_compress_kv_cache_insert",
    "deepseek_v4_prepare_indexer_q_fp8",
    "deepseek_v4_prepare_indexer_q_mxfp4",
    "dequantize_deepseek_v4_fp8_ds_mla_cache",
    "gather_paged_indexer_fp8_cache",
    "fused_qnorm_rope_kv_insert",
    "read_deepseek_v4_indexer_fp8_cache",
    "read_deepseek_v4_indexer_mxfp4_cache",
    "save_deepseek_v4_compressor_state",
    "write_deepseek_v4_indexer_fp8_cache",
    "write_deepseek_v4_indexer_mxfp4_cache",
)


def _indexer_mxfp4_layout_from_cache(
    cache_2d: torch.Tensor,
    block_size: int,
) -> tuple[int, int, int]:
    if cache_2d.dim() != 2:
        raise ValueError(f"cache_2d must be 2-D, got {tuple(cache_2d.shape)}")
    row_bytes = cache_2d.shape[1] // block_size
    if cache_2d.shape[1] % block_size != 0:
        raise ValueError(
            "MXFP4 indexer cache row size must match value+scale layout, "
            f"got cache shape {tuple(cache_2d.shape)} and block_size={block_size}"
        )
    return deepseek_v4_indexer_mxfp4_layout_from_row_bytes(row_bytes)


def _indexer_fp8_layout_from_cache(
    cache_2d: torch.Tensor,
    block_size: int,
) -> tuple[int, int]:
    if cache_2d.dim() != 2:
        raise ValueError(f"cache_2d must be 2-D, got {tuple(cache_2d.shape)}")
    row_bytes = cache_2d.shape[1] // block_size
    if cache_2d.shape[1] % block_size != 0:
        raise ValueError(
            "FP8 indexer cache row size must match value+scale layout, "
            f"got cache shape {tuple(cache_2d.shape)} and block_size={block_size}"
        )
    return deepseek_v4_indexer_fp8_layout_from_row_bytes(row_bytes)


def fused_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_eps: float,
    block_size: int,
) -> None:
    """Run the DeepSeek V4 fused SWA cache insert op.

    Expected contract:
    - q: [tokens, local_heads, 512], mutated in place by RMSNorm/RoPE
    - kv: [tokens, 512], source KV latent before RoPE/quant insert
    - swa_kv_cache_2d: uint8 cache blocks flattened as [num_blocks, block_bytes]
    - slot_mapping: output token slots in the paged SWA cache
    - positions: absolute token positions
    """

    if not _cuda_has_fused_qnorm_rope_kv_insert():
        raise RuntimeError(
            "DeepSeek V4 fused SWA cache insert op "
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert is unavailable. "
            "Build `tokenspeed-kernel/python` so the deepseek_v4_attention CUDA "
            "library is present before running this path."
        )

    _cuda_fused_qnorm_rope_kv_insert(
        q,
        kv,
        swa_kv_cache_2d,
        slot_mapping,
        positions.to(torch.int64),
        cos_sin_cache,
        rms_norm_eps,
        block_size,
    )


def _apply_gptj_rope_tail_rows(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    out = x.float().clone()
    half_rope = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    cos = cos_sin_cache[positions.long(), :half_rope].float()
    sin = cos_sin_cache[positions.long(), half_rope:rope_dim].float()
    even = out[..., nope_dim::2].clone()
    odd = out[..., nope_dim + 1 :: 2].clone()
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    out[..., nope_dim::2] = even * cos - odd * sin
    out[..., nope_dim + 1 :: 2] = even * sin + odd * cos
    return out


def _fp8_e4m3_pow2_bytes(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = max(float(block.detach().abs().max()) / DEEPSEEK_V4_FP8_MAX, 1.0e-10)
    scale = 2.0 ** math.ceil(math.log2(scale))
    scaled = torch.clamp(block / scale, -DEEPSEEK_V4_FP8_MAX, DEEPSEEK_V4_FP8_MAX)
    return scaled.to(torch.float8_e4m3fn).view(torch.uint8), block.new_tensor(scale)


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    table = nibbles.new_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    magnitude = table[(nibbles & 0x7).long()]
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    return magnitude * sign


def _deepseek_v4_hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    rotated = hadamard_transform(
        x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous(),
        scale=shape[-1] ** -0.5,
    )
    return rotated.reshape(shape)


def dequantize_deepseek_v4_fp8_ds_mla_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
    *,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Dequantize DeepSeek V4 `fp8_ds_mla` rows selected by global slots."""

    nope_dim = deepseek_v4_nope_dim(head_dim, rope_dim)
    token_stride = deepseek_v4_swa_token_stride(head_dim, rope_dim)
    scale_dim = deepseek_v4_swa_scale_dim(head_dim, rope_dim)
    min_stride = block_size * (token_stride + scale_dim)
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out_shape = (slot_mapping.numel(), head_dim)
    if slot_mapping.numel() == 0:
        return torch.empty(out_shape, device=cache_2d.device, dtype=torch.bfloat16)

    num_nope_blocks = nope_dim // DEEPSEEK_V4_FP8_QUANT_BLOCK

    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    pages = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos = safe_slots % block_size
    # Index pages via advanced indexing on the 2-D cache, NOT via
    # page * stride(0) into reshape(-1): on a strided field view of a larger
    # LCM arena, reshape(-1) copies only the logical elements and
    # physical-stride offsets read past its end.
    value_base = pos * token_stride
    scale_base = block_size * token_stride + pos * scale_dim

    value_offsets = (
        value_base[:, None]
        + torch.arange(token_stride, device=cache_2d.device, dtype=torch.int64)[None, :]
    )
    row_bytes = cache_2d[pages[:, None], value_offsets]
    nope = row_bytes[:, :nope_dim].contiguous().view(torch.float8_e4m3fn)

    scale_offsets = (
        scale_base[:, None]
        + torch.arange(num_nope_blocks, device=cache_2d.device, dtype=torch.int64)[
            None, :
        ]
    )
    scales = torch.pow(
        2.0, cache_2d[pages[:, None], scale_offsets].to(torch.int32) - 127
    )
    scales = scales.float().repeat_interleave(DEEPSEEK_V4_FP8_QUANT_BLOCK, dim=1)

    rope = row_bytes[:, nope_dim:token_stride].contiguous()
    out = torch.cat([nope.float() * scales, rope.view(torch.bfloat16).float()], dim=1)
    out = out.to(torch.bfloat16)
    return torch.where(valid[:, None], out, torch.zeros_like(out))


def deepseek_v4_prepare_indexer_q_mxfp4(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Apply indexer Q RoPE and return DeepGEMM-ready MXFP4 values/scales."""

    if index_q.dim() != 3:
        raise ValueError(f"index_q must be [tokens, heads, dim], got {index_q.shape}")
    if index_q.shape[-1] % DEEPSEEK_V4_MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            "MXFP4 index_q dim must be divisible by "
            f"{DEEPSEEK_V4_MXFP4_BLOCK_SIZE}, got {index_q.shape[-1]}"
        )
    rope_dim = int(cos_sin_cache.shape[-1])
    if index_q.shape[-1] <= rope_dim:
        raise ValueError(
            f"index_q dim must be larger than rope_dim={rope_dim}, got {index_q.shape}"
        )
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    if weights.shape != index_q.shape[:2]:
        raise ValueError(f"weights must be [tokens, heads], got {tuple(weights.shape)}")
    if not index_q.is_cuda:
        raise ValueError(
            "deepseek_v4_prepare_indexer_q_mxfp4 only supports CUDA tensors."
        )
    return _triton_fused_indexer_q_rope_hadamard_mxfp4(
        index_q=index_q,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=weights,
        softmax_scale=softmax_scale,
        head_scale=head_scale,
    )


def deepseek_v4_prepare_indexer_q_fp8(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply indexer Q RoPE + Hadamard and return DeepGEMM-ready FP8 values.

    Non-Blackwell (SM90) analogue of ``deepseek_v4_prepare_indexer_q_mxfp4``.
    ``deep_gemm.fp8_paged_mqa_logits`` / ``fp8_mqa_logits`` consume the query as
    a plain ``float8_e4m3fn`` tensor (no per-token query scale) with the softmax
    and head scales folded into ``weights`` -- mirroring SGLang's
    ``fused_q_indexer_rope_hadamard_quant``.

    Args:
        index_q: ``[tokens, heads, index_head_dim]`` bf16/fp32 indexer queries.
        positions: ``[tokens]`` absolute token positions for RoPE.
        cos_sin_cache: ``[max_pos, rope_dim]`` fused cos/sin cache.
        weights: ``[tokens, heads]`` (or trailing singleton) indexer weights.
        softmax_scale: ``index_head_dim**-0.5`` attention softmax scale.
        head_scale: ``n_head**-0.5`` head normalization scale.

    Returns:
        Tuple of ``(q_fp8, weights_out)`` where ``q_fp8`` is
        ``[tokens, heads, index_head_dim]`` ``float8_e4m3fn`` and
        ``weights_out`` is ``[tokens, heads]`` float32 with both scales folded
        in.
    """

    if index_q.dim() != 3:
        raise ValueError(f"index_q must be [tokens, heads, dim], got {index_q.shape}")
    rope_dim = int(cos_sin_cache.shape[-1])
    if index_q.shape[-1] <= rope_dim:
        raise ValueError(
            f"index_q dim must be larger than rope_dim={rope_dim}, got {index_q.shape}"
        )
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    if weights.shape != index_q.shape[:2]:
        raise ValueError(f"weights must be [tokens, heads], got {tuple(weights.shape)}")
    if not index_q.is_cuda:
        raise ValueError(
            "deepseek_v4_prepare_indexer_q_fp8 only supports CUDA tensors."
        )

    weights_out = (weights.float() * float(softmax_scale) * float(head_scale)).float()
    if index_q.shape[0] == 0:
        q_fp8 = index_q.new_empty(index_q.shape, dtype=torch.float8_e4m3fn)
        return q_fp8, weights_out

    rotated = _apply_gptj_rope_tail_rows(
        index_q,
        positions,
        cos_sin_cache,
        rope_dim,
    )
    rotated = _deepseek_v4_hadamard_rotate(rotated)
    q_fp8 = rotated.to(torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
    return q_fp8, weights_out


def gather_paged_indexer_fp8_cache(
    cache_2d: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ragged FP8 indexer keys for ``deep_gemm.fp8_mqa_logits``.

    Torch analogue of the CUDA MXFP4 paged gather. Produces the fp8 keys and
    their fp32 per-token scales laid out contiguously in ``cu_seq_lens`` order.

    Args:
        cache_2d: ``[pages, block_size * (index_head_dim + 4)]`` uint8 fp8 cache.
        block_table: ``[num_reqs, max_blocks]`` int32 logical->physical pages.
        cu_seq_lens: ``[num_reqs + 1]`` int32 cumulative compressed key lengths.
        block_size: cache block size (tokens per page), e.g. 64.

    Returns:
        Tuple ``(k_fp8, k_scale)`` with ``k_fp8`` of shape ``[total_k, dim]``
        ``float8_e4m3fn`` and ``k_scale`` of shape ``[total_k]`` float32.
    """

    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    index_head_dim, scale_bytes = _indexer_fp8_layout_from_cache(cache_2d, block_size)
    device = cache_2d.device
    cu_seq_lens = cu_seq_lens.to(device=device, dtype=torch.int64)
    total_rows = int(cu_seq_lens[-1].item()) if cu_seq_lens.numel() else 0
    if total_rows == 0:
        return (
            torch.empty((0, index_head_dim), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    block_table = block_table.to(device=device, dtype=torch.int64)
    row_ids = torch.arange(total_rows, device=device, dtype=torch.int64)
    # searchsorted over the per-req end offsets maps each output row to its req.
    req = torch.searchsorted(cu_seq_lens[1:].contiguous(), row_ids, right=True)
    req = req.clamp_max(block_table.shape[0] - 1)
    local = row_ids - cu_seq_lens[req]
    logical_block = torch.div(local, block_size, rounding_mode="floor")
    logical_block = logical_block.clamp_max(block_table.shape[1] - 1)
    in_block = local % block_size
    phys = block_table[req, logical_block]

    # Index pages via advanced indexing on the 2-D cache, NOT via
    # page * stride(0) into reshape(-1): the indexer cache can be a strided
    # field view of a larger LCM arena (stride(0) > shape[1]), where a
    # flattened view only covers the logical elements and physical-stride
    # offsets read past its end.
    value_offsets = (
        in_block[:, None] * index_head_dim
        + torch.arange(index_head_dim, device=device, dtype=torch.int64)[None, :]
    )
    scale_offsets = (
        block_size * index_head_dim
        + in_block[:, None] * scale_bytes
        + torch.arange(scale_bytes, device=device, dtype=torch.int64)[None, :]
    )
    k_fp8 = (
        cache_2d[phys[:, None], value_offsets].contiguous().view(torch.float8_e4m3fn)
    )
    k_scale = (
        cache_2d[phys[:, None], scale_offsets]
        .contiguous()
        .view(torch.float32)
        .reshape(total_rows)
    )
    return k_fp8, k_scale


def _fp8_ds_mla_cache_rows(
    normed: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_rows = normed.shape[0]
    head_dim = int(normed.shape[-1])
    rope_dim = int(cos_sin_cache.shape[-1])
    nope_dim = deepseek_v4_nope_dim(head_dim, rope_dim)
    scale_dim = deepseek_v4_swa_scale_dim(head_dim, rope_dim)
    quant_input = normed.to(torch.bfloat16).float()
    nope_blocks = quant_input[:, :nope_dim].reshape(
        num_rows,
        nope_dim // DEEPSEEK_V4_FP8_QUANT_BLOCK,
        DEEPSEEK_V4_FP8_QUANT_BLOCK,
    )
    absmax = nope_blocks.detach().abs().amax(dim=-1).clamp_min(1.0e-4)
    exponent = torch.ceil(torch.log2(absmax / DEEPSEEK_V4_FP8_MAX))
    scaled = torch.clamp(
        nope_blocks * torch.pow(2.0, -exponent).unsqueeze(-1),
        -DEEPSEEK_V4_FP8_MAX,
        DEEPSEEK_V4_FP8_MAX,
    )
    value_bytes = (
        scaled.to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(
            num_rows,
            nope_dim,
        )
    )
    scale_bytes = torch.clamp(exponent + 127.0, 0.0, 255.0).to(torch.uint8)
    scale_pad = scale_dim - scale_bytes.shape[1]
    if scale_pad > 0:
        scale_bytes = torch.cat(
            [scale_bytes, torch.zeros_like(scale_bytes[:, :scale_pad])],
            dim=-1,
        )

    compressed_positions = (
        torch.div(positions.to(torch.int64), compress_ratio, rounding_mode="floor")
        * compress_ratio
    )
    rotated = _apply_gptj_rope_tail_rows(
        normed,
        compressed_positions,
        cos_sin_cache,
        rope_dim,
    ).to(torch.bfloat16)
    rope_bytes = rotated[:, nope_dim:].contiguous().view(torch.uint8)
    rope_bytes = rope_bytes.reshape(num_rows, rope_dim * 2)
    return value_bytes, scale_bytes, rope_bytes


def _write_fp8_ds_mla_cache_rows_capturable(
    normed: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    valid: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
) -> None:
    num_rows = normed.shape[0]
    if num_rows == 0:
        return

    head_dim = int(normed.shape[-1])
    rope_dim = int(cos_sin_cache.shape[-1])
    nope_dim = deepseek_v4_nope_dim(head_dim, rope_dim)
    token_stride = deepseek_v4_swa_token_stride(head_dim, rope_dim)
    scale_dim = deepseek_v4_swa_scale_dim(head_dim, rope_dim)
    slots = kv_slot_mapping[:num_rows].to(torch.int64)
    valid = valid[:num_rows] & (slots >= 0)
    if not (slots.is_cuda and torch.cuda.is_current_stream_capturing()):
        if not bool(valid.any()):
            return
        normed = normed[:num_rows][valid]
        positions = positions[:num_rows][valid]
        slots = slots[valid]
        valid = torch.ones_like(slots, dtype=torch.bool)
        num_rows = slots.numel()
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    block_idx = torch.div(safe_slots, kv_cache_block_size, rounding_mode="floor")
    pos_in_block = safe_slots % kv_cache_block_size
    token_base = pos_in_block * token_stride
    scale_base = kv_cache_block_size * token_stride + pos_in_block * scale_dim

    value_bytes, scale_bytes, rope_bytes = _fp8_ds_mla_cache_rows(
        normed[:num_rows], positions[:num_rows], cos_sin_cache, compress_ratio
    )
    # Index blocks via advanced indexing on the 2-D cache, NOT via
    # block * stride(0) into reshape(-1): the cache can be a strided field
    # view of a larger LCM arena (stride(0) > shape[1]), where a flattened
    # view only covers the logical elements and physical-stride offsets run
    # past its end (or corrupt neighboring fields).
    value_offsets = (
        token_base[:, None]
        + torch.arange(
            nope_dim,
            device=kv_cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            scale_dim,
            device=kv_cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    rope_offsets = (
        token_base[:, None]
        + nope_dim
        + torch.arange(
            rope_dim * 2,
            device=kv_cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    kv_cache_2d[block_idx[:, None], value_offsets] = value_bytes
    kv_cache_2d[block_idx[:, None], scale_offsets] = scale_bytes
    kv_cache_2d[block_idx[:, None], rope_offsets] = rope_bytes


def save_deepseek_v4_compressor_state(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    compress_ratio: int,
) -> None:
    """Save DeepSeek V4 compressor residual state into paged SWA-style cache.

    This correctness-first state write packs `[kv_state, score_state]`, each
    with width `coff * head_dim`; score state includes the APE row selected by
    `position % compress_ratio`.
    """

    if kv.shape != score.shape:
        raise ValueError(
            f"kv and score shapes must match, got {kv.shape} vs {score.shape}"
        )
    if kv.dim() != 2:
        raise ValueError(f"kv/score must be [tokens, state_width], got {kv.shape}")
    if state_cache.dim() != 3:
        raise ValueError(
            "state_cache must be [blocks, block_size, 2 * state_width], "
            f"got {state_cache.shape}"
        )
    if block_size != state_cache.shape[1]:
        raise ValueError(
            f"block_size={block_size} does not match "
            f"state_cache.shape[1]={state_cache.shape[1]}"
        )
    state_width = kv.shape[-1]
    if state_cache.shape[-1] != state_width * 2:
        raise ValueError(
            f"state_cache last dim must be {state_width * 2}, "
            f"got {state_cache.shape[-1]}"
        )
    if ape.shape != (compress_ratio, state_width):
        raise ValueError(
            f"ape must be [{compress_ratio}, {state_width}], got {tuple(ape.shape)}"
        )

    num_actual = min(slot_mapping.numel(), kv.shape[0])
    if num_actual == 0:
        return
    if not state_cache.is_cuda:
        raise ValueError(
            "save_deepseek_v4_compressor_state only supports CUDA tensors."
        )

    _triton_save_compressor_state(
        kv=kv,
        score=score,
        ape=ape,
        state_cache=state_cache,
        slot_mapping=slot_mapping,
        positions=positions,
        block_size=block_size,
        compress_ratio=compress_ratio,
    )


def write_deepseek_v4_indexer_fp8_cache(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> None:
    """Write FP8 indexer keys using `[values | fp32 scale]` page layout."""

    if index_k.dim() != 2:
        raise ValueError(f"index_k must be [tokens, dim], got {tuple(index_k.shape)}")
    index_head_dim = int(index_k.shape[-1])
    scale_bytes = deepseek_v4_indexer_fp8_scale_bytes(index_head_dim)
    row_bytes = deepseek_v4_indexer_fp8_row_bytes(index_head_dim)
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    min_stride = block_size * row_bytes
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], "
            f"got {tuple(cache_2d.shape)}"
        )

    num_actual = min(slot_mapping.numel(), index_k.shape[0])
    for token_idx in range(num_actual):
        slot = int(slot_mapping[token_idx].item())
        if slot < 0:
            continue
        page = slot // block_size
        pos = slot % block_size
        # Row-index the 2-D cache (strided-view safe) instead of computing
        # page * stride(0) offsets into reshape(-1).
        page_row = cache_2d[page]
        value_base = pos * index_head_dim
        scale_base = block_size * index_head_dim + pos * scale_bytes
        q_bytes, scale = _fp8_e4m3_pow2_bytes(index_k[token_idx].float())
        page_row[value_base : value_base + index_head_dim].copy_(q_bytes)
        page_row[scale_base : scale_base + scale_bytes].copy_(
            scale.reshape(1).view(torch.uint8)
        )


def write_deepseek_v4_indexer_mxfp4_cache(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> None:
    """Write MXFP4 indexer keys using the `[values | ue8m0 scales]` layout."""

    if index_k.dim() != 2:
        raise ValueError(f"index_k must be [tokens, dim], got {tuple(index_k.shape)}")
    index_head_dim = int(index_k.shape[-1])
    row_bytes = deepseek_v4_indexer_mxfp4_row_bytes(index_head_dim)
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    min_stride = block_size * row_bytes
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    num_actual = min(slot_mapping.numel(), index_k.shape[0])
    if num_actual == 0:
        return
    if not index_k.is_cuda:
        raise ValueError(
            "write_deepseek_v4_indexer_mxfp4_cache only supports CUDA tensors."
        )
    valid = torch.ones(num_actual, device=index_k.device, dtype=torch.bool)
    _triton_write_indexer_mxfp4_cache_cuda(
        index_k[:num_actual],
        cache_2d,
        slot_mapping[:num_actual],
        valid,
        block_size,
    )


def _write_deepseek_v4_indexer_fp8_cache_capturable(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    valid: torch.Tensor,
    block_size: int = 64,
) -> None:
    num_rows = min(slot_mapping.numel(), index_k.shape[0])
    if num_rows == 0:
        return

    index_head_dim = int(index_k.shape[-1])
    scale_bytes = deepseek_v4_indexer_fp8_scale_bytes(index_head_dim)
    _w0 = _time.perf_counter()
    rows = index_k[:num_rows].float()
    scale = (rows.detach().abs().amax(dim=-1) / DEEPSEEK_V4_FP8_MAX).clamp_min(1.0e-10)
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    value_bytes = (
        torch.clamp(
            rows / scale.unsqueeze(-1),
            -DEEPSEEK_V4_FP8_MAX,
            DEEPSEEK_V4_FP8_MAX,
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    _w1 = _time.perf_counter()

    slots = slot_mapping[:num_rows].to(torch.int64)
    valid = valid[:num_rows] & (slots >= 0)
    _w2 = _time.perf_counter()
    synced = False
    if not (slots.is_cuda and torch.cuda.is_current_stream_capturing()):
        if not bool(valid.any()):
            _w3 = _time.perf_counter()
            _ins_write_timing({"scale": _w1 - _w0, "slots": _w2 - _w1, "sync": _w3 - _w2, "compact": 0.0, "scatter": 0.0, "empty": True})
            return
        rows = rows[valid]
        slots = slots[valid]
        scale = scale[valid]
        value_bytes = value_bytes[valid]
        valid = torch.ones_like(slots, dtype=torch.bool)
        num_rows = slots.numel()
        synced = True
    _w3 = _time.perf_counter()
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    pages = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos = safe_slots % block_size

    # Index pages via advanced indexing on the 2-D cache, NOT via
    # page * stride(0) into reshape(-1): the indexer cache can be a strided
    # field view of a larger LCM arena (stride(0) > shape[1]), where a
    # flattened view only covers the logical elements and physical-stride
    # offsets run past its end (or silently corrupt neighboring fields).
    value_offsets = (
        pos[:, None] * index_head_dim
        + torch.arange(
            index_head_dim,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scale_offsets = (
        block_size * index_head_dim
        + pos[:, None] * scale_bytes
        + torch.arange(scale_bytes, device=cache_2d.device, dtype=torch.int64)[None, :]
    )
    _w4 = _time.perf_counter()
    # A captured scatter cannot drop rows, so rows that are masked out are sent
    # to slot 0.  Storing their bytes there would destroy a live slot: `valid` is
    # the compress-boundary mask, so on a decode step all but a handful of the
    # batch are redirected onto that one slot on every replayed step.  Reading
    # the target back and storing it unchanged turns those rows into a no-op --
    # they all gather the same pre-write contents, so whichever one the scatter
    # happens to order last still leaves the slot exactly as it was.
    value_targets = (pages[:, None], value_offsets)
    scale_targets = (pages[:, None], scale_offsets)
    cache_2d[value_targets] = torch.where(
        valid[:, None], value_bytes, cache_2d[value_targets]
    )
    cache_2d[scale_targets] = torch.where(
        valid[:, None],
        scale.view(torch.uint8).reshape(num_rows, scale_bytes),
        cache_2d[scale_targets],
    )
    _w5 = _time.perf_counter()
    _ins_write_timing({
        "scale": _w1 - _w0,
        "slots": _w2 - _w1,
        "sync": _w3 - _w2,
        "compact": _w4 - _w3,
        "scatter": _w5 - _w4,
        "synced": synced,
    })


def read_deepseek_v4_indexer_mxfp4_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Dequantize MXFP4 indexer cache rows selected by `slot_mapping`."""

    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    index_head_dim, value_bytes, scale_bytes = _indexer_mxfp4_layout_from_cache(
        cache_2d, block_size
    )
    min_stride = block_size * (value_bytes + scale_bytes)
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out_shape = (slot_mapping.numel(), index_head_dim)
    if slot_mapping.numel() == 0:
        return torch.empty(out_shape, device=cache_2d.device, dtype=torch.float32)

    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    pages = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos = safe_slots % block_size
    # Index pages via advanced indexing on the 2-D cache, NOT via
    # page * stride(0) into reshape(-1): on a strided field view of a larger
    # LCM arena, reshape(-1) copies only the logical elements and
    # physical-stride offsets read past its end.
    value_base = pos * value_bytes
    scale_base = block_size * value_bytes + pos * scale_bytes

    value_offsets = (
        value_base[:, None]
        + torch.arange(
            value_bytes,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    packed = cache_2d[pages[:, None], value_offsets]

    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            scale_bytes,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scales = torch.pow(
        2.0, cache_2d[pages[:, None], scale_offsets].to(torch.int32) - 127
    )
    byte_scales = scales.float().repeat_interleave(
        DEEPSEEK_V4_MXFP4_BLOCK_SIZE // 2, dim=1
    )

    even = _e2m1_values(packed & 0xF) * byte_scales
    odd = _e2m1_values(packed >> 4) * byte_scales
    out = torch.empty(out_shape, device=cache_2d.device, dtype=torch.float32)
    out[:, 0::2] = even
    out[:, 1::2] = odd
    return torch.where(valid[:, None], out, torch.zeros_like(out))


def read_deepseek_v4_indexer_fp8_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Dequantize FP8 indexer cache rows selected by `slot_mapping`."""

    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    index_head_dim, scale_bytes = _indexer_fp8_layout_from_cache(cache_2d, block_size)
    min_stride = block_size * (index_head_dim + scale_bytes)
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out = torch.zeros(
        slot_mapping.numel(),
        index_head_dim,
        device=cache_2d.device,
        dtype=torch.float32,
    )
    for token_idx, raw_slot in enumerate(slot_mapping.tolist()):
        slot = int(raw_slot)
        if slot < 0:
            continue
        page = slot // block_size
        pos = slot % block_size
        # Row-index the 2-D cache (strided-view safe) instead of computing
        # page * stride(0) offsets into reshape(-1).
        page_row = cache_2d[page]
        value_base = pos * index_head_dim
        scale_base = block_size * index_head_dim + pos * scale_bytes
        scale = page_row[scale_base : scale_base + scale_bytes].view(torch.float32)[0]
        values = page_row[value_base : value_base + index_head_dim].view(
            torch.float8_e4m3fn
        )
        out[token_idx].copy_(values.float() * scale)
    return out


def _compress_v4_state_windows_capturable(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_table_base_offsets: torch.Tensor | None,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    compress_ratio: int,
    head_dim: int,
    overlap: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    if num_actual == 0:
        return (
            torch.empty((0, head_dim), device=state_cache.device, dtype=torch.float32),
            torch.empty((0,), device=state_cache.device, dtype=torch.bool),
        )

    token_positions = positions[:num_actual].to(torch.int64)
    state_slots = compressor_slot_mapping[:num_actual].to(torch.int64)
    valid_token = (state_slots >= 0) & (
        torch.remainder(token_positions + 1, compress_ratio) == 0
    )

    window = (2 if overlap else 1) * compress_ratio
    offsets = torch.arange(window, device=state_cache.device, dtype=torch.int64)
    window_positions = token_positions[:, None] - window + 1 + offsets[None, :]
    table_idx_raw = torch.div(
        window_positions, compressor_block_size, rounding_mode="floor"
    )
    req_idx = token_to_req_indices[:num_actual].to(torch.int64).clamp_min(0)
    if block_table_base_offsets is not None:
        safe_req_for_base = req_idx.clamp(
            0, max(int(block_table_base_offsets.shape[0]) - 1, 0)
        )
        base_logical_page = block_table_base_offsets.to(
            device=state_cache.device,
            dtype=torch.int64,
        )[safe_req_for_base]
        table_idx_raw = table_idx_raw - base_logical_page[:, None]
    valid_window = (
        (window_positions >= 0)
        & (table_idx_raw >= 0)
        & (table_idx_raw < block_table.shape[1])
    )
    table_idx = table_idx_raw.clamp(0, max(block_table.shape[1] - 1, 0))
    block_number = block_table[req_idx[:, None], table_idx]
    valid_window = valid_window & (block_number >= 0)

    safe_block = block_number.to(torch.int64).clamp_min(0)
    pos_in_block = torch.remainder(window_positions.clamp_min(0), compressor_block_size)
    rows = state_cache[safe_block, pos_in_block]
    state_width = state_cache.shape[-1] // 2

    if overlap:
        head_offsets = torch.where(
            offsets >= compress_ratio,
            torch.full_like(offsets, head_dim),
            torch.zeros_like(offsets),
        )
    else:
        head_offsets = torch.zeros_like(offsets)
    dim_indices = (
        head_offsets[:, None]
        + torch.arange(head_dim, device=state_cache.device, dtype=torch.int64)[None, :]
    )
    dim_indices = dim_indices[None, :, :].expand(num_actual, -1, -1)

    kv_rows = torch.gather(rows[..., :state_width], -1, dim_indices).float()
    score_rows = torch.gather(rows[..., state_width:], -1, dim_indices).float()
    valid_window_f = valid_window.unsqueeze(-1)
    score_rows = torch.where(
        valid_window_f, score_rows, score_rows.new_full((), -1.0e30)
    )
    weights = torch.softmax(score_rows, dim=1)
    kv_rows = torch.where(valid_window_f, kv_rows, torch.zeros_like(kv_rows))
    compressed = torch.sum(kv_rows * weights, dim=1)
    variance = compressed.square().sum(dim=-1, keepdim=True) / float(head_dim)
    normed = compressed * torch.rsqrt(variance + rms_norm_eps)
    return normed * rms_norm_weight.float(), valid_token


def deepseek_v4_hca_compress_kv_cache_insert(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int = 128,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Compress HCA state, normalize/RoPE/FP8-quantize, and insert KV cache.

    The HCA path writes one compressed cache entry only at positions where
    `(position + 1) % 128 == 0`.
    """

    if compress_ratio != 128:
        raise ValueError(
            f"HCA cache insert requires compress_ratio=128, got {compress_ratio}"
        )
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")
    state_width = state_cache.shape[-1] // 2
    head_dim = int(rms_norm_weight.numel())
    if state_width != head_dim:
        raise ValueError(f"HCA state width must be {head_dim}, got {state_width}")
    if compressor_block_size != state_cache.shape[1]:
        raise ValueError(
            "compressor_block_size must match state_cache page size, "
            f"got {compressor_block_size} vs {state_cache.shape[1]}"
        )
    rope_dim = int(cos_sin_cache.shape[-1])
    min_block_stride = kv_cache_block_size * deepseek_v4_swa_row_bytes(
        state_width, rope_dim
    )
    if kv_cache_2d.dim() != 2 or kv_cache_2d.shape[1] < min_block_stride:
        raise ValueError(
            f"kv_cache_2d must be [blocks, >= {min_block_stride}] uint8, "
            f"got {tuple(kv_cache_2d.shape)}"
        )
    if kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {kv_cache_2d.dtype}")

    num_actual = min(
        compressor_slot_mapping.numel(),
        positions.numel(),
        kv_slot_mapping.numel(),
    )
    if num_actual == 0:
        return
    if not state_cache.is_cuda:
        raise ValueError(
            "deepseek_v4_hca_compress_kv_cache_insert only supports CUDA tensors."
        )

    _triton_fused_sparse_compress_cache_insert(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        compressor_slot_mapping=compressor_slot_mapping,
        block_table=block_table,
        compressor_block_size=compressor_block_size,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        cos_sin_cache=cos_sin_cache,
        kv_cache_2d=kv_cache_2d,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_cache_block_size,
        compress_ratio=compress_ratio,
        overlap=False,
        block_table_base_offsets=block_table_base_offsets,
    )


def deepseek_v4_csa_compress_kv_cache_insert(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int = 4,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Compress CSA state and insert one `fp8_ds_mla` row per 4 tokens.

    CSA uses overlap: the compression window spans eight token positions and
    selects the first 512-wide slice from the older four positions and the
    second slice from the newer four positions before the softmax-weighted sum.
    """

    if compress_ratio != 4:
        raise ValueError(
            f"CSA cache insert requires compress_ratio=4, got {compress_ratio}"
        )
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")
    state_width = state_cache.shape[-1] // 2
    head_dim = int(rms_norm_weight.numel())
    expected_width = head_dim * 2
    if state_width != expected_width:
        raise ValueError(f"CSA state width must be {expected_width}, got {state_width}")
    if compressor_block_size != state_cache.shape[1]:
        raise ValueError(
            "compressor_block_size must match state_cache page size, "
            f"got {compressor_block_size} vs {state_cache.shape[1]}"
        )
    rope_dim = int(cos_sin_cache.shape[-1])
    min_block_stride = kv_cache_block_size * deepseek_v4_swa_row_bytes(
        head_dim, rope_dim
    )
    if kv_cache_2d.dim() != 2 or kv_cache_2d.shape[1] < min_block_stride:
        raise ValueError(
            f"kv_cache_2d must be [blocks, >= {min_block_stride}] uint8, "
            f"got {tuple(kv_cache_2d.shape)}"
        )
    if kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {kv_cache_2d.dtype}")

    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    if num_actual == 0:
        return
    if not state_cache.is_cuda:
        raise ValueError(
            "deepseek_v4_csa_compress_kv_cache_insert only supports CUDA tensors."
        )

    _triton_fused_sparse_compress_cache_insert(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        compressor_slot_mapping=compressor_slot_mapping,
        block_table=block_table,
        compressor_block_size=compressor_block_size,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        cos_sin_cache=cos_sin_cache,
        kv_cache_2d=kv_cache_2d,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_cache_block_size,
        compress_ratio=compress_ratio,
        overlap=True,
        block_table_base_offsets=block_table_base_offsets,
    )


import os as _os  # noqa: E402
import time as _time  # noqa: E402

_INS_TIMING = {"n": 0, "compress": 0.0, "div": 0.0, "rope": 0.0, "had": 0.0, "write": 0.0}
_INS_ENABLE = _os.environ.get("DSV4_INS_STAGE") == "1"
# The fused FP8 indexer cache insert replaces ~65 ATen kernels per CSA layer,
# baked into the decode CUDA graph and replayed every step, which makes it the
# largest single component of our decode gap against vLLM. It is on by default
# because its written bytes were measured against the eager chain on live
# traffic: a 4,374-token batch with 1,092 compress boundaries writing 1,092
# DISTINCT indexer slots gave 1 differing byte of 139,776 (one e4m3 ULP), 0 of
# 1,092 rows picking a different divisor, and 0 of 139,776 elements more than one
# ULP away. The indexer top-k survives that: it selects 512 of thousands, and a
# 1-ULP change can only move a selection on an exact tie at the 512th boundary.
#
# Two earlier "~0.7% of bytes disagree" readings were the engine's warm-up
# dummies rather than traffic -- one mapped every page index onto a single
# physical page, so 32 compress groups claimed each slot, and one wrote nothing
# at all. The verifier now waits for a batch that writes real slots; see
# _indexer_verify_is_meaningful. Set the variable to 0 to force the eager chain.
_FP8_INDEXER_FUSED = _os.environ.get("TOKENSPEED_DSV4_FP8_INDEXER_FUSED", "1") != "0"
_FP8_INDEXER_FUSED_VERIFY = (
    _os.environ.get("TOKENSPEED_DSV4_FP8_INDEXER_FUSED_VERIFY") == "1"
)
_FP8_INDEXER_FUSED_VERIFY_STATE = {"done": False}
_INS_W_TIMING = {"n": 0, "scale": 0.0, "slots": 0.0, "sync": 0.0, "compact": 0.0, "scatter": 0.0, "empty": 0}


def _ins_write_timing(d: dict) -> None:
    if not _INS_ENABLE:
        return
    t = _INS_W_TIMING
    t["n"] += 1
    for k in ("scale", "slots", "sync", "compact", "scatter"):
        t[k] += d.get(k, 0.0)
    if d.get("empty"):
        t["empty"] += 1
    if t["n"] % 128 == 0:
        e = t["empty"]
        snap = {k: t[k] for k in ("scale", "slots", "sync", "compact", "scatter")}
        t["n"] = 0
        t["empty"] = 0
        for k in ("scale", "slots", "sync", "compact", "scatter"):
            t[k] = 0.0
        print(
            f"[W_TIMING] #{128} empty={e} ms="
            + " ".join(f"{k}:{v * 1e3:.2f}" for k, v in snap.items()),
            flush=True,
        )


def _ins_term(stage: str, dur: float) -> None:
    if not _INS_ENABLE:
        return
    _INS_TIMING[stage] += dur
    _INS_TIMING["n"] += 1
    if _INS_TIMING["n"] % 128 == 0:
        d = dict(_INS_TIMING)
        d.pop("n", None)
        _INS_TIMING["n"] = 0
        for k in ("compress", "div", "rope", "had", "write"):
            _INS_TIMING[k] = 0.0
        print(
            f"[INS_TIMING] #{128} ms="
            + " ".join(f"{k}:{v * 1e3:.2f}" for k, v in d.items()),
            flush=True,
        )


def _looks_like_warmup_batch(
    positions: torch.Tensor,
    block_table: torch.Tensor,
) -> bool:
    """Whether this call is one of the engine's synthetic warm-up batches.

    The one-shot verifier used to fire on the first non-captured forward, which
    is a warm-up pass, not a request.  The warm-up batch is recognisable and it
    is not merely noisy -- it is ill-posed: its block table maps every page index
    to the same physical page, so all 2,048 compress groups of an 8,192-token
    batch resolve onto the same 64 slots and 32 of them claim each slot.  The
    eager scatter and the fused kernel then legitimately disagree about which
    claimant survives, and the byte diff measures that race rather than the
    kernel.  Both signals are required, so a real single long prefill -- whose
    positions are also a plain arange, but whose block table spans many pages --
    is still measured.
    """
    arange_like = False
    if positions.is_cuda and positions.numel() > 1:
        steps = positions[1:] - positions[:-1]
        arange_like = bool((steps == 1).all().item())
    if not arange_like:
        return False
    if not (block_table.is_cuda and block_table.numel() > 1):
        return False
    sample = block_table.reshape(-1)[:4096]
    return int(sample.min().item()) == int(sample.max().item())


def _indexer_verify_is_meaningful(
    positions: torch.Tensor,
    block_table: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
) -> bool:
    """Whether a call writes enough for an eager-vs-fused byte diff to mean anything.

    The one-shot verifier fires on the first non-captured forward, and the first
    non-captured forwards are the engine's warm-up dummies.  Two dummy shapes
    have been caught with the dump, and neither can answer the question:

    * a block table whose every entry is one physical page, so all 2,048 compress
      groups of an 8,192-token batch resolve onto 64 slots with 32 claimants
      each -- the two writers then disagree about which claimant survives, which
      is an unspecified race, not a kernel difference;
    * a batch whose indexer slot mapping is entirely `-1`, so nothing is written
      and a zero diff only reports that both writers wrote nothing.

    So wait for a batch that writes real slots.  Whether those targets are
    distinct is *reported*, not required: if the awake engine collapses pages
    too, that is a finding rather than something to gate away.
    """
    num_tokens = min(kv_slot_mapping.numel(), positions.numel())
    if num_tokens == 0:
        return False
    if int((kv_slot_mapping[:num_tokens] >= 0).sum().item()) < 2:
        return False
    return not _looks_like_warmup_batch(positions, block_table)


def _verify_fused_indexer_fp8(
    *,
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Replay the eager reference insert on a cache copy and log any diff."""
    import logging

    _log = logging.getLogger(__name__)
    index_head_dim = int(rms_norm_weight.numel())
    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    ref_cache = kv_cache_2d.clone()
    normed, valid = _compress_v4_state_windows_capturable(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        compressor_slot_mapping=compressor_slot_mapping,
        block_table=block_table,
        block_table_base_offsets=block_table_base_offsets,
        compressor_block_size=compressor_block_size,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        compress_ratio=compress_ratio,
        head_dim=index_head_dim,
        overlap=True,
    )
    cpos = (
        torch.div(
            positions[:num_actual].to(torch.int64),
            compress_ratio,
            rounding_mode="floor",
        )
        * compress_ratio
    )
    rotated = _apply_gptj_rope_tail_rows(
        normed, cpos, cos_sin_cache, int(cos_sin_cache.shape[-1])
    )
    rotated = _deepseek_v4_hadamard_rotate(rotated).float()
    _dump_path = _os.environ.get("TOKENSPEED_DSV4_FP8_INDEXER_DUMP")
    if _dump_path and not getattr(_verify_fused_indexer_fp8, "_dumped", False):
        # Save one captured live step -- inputs, plus the eager chain's own
        # intermediates -- so the offline reproducer can be driven by the real
        # tensors instead of by synthetic geometry.  The dump is what tells
        # apart "the kernel's compression differs" from "the kernel's write
        # differs", which is the question the byte-count alone cannot answer.
        _verify_fused_indexer_fp8._dumped = True
        torch.save(
            {
                "state_cache": state_cache.detach().clone(),
                "token_to_req_indices": token_to_req_indices.detach().clone(),
                "positions": positions.detach().clone(),
                "compressor_slot_mapping": compressor_slot_mapping.detach().clone(),
                "block_table": block_table.detach().clone(),
                "block_table_base_offsets": (
                    None
                    if block_table_base_offsets is None
                    else block_table_base_offsets.detach().clone()
                ),
                "compressor_block_size": compressor_block_size,
                "rms_norm_weight": rms_norm_weight.detach().clone(),
                "rms_norm_eps": rms_norm_eps,
                "cos_sin_cache": cos_sin_cache.detach().clone(),
                "kv_cache_2d": kv_cache_2d.detach().clone(),
                "kv_slot_mapping": kv_slot_mapping.detach().clone(),
                "kv_cache_block_size": kv_cache_block_size,
                "compress_ratio": compress_ratio,
                "normed": normed.detach().clone(),
                "rotated": rotated.detach().clone(),
            },
            _dump_path,
        )
        _log.info("[FSVERIFY] dumped live inputs to %s", _dump_path)
    _write_deepseek_v4_indexer_fp8_cache_capturable(
        rotated,
        ref_cache,
        kv_slot_mapping[:num_actual],
        valid,
        block_size=kv_cache_block_size,
    )
    diff = kv_cache_2d != ref_cache
    n = int(diff.sum().item())
    vslots = kv_slot_mapping[:num_actual][valid].tolist()
    # Whether the comparison had an answer at all: how many slots this batch
    # really wrote, and how many of them are distinct.  A batch that writes
    # nothing, or writes N groups into fewer than N slots, cannot adjudicate
    # between the two writers however small its byte diff is.
    written = kv_slot_mapping[:num_actual][valid]
    written = written[written >= 0]
    _log.info(
        "[FSVERIFY] state=%s stride=%s npos=%d nslot=%d nkv=%d num_actual=%d "
        "valid=%d writes=%d distinct_targets=%d diff_bytes=%d valid_slots=%s",
        tuple(state_cache.shape),
        tuple(state_cache.stride()),
        positions.numel(),
        compressor_slot_mapping.numel(),
        kv_slot_mapping.numel(),
        num_actual,
        int(valid.sum().item()),
        int(written.numel()),
        int(written.unique().numel()),
        n,
        vslots[:64],
    )
    if n:
        nz = diff.nonzero()
        pages = sorted(set(nz[:, 0].tolist()))
        value_bytes = kv_cache_block_size * index_head_dim
        slot_of = []
        for r, c in nz:
            r = int(r)
            c = int(c)
            if c < value_bytes:
                slot_of.append((r, c // index_head_dim))
            else:
                slot_of.append((r, (c - value_bytes) // 4))
        slots = sorted(set(slot_of))
        r0, c0 = int(nz[0, 0]), int(nz[0, 1])
        head = "value" if c0 < value_bytes else "scale"
        if head == "value":
            token = c0 // index_head_dim
            byte = c0 % index_head_dim
        else:
            token = (c0 - value_bytes) // 4
            byte = (c0 - value_bytes) % 4
        _log.info(
            "[FSVERIFY] first_diff page=%d col=%d kind=%s slot_token=%d byte=%d "
            "fused=%d ref=%d ndistinct_slots=%d slots=%s",
            r0,
            c0,
            head,
            token,
            byte,
            int(kv_cache_2d[r0, c0]),
            int(ref_cache[r0, c0]),
            len(slots),
            slots[:32],
        )
        # Decompose value vs scale diffs and decode fp8 magnitude for a few slots.
        page_val = value_bytes
        value_diff = int(diff[:, :page_val].sum().item())
        scale_diff = int(diff[:, page_val:].sum().item())
        _log.info(
            "[FSVERIFY] decompose value_diff=%d scale_diff=%d pages=%s",
            value_diff,
            scale_diff,
            pages[:8],
        )
        # How many bytes differ is the wrong question: the two arms sum the
        # Hadamard in different orders (the kernel reduces a signed matrix
        # product, the eager path calls fast_hadamard_transform), so a last-bit
        # difference can move a stored byte by exactly one fp8 ULP without any
        # real divergence.  The divisor is a discrete choice and must match
        # exactly, and any deviation beyond one ULP is a defect.  Report those
        # two, so this log can distinguish the two cases instead of counting
        # bytes.
        valid_slots_t = kv_slot_mapping[:num_actual][valid].to(torch.int64)
        if valid_slots_t.numel() and index_head_dim:
            blk = valid_slots_t // kv_cache_block_size
            off = valid_slots_t % kv_cache_block_size
            dims = torch.arange(index_head_dim, device=kv_cache_2d.device)[None, :]
            vcols = off[:, None] * index_head_dim + dims
            sbase = kv_cache_block_size * index_head_dim
            sdims = torch.arange(4, device=kv_cache_2d.device)[None, :]
            scols = sbase + off[:, None] * 4 + sdims
            fv = kv_cache_2d[blk[:, None], vcols].view(torch.float8_e4m3fn).float()
            rv = ref_cache[blk[:, None], vcols].view(torch.float8_e4m3fn).float()
            fs = kv_cache_2d[blk[:, None], scols].view(torch.float32)
            rs = ref_cache[blk[:, None], scols].view(torch.float32)
            div_mismatch = int((fs != rs).any(dim=-1).sum().item())
            mag = torch.maximum(fv.abs(), rv.abs()).clamp_min(2.0**-6)
            ulp = torch.pow(
                torch.full_like(mag, 2.0), torch.floor(torch.log2(mag)) - 3.0
            )
            excess = (fv - rv).abs() - ulp
            _log.info(
                "[FSVERIFY] steps: rows=%d divisor_mismatch=%d over_one_ulp=%d/%d "
                "worst_excess_steps=%.3f worst_reconstruction=%.6g",
                int(valid_slots_t.numel()),
                div_mismatch,
                int((excess > 0).sum().item()),
                int(fv.numel()),
                float((excess / ulp).max().item()),
                float((fv * fs - rv * rs).abs().max().item()),
            )
        # Decode one diverging slot in each arm: real ≈ fp8(byte) * divisor.
        shown = 0
        for r, slot in sorted(set((int(r_), int(c) // index_head_dim) for r_, c in nz if int(c) < page_val)):
            div_fused = (
                int(kv_cache_2d[r, page_val + slot * 4])
                | (int(kv_cache_2d[r, page_val + slot * 4 + 1]) << 8)
                | (int(kv_cache_2d[r, page_val + slot * 4 + 2]) << 16)
                | (int(kv_cache_2d[r, page_val + slot * 4 + 3]) << 24)
            )
            div_ref = (
                int(ref_cache[r, page_val + slot * 4])
                | (int(ref_cache[r, page_val + slot * 4 + 1]) << 8)
                | (int(ref_cache[r, page_val + slot * 4 + 2]) << 16)
                | (int(ref_cache[r, page_val + slot * 4 + 3]) << 24)
            )
            fbytes = kv_cache_2d[r, slot * index_head_dim : (slot + 1) * index_head_dim]
            rbytes = ref_cache[r, slot * index_head_dim : (slot + 1) * index_head_dim]
            div_fused_f = struct.unpack("<f", struct.pack("<I", div_fused))[0]
            div_ref_f = struct.unpack("<f", struct.pack("<I", div_ref))[0]
            fv = (
                fbytes.to(torch.uint8)
                .view(torch.float8_e4m3fn)
                .float()
                .mul(div_fused_f)
            )
            rv = (
                rbytes.to(torch.uint8)
                .view(torch.float8_e4m3fn)
                .float()
                .mul(div_ref_f)
            )
            maxabs = float((fv - rv).abs().max().item())
            gslot = int(r) * kv_cache_block_size + int(slot)
            mat = (
                (kv_slot_mapping[:num_actual] == gslot)
                & valid
            )
            tvec = mat.nonzero()
            tt = int(tvec[0, 0].item()) if tvec.numel() else -1
            log_extra = ""
            if tt >= 0:
                nm = normed[tt].to(torch.float32).tolist()
                rt = rotated[tt].to(torch.float32).tolist()
                log_extra = (
                    " tok=%d pos=%d normed=%s rotated=%s fv=%s rv=%s"
                    % (
                        tt,
                        int(positions[tt]),
                        [round(v, 3) for v in nm[:8]],
                        [round(v, 3) for v in rt[:8]],
                        [round(v, 3) for v in fv[:8].tolist()],
                        [round(v, 3) for v in rv[:8].tolist()],
                    )
                )
            _log.info(
                "[FSVERIFY] slot page=%d slot=%d gslot=%d div_fused=%d div_ref=%d "
                "scale_eq=%s val_maxabs=%.6g nbyte_diff=%d%s",
                r,
                slot,
                gslot,
                div_fused,
                div_ref,
                div_fused == div_ref,
                maxabs,
                int((fbytes != rbytes).sum().item()),
                log_extra,
            )
            shown += 1
            if shown >= 4:
                break
    vpos = positions[:num_actual][valid]
    vtslot = compressor_slot_mapping[:num_actual][valid]
    _log.info(
        "[FSVERIFY] positions=%s state_slots=%s",
        vpos[:16].tolist(),
        vtslot[:16].tolist(),
    )
    # Dump the eager window for the first valid token.
    if valid.any() and torch.cuda.is_available():
        j0 = int(valid.nonzero()[0, 0])
        pos0 = int(positions[j0].item())
        req0 = int(token_to_req_indices[j0].item())
        if block_table_base_offsets is not None:
            base0 = int(block_table_base_offsets[req0].item())
        else:
            base0 = 0
        win = 2 * compress_ratio
        details = []
        for o in range(win):
            wp = pos0 - win + 1 + o
            ti = wp // compressor_block_size - base0
            ti_c = max(0, min(int(block_table.shape[1]) - 1, ti))
            bn = int(block_table[req0, ti_c].item())
            pp = (max(0, wp)) % compressor_block_size
            details.append((o, wp, ti, bn, pp))
        _log.info(
            "[FSVERIFY] win_dump pos0=%d req0=%d base0=%d state_block=%d "
            "col block_table_WIDTH=%d details=%s",
            pos0,
            req0,
            base0,
            compressor_block_size,
            int(block_table.shape[1]),
            details,
        )


def deepseek_v4_csa_indexer_cache_insert(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    use_fp4_cache: bool,
    compress_ratio: int = 4,
    block_table_base_offsets: torch.Tensor | None = None,
) -> None:
    """Compress CSA indexer state and insert FP8/MXFP4 indexer cache rows."""

    if compress_ratio != 4:
        raise ValueError(
            f"CSA indexer cache insert requires compress_ratio=4, got {compress_ratio}"
        )
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")
    state_width = state_cache.shape[-1] // 2
    index_head_dim = int(rms_norm_weight.numel())
    expected_width = index_head_dim * 2
    if state_width != expected_width:
        raise ValueError(
            f"CSA indexer state width must be {expected_width}, got {state_width}"
        )

    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    if num_actual == 0:
        return
    if not state_cache.is_cuda:
        raise ValueError(
            "deepseek_v4_csa_indexer_cache_insert only supports CUDA tensors."
        )
    if use_fp4_cache:
        _triton_fused_csa_indexer_mxfp4_cache_insert(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            kv_cache_2d=kv_cache_2d,
            kv_slot_mapping=kv_slot_mapping,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
            block_table_base_offsets=block_table_base_offsets,
        )
        return

    if _FP8_INDEXER_FUSED:
        _triton_fused_csa_indexer_fp8_cache_insert(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            kv_cache_2d=kv_cache_2d,
            kv_slot_mapping=kv_slot_mapping,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
            block_table_base_offsets=block_table_base_offsets,
        )
        if (
            _FP8_INDEXER_FUSED_VERIFY
            and not _FP8_INDEXER_FUSED_VERIFY_STATE["done"]
            and not torch.cuda.is_current_stream_capturing()
            and _indexer_verify_is_meaningful(positions, block_table, kv_slot_mapping)
        ):
            _FP8_INDEXER_FUSED_VERIFY_STATE["done"] = True
            try:
                _verify_fused_indexer_fp8(
                    state_cache=state_cache,
                    token_to_req_indices=token_to_req_indices,
                    positions=positions,
                    compressor_slot_mapping=compressor_slot_mapping,
                    block_table=block_table,
                    compressor_block_size=compressor_block_size,
                    rms_norm_weight=rms_norm_weight,
                    rms_norm_eps=rms_norm_eps,
                    cos_sin_cache=cos_sin_cache,
                    kv_cache_2d=kv_cache_2d,
                    kv_slot_mapping=kv_slot_mapping,
                    kv_cache_block_size=kv_cache_block_size,
                    compress_ratio=compress_ratio,
                    block_table_base_offsets=block_table_base_offsets,
                )
            except Exception as _e:  # noqa: BLE001
                import logging as _lg

                _lg.getLogger(__name__).error("[FSVERIFY] exception %r", _e)
        return

    _t0 = _time.perf_counter()
    normed, valid = _compress_v4_state_windows_capturable(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        compressor_slot_mapping=compressor_slot_mapping,
        block_table=block_table,
        block_table_base_offsets=block_table_base_offsets,
        compressor_block_size=compressor_block_size,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        compress_ratio=compress_ratio,
        head_dim=index_head_dim,
        overlap=True,
    )
    _t1 = _time.perf_counter()
    _ins_term("compress", _t1 - _t0)
    compressed_positions = (
        torch.div(
            positions[:num_actual].to(torch.int64),
            compress_ratio,
            rounding_mode="floor",
        )
        * compress_ratio
    )
    _t2 = _time.perf_counter()
    _ins_term("div", _t2 - _t1)
    rotated = _apply_gptj_rope_tail_rows(
        normed,
        compressed_positions,
        cos_sin_cache,
        int(cos_sin_cache.shape[-1]),
    )
    _t3 = _time.perf_counter()
    _ins_term("rope", _t3 - _t2)
    rotated = _deepseek_v4_hadamard_rotate(rotated).float()
    _t4 = _time.perf_counter()
    _ins_term("had", _t4 - _t3)
    _write_deepseek_v4_indexer_fp8_cache_capturable(
        rotated,
        kv_cache_2d,
        kv_slot_mapping[:num_actual],
        valid,
        block_size=kv_cache_block_size,
    )
    _t5 = _time.perf_counter()
    _ins_term("write", _t5 - _t4)
