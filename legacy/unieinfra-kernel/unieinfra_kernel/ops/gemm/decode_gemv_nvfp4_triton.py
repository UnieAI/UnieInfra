# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Decode-tuned W4A16 GEMV: NVFP4 weights, bf16/fp16 activations.

NVFP4 stores two E2M1 values per byte plus one FP8-E4M3 scale per 16 contiguous
``K`` elements, and a per-tensor global scale. A checkpoint quantized this way
is 4.25 bits/weight, so on a decode step — where the projection is a pure
weight-streaming problem (E26) — it should be the *fastest* format available,
and E30 measured exactly that on the tokenspeed base (C=128 4341 tok/s,
+59% over bf16).

The catch is that the stock path (``scaled_fp4_quant`` + ``cutlass_scaled_fp4_mm``)
also quantizes the **activation** to NVFP4 and runs a tensor-core W4A4 GEMM. At
``M = 1`` that is the wrong kernel twice over:

* The activation is ~0.03% of the bytes moved, so quantizing it saves nothing
  and costs a launch plus a block-scale round trip.
* A W4A4 MMA tile is at least 64 rows of ``M``; at ``M = 1`` over 98% of the
  tensor-core work is padding, and the achievable bandwidth is capped by how
  the kernel tiles ``N``.

This kernel therefore does **W4A16**: activations stay bf16, the packed weight
is streamed once, unpacked and scaled in-register, and the reduction runs in
fp32. It moves the same ``N x K / 2`` weight bytes the W4A4 kernel does, is more
accurate (the activation is never quantized), and — being a plain streaming
kernel — parallelizes as ``N`` independent programs.

Layout contract
---------------
``B`` is ``[N, K // 2]`` ``uint8`` (two E2M1 nibbles per byte, low nibble first,
i.e. the ModelOpt/vLLM packing). ``B_scales`` is ``[N, K // 16]``
``float8_e4m3fn`` in **linear** (un-swizzled) layout — *not* the 128x4-tiled
layout CUTLASS wants. ``alpha`` is the scalar weight global scale, so

    ``W[n, k] = e2m1(nibble) * float(B_scales[n, k // 16]) * alpha``

which is the same dequantization ``run_nvfp4_emulations`` performs, and the
weight half of what ``cutlass_scaled_fp4_mm`` folds into its own ``alpha``.
"""

from __future__ import annotations

import logging
import os

import torch
from unieinfra_kernel.boundary import Priority, register_kernel
from unieinfra_kernel.contracts.gemm import (
    DECODE_GEMV_CAPABILITY,
    FAMILY,
    MODE,
    W4A16_NVFP4_SIGNATURES,
)

logger = logging.getLogger(__name__)

__all__ = [
    "decode_gemv_nvfp4",
    "decode_gemv_nvfp4_reference",
    "nvfp4_gemv_stats",
    "reset_nvfp4_gemv_stats",
    "supports_decode_gemv_nvfp4",
    "unieinfra_decode_gemv_nvfp4_triton",
]

try:  # pragma: no cover - exercised on GPU boxes only
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001 - CPU-only test boxes have no triton
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

# NVFP4 block size: 16 K-elements share one FP8 scale.
GROUP_SIZE = 16

_ENABLED = os.environ.get("UNIEINFRA_DECODE_GEMV_NVFP4", "1") != "0"

# Largest M the kernel claims. Measured on sm_120 with L2-defeating weight
# rotation, ratio = vLLM's W4A4 (``scaled_fp4_quant`` + ``cutlass_scaled_fp4_mm``)
# time / this kernel's time, so > 1 means we win:
#
#   shape (N, K)             M=1    M=2    M=4    M=8
#   qkv_proj   (6144,  4096) 2.01   1.94   1.20   0.78
#   o_proj     (4096,  4096) 2.04   2.07   1.76   1.07
#   gate_up    (24576, 4096) 1.27   1.10   0.77   0.43
#   down_proj  (4096, 12288) 1.46   1.19   0.77   0.47
#
# Same cut as the fp8 and bf16 siblings, and for the same structural reason:
# one block-wide reduction per row per K tile. M <= 2 wins on every shape;
# M = 4 wins on two of four, which is not a green light.
_MAX_M = int(os.environ.get("UNIEINFRA_DECODE_GEMV_NVFP4_MAX_M", "2"))

_STATS: dict[str, int] = {"fast": 0, "delegated": 0}


def nvfp4_gemv_stats() -> dict[str, int]:
    """Return a snapshot of the fast/delegated call counters."""
    return dict(_STATS)


def reset_nvfp4_gemv_stats() -> None:
    """Zero the fast/delegated call counters."""
    _STATS["fast"] = 0
    _STATS["delegated"] = 0


if _HAS_TRITON:

    @triton.jit
    def _e2m1_decode(nibble):
        """Decode a 4-bit E2M1 nibble to fp32 by direct IEEE-754 construction.

        For magnitudes 2..7 the fp32 bit pattern is ``0x3F000000 + (mag << 22)``
        (one shift, one add, one bitcast); magnitudes 0 and 1 are patched.
        """
        magnitude = nibble & 0x07
        sign = (nibble >> 3) & 1
        bits = 0x3F000000 + (magnitude.to(tl.int32) << 22)
        val = bits.to(tl.float32, bitcast=True)
        val = tl.where(magnitude == 0, 0.0, val)
        val = tl.where(magnitude == 1, 0.5, val)
        return tl.where(sign == 1, -val, val)

    @triton.jit
    def _decode_gemv_nvfp4_kernel(
        a_ptr,
        b_ptr,  # uint8, [N, K // 2]
        s_ptr,  # uint8 view of FP8-E4M3 block scales, [N, K // 16]
        c_ptr,
        bias_ptr,
        alpha_ptr,
        M,
        N,
        NUM_BLOCKS,  # K // 16
        stride_am,
        stride_bn,
        stride_sn,
        stride_cm,
        NUM_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        NBLK: tl.constexpr,
        GS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        EVEN_BLOCKS: tl.constexpr,
    ):
        """One program per output column ``n``; whole ``K`` reduction inside.

        Each iteration consumes ``NBLK`` scale-blocks, i.e. ``NBLK * 8`` weight
        bytes and ``NBLK * 16`` activation elements.
        """
        n = tl.program_id(0)
        if n >= N:
            return

        offs_m = tl.arange(0, BLOCK_M)
        offs_blk = tl.arange(0, NBLK)
        offs_byte = tl.arange(0, GS // 2)
        offs_elem = tl.arange(0, GS)

        b_row = b_ptr + n * stride_bn
        s_row = s_ptr + n * stride_sn

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for blk0 in range(0, NUM_BLOCKS, NBLK):
            blk = blk0 + offs_blk
            blk_ok = blk < NUM_BLOCKS

            byte_idx = blk[:, None] * (GS // 2) + offs_byte[None, :]
            if EVEN_BLOCKS:
                raw = tl.load(b_row + byte_idx, eviction_policy="evict_first")
                scale_raw = tl.load(s_row + blk, eviction_policy="evict_first")
            else:
                raw = tl.load(
                    b_row + byte_idx,
                    mask=blk_ok[:, None],
                    other=0,
                    eviction_policy="evict_first",
                )
                scale_raw = tl.load(
                    s_row + blk,
                    mask=blk_ok,
                    other=0,
                    eviction_policy="evict_first",
                )

            low = _e2m1_decode(raw & 0x0F)
            high = _e2m1_decode((raw >> 4) & 0x0F)
            # Byte j holds elements 2j (low nibble) and 2j+1 (high nibble).
            vals = tl.interleave(low, high)  # [NBLK, GS]

            scale = tl.cast(scale_raw, tl.float8e4nv, bitcast=True).to(tl.float32)
            w = vals * scale[:, None]

            elem_idx = blk[:, None] * GS + offs_elem[None, :]
            for i in tl.static_range(NUM_M):
                if EVEN_BLOCKS:
                    a_i = tl.load(a_ptr + i * stride_am + elem_idx)
                else:
                    a_i = tl.load(
                        a_ptr + i * stride_am + elem_idx,
                        mask=blk_ok[:, None],
                        other=0.0,
                    )
                s = tl.sum(a_i.to(tl.float32) * w)
                if BLOCK_M == 1:
                    acc += s
                else:
                    acc += tl.where(offs_m == i, s, 0.0)

        acc *= tl.load(alpha_ptr).to(tl.float32)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + n).to(tl.float32)

        tl.store(
            c_ptr + offs_m * stride_cm + n,
            acc.to(c_ptr.dtype.element_ty),
            mask=offs_m < M,
        )

else:  # pragma: no cover - CPU-only import path
    _decode_gemv_nvfp4_kernel = None


def _select_nblk(num_blocks: int) -> tuple[int, int]:
    """Return ``(NBLK, num_warps)``: scale-blocks per iteration and warps.

    ``NBLK * 8`` weight bytes are read per iteration. With 4 warps (128 threads)
    ``NBLK = 256`` gives one 128-bit load per thread, which is the same
    "one vector load per thread per iteration" rule the bf16 sweep settled on
    (E33). Shorter reductions shrink ``NBLK`` so the loop still has work.
    """
    num_warps = 4
    nblk = 256
    while nblk > 16 and nblk > num_blocks:
        nblk //= 2
    return nblk, num_warps


def supports_decode_gemv_nvfp4(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scales: torch.Tensor | None,
    alpha: torch.Tensor | None,
    *,
    out: torch.Tensor | None = None,
) -> bool:
    """Whether the fast path can serve this request (never a correctness gate)."""
    if not (_HAS_TRITON and _ENABLED):
        return False
    if B_scales is None or alpha is None:
        return False
    if A.ndim != 2 or B.ndim != 2 or B_scales.ndim != 2:
        return False
    if A.dtype not in (torch.bfloat16, torch.float16):
        return False
    if B.dtype != torch.uint8:
        return False
    if B_scales.dtype not in (torch.float8_e4m3fn, torch.uint8):
        return False
    if not (A.is_cuda and B.is_cuda and B_scales.is_cuda):
        return False
    if alpha.numel() != 1:
        return False
    M, K = A.shape
    N = B.shape[0]
    if B.shape[1] * 2 != K:
        return False
    if K % GROUP_SIZE != 0:
        return False
    if B_scales.shape != (N, K // GROUP_SIZE):
        return False
    if not (1 <= M <= _MAX_M):
        return False
    if A.stride(1) != 1 or B.stride(1) != 1 or B_scales.stride(1) != 1:
        return False
    if out is not None and out.stride(1) != 1:
        return False
    return True


def decode_gemv_nvfp4(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute ``(A @ dequant_nvfp4(B).T) (+ bias)``.

    Args:
        A: Activations ``[M, K]`` in bf16/fp16, contiguous along ``K``.
        B: Packed NVFP4 weights ``[N, K // 2]`` ``uint8``.
        B_scales: ``[N, K // 16]`` FP8-E4M3 block scales in linear layout.
        alpha: Scalar fp32 weight global scale.
        bias: Optional ``[N]`` bias, added after ``alpha``.
        out: Optional ``[M, N]`` destination with ``stride(1) == 1``.
        out_dtype: Output dtype; defaults to ``A.dtype``.

    Returns:
        ``[M, N]`` tensor (``out`` when supplied).

    Raises:
        RuntimeError: If Triton is unavailable.
    """
    if not _HAS_TRITON:
        raise RuntimeError("decode_gemv_nvfp4 requires triton")

    M, K = A.shape
    N = B.shape[0]
    num_blocks = K // GROUP_SIZE
    out_dtype = out_dtype or A.dtype
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype, device=A.device)

    scale_bytes = (
        B_scales if B_scales.dtype == torch.uint8 else B_scales.view(torch.uint8)
    )
    nblk, num_warps = _select_nblk(num_blocks)
    block_m = max(1, 1 << (M - 1).bit_length())

    _decode_gemv_nvfp4_kernel[(N,)](
        A,
        B,
        scale_bytes,
        out,
        bias,
        alpha,
        M,
        N,
        num_blocks,
        A.stride(0),
        B.stride(0),
        scale_bytes.stride(0),
        out.stride(0),
        NUM_M=M,
        BLOCK_M=block_m,
        NBLK=nblk,
        GS=GROUP_SIZE,
        HAS_BIAS=bias is not None,
        EVEN_BLOCKS=(num_blocks % nblk == 0),
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def decode_gemv_nvfp4_reference(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scales: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize-then-``F.linear`` reference for the W4A16 NVFP4 GEMV.

    Args:
        A: Activations ``[M, K]``.
        B: Packed NVFP4 weights ``[N, K // 2]`` ``uint8``.
        B_scales: ``[N, K // 16]`` FP8-E4M3 block scales, linear layout.
        alpha: Scalar fp32 weight global scale.
        bias: Optional ``[N]`` bias.

    Returns:
        ``[M, N]`` tensor in ``A``'s dtype.
    """
    low = (B & 0x0F).to(torch.int32)
    high = ((B >> 4) & 0x0F).to(torch.int32)
    nibbles = torch.stack((low, high), dim=-1).reshape(B.shape[0], -1)
    magnitude = nibbles & 0x07
    # Same IEEE-754 construction the Triton kernel uses, so the reference has no
    # host-side lookup table: building one with ``torch.tensor([...])`` is a
    # CPU->GPU copy, which is illegal inside a CUDA-graph capture.
    values = (0x3F000000 + (magnitude << 22)).view(torch.float32)
    values = torch.where(magnitude == 0, 0.0, values)
    values = torch.where(magnitude == 1, 0.5, values)
    values = torch.where((nibbles & 0x08) != 0, -values, values)  # [N, K]

    scales = B_scales.view(torch.float8_e4m3fn).to(torch.float32)
    weight = values.reshape(values.shape[0], -1, GROUP_SIZE) * scales.unsqueeze(-1)
    weight = weight.reshape(values.shape) * alpha.to(torch.float32).reshape(())

    out = torch.nn.functional.linear(A.to(torch.float32), weight)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(A.dtype)


def _maybe_register(**kwargs):
    """Register the kernel unless it was disabled by the environment.

    Disabling must *unregister*, not merely make the wrapper decline: a declined
    request would still have been selected, and the wrapper's own fallback is a
    dequantize-then-matmul reference, not what the engine would have run. With
    the kernel absent from the registry, dispatch finds nothing and calls the
    engine's own quantized GEMM — which is exactly the control arm an A/B needs.
    """

    def decorate(fn):
        if not _ENABLED:
            return fn
        return register_kernel(FAMILY, MODE, **kwargs)(fn)

    return decorate


@_maybe_register(
    name="unieinfra_decode_gemv_nvfp4_triton",
    solution="triton",
    capability=DECODE_GEMV_CAPABILITY,
    signatures=W4A16_NVFP4_SIGNATURES,
    traits={
        # `m_le_2`, not `m_le_8`: the declarative gate must match the
        # profitability rule exactly. Above the cut this kernel must not be
        # *selected* at all, so dispatch finds nothing and hands the shape to
        # the engine's own scaled GEMM — which is what the measurement compares
        # against and what CUDA-graph capture (M = 4, 8, 16, ... 512) needs.
        "m_le_2": frozenset({True}),
        "n_le_16384": frozenset({True}),
        "k_align_16": frozenset({True}),
    },
    priority=Priority.SPECIALIZED + 1,
    tags={"latency"},
    origin="first_party",
)
def unieinfra_decode_gemv_nvfp4_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``gemm.mm`` entry point for decode-shaped W4A16 NVFP4 GEMVs.

    Args:
        A: Activations ``[M, K]`` in bf16/fp16 (**not** quantized).
        B: Packed NVFP4 weights ``[N, K // 2]`` ``uint8``.
        A_scales: Must be ``None`` — this signature has a dense activation.
        B_scales: ``[N, K // 16]`` FP8-E4M3 block scales, linear layout.
        out_dtype: Requested output dtype.
        alpha: Scalar fp32 weight global scale (required).
        block_size: Unused; the NVFP4 group size is fixed at 16.
        bias: Optional ``[N]`` bias.
        out: Optional ``[M, N]`` output buffer.

    Returns:
        ``[M, N]`` tensor equal to ``A @ dequant_nvfp4(B).T (+ bias)``.
    """
    del block_size

    if (
        A_scales is not None
        or B_scales is None
        or alpha is None
        or out_dtype != A.dtype
        or not supports_decode_gemv_nvfp4(A, B, B_scales, alpha, out=out)
    ):
        _STATS["delegated"] += 1
        result = decode_gemv_nvfp4_reference(
            A,
            B,
            B_scales,  # type: ignore[arg-type]
            alpha if alpha is not None else torch.ones((), device=A.device),
            bias,
        ).to(out_dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    _STATS["fast"] += 1
    return decode_gemv_nvfp4(
        A, B, B_scales, alpha, bias=bias, out=out, out_dtype=out_dtype
    )
