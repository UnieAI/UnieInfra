# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Decode-tuned W8A16 GEMV: FP8-E4M3 weights, bf16/fp16 activations.

Why weight-only, when the checkpoint is W8A8
--------------------------------------------
Every engine that serves an fp8 checkpoint quantizes the *activation* too and
calls a tensor-core scaled GEMM (vLLM: ``QuantFP8`` + ``cutlass_scaled_mm``).
That is the right shape for prefill. It is the wrong shape for decode:

* At ``M <= 8`` the projection moves ``N x K`` weight bytes and ``M x K``
  activation bytes. With ``M = 1``, ``K = 4096``, ``N = 6144`` the activation is
  8 KB against 24 MB of weights — 0.03% of the traffic. Quantizing it saves
  4 KB and costs a kernel launch.
* A tensor-core GEMM tile is ``>= 64`` rows of ``M``. At ``M = 1`` more than
  98% of every MMA is padding, and the kernel's occupancy is set by the ``N``
  tiling, which is exactly the under-parallelization E26 measured on the bf16
  path (67-70% of peak DRAM bandwidth on the narrow projections).

So this kernel keeps the activation in bf16, streams the fp8 weight once,
converts in-register and accumulates in fp32. It reads the *same* number of
weight bytes as the W8A8 path — the whole saving of fp8 — while being a pure
streaming kernel, and it is strictly *more* accurate than W8A8 because the
activation never leaves bf16.

Structure is the fp8 sibling of
:mod:`unieinfra_kernel.ops.gemm.decode_gemv_triton`: one program per output
column (``BLOCK_N = 1`` maximizes the grid on a 188-SM part), the whole ``K``
reduction inside the program, ``evict_first`` on the weight because each byte is
read exactly once, and a static config table so the kernel is CUDA-graph safe.

Scales
------
``B_scales`` is either a single per-tensor value or one value per output column
(per-channel). Both are applied in the epilogue, in fp32, before the bias — the
same order ``cutlass_scaled_mm`` uses. Per-channel scales must be laid out
``[N]`` or ``[N, 1]``; the ``[1, N]`` orientation is rejected rather than
silently transposed (E23 lost a day to exactly that ambiguity).
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
    W8A16_FP8_SIGNATURES,
)
from unieinfra_kernel.ops.gemm.decode_gemv_triton import GemvConfig

logger = logging.getLogger(__name__)

__all__ = [
    "decode_gemv_fp8",
    "fp8_gemv_stats",
    "reset_fp8_gemv_stats",
    "supports_decode_gemv_fp8",
    "unieinfra_decode_gemv_fp8_triton",
]

try:  # pragma: no cover - exercised on GPU boxes only
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001 - CPU-only test boxes have no triton
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

# ``UNIEINFRA_DECODE_GEMV_FP8=0`` disables the kernel in place, so an A/B can be
# run without unregistering it (the wrapper then delegates to the reference).
_ENABLED = os.environ.get("UNIEINFRA_DECODE_GEMV_FP8", "1") != "0"

# Largest M the kernel claims. Measured on sm_120 with L2-defeating weight
# rotation, ratio = vLLM's W8A8 (``scaled_fp8_quant`` per-token +
# ``cutlass_scaled_mm``) time / this kernel's time, so > 1 means we win:
#
#   shape (N, K)             M=1    M=2    M=4    M=8
#   qkv_proj   (6144,  4096) 1.31   1.30   1.04   0.57
#   o_proj     (4096,  4096) 1.37   1.27   1.35   0.76
#   gate_up    (24576, 4096) 1.12   1.10   0.74   0.38
#   down_proj  (4096, 12288) 1.21   1.14   0.70   0.37
#
# The cut is at M = 2, the same place the bf16 sibling cuts and for the same
# reason: this kernel performs one block-wide reduction *per row* per K tile, so
# its cost grows with M while a tensor-core GEMM's does not. Above the cut the
# wrapper delegates, i.e. does exactly what the engine would have done.
_MAX_M = int(os.environ.get("UNIEINFRA_DECODE_GEMV_FP8_MAX_M", "2"))

_STATS: dict[str, int] = {"fast": 0, "delegated": 0}


def fp8_gemv_stats() -> dict[str, int]:
    """Return a snapshot of the fast/delegated call counters."""
    return dict(_STATS)


def reset_fp8_gemv_stats() -> None:
    """Zero the fast/delegated call counters."""
    _STATS["fast"] = 0
    _STATS["delegated"] = 0


if _HAS_TRITON:

    @triton.jit
    def _decode_gemv_fp8_kernel(
        a_ptr,
        b_ptr,  # uint8 view of the FP8-E4M3 weight, [N, K]
        c_ptr,
        bias_ptr,
        wscale_ptr,
        M,
        N,
        K,
        stride_am,
        stride_bn,
        stride_cm,
        NUM_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        PER_CHANNEL: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        """``C[M, N] = (A[M, K] @ dequant(B)[N, K].T) * wscale (+ bias)``.

        ``B`` is passed as ``uint8`` and bit-cast to ``float8e4nv`` in-register:
        Triton 3.6 loads raw bytes reliably on every backend, whereas a typed
        fp8 pointer is not uniformly supported.
        """
        pid_n = tl.program_id(0)

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        n_mask = offs_n < N
        b_base = b_ptr + offs_n[:, None] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            cur_k = k0 + offs_k
            k_ok = cur_k < K

            if EVEN_K and EVEN_N:
                b_raw = tl.load(b_base + cur_k[None, :], eviction_policy="evict_first")
            else:
                if EVEN_K:
                    b_mask = n_mask[:, None] | tl.zeros([BLOCK_K], tl.int1)[None, :]
                elif EVEN_N:
                    b_mask = k_ok[None, :] | tl.zeros([BLOCK_N], tl.int1)[:, None]
                else:
                    b_mask = n_mask[:, None] & k_ok[None, :]
                b_raw = tl.load(
                    b_base + cur_k[None, :],
                    mask=b_mask,
                    other=0,
                    eviction_policy="evict_first",
                )
            b_f = tl.cast(b_raw, tl.float8e4nv, bitcast=True).to(tl.float32)

            for i in tl.static_range(NUM_M):
                if EVEN_K:
                    a_i = tl.load(a_ptr + i * stride_am + cur_k)
                else:
                    a_i = tl.load(a_ptr + i * stride_am + cur_k, mask=k_ok, other=0.0)
                s = tl.sum(a_i.to(tl.float32)[None, :] * b_f, axis=1)
                if BLOCK_M == 1:
                    acc += s[None, :]
                else:
                    acc += tl.where(offs_m[:, None] == i, s[None, :], 0.0)

        if PER_CHANNEL:
            wscale = tl.load(wscale_ptr + offs_n, mask=n_mask, other=0.0)
            acc *= wscale.to(tl.float32)[None, :]
        else:
            acc *= tl.load(wscale_ptr).to(tl.float32)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
            acc += bias.to(tl.float32)[None, :]

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        store_mask = (offs_m < M)[:, None] & n_mask[None, :]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=store_mask)

else:  # pragma: no cover - CPU-only import path
    _decode_gemv_fp8_kernel = None


def _select_config(M: int, N: int, K: int) -> GemvConfig:
    """Launch config for a shape (no runtime autotuning: CUDA-graph safe).

    Same two rules the bf16 sweep produced (E33): one output column per program
    so the grid is ``N`` blocks, and ``BLOCK_K`` sized for one 128-bit load per
    thread. fp8 is one byte per element, so 16 values per thread instead of 8.
    """
    del M, N
    num_warps = 4
    block_k = 16 * 32 * num_warps  # 2048: one 128-bit load per thread
    while block_k > 256 and block_k > K:
        block_k //= 2
    return GemvConfig(
        block_n=1,
        block_k=block_k,
        split_k=1,
        num_warps=num_warps,
        num_stages=2,
        stream_b=True,
    )


def _normalize_weight_scale(
    B_scales: torch.Tensor, N: int
) -> tuple[torch.Tensor, bool] | None:
    """Return ``(contiguous_scale, per_channel)`` or ``None`` if unsupported.

    Accepts a scalar (per-tensor) or an ``[N]`` / ``[N, 1]`` per-channel vector.
    A ``[1, N]`` layout is *rejected*, not transposed: silently accepting both
    orientations is how E23's fp8 regression hid for a day.
    """
    if B_scales.numel() == 1:
        return B_scales.reshape(1), False
    if B_scales.dim() == 1 and B_scales.shape[0] == N:
        return B_scales, True
    if B_scales.dim() == 2 and B_scales.shape == (N, 1):
        return B_scales.reshape(N), True
    return None


def supports_decode_gemv_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scales: torch.Tensor | None,
    *,
    out: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
) -> bool:
    """Whether the fast path can serve this request.

    Returning ``False`` makes the wrapper delegate to the dequant reference, so
    a declined shape is never *wrong* — only no faster than the incumbent.
    """
    if not (_HAS_TRITON and _ENABLED):
        return False
    if alpha is not None or B_scales is None:
        return False
    if A.ndim != 2 or B.ndim != 2:
        return False
    if A.dtype not in (torch.bfloat16, torch.float16):
        return False
    if B.dtype not in (torch.float8_e4m3fn, torch.uint8):
        return False
    if not A.is_cuda or not B.is_cuda:
        return False
    M, K = A.shape
    N = B.shape[0]
    if B.shape[1] != K:
        return False
    if not (1 <= M <= _MAX_M):
        return False
    # Both operands contiguous along K -> coalesced 128-bit loads.
    if A.stride(1) != 1 or B.stride(1) != 1:
        return False
    if K % 16 != 0:
        return False
    if out is not None and out.stride(1) != 1:
        return False
    return _normalize_weight_scale(B_scales, N) is not None


def decode_gemv_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scales: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    config: GemvConfig | None = None,
) -> torch.Tensor:
    """Compute ``(A @ dequant(B).T) (+ bias)`` for an FP8-E4M3 weight.

    Args:
        A: Activations ``[M, K]`` in bf16/fp16, contiguous along ``K``.
        B: Weights ``[N, K]`` in ``float8_e4m3fn`` (or its ``uint8`` view),
            contiguous along ``K`` — the ``torch.nn.Linear`` orientation.
        B_scales: Per-tensor scalar or per-channel ``[N]`` / ``[N, 1]`` fp32
            weight scale, applied in the fp32 epilogue.
        bias: Optional ``[N]`` bias, added after the scale.
        out: Optional ``[M, N]`` destination with ``stride(1) == 1``.
        out_dtype: Output dtype; defaults to ``A.dtype``.
        config: Launch configuration; defaults to :func:`_select_config`.

    Returns:
        ``[M, N]`` tensor (``out`` when supplied).

    Raises:
        RuntimeError: If Triton is unavailable.
        ValueError: If ``B_scales`` has an unsupported layout.
    """
    if not _HAS_TRITON:
        raise RuntimeError("decode_gemv_fp8 requires triton")

    M, K = A.shape
    N = B.shape[0]
    out_dtype = out_dtype or A.dtype
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype, device=A.device)

    normalized = _normalize_weight_scale(B_scales, N)
    if normalized is None:
        raise ValueError(
            f"unsupported fp8 weight-scale layout {tuple(B_scales.shape)} for N={N}"
        )
    wscale, per_channel = normalized
    wscale = wscale.contiguous()

    b_bytes = B if B.dtype == torch.uint8 else B.view(torch.uint8)
    cfg = config or _select_config(M, N, K)
    block_m = max(1, 1 << (M - 1).bit_length())

    _decode_gemv_fp8_kernel[(triton.cdiv(N, cfg.block_n),)](
        A,
        b_bytes,
        out,
        bias,
        wscale,
        M,
        N,
        K,
        A.stride(0),
        b_bytes.stride(0),
        out.stride(0),
        NUM_M=M,
        BLOCK_M=block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        HAS_BIAS=bias is not None,
        PER_CHANNEL=per_channel,
        EVEN_N=(N % cfg.block_n == 0),
        EVEN_K=(K % cfg.block_k == 0),
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return out


def decode_gemv_fp8_reference(
    A: torch.Tensor,
    B: torch.Tensor,
    B_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize-then-``F.linear`` reference for the W8A16 fp8 GEMV.

    Args:
        A: Activations ``[M, K]``.
        B: FP8 weights ``[N, K]``.
        B_scales: Per-tensor or per-channel fp32 weight scale.
        bias: Optional ``[N]`` bias.

    Returns:
        ``[M, N]`` tensor in ``A``'s dtype.
    """
    weight = B.to(torch.float32)
    scale = B_scales.to(torch.float32)
    if scale.numel() > 1:
        weight = weight * scale.reshape(-1, 1)
    else:
        weight = weight * scale.reshape(())
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
    name="unieinfra_decode_gemv_fp8_triton",
    solution="triton",
    capability=DECODE_GEMV_CAPABILITY,
    signatures=W8A16_FP8_SIGNATURES,
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
def unieinfra_decode_gemv_fp8_triton(
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
    """``gemm.mm`` entry point for decode-shaped W8A16 fp8 GEMVs.

    Args:
        A: Activations ``[M, K]`` in bf16/fp16 (**not** quantized).
        B: FP8-E4M3 weights ``[N, K]``.
        A_scales: Must be ``None`` — this signature has a dense activation.
        B_scales: Per-tensor or per-channel fp32 weight scale.
        out_dtype: Requested output dtype.
        alpha: Optional global scale (nvfp4 only); its presence delegates.
        block_size: Unused on this signature.
        bias: Optional ``[N]`` bias.
        out: Optional ``[M, N]`` output buffer.

    Returns:
        ``[M, N]`` tensor equal to ``A @ dequant(B).T (+ bias)``.
    """
    del block_size

    if (
        A_scales is not None
        or B_scales is None
        or out_dtype != A.dtype
        or not supports_decode_gemv_fp8(A, B, B_scales, out=out, alpha=alpha)
    ):
        _STATS["delegated"] += 1
        result = decode_gemv_fp8_reference(A, B, B_scales, bias)  # type: ignore[arg-type]
        result = result.to(out_dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    _STATS["fast"] += 1
    return decode_gemv_fp8(A, B, B_scales, bias=bias, out=out, out_dtype=out_dtype)
