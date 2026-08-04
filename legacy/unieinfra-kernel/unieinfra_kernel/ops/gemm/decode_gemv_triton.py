# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Decode-tuned GEMV (skinny ``M``, narrow ``N``) for dense bf16/fp16 weights.

Why this kernel exists
----------------------
``docs/experiments-log.md`` E26 profiled batch-1 Qwen3-8B decode on sm_120 with
Nsight Compute hardware counters. Every weight projection is served by the same
generic cuBLAS ``gemvx::kernel<...cublasGemvParamsEx...>`` template at 216
registers/thread, and the achieved DRAM bandwidth tracks how many threads the
launch puts on the machine rather than any hardware ceiling::

    Gate+Up  786432 threads -> 89.7% of peak DRAM BW
    Down     131072 threads -> 83.6%
    QKV       98304 threads -> 70.5%
    O-proj    65536 threads -> 67.1%

Decode weight projections are pure streaming problems (~2 FLOP per byte), so the
*only* figure of merit is achieved bandwidth. The narrow-output projections lose
~20 points of it to a latency-hiding deficit, not to physics.

Design
------
One program per ``BLOCK_N`` output columns; the whole ``K`` reduction happens
inside that program. Parallelism therefore comes from making ``BLOCK_N``
*small* (1/2/4/8 columns per block) so the grid is 2-4k blocks on a 188-SM part,
instead of from split-K. Split-K is implemented too (``SPLIT_K > 1``, fp32
atomics into a zeroed workspace) because it was the obvious first hypothesis,
but on these shapes it loses — see E33 in the experiments log.

Key implementation points:

* ``BLOCK_K`` is sized so each thread moves 8 bf16 values (one 128-bit load) per
  iteration, e.g. ``BLOCK_K = 1024`` with ``num_warps=4``.
* The weight tile is loaded with ``eviction_policy="evict_first"``. Each weight
  byte is read exactly once per call, so keeping it resident in L2 only evicts
  data that is still wanted. (Triton 3.6 accepts the ``.cs`` cache modifier on
  stores only, so the eviction policy is the portable spelling of ``__ldcs``.)
* fp32 accumulation; the reduction over ``BLOCK_K`` is a ``tl.sum`` (Triton
  lowers it to per-thread partials plus a warp shuffle tree).
* No autotuning at call time. Configs come from a static table keyed on
  ``(M, N, K)`` so the kernel is safe to capture in a CUDA graph — a runtime
  autotuner benchmarks candidates and synchronizes, which is illegal during
  capture.

Provenance
----------
The block-per-output-column mapping, the "narrow blocks minimize wave
quantization" tile choice, the 128-bit (8 x bf16) per-thread load, and the
evict-first weight load are ported from vLLM's ``bf16_skinny_gemm.cu``
(Apache-2.0, ``third_party/vllm/csrc/libtorch_stable/bf16_skinny_gemm.cu``). No
code was copied; the techniques were re-expressed in portable Triton so the same
kernel runs on NVIDIA and AMD. See ``ACKNOWLEDGEMENTS.md``.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass

import torch
from unieinfra_kernel.boundary import Priority, register_kernel
from unieinfra_kernel.contracts.gemm import (
    DECODE_GEMV,
    DECODE_GEMV_CAPABILITY,
    decode_gemv_reference,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GemvConfig",
    "decode_gemv",
    "gemv_stats",
    "reset_gemv_stats",
    "select_gemv_config",
    "supports_decode_gemv",
    "unieinfra_decode_gemv_triton",
]

# Call accounting, so an integration can *prove* the fast path ran instead of
# inferring it from a timing delta. ``fast`` counts calls that launched the
# Triton kernel; ``delegated`` counts calls the wrapper handed to ``F.linear``
# (a shape the runtime profitability rule declined). Under CUDA-graph replay
# neither advances — the Python wrapper only runs during capture and eager
# warm-up — so treat these as "was it captured", not "how many steps ran".
_STATS: dict[str, int] = {"fast": 0, "delegated": 0}


def gemv_stats() -> dict[str, int]:
    """Return a snapshot of the fast/delegated call counters."""
    return dict(_STATS)


def reset_gemv_stats() -> None:
    """Zero the fast/delegated call counters."""
    _STATS["fast"] = 0
    _STATS["delegated"] = 0


try:  # pragma: no cover - exercised on GPU boxes only
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001 - CPU-only test boxes have no triton
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


# Largest M this kernel claims. Above it the activation re-reads (M x K bytes
# per block) stop being free and cuBLAS's tensor-core GEMM wins.
MAX_DECODE_M = 8

# Dtypes the kernel handles.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

# ``UNIEINFRA_DECODE_GEMV=0`` disables the kernel without unregistering it, so a
# benchmark or a bisect can A/B it in-place.
_ENABLED = os.environ.get("UNIEINFRA_DECODE_GEMV", "1") != "0"

# ``UNIEINFRA_DECODE_GEMV_FORCE=1`` bypasses the profitability rule so the
# benchmark can measure shapes the kernel would normally decline.
_FORCE = os.environ.get("UNIEINFRA_DECODE_GEMV_FORCE", "0") == "1"


@dataclass(frozen=True)
class GemvConfig:
    """One launch configuration for :func:`decode_gemv`.

    Attributes:
        block_n: Output columns computed per program. Small values raise the
            grid size, which is the whole point on narrow ``N``.
        block_k: Reduction elements consumed per program per iteration. Choose
            ``block_k == 8 * 32 * num_warps`` for one 128-bit load per thread.
        split_k: Number of programs cooperating on the ``K`` reduction. ``1``
            means a single deterministic pass; ``> 1`` uses fp32 atomics into a
            zeroed workspace and a cast/bias epilogue launch.
        num_warps: Warps per program.
        num_stages: Software pipeline depth for the ``K`` loop.
        stream_b: Load the weight with the evict-first cache hint.
    """

    block_n: int = 2
    block_k: int = 1024
    split_k: int = 1
    num_warps: int = 4
    num_stages: int = 2
    stream_b: bool = True


if _HAS_TRITON:

    @triton.jit
    def _decode_gemv_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        bias_ptr,
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
        SPLIT_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_K: tl.constexpr,
        STREAM_B: tl.constexpr,
    ):
        """``C[M, N] = A[M, K] @ B[N, K].T`` for tiny ``M``.

        One program owns ``BLOCK_N`` columns of ``C`` (and, when ``SPLIT_K > 1``,
        a strided slice of ``K``). ``A`` and ``B`` must both be contiguous along
        ``K``.

        ``NUM_M`` is the true row count, specialized as a constexpr so the row
        loop is fully unrolled — the same trick as vLLM's ``kNumTokens``
        template parameter. Rows are handled one at a time against a single
        ``[BLOCK_N, BLOCK_K]`` weight tile: the natural
        ``a[:, None, :] * b[None, :, :]`` formulation materializes a
        ``BLOCK_M x BLOCK_N x BLOCK_K`` fp32 intermediate, which forces
        ``BLOCK_K`` down to 128 at ``M = 8`` and costs ~12x (measured, E33).
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        n_mask = offs_n < N
        b_base = b_ptr + offs_n[:, None] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
            cur_k = k0 + offs_k
            k_ok = cur_k < K

            # Evict-first: every weight byte is touched once per call, so
            # keeping it in L2 only displaces data that is still live. This is
            # Triton's portable spelling of vLLM's ``__ldcs`` weight load; the
            # ``.cs`` load cache modifier itself is store-only in Triton 3.6.
            if EVEN_K and EVEN_N:
                b = tl.load(
                    b_base + cur_k[None, :],
                    eviction_policy="evict_first" if STREAM_B else "",
                )
            else:
                if EVEN_K:
                    b_mask = n_mask[:, None] | tl.zeros([BLOCK_K], tl.int1)[None, :]
                elif EVEN_N:
                    b_mask = k_ok[None, :] | tl.zeros([BLOCK_N], tl.int1)[:, None]
                else:
                    b_mask = n_mask[:, None] & k_ok[None, :]
                b = tl.load(
                    b_base + cur_k[None, :],
                    mask=b_mask,
                    other=0.0,
                    eviction_policy="evict_first" if STREAM_B else "",
                )
            b_f = b.to(tl.float32)

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

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        store_mask = (offs_m < M)[:, None] & n_mask[None, :]

        if SPLIT_K == 1:
            if HAS_BIAS:
                bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
                acc += bias.to(tl.float32)[None, :]
            tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=store_mask)
        else:
            # c_ptr is an fp32 workspace here; bias and the output cast are
            # applied by _split_k_epilogue_kernel.
            tl.atomic_add(c_ptrs, acc, mask=store_mask)

    @triton.jit
    def _split_k_epilogue_kernel(
        acc_ptr,
        out_ptr,
        bias_ptr,
        M,
        N,
        stride_om,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Cast the fp32 split-K accumulator to the output dtype, adding bias."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        v = tl.load(acc_ptr + pid_m * N + offs_n, mask=mask, other=0.0)
        if HAS_BIAS:
            v += tl.load(bias_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + pid_m * stride_om + offs_n,
            v.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

else:  # pragma: no cover - CPU-only import path
    _decode_gemv_kernel = None
    _split_k_epilogue_kernel = None


@functools.lru_cache(maxsize=8)
def _sm_count(device_index: int) -> int:
    """Streaming multiprocessor count, used to size the grid."""
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:  # noqa: BLE001 - non-CUDA or driver hiccup
        return 128


# Static, offline-tuned table. Keys are ``(M, N, K)``; produced by
# ``benchmark/scripts/bench_decode_gemv.py --mode sweep`` over BLOCK_N x BLOCK_K
# x SPLIT_K x num_warps x num_stages, on an RTX PRO 6000 (sm_120, 188 SMs). See
# E33 in ``docs/experiments-log.md``. Shapes outside the table use
# :func:`_heuristic_config`, which encodes the same rules the sweep found.
_TUNED_CONFIGS: dict[tuple[int, int, int], GemvConfig] = {
    # Qwen3-8B qkv_proj (hidden 4096 -> 6144)
    (1, 6144, 4096): GemvConfig(1, 2048, 1, 2, 2),
    # Qwen3-8B o_proj (4096 -> 4096)
    (1, 4096, 4096): GemvConfig(1, 1024, 1, 4, 2),
    (2, 4096, 4096): GemvConfig(1, 1024, 1, 4, 2),
    # Qwen3.6-35B-A3B qkv_proj (hidden 2048 -> 5120; 16x256 q + 2x256 kv x2)
    (1, 5120, 2048): GemvConfig(4, 2048, 1, 4, 2),
    (2, 5120, 2048): GemvConfig(2, 2048, 1, 2, 2),
    # Qwen3.6-35B-A3B o_proj (4096 -> 2048). M=4 and M=8 are listed because
    # _PROFITABLE_TO_M8 lets this shape run there; without them the newly
    # reachable batch sizes would fall to the heuristic, which picks
    # block_k=1024/num_warps=4 -- close to the M=8 optimum but not the M=4 one.
    (1, 2048, 4096): GemvConfig(4, 512, 1, 4, 1),
    (2, 2048, 4096): GemvConfig(2, 1024, 1, 2, 1),
    (4, 2048, 4096): GemvConfig(1, 2048, 1, 2, 1),
    (8, 2048, 4096): GemvConfig(1, 1024, 1, 2, 1),
    # The four entries above break the BLOCK_N=1 rule the heuristic encodes,
    # and that is the point of measuring rather than assuming. This model's
    # K is 2048 -- half of anything the original sweep saw -- so the per-column
    # reduction is cheap and wider tiles pay for themselves by amortizing the
    # activation re-read, which is the trade the heuristic's docstring says
    # "only starts to matter" at larger tiles. Swept with
    # `bench_decode_gemv.py --mode sweep --shapes m35_o,m35_qkv`; only M<=2 is
    # listed because `_is_profitable` hands M>=4 to cuBLAS, and the sweep agrees
    # it should: the best config found at M=8 still loses at 0.837x.
}


def _heuristic_config(M: int, N: int, K: int, device_index: int) -> GemvConfig:
    """Pick a launch config from shape alone (no runtime benchmarking).

    The sweep's verdict, in two rules:

    * ``BLOCK_N = 1`` — one output column per program. This is the opposite of
      the usual GEMM instinct and it is the whole trick: it maximizes the grid
      (``N`` blocks) and minimizes wave quantization on a 188-SM part. vLLM
      reached the same conclusion from the other direction, narrowing
      ``kNPB`` from 8 to 2 for ``N = 6144``; on sm_120 the optimum is 1. Wider
      tiles amortize the activation re-read, which only starts to matter at
      ``M >= 4`` — and that is outside the region where this kernel is used at
      all (see :func:`_is_profitable`).
    * ``BLOCK_K`` large enough that each thread issues one 128-bit load per
      iteration (``8 values x 32 lanes x num_warps``).

    ``SPLIT_K`` stays 1: with ``BLOCK_N = 1`` the grid is already ``N`` blocks,
    so splitting the reduction only adds an atomics/epilogue tax. The sweep
    never picked ``SPLIT_K > 1`` on any shape or batch size.
    """
    del device_index  # grid size is N/block_n; block_n is already minimal

    block_n = 1
    num_warps = 4
    block_k = 8 * 32 * num_warps  # 1024: one 128-bit load per thread
    while block_k > 128 and block_k > K:
        block_k //= 2

    # Cap the live weight tile so the fp32 conversion does not spill. The row
    # loop is unrolled rather than broadcast, so M does not enter here.
    while block_n * block_k > 8192 and block_k > 128:
        block_k //= 2

    return GemvConfig(
        block_n=block_n,
        block_k=block_k,
        split_k=1,
        num_warps=num_warps,
        num_stages=2,
        stream_b=True,
    )


#: Shapes (N, K) measured profitable for this kernel across the whole decode
#: batch range 1..8, rather than only M<=2. Membership must come from a sweep
#: whose two timing methodologies agree; see :func:`_is_profitable`.
_PROFITABLE_TO_M8: frozenset[tuple[int, int]] = frozenset({
    (2048, 4096),   # Qwen3.6-35B-A3B o_proj
})


def _is_profitable(M: int, N: int, K: int) -> bool:
    """Whether this kernel is measurably faster than cuBLAS for a shape.

    Correctness never depends on this; speed does. Measured on sm_120 with
    L2-defeating buffer rotation and CUDA-graph timing (E33, ratios vs cuBLAS):

    ====================== ===== ===== ===== =====
    shape (N, K)           M=1   M=2   M=4   M=8
    ====================== ===== ===== ===== =====
    qkv_proj  (6144, 4096) 1.16  1.04  0.98  0.62
    o_proj    (4096, 4096) 1.16  1.16  1.11  0.66
    down_proj (4096,12288) 1.04  1.05  1.00  0.65
    ====================== ===== ===== ===== =====

    The pattern is not arbitrary. cuBLAS's ``gemvx`` under-parallelizes at
    ``M = 1`` on every narrow output — that is the E26 gap. From ``M = 2`` it
    switches to a ``cutlass_80_wmma_tensorop`` GEMM which recovers most of it.
    By ``M = 8`` the per-row reduction this kernel performs (one block-wide
    ``tl.sum`` per row per K tile) dominates and cuBLAS wins everywhere — a
    Triton structural limit, not a limit of the technique: vLLM's hand-written
    CUDA equivalent holds ~80% of peak all the way to ``M = 8``.

    The cut is at ``M = 2``, not ``M = 4``, on purpose. ``M = 4`` measures +11%
    under production-like timing (warm activation, cold weights) but *slower*
    than cuBLAS under ncu's flushed-cache replay, because the activation
    re-read this kernel does once per output column stops being free once it
    misses. Two methodologies disagreeing is not a green light, so ``M = 4`` is
    left to cuBLAS. ``M <= 2`` wins under both.

    Returns ``False`` -> the wrapper delegates to ``F.linear``, i.e. exactly what
    the incumbent would have done.
    """
    if _FORCE:
        return True

    # Shape-specific overrides, for shapes where the M<=2 rule above was
    # measured to be wrong. It was derived entirely from Qwen3-8B projections
    # and applied to every shape; `del N, K` used to make that explicit.
    #
    # Qwen3.6-35B-A3B o_proj (N=2048, K=4096) profits at every M in 1..8:
    #
    #   M         1      2      4      8
    #   sweep   1.160  1.856  1.810  1.400   (L2-defeating rotation)
    #   warm    1.150  1.947  1.771  1.102   (--mode compare)
    #
    # Both methodologies agree at all four points, which is the standard this
    # docstring sets for the 8B M=4 case ("Two methodologies disagreeing is not
    # a green light"). The reason it differs from 8B is cuBLAS, not this kernel:
    # on this narrow output cuBLAS jumps from 16.4us at M=1 to ~26.7us at M>=2
    # and stays there, so the ratio widens rather than closing.
    #
    # Its sibling projection does NOT behave this way -- qkv (N=5120, K=2048)
    # measures 1.005x at M=4 and 0.837x at M=8 -- so this cannot be a per-model
    # rule either. It is genuinely per-shape.
    if (N, K) in _PROFITABLE_TO_M8:
        return M <= 8
    return M <= 2


def select_gemv_config(M: int, N: int, K: int, device_index: int = 0) -> GemvConfig:
    """Return the launch config for a shape, preferring the offline-tuned table.

    Args:
        M: Rows of the activation (tokens in the step).
        N: Output width of the projection.
        K: Reduction width.
        device_index: CUDA device ordinal, used only for the SM count.

    Returns:
        The :class:`GemvConfig` to launch with.
    """
    tuned = _TUNED_CONFIGS.get((M, N, K))
    if tuned is not None:
        return tuned
    return _heuristic_config(M, N, K, device_index)


def supports_decode_gemv(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
) -> bool:
    """Whether this kernel can serve the request at all.

    This is the *runtime* half of the gate; the declarative half is the
    ``m_le_8`` / ``n_le_8192`` trait pair the kernel registers, which keeps
    selection from ever offering it a prefill-shaped GEMM. Returning ``False``
    here makes the wrapper delegate to the reference path, so a shape this
    kernel does not like can never be slower than today — only equal.
    """
    if not (_HAS_TRITON and _ENABLED):
        return False
    if alpha is not None:
        return False
    if A.ndim != 2 or B.ndim != 2:
        return False
    if A.dtype not in _SUPPORTED_DTYPES or B.dtype != A.dtype:
        return False
    if not A.is_cuda or not B.is_cuda:
        return False
    M, K = A.shape
    if B.shape[1] != K:
        return False
    if not (1 <= M <= MAX_DECODE_M):
        return False
    # Both operands must be contiguous along K: that is what makes the loads
    # 128-bit and coalesced.
    if A.stride(1) != 1 or B.stride(1) != 1:
        return False
    # 8 x bf16 = 16 B vector loads need K divisible by 8.
    if K % 8 != 0:
        return False
    if out is not None and out.stride(1) != 1:
        return False
    return _is_profitable(M, B.shape[0], K)


def decode_gemv(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    config: GemvConfig | None = None,
) -> torch.Tensor:
    """Compute ``A @ B.T (+ bias)`` with the decode-tuned GEMV.

    Args:
        A: Activations ``[M, K]``, contiguous along ``K``, ``M <= 8``.
        B: Weights ``[N, K]``, contiguous along ``K`` (the ``nn.Linear``
            layout).
        bias: Optional ``[N]`` bias, added in the epilogue in fp32.
        out: Optional ``[M, N]`` destination. Must have ``stride(1) == 1``.
        out_dtype: Output dtype; defaults to ``A.dtype``.
        config: Launch configuration. Defaults to :func:`select_gemv_config`.

    Returns:
        ``[M, N]`` result tensor (``out`` when supplied).

    Raises:
        RuntimeError: If Triton is unavailable.
    """
    if not _HAS_TRITON:
        raise RuntimeError("decode_gemv requires triton")

    M, K = A.shape
    N = B.shape[0]
    out_dtype = out_dtype or A.dtype
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype, device=A.device)

    cfg = config or select_gemv_config(M, N, K, A.device.index or 0)
    block_m = max(1, 1 << (M - 1).bit_length())
    grid = (triton.cdiv(N, cfg.block_n), cfg.split_k)

    common = dict(
        NUM_M=M,
        BLOCK_M=block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        HAS_BIAS=bias is not None,
        EVEN_N=(N % cfg.block_n == 0),
        EVEN_K=(K % (cfg.block_k * cfg.split_k) == 0),
        STREAM_B=cfg.stream_b,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    if cfg.split_k == 1:
        _decode_gemv_kernel[grid](
            A,
            B,
            out,
            bias,
            M,
            N,
            K,
            A.stride(0),
            B.stride(0),
            out.stride(0),
            SPLIT_K=1,
            **common,
        )
        return out

    # Split-K: fp32 atomics into a zeroed workspace, then a cast/bias pass.
    acc = torch.zeros((M, N), dtype=torch.float32, device=A.device)
    _decode_gemv_kernel[grid](
        A,
        B,
        acc,
        bias,
        M,
        N,
        K,
        A.stride(0),
        B.stride(0),
        acc.stride(0),
        SPLIT_K=cfg.split_k,
        **{**common, "HAS_BIAS": False},
    )
    _split_k_epilogue_kernel[(M, triton.cdiv(N, 1024))](
        acc,
        out,
        bias,
        M,
        N,
        out.stride(0),
        HAS_BIAS=bias is not None,
        BLOCK_N=1024,
    )
    return out


@register_kernel(
    DECODE_GEMV.family,
    DECODE_GEMV.mode,
    name="unieinfra_decode_gemv_triton",
    solution="triton",
    capability=DECODE_GEMV_CAPABILITY,
    signatures=DECODE_GEMV.signatures,
    # The gate. ``m_le_8`` and ``n_le_16384`` are shape buckets published by
    # ``tokenspeed_kernel.ops.gemm.mm``; a kernel that declares them is only
    # ever offered requests in that bucket, and kernels that do not declare them
    # (i.e. every pre-existing kernel) are completely unaffected.
    traits={
        "m_le_8": frozenset({True}),
        "n_le_16384": frozenset({True}),
        "k_align_16": frozenset({True}),
    },
    # SPECIALIZED, not PLUGIN: this is narrowly gated on arch + shape, which is
    # exactly what that band is for. It beats the incumbent ``torch_mm``
    # (PORTABLE+3) on the shapes it declares and is invisible everywhere else.
    priority=Priority.SPECIALIZED + 1,
    tags={"latency"},
    origin="first_party",
)
def unieinfra_decode_gemv_triton(
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
    """``gemm.mm`` entry point for decode-shaped dense GEMVs.

    Args:
        A: Activations ``[M, K]``.
        B: Weights ``[N, K]``.
        A_scales: Unused (dense path); must be ``None``.
        B_scales: Unused (dense path); must be ``None``.
        out_dtype: Requested output dtype.
        alpha: Optional global scale. Present only on the nvfp4 path; if given,
            the request is delegated to the reference.
        block_size: Unused (dense path).
        bias: Optional ``[N]`` bias, fused into the epilogue.
        out: Optional output buffer ``[M, N]``.

    Returns:
        ``[M, N]`` tensor equal to ``A @ B.T (+ bias)``.
    """
    del A_scales, B_scales, block_size

    if not supports_decode_gemv(A, B, out=out, alpha=alpha) or out_dtype != A.dtype:
        _STATS["delegated"] += 1
        # Safety net: anything the fast path does not like goes to the exact
        # function the incumbent kernel calls, so selection can never regress a
        # shape it happens to route here.
        result = decode_gemv_reference(A, B, bias)
        if alpha is not None:
            result = result * alpha.to(dtype=result.dtype)
        result = result.to(out_dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    _STATS["fast"] += 1
    return decode_gemv(A, B, bias=bias, out=out, out_dtype=out_dtype)
