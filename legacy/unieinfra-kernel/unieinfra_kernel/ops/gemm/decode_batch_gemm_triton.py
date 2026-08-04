# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Mid-``M`` decode GEMM (batched decode projections) for dense bf16/fp16.

Why this kernel exists
----------------------
``docs/experiments-log.md`` E44 audited vLLM's own ``C = 128`` decode step on
sm_120 with Nsight Compute hardware counters. The step is memory-bound overall,
but the weight projections lose a measurable slice of the machine to **wave
quantization** rather than to any hardware ceiling. The counters for the
incumbent cuBLASLt/CUTLASS kernels, at ``M = 128``:

======================= ====== ======== ========= ========= ==========
projection              grid   CTA/SM   waves/SM  SM thr.   tensor pipe
======================= ====== ======== ========= ========= ==========
gate_up (64x256_32x4)   384    1        2.04      60.6%     90.5%
down    (256x64_32x4)   160    1        0.85      39.0%     51.8%
qkv     (128x128_32x4)  144    1        0.77      44.1%     61.6%
======================= ====== ======== ========= ========= ==========

The gap between "tensor pipe active" (measured over cycles the SM is *active*)
and "SM throughput" (measured over *elapsed* cycles) is the waste, and it is
pure scheduling. ``gate_up`` launches 384 CTAs that are shared-memory limited to
one resident CTA per SM on a 188-SM part, so the grid drains in three discrete
rounds of 188/188/8: ``384 / (3 x 188) = 68%`` of the machine is used, and
``60.6 / 90.5 = 67%`` is exactly what the counters report. ``down`` and ``qkv``
do not even fill the GPU once — 28 and 44 SMs are idle for the entire kernel.

Design, and one hypothesis the measurements refuted
---------------------------------------------------
The diagnosis above says the incumbent wastes a third of the machine because
its grid drains in three rounds, so the obvious fix is "launch more, smaller
CTAs until the rounds divide evenly". **That is not what won.** Sweeping the
tile space on the real shapes (E45) put the utilisation model and the clock in
direct conflict, and the clock won:

    gate_up, M=128, 188 SMs      tiles   modelled utilisation   measured
    ---------------------------  ------  --------------------   ---------
    64x64x128, split_k=4          3072                    96%    152.2 us
    64x64x128, split_k=1           768                    82%    155.9 us
    128x256x64, split_k=1           96                    51%    150.9 us  <-- best

The configuration that leaves *half the SMs empty* is the fastest one, and it
runs at 94.7% of measured peak DRAM bandwidth. So filling SMs was never the
mechanism; **memory-level parallelism per CTA** was. A ``128x256x64`` tile at
three pipeline stages keeps 96 KiB of operand in flight per CTA — essentially
the whole shared-memory budget — and 96 such CTAs saturate the memory system on
their own, at eight warps per SM against the incumbent's four. ncu confirms the
mechanism directly: same DRAM bytes (208.25 MB vs 207.93 MB, so no traffic was
saved), DRAM throughput 53.9% -> 75.1%, and **waves per SM 2.04 -> 0.51**. The
winning kernel does not fill the GPU once. The incumbent's problem is not
that it uses 384 CTAs, it is that each one is too thin (four warps, one CTA
resident per SM) to keep DRAM busy, so its three rounds are three rounds of a
half-idle memory system.

:func:`wave_utilisation` is therefore kept as a **diagnostic** — it reproduces
the incumbent's measured 67% loss from its launch geometry alone, which is what
makes the E44 attribution checkable — and is deliberately *not* the objective
:func:`_heuristic_config` maximises. Encoding a refuted model as the selection
rule would have shipped the 96%-utilisation configuration and lost.

What the search actually settles on:

* ``BLOCK_M`` covers the batch in one tile row wherever it can. The incumbent's
  ``BLOCK_M = 64`` at ``M = 128`` splits the batch in two, doubling the weight
  stream the memory system must serve; L2 absorbs it, at the cost of the
  bandwidth the kernel could have used.
* ``BLOCK_N`` and ``BLOCK_K`` as large as shared memory allows, because that is
  the term that buys memory-level parallelism.
* Split-K (deterministic: a ``[SPLIT_K, M, N]`` fp32 workspace plus a reduction
  pass, no atomics) earns its ``SPLIT_K * M * N * 4`` bytes each way only on the
  long-``K``, narrow-``N`` shape — ``down_proj`` (``K = 12288``, ``N = 4096``),
  where the tuned table does pick it.
* A persistent form (``programs < tiles``, each program strides the tile space)
  is implemented and was measured: it is within noise of the plain launch on
  every shape, so the table does not use it.

No autotuning at call time. Configs come from a static table keyed on
``(M, N, K)`` so the kernel is safe to capture in a CUDA graph — a runtime
autotuner benchmarks candidates and synchronizes, which is illegal during
capture.

Relationship to :mod:`decode_gemv_triton`
-----------------------------------------
That kernel targets ``M <= 2`` batch-1 GEMV, where there is no reduction to tile
and parallelism comes from making ``BLOCK_N`` as small as one column. This is a
different regime — ``M >= 16`` uses tensor cores and real tiles — so it is a
separate registration with disjoint traits (``m_min_16`` vs ``m_le_2``), not a
change to the existing one.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass

import torch
from unieinfra_kernel.boundary import Priority, register_kernel
from unieinfra_kernel.contracts.gemm import (
    DECODE_BATCH_GEMM,
    DECODE_GEMV_CAPABILITY,
    decode_gemv_reference,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BatchGemmConfig",
    "batch_gemm_stats",
    "decode_batch_gemm",
    "reset_batch_gemm_stats",
    "select_batch_gemm_config",
    "supports_decode_batch_gemm",
    "unieinfra_decode_batch_gemm_triton",
    "wave_utilisation",
]

# Call accounting, so an integration can *prove* the fast path ran instead of
# inferring it from a timing delta. Under CUDA-graph replay neither counter
# advances — the Python wrapper only runs during capture and eager warm-up — so
# treat these as "was it captured", not "how many steps ran".
_STATS: dict[str, int] = {"fast": 0, "delegated": 0}


def batch_gemm_stats() -> dict[str, int]:
    """Return a snapshot of the fast/delegated call counters."""
    return dict(_STATS)


def reset_batch_gemm_stats() -> None:
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


# The decode batch band this kernel claims. Below ``MIN_BATCH_M`` the problem is
# a GEMV and ``decode_gemv_triton`` owns it; above ``MAX_BATCH_M`` the shape is
# compute-bound, cuBLASLt's tile choice is sound, and there is no quantization
# waste left to recover.
MIN_BATCH_M = 16
MAX_BATCH_M = 256

# Widest projection this kernel claims. A decode projection's ``N`` is one to
# six times the hidden size; ``lm_head``'s is the vocabulary, which at 151936 is
# a 1.2 GB weight matrix and a different kernel problem that E45 did not
# measure. Bounding it here keeps the kernel out of a shape it has no evidence
# for, and the bound is declarative (``n_le_32768``) so selection enforces it
# too.
MAX_PROJECTION_N = 32768

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

# Shared memory a single CTA may claim on the parts this kernel is tuned for.
# 99 KiB is the sm_120 limit Triton reports (``Hardware limit: 101376``); it is
# a hard ceiling, not a tuning knob -- exceeding it raises ``OutOfResources`` at
# *first call*, which for an engine means at CUDA-graph capture time.
_SMEM_PER_SM = 99 * 1024

# ``UNIEINFRA_DECODE_BATCH_GEMM=0`` disables the kernel without unregistering
# it, so a benchmark or a bisect can A/B it in-place.
_ENABLED = os.environ.get("UNIEINFRA_DECODE_BATCH_GEMM", "1") != "0"

# ``UNIEINFRA_DECODE_BATCH_GEMM_FORCE=1`` bypasses the profitability rule so the
# benchmark can measure shapes the kernel would normally decline.
_FORCE = os.environ.get("UNIEINFRA_DECODE_BATCH_GEMM_FORCE", "0") == "1"


@dataclass(frozen=True)
class BatchGemmConfig:
    """One launch configuration for :func:`decode_batch_gemm`.

    Attributes:
        block_m: Rows of ``C`` per program. ``>= M`` keeps the batch in a single
            M-tile, so each weight byte is read exactly once.
        block_n: Output columns per program. The main grid lever.
        block_k: Reduction elements consumed per program per pipeline stage.
        group_m: Tiles per L2 swizzle group. ``1`` disables the swizzle.
        split_k: Programs cooperating on the ``K`` reduction. ``1`` is a single
            deterministic pass; ``> 1`` uses a ``[split_k, M, N]`` fp32
            workspace plus a reduction launch (no atomics, so still
            deterministic).
        num_warps: Warps per program.
        num_stages: Software pipeline depth for the ``K`` loop.
        persist_factor: ``0`` launches one program per tile. ``k > 0`` launches
            ``min(tiles, k * num_sms)`` programs, each striding the tile space.
        stream_b: Load the weight tile with the evict-first cache hint.
    """

    block_m: int = 128
    block_n: int = 64
    block_k: int = 64
    group_m: int = 8
    split_k: int = 1
    num_warps: int = 4
    num_stages: int = 3
    persist_factor: int = 0
    stream_b: bool = True


if _HAS_TRITON:

    @triton.jit
    def _decode_batch_gemm_kernel(
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
        stride_ck,
        NUM_TILES: tl.constexpr,
        NUM_PID_N: tl.constexpr,
        NUM_PROGRAMS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        STREAM_B: tl.constexpr,
    ):
        """``C[M, N] = A[M, K] @ B[N, K].T (+ bias)`` for a decode batch.

        ``A`` and ``B`` are both contiguous along ``K`` (the ``nn.Linear``
        layout), which is the native operand layout for an NT tensor-core MMA:
        the weight tile is loaded ``[BLOCK_N, BLOCK_K]`` — fully coalesced —
        and transposed in registers/shared memory by ``tl.trans``.

        The program-to-tile map is a loop rather than a bijection. With
        ``NUM_PROGRAMS == NUM_TILES`` the loop body runs once and this is an
        ordinary tiled GEMM; with ``NUM_PROGRAMS < NUM_TILES`` it is a
        persistent kernel. Both forms come out of the same compiled code, so the
        tuned table can choose between them per shape.

        When ``SPLIT_K > 1``, ``c_ptr`` is an fp32 ``[SPLIT_K, M, N]`` workspace
        and each ``pid_k`` owns one slice; the caller reduces it. Writing
        disjoint slices instead of accumulating with atomics keeps the result
        bit-reproducible run to run.
        """
        pid = tl.program_id(0)
        pid_k = tl.program_id(1)
        num_pid_m: tl.constexpr = NUM_TILES // NUM_PID_N

        for tile_id in tl.range(pid, NUM_TILES, NUM_PROGRAMS):
            # Group-M swizzle: consecutive programs share weight columns, so the
            # B tile they stream stays L2-resident across the group.
            if GROUP_M == 1:
                pid_m = tile_id // NUM_PID_N
                pid_n = tile_id % NUM_PID_N
            else:
                num_in_group = GROUP_M * NUM_PID_N
                group_id = tile_id // num_in_group
                first_m = group_id * GROUP_M
                group_m = min(num_pid_m - first_m, GROUP_M)
                pid_m = first_m + ((tile_id % num_in_group) % group_m)
                pid_n = (tile_id % num_in_group) // group_m

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            m_mask = offs_m < M
            n_mask = offs_n < N
            # Clamp the out-of-range lanes onto row/column 0 instead of masking
            # every load in the K loop: the lanes are discarded by the store
            # mask, and a predicated load in the inner loop costs more than the
            # duplicated fetch.
            a_ptrs = a_ptr + tl.where(m_mask, offs_m, 0)[:, None] * stride_am
            b_ptrs = b_ptr + tl.where(n_mask, offs_n, 0)[:, None] * stride_bn
            a_ptrs = a_ptrs + offs_k[None, :]
            b_ptrs = b_ptrs + offs_k[None, :]

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in tl.range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
                if EVEN_K:
                    a = tl.load(a_ptrs + k0)
                    b = tl.load(
                        b_ptrs + k0,
                        eviction_policy="evict_first" if STREAM_B else "",
                    )
                else:
                    k_ok = (k0 + offs_k) < K
                    a = tl.load(a_ptrs + k0, mask=k_ok[None, :], other=0.0)
                    b = tl.load(
                        b_ptrs + k0,
                        mask=k_ok[None, :],
                        other=0.0,
                        eviction_policy="evict_first" if STREAM_B else "",
                    )
                acc = tl.dot(a, tl.trans(b), acc)

            store_mask = m_mask[:, None] & n_mask[None, :]
            if SPLIT_K == 1:
                if HAS_BIAS:
                    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
                    acc += bias.to(tl.float32)[None, :]
                c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
                tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=store_mask)
            else:
                c_ptrs = (
                    c_ptr
                    + pid_k * stride_ck
                    + offs_m[:, None] * stride_cm
                    + offs_n[None, :]
                )
                tl.store(c_ptrs, acc, mask=store_mask)

    @triton.jit
    def _split_k_reduce_kernel(
        acc_ptr,
        out_ptr,
        bias_ptr,
        M,
        N,
        stride_om,
        stride_ak,
        SPLIT_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Sum the ``SPLIT_K`` fp32 partials, add bias, cast to the out dtype."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in tl.static_range(SPLIT_K):
            acc += tl.load(
                acc_ptr + k * stride_ak + pid_m * N + offs_n, mask=mask, other=0.0
            )
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + pid_m * stride_om + offs_n,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

else:  # pragma: no cover - CPU-only import path
    _decode_batch_gemm_kernel = None
    _split_k_reduce_kernel = None


@functools.lru_cache(maxsize=8)
def _sm_count(device_index: int) -> int:
    """Streaming multiprocessor count, used to size the grid."""
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:  # noqa: BLE001 - non-CUDA or driver hiccup
        return 128


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _smem_bytes(cfg: BatchGemmConfig, itemsize: int = 2) -> int:
    """Shared memory the pipelined ``K`` loop needs, in bytes.

    Triton's software pipeline holds ``num_stages - 1`` copies of the operand
    tiles in flight, not ``num_stages``. Modelling it as ``num_stages``
    over-estimates by one buffer and, worse, an over-estimate that is then
    compared against a doubled budget under-constrains the search: a
    ``128x128x128`` tile at three stages needs 128 KiB against a 99 KiB limit
    and fails at first call. Caught by the ragged-batch test at ``M = 127``.
    """
    buffers = max(1, cfg.num_stages - 1)
    return (cfg.block_m + cfg.block_n) * cfg.block_k * itemsize * buffers


def wave_utilisation(
    M: int,
    N: int,
    K: int,
    cfg: BatchGemmConfig,
    num_sms: int,
    ctas_per_sm: int | None = None,
) -> tuple[float, int, int]:
    """Fraction of the machine a launch geometry keeps busy. **Diagnostic only.**

    A grid of ``T`` tiles over ``R`` simultaneously resident CTAs drains in
    ``ceil(T / R)`` rounds, and everything the last, partial round leaves idle is
    wasted. This reproduces the loss E44 measured on the incumbent ``gate_up``
    tile — 384 CTAs, one resident per SM on 188 SMs, ``384 / (3 * 188) = 68%``
    against the ``60.6 / 90.5 = 67%`` its counters reported — which is what makes
    that attribution checkable rather than asserted.

    It is **not** the objective :func:`_heuristic_config` maximises, because the
    sweep refuted it as a predictor of our own configs: the fastest ``gate_up``
    configuration scores 51% here and is still the fastest, by 3%, over one
    scoring 96%. See the module docstring.

    Args:
        M: Rows of the activation (tokens in the step).
        N: Output width of the projection.
        K: Reduction width.
        cfg: Candidate launch configuration.
        num_sms: Streaming multiprocessors on the device.
        ctas_per_sm: Measured residency, when it is known. Defaults to a
            shared-memory estimate, which is only ever an estimate — the real
            limit also depends on registers and on how the compiler laid the
            tile out. Pass ncu's ``launch__occupancy_limit_shared_mem`` here to
            model a kernel that has actually been profiled, rather than guessing
            a number the profiler already reported.

    Returns:
        ``(utilisation, tiles, ctas_per_sm)`` where ``utilisation`` is in
        ``(0, 1]`` and ``tiles`` counts programs' worth of work including the
        split-K factor.
    """
    tiles = _cdiv(M, cfg.block_m) * _cdiv(N, cfg.block_n) * cfg.split_k
    if ctas_per_sm is None:
        ctas_per_sm = max(1, min(4, _SMEM_PER_SM // max(_smem_bytes(cfg), 1)))
    resident = num_sms * ctas_per_sm
    rounds = _cdiv(tiles, resident)
    return tiles / (resident * rounds), tiles, ctas_per_sm


# Static, offline-tuned table. Keys are ``(M, N, K)``; produced by
# ``benchmark/scripts/bench_decode_batch_gemm.py --mode sweep`` on an
# RTX PRO 6000 (sm_120, 188 SMs), against cold (L2-defeating) weight buffers.
# The trailing comment is the measured speedup over cuBLASLt on that shape.
# See E45 in ``docs/experiments-log.md``. Shapes outside the table use
# :func:`_heuristic_config`.
#
# Qwen3-8B (hidden 4096) at the decode concurrencies a served engine runs at.
_TUNED_CONFIGS: dict[tuple[int, int, int], BatchGemmConfig] = {
    # gate_up_proj (4096 -> 24576)
    (32, 24576, 4096): BatchGemmConfig(32, 128, 128, 8, 1, 4, 2),  # 1.38x
    (64, 24576, 4096): BatchGemmConfig(64, 128, 64, 8, 1, 4, 3),  # 1.24x
    (128, 24576, 4096): BatchGemmConfig(128, 256, 64, 8, 1, 8, 3),  # 1.34x
    (256, 24576, 4096): BatchGemmConfig(128, 64, 64, 8, 1, 4, 3),  # 1.08x
    # down_proj (12288 -> 4096): long K, narrow N -- the one shape where the
    # split-K reduction pass pays for itself.
    (32, 4096, 12288): BatchGemmConfig(32, 128, 128, 8, 4, 4, 3),  # 1.08x
    (64, 4096, 12288): BatchGemmConfig(64, 64, 128, 8, 4, 4, 4),  # 1.42x
    (128, 4096, 12288): BatchGemmConfig(64, 128, 128, 8, 2, 4, 3),  # 1.17x
    (256, 4096, 12288): BatchGemmConfig(64, 128, 64, 8, 4, 4, 3),  # 0.96x
    # qkv_proj (4096 -> 6144)
    (32, 6144, 4096): BatchGemmConfig(32, 128, 128, 8, 1, 8, 3),  # 1.15x
    (64, 6144, 4096): BatchGemmConfig(64, 64, 128, 8, 1, 8, 3),  # 1.50x
    (128, 6144, 4096): BatchGemmConfig(64, 128, 64, 8, 1, 4, 4),  # 1.04x
    (256, 6144, 4096): BatchGemmConfig(128, 128, 64, 8, 1, 4, 3),  # 1.99x
    # o_proj (4096 -> 4096)
    (32, 4096, 4096): BatchGemmConfig(32, 128, 128, 8, 2, 4, 3),  # 1.12x
    (64, 4096, 4096): BatchGemmConfig(64, 64, 128, 8, 1, 8, 4),  # 1.29x
    (128, 4096, 4096): BatchGemmConfig(64, 64, 128, 8, 1, 8, 4),  # 1.19x
    (256, 4096, 4096): BatchGemmConfig(64, 128, 64, 8, 1, 4, 4),  # 1.00x
}


def _heuristic_config(M: int, N: int, K: int, device_index: int) -> BatchGemmConfig:
    """Pick a launch config from shape alone, for shapes not in the tuned table.

    This is the *generalisation* of what the sweep found, not a model of it. The
    two rules the tuned entries above have in common:

    * Cover the batch in as few tile rows as possible (``BLOCK_M`` up to 128),
      so each weight byte is streamed once.
    * Buy memory-level parallelism with the tile, not with the grid: a wide
      ``BLOCK_N x BLOCK_K`` beats a larger number of thin CTAs on every shape
      measured, including ones where it leaves half the SMs idle.

    Split-K is used only where the reduction is long and the output narrow
    enough that the extra ``SPLIT_K * M * N`` fp32 round trip is small next to
    the weight stream — the ``down_proj`` regime. The threshold is expressed as
    a traffic ratio rather than a shape whitelist so it transfers to other
    models.

    Args:
        M: Rows of the activation.
        N: Output width of the projection.
        K: Reduction width.
        device_index: CUDA device ordinal, used only for the SM count.

    Returns:
        The :class:`BatchGemmConfig` to launch with.
    """
    num_sms = _sm_count(device_index)
    block_m = min(128, max(16, 1 << (M - 1).bit_length()))
    block_n = 128 if N >= 128 else 64
    block_k = 128 if K % 128 == 0 else 64
    num_warps = 8 if block_m * block_n >= 16384 else 4
    num_stages = 3

    split_k = 1
    tiles = _cdiv(M, block_m) * _cdiv(N, block_n)
    if tiles < num_sms:
        # Too few tiles to occupy the machine even once. Split the reduction --
        # which, unlike a finer N tiling, adds parallelism without re-reading a
        # single weight byte -- but only while the fp32 round trip it costs
        # stays under ~10% of the weight stream it is trying to keep busy.
        weight_bytes = N * K * 2
        for candidate in (4, 2):
            if K % (block_k * candidate) != 0:
                continue
            if tiles * candidate > 4 * num_sms:
                continue
            if 2 * candidate * M * N * 4 > 0.1 * weight_bytes:
                continue
            split_k = candidate
            break

    cfg = BatchGemmConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=8,
        split_k=split_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    # Shared memory is the one hard limit: back off BLOCK_K until the pipelined
    # tile fits, rather than letting the launch fail at run time.
    while _smem_bytes(cfg) > _SMEM_PER_SM and cfg.block_k > 32:
        cfg = BatchGemmConfig(
            block_m=cfg.block_m,
            block_n=cfg.block_n,
            block_k=cfg.block_k // 2,
            group_m=cfg.group_m,
            split_k=cfg.split_k if K % (cfg.block_k // 2 * cfg.split_k) == 0 else 1,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    return cfg


def _is_profitable(M: int, N: int, K: int) -> bool:
    """Whether this kernel is measurably faster than cuBLASLt for a shape.

    Correctness never depends on this; speed does. The rule is exactly the
    predicate the declarative traits express — ``16 <= M <= 256`` and
    ``N <= 32768`` — and nothing else, on purpose. CUDA-graph capture walks
    ``M = 1 ... 512`` at engine
    construction (E38), so a rule the traits cannot express is a rule that first
    fires *during capture*; keeping the two identical is what makes the gate
    checkable, and ``test_declarative_gate_equals_the_profitability_rule`` checks
    it for every M in that walk.

    That simplicity is paid for. On the E45 sweep the band is a clear net win
    per decoder layer (1.21x at ``M = 256``, more below it), but two individual
    cells inside it are not: ``down_proj`` at ``M = 256`` measures 0.96x and
    ``o_proj`` at ``M = 256`` measures 1.00x, because by that batch size those
    two shapes are compute-bound and cuBLASLt's tile is sound. A shape-keyed
    deny-list would recover those two cells and would have to be re-measured for
    every new model; the band is stated in ``M`` alone so that it transfers.

    Returns ``False`` -> the wrapper delegates to ``F.linear``, i.e. exactly what
    the incumbent would have done.
    """
    del K
    if _FORCE:
        return True
    return MIN_BATCH_M <= M <= MAX_BATCH_M and N <= MAX_PROJECTION_N


def select_batch_gemm_config(
    M: int, N: int, K: int, device_index: int = 0
) -> BatchGemmConfig:
    """Return the launch config for a shape, preferring the offline-tuned table.

    Args:
        M: Rows of the activation (tokens in the step).
        N: Output width of the projection.
        K: Reduction width.
        device_index: CUDA device ordinal, used only for the SM count.

    Returns:
        The :class:`BatchGemmConfig` to launch with.
    """
    tuned = _TUNED_CONFIGS.get((M, N, K))
    if tuned is not None:
        return tuned
    return _heuristic_config(M, N, K, device_index)


def supports_decode_batch_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
) -> bool:
    """Whether this kernel can serve the request at all.

    This is the *runtime* half of the gate; the declarative half is the
    ``m_min_16`` / ``m_le_256`` / ``n_align_64`` / ``k_align_128`` trait set the
    kernel registers. Returning ``False`` here makes the wrapper delegate to the
    reference path, so a shape this kernel does not like can never be slower
    than today — only equal.
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
    N = B.shape[0]
    if B.shape[1] != K:
        return False
    if not (MIN_BATCH_M <= M <= MAX_BATCH_M):
        return False
    # Both operands must be contiguous along K: that is what makes the weight
    # load coalesced and the MMA operand layout native.
    if A.stride(1) != 1 or B.stride(1) != 1:
        return False
    if N % 64 != 0 or K % 128 != 0 or N > MAX_PROJECTION_N:
        return False
    if out is not None and out.stride(1) != 1:
        return False
    return _is_profitable(M, N, K)


def decode_batch_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    config: BatchGemmConfig | None = None,
) -> torch.Tensor:
    """Compute ``A @ B.T (+ bias)`` with the decode-batch GEMM.

    Args:
        A: Activations ``[M, K]``, contiguous along ``K``.
        B: Weights ``[N, K]``, contiguous along ``K`` (the ``nn.Linear``
            layout).
        bias: Optional ``[N]`` bias, added in the epilogue in fp32.
        out: Optional ``[M, N]`` destination. Must have ``stride(1) == 1``.
        out_dtype: Output dtype; defaults to ``A.dtype``.
        config: Launch configuration. Defaults to
            :func:`select_batch_gemm_config`.

    Returns:
        ``[M, N]`` result tensor (``out`` when supplied).

    Raises:
        RuntimeError: If Triton is unavailable.
    """
    if not _HAS_TRITON:
        raise RuntimeError("decode_batch_gemm requires triton")

    M, K = A.shape
    N = B.shape[0]
    out_dtype = out_dtype or A.dtype
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype, device=A.device)

    device_index = A.device.index or 0
    cfg = config or select_batch_gemm_config(M, N, K, device_index)

    num_pid_m = _cdiv(M, cfg.block_m)
    num_pid_n = _cdiv(N, cfg.block_n)
    num_tiles = num_pid_m * num_pid_n
    if cfg.persist_factor:
        programs = min(num_tiles, cfg.persist_factor * _sm_count(device_index))
    else:
        programs = num_tiles

    common = dict(
        NUM_TILES=num_tiles,
        NUM_PID_N=num_pid_n,
        NUM_PROGRAMS=programs,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        GROUP_M=cfg.group_m,
        SPLIT_K=cfg.split_k,
        EVEN_K=(K % (cfg.block_k * cfg.split_k) == 0),
        STREAM_B=cfg.stream_b,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    if cfg.split_k == 1:
        _decode_batch_gemm_kernel[(programs, 1)](
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
            0,
            HAS_BIAS=bias is not None,
            **common,
        )
        return out

    # Split-K: disjoint fp32 slices, then a deterministic reduction pass.
    acc = torch.empty((cfg.split_k, M, N), dtype=torch.float32, device=A.device)
    _decode_batch_gemm_kernel[(programs, cfg.split_k)](
        A,
        B,
        acc,
        bias,
        M,
        N,
        K,
        A.stride(0),
        B.stride(0),
        acc.stride(1),
        acc.stride(0),
        HAS_BIAS=False,
        **common,
    )
    _split_k_reduce_kernel[(M, _cdiv(N, 1024))](
        acc,
        out,
        bias,
        M,
        N,
        out.stride(0),
        acc.stride(0),
        SPLIT_K=cfg.split_k,
        HAS_BIAS=bias is not None,
        BLOCK_N=1024,
    )
    return out


@register_kernel(
    DECODE_BATCH_GEMM.family,
    DECODE_BATCH_GEMM.mode,
    name="unieinfra_decode_batch_gemm_triton",
    solution="triton",
    capability=DECODE_GEMV_CAPABILITY,
    signatures=DECODE_BATCH_GEMM.signatures,
    # The gate, and it matches :func:`_is_profitable` exactly. ``m_min_16`` and
    # ``m_le_256`` are shape buckets published by
    # ``unieinfra_kernel.dispatch.shape_traits``; a kernel that declares them is
    # only ever offered requests in that bucket, and kernels that do not declare
    # them (i.e. every pre-existing kernel, including the ``m_le_2`` decode
    # GEMV) are completely unaffected.
    traits={
        "m_min_16": frozenset({True}),
        "m_le_256": frozenset({True}),
        "n_le_32768": frozenset({True}),
        "n_align_64": frozenset({True}),
        "k_align_128": frozenset({True}),
    },
    # SPECIALIZED, not PLUGIN: narrowly gated on arch + shape band, which is
    # exactly what that band is for. It beats the incumbent ``torch_mm``
    # (PORTABLE+3) on the shapes it declares and is invisible everywhere else.
    priority=Priority.SPECIALIZED + 1,
    tags={"latency"},
    origin="first_party",
)
def unieinfra_decode_batch_gemm_triton(
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
    """``gemm.mm`` entry point for decode-batch dense GEMMs.

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

    if (
        not supports_decode_batch_gemm(A, B, out=out, alpha=alpha)
        or out_dtype != A.dtype
    ):
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
    return decode_batch_gemm(A, B, bias=bias, out=out, out_dtype=out_dtype)
