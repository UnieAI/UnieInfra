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

"""Triton fused MoE expert-apply for sm_120 (RTX PRO 6000 Blackwell).

Why this exists
---------------
``moe/apply`` on sm_120 is thin and, for one weight dtype, empty. Measured on
the device (see ``docs/`` / the porting report), of the seven candidates the
registry declares admissible on sm_120:

* ``flashinfer_cutlass_unquant_moe_apply``  (PERFORMANT, 8)  -- runs
* ``flashinfer_cutlass_nvfp4_moe_apply``    (PERFORMANT, 8)  -- runs
* ``triton_mxfp8_precomputed_moe_apply``    (PORTABLE,   4)  -- runs
* ``flashinfer_trtllm_fp8_moe_apply``       (SPECIALIZED, 12) -- raises
* ``flashinfer_cutlass_fp8_moe_apply``      (PERFORMANT, 8)  -- raises
  ``NotImplementedError: FP8 block scaling not yet implemented for Blackwell``
* ``triton_mxfp4_moe_apply``                (PORTABLE,   4)  -- raises
  ``ptxas: Feature '.tile::scatter4' not supported on .target 'sm_120a'``
* ``flashinfer_cutedsl_deepep_nvfp4_moe_apply`` -- unreachable by plan

So a **block-scaled FP8 MoE has no working apply kernel on sm_120 at all**:
the SPECIALIZED entry that wins selection raises, and the PERFORMANT entry
behind it raises too. That is the hole this module fills.

Design
------
The vendored frameworks agree on one arch-portable shape, and it is the only
one that survives sm_120's constraints:

* TensorRT-LLM's trtllm-gen MoE has no sm_120 cubins at all
  (``CudaArch`` enumerates Sm90a/Sm100a/Sm100f/Sm103a only), and its CUTLASS
  grouped GEMM only instantiates sm_120 for NVFP4xNVFP4 and FP8xFP4 --
  never FP8xFP8 or BF16.
* vLLM excludes its own CUTLASS grouped GEMM on 120
  (``cap < 90 or cap >= 110 -> False``) and every ``trtllm_*`` experts class
  behind ``is_device_capability_family(100)``, which is false for 12.x.
* SGLang's default for unquantized MoE on sm_120 is its Triton
  ``fused_moe_kernel``, and it explicitly disables the persistent/TMA-gather
  path there (``if is_sm120_supported(): is_persistent = False``).

That leaves the vLLM/SGLang/lmdeploy Triton design, which all three share:
``moe_align_block_size`` produces ``(sorted_token_ids, expert_ids,
num_tokens_post_pad)``; the GEMM gathers A rows by ``offs_token // top_k`` and
scatters C rows by ``offs_token``, so tokens are never materialised in expanded
form; the router weight is folded into the *second* GEMM's accumulator; the
activation and the top-k reduction are separate launches. None of the three
fuses the two GEMMs -- neither does this.

Tiling is taken from the tuned tables the vendored trees already ship *for this
exact GPU*: SGLang's ``E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_
Server_Edition.json`` (BF16) and vLLM's ``...,dtype=fp8_w8a8,block_shape=
[128,128].json`` (FP8). Both are the Qwen3.6-35B-A3B geometry. Note how small
they stay -- ``BLOCK_N <= 128`` and ``BLOCK_M <= 64`` until M is in the
thousands -- which is what a 100 KB shared-memory SM affords; the H100 tables
in the same directories use 128x256x128. A shared-memory clamp
(:func:`_clamp_stages`) enforces that budget for geometries the tables do not
cover, the same guard SGLang applies via ``shared_memory_per_block_optin``.

Scope, stated as narrowly as the traits declare it
--------------------------------------------------
Implemented: unquantized (bf16/fp16) weights, and FP8 weights with (128, 128)
block scales; SwiGLU/SiLU gate-up activation; externally computed top-k;
expert parallelism by contiguous local-expert ownership.

Not implemented, and therefore not declared: in-kernel routing, expert bias,
deferred finalize, all-to-all EP, NVFP4/MXFP4/MXINT4 weights, FP8 scale block
shapes other than (128, 128), and internal activation dtypes other than the
input's. Selection falls back to the in-tree kernels for all of those. A
kernel that claims a capability it lacks wins selection and then returns wrong
answers, which is strictly worse than not being selected.
"""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:  # tokenspeed-kernel's single point of indirection for its Triton build.
    from tokenspeed_kernel._triton import tl, triton
except ImportError:  # pragma: no cover - stock Triton install
    import triton
    import triton.language as tl

__all__ = [
    "register_moe",
    "unieinfra_triton_moe_apply",
]

# FP8 block-scale recipe this kernel implements. Anything else is rejected by
# the declared ``fp8_scale_block_shape`` trait before it reaches the kernel.
_FP8_BLOCK_N = 128
_FP8_BLOCK_K = 128

# Fallback only. The real budget is read off the device by _smem_budget();
# sm_120 reports 101376 B opt-in against sm_100's 232448 B, and branching on
# the architecture family instead of the reported number is exactly how a tile
# budgeted for a 227 KB SM ends up on a 100 KB one.
_SMEM_BUDGET_FALLBACK_BYTES = 96 * 1024


@functools.lru_cache(maxsize=8)
def _smem_budget(device_index: int) -> int:
    """Shared memory a kernel may request on this device, with headroom.

    Read from ``shared_memory_per_block_optin`` rather than inferred from the
    architecture: the two Blackwells disagree (227 KB on sm_100, 100 KB on
    sm_120) and every vendored tree that gets this right reads the attribute.
    SGLang gates its MoE defaults on it directly; TensorRT-LLM queries
    ``cudaDevAttrMaxSharedMemoryPerBlockOptin`` and drops all but one sm_120
    CUTLASS tile below 120 KB.
    """
    props = torch.cuda.get_device_properties(device_index)
    optin = getattr(props, "shared_memory_per_block_optin", None)
    if not optin:
        return _SMEM_BUDGET_FALLBACK_BYTES
    # Leave ~4 KB for whatever the compiler places alongside the pipelined
    # operand tiles rather than budgeting the whole opt-in allowance.
    return max(int(optin) - 4096, 32 * 1024)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def _moe_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    TOP_K: tl.constexpr,
    COMPUTE_TYPE: tl.constexpr,
    USE_FP8_W8A8: tl.constexpr,
):
    """One grouped expert GEMM over block-sorted, expert-major token rows.

    ``A`` is ``[num_tokens, K]`` (gemm1) or ``[num_tokens * top_k, K]``
    (gemm2); ``B`` is ``[num_local_experts, N, K]``; ``C`` is
    ``[num_tokens * top_k, N]``. ``sorted_token_ids`` holds flat
    ``token * top_k + slot`` indices grouped by expert and padded to
    ``BLOCK_M``, with ``num_valid_tokens`` used as the padding sentinel, so a
    padded lane simply masks off. ``expert_ids[block]`` names the local expert
    whose weight slice the block multiplies against.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # Grouped ordering along M so consecutive programs reuse the same B tiles
    # in L2 -- the expert weight slice is the large operand here.
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # The grid is sized for the worst-case padding; the real extent is only
    # known on device, so blocks past it retire here.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // TOP_K * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    if USE_FP8_W8A8:
        a_scale_ptrs = a_scale_ptr + (offs_token // TOP_K) * stride_asm
        offs_bsn = offs_bn // GROUP_N
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        if USE_FP8_W8A8:
            # BLOCK_K is pinned to GROUP_K by _fp8_config, so each iteration
            # consumes exactly one scale group and the dequant can stay in the
            # epilogue of the dot rather than on the operands.
            offs_ks = (k * BLOCK_K) // GROUP_K
            a_scale = tl.load(
                a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
            )
            b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
            accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        else:
            accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    out = accumulator.to(COMPUTE_TYPE)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, out, mask=token_mask[:, None] & (offs_cn[None, :] < N))


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------


def _align_blocks(
    local_ids: torch.Tensor,
    valid: torch.Tensor,
    block_m: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Group ``(token, slot)`` routes by expert and pad each run to ``block_m``.

    This is the pure-PyTorch equivalent of vLLM/SGLang's ``moe_align_block_size``
    CUDA kernel. It is written to avoid a device-to-host sync: the padded total
    is returned as a device tensor the kernel reads, and the grid is sized from
    the static worst case ``num_routes + num_local_experts * (block_m - 1)``.

    Args:
        local_ids: ``[num_tokens, top_k]`` expert ids already rebased to this
            rank's local range.
        valid: ``[num_tokens, top_k]`` bool; False for routes owned by another
            expert-parallel rank.
        block_m: M tile the runs are padded to.
        num_local_experts: number of experts whose weights live on this rank.

    Returns:
        ``(sorted_token_ids, expert_ids, num_tokens_post_pad, dropped)`` where
        ``sorted_token_ids`` holds flat ``token * top_k + slot`` indices padded
        with the ``num_routes`` sentinel, ``expert_ids`` names the local expert
        per M block, and ``dropped`` says whether any route was masked out (the
        caller must then zero-fill the output cache, since unwritten rows would
        otherwise contribute uninitialised memory to the top-k sum).
    """
    device = local_ids.device
    num_routes = local_ids.numel()
    # Route ids outside this rank land in a sentinel bucket that sorts last and
    # is never emitted, rather than being handled with an ``expert_ids == -1``
    # zero-writing branch in the kernel.
    flat = torch.where(
        valid.reshape(-1),
        local_ids.reshape(-1).to(torch.int64),
        torch.full((), num_local_experts, dtype=torch.int64, device=device),
    )
    order = torch.argsort(flat, stable=True)
    sorted_experts = flat[order]

    counts = torch.bincount(flat, minlength=num_local_experts + 1)[:num_local_experts]
    padded = ((counts + block_m - 1) // block_m) * block_m
    padded_start = torch.cumsum(padded, 0) - padded
    counts_start = torch.cumsum(counts, 0) - counts

    clamped = sorted_experts.clamp(max=num_local_experts - 1)
    position = torch.arange(num_routes, device=device, dtype=torch.int64)
    dest = padded_start[clamped] + (position - counts_start[clamped])

    max_len = num_routes + num_local_experts * (block_m - 1)
    sorted_token_ids = torch.full(
        (max_len,), num_routes, dtype=torch.int32, device=device
    )
    keep = sorted_experts < num_local_experts
    sorted_token_ids[dest[keep]] = order[keep].to(torch.int32)

    num_blocks = (max_len + block_m - 1) // block_m
    block_start = torch.arange(num_blocks, device=device, dtype=torch.int64) * block_m
    padded_end = padded_start + padded
    expert_ids = (
        torch.searchsorted(padded_end, block_start, right=True)
        .clamp(max=num_local_experts - 1)
        .to(torch.int32)
    )
    num_tokens_post_pad = (padded_start[-1] + padded[-1]).reshape(1).to(torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_pad, bool(not valid.all())


def _per_token_group_quant_fp8(
    x: torch.Tensor, group_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, K]`` activations to FP8 with per-``group_k`` row scales.

    Returns ``(x_fp8, scale)`` with ``scale`` shaped ``[M, K // group_k]`` in
    float32 -- the orientation the GEMM's ``stride_asm``/``stride_ask`` expect.

    Deliberately plain PyTorch: this is a memory-bound pass next to two grouped
    GEMMs, and the in-tree ``per_token_group_quant_fp8`` returns a flattened
    scale on this platform (which is what makes
    ``flashinfer_trtllm_fp8_moe_apply`` raise here), so depending on it would
    import that ambiguity.
    """
    m, k = x.shape
    grouped = x.reshape(m, k // group_k, group_k).float()
    amax = grouped.abs().amax(dim=-1).clamp(min=1e-12)
    scale = amax / 448.0
    q = (grouped / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    return q.reshape(m, k).to(torch.float8_e4m3fn), scale.contiguous()


def _clamp_stages(
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    itemsize: int,
    device_index: int,
) -> int:
    """Reduce ``num_stages`` until the pipelined A/B tiles fit this device's SMEM.

    The tuned tables cover one geometry each; anything else falls on the
    heuristics below, and a 100 KB SM will refuse (``OutOfResources``) a tile
    budgeted for a 227 KB one. TensorRT-LLM applies the same guard to its
    sm_120 CUTLASS tiles, keeping only 128x128x64 once
    ``cudaDevAttrMaxSharedMemoryPerBlockOptin`` drops below 120 KB; SGLang
    swaps its MoE defaults below 128 KB. vLLM has no such guard, which is the
    one place its defaults are wrong on this GPU.
    """
    budget = _smem_budget(device_index)
    per_stage = (block_m * block_k + block_k * block_n) * itemsize
    while stages > 2 and per_stage * stages > budget:
        stages -= 1
    return stages


# (max_tokens, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages).
# Transcribed from SGLang's tuned table for E=256, N=512 on this exact device
# (configs/triton_3_6_0/...NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json),
# collapsed to the buckets where the tuner actually changed its answer.
_BF16_TABLE = (
    (1, 32, 32, 64, 32, 4, 4),
    (2, 16, 32, 64, 16, 4, 3),
    (4, 16, 32, 128, 1, 4, 2),
    (8, 16, 64, 128, 16, 8, 2),
    (32, 16, 32, 128, 32, 4, 4),
    (192, 16, 32, 256, 1, 4, 3),
    (256, 16, 128, 128, 32, 8, 3),
    (768, 32, 128, 128, 32, 8, 3),
    (1024, 64, 32, 128, 64, 8, 3),
    (2048, 64, 128, 128, 64, 8, 3),
    (5120, 64, 128, 64, 64, 4, 3),
)
_BF16_LARGE = (128, 256, 64, 64, 8, 3)

# Same, from vLLM's tuned table for the FP8 (128, 128) block-scale recipe on
# this device. BLOCK_K is forced to GROUP_K below regardless of what the table
# says, because this kernel loads exactly one scale group per K iteration.
_FP8_TABLE = (
    (1, 16, 128, 1, 4, 4),
    (8, 32, 128, 16, 4, 4),
    (16, 16, 128, 64, 8, 4),
    (128, 16, 128, 16, 4, 2),
    (512, 32, 128, 16, 4, 2),
    (1024, 32, 128, 64, 8, 3),
    (2048, 64, 128, 64, 4, 3),
)
_FP8_LARGE = (64, 128, 64, 4, 3)


def _bf16_config(num_tokens: int, itemsize: int, device_index: int) -> dict:
    for bound, bm, bn, bk, gm, warps, stages in _BF16_TABLE:
        if num_tokens <= bound:
            break
    else:
        bm, bn, bk, gm, warps, stages = _BF16_LARGE
    return {
        "BLOCK_M": bm,
        "BLOCK_N": bn,
        "BLOCK_K": bk,
        "GROUP_SIZE_M": gm,
        "num_warps": warps,
        "num_stages": _clamp_stages(bm, bn, bk, stages, itemsize, device_index),
    }


def _fp8_config(num_tokens: int, device_index: int) -> dict:
    for bound, bm, bn, gm, warps, stages in _FP8_TABLE:
        if num_tokens <= bound:
            break
    else:
        bm, bn, gm, warps, stages = _FP8_LARGE
    return {
        "BLOCK_M": bm,
        "BLOCK_N": bn,
        "BLOCK_K": _FP8_BLOCK_K,
        "GROUP_SIZE_M": gm,
        "num_warps": warps,
        "num_stages": _clamp_stages(bm, bn, _FP8_BLOCK_K, stages, 1, device_index),
    }


@functools.lru_cache(maxsize=8)
def _compute_type(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    raise NotImplementedError(f"unieinfra sm120 MoE: unsupported input dtype {dtype}")


def _launch_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scale: torch.Tensor | None,
    b_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    num_routes: int,
    top_k: int,
    mul_routed_weight: bool,
    config: dict,
    compute_type,
    use_fp8: bool,
) -> None:
    em = sorted_token_ids.numel()
    n = b.shape[1]
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(em, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _moe_gemm_kernel[grid](
        a,
        b,
        c,
        a_scale,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        n,
        b.shape[2],
        em,
        num_routes,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scale.stride(0) if use_fp8 else 0,
        a_scale.stride(1) if use_fp8 else 0,
        b_scale.stride(0) if use_fp8 else 0,
        b_scale.stride(2) if use_fp8 else 0,
        b_scale.stride(1) if use_fp8 else 0,
        GROUP_N=_FP8_BLOCK_N,
        GROUP_K=_FP8_BLOCK_K,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        TOP_K=top_k,
        COMPUTE_TYPE=compute_type,
        USE_FP8_W8A8=use_fp8,
        **config,
    )


# ---------------------------------------------------------------------------
# Registered entry point
# ---------------------------------------------------------------------------


def unieinfra_triton_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
):
    """Apply routed experts as ``gemm1 -> SwiGLU -> gemm2 -> top-k sum``.

    Args:
        plan: Execution plan from ``moe_plan``; ``weight_dtype`` selects the
            unquantized or FP8 block-scale path.
        x: Hidden states ``[num_tokens, hidden_size]``, fp16 or bf16.
        w: Module holding ``w13_weight`` ``[E_local, 2 * ispp, hidden_size]``
            with gate rows ``[0:ispp]`` and up rows ``[ispp:2*ispp]``, and
            ``w2_weight`` ``[E_local, hidden_size, ispp]``. For the FP8 path,
            also ``w13_weight_scale_inv`` / ``w2_weight_scale_inv``. No weight
            preprocessing is registered, so the loader's ``[gate | up]``
            ordering is consumed as-is.
        router_logits: Unused; this kernel requires precomputed top-k and
            declares ``routing_mode = precomputed_topk`` accordingly.
        topk_weights: ``[num_tokens, top_k]`` routing weights. Required.
        topk_ids: ``[num_tokens, top_k]`` global expert ids. Required.
        num_tokens_global: Unused (no all-to-all EP).
        max_num_tokens_per_gpu: Unused.
        do_finalize: Must be True; deferred finalize is not declared.
        enable_pdl: Unused.

    Returns:
        ``[num_tokens, hidden_size]`` in ``x``'s dtype -- the top-k
        gate-weighted sum over the routed experts. Not reduced across
        tensor-parallel ranks: with intermediate-dim sharding each rank holds a
        partial sum and the caller must all-reduce.

    Raises:
        RuntimeError: if top-k tensors are missing, ``do_finalize`` is False,
            or the weights do not match the plan's declared dtype.
    """
    if topk_weights is None or topk_ids is None:
        raise RuntimeError(
            "unieinfra sm120 MoE requires precomputed topk_weights/topk_ids"
        )
    if not do_finalize:
        raise RuntimeError("unieinfra sm120 MoE does not support deferred finalize")

    use_fp8 = plan.get("weight_dtype") == "fp8"
    w13 = w.w13_weight
    w2 = w.w2_weight
    num_local_experts, two_ispp, hidden_size = w13.shape
    ispp = two_ispp // 2
    num_tokens, top_k = topk_ids.shape

    if x.shape[0] == 0:
        return x.new_empty((0, hidden_size))

    # Expert parallelism: contiguous ownership, so a global id maps to a local
    # one by subtracting this rank's offset. Routes outside the local range are
    # dropped here and their output rows are left at zero.
    ep_size = int(getattr(w, "ep_size", 1))
    ep_rank = int(getattr(w, "ep_rank", 0))
    if ep_size > 1:
        local_ids = topk_ids.to(torch.int64) - ep_rank * num_local_experts
        valid = (local_ids >= 0) & (local_ids < num_local_experts)
        local_ids = torch.where(valid, local_ids, torch.zeros_like(local_ids))
    else:
        local_ids = topk_ids.to(torch.int64)
        valid = torch.ones_like(local_ids, dtype=torch.bool)

    compute_type = _compute_type(x.dtype)
    num_routes = num_tokens * top_k
    flat_weights = topk_weights.reshape(-1).to(torch.float32)

    device_index = x.device.index if x.device.index is not None else 0
    config1 = (
        _fp8_config(num_tokens, device_index)
        if use_fp8
        else _bf16_config(num_tokens, x.element_size(), device_index)
    )
    sorted_ids, expert_ids, post_pad, dropped = _align_blocks(
        local_ids, valid, config1["BLOCK_M"], num_local_experts
    )

    if use_fp8:
        a1, a1_scale = _per_token_group_quant_fp8(x, _FP8_BLOCK_K)
        b1_scale, b2_scale = w.w13_weight_scale_inv, w.w2_weight_scale_inv
    else:
        a1, a1_scale = x, None
        b1_scale = b2_scale = None

    cache1 = torch.empty((num_routes, two_ispp), dtype=x.dtype, device=x.device)
    _launch_gemm(
        a1,
        w13,
        cache1,
        a1_scale,
        b1_scale,
        None,
        sorted_ids,
        expert_ids,
        post_pad,
        num_routes,
        top_k=top_k,
        mul_routed_weight=False,
        config=config1,
        compute_type=compute_type,
        use_fp8=use_fp8,
    )

    # SwiGLU on the [gate | up] halves. Kept as a separate launch, as in vLLM,
    # SGLang and lmdeploy -- the only trees that fuse it into the expert GEMM
    # are the vendor kernels that have no sm_120 build.
    cache2 = F.silu(cache1[:, :ispp]) * cache1[:, ispp:]

    # gemm2's rows are already expert-major, so it consumes the same block map
    # only when BLOCK_M is unchanged; the config is keyed on the same token
    # count, so it is.
    config2 = dict(config1)
    if use_fp8:
        a2, a2_scale = _per_token_group_quant_fp8(cache2.contiguous(), _FP8_BLOCK_K)
    else:
        a2, a2_scale = cache2.contiguous(), None

    cache3 = (
        torch.zeros((num_routes, hidden_size), dtype=x.dtype, device=x.device)
        if dropped
        else torch.empty((num_routes, hidden_size), dtype=x.dtype, device=x.device)
    )
    _launch_gemm(
        a2,
        w2,
        cache3,
        a2_scale,
        b2_scale,
        flat_weights,
        sorted_ids,
        expert_ids,
        post_pad,
        num_routes,
        # A is already one row per (token, slot), so no ``// top_k`` folding.
        top_k=1,
        mul_routed_weight=True,
        config=config2,
        compute_type=compute_type,
        use_fp8=use_fp8,
    )

    return cache3.view(num_tokens, top_k, hidden_size).sum(dim=1)


def register_moe() -> None:
    """Register this module's ``moe/apply`` kernels.

    Safe to call on any platform: registration is skipped with a log line
    rather than an exception when the host is not sm_120, because a plugin that
    raises during discovery takes the host process down with it.
    """
    from tokenspeed_kernel.platform import (
        ArchVersion,
        CapabilityRequirement,
        current_platform,
    )
    from tokenspeed_kernel.registry import Priority, register_kernel
    from tokenspeed_kernel.signature import format_signatures

    platform = current_platform()
    if not (platform.is_nvidia and platform.arch_version == ArchVersion(12, 0)):
        logger.info(
            "unieinfra-kernel-sm120 moe: not registering, platform is %s sm_%s%s",
            platform.vendor,
            platform.arch_version.major,
            platform.arch_version.minor,
        )
        return

    capability = CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 9),
    )
    signatures = format_signatures("x", "dense", {torch.float16, torch.bfloat16})
    shared_traits = {
        "activation": frozenset({"silu", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        # Everything below is "not implemented here", not "off". Selection
        # falls back to the in-tree kernels for a plan that needs any of them.
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({False, True}),
        "supports_all_to_all_ep": frozenset({False}),
        "supports_bias": frozenset({False}),
        "internal_activation_dtype": frozenset({"input"}),
    }

    register_kernel(
        "moe",
        "apply",
        name="unieinfra_triton_fp8_block_moe_apply_sm120",
        solution="unieinfra_triton",
        capability=capability,
        signatures=signatures,
        traits={
            **shared_traits,
            "weight_dtype": frozenset({"fp8"}),
            "fp8_scale_block_shape": frozenset({(_FP8_BLOCK_N, _FP8_BLOCK_K)}),
            # The loader lays w13's block scales out as
            # ``2 * ceil(ispp / 128)`` rows, which only equals
            # ``ceil(2 * ispp / 128)`` when ispp is a multiple of 128. Below
            # that the two halves' scale grids do not tile the N axis and this
            # kernel would index the wrong scale, so require the alignment.
            "ispp_alignment": frozenset({_FP8_BLOCK_N}),
        },
        # Both in-tree FP8 candidates raise on this arch (the SPECIALIZED
        # trtllm entry and the PERFORMANT cutlass one), so this is the only
        # working block-scaled FP8 apply on sm_120.
        priority=Priority.PLUGIN,
        tags={"unieinfra", "sm120"},
    )(unieinfra_triton_moe_apply)

    register_kernel(
        "moe",
        "apply",
        name="unieinfra_triton_unquant_moe_apply_sm120",
        solution="unieinfra_triton",
        capability=capability,
        signatures=signatures,
        traits={
            **shared_traits,
            "weight_dtype": frozenset({"unquant"}),
            "ispp_alignment": frozenset({1}),
        },
        # NOTE: unlike the FP8 entry this displaces a kernel that *works* --
        # flashinfer_cutlass_unquant_moe_apply at PERFORMANT (8), verified on
        # device at cos 0.999996. The case for it is arch-portability, not a
        # measured speedup: it needs no FlashInfer JIT or startup autotune, and
        # its tiles come from SGLang's tuned table for this GPU whereas the
        # cutlass path reports "no tuned config covers ..." and falls back to
        # tactic -1 at the token counts tested. That argument is unproven until
        # somebody runs a throughput A/B; if the A/B says otherwise, drop this
        # registration and keep the FP8 one.
        #
        # So it is registered BELOW the in-tree cutlass kernel (PERFORMANT, 8)
        # rather than in the plugin band. The kernel it would displace works and
        # verifies better (cosine 0.999996 against this one's 0.99998); winning
        # selection on an unmeasured portability argument is how a regression
        # ships. It stays reachable — as a fallback, and for the A/B that would
        # justify promoting it — without being chosen by default.
        priority=Priority.PORTABLE + 3,
        tags={"unieinfra", "sm120"},
    )(unieinfra_triton_moe_apply)

    logger.info(
        "unieinfra-kernel-sm120: registered Triton MoE apply (unquant, fp8 "
        "block-scale) at priority %d",
        int(Priority.PLUGIN),
    )
