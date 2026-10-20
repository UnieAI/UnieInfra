# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Out-of-tree GEMM kernels for NVIDIA sm_120 (RTX PRO 6000 Blackwell).

Two gaps in the in-tree ``gemm`` family on sm_120, both measured on an
RTX PRO 6000 Blackwell Server Edition:

``gemm.bmm``
    The tree registers *only* reference kernels for batched matmul. For dense
    bf16 that is fine -- ``torch.bmm`` is cuBLAS strided-batched and a Triton
    replacement loses to it at every shape measured here. For **FP8** it is
    not: ``torch_bmm_fp8_scaled`` materializes ``A.float()`` and ``B.float()``
    and runs an FP32 ``torch.bmm``, which costs 2.25 ms where a real FP8
    batched GEMM costs 0.18 ms (B=16, M=1024, N=5120, K=512). FlashInfer's
    ``bmm_fp8`` runs on sm_120 and closes that 12x gap.

``gemm.mm`` FP8 block-scale
    ``flashinfer_mm_fp8_blockscale`` (SPECIALIZED+3) outranks
    ``triton_mm_fp8_blockscale`` (PERFORMANT+3) at *every* M, because priority
    is a per-kernel constant and the request traits carry only N/K alignment --
    there is no way for an in-tree registration to say "I win above M=256".
    Measured on three Qwen3.5-27B weight shapes, Triton is 1.04x-2.06x faster
    for M <= 256 and FlashInfer is 1.01x-1.32x faster from M >= 320. This
    plugin registers one kernel that picks between them at call time.

Both registrations declare narrow traits on purpose: a kernel that claims a
shape it cannot serve wins selection and then produces wrong output, which is
worse than not being selected.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["register_gemm"]

# M at or below which the Triton FP8 block-scale GEMM beats FlashInfer's.
# Measured crossover is between 256 and 320 on all three of N=5120/K=5120,
# N=34816/K=5120 and N=5120/K=17408; 256 is the last measured Triton win.
_FP8_BLOCKSCALE_TRITON_MAX_M = 256

# ``bmm_fp8`` costs a flat ~0.065-0.090 ms per call on sm_120 (cuBLASLt
# heuristic plus FlashInfer workspace setup) almost independently of shape,
# while the FP32-dequant reference is dominated by materializing ``B.float()``
# and so scales with ``batch * N * K``. Below these thresholds the reference is
# up to 1.3x *faster*, on problems small enough that the absolute difference is
# under 30 us; above them the FP8 path wins by 1.2x-15x. Both conditions were
# fitted to a measured sweep over batch/M/N/K -- see docs/kernel-provenance.md.
_BMM_FP8_MIN_WEIGHT_ELEMS = 8 * 1024 * 1024
_BMM_FP8_MIN_M = 256


# ---------------------------------------------------------------------------
# gemm.bmm -- FP8 per-tensor-scaled batched GEMM
# ---------------------------------------------------------------------------


def unieinfra_bmm_fp8_scaled_sm120(
    A,
    B,
    A_scales,
    B_scales,
    out_dtype,
    *,
    alpha=None,
    block_size=None,
    out=None,
):
    """Per-tensor-scaled FP8 batched GEMM via FlashInfer's cuBLASLt path.

    Replaces the in-tree ``torch_bmm_fp8_scaled`` reference, which dequantizes
    to FP32 and runs ``torch.bmm`` in FP32. Problems too small to clear
    cuBLASLt's per-call cost are delegated back to that reference; see
    ``_BMM_FP8_MIN_WEIGHT_ELEMS``.

    Args:
        A: Activations ``[batch, M, K]``, ``float8_e4m3fn``, row-major.
        B: Weights ``[batch, N, K]``, ``float8_e4m3fn``, row-major. Passed to
            cuBLASLt as the transposed view ``[batch, K, N]``; the underlying
            buffer must be ``[batch, N, K]``-contiguous, so this must not be
            materialized as a contiguous ``[batch, K, N]`` tensor (doing so is
            accepted by the shape check and silently returns garbage).
        A_scales: Scalar activation scale, ``numel() == 1``.
        B_scales: Scalar weight scale, ``numel() == 1``.
        out_dtype: ``torch.bfloat16`` or ``torch.float16``.
        alpha: Not supported; must be ``None``.
        block_size: Not applicable to per-tensor scaling; must be ``None``.
        out: Optional ``[batch, M, N]`` output buffer. May be a strided view
            with contiguous rows.

    Returns:
        ``[batch, M, N]`` tensor of ``out_dtype``.
    """
    import torch
    from flashinfer.gemm import bmm_fp8

    if A.ndim != 3 or B.ndim != 3:
        raise ValueError(
            f"expected 3D inputs, got A={tuple(A.shape)} B={tuple(B.shape)}"
        )
    if block_size is not None:
        raise ValueError("block_size is not supported for per-tensor FP8 BMM")
    if alpha is not None:
        # Neither cuBLASLt call site applies a global alpha, and folding it
        # into A_scales would change the rounding of the scale itself.
        raise ValueError("alpha is not supported by unieinfra_bmm_fp8_scaled_sm120")
    if A_scales is None or B_scales is None:
        raise ValueError("A_scales and B_scales are required")
    if A_scales.numel() != 1 or B_scales.numel() != 1:
        raise ValueError(
            "unieinfra_bmm_fp8_scaled_sm120 handles per-tensor scales only, got "
            f"A_scales={tuple(A_scales.shape)} B_scales={tuple(B_scales.shape)}"
        )
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"unsupported out_dtype {out_dtype}")

    batch, M, K = A.shape
    B_batch, N, B_K = B.shape
    if B_batch != batch or B_K != K:
        raise ValueError(f"shape mismatch: A={tuple(A.shape)} B={tuple(B.shape)}")

    # Small problems do not clear cuBLASLt's fixed per-call cost; hand them back
    # to the reference rather than shipping a regression on the decode-shaped
    # MLA absorb (batch 32, M 1, N 512, K 256), which is exactly one of them.
    if batch * N * K < _BMM_FP8_MIN_WEIGHT_ELEMS and M < _BMM_FP8_MIN_M:
        from tokenspeed_kernel.registry import KernelRegistry

        reference = KernelRegistry.get().get_impl("torch_bmm_fp8_scaled")
        if reference is not None:
            return reference(
                A,
                B,
                A_scales,
                B_scales,
                out_dtype,
                alpha=None,
                block_size=None,
                out=out,
            )

    # cuBLASLt FP8 requires the TN layout: A row-major [M, K], B column-major
    # [K, N]. B arriving as row-major [N, K] is exactly a column-major [K, N],
    # so the transpose is a free view -- but only if B's own rows are
    # contiguous.
    if B.stride(-1) != 1:
        raise ValueError(
            "B must be [batch, N, K] with contiguous rows (stride(-1) == 1), got "
            f"strides {tuple(B.stride())}"
        )
    if A.stride(-1) != 1:
        A = A.contiguous()

    if out is None:
        out = torch.empty((batch, M, N), device=A.device, dtype=out_dtype)

    bmm_fp8(A, B.transpose(1, 2), A_scales, B_scales, out_dtype, out)
    return out


# ---------------------------------------------------------------------------
# gemm.mm -- FP8 (128,128) block-scale GEMM with an M-aware backend split
# ---------------------------------------------------------------------------


def unieinfra_mm_fp8_blockscale_sm120(
    A,
    B,
    A_scales,
    B_scales,
    out_dtype,
    *,
    alpha=None,
    block_size=None,
    out=None,
):
    """FP8 (128,128) block-scale GEMM that picks Triton or FlashInfer by M.

    Both backends are the in-tree kernels, called through the registry, so
    this adds a dispatch decision and no new numerics.

    Because this kernel declares the ``self_quantizes_mxfp8`` trait, dispatch
    hands it unquantized activations with ``A_scales=None`` rather than
    routing through ``_online_quantize_mxfp8`` (which dispatches on kernel
    name and does not know out-of-tree names). The quantization performed here
    is the same one the in-tree dispatcher already applies on sm_120: Triton,
    token-group, float32 scales -- which is what both backends consume.

    Args:
        A: Activations ``[M, K]``. Either ``float8_e4m3fn`` with ``A_scales``
            supplied, or an unquantized ``bfloat16``/``float16`` tensor with
            ``A_scales=None`` for online quantization.
        B: Weights ``[N, K]``, ``float8_e4m3fn``.
        A_scales: ``[M, K // block_k]`` float32 scales, or ``None``.
        B_scales: ``[N // block_n, K // block_k]`` float32 scales.
        out_dtype: Output dtype.
        alpha: Forwarded to the selected backend unchanged.
        block_size: ``[block_n, block_k]``; required.
        out: Optional ``[M, N]`` output buffer.

    Returns:
        ``[M, N]`` tensor of ``out_dtype``.
    """
    from tokenspeed_kernel.registry import KernelRegistry

    if block_size is None:
        raise ValueError("block_size is required for FP8 block-scale GEMM")
    if B_scales is None:
        raise ValueError("B_scales is required for FP8 block-scale GEMM")

    if A_scales is None:
        from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

        A, A_scales = quantize_fp8_with_scale(
            A,
            granularity="token_group",
            group_size=block_size[1],
            scale_encoding="float32",
            solution="triton",
        )

    registry = KernelRegistry.get()
    name = (
        "triton_mm_fp8_blockscale"
        if A.shape[0] <= _FP8_BLOCKSCALE_TRITON_MAX_M
        else "flashinfer_mm_fp8_blockscale"
    )
    impl = registry.get_impl(name)
    if impl is None:  # pragma: no cover - guarded at registration time
        raise RuntimeError(f"backing kernel {name!r} is not registered")

    kwargs = {"alpha": alpha, "block_size": block_size}
    if out is not None:
        kwargs["out"] = out
    return impl(A, B, A_scales, B_scales, out_dtype, **kwargs)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_gemm() -> None:
    """Register this plugin's ``gemm`` kernels.

    Registration is skipped, with a log line rather than an exception, on any
    platform or build this plugin was not written for. A plugin that raises
    during discovery takes the host process down with it.
    """
    import torch
    from tokenspeed_kernel.platform import (
        ArchVersion,
        CapabilityRequirement,
        current_platform,
    )
    from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
    from tokenspeed_kernel.signature import ScaleFormat, format_signatures

    platform = current_platform()
    if not (platform.is_nvidia and platform.arch_version.major == 12):
        logger.info(
            "unieinfra-kernel-sm120 gemm: not registering, platform is %s sm_%s%s",
            platform.vendor,
            platform.arch_version.major,
            platform.arch_version.minor,
        )
        return

    # PlatformInfo.fp8e4m3fn went away upstream with gfx942 (#889), the one
    # target whose FP8 was e4m3fnuz; on NVIDIA it is always float8_e4m3fn.
    fp8_dtype = torch.float8_e4m3fn
    capability = CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 9),
    )
    registry = KernelRegistry.get()

    # ---- gemm.bmm: FP8 per-tensor scaled ----
    try:
        from flashinfer.gemm import bmm_fp8 as _bmm_fp8  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the FlashInfer build
        logger.info(
            "unieinfra-kernel-sm120 gemm: skipping bmm, flashinfer.gemm.bmm_fp8 "
            "is unavailable (%s)",
            exc,
        )
    else:
        register_kernel(
            "gemm",
            "bmm",
            name="unieinfra_bmm_fp8_scaled_sm120",
            solution="flashinfer",
            capability=capability,
            # Per-tensor only. The channel-granularity signature that
            # torch_bmm_fp8_scaled also serves stays with the reference:
            # bmm_fp8 takes scalar scales and cannot express per-row scaling.
            signatures=format_signatures(
                ("a", "b"),
                "scaled-fp8",
                {fp8_dtype},
                scale=ScaleFormat(storage_dtype=torch.float32, granularity="tensor"),
            ),
            # cuBLASLt FP8 rejects N or K that are not multiples of 16
            # ("the requested functionality is not supported"). Selection falls
            # back to the reference for those shapes.
            traits={
                "n_align_16": frozenset({True}),
                "k_align_16": frozenset({True}),
            },
            # Beats the in-tree reference (PORTABLE, 4) by ~12x at large
            # batched shapes; there is no other bmm kernel to outrank.
            priority=Priority.PLUGIN,
        )(unieinfra_bmm_fp8_scaled_sm120)
        logger.info(
            "unieinfra-kernel-sm120 gemm: registered "
            "unieinfra_bmm_fp8_scaled_sm120 at priority %d",
            int(Priority.PLUGIN),
        )

    # ---- gemm.mm: FP8 (128,128) block-scale, M-aware backend split ----
    missing = [
        name
        for name in ("triton_mm_fp8_blockscale", "flashinfer_mm_fp8_blockscale")
        if registry.get_impl(name) is None
    ]
    if missing:
        logger.info(
            "unieinfra-kernel-sm120 gemm: skipping mm, backing kernels %s are "
            "not registered",
            ", ".join(missing),
        )
        return

    register_kernel(
        "gemm",
        "mm",
        name="unieinfra_mm_fp8_blockscale_sm120",
        solution="unieinfra",
        capability=capability,
        signatures=format_signatures(
            ("a", "b"),
            "mxfp8",
            {fp8_dtype},
            scale=ScaleFormat(
                storage_dtype=torch.float32,
                granularity="block",
                block_shape=(128, 128),
            ),
        ),
        traits={
            # Inherited from the FlashInfer backend used above the split.
            "n_align_128": frozenset({True}),
            "k_align_128": frozenset({True}),
            # Take unquantized activations and A_scales=None; see
            # _kernel_handles_online_mxfp8 in tokenspeed_kernel.ops.gemm.
            "self_quantizes_mxfp8": frozenset({True}),
        },
        priority=Priority.PLUGIN,
    )(unieinfra_mm_fp8_blockscale_sm120)
    logger.info(
        "unieinfra-kernel-sm120 gemm: registered "
        "unieinfra_mm_fp8_blockscale_sm120 at priority %d (Triton for M <= %d, "
        "FlashInfer above)",
        int(Priority.PLUGIN),
        _FP8_BLOCKSCALE_TRITON_MAX_M,
    )
