"""Quantization kernels for NVIDIA sm_120 (RTX PRO 6000 Blackwell).

Why this exists
---------------
A registry inventory taken on sm_120 shows the quantization family is not
merely running slow here, it is partly *unreachable*:

===========================  ==========================================
``quantization/mxfp4``       no admissible kernel at all
``quantization/nvfp4``       nothing registered at all
``fp8_with_scale`` tensor    no admissible kernel at all
``fp8_with_scale`` token     no admissible kernel at all
``quantization/fp8``         Triton only, PORTABLE band (priority 4)
``quantization/mxfp8``       FlashInfer, PERFORMANT band -- already fine
===========================  ==========================================

The causes are all gating, not hardware:

* ``ops/quantization/triton.py`` registered ``triton_quantize_mxfp4`` with
  ``CapabilityRequirement(vendors=frozenset({"amd"}))``, and it is the only
  MXFP4 entry, so NVIDIA got nothing.  That gate is corrected in the tree
  alongside this module -- together with a scale-byte bug the gate was hiding,
  see below -- but the correction is not sufficient on its own: the Triton
  kernel declares ``scale_layout="linear"`` only, while ``quantize_mxfp4``'s
  own default is ``"128x4"``.  Without the registration here the default call
  still finds nothing.

  The gate was hiding a real defect.  The kernel computed its UE8M0 byte as
  ``scale_e8m0_unbiased.to(tl.uint8) + 127``, and the shared exponent
  ``floor(log2(amax)) - 2`` is negative for any block with ``amax < 4``, which
  is essentially all real activations.  Triton's float-to-uint8 conversion
  saturates negatives to 0 on the NVIDIA backend rather than wrapping, so the
  stored byte collapsed to 127 (scale ``2^0``) while the E2M1 nibbles -- taken
  from the float exponent -- stayed correct.  The tensor dequantized ``2^-e``
  too large with no other symptom; at ``amax ~ 0.023`` that is 600x the input
  RMS.  Fixed in the tree by biasing before converting.
* ``ops/quantization/flashinfer.py`` guards its NVFP4 registration on
  ``platform.is_blackwell``, and ``PlatformInfo.is_blackwell`` is
  ``arch_version.major == 10``.  sm_120 reports major 12, so the whole block
  is skipped and NVFP4 has no kernel -- even though FlashInfer ships a
  purpose-built ``sm120a`` FP4 quantization module.
* ``ops/quantization/trtllm.py`` caps its FP8 entry at
  ``max_arch_version=ArchVersion(10, 9)``.  That cap is *correct* and this
  plugin does not try to lift it: every TRT-LLM FP8 quantize op
  (``quantize_e4m3_per_tensor``, ``quantize_e4m3_activation``,
  ``fp8_quantize_1x128``) fails on this card with CUDA error 209,
  ``cudaErrorNoKernelImageForDevice``, because the shipped
  ``libtrtllm_kernel.so`` has no sm_120 cubin and no forward-compatible PTX.
  Worse, the launch return code is not checked, so the op returns
  *uninitialized* output instead of raising and leaves a pending CUDA error
  that detonates on the next unrelated torch op.  The cap is the only thing
  standing between sm_120 and silently corrupt activations.

So this module fills the reachability holes with FlashInfer's sm_120 FP4
kernels and with Triton for the FP8 granularities nothing covers, and
replaces the in-tree FP8 cast with one that does not degrade on wide rows.

Honest traits
-------------
Every registration below declares only what it implements and has been
checked against a float32 reference.  In particular ``scale_encoding="ue8m0"``
for ``fp8_with_scale`` is *not* claimed: the public contract leaves its
encoded integer layout backend-specific, there is no consumer on this card to
validate against, and a kernel that wins selection and then hands back a
differently-encoded scale is worse than no kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["register_quant"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _flatten_to_2d(x: torch.Tensor) -> tuple[int, int, int]:
    """Return ``(M, N, row_stride)`` for a tensor viewed as rows of size N.

    Mirrors the in-tree helper's contract: ``stride(-1)`` must be 1 and every
    leading dim must pack onto the next, which holds for contiguous tensors
    and for last-dim slice views.
    """
    if x.stride(-1) != 1:
        raise ValueError(f"expected stride-1 inner dim, got stride={x.stride(-1)}")
    n = x.shape[-1]
    if x.ndim == 1:
        return 1, n, n
    m = x.numel() // n
    for d in range(x.ndim - 2):
        expected = x.shape[d + 1] * x.stride(d + 1)
        if x.stride(d) != expected:
            raise ValueError(
                f"cannot flatten dim {d}: stride={x.stride(d)} but expected "
                f"shape[{d + 1}]*stride[{d + 1}]={expected}"
            )
    return m, n, x.stride(-2)


# ---------------------------------------------------------------------------
# FP8 cast  (quantization/fp8)
# ---------------------------------------------------------------------------


@triton.jit
def _fp8_cast_flat_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    n_elements,
    FP8_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SCALE_TENSOR: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Flat elementwise FP8 cast over a contiguous buffer.

    The in-tree kernel tiles as ``BLOCK_M x next_pow2(N)``, so one program
    holds a whole padded row.  At N=7168 that is 8192 columns times BLOCK_M,
    which is far past the register budget and spills.  Indexing flat decouples
    the tile from the row width entirely, so cost tracks ``numel`` and nothing
    else.  It costs contiguity, which the caller-side wrapper enforces.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    if HAS_SCALE:
        if HAS_SCALE_TENSOR:
            inv = 1.0 / tl.load(scale_ptr)
        else:
            inv = 1.0 / scale_ptr
        x = x * inv
    tl.store(out_ptr + offs, x.to(FP8_DTYPE), mask=mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


_TL_FP8 = {
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
}


def quantize_fp8_flat_sm120(
    x: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Cast a BF16/FP16 tensor to FP8-E4M3 with an optional scalar scale.

    Computes ``out = saturate_e4m3(x / scale)`` elementwise, or a pure cast
    when ``scale`` is omitted.  Semantics are identical to the in-tree Triton
    kernel; only the launch geometry differs.

    Args:
        x: BF16 or FP16 tensor whose last dim has stride 1 and whose leading
            dims pack onto the row stride.
        scale: Optional scalar divisor, a Python float or a scalar tensor.
        enable_pdl: Opt into Programmatic Dependent Launch.

    Returns:
        FP8-E4M3 tensor with the same shape as ``x``.
    """
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"fp8 quantize input must be bf16/fp16, got {x.dtype}")

    m, n, row_stride = _flatten_to_2d(x)
    # The flat indexing is only valid over a gap-free buffer. A last-dim slice
    # view has row_stride != n; densify it rather than silently reading the
    # neighbouring columns.
    src = x if row_stride == n else x.contiguous()

    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    n_elements = src.numel()

    has_scale = scale is not None
    has_scale_tensor = isinstance(scale, torch.Tensor)
    if has_scale_tensor:
        if scale.numel() != 1:
            raise ValueError("scale tensor must be scalar")
        scale = scale.contiguous()

    # BLOCK=2048/num_warps=4 from an interleaved A/B sweep over
    # {1024, 2048, 4096, 8192} x {4, 8} on this card. 1024 is the only setting
    # that loses to the in-tree kernel anywhere (roughly 2x at [16384, 128],
    # where the grid gets launch-bound); 2048 reaches parity there and matches
    # the best config everywhere else.
    block = 2048
    grid = (triton.cdiv(n_elements, block),)
    extra = {"launch_pdl": True} if enable_pdl else {}
    _fp8_cast_flat_kernel[grid](
        src,
        out,
        1.0 if scale is None else scale,
        n_elements,
        FP8_DTYPE=_TL_FP8[torch.float8_e4m3fn],
        BLOCK=block,
        HAS_SCALE=has_scale,
        HAS_SCALE_TENSOR=has_scale_tensor,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        num_stages=2,
        **extra,
    )
    return out


# ---------------------------------------------------------------------------
# FP8 dynamic quantization  (quantization/fp8_with_scale)
# ---------------------------------------------------------------------------


@triton.jit
def _row_amax_kernel(x_ptr, amax_ptr, N, x_row_stride, BLOCK: tl.constexpr):
    """Per-row |max| into an fp32 ``[M]`` vector, looping over N in tiles."""
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row * x_row_stride + offs, mask=offs < N, other=0.0)
        acc = tl.maximum(acc, tl.abs(v.to(tl.float32)))
    tl.store(amax_ptr + row, tl.max(acc, axis=0))


@triton.jit
def _fp8_apply_scale_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    N,
    x_row_stride,
    out_row_stride,
    FP8_MIN,
    FP8_MAX,
    PER_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Divide each row by its scale and clamp into E4M3 range."""
    row = tl.program_id(0)
    if PER_TOKEN:
        scale = tl.load(scale_ptr + row)
    else:
        scale = tl.load(scale_ptr)
    inv = 1.0 / scale
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        v = tl.load(x_ptr + row * x_row_stride + offs, mask=mask).to(tl.float32)
        q = tl.clamp(v * inv, FP8_MIN, FP8_MAX)
        tl.store(out_ptr + row * out_row_stride + offs, q.to(tl.float8e4nv), mask=mask)


# Floor on the amax so an all-zero row yields a finite, invertible scale
# instead of 0. Matches the in-tree token-group kernel's constant.
_AMAX_EPS = 1e-10


def quantize_fp8_with_scale_sm120(
    x: torch.Tensor,
    granularity: str = "tensor",
    group_size: int | None = None,
    scale_encoding: str = "float32",
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize to FP8-E4M3 at per-tensor or per-token granularity.

    The scheme is the standard one: ``scale = max(|x|) / 448`` over the
    reduction domain, then ``q = clamp(x / scale, -448, 448)`` rounded to
    nearest even by the FP8 conversion.

    Args:
        x: BF16 or FP16 tensor. Reduced as ``[M, N]`` with
            ``M = x.numel() // x.shape[-1]``.
        granularity: ``"tensor"`` for one scale overall, ``"token"`` for one
            scale per row. ``"token_group"`` is deliberately not handled here;
            the in-tree Triton kernel already covers it on this card.
        group_size: Unused for the granularities this kernel serves.
        scale_encoding: Must be ``"float32"``.
        enable_pdl: Accepted for launch-site parity; unused.

    Returns:
        ``(q, scale)`` where ``q`` is FP8-E4M3 with the shape of ``x`` and
        ``scale`` is float32, shaped ``[1]`` for tensor granularity and
        ``[*x.shape[:-1], 1]`` for token granularity.
    """
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"fp8 dynamic quantize input must be bf16/fp16, got {x.dtype}")
    if granularity not in {"tensor", "token"}:
        raise ValueError(
            "unieinfra sm120 FP8 dynamic quantization serves granularity "
            f"'tensor' or 'token', got {granularity!r}"
        )
    if scale_encoding != "float32":
        raise ValueError(
            f"unieinfra sm120 FP8 dynamic quantization requires "
            f"scale_encoding='float32', got {scale_encoding!r}"
        )

    m, n, x_row_stride = _flatten_to_2d(x)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    _, _, out_row_stride = _flatten_to_2d(out)

    block = min(1024, max(32, triton.next_power_of_2(n)))
    num_warps = max(1, min(8, block // 256))

    row_amax = torch.empty(m, dtype=torch.float32, device=x.device)
    _row_amax_kernel[(m,)](
        x, row_amax, n, x_row_stride, BLOCK=block, num_warps=num_warps
    )

    if granularity == "token":
        scale = row_amax.clamp_min(_AMAX_EPS) / fp8_max
        per_token = True
        out_scale = scale.reshape(*x.shape[:-1], 1)
    else:
        # A max over the [M] row-amax vector, not over [M, N] -- no second
        # full pass across the activations.
        scale = (row_amax.amax().clamp_min(_AMAX_EPS) / fp8_max).reshape(1)
        per_token = False
        out_scale = scale

    _fp8_apply_scale_kernel[(m,)](
        x,
        out,
        scale,
        n,
        x_row_stride,
        out_row_stride,
        -fp8_max,
        fp8_max,
        PER_TOKEN=per_token,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out, out_scale


# ---------------------------------------------------------------------------
# MXFP4 / NVFP4  (quantization/mxfp4, quantization/nvfp4)
# ---------------------------------------------------------------------------

# ``fp4_quantize`` takes a global scale unconditionally, but in UE8M0 mode the
# kernel never reads it: the branch computes the shared exponent purely from
# the block amax (``vecMax *= 1/6; __nv_cvt_float_to_e8m0(vecMax, SATFINITE,
# cudaRoundPosInf)``).  Verified on device -- quantizing the same input with
# global scales 1.0 and 137.0 gives byte-identical data *and* scale factors.
# So passing a constant 1.0 here is inert, not an unstated assumption.
_MXFP4_INERT_GLOBAL_SCALE: dict[torch.device, torch.Tensor] = {}


def _inert_global_scale(device: torch.device) -> torch.Tensor:
    t = _MXFP4_INERT_GLOBAL_SCALE.get(device)
    if t is None:
        t = torch.ones(1, dtype=torch.float32, device=device)
        _MXFP4_INERT_GLOBAL_SCALE[device] = t
    return t


def quantize_mxfp4_sm120(
    x: torch.Tensor,
    global_scale: float | None = None,
    scale_size: int = 32,
    scale_layout: str = "128x4",
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to packed MXFP4 via FlashInfer's sm_120 FP4 kernel.

    One UE8M0 scale factor per 32 values along the last dim, two E2M1 values
    packed per output byte.

    Args:
        x: BF16 or FP16 tensor, contiguous, last dim divisible by 32.
        global_scale: Must be ``None``. MXFP4 has no global scale, and the
            UE8M0 path ignores one if given.
        scale_size: Must be 32; UE8M0 scale factors are only defined for
            32-element blocks in this kernel.
        scale_layout: ``"linear"`` for unswizzled ``[M, K/32]`` scales,
            ``"128x4"`` or ``"8x4"`` for the swizzled layouts the FP4 GEMM and
            MoE kernels consume directly.
        enable_pdl: Opt into Programmatic Dependent Launch.

    Returns:
        ``(packed, scales)``. ``packed`` is uint8 ``[*x.shape[:-1], K/2]``.
        ``scales`` is uint8: exactly ``[M, K/32]`` for ``"linear"``, and the
        padded swizzled buffer (rows to a multiple of 128 or 8, columns to a
        multiple of 4) reshaped to ``[-1, round_up(K/32, 4)]`` otherwise.
    """
    from flashinfer import fp4_quantize

    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"mxfp4 quantize input must be bf16/fp16, got {x.dtype}")
    if global_scale is not None:
        raise ValueError("MXFP4 quantization does not take a global scale")
    if scale_size != 32:
        raise ValueError(
            f"MXFP4 UE8M0 scale factors require scale_size=32, got {scale_size}"
        )
    if scale_layout not in {"linear", "128x4", "8x4"}:
        raise ValueError(f"unsupported MXFP4 scale_layout: {scale_layout!r}")
    if x.shape[-1] % scale_size != 0:
        raise ValueError(
            f"MXFP4 requires the last dim divisible by {scale_size}, got {x.shape[-1]}"
        )
    # fp4_quantize treats stride(-2) == 1 as column-major and transposes. A
    # non-contiguous view would silently be quantized along the wrong axis.
    src = x if x.is_contiguous() else x.contiguous()

    return fp4_quantize(
        src,
        _inert_global_scale(src.device),
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=scale_layout != "linear",
        is_sf_8x4_layout=scale_layout == "8x4",
        enable_pdl=enable_pdl,
    )


def quantize_nvfp4_sm120(
    x: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    scale_layout: str = "swizzled",
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to packed NVFP4 via FlashInfer's sm_120 FP4 kernel.

    One E4M3 scale factor per 16 values, two E2M1 values packed per byte.

    ``scale`` is the actual input scale, matching the public API. FlashInfer
    expects the inverse, so it is reciprocated here -- the same convention the
    in-tree ``flashinfer_quantize_nvfp4`` uses on sm_100, kept identical so
    the two architectures do not disagree about what ``scale`` means.

    Args:
        x: BF16 or FP16 tensor, contiguous, last dim divisible by 16.
        scale: Scalar input scale, required.
        scale_layout: ``"swizzled"`` for the 128x4 layout the FP4 GEMM
            consumes, ``"linear"`` for unswizzled ``[M, K/16]`` scales.
        enable_pdl: Opt into Programmatic Dependent Launch.

    Returns:
        ``(packed, scale_factors)``; ``packed`` is uint8 ``[*x.shape[:-1], K/2]``.
    """
    from flashinfer import fp4_quantize

    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"nvfp4 quantize input must be bf16/fp16, got {x.dtype}")
    if scale is None:
        raise ValueError("NVFP4 quantization requires a global scale")
    if scale_layout not in {"linear", "swizzled"}:
        raise ValueError(f"unsupported NVFP4 scale_layout: {scale_layout!r}")
    if x.shape[-1] % 16 != 0:
        raise ValueError(
            f"NVFP4 requires the last dim divisible by 16, got {x.shape[-1]}"
        )

    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError("NVFP4 scale must be scalar")
        scale_inv = (1.0 / scale.to(dtype=torch.float32, device=x.device)).reshape(1)
    else:
        scale_inv = torch.full(
            (1,), 1.0 / float(scale), dtype=torch.float32, device=x.device
        )

    src = x if x.is_contiguous() else x.contiguous()
    return fp4_quantize(
        src,
        scale_inv,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=scale_layout == "swizzled",
        enable_pdl=enable_pdl,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_quant() -> None:
    """Register this module's quantization kernels.

    Safe to call on any platform: on anything that is not NVIDIA sm_12x it
    logs and returns rather than raising, because a plugin that raises during
    discovery takes the host process down with it.
    """
    import logging

    from tokenspeed_kernel.platform import (
        ArchVersion,
        CapabilityRequirement,
        current_platform,
    )
    from tokenspeed_kernel.registry import Priority, register_kernel
    from tokenspeed_kernel.signature import format_signatures

    logger = logging.getLogger(__name__)

    platform = current_platform()
    if not (platform.is_nvidia and platform.arch_version == ArchVersion(12, 0)):
        logger.info(
            "unieinfra-kernel-sm120 quant: not registering, platform is %s sm_%s%s",
            platform.vendor,
            platform.arch_version.major,
            platform.arch_version.minor,
        )
        return

    # AMD CDNA5 also reports ArchVersion(12, 5), so the arch window alone is
    # not enough to keep these NVIDIA-only kernels off that card.
    sm12x = CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 9),
    )
    sigs = format_signatures("x", "dense", {torch.bfloat16, torch.float16})

    register_kernel(
        "quantization",
        "mxfp4",
        name="unieinfra_flashinfer_quantize_mxfp4_sm120",
        solution="flashinfer",
        capability=sm12x,
        signatures=sigs,
        # The only kernel that serves the API's default "128x4" layout here,
        # and 2.5-68x faster than the in-tree Triton entry on the "linear"
        # layout they share -- that one launches (M, N/32) programs of 32
        # elements at num_warps=1.
        priority=Priority.PLUGIN,
        traits={
            "scale_size": frozenset({32}),
            "scale_layout": frozenset({"linear", "128x4", "8x4"}),
            # MXFP4 has no global scale and the UE8M0 path ignores one.
            "has_global_scale": frozenset({False}),
            "scale_encoding": frozenset({"ue8m0"}),
        },
        tags={"unieinfra", "sm120"},
    )(quantize_mxfp4_sm120)

    register_kernel(
        "quantization",
        "nvfp4",
        name="unieinfra_flashinfer_quantize_nvfp4_sm120",
        solution="flashinfer",
        capability=sm12x,
        signatures=sigs,
        priority=Priority.PLUGIN,
        traits={
            "scale_layout": frozenset({"linear", "swizzled"}),
            "has_scale": frozenset({True}),
        },
        tags={"unieinfra", "sm120"},
    )(quantize_nvfp4_sm120)

    register_kernel(
        "quantization",
        "fp8_with_scale",
        name="unieinfra_triton_quantize_fp8_with_scale_sm120",
        solution="triton",
        capability=sm12x,
        signatures=sigs,
        priority=Priority.PLUGIN,
        traits={
            # token_group is left to the in-tree Triton kernel, which already
            # covers it here. ue8m0 is left unclaimed: its encoded integer
            # layout is backend-specific and unverified on this card.
            "granularity": frozenset({"tensor", "token"}),
            "scale_encoding": frozenset({"float32"}),
        },
        tags={"unieinfra", "sm120"},
    )(quantize_fp8_with_scale_sm120)

    register_kernel(
        "quantization",
        "fp8",
        name="unieinfra_triton_quantize_fp8_sm120",
        solution="triton",
        capability=sm12x,
        signatures=sigs,
        # Same numerics as the in-tree PORTABLE kernel, flat launch geometry.
        priority=Priority.PLUGIN,
        traits={"has_scale": frozenset({True, False})},
        tags={"unieinfra", "sm120"},
    )(quantize_fp8_flat_sm120)

    logger.info(
        "unieinfra-kernel-sm120: registered mxfp4, nvfp4, fp8 and fp8_with_scale "
        "quantization at priority %d",
        int(Priority.PLUGIN),
    )
