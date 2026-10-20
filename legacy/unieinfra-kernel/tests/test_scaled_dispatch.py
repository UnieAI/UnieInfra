# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Format-signature and dispatch tests for the quantized (fp8 / nvfp4) path.

All CPU-only: signature construction and kernel selection are pure Python, so
these protect the dispatch contract on any box. The numeric half lives in
``tests/kernels/gemm/test_decode_gemv_scaled.py``.

The invariant these tests exist to hold is: **a request the registry has no
kernel for must resolve to nothing**, so the engine runs its own quantized GEMM.
For the quantized path that is stricter than for the dense one — the dense
façade can safely fall through to ``F.linear`` (what the incumbent kernel calls
anyway), whereas a quantized fallback inside our kernel would be a slow
dequantize the engine would never have run.
"""

from __future__ import annotations

import pytest
import torch
import unieinfra_kernel  # noqa: F401  (import triggers kernel discovery)
from unieinfra_kernel.boundary import (
    ArchVersion,
    KernelRegistry,
    NoKernelFoundError,
    PlatformInfo,
    ScaleFormat,
    _get_cuda_sm_features,
    dense_tensor_format,
    format_signature,
)
from unieinfra_kernel.boundary import select_kernel as _select_kernel
from unieinfra_kernel.boundary import (
    tensor_format,
)
from unieinfra_kernel.contracts.gemm import (
    W4A16_NVFP4_SIGNATURES,
    W8A16_FP8_SIGNATURES,
)
from unieinfra_kernel.dispatch import (
    SCALED_KINDS,
    gemm_format_signature,
    shape_traits,
)

FP8_KERNEL = "unieinfra_decode_gemv_fp8_triton"
NVFP4_KERNEL = "unieinfra_decode_gemv_nvfp4_triton"

# Qwen3-8B decode projections (hidden 4096).
QWEN3_8B_SHAPES = [
    ("qkv_proj", 6144, 4096),
    ("o_proj", 4096, 4096),
    ("gate_up_proj", 24576, 4096),
    ("down_proj", 4096, 12288),
]


def _meta(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A zero-cost stand-in tensor: only dtype/shape reach the signature."""
    return torch.empty(shape, dtype=dtype, device="meta")


# Selection is pure Python, but it needs *a* platform. Synthesizing sm_120
# keeps these tests runnable on a CPU-only box and pins the arch the kernels
# were measured on.
SM120 = PlatformInfo(
    vendor="nvidia",
    arch_version=ArchVersion(12, 0),
    device_name="synthetic-sm_120",
    device_count=1,
    total_memory=0,
    memory_bandwidth=0.0,
    sm_count=188,
    max_threads_per_sm=0,
    max_shared_memory_per_sm=0,
    # Derived from the real detector, never hand-copied (E30).
    sm_features=_get_cuda_sm_features(ArchVersion(12, 0)),
)


def select_kernel(family, mode, signature, traits=None):
    """``select_kernel`` pinned to the synthetic sm_120 platform."""
    return _select_kernel(family, mode, signature, traits=traits, platform=SM120)


# ---------------------------------------------------------------------------
# gemm_format_signature
# ---------------------------------------------------------------------------


def test_dense_branch_matches_the_dense_facade():
    """``quant=None`` reproduces exactly what :func:`linear` builds."""
    a = _meta((1, 4096), torch.bfloat16)
    b = _meta((6144, 4096), torch.bfloat16)
    assert gemm_format_signature(a, b, None, None, None) == format_signature(
        a=dense_tensor_format(torch.bfloat16), b=dense_tensor_format(torch.bfloat16)
    )


def test_fp8_weight_only_signature_has_a_dense_activation():
    """An un-quantized bf16 activation yields the W8A16 signature we register."""
    a = _meta((1, 4096), torch.bfloat16)
    b = _meta((6144, 4096), torch.float8_e4m3fn)
    scale = torch.empty((1,), dtype=torch.float32, device="meta")
    signature = gemm_format_signature(a, b, None, scale, "fp8")
    assert signature in W8A16_FP8_SIGNATURES
    assert signature.format_for("a").format == "dense"
    assert signature.format_for("b").format == "scaled-fp8"


def test_fp8_symmetric_signature_when_the_activation_is_already_quantized():
    """A pre-quantized activation yields the W8A8 signature, not the W8A16 one."""
    a = _meta((1, 4096), torch.float8_e4m3fn)
    b = _meta((6144, 4096), torch.float8_e4m3fn)
    a_scale = torch.empty((1, 1), dtype=torch.float32, device="meta")
    b_scale = torch.empty((1,), dtype=torch.float32, device="meta")
    signature = gemm_format_signature(a, b, a_scale, b_scale, "fp8")
    assert signature not in W8A16_FP8_SIGNATURES
    assert signature.format_for("a").format == "scaled-fp8"


def test_nvfp4_weight_only_signature():
    """NVFP4 weights + bf16 activation is the signature our GEMV registers."""
    a = _meta((1, 4096), torch.bfloat16)
    b = _meta((6144, 2048), torch.uint8)
    scale = torch.empty((6144, 256), dtype=torch.float8_e4m3fn, device="meta")
    signature = gemm_format_signature(a, b, None, scale, "nvfp4")
    assert signature in W4A16_NVFP4_SIGNATURES
    assert signature.format_for("b").scale.block_shape == (16,)


def test_mxfp4_and_mxfp8_branches_exist_and_carry_their_block_shapes():
    """The remaining ``gemm.mm`` scaled branches are reproduced verbatim."""
    a = _meta((1, 4096), torch.bfloat16)
    b_mxfp4 = _meta((6144, 2048), torch.uint8)
    mx4_scale = torch.empty((6144, 128), dtype=torch.uint8, device="meta")
    sig4 = gemm_format_signature(a, b_mxfp4, None, mx4_scale, "mxfp4")
    assert sig4.format_for("b").scale.block_shape == (32,)

    b_mxfp8 = _meta((6144, 4096), torch.float8_e4m3fn)
    mx8_scale = torch.empty((48, 32), dtype=torch.float32, device="meta")
    sig8 = gemm_format_signature(a, b_mxfp8, None, mx8_scale, "mxfp8", [128, 128])
    assert sig8.format_for("b").scale.block_shape == (128, 128)


def test_mxfp8_requires_block_size():
    """The mxfp8 branch cannot invent a block shape."""
    a = _meta((1, 4096), torch.bfloat16)
    b = _meta((6144, 4096), torch.float8_e4m3fn)
    scale = torch.empty((48, 32), dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="block_size"):
        gemm_format_signature(a, b, None, scale, "mxfp8")


def test_unknown_quant_is_rejected():
    """An unrecognised format name is a programming error, not a silent dense."""
    a = _meta((1, 4096), torch.bfloat16)
    b = _meta((6144, 4096), torch.bfloat16)
    with pytest.raises(ValueError, match="unknown quant"):
        gemm_format_signature(a, b, None, None, "int4")
    assert "fp8" in SCALED_KINDS and "nvfp4" in SCALED_KINDS


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_name", [FP8_KERNEL, NVFP4_KERNEL])
def test_kernels_are_registered(kernel_name):
    """Both quantized decode GEMVs self-register on import."""
    spec = KernelRegistry.get().get_by_name(kernel_name)
    assert spec is not None
    assert spec.origin == "first_party"
    # The gate must be the profitability cut, not the wider decode bucket.
    assert spec.traits["m_le_2"] == frozenset({True})


@pytest.mark.parametrize("label,N,K", QWEN3_8B_SHAPES)
@pytest.mark.parametrize("M", [1, 2])
def test_fp8_selection_inside_the_decode_slice(label, N, K, M):
    """Narrow decode projections select our fp8 GEMV; wide ones do not."""
    a = _meta((M, K), torch.bfloat16)
    b = _meta((N, K), torch.float8_e4m3fn)
    scale = torch.empty((1,), dtype=torch.float32, device="meta")
    signature = gemm_format_signature(a, b, None, scale, "fp8")
    traits = shape_traits(M, N, K)
    if N > 16384:  # gate_up_proj: `n_le_16384` excludes it
        with pytest.raises(NoKernelFoundError):
            select_kernel("gemm", "mm", signature, traits=traits)
    else:
        assert select_kernel("gemm", "mm", signature, traits=traits).name == FP8_KERNEL


@pytest.mark.parametrize("M", [3, 4, 8, 16, 128, 512])
def test_no_kernel_above_the_profitability_cut(M):
    """Above ``M = 2`` nothing matches, so dispatch hands the shape to the engine.

    This is the property that keeps CUDA-graph capture (which runs M = 4, 8,
    ... 512) on the engine's own kernel instead of our dequantize reference.
    """
    a = _meta((M, 4096), torch.bfloat16)
    b = _meta((4096, 2048), torch.uint8)
    scale = torch.empty((4096, 256), dtype=torch.float8_e4m3fn, device="meta")
    signature = gemm_format_signature(a, b, None, scale, "nvfp4")
    with pytest.raises(NoKernelFoundError):
        select_kernel("gemm", "mm", signature, traits=shape_traits(M, 4096, 4096))


def test_quantized_kernels_never_claim_a_dense_request():
    """The dense bf16 path is untouched by the new registrations."""
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16), b=dense_tensor_format(torch.bfloat16)
    )
    winner = select_kernel("gemm", "mm", signature, traits=shape_traits(1, 4096, 4096))
    assert winner.name not in (FP8_KERNEL, NVFP4_KERNEL)


def test_no_first_party_kernel_claims_the_symmetric_w8a8_request():
    """A pre-quantized activation is not something we serve yet.

    Recorded deliberately: the W8A8/W4A4 signatures are *built* by
    :func:`gemm_format_signature` so a future kernel can claim them, but no
    first-party kernel fulfils them today — prefill and every ``M > 2`` decode
    stay on the engine's own CUTLASS path.

    Asserted as "not one of ours" rather than "nothing at all", because on a box
    where ``tokenspeed_kernel``'s own op corpus is importable its
    ``triton_mm_fp8_scaled`` legitimately claims this signature.
    """
    a = _meta((1, 4096), torch.float8_e4m3fn)
    b = _meta((4096, 4096), torch.float8_e4m3fn)
    a_scale = torch.empty((1, 1), dtype=torch.float32, device="meta")
    b_scale = torch.empty((1,), dtype=torch.float32, device="meta")
    signature = gemm_format_signature(a, b, a_scale, b_scale, "fp8")
    try:
        winner = select_kernel(
            "gemm", "mm", signature, traits=shape_traits(1, 4096, 4096)
        ).name
    except NoKernelFoundError:
        return  # the standalone registry: nothing claims it, which is the point
    assert winner not in (FP8_KERNEL, NVFP4_KERNEL)


def test_shape_traits_adds_finer_decode_buckets_without_changing_the_old_ones():
    """``m_le_2`` / ``m_le_4`` are additive next to the pre-existing vocabulary."""
    traits = shape_traits(2, 4096, 4096)
    assert traits["m_le_2"] is True
    assert traits["m_le_4"] is True
    assert traits["m_le_8"] is True
    assert shape_traits(4, 4096, 4096)["m_le_2"] is False
    # Every name the tokenspeed façade publishes is still present.
    for name in ("n_le_16384", "n_align_16", "k_align_128", "n_min_128", "k_min_128"):
        assert name in traits


def test_scale_granularity_is_inferred_from_the_weight_scale():
    """A per-channel weight scale produces the channel-granularity signature."""
    a = _meta((1, 4096), torch.bfloat16)
    b = _meta((4096, 4096), torch.float8_e4m3fn)
    per_tensor = gemm_format_signature(
        a, b, None, torch.empty((1,), dtype=torch.float32, device="meta"), "fp8"
    )
    per_channel = gemm_format_signature(
        a, b, None, torch.empty((4096,), dtype=torch.float32, device="meta"), "fp8"
    )
    assert per_tensor.format_for("b").scale.granularity == "tensor"
    assert per_channel.format_for("b").scale.granularity == "channel"
    assert per_tensor != per_channel
    # Both are covered by the registered kernel.
    assert {per_tensor, per_channel} <= W8A16_FP8_SIGNATURES
    assert ScaleFormat is not None and tensor_format is not None
