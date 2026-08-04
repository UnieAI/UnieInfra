# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""GPU numerics for the quantized decode GEMVs (fp8 W8A16, NVFP4 W4A16).

Each kernel is checked against its own dequantize-then-``F.linear`` reference,
which is the exact arithmetic the engine's scaled GEMM approximates. That is the
right oracle: comparing a 4-bit GEMM against a bf16 one only measures the
quantization error of the checkpoint, not whether the kernel is correct.

Also asserted here: the wrapper's *delegate* path (a layout the fast path does
not accept) still returns the reference answer, so a declined request is never
wrong — only no faster.
"""

from __future__ import annotations

import pytest
import torch
import unieinfra_kernel  # noqa: F401  (import triggers kernel discovery)
from unieinfra_kernel.ops.gemm.decode_gemv_fp8_triton import (
    decode_gemv_fp8,
    decode_gemv_fp8_reference,
    supports_decode_gemv_fp8,
    unieinfra_decode_gemv_fp8_triton,
)
from unieinfra_kernel.ops.gemm.decode_gemv_nvfp4_triton import (
    GROUP_SIZE,
    decode_gemv_nvfp4,
    decode_gemv_nvfp4_reference,
    supports_decode_gemv_nvfp4,
    unieinfra_decode_gemv_nvfp4_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="quantized decode GEMVs need a GPU"
)

# Qwen3-8B decode projections (hidden 4096); gate_up is excluded by the trait
# gate but the kernel must still be *correct* on it if handed one.
SHAPES = [(6144, 4096), (4096, 4096), (4096, 12288)]
MS = [1, 2]


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref.float()).norm() / ref.float().norm()).item()


def _quantize_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor fp8-e4m3 quantization of ``[N, K]`` weights."""
    amax = weight.abs().max().float().clamp(min=1e-12)
    scale = (amax / 448.0).reshape(1)
    q = (weight.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


def _quantize_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference NVFP4 quantization of ``[N, K]`` weights.

    Returns:
        ``(packed_uint8 [N, K//2], block_scales_fp8 [N, K//16], global_scale)``
        in the ModelOpt convention, i.e.
        ``W ~= e2m1(nibble) * block_scale * global_scale``.
    """
    n, k = weight.shape
    w = weight.float().reshape(n, k // GROUP_SIZE, GROUP_SIZE)
    global_scale = (weight.abs().max().float() / (6.0 * 448.0)).clamp(min=1e-12)
    block_scale = (w.abs().amax(dim=-1, keepdim=True) / 6.0) / global_scale
    block_scale = block_scale.clamp(max=448.0).to(torch.float8_e4m3fn)
    denom = (block_scale.float() * global_scale).clamp(min=1e-12)
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device
    )
    scaled = (w / denom).clamp(-6.0, 6.0)
    idx = (scaled.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    nibble = idx.to(torch.uint8) | torch.where(
        scaled < 0,
        torch.tensor(8, device=weight.device, dtype=torch.uint8),
        torch.tensor(0, device=weight.device, dtype=torch.uint8),
    )
    nibble = nibble.reshape(n, k)
    packed = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).contiguous()
    return (
        packed,
        block_scale.reshape(n, k // GROUP_SIZE).contiguous(),
        global_scale.reshape(1),
    )


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("per_channel", [False, True])
def test_fp8_gemv_matches_its_dequant_reference(N, K, M, per_channel):
    """W8A16 fp8 GEMV reproduces dequantize-then-``F.linear``."""
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 30
    qw, scale = _quantize_fp8(w)
    if per_channel:
        scale = torch.rand(N, device="cuda", dtype=torch.float32) * 1e-3 + 5e-4
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)

    got = decode_gemv_fp8(x, qw, scale, bias=bias)
    ref = decode_gemv_fp8_reference(x, qw, scale, bias)
    assert _rel_err(got, ref) < 1e-3


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("M", MS)
def test_nvfp4_gemv_matches_its_dequant_reference(N, K, M):
    """W4A16 NVFP4 GEMV reproduces dequantize-then-``F.linear``."""
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 30
    packed, block_scale, global_scale = _quantize_nvfp4(w)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)

    got = decode_gemv_nvfp4(x, packed, block_scale, global_scale, bias=bias)
    ref = decode_gemv_nvfp4_reference(x, packed, block_scale, global_scale, bias)
    assert _rel_err(got, ref) < 1e-3


def test_nvfp4_quantization_error_is_the_expected_order():
    """The 4-bit round trip costs ~10% relative error, not more.

    A guard against a silently wrong scale convention: a mis-applied global
    scale shows up as a huge relative error against the bf16 answer while the
    dequant reference test above would still pass.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) / 30
    packed, block_scale, global_scale = _quantize_nvfp4(w)
    got = decode_gemv_nvfp4(x, packed, block_scale, global_scale)
    dense = torch.nn.functional.linear(x.float(), w.float())
    assert _rel_err(got, dense) < 0.15


def test_fp8_rejects_the_transposed_per_channel_scale_layout():
    """``[1, N]`` is refused rather than silently transposed (E23)."""
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) / 30
    qw, _ = _quantize_fp8(w)
    bad = torch.rand(1, 4096, device="cuda", dtype=torch.float32)
    assert not supports_decode_gemv_fp8(x, qw, bad)
    good = torch.rand(4096, 1, device="cuda", dtype=torch.float32)
    assert supports_decode_gemv_fp8(x, qw, good)


def test_wrappers_delegate_correctly_for_declined_requests():
    """A shape the fast path declines still returns the reference answer."""
    torch.manual_seed(0)
    M, N, K = 5, 4096, 4096  # M above the profitability cut
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 30

    qw, scale = _quantize_fp8(w)
    assert not supports_decode_gemv_fp8(x, qw, scale)
    got = unieinfra_decode_gemv_fp8_triton(x, qw, None, scale, torch.bfloat16)
    assert _rel_err(got, decode_gemv_fp8_reference(x, qw, scale)) < 1e-6

    packed, block_scale, global_scale = _quantize_nvfp4(w)
    assert not supports_decode_gemv_nvfp4(x, packed, block_scale, global_scale)
    got4 = unieinfra_decode_gemv_nvfp4_triton(
        x, packed, None, block_scale, torch.bfloat16, alpha=global_scale
    )
    ref4 = decode_gemv_nvfp4_reference(x, packed, block_scale, global_scale)
    assert _rel_err(got4, ref4) < 1e-6
