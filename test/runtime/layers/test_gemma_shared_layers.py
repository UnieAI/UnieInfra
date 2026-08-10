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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Numerical parity for the two shared layers the Gemma 3/4 decoders need.

Both replace inline arithmetic that the Gemma port previously carried in the
model file: ``chunk(2) + F.gelu(approximate="tanh") * up`` and a hand-written
value RMSNorm. The references here are written against the HF formulation
rather than against the implementation, and run in float64 where the
implementation runs in float32, so a shared algebraic slip does not cancel out.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.activation import GeluTanhAndMul
from tokenspeed.runtime.layers.layernorm import RMSNormWithoutScale


def _ref_geglu(x: torch.Tensor) -> torch.Tensor:
    """HF ``gelu_pytorch_tanh`` GeGLU, expanded in float64.

    The tanh approximation is spelled out rather than delegated to
    ``F.gelu(approximate="tanh")`` so the two sides are not the same code.
    """
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float64)
    up = x[..., d:].to(torch.float64)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate.pow(3))
    return 0.5 * gate * (1.0 + torch.tanh(inner)) * up


def _ref_rms_no_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.to(torch.float64)
    return xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


# --------------------------------------------------------------------------
# GeluTanhAndMul
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 8), (7, 128), (4, 3, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_geglu_matches_the_hf_reference_on_cpu(shape, dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype)

    got = GeluTanhAndMul()(x)
    want = _ref_geglu(x)

    assert got.shape == x.shape[:-1] + (x.shape[-1] // 2,)
    assert got.dtype == dtype
    tol = 2e-6 if dtype is torch.float32 else 3e-2
    torch.testing.assert_close(got.to(torch.float64), want, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_geglu_cuda_kernel_matches_the_native_path(dtype) -> None:
    """The fused kernel and the fallback must agree; only the launch differs."""
    torch.manual_seed(0)
    x = torch.randn(37, 256, dtype=dtype, device="cuda")

    fused = GeluTanhAndMul()(x)
    native = GeluTanhAndMul()(x.cpu()).to(x.device)

    torch.testing.assert_close(fused, native, rtol=3e-2, atol=3e-2)


def test_geglu_rejects_an_odd_width() -> None:
    with pytest.raises(ValueError, match="even"):
        GeluTanhAndMul()(torch.randn(2, 7))


def test_geglu_splits_gate_before_up() -> None:
    """Pin the halving order: a swapped split is silently plausible.

    With gate=0 the GeLU branch is 0, so the product is 0 regardless of ``up``;
    the reverse split would instead return ``gelu(up) * 0``, also 0. Use a
    nonzero gate and a zero up to break the symmetry in one direction, then the
    mirrored case for the other.
    """
    x = torch.cat([torch.full((1, 4), 2.0), torch.zeros(1, 4)], dim=-1)
    torch.testing.assert_close(GeluTanhAndMul()(x), torch.zeros(1, 4))

    y = torch.cat([torch.zeros(1, 4), torch.full((1, 4), 2.0)], dim=-1)
    torch.testing.assert_close(GeluTanhAndMul()(y), torch.zeros(1, 4))

    z = torch.cat([torch.full((1, 4), 2.0), torch.full((1, 4), 3.0)], dim=-1)
    # gelu_tanh(2.0) ~= 1.9545977, times up=3
    torch.testing.assert_close(
        GeluTanhAndMul()(z), torch.full((1, 4), 5.8637933), rtol=1e-5, atol=1e-5
    )


# --------------------------------------------------------------------------
# RMSNormWithoutScale
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 16), (5, 128), (2, 3, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_without_scale_matches_the_reference(shape, dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype)
    eps = 1e-6

    got = RMSNormWithoutScale(eps)(x)
    want = _ref_rms_no_scale(x, eps)

    assert got.shape == x.shape
    assert got.dtype == dtype
    tol = 2e-6 if dtype is torch.float32 else 3e-2
    torch.testing.assert_close(got.to(torch.float64), want, rtol=tol, atol=tol)


def test_rms_without_scale_holds_no_parameters() -> None:
    """``with_scale=False`` means there is no weight for the loader to fill.

    A stray parameter here would make the checkpoint loader look for a tensor
    the Gemma 4 weights do not contain.
    """
    norm = RMSNormWithoutScale(1e-6)

    assert list(norm.parameters()) == []
    assert list(norm.buffers()) == []
    assert norm.state_dict() == {}


def test_rms_without_scale_reduces_in_float32() -> None:
    """A bf16 reduction on a large row loses enough precision to matter."""
    x = torch.full((1, 4096), 0.01, dtype=torch.bfloat16)

    got = RMSNormWithoutScale(1e-6)(x).to(torch.float64)
    naive_bf16 = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(
        torch.float64
    )
    want = _ref_rms_no_scale(x, 1e-6)

    assert (got - want).abs().max() <= (naive_bf16 - want).abs().max()
    torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)


def test_rms_without_scale_normalises_the_last_dim_only() -> None:
    """Rows must be independent: scaling one row must not move another."""
    x = torch.randn(3, 32)
    norm = RMSNormWithoutScale(1e-6)

    base = norm(x)
    scaled = x.clone()
    scaled[0] *= 10.0

    torch.testing.assert_close(norm(scaled)[1:], base[1:])
    # RMSNorm is scale-invariant up to eps, so the scaled row lands back on itself.
    torch.testing.assert_close(norm(scaled)[0], base[0], rtol=1e-4, atol=1e-4)
