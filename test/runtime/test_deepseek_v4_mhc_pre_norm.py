# Copyright (c) 2026 LightSeek Foundation
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
"""The fused mHC pre epilogue must equal the three-launch path plus a norm.

`mhc_pre` collapses `hc_mult` residual streams into one hidden row, then the
model normalises that row before feeding it to attention or the FFN. That is
three launches per seam in the unfused path (mix epilogue, collapse, RMSNorm),
and a seam runs twice per layer, so the fusion saves two launches per seam.

Fusing them forces the grid from one program per ``(token, hidden block)`` to
one program per token, because the norm needs a row-wide reduction. The gate is
therefore recomputed inside the collapse instead of being read back from the
``pre_mix`` buffer that the fused path no longer allocates at all.

What this test pins, and why each half matters:

* ``post_mix`` and ``comb_mix`` are handed to the next seam, and both paths use
  the same arithmetic for them, so they must stay *bitwise* equal. Drift there
  would silently change every following layer.
* The normalised row must match a torch RMSNorm of the unfused row. The norm's
  reduction order differs between Triton and the FlashInfer kernel the module
  uses, so an absolute tolerance is meaningless -- bf16 keeps 8 mantissa bits,
  making one ULP 0.008 near 1.0 but 0.03 near 4.0. The bound here is in ULPs,
  which tolerates the rounding of a slightly different ``rstd`` and nothing
  more.
"""

from __future__ import annotations

from test.runtime.conftest import requires_cuda

import pytest
import torch

from tokenspeed.runtime.layers import deepseek_v4_mhc as mhc

HC_MULT = 4
HC_COMB = HC_MULT * HC_MULT
HC_MULT3 = HC_MULT * 2 + HC_COMB
HIDDEN = 512
RMS_EPS = 1e-6
HC_EPS = 1e-6
SINKHORN_ITERS = 20

pytestmark = requires_cuda


def _require_deep_gemm():
    if mhc.deep_gemm is None:
        pytest.skip("deep_gemm.tf32_hc_prenorm_gemm is unavailable")


def _inputs(num_tokens: int, *, hidden: int = HIDDEN, seed: int = 0):
    """Build tensors shaped like the model's mHC parameters."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hc_dim = HC_MULT * hidden
    residual = torch.randn(
        num_tokens, HC_MULT, hidden, generator=gen, device="cuda"
    ).to(torch.bfloat16)
    fn = torch.randn(HC_MULT3, hc_dim, generator=gen, device="cuda") / hc_dim**0.5
    hc_scale = torch.tensor([1.0, 0.5, 0.25], device="cuda", dtype=torch.float32)
    hc_base = torch.randn(HC_MULT3, generator=gen, device="cuda") * 0.1
    weight = torch.randn(hidden, generator=gen, device="cuda").to(torch.bfloat16)
    return residual, fn, hc_scale, hc_base, weight


def _torch_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (
        xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    ).to(torch.bfloat16)


def _ulp(x: torch.Tensor) -> torch.Tensor:
    """One bf16 unit in the last place of ``x`` (8 exponent bits, 7 mantissa)."""
    return torch.pow(2.0, torch.floor(torch.log2(x.abs().clamp_min(1e-30))) - 8.0)


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, what: str, max_ulp: float = 2.0
) -> None:
    """Assert the fused row is the same row, rounded once fewer.

    The unfused path writes the collapsed row to bf16 and the model normalises
    that rounded copy; the fused path normalises the fp32 accumulator directly.
    The fused result is the *more* accurate one, and the two differ by exactly
    one rounding step -- about 2.5e-3 in relative L2, never more than two bf16
    ULPs, with roughly a quarter of the elements tipped in the last place. A
    tighter bound would be asserting the extra rounding, not the math.
    """
    diff = (actual.float() - expected.float()).abs()
    beyond = diff > _ulp(expected) * max_ulp
    assert not beyond.any(), (
        f"{what}: {beyond.sum().item()} of {diff.numel()} elements beyond "
        f"{max_ulp} ULP (max |diff| {diff.max().item():.6f})"
    )
    rel_l2 = (diff.pow(2).sum().sqrt() / expected.float().pow(2).sum().sqrt()).item()
    assert rel_l2 <= 5e-3, f"{what}: relative L2 error {rel_l2:.3e}"


def test_fused_pre_epilogue_matches_unfused_path_on_one_token():
    _require_deep_gemm()
    _check_against_unfused_path(1)


@pytest.mark.parametrize("num_tokens", [8, 128])
def test_fused_pre_epilogue_matches_unfused_path(num_tokens: int):
    _require_deep_gemm()
    _check_against_unfused_path(num_tokens)


def _check_against_unfused_path(num_tokens: int) -> None:
    residual, fn, hc_scale, hc_base, weight = _inputs(num_tokens)

    ref_row, ref_post, ref_comb = mhc.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        SINKHORN_ITERS,
    )
    fused_row, post, comb = mhc.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        SINKHORN_ITERS,
        norm_weight=weight,
        norm_eps=RMS_EPS,
    )

    # The next seam consumes these; both paths compute them identically.
    assert torch.equal(post, ref_post), "post mix drifted"
    assert torch.equal(comb, ref_comb), "comb mix drifted"

    _assert_close(fused_row, _torch_norm(ref_row, weight, RMS_EPS), "layer input")

    # A normalised row is sane only if it is actually populated: a fusion that
    # dropped the collapse or the norm cannot pass on the tolerance alone.
    rms = fused_row.float().pow(2).mean().sqrt().item()
    assert 0.1 < rms < 100.0


def test_fused_pre_epilogue_normalises_only_when_asked():
    """Without a weight the call must still return the un-normalised row."""
    _require_deep_gemm()
    residual, fn, hc_scale, hc_base, weight = _inputs(8)

    with_weight, _, _ = mhc.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        SINKHORN_ITERS,
        norm_weight=weight,
        norm_eps=RMS_EPS,
    )
    without_weight, _, _ = mhc.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        SINKHORN_ITERS,
    )
    _assert_close(
        with_weight, _torch_norm(without_weight, weight, RMS_EPS), "layer input"
    )


def test_mhc_fused_hc_forwards_norm_request():
    """`mhc_fused_hc` must pass the norm through, or the caller double-normalises."""
    _require_deep_gemm()
    residual, fn, hc_scale, hc_base, weight = _inputs(8)
    gen = torch.Generator(device="cuda").manual_seed(1)
    x_prev = torch.randn(8, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
    post_prev = torch.rand(8, HC_MULT, 1, generator=gen, device="cuda")
    comb_prev = torch.rand(8, HC_MULT, HC_MULT, generator=gen, device="cuda")

    ref_residual = mhc.mhc_post(x_prev, residual, post_prev, comb_prev)
    ref_row, ref_post, ref_comb = mhc.mhc_pre(
        ref_residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, SINKHORN_ITERS
    )

    residual_cur, row, post, comb = mhc.mhc_fused_hc(
        x_prev,
        residual,
        post_prev,
        comb_prev,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        SINKHORN_ITERS,
        norm_weight=weight,
        norm_eps=RMS_EPS,
    )

    assert torch.equal(residual_cur, ref_residual), "post mapping drifted"
    assert torch.equal(post, ref_post), "post mix drifted"
    assert torch.equal(comb, ref_comb), "comb mix drifted"
    _assert_close(row, _torch_norm(ref_row, weight, RMS_EPS), "layer input")
