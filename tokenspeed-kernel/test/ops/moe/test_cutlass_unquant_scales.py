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

"""``token_final_scales`` handed to cutlass_fused_moe must stay float32.

The engine's precomputed-topk path already passes float32, so only the branch
where the kernel does its own routing can get this wrong -- and there it used
to follow ``x.dtype``, producing a half/bfloat16 scale tensor the cutlass
fused-MoE kernel rejects. The kernel call itself is stubbed: what is under test
is the dtype of the argument, not the MoE result.
"""

import pytest
import torch
from tokenspeed_kernel.platform import Platform

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not Platform.get().is_nvidia:
    pytest.skip("NVIDIA-only registration under test", allow_module_level=True)

cutlass_unquant = pytest.importorskip(
    "tokenspeed_kernel.ops.moe.flashinfer.cutlass_unquant"
)


class _Weights(torch.nn.Module):
    """Minimal stand-in for the MoE weight module the op reads attributes off."""

    def __init__(self, num_experts: int, top_k: int, hidden: int, inter: int):
        super().__init__()
        self.top_k = top_k
        self.w13_weight = torch.zeros(num_experts, 2 * inter, hidden, device="cuda")
        self.w2_weight = torch.zeros(num_experts, hidden, inter, device="cuda")


def _run_capturing_scales(monkeypatch, x_dtype, topk_weights):
    """Call the op with the cutlass kernel stubbed; return the captured kwargs."""
    captured = {}

    def _stub(**kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["input"])

    monkeypatch.setattr(cutlass_unquant, "cutlass_fused_moe", _stub)
    # Freeze autotuning so the op takes the nullcontext branch: the real
    # autotuner would try to profile the stub.
    monkeypatch.setattr(cutlass_unquant, "autotune_frozen", lambda: True)

    num_experts, top_k, hidden, inter, tokens = 8, 2, 16, 32, 4
    w = _Weights(num_experts, top_k, hidden, inter)
    x = torch.randn(tokens, hidden, dtype=x_dtype, device="cuda")
    router_logits = torch.randn(tokens, num_experts, dtype=torch.float32, device="cuda")

    ids = None
    if topk_weights is not None:
        ids = torch.zeros(tokens, top_k, dtype=torch.int32, device="cuda")

    cutlass_unquant.flashinfer_cutlass_unquant_moe_apply(
        {},
        x,
        w,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=ids,
    )
    return captured


@pytest.mark.parametrize("x_dtype", [torch.float16, torch.bfloat16])
def test_internal_routing_scales_are_float32(monkeypatch, x_dtype):
    """The kernel-routes-itself branch must not follow ``x.dtype``."""
    captured = _run_capturing_scales(monkeypatch, x_dtype, topk_weights=None)

    assert captured["token_final_scales"].dtype == torch.float32
    # The activation itself is unaffected -- only the scales are promoted.
    assert captured["input"].dtype == x_dtype


@pytest.mark.parametrize("x_dtype", [torch.float16, torch.bfloat16])
def test_precomputed_scales_are_passed_through(monkeypatch, x_dtype):
    """A caller-supplied float32 scale tensor reaches the kernel unchanged."""
    tokens, top_k = 4, 2
    supplied = torch.rand(tokens, top_k, dtype=torch.float32, device="cuda")
    captured = _run_capturing_scales(monkeypatch, x_dtype, topk_weights=supplied)

    assert captured["token_final_scales"].dtype == torch.float32
    assert torch.equal(captured["token_final_scales"], supplied)
