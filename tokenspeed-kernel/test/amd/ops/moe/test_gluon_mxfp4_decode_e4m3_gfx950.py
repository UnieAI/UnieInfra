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

"""Decode-sized gfx950 dynamic-A4W4 MoE with E4M3 stage-1 activations.

The E4M3 package (sorted-route MXFP8 quantization, async-LDS stage 1 with the
SwiGLU epilogue, MXFP8 intermediate) must track a BF16-activation reference at
least as closely as the E2M1 package it replaces at decode batch sizes.
"""

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.numerics.torch.quantization import dequantize_mxfp4
from utils import is_cdna4, make_mxfp4_moe_weights

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    gluon_mxfp_precomputed_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)

# DeepSeek V4.1 routed experts: SwiGLU with a 10.0 clamp, standard (beta 0).
_ALPHA, _LIMIT, _BETA = 1.0, 10.0, 0.0


def _module(raw):
    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)
    for name, key in (
        ("w13_weight", "w13_weight"),
        ("w13_weight_scale", "w13_scale"),
        ("w2_weight", "w2_weight"),
        ("w2_weight_scale", "w2_scale"),
    ):
        setattr(module, name, torch.nn.Parameter(raw[key].clone(), requires_grad=False))
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)
    return module


def _run(module, x, topk_weights, topk_ids, num_experts, fmt):
    return gluon_mxfp_precomputed_mxfp4_fused_moe(
        x,
        topk_weights,
        topk_ids,
        module.w13_weight_triton_tensor,
        module.w2_weight_triton_tensor,
        w13_mx_scale=module.w13_precision_config.b_mx_scale,
        w2_mx_scale=module.w2_precision_config.b_mx_scale,
        expert_start=0,
        global_num_experts=num_experts,
        do_finalize=True,
        x_quantized=None,
        out_dtype=torch.bfloat16,
        swiglu_alpha=_ALPHA,
        swiglu_limit=_LIMIT,
        swiglu_beta=_BETA,
        out=None,
        decode_activation_format=fmt,
    )


def _reference(raw, x, topk_weights, topk_ids):
    """FP32 experts on the BF16 input and unquantized intermediate."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in torch.unique(topk_ids).tolist():
        token, slot = (topk_ids == expert).nonzero(as_tuple=True)
        w13 = dequantize_mxfp4(raw["w13_weight"][expert], raw["w13_scale"][expert])
        w2 = dequantize_mxfp4(raw["w2_weight"][expert], raw["w2_scale"][expert])
        gate_up = x[token].float() @ w13.float().T
        gate = gate_up[:, 0::2].clamp(max=_LIMIT)
        up = gate_up[:, 1::2].clamp(-_LIMIT, _LIMIT)
        inter = gate * torch.sigmoid(_ALPHA * gate) * (up + _BETA)
        out.index_add_(
            0, token, (inter @ w2.float().T) * topk_weights[token, slot, None]
        )
    return out


@pytest.mark.parametrize("num_tokens", [9, 64, 192, 768])
def test_e4m3_decode_tracks_bf16_reference(num_tokens):
    generator = torch.Generator(device="cuda").manual_seed(num_tokens)
    num_experts, hidden, inter, top_k = 64, 1024, 512, 6
    raw = make_mxfp4_moe_weights(num_experts, hidden, inter, generator)
    module = _module(raw)
    x = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    scores = torch.rand(num_tokens, num_experts, device="cuda", generator=generator)
    topk_ids = scores.topk(top_k, dim=1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device="cuda", generator=generator), dim=1
    )
    expected = _reference(raw, x, topk_weights, topk_ids)

    def rel_error(fmt):
        actual = _run(module, x, topk_weights, topk_ids, num_experts, fmt).float()
        return ((actual - expected).norm() / expected.norm()).item()

    e4m3 = rel_error("e4m3")
    e2m1 = rel_error("e2m1")
    # MXFP8 activations and intermediates lose far less than MXFP4 ones.
    assert e4m3 < 0.04
    assert e4m3 < 0.5 * e2m1
