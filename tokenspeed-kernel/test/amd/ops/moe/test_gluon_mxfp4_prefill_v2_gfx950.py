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

"""Opt-in package-prefill MXFP4 paths must match the default path bit for bit.

``TOKENSPEED_MOE_PREFILL_S1_V2`` (pipelined stage-1 K loop with dword scale
cells, BLOCK_M 64) and ``TOKENSPEED_MOE_PREFILL_S2_V2`` (dword scale loads in
stage 2) are read at import; the tests flip the module flags directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from utils import is_cdna4, make_mxfp4_moe_weights

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Gluon MXFP4 package prefill tests",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import (  # noqa: E402
    prefill_stage1,
    prefill_stage2,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    gluon_mxfp_precomputed_mxfp4_fused_moe,
    tuning,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)


def _make_module(num_experts, hidden_size, intermediate_size, generator):
    raw = make_mxfp4_moe_weights(num_experts, hidden_size, intermediate_size, generator)
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


def _run(module, hidden_states, topk_weights, topk_ids, num_experts):
    return gluon_mxfp_precomputed_mxfp4_fused_moe(
        hidden_states,
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
        swiglu_alpha=1.0,
        swiglu_limit=10.0,
        swiglu_beta=0.0,
        out=None,
    )


# Hidden 1280 gives an odd number of BK256 stage-1 tiles (pipeline tail) and
# intermediate 576 an odd number of BK128 stage-2 tiles (K padded to 640).
# 2304 tokens with top-2 over 32 experts is 144 routes per expert: BLOCK_M 64
# with and without the opt-in selection, and above the 2048-token atomic
# combine bound, so both runs use the deterministic FP32 reduction (covering
# the M-major tile order and ``.cg`` partial reads of the reduce path).
@pytest.mark.parametrize("hidden_size", [1024, 1280])
def test_package_prefill_v2_paths_match_default_gfx950(
    monkeypatch: pytest.MonkeyPatch, hidden_size: int
) -> None:
    num_experts, intermediate_size, top_k, num_tokens = 32, 576, 2, 2304
    generator = torch.Generator(device="cuda").manual_seed(20261115)
    module = _make_module(num_experts, hidden_size, intermediate_size, generator)
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_ids = (
        torch.rand(num_tokens, num_experts, device="cuda", generator=generator)
        .topk(top_k, dim=1)
        .indices.to(torch.int32)
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device="cuda", generator=generator), dim=1
    )
    assert tuning._select_grouped_moe_block_m(num_tokens, top_k, num_experts) == 64

    monkeypatch.setattr(prefill_stage1, "_S1_PIPELINE", False)
    monkeypatch.setattr(prefill_stage2, "_S2_SCALE_DWORD", False)
    monkeypatch.setattr(prefill_stage2, "_S2_REDUCE_V2", False)
    monkeypatch.setattr(tuning, "_PREFILL_S1_V2", False)
    expected = _run(module, hidden_states, topk_weights, topk_ids, num_experts)

    monkeypatch.setattr(prefill_stage1, "_S1_PIPELINE", True)
    monkeypatch.setattr(prefill_stage2, "_S2_SCALE_DWORD", True)
    monkeypatch.setattr(prefill_stage2, "_S2_REDUCE_V2", True)
    monkeypatch.setattr(tuning, "_PREFILL_S1_V2", True)
    actual = _run(module, hidden_states, topk_weights, topk_ids, num_experts)

    assert expected.abs().max() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("routes_per_expert", "expected"),
    [(16, 32), (24, 32), (25, 64), (64, 64), (128, 64), (320, 64), (321, 128)],
)
def test_package_prefill_v2_block_m_selection(
    monkeypatch: pytest.MonkeyPatch, routes_per_expert: int, expected: int
) -> None:
    num_experts, top_k = 384, 6
    monkeypatch.setattr(tuning, "_PREFILL_S1_V2", True)
    num_tokens = routes_per_expert * num_experts // top_k
    assert tuning._select_grouped_moe_block_m(num_tokens, top_k, num_experts) == (
        expected
    )
