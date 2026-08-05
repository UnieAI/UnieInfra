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

"""Dense Qwen3 ``load_weights`` skips KV-cache scales it has no parameter for.

ModelOpt checkpoints that declare ``kv_cache_quant_algo`` carry per-layer
``.k_scale``/``.v_scale`` tensors. They only become parameters when the KV cache
is actually quantised, so on an unquantised-KV load they have no home and the
loader used to die with a ``KeyError`` before the first forward.
``qwen3_moe.py`` already skips them; this covers the dense path.

``load_weights`` reads only ``self.config``, ``self.model`` and
``self.named_parameters()``, so it is exercised here against a stub rather than
a constructed model -- no GPU, no checkpoint.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM  # noqa: E402


class _Param(torch.nn.Parameter):
    """A parameter that records what the loader wrote into it."""

    def __new__(cls, name: str):
        obj = super().__new__(cls, torch.zeros(1), requires_grad=False)
        obj.loaded = []
        obj.weight_loader = lambda param, weight, *a: param.loaded.append((weight, a))
        return obj


def _stub_model(param_names: list[str]):
    params = {name: _Param(name) for name in param_names}
    return (
        SimpleNamespace(
            config=SimpleNamespace(
                name_or_path="Qwen/Qwen3-8B", tie_word_embeddings=False
            ),
            model=SimpleNamespace(),
            named_parameters=lambda: list(params.items()),
        ),
        params,
    )


_ATTN = "model.layers.0.self_attn"
# Both spellings ModelOpt emits, on the fused qkv name and on the raw
# projections that the stacked-params mapping rewrites to it.
KV_SCALE_NAMES = [
    f"{_ATTN}.k_proj.k_scale",
    f"{_ATTN}.v_proj.v_scale",
    f"{_ATTN}.qkv_proj.k_scale",
    f"{_ATTN}.qkv_proj.v_scale",
    f"{_ATTN}.qkv_proj_k_scale",
    f"{_ATTN}.qkv_proj_v_scale",
]


@pytest.mark.parametrize("scale_name", KV_SCALE_NAMES)
def test_absent_kv_scale_is_skipped(scale_name):
    """A KV scale with no matching parameter is dropped, not raised on."""
    stub, params = _stub_model([f"{_ATTN}.o_proj.weight"])

    Qwen3ForCausalLM.load_weights(stub, [(scale_name, torch.tensor([0.5]))])

    assert params[f"{_ATTN}.o_proj.weight"].loaded == []


@pytest.mark.parametrize(
    "ckpt_name,param_name,shard",
    [
        (f"{_ATTN}.k_proj.k_scale", f"{_ATTN}.qkv_proj.k_scale", "k"),
        (f"{_ATTN}.v_proj.v_scale", f"{_ATTN}.qkv_proj.v_scale", "v"),
    ],
)
def test_present_kv_scale_is_still_loaded(ckpt_name, param_name, shard):
    """When the KV cache *is* quantised the scale has a parameter and loads.

    Uses the raw-projection spelling ModelOpt actually emits, which the
    stacked-params mapping rewrites onto the fused ``qkv_proj`` parameter. A
    checkpoint key already spelled ``qkv_proj.k_scale`` cannot reach its
    parameter at all -- ``"v_proj" in "qkv_proj"`` is true, so the rewrite
    mangles it to ``qkqkv_proj``. That is a pre-existing quirk of the mapping
    loop that also affects ``qkv_proj.bias``, and is out of scope here.
    """
    stub, params = _stub_model([param_name])
    weight = torch.tensor([0.25])

    Qwen3ForCausalLM.load_weights(stub, [(ckpt_name, weight)])

    assert len(params[param_name].loaded) == 1
    assert torch.equal(params[param_name].loaded[0][0], weight)
    assert params[param_name].loaded[0][1] == (shard,)


def test_ordinary_weights_still_load():
    """The skip must not swallow anything else -- the control case."""
    fused = f"{_ATTN}.qkv_proj.weight"
    stub, params = _stub_model([fused, f"{_ATTN}.o_proj.weight"])

    Qwen3ForCausalLM.load_weights(
        stub,
        [
            (f"{_ATTN}.k_proj.weight", torch.tensor([1.0])),
            (f"{_ATTN}.o_proj.weight", torch.tensor([2.0])),
        ],
    )

    # k_proj is rewritten onto the fused qkv parameter with shard id "k".
    assert len(params[fused].loaded) == 1
    assert params[fused].loaded[0][1] == ("k",)
    assert len(params[f"{_ATTN}.o_proj.weight"].loaded) == 1


def test_missing_kv_scale_without_the_skip_would_raise():
    """Pin the failure mode: a *non*-scale name absent from params_dict still
    raises, so the new branch is doing the skipping and not some pre-existing
    catch-all."""
    stub, _ = _stub_model([f"{_ATTN}.o_proj.weight"])

    with pytest.raises(KeyError):
        Qwen3ForCausalLM.load_weights(
            stub, [(f"{_ATTN}.q_norm.weight", torch.tensor([1.0]))]
        )
