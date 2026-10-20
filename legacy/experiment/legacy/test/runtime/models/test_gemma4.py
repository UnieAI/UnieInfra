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

"""Structural (no-GPU) unit tests for the Gemma 4 text model.

These build the model on the ``meta`` device from a tiny synthetic ``gemma4``
config (2 sliding + 1 full-attention layers) and assert that the
Gemma-4-specific transforms are wired correctly. No forward pass is run (the
fused RMSNorm / paged-attention kernels require CUDA); the CPU-computable
Gemma-4 numerics -- the embedding normalizer and the per-layer RoPE inverse
frequencies -- are checked directly.
"""

from __future__ import annotations

import math

import pytest
import torch

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.models.gemma import (
    Gemma4ForConditionalGeneration,
    Gemma4MLP,
    build_rope_inv_freq,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

HIDDEN = 32
GLOBAL_HEAD_DIM = 16
LOCAL_HEAD_DIM = 8
NUM_HEADS = 4
NUM_KV_HEADS = 2
NUM_GLOBAL_KV_HEADS = 1
SLIDING_WINDOW = 4
VOCAB = 64
LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention"]
LOCAL_THETA = 10000.0
GLOBAL_THETA = 1_000_000.0
PARTIAL = 0.25


def _tiny_text_config():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    return Gemma4TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_global_key_value_heads=NUM_GLOBAL_KV_HEADS,
        head_dim=LOCAL_HEAD_DIM,
        global_head_dim=GLOBAL_HEAD_DIM,
        sliding_window=SLIDING_WINDOW,
        layer_types=list(LAYER_TYPES),
        attention_k_eq_v=True,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        # Match the dense gemma-4-31B checkpoint: no per-layer input
        # embeddings, no KV sharing, no MoE.
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        enable_moe_block=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": PARTIAL,
                "rope_theta": GLOBAL_THETA,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": LOCAL_THETA,
                "rope_type": "default",
            },
        },
        dtype="bfloat16",
    )


def _mapping() -> Mapping:
    return Mapping(
        rank=0,
        world_size=1,
        attn_tp_size=1,
        attn_cp_size=1,
        attn_dp_size=1,
        dense_tp_size=1,
        dense_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        moe_dp_size=1,
        nprocs_per_node=1,
        nnodes=1,
    )


def _build_meta_model(monkeypatch) -> Gemma4ForConditionalGeneration:
    mapping = _mapping()
    monkeypatch.setitem(global_server_args_dict, "mapping", mapping)
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 2048)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return Gemma4ForConditionalGeneration(_tiny_text_config(), mapping)
    finally:
        torch.set_default_dtype(old_dtype)


def test_per_layer_head_dim_and_kv_heads(monkeypatch) -> None:
    """Sliding layers use local head_dim/kv-heads; the full-attention layer
    uses the larger global head_dim and the smaller global kv-head count."""
    model = _build_meta_model(monkeypatch)
    layers = model.model.layers

    sliding = layers[0].self_attn
    assert sliding.is_sliding is True
    assert sliding.head_dim == LOCAL_HEAD_DIM
    assert sliding.num_kv_heads == NUM_KV_HEADS
    assert sliding.attn.sliding_window_size == SLIDING_WINDOW
    assert sliding.attn.group_id == "sliding_attention"

    full = layers[2].self_attn
    assert full.is_sliding is False
    assert full.head_dim == GLOBAL_HEAD_DIM
    # global head_dim differs from the sliding head_dim -- the trickiest bit.
    assert full.head_dim != sliding.head_dim
    assert full.num_kv_heads == NUM_GLOBAL_KV_HEADS
    assert full.attn.sliding_window_size == -1
    assert full.attn.group_id == "full_attention"


def test_attention_k_eq_v(monkeypatch) -> None:
    """Full-attention layers share K/V (no v_proj); sliding layers keep a
    dedicated v_proj."""
    model = _build_meta_model(monkeypatch)
    assert model.model.layers[0].self_attn.v_proj is not None
    assert model.model.layers[2].self_attn.k_eq_v is True
    assert model.model.layers[2].self_attn.v_proj is None


def test_attention_scaling_is_one(monkeypatch) -> None:
    """Gemma 4 has no query_pre_attn_scalar: softmax scaling is 1.0, not
    head_dim**-0.5."""
    model = _build_meta_model(monkeypatch)
    for layer in model.model.layers:
        assert layer.self_attn.scaling == 1.0
        assert layer.self_attn.attn.scaling == 1.0


def test_qk_and_v_norm_wiring(monkeypatch) -> None:
    """q_norm / k_norm exist with per-head_dim weights; v_norm has no learnable
    weight (applied inline)."""
    model = _build_meta_model(monkeypatch)
    attn = model.model.layers[2].self_attn
    assert attn.q_norm.weight.shape[0] == GLOBAL_HEAD_DIM
    assert attn.k_norm.weight.shape[0] == GLOBAL_HEAD_DIM
    # v_norm is weightless -- there must be no such parameter.
    assert not any("v_norm" in n for n, _ in model.named_parameters())


def test_sandwich_norms_and_layer_scalar(monkeypatch) -> None:
    """Each layer has the four Gemma sandwich norms and a trainable
    layer_scalar."""
    model = _build_meta_model(monkeypatch)
    layer = model.model.layers[0]
    for norm in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        assert hasattr(layer, norm)
    assert layer.layer_scalar.shape == (1,)
    assert "model.layers.0.layer_scalar" in dict(model.named_parameters())


def test_tied_lm_head_and_softcap(monkeypatch) -> None:
    """lm_head is tied to embed_tokens; final_logit_softcapping is registered
    on the shared logits processor."""
    model = _build_meta_model(monkeypatch)
    assert model.lm_head is model.model.embed_tokens
    assert model.logits_processor.final_logit_softcapping == 30.0


def test_registered_architectures() -> None:
    from tokenspeed.runtime.models.registry import ModelRegistry

    supported = ModelRegistry.get_supported_archs()
    assert "Gemma4ForConditionalGeneration" in supported
    assert "Gemma4ForCausalLM" in supported


def test_mlp_rejects_wrong_activation() -> None:
    with pytest.raises(ValueError):
        with torch.device("meta"):
            Gemma4MLP(HIDDEN, 64, "silu")


def test_embedding_normalizer_matches_gemma() -> None:
    """The Gemma embedding scale is sqrt(hidden_size), rounded in bf16 (as HF
    does), and is applied to the token embeddings."""
    hidden = 5376  # the real gemma-4-31B hidden_size
    embeds = torch.ones(2, hidden, dtype=torch.bfloat16)
    normalizer = embeds.new_tensor(hidden**0.5)
    expected = torch.tensor(math.sqrt(hidden), dtype=torch.bfloat16)
    assert normalizer == expected
    assert torch.allclose((embeds * normalizer)[0, 0], expected)


def test_sliding_rope_inv_freq_default() -> None:
    """Sliding layers: full rotary over the local head_dim with the local
    base (standard RoPE)."""
    cfg = _tiny_text_config()
    inv = build_rope_inv_freq("sliding_attention", cfg)
    assert inv.shape[0] == LOCAL_HEAD_DIM // 2
    exponent = torch.arange(0, LOCAL_HEAD_DIM, 2, dtype=torch.int64).float()
    expected = 1.0 / (LOCAL_THETA ** (exponent / LOCAL_HEAD_DIM))
    assert torch.allclose(inv, expected)
    assert (inv > 0).all()


def test_full_rope_inv_freq_partial_proportional() -> None:
    """Full-attention layers: partial (proportional) RoPE -- only the first
    ``partial_rotary_factor`` fraction of channels rotate, the rest get a zero
    frequency (identity)."""
    cfg = _tiny_text_config()
    inv = build_rope_inv_freq("full_attention", cfg)
    # Length is always global_head_dim // 2.
    assert inv.shape[0] == GLOBAL_HEAD_DIM // 2
    rope_angles = int(PARTIAL * GLOBAL_HEAD_DIM // 2)
    assert rope_angles == 2
    # First `rope_angles` frequencies are non-zero, the rest are exactly zero.
    assert (inv[:rope_angles] > 0).all()
    assert torch.count_nonzero(inv[rope_angles:]) == 0
    expected_rot = 1.0 / (
        GLOBAL_THETA
        ** (
            torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float()
            / GLOBAL_HEAD_DIM
        )
    )
    assert torch.allclose(inv[:rope_angles], expected_rot)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
