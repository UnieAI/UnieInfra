"""CPU tests for the encoder-only embedding models.

These cover the parts that a parity run would catch only as "the cosine is
0.7 for reasons unknown": the RoBERTa position offset, the NTK rope
frequencies, the batch layout derived from gather_ids, and the weight rename
tables. All are pure functions of small tensors, so none of it needs a GPU.
"""

import math

import pytest
import torch

from tokenspeed.runtime.models.bert import (
    BertEmbedding,
    BertModel,
    NewModel,
    NTKScalingRotaryCache,
    RobertaEmbedding,
    XLMRobertaModel,
    encoder_batch_layout,
    _resolve_bert_activation,
)


class _Ctx:
    def __init__(self, gather_ids, input_num_tokens):
        self.gather_ids = gather_ids
        self.input_num_tokens = input_num_tokens
        self.forward_mode = None
        self.bs = 0
        self.num_extends = 0


def _config(**overrides):
    # A namespace, not a real PretrainedConfig: transformers 5.12 drops
    # unknown kwargs (rope_scaling among them) instead of setting them, so a
    # PretrainedConfig here would silently test the unscaled path.
    from types import SimpleNamespace

    base = dict(
        vocab_size=64,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=16,
        max_position_embeddings=32,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        hidden_act="gelu",
        pad_token_id=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------- batch layout


def test_batch_layout_recovers_cu_seqlens():
    # gather_ids is cumsum(lens) - 1 for lens [3, 5, 2].
    ctx = _Ctx(torch.tensor([2, 7, 9]), 10)
    cu, max_seqlen = encoder_batch_layout(ctx)
    assert cu.tolist() == [0, 3, 8, 10]
    assert cu.dtype == torch.int32
    # An upper bound, not the exact maximum: deriving the exact value needs
    # a device sync that is illegal under CUDA graph capture.
    assert max_seqlen >= 5


def test_batch_layout_single_request():
    ctx = _Ctx(torch.tensor([6]), 7)
    cu, _ = encoder_batch_layout(ctx)
    assert cu.tolist() == [0, 7]


def test_batch_layout_refuses_a_decode_forward():
    # No gather_ids means this is not an extend forward, and attending with
    # a guessed layout would return a well-formed wrong vector.
    with pytest.raises(RuntimeError, match="gather_ids"):
        encoder_batch_layout(_Ctx(None, 4))


# ------------------------------------------------------------ position offset


def test_roberta_offsets_positions_by_pad_plus_one():
    config = _config(pad_token_id=1, max_position_embeddings=34)
    bert = BertEmbedding(config)
    roberta = RobertaEmbedding(config)
    assert bert.position_offset == 0
    assert roberta.position_offset == 2

    # Same table, same tokens: RoBERTa must read two rows further in.
    # A non-linear table on purpose: LayerNorm maps every row of an
    # arithmetic sequence onto the same normalized pattern, so a table of
    # aranges would make offset and no-offset agree and the test would pass
    # even with the offset dropped.
    table = torch.sin(torch.arange(34 * 8, dtype=torch.float32) * 0.7).reshape(34, 8)
    for module in (bert, roberta):
        module.position_embeddings.weight.data.copy_(table)
        module.word_embeddings.weight.data.zero_()
        module.token_type_embeddings.weight.data.zero_()
        module.LayerNorm.weight.data.fill_(1.0)
        module.LayerNorm.bias.data.zero_()

    ids = torch.zeros(3, dtype=torch.long)
    positions = torch.arange(3)
    with torch.no_grad():
        a = bert(ids, positions)
        b = roberta(ids, positions)
        shifted = bert(ids, positions + 2)
    assert torch.allclose(b, shifted)
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------- rope


def test_ntk_rope_matches_the_closed_form():
    """Reproduce the checkpoint's own two-step NTK rescale.

    base' = base * factor, then inv_freq /= factor ** (2 / head_dim). Both
    steps change every position, not just positions past the training
    window, which is why an unscaled cache is wrong even for short prompts.
    """
    config = _config(
        hidden_size=64,
        num_attention_heads=2,
        rope_theta=500000.0,
        max_position_embeddings=16,
        rope_scaling={"type": "ntk", "factor": 2.0},
    )
    cache = NTKScalingRotaryCache(config, torch.float32)

    head_dim = 32
    base = 500000.0 * 2.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    inv_freq = inv_freq / 2.0 ** (2 / head_dim)

    positions = torch.arange(5)
    cos, sin = cache(positions)
    expect = torch.outer(positions.float(), inv_freq)
    assert torch.allclose(cos, expect.cos(), atol=1e-6)
    assert torch.allclose(sin, expect.sin(), atol=1e-6)
    # Half the head dim; the attention layer doubles it, matching the
    # reference's cat((freqs, freqs)).
    assert cos.shape == (5, head_dim // 2)


def test_ntk_cache_is_long_enough_for_the_scaled_window():
    config = _config(
        hidden_size=64,
        num_attention_heads=2,
        rope_theta=10000.0,
        max_position_embeddings=16,
        rope_scaling={"type": "ntk", "factor": 2.0},
    )
    cache = NTKScalingRotaryCache(config, torch.float32)
    assert cache.cos_cached.shape[0] == 32


def test_rope_without_scaling_uses_the_plain_base():
    config = _config(
        hidden_size=64,
        num_attention_heads=2,
        rope_theta=10000.0,
        max_position_embeddings=16,
        rope_scaling=None,
    )
    cache = NTKScalingRotaryCache(config, torch.float32)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    cos, _ = cache(torch.arange(3))
    assert torch.allclose(
        cos, torch.outer(torch.arange(3).float(), inv_freq).cos(), atol=1e-6
    )


def test_unknown_rope_scaling_is_refused_not_ignored():
    config = _config(
        hidden_size=64,
        num_attention_heads=2,
        rope_theta=10000.0,
        rope_scaling={"type": "yarn", "factor": 2.0},
    )
    with pytest.raises(ValueError, match="ntk"):
        NTKScalingRotaryCache(config, torch.float32)


def test_mixed_b_ntk_is_refused():
    config = _config(
        hidden_size=64,
        num_attention_heads=2,
        rope_theta=10000.0,
        rope_scaling={"type": "ntk", "factor": 2.0, "mixed_b": 0.5},
    )
    with pytest.raises(ValueError, match="mixed_b"):
        NTKScalingRotaryCache(config, torch.float32)


# ------------------------------------------------------------------ renaming


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "encoder.layer.3.attention.output.dense.weight",
            "encoder.layers.3.attention.attn.proj.weight",
        ),
        (
            "encoder.layer.3.attention.output.LayerNorm.bias",
            "encoder.layers.3.attention_LayerNorm.bias",
        ),
        (
            "encoder.layer.0.intermediate.dense.weight",
            "encoder.layers.0.intermediate.weight",
        ),
        ("encoder.layer.0.output.dense.bias", "encoder.layers.0.output.bias"),
        (
            "encoder.layer.0.output.LayerNorm.weight",
            "encoder.layers.0.output_LayerNorm.weight",
        ),
        ("embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"),
    ],
)
def test_bert_rename_table(source, expected):
    assert BertModel._rewrite(BertModel, source) == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "encoder.layer.1.attention.o_proj.weight",
            "encoder.layers.1.attention.attn.proj.weight",
        ),
        (
            "encoder.layer.1.mlp.up_gate_proj.weight",
            "encoder.layers.1.up_gate_proj.weight",
        ),
        ("encoder.layer.1.mlp.down_proj.bias", "encoder.layers.1.down_proj.bias"),
        ("encoder.layer.1.attn_ln.weight", "encoder.layers.1.attn_ln.weight"),
        ("encoder.layer.1.mlp_ln.bias", "encoder.layers.1.mlp_ln.bias"),
    ],
)
def test_new_model_rename_table(source, expected):
    assert NewModel._rewrite(NewModel, source) == expected


def test_new_model_splits_packed_qkv_into_three_shards():
    weight = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3 * 4, 2)
    out = list(
        NewModel._iter_weights(
            NewModel, [("encoder.layer.0.attention.qkv_proj.weight", weight)]
        )
    )
    assert [n for n, _ in out] == [
        "encoder.layer.0.attention.self.query.weight",
        "encoder.layer.0.attention.self.key.weight",
        "encoder.layer.0.attention.self.value.weight",
    ]
    # Contiguous [q; k; v], the layout the reference's
    # ``qkv_proj(x).split(all_head_size, dim=-1)`` implies.
    assert torch.equal(out[0][1], weight[:4])
    assert torch.equal(out[1][1], weight[4:8])
    assert torch.equal(out[2][1], weight[8:])


def test_non_qkv_weights_pass_through_unchanged():
    w = torch.zeros(2)
    out = list(NewModel._iter_weights(NewModel, [("encoder.layer.0.mlp_ln.bias", w)]))
    assert out == [("encoder.layer.0.mlp_ln.bias", w)]


def test_xlm_roberta_reuses_the_roberta_stack():
    from tokenspeed.runtime.models.bert import RobertaTransformer

    assert XLMRobertaModel.model_cls is RobertaTransformer


# --------------------------------------------------------------- activations


def test_geglu_config_string_resolves_to_exact_gelu():
    # GTE's config says "gelu" while its MLP is gated; vLLM rewrites the
    # field to "geglu". Both must land on the erf gelu, not the tanh
    # approximation, because that is what the reference's ACT2FN["gelu"] is.
    x = torch.tensor([0.5, -1.25, 2.0])
    for name in ("gelu", "geglu"):
        assert torch.allclose(
            _resolve_bert_activation(name)(x),
            torch.erf(x / math.sqrt(2.0)).add(1.0).mul(0.5).mul(x),
            atol=1e-6,
        )


def test_tanh_gelu_is_a_different_function():
    x = torch.tensor([2.0])
    exact = _resolve_bert_activation("gelu")(x)
    tanh = _resolve_bert_activation("gelu_pytorch_tanh")(x)
    assert not torch.allclose(exact, tanh, atol=1e-9)


def test_unsupported_activation_is_refused():
    with pytest.raises(ValueError, match="hidden_act"):
        _resolve_bert_activation("swiglu")
