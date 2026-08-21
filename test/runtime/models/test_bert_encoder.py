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
    EncoderRotaryCache,
    NewModel,
    RobertaEmbedding,
    XLMRobertaModel,
    _resolve_bert_activation,
    encoder_batch_layout,
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
    cache = EncoderRotaryCache.from_config(config, torch.float32)

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
    cache = EncoderRotaryCache.from_config(config, torch.float32)
    assert cache.cos_cached.shape[0] == 32


def test_rope_without_scaling_uses_the_plain_base():
    config = _config(
        hidden_size=64,
        num_attention_heads=2,
        rope_theta=10000.0,
        max_position_embeddings=16,
        rope_scaling=None,
    )
    cache = EncoderRotaryCache.from_config(config, torch.float32)
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
        EncoderRotaryCache.from_config(config, torch.float32)


def test_mixed_b_ntk_is_refused():
    config = _config(
        hidden_size=64,
        num_attention_heads=2,
        rope_theta=10000.0,
        rope_scaling={"type": "ntk", "factor": 2.0, "mixed_b": 0.5},
    )
    with pytest.raises(ValueError, match="mixed_b"):
        EncoderRotaryCache.from_config(config, torch.float32)


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
    dest = "encoder.layer.0.attention.attn.qkv_proj.weight"
    assert [(n, sh) for n, _, sh in out] == [
        (dest, "q"),
        (dest, "k"),
        (dest, "v"),
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


# ------------------------------------------------------- modernbert / nomic


def _modernbert_config(**overrides):
    from types import SimpleNamespace

    base = dict(
        vocab_size=64,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=4,
        intermediate_size=16,
        max_position_embeddings=32,
        norm_eps=1e-5,
        norm_bias=False,
        attention_bias=False,
        mlp_bias=False,
        hidden_activation="gelu",
        hidden_act="gelu",
        global_attn_every_n_layers=3,
        local_attention=8,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_modernbert_alternates_global_and_local_layers():
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.bert import ModernBertLayer

    config = _modernbert_config(num_hidden_layers=6)
    mapping = Mapping(world_size=1, rank=0)
    kinds = []
    for layer_id in range(6):
        layer = ModernBertLayer(config, mapping, layer_id=layer_id)
        kinds.append((layer.is_global, layer.attention.window_size))
    # global_attn_every_n_layers=3 -> layers 0, 3 are global; the rest local
    # with a window of local_attention // 2 on each side.
    assert [g for g, _ in kinds] == [True, False, False, True, False, False]
    assert [w for _, w in kinds] == [None, (4, 4), (4, 4), None, (4, 4), (4, 4)]


def test_modernbert_reads_either_config_vocabulary():
    from tokenspeed.runtime.models.bert import modernbert_layer_plan

    v4 = _modernbert_config(num_hidden_layers=6)
    # transformers v5 says the same thing in different words: explicit layer
    # types, per-type rope parameters, and sliding_window as the per-side
    # offset rather than the total width.
    v5 = _modernbert_config(
        num_hidden_layers=6,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
        ],
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
        sliding_window=4,
    )
    assert modernbert_layer_plan(v4) == modernbert_layer_plan(v5)


def test_modernbert_plan_pairs_each_layer_with_its_own_theta():
    from tokenspeed.runtime.models.bert import modernbert_layer_plan

    plan = modernbert_layer_plan(_modernbert_config(num_hidden_layers=4))
    assert [theta for _, _, theta in plan] == [160000.0, 10000.0, 10000.0, 160000.0]


def test_modernbert_layer_zero_has_no_attention_norm():
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.bert import ModernBertLayer

    config = _modernbert_config()
    mapping = Mapping(world_size=1, rank=0)
    # The checkpoint ships no layers.0.attn_norm: the embedding LayerNorm
    # already normalised that input. Identity, not a missing weight.
    assert isinstance(
        ModernBertLayer(config, mapping, layer_id=0).attn_norm, torch.nn.Identity
    )
    assert isinstance(
        ModernBertLayer(config, mapping, layer_id=1).attn_norm, torch.nn.LayerNorm
    )


def test_modernbert_uses_two_rope_tables_when_thetas_differ():
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.bert import ModernBertTransformer

    mapping = Mapping(world_size=1, rank=0)
    model = ModernBertTransformer(_modernbert_config(), mapping)
    # Two distinct thetas -> two tables, and every layer points at the one
    # for its own type.
    assert len(model.rotary_caches) == 2
    assert model._layer_rotary == [0, 1, 1, 0]
    assert not torch.allclose(
        model.rotary_caches[0].cos_cached, model.rotary_caches[1].cos_cached
    )
    # Same theta -> one shared table, not two identical ones.
    same = ModernBertTransformer(_modernbert_config(local_rope_theta=160000.0), mapping)
    assert len(same.rotary_caches) == 1
    assert same._layer_rotary == [0, 0, 0, 0]


def test_modernbert_splits_packed_wqkv_with_shard_ids():
    from tokenspeed.runtime.models.bert import ModernBertModel

    weight = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3 * 4, 2)
    out = list(
        ModernBertModel._iter_weights(
            ModernBertModel, [("layers.0.attn.Wqkv.weight", weight)]
        )
    )
    assert [(n, sh) for n, _, sh in out] == [
        ("layers.0.attention.attn.qkv_proj.weight", "q"),
        ("layers.0.attention.attn.qkv_proj.weight", "k"),
        ("layers.0.attention.attn.qkv_proj.weight", "v"),
    ]
    assert torch.equal(out[0][1], weight[:4])


def test_modernbert_rename_keeps_the_mlp_output_projection():
    from tokenspeed.runtime.models.bert import ModernBertModel

    # attn.Wo must become the attention out-projection while mlp.Wo, which
    # shares the name, must survive untouched.
    assert (
        ModernBertModel._rewrite(ModernBertModel, "layers.3.attn.Wo.weight")
        == "encoder.layers.3.attention.attn.proj.weight"
    )
    assert (
        ModernBertModel._rewrite(ModernBertModel, "layers.3.mlp.Wo.weight")
        == "encoder.layers.3.mlp.Wo.weight"
    )


def test_nomic_maps_fc11_to_up_and_fc12_to_gate():
    from tokenspeed.runtime.models.bert import NomicBertModel

    w = torch.zeros(4, 2)
    out = list(
        NomicBertModel._iter_weights(
            NomicBertModel,
            [
                ("encoder.layers.0.mlp.fc11.weight", w),
                ("encoder.layers.0.mlp.fc12.weight", w),
            ],
        )
    )
    # Shard 0 is up, shard 1 is gate. Swapping them moves the activation onto
    # the other half and still returns a well-formed unit vector.
    assert [(n, sh) for n, _, sh in out] == [
        ("encoder.layers.0.up_gate_proj.weight", 0),
        ("encoder.layers.0.up_gate_proj.weight", 1),
    ]


def test_gte_splits_packed_up_gate_into_up_then_gate():
    from tokenspeed.runtime.models.bert import NewModel

    weight = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2 * 3, 2)
    out = list(
        NewModel._iter_weights(
            NewModel, [("encoder.layer.0.mlp.up_gate_proj.weight", weight)]
        )
    )
    assert [(n, sh) for n, _, sh in out] == [
        ("encoder.layer.0.up_gate_proj.weight", 0),
        ("encoder.layer.0.up_gate_proj.weight", 1),
    ]
    assert torch.equal(out[0][1], weight[:3])
    assert torch.equal(out[1][1], weight[3:])


def test_swiglu_config_string_resolves_to_silu():
    from tokenspeed.runtime.models.bert import _resolve_bert_activation

    x = torch.tensor([0.5, -1.25, 2.0])
    assert torch.allclose(_resolve_bert_activation("silu")(x), x * torch.sigmoid(x))


# ---------------------------------------------------------------- jina v3


def _jina_config(**overrides):
    from types import SimpleNamespace

    base = dict(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=16,
        max_position_embeddings=32,
        type_vocab_size=1,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        pad_token_id=1,
        position_embedding_type="rotary",
        rotary_emb_base=20000.0,
        lora_adaptations=[
            "retrieval.query",
            "retrieval.passage",
            "separation",
            "classification",
            "text-matching",
        ],
        lora_alpha=1,
        lora_rank=4,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _lora_triple(name, base_shape, rank, tasks, embedding_layout):
    """One parametrization triple, laid out the way the checkpoint stores it."""
    rows, cols = base_shape
    base = torch.zeros(rows, cols)
    if embedding_layout:
        a = torch.zeros(tasks, rows, rank)
        b = torch.zeros(tasks, rank, cols)
    else:
        a = torch.zeros(tasks, rank, cols)
        b = torch.zeros(tasks, rows, rank)
    return [
        (f"{name}.original", base),
        (f"{name}.0.lora_A", a),
        (f"{name}.0.lora_B", b),
    ]


def test_jina_lora_merge_picks_the_task_by_name():
    from tokenspeed.runtime.models.bert import merge_jina_lora

    config = _jina_config()
    rank, tasks = 4, 5
    weights = _lora_triple(
        "encoder.layers.0.mixer.Wqkv.parametrizations.weight",
        (6, 8),
        rank,
        tasks,
        embedding_layout=False,
    )
    # Make only the text-matching slice non-zero, so a merge that picked any
    # other adaptation would leave the base untouched.
    lookup = dict(weights)
    lookup["encoder.layers.0.mixer.Wqkv.parametrizations.weight.0.lora_A"][4].fill_(1.0)
    lookup["encoder.layers.0.mixer.Wqkv.parametrizations.weight.0.lora_B"][4].fill_(1.0)

    merged = dict(merge_jina_lora(list(lookup.items()), config, "text-matching"))
    name = "encoder.layers.0.mixer.Wqkv.weight"
    assert name in merged
    # rank 4, alpha 1 -> scaling 0.25; each entry is sum over rank of 1*1.
    assert torch.allclose(merged[name], torch.full((6, 8), rank * 0.25))

    other = dict(merge_jina_lora(list(lookup.items()), config, "classification"))
    assert torch.allclose(other[name], torch.zeros(6, 8))


def test_jina_lora_merge_handles_the_transposed_embedding_layout():
    from tokenspeed.runtime.models.bert import merge_jina_lora

    config = _jina_config()
    weights = dict(
        _lora_triple(
            "embeddings.word_embeddings.parametrizations.weight",
            (32, 8),
            4,
            5,
            embedding_layout=True,
        )
    )
    weights["embeddings.word_embeddings.parametrizations.weight.0.lora_A"][4].fill_(1.0)
    weights["embeddings.word_embeddings.parametrizations.weight.0.lora_B"][4].fill_(1.0)

    merged = dict(merge_jina_lora(list(weights.items()), config, "text-matching"))
    # The embedding's factors are stored transposed relative to a linear's;
    # multiplying them in the linear order would raise on the shapes rather
    # than quietly produce something.
    assert merged["embeddings.word_embeddings.weight"].shape == (32, 8)
    assert torch.allclose(
        merged["embeddings.word_embeddings.weight"], torch.full((32, 8), 1.0)
    )


def test_jina_lora_merge_refuses_an_unknown_task():
    from tokenspeed.runtime.models.bert import merge_jina_lora

    with pytest.raises(ValueError, match="not one of"):
        merge_jina_lora([], _jina_config(), "retrieval")


def test_jina_lora_merge_passes_unparametrised_weights_through():
    from tokenspeed.runtime.models.bert import merge_jina_lora

    bias = torch.ones(6)
    out = dict(
        merge_jina_lora(
            [("encoder.layers.0.mixer.Wqkv.bias", bias)],
            _jina_config(),
            "text-matching",
        )
    )
    assert torch.equal(out["encoder.layers.0.mixer.Wqkv.bias"], bias)


def test_xlm_roberta_dispatches_on_position_embedding_type():
    from tokenspeed.runtime.models.bert import (
        JinaTransformer,
        RobertaTransformer,
        XLMRobertaModel,
    )

    # Two unrelated models answer to XLMRobertaModel; only the config
    # separates multilingual-e5 (absolute) from jina-v3 (rotary).
    assert XLMRobertaModel.model_cls is RobertaTransformer
    assert JinaTransformer.layer_cls.gated_mlp is False
    assert RobertaTransformer.embedding_cls.__name__ == "RobertaEmbedding"


def test_jina_layer_is_the_ungated_member():
    from tokenspeed.runtime.models.bert import JinaLayer, NewLayer

    assert NewLayer.gated_mlp is True
    assert JinaLayer.gated_mlp is False


def test_rope_base_reads_whichever_name_the_family_publishes():
    from tokenspeed.runtime.models.bert import _rope_base

    assert _rope_base(_config(rope_theta=500000.0)) == 500000.0
    # Nomic publishes rotary_emb_base; jina too. A 10000 default here reads
    # as a small systematic error, not a failure.
    assert _rope_base(_jina_config()) == 20000.0
    # rope_parameters wins when it carries the theta.
    assert (
        _rope_base(_config(rope_parameters={"rope_theta": 1000.0}, rope_theta=9.0))
        == 1000.0
    )


def test_rope_base_refuses_to_guess():
    from types import SimpleNamespace

    from tokenspeed.runtime.models.bert import _rope_base

    with pytest.raises(ValueError, match="rope base"):
        _rope_base(SimpleNamespace(hidden_size=8, num_attention_heads=2))
