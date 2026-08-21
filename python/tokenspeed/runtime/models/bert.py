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

"""Encoder-only text embedding models: BERT, RoBERTa/XLM-R, and GTE-v1.5.

Until now every pooling model this engine served was a *decoder* with a
pooler bolted on (Qwen3-Embedding): causal attention, a paged KV pool, LAST
reduction. The BERT family is the other shape -- bidirectional attention, no
KV at all, one forward and done. ``unieinfra/xref/15-encoder-embedding.md``
§1 is the four-engine survey; this file is the port.

Three architectures, because "BERT / GTE / BGE" is not three checkpoints of
one architecture:

  * ``BertModel``     -- ``bert-base``, ``BAAI/bge-*-en-v1.5``,
                         ``thenlper/gte-*``. Learned absolute positions,
                         ungated MLP, post-LN.
  * ``XLMRobertaModel`` / ``RobertaModel`` -- ``intfloat/multilingual-e5-*``,
                         ``BAAI/bge-m3``. Identical to BERT except that
                         position ids start at ``pad_token_id + 1``, which is
                         why those configs carry ``max_position_embeddings``
                         514 rather than 512.
  * ``NewModel``      -- ``Alibaba-NLP/gte-*-en-v1.5``. RoPE instead of
                         learned positions, a *gated* MLP, and NTK rope
                         scaling.

Two traps in the third one, both of which produce a plausible wrong vector
rather than an error:

  1. Its config says ``hidden_act: "gelu"`` but the checkpoint ships
     ``mlp.up_gate_proj`` -- the MLP is GeGLU. vLLM rewrites the field
     (``model_executor/models/config.py:391 config.hidden_act = "geglu"``);
     the reference implementation reaches the same place because its
     ``NewGatedMLP`` applies ``ACT2FN[hidden_act]`` to the *gate* half only.
     Reading the config at face value gives an ungated MLP that loads
     cleanly against half the weights.
  2. ``rope_scaling {"type": "ntk", "factor": 2.0}`` rescales the *base
     frequency*, so it changes every position, not just positions past the
     training window. A short-prompt test would not catch skipping it.

Attention runs through ``mm_encoder_attention``'s varlen backends -- the same
pool-free non-causal path Whisper's encoder uses. Nothing here allocates,
reads or writes a KV page, and nothing here consults the attention backend:
the batch layout comes from ``ForwardContext.gather_ids``, which is the same
tensor ``BaseCausalLM.pool`` already trusts to say where each request's rows
start and end.
"""

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.mm_encoder_attention import (
    MultimodalEncoderAttention,
)
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.pooler import Pooler, PoolerOutput, PoolingType
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.utils import add_prefix, get_colorful_logger

# Set TOKENSPEED_ENCODER_DEBUG=1 to log the batch layout each forward.
_DEBUG_BATCH = bool(os.environ.get("TOKENSPEED_ENCODER_DEBUG"))

logger = get_colorful_logger(__name__)

__all__ = [
    "BertModel",
    "RobertaModel",
    "XLMRobertaModel",
    "NewModel",
    "GteModel",
    "NomicBertModel",
    "ModernBertModel",
    "BertForSequenceClassification",
    "RobertaForSequenceClassification",
    "XLMRobertaForSequenceClassification",
    "NewForSequenceClassification",
]


def encoder_batch_layout(ctx: ForwardContext) -> tuple[torch.Tensor, int]:
    """Return ``(cu_seqlens, max_seqlen_bound)`` for this forward.

    Called from inside the attention break, so ``ctx`` is the *live* forward
    context even under a graph replay -- ``break_point`` rebinds the context
    argument by identity to the ambient one on every replay. Computing this
    in the captured region instead would freeze the capture-time dummy
    batch's boundaries into every replay, which is exactly the bug this
    file's history records.

    Taken from ``ctx.gather_ids`` rather than from the attention backend's
    extend metadata. Two reasons: an encoder-only model has no attention
    layer to drive that backend, and ``gather_ids`` is already the tensor
    ``BaseCausalLM.pool`` reduces over, so attention and pooling cannot
    disagree about where a request's rows are.

    ``gather_ids`` on an extend forward is ``cumsum(extend_lens) - 1``, so
    ``cat([0, gather_ids + 1])`` *is* the cu_seqlens vector.

    The second return value is an upper bound on the longest sequence, not
    the exact maximum: deriving the exact value needs ``.item()``, a device
    sync. Under a padded replay ``input_num_tokens`` reads as the bucket
    (``PrefillGraph._padded_to`` sets it), which is still a bound on any one
    request. Varlen kernels use it only to size the grid and mask by the
    per-request lengths, so an over-estimate costs empty blocks, never wrong
    rows.
    """
    if ctx.gather_ids is None:
        raise RuntimeError(
            "encoder-only attention requires ForwardContext.gather_ids, which "
            "only an extend forward carries. Got forward_mode="
            f"{ctx.forward_mode} bs={ctx.bs} num_extends={ctx.num_extends} "
            f"input_num_tokens={ctx.input_num_tokens}"
        )
    ends = (ctx.gather_ids + 1).to(torch.int32)
    cu_seqlens = torch.cat([ends.new_zeros(1), ends])
    return cu_seqlens, int(ctx.input_num_tokens)


# A cross-encoder scores a *pair*, and BERT tells the two halves apart with
# segment ids. Rather than thread a second per-token tensor through the
# engine -- a new persistent buffer, a new request field, and a new thing for
# the prefill graph to refresh -- the segment rides in the high bits of the
# token id, the same trick vLLM uses
# (``model_executor/models/bert.py:524 _encode_token_type_ids``).
# Vocabularies here reach ~250k, so bit 30 of an id is free.
TOKEN_TYPE_SHIFT = 30


def encode_token_type_ids(
    input_ids: torch.Tensor, token_type_ids: torch.Tensor
) -> torch.Tensor:
    """Pack segment ids into ``input_ids``; returns a new tensor."""
    return input_ids | (token_type_ids << TOKEN_TYPE_SHIFT)


def decode_token_type_ids(
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack ``(real_ids, token_type_ids)``.

    Deliberately not in place. vLLM masks ``input_ids`` in place because
    everything entering a ``@support_torch_compile`` region has to be a
    persistent tensor; here the ids arrive in a static buffer the prefill
    graph refreshes, and mutating it would leave a half-decoded buffer for
    whatever reads it after this layer -- including the graph's own
    out-of-capture word gather.
    """
    mask = 1 << TOKEN_TYPE_SHIFT
    return input_ids & ~mask, (input_ids & mask) >> TOKEN_TYPE_SHIFT


class BertEmbedding(nn.Module):
    """Word + learned-position + token-type, then LayerNorm."""

    # Positions start here; RoBERTa overrides it (see class below).
    position_offset: int = 0

    def __init__(self, config: PretrainedConfig, prefix: str = "") -> None:
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("word_embeddings", prefix),
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        # ``type_vocab_size`` is 0 on some encoder configs; a single-sequence
        # embedding request is all-zeros token types either way, so the
        # segment embedding is simply absent rather than a zero lookup.
        self.token_type_embeddings = (
            nn.Embedding(config.type_vocab_size, config.hidden_size)
            if getattr(config, "type_vocab_size", 0) > 0
            else None
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        word_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ``word_embeds`` is the prefill graph's static input-embeds buffer:
        # the graph runs the word gather outside the capture and hands the
        # result back. Everything after it reads ``positions``, so it stays
        # inside.
        token_type_ids = None
        if self.token_type_embeddings is not None:
            input_ids, token_type_ids = decode_token_type_ids(input_ids)
        embeddings = (
            self.word_embeddings(input_ids) if word_embeds is None else word_embeds
        )
        embeddings = embeddings + self.position_embeddings(
            positions + self.position_offset
        )
        if token_type_ids is not None:
            # Segment 0 everywhere for an ordinary embedding request; 0 for
            # the query and 1 for the document on a cross-encoder pair.
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        return self.LayerNorm(embeddings)


class RobertaEmbedding(BertEmbedding):
    """BERT's embedding with RoBERTa's position offset.

    RoBERTa reserves ``[0, pad_token_id]`` in the position table, so real
    token positions begin at ``pad_token_id + 1``. This is the whole
    difference between the two families' forwards, and getting it wrong
    yields a fluent-looking but uniformly shifted embedding.
    """

    def __init__(self, config: PretrainedConfig, prefix: str = "") -> None:
        super().__init__(config, prefix)
        self.position_offset = int(config.pad_token_id) + 1


class NewEmbedding(nn.Module):
    """GTE-v1.5: word (+ optional token type), LayerNorm, no position table.

    Positions enter through RoPE inside each attention layer instead.
    """

    def __init__(self, config: PretrainedConfig, prefix: str = "") -> None:
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("word_embeddings", prefix),
        )
        self.token_type_embeddings = (
            nn.Embedding(config.type_vocab_size, config.hidden_size)
            if getattr(config, "type_vocab_size", 0) > 0
            else None
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        word_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_type_ids = None
        if self.token_type_embeddings is not None:
            input_ids, token_type_ids = decode_token_type_ids(input_ids)
        embeddings = (
            self.word_embeddings(input_ids) if word_embeds is None else word_embeds
        )
        if token_type_ids is not None:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        return self.LayerNorm(embeddings)


def _rope_base(config: PretrainedConfig) -> float:
    """The rope theta, under whichever name this family publishes it.

    GTE says ``rope_theta``. Nomic says ``rotary_emb_base`` (1000 -- an order
    of magnitude off the usual default) and also carries the normalised
    ``rope_parameters``. Falling through to a 10000 default reads as a small
    systematic error, not as a failure: measured cosine 0.97-0.99 against the
    reference, which is wrong but not obviously wrong.
    """
    parameters = getattr(config, "rope_parameters", None)
    if isinstance(parameters, dict) and "rope_theta" in parameters:
        return float(parameters["rope_theta"])
    for name in ("rope_theta", "rotary_emb_base"):
        value = getattr(config, name, None)
        if value is not None:
            return float(value)
    raise ValueError(
        f"{type(config).__name__} declares no rope base "
        "(rope_parameters/rope_theta/rotary_emb_base); refusing to guess one, "
        "because a wrong base returns a plausible embedding."
    )


class EncoderRotaryCache(nn.Module):
    """RoPE cos/sin for the encoder families that use it.

    Built from explicit ``base`` / ``max_position`` rather than from the
    config, because ModernBERT needs two of these with *different* thetas in
    one model -- its local layers use ``local_rope_theta`` and its global
    layers ``global_rope_theta``. :meth:`from_config` is the GTE path, which
    reads them off the config and applies NTK scaling.

    Ported from the checkpoint's own reference implementation
    (``Alibaba-NLP/new-impl`` ``modeling.py`` ``NTKScalingRotaryEmbedding``),
    which vLLM mirrors as ``NTKScalingRotaryEmbedding`` in
    ``model_executor/layers/rotary_embedding/``. This tree's ``get_rope``
    has no ``"ntk"`` scaling type, and adding one would touch the rotary
    layer every decoder model shares, so the cache lives here.

    The rescale is *not* conditional in practice. The reference builds the
    cache twice: once for ``max_position_embeddings`` (no rescale, since the
    guard is a strict ``>``) and then again for
    ``max_position_embeddings * factor``, which does trip the guard. The
    second call wins, so the served frequencies are always the scaled ones:

        base' = base * factor          inv_freq = 1 / base'^(2i/d)
        inv_freq /= factor^(2/d)

    Both steps matter at every position, which is why a short-prompt parity
    test would still fail if this were skipped -- it is not a long-context
    refinement.
    """

    @classmethod
    def from_config(
        cls, config: PretrainedConfig, dtype: torch.dtype
    ) -> "EncoderRotaryCache":
        return cls(
            head_dim=config.hidden_size // config.num_attention_heads,
            base=_rope_base(config),
            max_position=int(config.max_position_embeddings),
            dtype=dtype,
            scaling=getattr(config, "rope_scaling", None),
        )

    def __init__(
        self,
        head_dim: int,
        base: float,
        max_position: int,
        dtype: torch.dtype,
        scaling: dict | None = None,
    ) -> None:
        super().__init__()
        scaling = scaling or {}
        scaling_type = scaling.get("type") or scaling.get("rope_type")
        factor = float(scaling.get("factor", 1.0)) if scaling else 1.0
        if scaling_type == "default":
            # Present-but-inert: transformers normalises an unscaled rope into
            # ``{"rope_type": "default"}``. Treat it as no scaling rather than
            # as an unknown type.
            factor = 1.0
            scaling = {}
        if scaling and scaling_type not in ("ntk",):
            raise ValueError(
                f"{type(self).__name__} supports rope_scaling type 'ntk'; got "
                f"{scaling_type!r}. Add the variant here rather than letting "
                "it fall through to unscaled frequencies, which would return "
                "a well-formed but wrong embedding."
            )
        if scaling.get("mixed_b") is not None:
            # The reference supports mixed-b NTK; no published GTE checkpoint
            # sets it, so refuse rather than silently ignore the field.
            raise ValueError("mixed_b NTK rope scaling is not implemented")

        positions = max_position
        if factor != 1.0:
            positions = int(max_position * factor)
            base = base * factor

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        if factor != 1.0:
            inv_freq = inv_freq / factor ** (2 / head_dim)

        freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv_freq)
        self.register_buffer("cos_cached", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(dtype), persistent=False)
        # Logged, not asserted: NTK scaling is invisible in the output shape,
        # so this line is how an operator tells a scaled server from an
        # unscaled one that returns equally well-formed vectors.
        logger.info(
            "encoder rope: base=%s factor=%s positions=%d head_dim=%d dtype=%s",
            base,
            factor,
            positions,
            head_dim,
            dtype,
        )

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [num_tokens, head_dim // 2]; the attention layer doubles it to the
        # full head dim, matching the reference's cat((freqs, freqs)).
        return (
            self.cos_cached.index_select(0, positions),
            self.sin_cached.index_select(0, positions),
        )


class EncoderSelfAttention(nn.Module):
    """Bidirectional self-attention over a varlen batch, with no KV cache.

    Thin wrapper over the shared ``MultimodalEncoderAttention``, which is
    already the pool-free non-causal path (Whisper's encoder uses it, and
    ``xref`` §7 Q1 records that this engine's answer to "where does the
    non-causal encoder pass live" was settled before this port started).
    The wrapper exists only to hold the varlen call shape in one place.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mm_attention_backend: str | None = None,
        window_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        # (left, right) for a local layer, None for a global one. Frozen at
        # construction because it is a property of the layer, not of the
        # batch -- a break's loose scalar args are frozen to their
        # capture-time value anyway, so per-request quantities could not live
        # here even if they wanted to.
        self.window_size = window_size
        self.attn = MultimodalEncoderAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            mapping=mapping,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            quant_config=quant_config,
            prefix=prefix,
            mm_attention_backend=mm_attention_backend,
        )

    @break_point
    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mix across the sequence. Runs eagerly, including under a replay.

        This is the one method in the file that must not be captured. It is
        where the batch layout enters the computation, and the layout is the
        one thing a padded-bucket replay changes from forward to forward:
        ``break_point`` reruns the body on every replay with ``ctx`` rebound
        by identity to the live ambient context, so ``cu_seqlens`` is
        recomputed each time. The norms, projections and MLP around it are
        token-shaped and stay captured, which is where the speedup comes
        from.

        ``cos``/``sin`` arrive as separate tensors rather than one tuple so
        the break machinery weak-refs them; a tuple would pass through as an
        opaque object and pin its pool block for the graph's lifetime.
        """
        cu_seqlens, max_seqlen = encoder_batch_layout(ctx)
        return self.attn(
            hidden_states.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_pos_emb_cos=cos,
            rotary_pos_emb_sin=sin,
            window_size=self.window_size,
        ).squeeze(0)


class BertLayer(nn.Module):
    """Post-LN block: LayerNorm *after* each residual add, not before.

    Every decoder in this tree is pre-LN, so the ordering here is the one
    thing worth reading twice.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        attn = mapping.vision
        self.attention = EncoderSelfAttention(
            config,
            mapping,
            quant_config=quant_config,
            prefix=add_prefix("attention", prefix),
            mm_attention_backend=mm_attention_backend,
        )
        self.attention_LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.intermediate = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("intermediate", prefix),
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
        )
        self.output = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("output", prefix),
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
            reduce_results=True,
        )
        self.output_LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.act_fn = _resolve_bert_activation(config.hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ``cos``/``sin`` are always None here -- BERT and RoBERTa carry
        # position information in the embedding table, not in the attention
        # scores. The parameters exist so the encoder stack calls every layer
        # type with one signature.
        attended = self.attention(hidden_states, ctx)
        hidden_states = self.attention_LayerNorm(hidden_states + attended)

        intermediate, _ = self.intermediate(hidden_states)
        intermediate = self.act_fn(intermediate)
        projected, _ = self.output(intermediate)
        return self.output_LayerNorm(hidden_states + projected)


class NewLayer(nn.Module):
    """Post-LN block with RoPE attention: GTE-v1.5, Nomic, Jina v3.

    ``gated_mlp`` is a class attribute rather than a config read because the
    three checkpoints that share this block cannot be told apart by their
    config: GTE and Jina both say ``hidden_act: "gelu"`` and only one of them
    is gated. vLLM distinguishes them by rewriting the field per architecture
    (``models/config.py``); the architecture name is the actual signal, and
    that is what the registry already keys on.
    """

    gated_mlp: bool = True

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        attn = mapping.vision
        # GTE carries a bias on qkv and o_proj; Nomic carries none. Read off
        # the config rather than defaulted -- a bias this model allocates but
        # the checkpoint never fills is uninitialised memory, and the
        # loader's unfilled-parameter check is what turns that into an error
        # instead of a plausible vector.
        attn_bias = bool(getattr(config, "qkv_proj_bias", True))
        self.attention = EncoderSelfAttention(
            config,
            mapping,
            quant_config=quant_config,
            prefix=add_prefix("attention", prefix),
            qkv_bias=attn_bias,
            proj_bias=attn_bias,
            mm_attention_backend=mm_attention_backend,
        )
        self.attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        mlp_in_bias = bool(getattr(config, "mlp_fc1_bias", not self.gated_mlp))
        if self.gated_mlp:
            # Two shards rather than one wide projection, so the same module
            # serves both layouts: GTE ships them packed as one
            # ``mlp.up_gate_proj`` tensor (split into shards at load), Nomic
            # ships them already separate as ``mlp.fc11`` / ``mlp.fc12``.
            # Shard 0 is up, shard 1 is gate.
            self.up_gate_proj = MergedColumnParallelLinear(
                config.hidden_size,
                [config.intermediate_size, config.intermediate_size],
                bias=mlp_in_bias,
                quant_config=quant_config,
                prefix=add_prefix("up_gate_proj", prefix),
                tp_rank=attn.tp_rank,
                tp_size=attn.tp_size,
                tp_group=attn.tp_group,
            )
        else:
            # Jina v3 is the ungated member: one projection, activation, one
            # projection back.
            self.fc1 = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                bias=mlp_in_bias,
                quant_config=quant_config,
                prefix=add_prefix("fc1", prefix),
                tp_rank=attn.tp_rank,
                tp_size=attn.tp_size,
                tp_group=attn.tp_group,
            )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=bool(getattr(config, "mlp_fc2_bias", True)),
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
            reduce_results=True,
        )
        self.mlp_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate_size = config.intermediate_size
        # GTE's config says "gelu" and means GeGLU; Nomic's says
        # "swiglu" and means silu. Both put the activation on the gate half.
        act = getattr(config, "activation_function", None) or config.hidden_act
        self.act_fn = _resolve_bert_activation("silu" if act == "swiglu" else act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self.attention(hidden_states, ctx, cos, sin)
        hidden_states = self.attn_ln(hidden_states + attended)

        if self.gated_mlp:
            up_gate, _ = self.up_gate_proj(hidden_states)
            # Shard 0 is up, shard 1 is gate -- the opposite of this tree's
            # fused gate_up convention, and the order both gated checkpoints
            # in this family use once their tensors are named.
            up_states, gate = up_gate.split(up_gate.shape[-1] // 2, dim=-1)
            inner = self.act_fn(gate) * up_states
        else:
            inner, _ = self.fc1(hidden_states)
            inner = self.act_fn(inner)
        projected, _ = self.down_proj(inner)
        return self.mlp_ln(hidden_states + projected)


class ModernBertEmbedding(nn.Module):
    """Token lookup then LayerNorm. No position table -- ModernBERT is RoPE."""

    def __init__(self, config: PretrainedConfig, prefix: str = "") -> None:
        super().__init__()
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("tok_embeddings", prefix),
        )
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=_modernbert_norm_eps(config),
            bias=bool(getattr(config, "norm_bias", True)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        word_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = (
            self.tok_embeddings(input_ids) if word_embeds is None else word_embeds
        )
        return self.norm(embeddings)


def modernbert_layer_plan(
    config: PretrainedConfig,
) -> list[tuple[bool, tuple[int, int] | None, float]]:
    """Per layer: ``(is_global, window, rope_theta)``.

    Two config vocabularies describe the same model, and a server has to read
    whichever one it is handed:

      * **transformers v5** publishes ``layer_types`` per layer, a
        ``rope_parameters`` dict keyed by layer type, and ``sliding_window``
        (64 here) as the per-side offset.
      * **transformers v4** publishes ``global_attn_every_n_layers``,
        ``global_rope_theta`` / ``local_rope_theta``, and ``local_attention``
        as the *total* window width, so each side is half of it.

    They agree on this checkpoint -- v5's ``layer_types`` is exactly
    ``i % 3 == 0``, and its ``sliding_window`` is exactly
    ``local_attention // 2`` -- which is worth knowing, because a server that
    reads the wrong vocabulary still runs and still returns a plausible
    embedding. transformers 5.12 drops the v4 names entirely.

    The window is ``(left, right)`` in ``flash_attn_varlen_func``'s terms.
    vLLM's ``sliding_window + 1`` (``models/modernbert.py:96``) is a
    different quantity: its attention layer counts window *positions* where
    this one takes offsets, and the reference implementation passes
    ``(local_attention // 2, local_attention // 2)``.
    """
    num_layers = int(config.num_hidden_layers)
    layer_types = getattr(config, "layer_types", None)
    half = getattr(config, "sliding_window", None)
    if half is None:
        half = int(config.local_attention) // 2
    window = (int(half), int(half))

    plan: list[tuple[bool, tuple[int, int] | None, float]] = []
    if layer_types:
        rope_parameters = config.rope_parameters
        for layer_id in range(num_layers):
            layer_type = layer_types[layer_id]
            is_global = layer_type != "sliding_attention"
            theta = float(rope_parameters[layer_type]["rope_theta"])
            plan.append((is_global, None if is_global else window, theta))
        return plan

    every = int(getattr(config, "global_attn_every_n_layers", 1) or 1)
    global_theta = float(config.global_rope_theta)
    local_theta = float(getattr(config, "local_rope_theta", None) or global_theta)
    for layer_id in range(num_layers):
        is_global = layer_id % every == 0
        plan.append(
            (
                is_global,
                None if is_global else window,
                global_theta if is_global else local_theta,
            )
        )
    return plan


def _modernbert_norm_eps(config: PretrainedConfig) -> float:
    return float(
        getattr(config, "norm_eps", None)
        or getattr(config, "layer_norm_eps", None)
        or 1e-5
    )


class ModernBertLayer(nn.Module):
    """Pre-LN block with alternating global and local attention.

    Three things differ from the other families in this file, and each one
    silently produces a plausible embedding if you get it wrong:

      * **Pre-LN.** The norm runs *before* attention and MLP and the residual
        adds their raw output; BERT, RoBERTa and GTE all post-LN.
      * **Layer 0 has no attention norm.** The checkpoint ships no
        ``layers.0.attn_norm``, because the embedding LayerNorm already
        normalised its input. It is ``Identity``, not a missing weight.
      * **Two of every three layers are local**, with their own rope theta.
        ``local_attention`` in the config is the *total* window width, so
        each side gets half.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        attn = mapping.vision
        eps = _modernbert_norm_eps(config)
        norm_bias = bool(getattr(config, "norm_bias", True))
        self.is_global, window, _ = modernbert_layer_plan(config)[layer_id]

        self.attn_norm = (
            nn.Identity()
            if layer_id == 0
            else nn.LayerNorm(config.hidden_size, eps=eps, bias=norm_bias)
        )
        self.attention = EncoderSelfAttention(
            config,
            mapping,
            quant_config=quant_config,
            prefix=add_prefix("attention", prefix),
            qkv_bias=bool(getattr(config, "attention_bias", True)),
            proj_bias=bool(getattr(config, "attention_bias", True)),
            # Not the platform default: the local layers need a windowed
            # kernel, and running the global and local halves of one model on
            # two different backends would make any divergence between them
            # impossible to attribute.
            #
            # ``sdpa_window`` rather than the faster ``fa_varlen``: that one
            # honours the window on this card but raises
            # cudaErrorIllegalAddress intermittently once a stack of
            # attention modules runs in one process. Override with
            # --mm-attention-backend fa_varlen once that is fixed; the
            # measurement and the reasoning are in xref §6.8.
            mm_attention_backend=mm_attention_backend or "sdpa_window",
            window_size=window,
        )
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=eps, bias=norm_bias)
        mlp_bias = bool(getattr(config, "mlp_bias", True))
        self.mlp = nn.Module()
        self.mlp.Wi = ColumnParallelLinear(
            config.hidden_size,
            int(config.intermediate_size) * 2,
            bias=mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("mlp.Wi", prefix),
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
        )
        self.mlp.Wo = RowParallelLinear(
            int(config.intermediate_size),
            config.hidden_size,
            bias=mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("mlp.Wo", prefix),
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
            reduce_results=True,
        )
        self.act_fn = _resolve_bert_activation(
            getattr(config, "hidden_activation", None) or config.hidden_act
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.attn_norm(hidden_states), ctx, cos, sin
        )
        gated, _ = self.mlp.Wi(self.mlp_norm(hidden_states))
        # ``act(first) * second``: the activation goes on the *first* half
        # here, the opposite of GTE's ``up_gate_proj``. Both orders load
        # cleanly and both return a unit-norm vector.
        first, second = gated.split(gated.shape[-1] // 2, dim=-1)
        projected, _ = self.mlp.Wo(self.act_fn(first) * second)
        return hidden_states + projected


def _resolve_bert_activation(name: str):
    """Map an HF ``hidden_act`` string to a callable.

    Small and local on purpose: the encoder configs in scope use only these
    three, and routing through the fused gate/up activation modules would
    misread GTE, whose gate and up halves are stored in the opposite order.
    """
    if name in ("gelu", "geglu"):
        return F.gelu
    if name == "gelu_new" or name == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "silu":
        return F.silu
    if name == "relu":
        return F.relu
    raise ValueError(f"unsupported encoder hidden_act: {name!r}")


class BertTransformer(nn.Module):
    """Embeddings + a stack of post-LN encoder blocks.

    Satisfies the call contract ``BaseCausalLM.forward`` expects --
    ``(input_ids, positions, ctx, out_cache_loc, **kwargs)`` returning
    ``(hidden_states, aux_hidden_states)`` -- without subclassing
    ``BaseTransformerModel``, which builds an RMSNorm final norm and a causal
    decoder stack that this architecture has neither of.

    ``out_cache_loc`` is accepted and ignored: the scheduler still allocates
    KV slots for the batch, and this model has no layer that writes them.
    Removing that allocation is a separate change with its own switch.
    """

    embedding_cls: type[nn.Module] = BertEmbedding
    layer_cls: type[nn.Module] = BertLayer

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        backend = getattr(config, "mm_attention_backend", None)
        self.embeddings = self.embedding_cls(
            config, prefix=add_prefix("embeddings", prefix)
        )
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [
                self.layer_cls(
                    config,
                    mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"encoder.layers.{idx}", prefix),
                    mm_attention_backend=backend,
                )
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.layers_to_capture: list[int] = []

    @property
    def embed_tokens(self) -> nn.Module:
        """The word lookup, under the name the prefill graph looks for.

        ``PrefillGraph`` runs this one gather outside the captured region and
        hands the result back as ``input_embeds``. For this family that is
        exactly the word table and nothing else: the position and segment
        terms read ``positions``, a static buffer the graph refreshes, so
        they belong inside the capture.

        Named ``embed_tokens`` and not ``get_input_embeddings`` for the
        *sizing*: the graph reads ``.weight`` off this to allocate its
        input-embeds buffer. The compute-form seam is
        :meth:`get_input_embeddings` below, which exists only because a
        cross-encoder's ids carry a packed segment bit -- gathering with
        those unmasked indexes past the end of the vocabulary.
        """
        return self.embeddings.word_embeddings

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The word gather the prefill graph runs outside its captured region.

        Only the *word* lookup moves out. The position and segment terms read
        ``positions`` and the packed ids, both of which the graph refreshes
        per replay, so they stay inside the capture.
        """
        real_ids, _ = decode_token_type_ids(input_ids)
        return self.embeddings.word_embeddings(real_ids)

    def rotary(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return None, None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if _DEBUG_BATCH:
            # Off by default, and worth keeping: every encoder-only bug found
            # during this port was invisible in the output and obvious in
            # this line -- which tokenizer ran, whether positions restart per
            # request, whether cu_seqlens agrees with them.
            cu_seqlens, _ = encoder_batch_layout(ctx)
            logger.info(
                "encoder batch: n=%d cu=%s pos[:12]=%s ids[:12]=%s",
                input_ids.shape[0],
                cu_seqlens.tolist(),
                positions[:12].tolist(),
                input_ids[:12].tolist(),
            )
        hidden_states = self.embeddings(
            input_ids, positions, kwargs.get("input_embeds")
        )
        for layer in self.encoder.layers:
            # Rope is recomputed per layer rather than hoisted out of the
            # loop. Both are correct eagerly; under capture the hoisted
            # version hands every break a tensor produced by segment 0, and
            # break arguments alias pool storage on the promise that "the
            # captured segment rewrites the aliased address before the break
            # reads it" -- which only holds for the segment immediately
            # before the break. Recomputing keeps that promise. The cost is
            # eleven extra index_selects on a cached table.
            cos, sin = self.rotary(positions)
            hidden_states = layer(hidden_states, ctx, cos, sin)
        return hidden_states, None


class RobertaTransformer(BertTransformer):
    embedding_cls = RobertaEmbedding


class NewTransformer(BertTransformer):
    embedding_cls = NewEmbedding
    layer_cls = NewLayer

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, mapping, quant_config, prefix)
        dtype = self.embeddings.LayerNorm.weight.dtype
        self.rotary_emb = EncoderRotaryCache.from_config(config, dtype)

    def rotary(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb(positions)


class ModernBertTransformer(BertTransformer):
    """ModernBERT's stack: two rope tables, a final norm, pre-LN blocks."""

    embedding_cls = ModernBertEmbedding

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.mapping = mapping
        backend = getattr(config, "mm_attention_backend", None)
        self.embeddings = self.embedding_cls(
            config, prefix=add_prefix("embeddings", prefix)
        )
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [
                ModernBertLayer(
                    config,
                    mapping,
                    layer_id=idx,
                    quant_config=quant_config,
                    prefix=add_prefix(f"encoder.layers.{idx}", prefix),
                    mm_attention_backend=backend,
                )
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_size,
            eps=_modernbert_norm_eps(config),
            bias=bool(getattr(config, "norm_bias", True)),
        )
        self.layers_to_capture: list[int] = []

        dtype = self.final_norm.weight.dtype
        head_dim = config.hidden_size // config.num_attention_heads
        max_position = int(config.max_position_embeddings)
        # One table per distinct theta, shared by every layer that uses it.
        # Reusing the global theta for the local layers is invisible in the
        # output, so the thetas come off the plan rather than from a default.
        plan = modernbert_layer_plan(config)
        caches: dict[float, EncoderRotaryCache] = {}
        for _, _, theta in plan:
            if theta not in caches:
                caches[theta] = EncoderRotaryCache(head_dim, theta, max_position, dtype)
        self.rotary_caches = nn.ModuleList(caches.values())
        self._layer_rotary = [list(caches).index(theta) for _, _, theta in plan]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        hidden_states = self.embeddings(
            input_ids, positions, kwargs.get("input_embeds")
        )
        for idx, layer in enumerate(self.encoder.layers):
            # Per layer, and per layer *type*: see BertTransformer.forward for
            # why this is not hoisted out of the loop.
            cos, sin = self.rotary_caches[self._layer_rotary[idx]](positions)
            hidden_states = layer(hidden_states, ctx, cos, sin)
        return self.final_norm(hidden_states), None


class JinaLayer(NewLayer):
    """jina-embeddings-v3's block: the ungated member of the RoPE post-LN family."""

    gated_mlp = False


class JinaTransformer(NewTransformer):
    layer_cls = JinaLayer


def merge_jina_lora(
    weights: Iterable[tuple[str, torch.Tensor]],
    config: PretrainedConfig,
    task: str,
) -> list[tuple[str, torch.Tensor]]:
    """Fold one of jina-v3's five task LoRAs into the base weights.

    The checkpoint ships every LoRA-adapted parameter as a torch
    parametrization triple -- ``X.parametrizations.weight.original`` plus
    ``.0.lora_A`` and ``.0.lora_B``, each stacked over the five adaptations.
    Serving one task means picking a slice and merging it; this engine has no
    per-request adapter switching, so the task is fixed at load.

    **The task is resolved by name, not by index.** vLLM takes ``i = -1`` and
    comments that it is text-matching (``bert_with_rope.py:637``), which is
    true for this checkpoint's ``lora_adaptations`` order and would silently
    become a different adapter if that list were ever reordered. Every task
    returns a well-formed unit vector, so the wrong one is invisible.

    The two factors are stored transposed between the embedding tables and
    the linear layers -- an embedding's ``lora_A`` is ``[tasks, vocab, rank]``
    while a linear's is ``[tasks, rank, in]`` -- so the multiplication order
    is chosen from the shapes and the result is checked against the base
    tensor rather than reshaped into it.
    """
    adaptations = list(getattr(config, "lora_adaptations", None) or [])
    if task not in adaptations:
        raise ValueError(
            f"jina lora task {task!r} is not one of {adaptations}; pass a "
            'valid one via --hf-overrides \'{"lora_task": "..."}\'. '
            "Guessing would serve a well-formed embedding from the wrong "
            "adapter."
        )
    index = adaptations.index(task)
    scaling = float(config.lora_alpha) / float(config.lora_rank)
    logger.info(
        "jina lora: merging adaptation %r (index %d of %d), scaling %.4f",
        task,
        index,
        len(adaptations),
        scaling,
    )

    table = dict(weights)
    merged: list[tuple[str, torch.Tensor]] = []
    for name in list(table):
        if not name.endswith(".original"):
            continue
        stem = name[: -len(".original")]
        base = table.pop(name)
        lora_a = table.pop(f"{stem}.0.lora_A")[index].float()
        lora_b = table.pop(f"{stem}.0.lora_B")[index].float()

        if lora_a.shape[0] == base.shape[0]:
            delta = lora_a @ lora_b  # embedding layout: [rows, rank] x [rank, cols]
        else:
            delta = lora_b @ lora_a  # linear layout: [out, rank] x [rank, in]
        if delta.shape != base.shape:
            raise ValueError(
                f"jina lora merge for {stem!r} produced {tuple(delta.shape)} "
                f"against a base of {tuple(base.shape)}"
            )
        merged.append(
            (
                stem.replace(".parametrizations", ""),
                (base.float() + delta * scaling).to(base.dtype),
            )
        )

    return list(table.items()) + merged


class EncoderOnlyEmbeddingModel(BaseCausalLM):
    """Shared wiring for the encoder-only pooling models in this file.

    ``BaseCausalLM`` already does the right thing for a pooling checkpoint:
    it resolves the pooler from the HF config, and skips ``lm_head`` and the
    logits processor when one exists. Reusing it also means these models
    inherit the pooling prefill graph without a second code path.
    """

    model_cls: type[nn.Module] = BertTransformer

    # (destination substring, source substring, shard id) for the three
    # separate projections BERT ships.
    _STACKED_QKV = (
        ("attention.attn.qkv_proj", "attention.self.query", "q"),
        ("attention.attn.qkv_proj", "attention.self.key", "k"),
        ("attention.attn.qkv_proj", "attention.self.value", "v"),
    )

    # Applied in order, longest-first where they overlap.
    _RENAMES = (
        ("encoder.layer.", "encoder.layers."),
        ("attention.output.LayerNorm", "attention_LayerNorm"),
        ("attention.output.dense", "attention.attn.proj"),
        ("intermediate.dense", "intermediate"),
        ("output.LayerNorm", "output_LayerNorm"),
        ("output.dense", "output"),
    )

    # Present in the checkpoint, unused by this server.
    #   pooler.dense       -- the BertPooler dense+tanh head. Every
    #                         sentence-transformers config in scope pools
    #                         straight off the encoder output
    #                         (``modules.json`` lists Pooling, then at most
    #                         Normalize, and never Dense), so loading it
    #                         would allocate a head nothing calls.
    #   embeddings.position_ids -- a buffer, not a parameter.
    _SKIP = ("pooler.dense", "embeddings.position_ids", "position_ids")

    def resolve_pooler(self, config: PretrainedConfig):
        pooler = super().resolve_pooler(config)
        if pooler is None:
            raise ValueError(
                f"{type(self).__name__} is an embedding-only architecture but "
                "no pooling config was resolved. Start the server with "
                "--is-embedding; there is no generative path for this model."
            )
        return pooler

    def _rewrite(self, name: str) -> str:
        for source, target in self._RENAMES:
            name = name.replace(source, target)
        return name

    def _iter_weights(self, weights):
        """Hook for per-architecture weight surgery before the common loader.

        Yields ``(name, weight)``, or ``(name, weight, shard_id)`` when the
        checkpoint packs several projections into one tensor and the split
        already knows which shard each piece is.
        """
        return weights

    @staticmethod
    def _split_packed(name, weight, source, dest, shards):
        """Cut a packed tensor along dim 0 into one triple per shard."""
        pieces = weight.chunk(len(shards), dim=0)
        for shard, piece in zip(shards, pieces):
            yield name.replace(source, dest), piece, shard

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        for item in self._iter_weights(weights):
            # A packed checkpoint tensor is split upstream and arrives as a
            # triple naming its destination shard directly; everything else
            # is an ordinary (name, weight) pair routed through the rename
            # table and ``_STACKED_QKV``.
            forced_shard = None
            if len(item) == 3:
                raw_name, loaded_weight, forced_shard = item
            else:
                raw_name, loaded_weight = item
            if any(skip in raw_name for skip in self._SKIP):
                continue

            name = self._rewrite(raw_name)

            shard_id = forced_shard
            if shard_id is None:
                for target, source, sid in self._STACKED_QKV:
                    if source in name:
                        name = name.replace(source, target)
                        shard_id = sid
                        break

            # Resolve the prefix last, after every rename has run: the
            # checkpoints are flat (``embeddings.``/``encoder.``) and this
            # tree nests the transformer under ``model.`` -- but not
            # everything, since a cross-encoder's classifier head hangs off
            # the outer model beside ``model`` rather than inside it. Trying
            # the name as written first lets the rename tables name either
            # without a second table saying which.
            if name not in params_dict and f"model.{name}" in params_dict:
                name = f"model.{name}"

            param = params_dict.get(name)
            if param is None:
                raise KeyError(
                    f"{type(self).__name__}: checkpoint weight {raw_name!r} "
                    f"maps to {name!r}, which this model does not define. "
                    "Fix the rename table rather than skipping it -- a "
                    "silently dropped encoder weight still produces a "
                    "well-formed embedding."
                )
            if shard_id is not None:
                param.weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight)
                else:
                    param.data.copy_(loaded_weight.to(param.dtype))
            loaded.add(name)

        missing = sorted(set(params_dict) - loaded)
        if missing:
            raise ValueError(
                f"{type(self).__name__}: {len(missing)} parameters were never "
                f"filled by the checkpoint, e.g. {missing[:5]}. An unfilled "
                "encoder parameter is uninitialised memory, which pools into "
                "a unit-norm vector that looks valid."
            )
        return loaded


class BertModel(EncoderOnlyEmbeddingModel):
    """``BertModel`` -- bert-base, BAAI/bge-*-en-v1.5, thenlper/gte-*."""

    model_cls = BertTransformer


# jina-embeddings-v3's rename table. Kept beside RobertaModel rather than on
# its own class because the checkpoint declares ``XLMRobertaModel`` -- the
# same architecture string as multilingual-e5 -- and only
# ``position_embedding_type`` tells them apart.
_JINA_RENAMES = (
    ("roberta.", ""),
    ("emb_ln", "embeddings.LayerNorm"),
    ("mixer.out_proj", "attention.attn.proj"),
    ("norm1", "attn_ln"),
    ("norm2", "mlp_ln"),
    ("mlp.fc1", "fc1"),
    ("mlp.fc2", "down_proj"),
)

_JINA_SKIP = ("pooler.", "position_ids")


class RobertaModel(EncoderOnlyEmbeddingModel):
    """``RobertaModel`` / ``XLMRobertaModel``.

    Two unrelated models answer to this architecture string, and the config
    is the only thing that separates them: multilingual-e5 and bge-m3 use
    learned absolute positions, while jina-embeddings-v3 sets
    ``position_embedding_type: "rotary"`` and is a different block entirely
    (RoPE, ungated MLP, LoRA-parametrised weights). vLLM branches at the same
    place (``roberta.py:141 _build_model``).
    """

    model_cls = RobertaTransformer

    def __init__(self, config: PretrainedConfig, *args, **kwargs) -> None:
        self.is_jina = getattr(config, "position_embedding_type", "absolute") in (
            "rotary",
            "rope",
        )
        if self.is_jina:
            # Instance attributes shadow the class tables the base loader
            # reads, so the shared loader needs no branch of its own.
            self.model_cls = JinaTransformer
            self._RENAMES = _JINA_RENAMES
            self._SKIP = _JINA_SKIP
            self._STACKED_QKV = ()
            self._lora_task = str(getattr(config, "lora_task", "text-matching"))
        super().__init__(config, *args, **kwargs)

    def _iter_weights(self, weights):
        if not self.is_jina:
            return super()._iter_weights(weights)
        return self._iter_jina_weights(weights)

    def _iter_jina_weights(self, weights):
        for name, weight in merge_jina_lora(weights, self.config, self._lora_task):
            if "mixer.Wqkv" in name:
                yield from self._split_packed(
                    name,
                    weight,
                    "mixer.Wqkv",
                    "attention.attn.qkv_proj",
                    ("q", "k", "v"),
                )
            else:
                yield name, weight


class XLMRobertaModel(RobertaModel):
    pass


class NewModel(EncoderOnlyEmbeddingModel):
    """``NewModel`` -- Alibaba-NLP/gte-*-en-v1.5 (RoPE + GeGLU)."""

    model_cls = NewTransformer

    # Both packed tensors are cut by ``_iter_weights`` below, which names
    # their destination shards directly, so no string table is needed here.
    _STACKED_QKV = ()

    _RENAMES = (
        ("encoder.layer.", "encoder.layers."),
        ("attention.o_proj", "attention.attn.proj"),
        ("mlp.down_proj", "down_proj"),
    )

    _SKIP = ("classifier", "position_ids")

    def _iter_weights(self, weights):
        """Split this checkpoint's two packed tensors into shards.

        ``attention.qkv_proj`` and ``mlp.up_gate_proj`` both ship fused. At
        TP=1 either could be copied whole; splitting serves TP>1 too, where a
        whole-tensor copy would silently give every rank the full weight.

        ``up_gate_proj`` is packed ``[up; gate]``, so shard 0 is up and shard
        1 is gate -- matching the reference's
        ``up_states, gate = split(up_gate)``.
        """
        for name, weight in weights:
            if "attention.qkv_proj" in name:
                yield from self._split_packed(
                    name,
                    weight,
                    "attention.qkv_proj",
                    "attention.attn.qkv_proj",
                    ("q", "k", "v"),
                )
            elif "mlp.up_gate_proj" in name:
                yield from self._split_packed(
                    name, weight, "mlp.up_gate_proj", "up_gate_proj", (0, 1)
                )
            else:
                yield name, weight


class GteModel(NewModel):
    """``GteModel`` -- Snowflake/snowflake-arctic-embed-*-v2.0."""


class NomicBertModel(NewModel):
    """``NomicBertModel`` -- nomic-ai/nomic-embed-text-v1.5, nomic-bert-2048.

    Structurally the same block as GTE-v1.5 (post-LN, RoPE, gated MLP), with
    three differences that all live in the config or the names:

      * the gate and up projections ship *unfused* as ``mlp.fc11`` /
        ``mlp.fc12``, so there is no packed tensor to split;
      * ``activation_function: "swiglu"`` means silu on the gate, where GTE
        says ``"gelu"`` and means GeGLU (vLLM rewrites both fields --
        ``model_executor/models/config.py:660`` for this one);
      * the projections carry no bias while the norms do, the opposite of
        GTE. Read off the config rather than assumed; an absent bias that
        this model allocates anyway would be caught by the loader's
        unfilled-parameter check, but a *present* one silently dropped would
        not.
    """

    _STACKED_QKV = ()

    _RENAMES = (
        ("emb_ln", "embeddings.LayerNorm"),
        ("attn.out_proj", "attention.attn.proj"),
        ("norm1", "attn_ln"),
        ("norm2", "mlp_ln"),
        ("mlp.fc2", "down_proj"),
    )

    _SKIP = ("position_ids",)

    def _iter_weights(self, weights):
        """Split the packed QKV; route the already-separate MLP halves.

        ``fc11`` is up and ``fc12`` is gate -- the mapping vLLM's
        ``NomicBertModel`` uses (``bert_with_rope.py:538``). Getting the two
        the wrong way round swaps which half the activation lands on and
        still returns a well-formed unit vector.
        """
        for name, weight in weights:
            if "attn.Wqkv" in name:
                yield from self._split_packed(
                    name,
                    weight,
                    "attn.Wqkv",
                    "attention.attn.qkv_proj",
                    ("q", "k", "v"),
                )
            elif "mlp.fc11" in name:
                yield name.replace("mlp.fc11", "up_gate_proj"), weight, 0
            elif "mlp.fc12" in name:
                yield name.replace("mlp.fc12", "up_gate_proj"), weight, 1
            else:
                yield name, weight


class ModernBertModel(EncoderOnlyEmbeddingModel):
    """``ModernBertModel`` -- answerdotai/ModernBERT-*, nomic-ai/modernbert-embed-*."""

    model_cls = ModernBertTransformer

    _STACKED_QKV = ()

    _RENAMES = (
        # ``attn.Wo`` first: the bare ``layers.`` rewrite must not reach it,
        # and the MLP has its own ``Wo`` that has to survive untouched.
        ("attn.Wo", "attention.attn.proj"),
        ("layers.", "encoder.layers."),
    )

    # ``head.*`` and ``decoder.*`` are the masked-LM head some ModernBERT
    # checkpoints ship; an embedding server never calls it.
    _SKIP = ("position_ids", "head.", "decoder.", "classifier")

    def _iter_weights(self, weights):
        for name, weight in weights:
            if "attn.Wqkv" in name:
                yield from self._split_packed(
                    name,
                    weight,
                    "attn.Wqkv",
                    "attention.attn.qkv_proj",
                    ("q", "k", "v"),
                )
            else:
                yield name, weight


class SequenceClassifierHead(nn.Module):
    """CLS -> dense -> tanh -> linear(num_labels).

    All three cross-encoder families in this file share this shape and differ
    only in where the checkpoint puts it: BERT splits it across
    ``bert.pooler.dense`` and a top-level ``classifier``, RoBERTa keeps both
    halves under ``classifier`` (``dense`` / ``out_proj``), and GTE splits it
    like BERT under ``new.pooler.dense``. The rename tables reconcile them.
    """

    def __init__(self, config: PretrainedConfig, num_labels: int) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.out_proj(torch.tanh(self.dense(pooled)))


def resolve_num_labels(config: PretrainedConfig) -> int:
    """``num_labels``, which these configs mostly decline to state.

    Every reranker checkpoint here leaves ``num_labels`` null and carries a
    one-entry ``id2label`` instead; transformers derives it the same way.
    Defaulting to 2 (the transformers default for a bare config) would build
    a head twice the checkpoint's width and fail the loader, which is the
    good case -- so this derives it rather than guessing.
    """
    declared = getattr(config, "num_labels", None)
    if declared:
        return int(declared)
    id2label = getattr(config, "id2label", None)
    if id2label:
        return len(id2label)
    raise ValueError(
        f"{type(config).__name__} states neither num_labels nor id2label; a "
        "classifier head cannot be sized from it."
    )


class CrossEncoderModel(EncoderOnlyEmbeddingModel):
    """A reranker: score one (query, document) pair, not embed one text.

    The output is still one row per request -- ``num_labels`` wide instead of
    ``hidden`` -- so it rides the existing pooled-embedding transport
    unchanged. What is genuinely different is the *input*: a pair, whose two
    halves BERT distinguishes with segment ids. Those ride packed into the
    token ids (see ``encode_token_type_ids``).

    Scores are the raw logits, matching what
    ``AutoModelForSequenceClassification`` returns and what every one of these
    checkpoints' own README compares against. bge-reranker's README says to
    apply sigmoid *if* a 0-1 range is wanted; doing that here would silently
    change the numbers a caller compares against the reference.
    """

    # Cross-encoders pool CLS and never normalize -- a normalized score is
    # not a score. Declared here rather than read from a 1_Pooling directory,
    # which none of these checkpoints ship.
    pooling_type = PoolingType.CLS

    def __init__(self, config: PretrainedConfig, *args, **kwargs) -> None:
        self.num_labels = resolve_num_labels(config)
        super().__init__(config, *args, **kwargs)
        self.classifier = SequenceClassifierHead(config, self.num_labels)

    def resolve_pooler(self, config: PretrainedConfig):
        return Pooler(pooling_type=self.pooling_type, normalize=False)

    def pool(self, ctx: ForwardContext, hidden_states: torch.Tensor):
        pooled = super().pool(ctx, hidden_states)
        scores = self.classifier(
            pooled.embeddings.to(self.classifier.dense.weight.dtype)
        )
        return PoolerOutput(embeddings=scores)


class BertForSequenceClassification(CrossEncoderModel):
    """``cross-encoder/ms-marco-*``, and other BERT rerankers."""

    model_cls = BertTransformer

    _RENAMES = (
        ("bert.encoder.layer.", "encoder.layers."),
        ("bert.embeddings", "embeddings"),
        ("attention.output.LayerNorm", "attention_LayerNorm"),
        ("attention.output.dense", "attention.attn.proj"),
        ("intermediate.dense", "intermediate"),
        ("output.LayerNorm", "output_LayerNorm"),
        ("output.dense", "output"),
        # The head: BERT keeps its two halves apart, we keep them together.
        ("bert.pooler.dense", "classifier.dense"),
        ("classifier.weight", "classifier.out_proj.weight"),
        ("classifier.bias", "classifier.out_proj.bias"),
    )

    _SKIP = ("position_ids",)


class RobertaForSequenceClassification(CrossEncoderModel):
    """``BAAI/bge-reranker-base``, ``bge-reranker-v2-m3``."""

    model_cls = RobertaTransformer

    _RENAMES = (
        ("roberta.encoder.layer.", "encoder.layers."),
        ("roberta.embeddings", "embeddings"),
        ("attention.output.LayerNorm", "attention_LayerNorm"),
        ("attention.output.dense", "attention.attn.proj"),
        ("intermediate.dense", "intermediate"),
        ("output.LayerNorm", "output_LayerNorm"),
        ("output.dense", "output"),
        # RoBERTa already keeps dense and out_proj under `classifier`.
    )

    _SKIP = ("position_ids",)


class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    pass


class NewForSequenceClassification(CrossEncoderModel):
    """``Alibaba-NLP/gte-multilingual-reranker-base``."""

    model_cls = NewTransformer

    _STACKED_QKV = ()

    _RENAMES = (
        ("new.encoder.layer.", "encoder.layers."),
        ("new.embeddings", "embeddings"),
        ("attention.o_proj", "attention.attn.proj"),
        ("mlp.down_proj", "down_proj"),
        ("new.pooler.dense", "classifier.dense"),
        ("classifier.weight", "classifier.out_proj.weight"),
        ("classifier.bias", "classifier.out_proj.bias"),
    )

    _SKIP = ("position_ids",)

    def _iter_weights(self, weights):
        for name, weight in weights:
            if "attention.qkv_proj" in name:
                yield from self._split_packed(
                    name,
                    weight,
                    "attention.qkv_proj",
                    "attention.attn.qkv_proj",
                    ("q", "k", "v"),
                )
            elif "mlp.up_gate_proj" in name:
                yield from self._split_packed(
                    name, weight, "mlp.up_gate_proj", "up_gate_proj", (0, 1)
                )
            else:
                yield name, weight


EntryClass = [
    BertModel,
    RobertaModel,
    XLMRobertaModel,
    NewModel,
    GteModel,
    NomicBertModel,
    ModernBertModel,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    XLMRobertaForSequenceClassification,
    NewForSequenceClassification,
]
