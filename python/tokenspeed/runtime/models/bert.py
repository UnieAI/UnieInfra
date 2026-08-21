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

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.mm_encoder_attention import (
    MultimodalEncoderAttention,
)
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.utils import add_prefix, get_colorful_logger

import os

# Set TOKENSPEED_ENCODER_DEBUG=1 to log the batch layout each forward.
_DEBUG_BATCH = bool(os.environ.get("TOKENSPEED_ENCODER_DEBUG"))

logger = get_colorful_logger(__name__)

__all__ = [
    "BertModel",
    "RobertaModel",
    "XLMRobertaModel",
    "NewModel",
    "GteModel",
]


def encoder_batch_layout(ctx: ForwardContext) -> tuple[torch.Tensor, int]:
    """Return ``(cu_seqlens, max_seqlen_bound)`` for this forward.

    Taken from ``ctx.gather_ids`` rather than from the attention backend's
    extend metadata. Two reasons: an encoder-only model has no attention
    layer to drive that backend, and ``gather_ids`` is already the tensor
    ``BaseCausalLM.pool`` reduces over, so attention and pooling cannot
    disagree about where a request's rows are.

    ``gather_ids`` on an extend forward is ``cumsum(extend_lens) - 1``, so
    ``cat([0, gather_ids + 1])`` *is* the cu_seqlens vector.

    The second return value is an upper bound on the longest sequence, not
    the exact maximum: deriving the exact value needs ``.item()``, which is a
    device sync and is illegal inside a captured CUDA graph (the same
    constraint Whisper's decoder hit). ``input_num_tokens`` is the total
    across the batch, so it bounds any single request. Varlen kernels use it
    only to size the grid and mask by the per-request lengths, so an
    over-estimate costs empty blocks, never wrong rows.
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
    ) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = embeddings + self.position_embeddings(
            positions + self.position_offset
        )
        if self.token_type_embeddings is not None:
            # Single-sequence embedding: segment 0 for every token. A
            # cross-encoder pair would need real token_type_ids plumbed from
            # the tokenizer; this server has no endpoint that produces them.
            embeddings = embeddings + self.token_type_embeddings.weight[0]
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
    ) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        if self.token_type_embeddings is not None:
            embeddings = embeddings + self.token_type_embeddings.weight[0]
        return self.LayerNorm(embeddings)


class NTKScalingRotaryCache(nn.Module):
    """RoPE cos/sin for GTE-v1.5, including its fixed NTK rescaling.

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

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype) -> None:
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = float(getattr(config, "rope_theta", 10000.0))
        max_position = int(config.max_position_embeddings)

        scaling = getattr(config, "rope_scaling", None) or {}
        scaling_type = scaling.get("type") or scaling.get("rope_type")
        factor = float(scaling.get("factor", 1.0)) if scaling else 1.0
        if scaling and scaling_type != "ntk":
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
    ) -> None:
        super().__init__()
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cos, sin = (None, None) if rotary is None else rotary
        return self.attn(
            hidden_states.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_pos_emb_cos=cos,
            rotary_pos_emb_sin=sin,
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
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # ``rotary`` is always None here -- BERT and RoBERTa carry position
        # information in the embedding table, not in the attention scores.
        # The parameter exists so the encoder stack calls every layer type
        # with one signature.
        attended = self.attention(hidden_states, cu_seqlens, max_seqlen)
        hidden_states = self.attention_LayerNorm(hidden_states + attended)

        intermediate, _ = self.intermediate(hidden_states)
        intermediate = self.act_fn(intermediate)
        projected, _ = self.output(intermediate)
        return self.output_LayerNorm(hidden_states + projected)


class NewLayer(nn.Module):
    """GTE-v1.5 block: RoPE attention, GeGLU MLP, post-LN."""

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
        self.attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # up and gate are one checkpoint tensor (``mlp.up_gate_proj``, no
        # bias) split at load time; keeping them as one column-parallel
        # projection preserves that layout and keeps the split a view.
        self.up_gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_gate_proj", prefix),
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=attn.tp_rank,
            tp_size=attn.tp_size,
            tp_group=attn.tp_group,
            reduce_results=True,
        )
        self.mlp_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate_size = config.intermediate_size
        # Exact gelu (erf), not the tanh approximation: the reference calls
        # ``ACT2FN["gelu"]`` on the gate half.
        self.act_fn = _resolve_bert_activation("gelu")

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        attended = self.attention(hidden_states, cu_seqlens, max_seqlen, rotary=rotary)
        hidden_states = self.attn_ln(hidden_states + attended)

        up_gate, _ = self.up_gate_proj(hidden_states)
        # Checkpoint order is [up; gate], the opposite of this tree's fused
        # gate_up convention.
        up_states, gate = up_gate.split(up_gate.shape[-1] // 2, dim=-1)
        gated, _ = self.down_proj(self.act_fn(gate) * up_states)
        return self.mlp_ln(hidden_states + gated)


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

    def rotary(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        cu_seqlens, max_seqlen = encoder_batch_layout(ctx)
        if _DEBUG_BATCH:
            # Off by default, and worth keeping: every encoder-only bug found
            # during this port was invisible in the output and obvious in
            # this line -- which tokenizer ran, whether positions restart per
            # request, whether cu_seqlens agrees with them.
            logger.info(
                "encoder batch: n=%d cu=%s pos[:12]=%s ids[:12]=%s",
                input_ids.shape[0],
                cu_seqlens.tolist(),
                positions[:12].tolist(),
                input_ids[:12].tolist(),
            )
        hidden_states = self.embeddings(input_ids, positions)
        rotary = self.rotary(positions)
        for layer in self.encoder.layers:
            hidden_states = layer(
                hidden_states, positions, cu_seqlens, max_seqlen, rotary
            )
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
        self.rotary_emb = NTKScalingRotaryCache(config, dtype)

    def rotary(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb(positions)


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

    def _iter_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Hook for per-architecture weight surgery before the common loader."""
        return weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        for raw_name, loaded_weight in self._iter_weights(weights):
            if any(skip in raw_name for skip in self._SKIP):
                continue

            name = self._rewrite(raw_name)
            # The checkpoints are flat (``embeddings.``/``encoder.``); this
            # tree nests the transformer under ``model.``.
            if not name.startswith("model."):
                name = "model." + name

            shard_id = None
            for target, source, sid in self._STACKED_QKV:
                if source in name:
                    name = name.replace(source, target)
                    shard_id = sid
                    break

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


class RobertaModel(EncoderOnlyEmbeddingModel):
    """``RobertaModel`` / ``XLMRobertaModel`` -- multilingual-e5-*, bge-m3."""

    model_cls = RobertaTransformer


class XLMRobertaModel(RobertaModel):
    pass


class NewModel(EncoderOnlyEmbeddingModel):
    """``NewModel`` -- Alibaba-NLP/gte-*-en-v1.5 (RoPE + GeGLU)."""

    model_cls = NewTransformer

    # Same fused destination as BERT; ``_iter_weights`` below cuts the
    # checkpoint's already-packed tensor into the three shard names first.
    _STACKED_QKV = (
        ("attention.attn.qkv_proj", "attention.self.query", "q"),
        ("attention.attn.qkv_proj", "attention.self.key", "k"),
        ("attention.attn.qkv_proj", "attention.self.value", "v"),
    )

    _RENAMES = (
        ("encoder.layer.", "encoder.layers."),
        ("attention.o_proj", "attention.attn.proj"),
        ("mlp.up_gate_proj", "up_gate_proj"),
        ("mlp.down_proj", "down_proj"),
    )

    _SKIP = ("classifier", "position_ids")

    def _iter_weights(self, weights):
        """Split the packed QKV so the fused projection loads shard by shard.

        The checkpoint already ships ``attention.qkv_proj`` fused, and at
        TP=1 it could be copied whole. Splitting it anyway means the same
        path serves TP>1, where the fused tensor has to be cut per rank and a
        whole-tensor copy would silently give every rank the full weight.
        """
        for name, weight in weights:
            if "attention.qkv_proj" not in name:
                yield name, weight
                continue
            q, k, v = weight.chunk(3, dim=0)
            yield name.replace("attention.qkv_proj", "attention.self.query"), q
            yield name.replace("attention.qkv_proj", "attention.self.key"), k
            yield name.replace("attention.qkv_proj", "attention.self.value"), v


class GteModel(NewModel):
    """``GteModel`` -- Snowflake/snowflake-arctic-embed-*-v2.0."""


EntryClass = [
    BertModel,
    RobertaModel,
    XLMRobertaModel,
    NewModel,
    GteModel,
]
