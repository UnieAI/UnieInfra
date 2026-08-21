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

"""Whisper (encoder-decoder ASR).

Structure, and why it differs from the other ASR model in this tree:

``qwen3_asr.py`` is decoder-only with an audio prefix -- the audio tower's
output is spliced into the LM's input embeddings and the rest is ordinary
causal decode. Whisper is a real encoder-decoder: the decoder cross-attends
to a fixed encoder output that is computed once and read by every decode
step. That fixed KV is the whole design question, and the four reference
engines disagree about it (``unieinfra/xref/15-encoder-embedding.md`` §3.3):
vLLM gives it its own KV spec, manager and cache group; TRT-LLM gives it a
separate resource manager with its own slice of the KV budget; SGLang folds
it into the ordinary paged pool by offsetting the layer id so cross-attention
looks like ``encoder_layers + decoder_layers + i``.

We are on vLLM/TRT-LLM's side -- cross KV lives outside the paged arena --
but implemented as a plain preallocated per-slot buffer rather than a fourth
cache family, because cross KV needs nothing the arena provides:

  * it never grows (one write of ``encoder_len`` rows per request),
  * it is never prefix-matched (encoder output is unique per audio, which is
    why vLLM's CrossAttentionManager raises on ``find_longest_cache_hit``),
  * it is never evicted independently of its request.

The cost is bounded and small: ``decoder_layers x encoder_len x d_model x 2``
per slot -- 30 MB per request on large-v3-turbo (4 decoder layers, 1500
encoder frames), against ~1.2 GB for the same request's share of a paged
group sized for it. Keeping it out of the arena also keeps this port off the
shared C++ scheduler, which several sessions share.

Cross-attention itself needs no new kernel: ``flash_attn_varlen_func``
already takes ``cu_seqlens_q`` and ``cu_seqlens_k`` separately (the encoder
wrappers in ``mm_encoder_attention.py`` merely pass the same tensor twice),
which is exactly the "q from the decoder, kv from the encoder" shape.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.mm_encoder_attention import (
    MultimodalEncoderAttention,
)
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.utils import add_prefix


class WhisperEncoderLayer(nn.Module):
    """One encoder block: bidirectional attention, no KV cache.

    Reuses ``MultimodalEncoderAttention`` -- the same non-causal, pool-free
    layer the vision and audio towers use. Whisper's encoder is exactly that
    shape, so it needs no encoder-specific attention code.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        self.self_attn = MultimodalEncoderAttention(
            embed_dim=embed_dim,
            num_heads=config.encoder_attention_heads,
            mapping=mapping,
            # Whisper ships no bias for k_proj. The fused QKV bias is
            # allocated anyway and the k slice is zero-filled at load time
            # (see load_weights) -- a zero bias is exactly "no bias".
            qkv_bias=True,
            proj_bias=True,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            mm_attention_backend=mm_attention_backend,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]

        vision = mapping.vision
        self.fc1 = ColumnParallelLinear(
            embed_dim,
            config.encoder_ffn_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            reduce_results=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.self_attn_layer_norm(hidden_states)
        attended = self.self_attn(
            normed.unsqueeze(0), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        ).squeeze(0)
        hidden_states = residual + attended

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return residual + hidden_states


class WhisperEncoder(nn.Module):
    """Log-mel frames -> a fixed-length sequence of encoder states.

    Runs once per request. The output is what every decode step cross-attends
    to; nothing here touches the paged KV pool.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.conv1 = nn.Conv1d(config.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        # Whisper's encoder positions are fixed sinusoids shipped as weights,
        # not a learned table; registered as a buffer so they load like one
        # and never appear in the optimizer's eyes.
        self.register_buffer(
            "embed_positions",
            torch.zeros(config.max_source_positions, embed_dim),
            persistent=True,
        )
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(
                    config,
                    mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                    mm_attention_backend=mm_attention_backend,
                )
                for i in range(config.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.conv1.weight.device

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """``input_features``: [batch, num_mel_bins, frames] -> [batch*T, d]."""
        features = input_features.to(device=self.device, dtype=self.dtype)
        hidden = F.gelu(self.conv1(features))
        hidden = F.gelu(self.conv2(hidden))
        hidden = hidden.permute(0, 2, 1)

        batch, seq_len, _ = hidden.shape
        if seq_len > self.embed_positions.shape[0]:
            raise ValueError(
                f"Whisper encoder got {seq_len} frames but the position table "
                f"holds {self.embed_positions.shape[0]}; the feature extractor "
                "must pad or trim to the model's window."
            )
        hidden = hidden + self.embed_positions[:seq_len].to(hidden.dtype)

        # One packed varlen sequence per batch row, all the same length: the
        # encoder window is fixed, so cu_seqlens is a plain arange.
        flat = hidden.reshape(batch * seq_len, -1)
        cu_seqlens = torch.arange(
            0,
            (batch + 1) * seq_len,
            seq_len,
            device=flat.device,
            dtype=torch.int32,
        )
        for layer in self.layers:
            flat = layer(flat, cu_seqlens=cu_seqlens, max_seqlen=seq_len)
        return self.layer_norm(flat)


class CrossAttentionCache:
    """Per-request encoder K/V for the decoder's cross-attention.

    Deliberately not a paged-arena cache group. Cross KV is written once,
    never grows, is never prefix-matched and is never evicted apart from its
    request, so none of the arena's machinery applies to it (see the module
    docstring, and §3.3 of the xref).

    Layout is ``[slot, layer, encoder_len, kv_heads, head_dim]`` so one
    ``index_select`` per forward gathers *every* layer's KV for the batch --
    the gather is unavoidable on sm_120, where no flash-attn path reads KV by
    page or index (measured; see ``15-encoder-embedding.md`` §6.1), so the
    thing to minimise is the number of gathers, not their existence.

    Slots are indexed by ``req_pool_index``, which the scheduler already
    bounds at ``per_rank_max_batch + 1`` (``event_loop.py:369``). The buffer
    grows to fit the largest index seen rather than taking that bound as a
    constructor argument: the model is built from the HF config alone and
    does not see the scheduler's sizing.
    """

    def __init__(
        self,
        num_layers: int,
        encoder_len: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_layers = num_layers
        self.encoder_len = encoder_len
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def bytes_per_slot(self) -> int:
        return (
            2  # k and v
            * self.num_layers
            * self.encoder_len
            * self.kv_heads
            * self.head_dim
            * torch.tensor([], dtype=self.dtype).element_size()
        )

    def _ensure(self, num_slots: int) -> None:
        if self.k is not None and self.k.shape[0] >= num_slots:
            return
        shape = (
            num_slots,
            self.num_layers,
            self.encoder_len,
            self.kv_heads,
            self.head_dim,
        )
        new_k = torch.zeros(shape, dtype=self.dtype, device=self.device)
        new_v = torch.zeros(shape, dtype=self.dtype, device=self.device)
        if self.k is not None:
            new_k[: self.k.shape[0]].copy_(self.k)
            new_v[: self.v.shape[0]].copy_(self.v)
        self.k, self.v = new_k, new_v

    def write(
        self,
        slot: int,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Store one request's encoder K/V for one decoder layer."""
        self._ensure(slot + 1)
        if k.shape[0] != self.encoder_len:
            raise ValueError(
                f"cross-attention KV for slot {slot} has {k.shape[0]} rows but "
                f"the cache was built for {self.encoder_len}; Whisper's encoder "
                "window is fixed, so a different length means the feature "
                "extractor did not pad or trim to it."
            )
        self.k[slot, layer_id].copy_(k)
        self.v[slot, layer_id].copy_(v)

    def gather(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch-ordered, contiguous K/V for every layer: [bs, layers, ...]."""
        if self.k is None:
            raise RuntimeError(
                "cross-attention KV was read before any encoder output was "
                "written; a decode step reached the decoder without its "
                "request having been prefilled"
            )
        self._ensure(int(slots.max().item()) + 1)
        index = slots.to(device=self.k.device, dtype=torch.long)
        return self.k.index_select(0, index), self.v.index_select(0, index)


class WhisperDecoderSelfAttention(nn.Module):
    """Causal self-attention over the decoded tokens; ordinary paged KV."""

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        total_heads = config.decoder_attention_heads
        self.tp_size = mapping.attn.tp_size
        self.num_heads = total_heads // self.tp_size
        self.head_dim = embed_dim // total_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            total_heads,
            total_heads,
            # Whisper's k_proj carries no bias; the fused bias exists and its
            # k slice is zero-filled at load time.
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
            reduce_results=False,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # The reference implementations pre-multiply q by the scale and build
        # the attention with scaling=1.0. Passing the scale instead is the
        # same arithmetic with one fewer elementwise pass.
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attended = self.attn(q, k, v, ctx, out_cache_loc)
        output, _ = self.out_proj(attended)
        return output


class WhisperCrossAttention(nn.Module):
    """Decoder attends to the fixed encoder output.

    Two shapes, one kernel. At prefill the encoder states are in hand, so
    K/V are projected, stored, and used directly. At decode the states are
    long gone and K/V come back from :class:`CrossAttentionCache`. Both go
    through ``flash_attn_varlen_func`` with *separate* q and k cu_seqlens --
    the API already supports that; the encoder wrappers in
    ``mm_encoder_attention.py`` simply pass one tensor twice because
    self-attention has q_len == kv_len.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        total_heads = config.decoder_attention_heads
        self.layer_id = layer_id
        self.tp_size = mapping.attn.tp_size
        self.num_heads = total_heads // self.tp_size
        self.head_dim = embed_dim // total_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # Whisper ships k_proj without a bias. Kept as a separate projection
        # rather than fused with v: the two are only ever applied together at
        # prefill, so fusing would buy nothing and would need the same
        # zero-bias patch as the self-attention path.
        self.k_proj = ColumnParallelLinear(
            embed_dim,
            embed_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.v_proj = ColumnParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
            reduce_results=False,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )

    def project_encoder_kv(
        self, encoder_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project one request's encoder output into this layer's K/V."""
        k, _ = self.k_proj(encoder_states)
        v, _ = self.v_proj(encoder_states)
        return (
            k.view(-1, self.num_heads, self.head_dim),
            v.view(-1, self.num_heads, self.head_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        from tokenspeed.runtime.layers.attention.mm_encoder_attention import (
            cross_attn_varlen,
        )

        q, _ = self.q_proj(hidden_states)
        q = q.view(-1, self.num_heads, self.head_dim)
        attended = cross_attn_varlen(
            q,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scaling,
        )
        output, _ = self.out_proj(attended.reshape(-1, self.num_heads * self.head_dim))
        return output


class WhisperDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        self.self_attn = WhisperDecoderSelfAttention(
            config,
            mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.encoder_attn = WhisperCrossAttention(
            config,
            mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("encoder_attn", prefix),
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = ColumnParallelLinear(
            embed_dim,
            config.decoder_ffn_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.fc2 = RowParallelLinear(
            config.decoder_ffn_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            reduce_results=True,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        cross_key: torch.Tensor,
        cross_value: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, ctx, out_cache_loc)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states,
            cross_key,
            cross_value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return residual + hidden_states


class WhisperDecoder(nn.Module):
    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            embed_dim,
            prefix=add_prefix("embed_tokens", prefix),
        )
        # Learned, unlike the encoder's fixed sinusoids.
        self.embed_positions = nn.Embedding(config.max_target_positions, embed_dim)
        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayer(
                    config,
                    mapping,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        cross_key: torch.Tensor,
        cross_value: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                ctx,
                out_cache_loc,
                cross_key[:, layer_id].reshape(-1, *cross_key.shape[3:]),
                cross_value[:, layer_id].reshape(-1, *cross_value.shape[3:]),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
        return self.layer_norm(hidden_states)


class WhisperForConditionalGeneration(nn.Module):
    """Whisper as this engine serves it: one encoder pass, then ordinary decode.

    The request carries its audio the same way a multimodal request carries an
    image. What is different from ``qwen3_asr.py`` is where the encoder output
    goes: not into the decoder's input embeddings, but into every decoder
    layer's cross-attention K/V, written once at prefill and read by every
    step after.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.encoder = WhisperEncoder(
            config,
            mapping,
            quant_config=None,  # the tower stays in bf16/fp16
            prefix=add_prefix("model.encoder", prefix),
            mm_attention_backend=mm_attention_backend,
        )
        self.decoder = WhisperDecoder(
            config,
            mapping,
            quant_config=quant_config,
            prefix=add_prefix("model.decoder", prefix),
        )
        self.proj_out = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            bias=False,
            prefix=add_prefix("proj_out", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        self.cross_cache = CrossAttentionCache(
            num_layers=config.decoder_layers,
            encoder_len=config.max_source_positions,
            kv_heads=config.decoder_attention_heads // mapping.attn.tp_size,
            head_dim=config.d_model // config.decoder_attention_heads,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )

    def encode_audio(self, input_features: torch.Tensor, slots: list[int]) -> None:
        """Run the encoder once and fill each request's cross-attention K/V."""
        encoder_states = self.encoder(input_features)
        encoder_len = self.config.max_source_positions
        if encoder_states.shape[0] != len(slots) * encoder_len:
            raise ValueError(
                f"encoder produced {encoder_states.shape[0]} rows for "
                f"{len(slots)} request(s); expected {len(slots) * encoder_len}"
            )
        for position, slot in enumerate(slots):
            span = encoder_states[position * encoder_len : (position + 1) * encoder_len]
            for layer_id, layer in enumerate(self.decoder.layers):
                key, value = layer.encoder_attn.project_encoder_kv(span)
                self.cross_cache.write(slot, layer_id, key, value)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    #
    # Three things the official layout forces:
    #
    #  * ``k_proj`` carries no bias. The fused QKV bias is allocated for all
    #    three shards, so the k slice must be an explicit zero rather than
    #    uninitialised memory -- a zero bias *is* no bias.
    #  * ``proj_out`` is absent: Whisper ties it to the decoder token
    #    embedding. Tied here from the checkpoint's actual contents rather
    #    than from a config flag, because the checkpoint is the thing that
    #    lacks the tensor.
    #  * the encoder's position table is a fixed sinusoid buffer, not a
    #    learned ``nn.Embedding``.

    # ``model.encoder.`` / ``model.decoder.`` in the checkpoint against
    # ``encoder.`` / ``decoder.`` here: this class is the whole model and
    # does not repeat the HF wrapper level.
    _PREFIX_REWRITES = (
        ("model.encoder.", "encoder."),
        ("model.decoder.", "decoder."),
    )
    _STACKED_QKV = (
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    )

    @classmethod
    def _rewrite(cls, name: str) -> str:
        for source, target in cls._PREFIX_REWRITES:
            if name.startswith(source):
                name = target + name[len(source) :]
                break
        # The encoder tower calls its output projection ``proj``; both stacks
        # are ``out_proj`` in the checkpoint.
        if name.startswith("encoder.layers.") and ".self_attn.out_proj." in name:
            name = name.replace(".self_attn.out_proj.", ".self_attn.proj.")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        for param_name, param in params_dict.items():
            if param_name.endswith("qkv_proj.bias"):
                param.data.zero_()

        loaded: set[str] = set()
        for raw_name, loaded_weight in weights:
            name = self._rewrite(raw_name)

            if name == "encoder.embed_positions.weight":
                self.encoder.embed_positions.copy_(
                    loaded_weight.to(self.encoder.embed_positions.dtype)
                )
                loaded.add(name)
                continue

            fused_target = None
            if ".self_attn." in name:
                for target, source, shard_id in self._STACKED_QKV:
                    if source in name and name.replace(source, target) in params_dict:
                        fused_target = (name.replace(source, target), shard_id)
                        break
            if fused_target is not None:
                fused_name, shard_id = fused_target
                param = params_dict[fused_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(name)
                continue

            if name not in params_dict:
                raise KeyError(
                    f"Whisper checkpoint tensor {raw_name!r} has no destination "
                    f"parameter (looked for {name!r}); the checkpoint layout "
                    "does not match this implementation"
                )
            param = params_dict[name]
            loader = getattr(param, "weight_loader", None)
            if loader is None:
                param.data.copy_(loaded_weight)
            else:
                loader(param, loaded_weight)
            loaded.add(name)

        # Tied output projection: the checkpoint ships no proj_out.
        self.proj_out.weight = self.decoder.embed_tokens.weight
        return loaded

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ):
        req_pool_indices = kwargs.get("req_pool_indices")
        if req_pool_indices is None:
            raise RuntimeError(
                "Whisper needs req_pool_indices to find each request's "
                "cross-attention KV slot"
            )

        multimodal_context = kwargs.pop("multimodal_context", None)
        if (
            multimodal_context is not None
            and multimodal_context.has_extend_inputs()
            and not ctx.forward_mode.is_decode_or_idle()
        ):
            features, slots = _audio_features_for_extend(
                multimodal_context, req_pool_indices
            )
            if features is not None:
                self.encode_audio(features, slots)

        cross_key, cross_value = self.cross_cache.gather(req_pool_indices)
        encoder_len = self.config.max_source_positions
        bs = int(req_pool_indices.shape[0])
        device = input_ids.device

        # One query segment per request. In decode every segment is a single
        # token; in extend the segment is that request's prefill span, which
        # the executor already lays out contiguously.
        query_lengths = _query_lengths(ctx, bs, device)
        cu_seqlens_q = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.int32),
                torch.cumsum(query_lengths, dim=0, dtype=torch.int32),
            ]
        )
        cu_seqlens_k = torch.arange(
            0,
            (bs + 1) * encoder_len,
            encoder_len,
            device=device,
            dtype=torch.int32,
        )

        hidden_states = self.decoder(
            input_ids,
            positions,
            ctx,
            out_cache_loc,
            cross_key,
            cross_value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(query_lengths.max().item()),
            max_seqlen_k=encoder_len,
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.proj_out,
            LogitsMetadata.from_forward_context(ctx),
            None,
        )


def _query_lengths(ctx: ForwardContext, bs: int, device: torch.device) -> torch.Tensor:
    """Per-request query-row count for this forward.

    Derived from ``gather_ids`` on an extend (it is ``cumsum(lens) - 1``, so
    differencing recovers the lengths) and fixed at one row per request on a
    decode. Kept here rather than threaded through the layers because
    cross-attention is the only consumer.
    """
    if ctx.num_extends == 0 or ctx.gather_ids is None:
        return torch.ones(bs, device=device, dtype=torch.int32)
    ends = ctx.gather_ids + 1
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    return (ends - starts).to(device=device, dtype=torch.int32)


def _audio_features_for_extend(
    multimodal_context: Any, req_pool_indices: torch.Tensor
) -> tuple[torch.Tensor | None, list[int]]:
    """Collect the log-mel features of the requests being prefilled.

    Returns ``(None, [])`` when this extend carries no fresh audio -- a
    chunked continuation, or a batch whose requests were all encoded already.
    """
    items = getattr(multimodal_context, "extend_items", None)
    if not items:
        return None, []
    features = []
    slots: list[int] = []
    for position, item in enumerate(items):
        feature = getattr(item, "feature", None)
        if feature is None:
            continue
        if feature.ndim == 2:
            feature = feature.unsqueeze(0)
        features.append(feature)
        slots.append(int(req_pool_indices[position].item()))
    if not features:
        return None, []
    return torch.cat(features, dim=0), slots


EntryClass = WhisperForConditionalGeneration
