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

"""Inference-only Gemma 4 text model compatible with HuggingFace weights.

This implements the *text* decoder of ``Gemma4ForConditionalGeneration`` (and
the equivalent ``Gemma4ForCausalLM``). The vision / audio towers of the
multimodal checkpoint are skipped -- text-only inputs are supported, which
covers the LLM benchmark. Only the dense variant (``enable_moe_block=False``,
e.g. ``gemma-4-31B-it``) is wired here; the MoE variant (``26B-A4B``) is a
follow-up (see the note on :class:`Gemma4DecoderLayer`).

Gemma-4-specific details that differ from a naive GQA port and from Gemma 3
(the source of truth is HF ``transformers.models.gemma4.modeling_gemma4`` +
the checkpoint ``config.json``):

* Embedding scaling by ``sqrt(hidden_size)`` in the compute dtype.
* RMSNorm uses the STANDARD ``x_normed * weight`` form (weights stored centred
  around 1). This is a change from Gemma 1/2/3, which used ``x_normed *
  (1 + weight)`` with zero-centred weights. Do NOT add 1 here, and do NOT
  reach for ``GemmaRMSNorm`` -- that class implements the Gemma 3 form.
* GeGLU MLP: ``down(gelu_tanh(gate) * up)``.
* Per-layer attention type: ``sliding_attention`` layers use ``head_dim=256``,
  ``num_key_value_heads`` and the local RoPE base; ``full_attention`` layers
  use ``head_dim=global_head_dim=512``, ``num_global_key_value_heads`` and the
  global (proportional / partially-rotary) RoPE base. The per-layer head_dim
  AND kv-head count both change -- the heterogeneous KV geometry this implies
  is planned by ``layers/attention/registry.py:_apply_gemma4_hetero_kv`` and
  needs a scheduler ext built with ``TOKENSPEED_FLAT_KVCACHE=ON``.
* ``query_pre_attn_scalar`` does NOT exist in Gemma 4 -- the attention softmax
  scaling is ``1.0`` (queries/keys are unit-RMS after qk-norm). This differs
  from Gemma 3.
* QK-norm: per-head RMSNorm on q and k before RoPE. Additionally a per-head
  RMSNorm *without* scale is applied to the values (``v_norm``).
* ``attention_k_eq_v=True``: ``full_attention`` layers have NO ``v_proj`` --
  the value stream is the raw ``k_proj`` output (taken before k-norm and RoPE)
  and then passed through ``v_norm``.
* A trained per-layer scalar ``layer_scalar`` multiplies each decoder layer's
  output.
* ``final_logit_softcapping=30`` (applied by the shared ``LogitsProcessor``).
* Tied embeddings: ``lm_head`` shares ``embed_tokens``.
"""

from __future__ import annotations

import os

from collections.abc import Iterable

import torch
from torch import nn

from tokenspeed.runtime.configs.paged_cache_spec import (
    split_groups_to_equal_depth,
)
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.activation import GeluTanhAndMul
from tokenspeed.runtime.layers.layernorm import RMSNorm, RMSNormWithoutScale
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import RotaryEmbedding
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.utils import add_prefix, get_colorful_logger, make_layers
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = get_colorful_logger(__name__)

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"


def _text_config(config):
    """Return the Gemma4 *text* sub-config for either the multimodal
    ``Gemma4Config`` (has ``.text_config``) or a bare text config."""
    return getattr(config, "text_config", None) or config


def layer_head_dim(config, layer_type: str) -> int:
    """Per-layer head dimension: global for full attention, local otherwise."""
    global_head_dim = getattr(config, "global_head_dim", None)
    if layer_type != SLIDING_ATTENTION and global_head_dim:
        return int(global_head_dim)
    return int(config.head_dim)


_HIDDEN_DUMP_DONE = False


_ATTN_DUMP_DONE = False


def _attn_dump_path(layer_id: int, num_tokens: int) -> str | None:
    """Path for a one-shot single-layer attention dump, or None.

    `TOKENSPEED_DUMP_ATTN=<path>` plus `TOKENSPEED_DUMP_ATTN_LAYER=<id>`, and
    the same exact-row-count gate as the hidden dump so the kernel tuner's
    8,192-token warm-up and the graph captures do not consume it.
    """
    if _ATTN_DUMP_DONE:
        return None
    path = os.environ.get("TOKENSPEED_DUMP_ATTN")
    if not path:
        return None
    want_layer = os.environ.get("TOKENSPEED_DUMP_ATTN_LAYER")
    if want_layer and int(want_layer) != int(layer_id):
        return None
    want = os.environ.get("TOKENSPEED_DUMP_HIDDEN_TOKENS")
    if want:
        allowed = {int(x) for x in want.replace(" ", "").split(",") if x}
        if int(num_tokens) not in allowed:
            return None
    elif num_tokens > 64:
        return None
    return path


def _mark_attn_dumped() -> None:
    global _ATTN_DUMP_DONE
    _ATTN_DUMP_DONE = True


def _forward_seq_lens(ctx) -> list[int] | None:
    """Per-sequence row counts for this forward, host-side, or None.

    ``ForwardContext`` deliberately carries no tensors, so the boundaries live
    on the attention backend's extend metadata. Without them a multi-sequence
    dump is one undivided block of rows and cannot be compared against a
    reference that runs each sequence on its own.
    """
    md = getattr(getattr(ctx, "attn_backend", None), "forward_extend_metadata", None)
    cu = getattr(md, "cu_extend_seq_lens_cpu", None)
    if not cu or len(cu) < 2:
        return None
    return [int(cu[i + 1]) - int(cu[i]) for i in range(len(cu) - 1)]


def _hidden_dump_path(num_tokens: int, num_seqs: int = 1) -> str | None:
    """Path for a one-shot layer dump, or None.

    Gated on an EXACT row count, not on "the first forward". The first forward
    a server does is `Kernel tuning with a dummy prefill of 8192 tokens`
    (model_executor.py), and after it come the prefill-graph bucket captures
    (16, 32, ...) and the decode-graph captures (batch 1, 2, 4, ... 64). All of
    those look like a forward and the first attempt at this dumped the kernel
    tuner: 8,192 rows x 60 layers x 5,376 floats = 10.9 GB on disk and not one
    row of the request.

    So the caller states how many tokens the request has and nothing else is
    dumped. Decode captures are batch-sized, so pick a prompt length that is not
    one of them.
    """
    if _HIDDEN_DUMP_DONE:
        return None
    path = os.environ.get("TOKENSPEED_DUMP_HIDDEN")
    if not path:
        return None
    # Multi-sequence mode. Every dump this project has taken held ONE sequence,
    # and one sequence cannot show a page mapping that lands on ANOTHER
    # sequence's pages -- the same blind spot as verifying a paged cache inside
    # a single page. Gate on the sequence count instead when this is set.
    min_seqs = os.environ.get("TOKENSPEED_DUMP_HIDDEN_MIN_SEQS")
    if min_seqs:
        return path if int(num_seqs) >= int(min_seqs) else None
    want = os.environ.get("TOKENSPEED_DUMP_HIDDEN_TOKENS")
    if want:
        # A comma list, because the caller usually knows the prompt but not
        # whether the server prepends BOS.
        allowed = {int(x) for x in want.replace(" ", "").split(",") if x}
        if int(num_tokens) not in allowed:
            return None
    if not want and num_tokens > 64:
        return None
    return path


def _mark_hidden_dumped() -> None:
    global _HIDDEN_DUMP_DONE
    _HIDDEN_DUMP_DONE = True


def build_rope_inv_freq(layer_type: str, config) -> torch.Tensor:
    """Inverse RoPE frequencies for one Gemma4 layer type.

    Matches HF ``Gemma4TextRotaryEmbedding``:

    * ``sliding_attention`` / ``rope_type == "default"``: full rotary over
      ``head_dim`` with the local base.
    * ``full_attention`` / ``rope_type == "proportional"``: partial rotary of
      ``partial_rotary_factor * global_head_dim`` channels with the global
      base, the remaining ("nope") channels get a zero frequency so RoPE is the
      identity there. The returned tensor always has length ``head_dim // 2``.

    The zero-tail encoding is load-bearing and is **not** interchangeable with
    ``get_rope(head_size=global_head_dim, rotary_dim=partial*global_head_dim)``.
    Neox rotation pairs channel ``i`` with ``i + head_dim // 2``; a shortened
    ``rotary_dim`` would instead pair ``i`` with ``i + rotary_dim // 2`` and
    rotate a different set of channels. Both run, and only one is Gemma 4.

    Args:
        layer_type: ``"sliding_attention"`` or ``"full_attention"``.
        config: The Gemma4 text config.

    Returns:
        1-D float32 tensor of inverse frequencies of length ``head_dim // 2``,
        where ``head_dim`` is the per-layer head dimension.
    """
    params = config.rope_parameters[layer_type]
    base = float(params["rope_theta"])
    rope_type = params.get("rope_type", "default")
    if rope_type == "proportional":
        head_dim = config.global_head_dim
        partial = float(params.get("partial_rotary_factor", 1.0))
        rope_angles = int(partial * head_dim // 2)
        inv_rot = 1.0 / (
            base
            ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float()
                / head_dim
            )
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            return torch.cat([inv_rot, torch.zeros(nope_angles, dtype=torch.float32)])
        return inv_rot
    # default rope
    head_dim = config.head_dim
    return 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
    )


class Gemma4RotaryEmbedding(RotaryEmbedding):
    """Shared-cache RoPE for one Gemma 4 layer type.

    Subclasses the engine's :class:`RotaryEmbedding` purely to substitute the
    inverse-frequency vector; everything downstream (the packed cos/sin cache,
    the fused ``apply_rope`` kernel, the fused KV pre-write path) is the
    stock implementation. ``rotary_dim == head_size`` because the proportional
    variant encodes its "nope" channels as zero frequencies rather than as a
    shortened rotary width -- see :func:`build_rope_inv_freq`.

    One instance is built per layer type and shared by every layer of that
    type: the cache is ``[max_position, head_dim]`` float32, so a per-layer
    instance would cost 60x what it needs to.

    Args:
        config: The Gemma4 text config.
        layer_type: ``"sliding_attention"`` or ``"full_attention"``.
        max_position: Rows to precompute; clamp this to the served context
            length rather than ``max_position_embeddings``.
        dtype: Model compute dtype (recorded; the cache itself is float32).
    """

    def __init__(
        self,
        config,
        layer_type: str,
        max_position: int,
        dtype: torch.dtype,
    ) -> None:
        self._inv_freq = build_rope_inv_freq(layer_type, config)
        head_dim = layer_head_dim(config, layer_type)
        if self._inv_freq.numel() * 2 != head_dim:
            raise ValueError(
                f"Gemma 4 {layer_type} rope has {self._inv_freq.numel()} angles "
                f"for head_dim {head_dim}; expected {head_dim // 2}."
            )
        super().__init__(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=max_position,
            base=float(config.rope_parameters[layer_type]["rope_theta"]),
            is_neox_style=True,
            dtype=dtype,
        )

    def _compute_inv_freq(self, base: int | float) -> torch.Tensor:
        # `base` is already folded into the per-layer-type vector built above.
        return self._inv_freq


def rope_cache_positions(config) -> int:
    """Rows to precompute in the RoPE cache.

    ``max_position_embeddings`` is 262144 on gemma-4-31B-it, and the cache is
    ``[rows, head_dim]`` float32 -- 512 MB for the full-attention layers plus
    256 MB for the sliding ones. Nothing can attend past the served context, so
    clamp to it. ``deepseek_v4`` reads the same key for the same reason.
    """
    max_position = int(getattr(config, "max_position_embeddings", 0) or 0)
    served = global_server_args_dict.get("max_model_len")
    if served:
        clamped = min(max_position, int(served)) if max_position else int(served)
        if max_position and clamped < max_position:
            logger.info(
                "Gemma 4 RoPE cache clamped to %d positions (config declares %d).",
                clamped,
                max_position,
            )
        return clamped
    return max_position


class Gemma4MLP(nn.Module):
    """GeGLU MLP: ``down(gelu_tanh(gate) * up)``."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: QuantizationConfig | None = None,
        tp_rank: int | None = None,
        tp_size: int | None = None,
        tp_group: tuple[int, ...] | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                f"Unsupported activation {hidden_activation!r}; Gemma 4 uses "
                "gelu_pytorch_tanh."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.act_fn = GeluTanhAndMul()
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class Gemma4Attention(nn.Module):
    """Gemma 4 attention with per-layer head_dim, kv-head count, sliding
    window, qk/v-norm and (for full-attention layers) shared K/V."""

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        layer_type: str,
        rotary_emb: Gemma4RotaryEmbedding,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        group_id: str | None = None,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.layer_type = layer_type
        self.layer_id = layer_id
        # The paged-cache group this layer's KV lives in. Distinct from
        # layer_type: the sliding layers are relabelled into equal-depth
        # sub-groups so no slab row is left dead (see
        # split_groups_to_equal_depth). Every geometry decision below
        # still keys off layer_type, which is what the checkpoint says.
        self.group_id = group_id or layer_type
        self.is_sliding = layer_type == SLIDING_ATTENTION
        self.rotary_emb = rotary_emb

        tp_rank = mapping.attn.tp_rank
        tp_size = mapping.attn.tp_size
        tp_group = mapping.attn.tp_group

        hidden_size = config.hidden_size
        # Per-layer head dim: global (512) for full-attention, local (256) for
        # sliding-attention layers.
        self.head_dim = layer_head_dim(config, layer_type)
        # attention_k_eq_v: full-attention layers derive V from K (no v_proj)
        # and use the smaller global kv-head count.
        self.k_eq_v = bool(getattr(config, "attention_k_eq_v", False)) and (
            not self.is_sliding
        )
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = (
            config.num_global_key_value_heads
            if self.k_eq_v
            else config.num_key_value_heads
        )
        validate_attention_partition(total_num_heads, total_num_kv_heads, tp_size)
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Gemma 4 has NO query_pre_attn_scalar: the softmax scaling is 1.0
        # (queries/keys are unit-RMS after qk-norm).
        self.scaling = 1.0

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            total_num_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            total_num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("k_proj", prefix),
        )
        if self.k_eq_v:
            self.v_proj = None
        else:
            self.v_proj = ColumnParallelLinear(
                hidden_size,
                total_num_kv_heads * self.head_dim,
                bias=config.attention_bias,
                quant_config=quant_config,
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_group=tp_group,
                prefix=add_prefix("v_proj", prefix),
            )
        self.o_proj = RowParallelLinear(
            total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("o_proj", prefix),
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # v_norm carries no learnable weight (HF ``with_scale=False``), so
        # there is nothing for the checkpoint loader to fill.
        self.v_norm = RMSNormWithoutScale(config.rms_norm_eps)

        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(config.sliding_window if self.is_sliding else -1),
            group_id=self.group_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q, _ = self.q_proj(hidden_states)
        k_raw, _ = self.k_proj(hidden_states)

        # Values: for attention_k_eq_v layers the value stream is the raw
        # k_proj output (BEFORE k-norm and RoPE); otherwise a dedicated v_proj.
        if self.v_proj is None:
            v = k_raw
        else:
            v, _ = self.v_proj(hidden_states)

        # qk-norm (per head) then RoPE. The norms reduce over head_dim, so the
        # head axis is folded into the row axis rather than materialised.
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(num_tokens, self.q_size)
        k = self.k_norm(k_raw.reshape(-1, self.head_dim)).view(num_tokens, self.kv_size)
        q, k = self.rotary_emb(positions, q, k)

        # v-norm (no learnable scale), no RoPE on values.
        v = self.v_norm(v.reshape(-1, self.head_dim)).view(num_tokens, self.kv_size)

        attn_output = self.attn(q, k, v, ctx, out_cache_loc)
        if attn_output.dim() == 3:
            attn_output = attn_output.reshape(attn_output.shape[0], -1)
        # One layer's intermediates, for the same reason as the per-layer dump
        # above: the per-layer comparison put the fault in the ten
        # full_attention layers and nowhere else, and the three things those
        # layers do differently (head_dim 512, attention_k_eq_v, partial rope)
        # cannot be told apart from the layer's output alone. Dumping q/k/v and
        # the attention output splits it: recomputing attention from the dumped
        # q/k/v with SDPA says whether the fault is the cache round trip or what
        # this function hands the kernel.
        _adump = _attn_dump_path(self.layer_id, hidden_states.shape[0])
        if _adump is not None:
            import torch as _t

            _t.save(
                {
                    "layer_id": self.layer_id,
                    "layer_type": self.layer_type,
                    "head_dim": self.head_dim,
                    "num_heads": self.num_heads,
                    "num_kv_heads": self.num_kv_heads,
                    "k_eq_v": self.k_eq_v,
                    "hidden_in": hidden_states.detach().float().cpu(),
                    "k_raw": k_raw.detach().float().cpu(),
                    "q": q.detach().float().cpu(),
                    "k": k.detach().float().cpu(),
                    "v": v.detach().float().cpu(),
                    "attn_out": attn_output.detach().float().cpu(),
                },
                _adump,
            )
            _mark_attn_dumped()
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4DecoderLayer(nn.Module):
    """Gemma 4 decoder layer: sandwich norms around attention + MLP and a
    trained per-layer output scalar.

    The sandwich shape (residual added *after* the post-norm) does not fit
    ``BaseDecoderLayer``'s pre-norm shell, so this layer is hand-written and
    therefore does not get ``CommManager``'s fused allreduce+norm. With TP > 1
    it pays two bare all-reduces per layer instead; expressing the layer as a
    ``CompiledDecoderLayer`` execution plan is the follow-up that recovers the
    fusion for Gemma 3 and 4 together.

    NOTE (MoE follow-up): the ``gemma-4-26B-A4B`` variant sets
    ``enable_moe_block=True`` and adds a routed-expert branch that runs in
    parallel with this dense MLP (its output combined via two extra
    post-feedforward norms). To add it, resolve ``self.mlp`` to a fused-experts
    block (reuse ``qwen3_moe`` / ``qwen3_5_moe`` MoELayer) alongside the dense
    MLP; the attention / norm / layer_scalar wiring below is shared.
    """

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        rotary_emb: Gemma4RotaryEmbedding,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        layer_type = config.layer_types[layer_id]
        # Same relabelling the KV pool published its groups from, so a
        # layer's group_id and the pool's groups cannot drift apart.
        group_labels = split_groups_to_equal_depth(config.layer_types)
        self.self_attn = Gemma4Attention(
            config=config,
            mapping=mapping,
            group_id=group_labels[layer_id],
            layer_id=layer_id,
            layer_type=layer_type,
            rotary_emb=rotary_emb,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Gemma4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            tp_rank=mapping.dense.tp_rank,
            tp_size=mapping.dense.tp_size,
            tp_group=mapping.dense.tp_group,
            prefix=add_prefix("mlp", prefix),
        )
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        # Trained per-layer output scalar (NOT all ones in the checkpoint).
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        # Attention block (sandwich norms; explicit residual -- Gemma adds the
        # residual AFTER the post-attention norm).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, ctx, out_cache_loc)
        # o_proj / down_proj are built with reduce_results=False, so the shard
        # sums are still partial here.
        if self.mapping.attn.tp_size > 1:
            hidden_states = all_reduce(hidden_states, self.mapping.attn.tp_group)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward block.
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mapping.dense.tp_size > 1:
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class Gemma4Model(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # One RoPE cache per layer type, shared by every layer of that type.
        max_position = rope_cache_positions(config)
        dtype = getattr(config, "dtype", None) or torch.get_default_dtype()
        self.rotary_emb = nn.ModuleDict(
            {
                layer_type: Gemma4RotaryEmbedding(
                    config, layer_type, max_position, dtype
                )
                for layer_type in sorted(set(config.layer_types))
            }
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Gemma4DecoderLayer(
                config=config,
                mapping=mapping,
                layer_id=idx,
                rotary_emb=self.rotary_emb[config.layer_types[idx]],
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size
        self._normalizer_by_dtype: dict[torch.dtype, float] = {}

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        # Gemma normalizer: sqrt(hidden_size), rounded into the embedding
        # compute dtype to match HF's bf16 ``embed_scale``.
        #
        # This used to be ``embeds.new_tensor(self.hidden_size**0.5)``, which
        # builds the scalar on CPU and copies it to the device on every forward.
        # That copy is illegal once CUDA graph capture is running:
        #
        #   RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA
        #   graph capture unless the CPU tensor is pinned.
        #
        # and it took gemma-4-31B down at startup before it served a request.
        #
        # Rounding still happens, just once and entirely on CPU: build the
        # scalar at the target dtype, read it back as a Python float, and cache
        # it. Multiplying a device tensor by a Python scalar folds the constant
        # into the kernel -- no allocation, no transfer, nothing for the capture
        # to object to. Keyed by dtype so a run in a different precision gets
        # its own correctly-rounded constant rather than bf16's.
        scale = self._normalizer_by_dtype.get(embeds.dtype)
        if scale is None:
            scale = float(torch.tensor(self.hidden_size**0.5, dtype=embeds.dtype))
            self._normalizer_by_dtype[embeds.dtype] = scale
        return embeds * scale

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if input_embeds is None:
            hidden_states = self.get_input_embeddings(input_ids)
        else:
            hidden_states = input_embeds
        # Layer-by-layer capture for reference comparison. Off unless
        # TOKENSPEED_DUMP_HIDDEN names a file, and it dumps ONE forward and then
        # stops: this exists because gemma-4 answered "The" with 'H' here and in
        # language on vLLM, and at one token of context the softmax has a single
        # element -- attention returns V exactly and RoPE is a zero-angle
        # rotation -- so every cheap explanation (kernels, rope, norms, the
        # weight loader) had already been eliminated and only a numerical
        # comparison against HF was left.
        _seq_lens = _forward_seq_lens(ctx)
        _dump = _hidden_dump_path(
            hidden_states.shape[0], len(_seq_lens) if _seq_lens else 1
        )
        if _dump is not None:
            import torch as _t

            _rec = {
                "input_ids": input_ids.detach().to("cpu"),
                "positions": positions.detach().to("cpu"),
                "seq_lens": _seq_lens,
                "embed": hidden_states.detach().float().to("cpu"),
            }
            for _i, layer in enumerate(self.layers):
                hidden_states = layer(positions, hidden_states, ctx, out_cache_loc)
                _rec[f"layer{_i}"] = hidden_states.detach().float().to("cpu")
            hidden_states = self.norm(hidden_states)
            _rec["final_norm"] = hidden_states.detach().float().to("cpu")
            _t.save(_rec, _dump)
            _mark_hidden_dumped()
            return hidden_states, None
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, ctx, out_cache_loc)
        hidden_states = self.norm(hidden_states)
        return hidden_states, None


class Gemma4ForConditionalGeneration(BaseCausalLM):
    """Text decoder of ``Gemma4ForConditionalGeneration``.

    The vision / audio towers are skipped: text-only inputs are supported
    (covers the LLM benchmark). ``final_logit_softcapping`` and the tied
    ``lm_head`` are handled by the shared ``BaseCausalLM`` / ``LogitsProcessor``
    (both read from the text config passed to ``super().__init__``).
    """

    model_cls = Gemma4Model

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        # The sandwich-norm layer keeps attention and MLP on one residual
        # stream, so a split attn/dense TP would need a shard exchange this
        # layer does not perform. Refuse rather than return wrong numbers.
        if mapping.attn.tp_size != mapping.dense.tp_size:
            raise ValueError(
                "Gemma 4 requires attn.tp_size == dense.tp_size, got "
                f"{mapping.attn.tp_size} != {mapping.dense.tp_size}."
            )
        # Drive the LM off the text sub-config so BaseCausalLM sees hidden_size,
        # vocab_size, tie_word_embeddings and final_logit_softcapping.
        super().__init__(
            config=_text_config(config),
            mapping=mapping,
            quant_config=quant_config,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs):
        # Gemma stores gate/up separately; fuse into gate_up_proj.
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        skipped: list[str] = []
        for name, loaded_weight in weights:
            # Skip the vision / audio towers and multimodal projectors.
            if (
                name.startswith("model.vision_tower")
                or name.startswith("model.audio_tower")
                or name.startswith("model.embed_vision")
                or name.startswith("model.embed_audio")
                or "rotary_emb.inv_freq" in name
            ):
                continue
            # Text weights live under model.language_model.* in the multimodal
            # checkpoint; map onto our model.* namespace.
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model.") :]
            elif name.startswith("language_model.model."):
                name = "model." + name[len("language_model.model.") :]
            # Tied lm_head is not stored separately.
            if "lm_head" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "mlp" not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(mapped)
                break
            else:
                if name not in params_dict:
                    skipped.append(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)

        # A checkpoint that renames a tensor would otherwise load "successfully"
        # and generate wrong text, which is the expensive way to find out.
        missing = sorted(set(params_dict) - loaded)
        if missing or skipped:
            raise ValueError(
                "Gemma 4 checkpoint did not match the model: "
                f"{len(missing)} parameter(s) never written "
                f"(e.g. {missing[:5]}), "
                f"{len(skipped)} checkpoint tensor(s) unclaimed "
                f"(e.g. {skipped[:5]})."
            )


# ``Gemma4ForCausalLM`` is the text-only architecture string; it maps to the
# same implementation (a bare text config has no ``.text_config``).
class Gemma4ForCausalLM(Gemma4ForConditionalGeneration):
    pass


EntryClass = [Gemma4ForConditionalGeneration, Gemma4ForCausalLM]
