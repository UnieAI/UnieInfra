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
follow-up (see the module-level notes below).

Gemma-4-specific details that differ from a naive GQA port and from Gemma 3
(the source of truth is HF ``transformers.models.gemma4.modeling_gemma4`` +
the checkpoint ``config.json``):

* Embedding scaling by ``sqrt(hidden_size)`` in the compute dtype.
* RMSNorm uses the STANDARD ``x_normed * weight`` form (weights stored centred
  around 1). This is a change from Gemma 1/2/3, which used ``x_normed *
  (1 + weight)`` with zero-centred weights. Do NOT add 1 here.
* GeGLU MLP: ``down(gelu_tanh(gate) * up)``.
* Per-layer attention type: ``sliding_attention`` layers use ``head_dim=256``,
  ``num_key_value_heads`` and the local RoPE base; ``full_attention`` layers
  use ``head_dim=global_head_dim=512``, ``num_global_key_value_heads`` and the
  global (proportional / partially-rotary) RoPE base. The per-layer head_dim
  AND kv-head count both change.
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

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.utils import add_prefix, make_layers


def _text_config(config):
    """Return the Gemma4 *text* sub-config for either the multimodal
    ``Gemma4Config`` (has ``.text_config``) or a bare text config."""
    return getattr(config, "text_config", None) or config


def build_rope_inv_freq(layer_type: str, config) -> torch.Tensor:
    """Inverse RoPE frequencies for one Gemma4 layer type.

    Matches HF ``Gemma4TextRotaryEmbedding``:

    * ``sliding_attention`` / ``rope_type == "default"``: full rotary over
      ``head_dim`` with the local base.
    * ``full_attention`` / ``rope_type == "proportional"``: partial rotary of
      ``partial_rotary_factor * global_head_dim`` channels with the global
      base, the remaining ("nope") channels get a zero frequency so RoPE is the
      identity there. The returned tensor always has length ``head_dim // 2``.

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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rms_norm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma4 ``v_norm``: RMSNorm with ``with_scale=False`` (no learnable
    weight). Computed in float32 then cast back, matching HF."""
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return xf.to(dtype)


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
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.gelu(gate, approximate="tanh") * up
        x, _ = self.down_proj(x)
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
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.layer_type = layer_type
        self.is_sliding = layer_type == "sliding_attention"
        self.rms_norm_eps = config.rms_norm_eps

        tp_rank = mapping.attn.tp_rank
        tp_size = mapping.attn.tp_size
        tp_group = mapping.attn.tp_group

        hidden_size = config.hidden_size
        # Per-layer head dim: global (512) for full-attention, local (256) for
        # sliding-attention layers.
        self.head_dim = (
            config.global_head_dim
            if (not self.is_sliding and getattr(config, "global_head_dim", None))
            else config.head_dim
        )
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
        # v_norm has no learnable weight (with_scale=False); applied inline.

        sliding_window = config.sliding_window if self.is_sliding else -1
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(sliding_window if self.is_sliding else -1),
            group_id=layer_type,
        )

        self.register_buffer(
            "rope_inv_freq", build_rope_inv_freq(layer_type, config), persistent=False
        )

    def _cos_sin(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.rope_inv_freq.to(device=positions.device, dtype=torch.float32)
        freqs = positions.float()[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _apply_rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # x: [T, H, head_dim]; cos/sin: [T, head_dim]
        return x * cos.unsqueeze(1) + _rotate_half(x) * sin.unsqueeze(1)

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

        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k_raw = k_raw.view(num_tokens, self.num_kv_heads, self.head_dim)

        # Values: for attention_k_eq_v layers the value stream is the raw
        # k_proj output (BEFORE k-norm and RoPE); otherwise a dedicated v_proj.
        if self.v_proj is None:
            v = k_raw
        else:
            v, _ = self.v_proj(hidden_states)
            v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        # qk-norm (per head) then RoPE.
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(
            num_tokens, self.num_heads, self.head_dim
        )
        k = self.k_norm(k_raw.reshape(-1, self.head_dim)).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        cos, sin = self._cos_sin(positions, q.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # v-norm (no learnable scale), no RoPE on values.
        v = _rms_norm_no_weight(v, self.rms_norm_eps)

        q = q.reshape(num_tokens, self.q_size)
        k = k.reshape(num_tokens, self.kv_size)
        v = v.reshape(num_tokens, self.kv_size)

        attn_output = self.attn(q, k, v, ctx, out_cache_loc)
        if attn_output.dim() == 3:
            attn_output = attn_output.reshape(attn_output.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4DecoderLayer(nn.Module):
    """Gemma 4 decoder layer: sandwich norms around attention + MLP and a
    trained per-layer output scalar.

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
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        layer_type = config.layer_types[layer_id]
        self.self_attn = Gemma4Attention(
            config=config,
            mapping=mapping,
            layer_id=layer_id,
            layer_type=layer_type,
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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward block.
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
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
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Gemma4DecoderLayer(
                config=config,
                mapping=mapping,
                layer_id=idx,
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
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# ``Gemma4ForCausalLM`` is the text-only architecture string; it maps to the
# same implementation (a bare text config has no ``.text_config``).
class Gemma4ForCausalLM(Gemma4ForConditionalGeneration):
    pass


EntryClass = [Gemma4ForConditionalGeneration, Gemma4ForCausalLM]
