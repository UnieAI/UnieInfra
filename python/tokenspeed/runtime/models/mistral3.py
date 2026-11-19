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

"""Mistral 3 / Ministral 3 (``Mistral3ForConditionalGeneration``), text only.

The language model (``text_config.model_type == "ministral3"``) is a dense
Llama-style decoder with YaRN RoPE and the Llama-4 position-dependent query
temperature (``rope_parameters.llama_4_scaling_beta``). It reuses the Llama
building blocks.

UNSUPPORTED: the Pixtral vision tower and the multimodal projector. Their
checkpoint tensors are skipped and the architecture is not registered as
multimodal, so image requests are rejected by the text-only request path.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from tokenspeed.runtime.configs.utils import get_rope_parameters
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM, BaseTransformerModel
from tokenspeed.runtime.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
)
from tokenspeed.runtime.utils import add_prefix

_VISION_PREFIXES = ("model.vision_tower.", "model.multi_modal_projector.")


class Ministral3Attention(LlamaAttention):
    """Llama attention + YaRN + Llama-4 query temperature scaling."""

    def __init__(self, config, *args, **kwargs) -> None:
        # LlamaAttention builds YaRN RoPE from ``config.rope_scaling``.
        super().__init__(config, *args, **kwargs)
        params = get_rope_parameters(config)
        self.llama4_beta = float(params.get("llama_4_scaling_beta") or 0.0)
        self.llama4_orig_max = int(
            params.get("original_max_position_embeddings")
            or getattr(config, "max_position_embeddings", 8192)
        )

    def _attn(self, positions, q, k, v, ctx: ForwardContext) -> torch.Tensor:
        if self.llama4_beta != 0.0:
            # 1 + beta * log(1 + floor(pos / original_max)); RoPE is a rotation,
            # so scaling q before it equals HF's scaling after it.
            scale = 1.0 + self.llama4_beta * torch.log1p(
                torch.floor(positions.float() / self.llama4_orig_max)
            )
            q = (
                q.view(q.shape[0], -1, self.head_dim) * scale.to(q.dtype)[:, None, None]
            ).view_as(q)
        return self.attn(q, k, v, positions, ctx)


class Ministral3DecoderLayer(LlamaDecoderLayer):

    def resolve_attn(self, prefix: str) -> nn.Module:
        return Ministral3Attention(
            config=self.config,
            mapping=self.mapping,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            layer_id=self.layer_id,
            quant_config=self.quant_config,
            prefix=add_prefix("self_attn", prefix),
            sliding_window_size=getattr(self.config, "sliding_window", None) or -1,
        )


class Ministral3Model(BaseTransformerModel):
    layer_cls = Ministral3DecoderLayer


class Mistral3ForConditionalGeneration(BaseCausalLM):
    model_cls = Ministral3Model

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        text_config = getattr(config, "text_config", config)
        super().__init__(
            config=text_config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.is_multimodal_active = False

    def get_stacked_params_mapping(self) -> list[tuple[str, str, int | str]]:
        return [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

    @staticmethod
    def _rename(name: str) -> str | None:
        if name.startswith(_VISION_PREFIXES) or name.startswith(
            ("vision_tower.", "multi_modal_projector.")
        ):
            return None
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model.") :]
        elif name.startswith("language_model.model."):
            name = "model." + name[len("language_model.model.") :]
        elif name.startswith("language_model.lm_head."):
            name = name[len("language_model.") :]
        # HF FP8 (per-tensor) naming -> TokenSpeed Fp8LinearMethod naming.
        if name.endswith(".weight_scale_inv"):
            name = name[: -len("_inv")]
        elif name.endswith(".activation_scale"):
            name = name[: -len("activation_scale")] + "input_scale"
        return name

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any
    ) -> set[str]:
        stacked_params_mapping = self.get_stacked_params_mapping()
        params_dict = dict(self.named_parameters())
        tie = getattr(self.config, "tie_word_embeddings", False)
        loaded: set[str] = set()

        for name, loaded_weight in weights:
            name = self._rename(name)
            if name is None or "rotary_emb." in name:
                continue
            if tie and name.startswith("lm_head."):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded.add(name)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if loaded_weight.dim() == 0 and param.dim() == 1 and param.numel() == 1:
                    loaded_weight = loaded_weight.reshape(1)
                weight_loader(param, loaded_weight)
                loaded.add(name)
        return loaded


EntryClass = [Mistral3ForConditionalGeneration]
