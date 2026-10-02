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

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.configs.paged_cache_spec import split_groups_to_equal_depth
from tokenspeed.runtime.layers.attention.configs.base import (
    BaseAttnConfig,
    resolve_dtype,
    resolve_speculative_num_tokens,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


def _per_layer_kv_geometry(
    text_config,
    raw_layer_types: tuple[str, ...],
    num_layers: int,
    default_heads: int,
    default_dim: int,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """Per-layer (kv_heads, head_dim) when the non-sliding layers differ.

    gemma-4 is the first model here whose head_dim varies per layer: the
    checkpoint carries `num_global_key_value_heads` (4) and `global_head_dim`
    (512) for the ten full-attention layers while the top level describes the
    sliding half (16 x 256). A pool built from the top level alone gives the
    global layers a row half the width they need, and every one of their tokens
    lands at half stride -- measured as garbage output on gemma-4-31B-it with
    the 50 sliding layers correct and the 10 global ones not.

    Returns (None, None) for every model that shares one geometry, which is
    every other model in this tree.

    Args:
        text_config: The decoder config (gemma ships as *ForConditionalGeneration,
            so this is `hf_config.text_config`).
        raw_layer_types: Checkpoint layer labels BEFORE the equal-depth split,
            because that split renames only the sliding half and the geometry
            decision must match `models/gemma.py`, which keys off the raw label.
        num_layers: Attention layer count; a label list that does not cover it
            is not trusted.
        default_heads: Top-level KV head count (pre-TP).
        default_dim: Top-level head_dim.

    Returns:
        `(heads_per_layer, dims_per_layer)`, or `(None, None)`.
    """
    global_heads = getattr(text_config, "num_global_key_value_heads", None)
    global_dim = getattr(text_config, "global_head_dim", None)
    if not global_heads and not global_dim:
        return None, None
    heads = int(global_heads or default_heads)
    dim = int(global_dim or default_dim)
    if heads == default_heads and dim == default_dim:
        # Declared but identical: nothing to describe, and staying None keeps
        # the pool on its uniform path.
        return None, None
    if len(raw_layer_types) != num_layers:
        # Without per-layer labels there is no way to say WHICH layers are
        # global. Refuse rather than guess: the uniform path is wrong here, but
        # a wrong guess is wrong everywhere instead of somewhere.
        raise ValueError(
            "model declares a distinct global attention geometry "
            f"({heads} x {dim} against {default_heads} x {default_dim}) but its "
            f"{len(raw_layer_types)} layer_types do not cover {num_layers} "
            "attention layers, so the global layers cannot be identified"
        )
    per_heads = []
    per_dims = []
    for label in raw_layer_types:
        if str(label).startswith("sliding"):
            per_heads.append(int(default_heads))
            per_dims.append(int(default_dim))
        else:
            per_heads.append(heads)
            per_dims.append(dim)
    return tuple(per_heads), tuple(per_dims)


@dataclass
class MHAConfig(BaseAttnConfig):
    # Per-layer attention-type labels + window, forwarded to the KV pool for
    # paged_cache_group_specs publication (empty -> single full-history group).
    layer_types: tuple[str, ...] = ()
    sliding_window_tokens: int | tuple[int | None, ...] | None = None
    max_scheduled_tokens: int = 0
    # True iff server_args.disaggregation_mode != "null"; used to reject
    # layouts whose aliased fields cannot use legacy per-layer transfers.
    pd_disaggregation_enabled: bool = False
    # Per-group scheduler page sizes, published by the registry for
    # group-aware backends (backends/cache_groups.py).
    group_page_sizes: dict[str, int] | None = None
    layer_kv_head_counts: tuple[int, ...] | None = None
    # Per-layer KV head_dim, for models whose layers do not share one.
    # `layer_kv_head_counts` alone cannot describe gemma-4: its ten global
    # layers are 4 heads x 512 against the sliding half's 16 x 256, so the
    # per-token width is half the allocation width while the head COUNT says a
    # quarter. See MHATokenToKVPool._layer_row_view.
    layer_kv_head_dims: tuple[int, ...] | None = None

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ):
        kwargs = {}
        if server_args.speculative_algorithm is not None:
            kwargs.update(
                speculative_num_steps=server_args.speculative_num_steps,
                speculative_num_draft_tokens=resolve_speculative_num_tokens(
                    server_args, is_draft
                ),
            )
        kv_cache_dtype = server_args.kv_cache_dtype
        draft_block_decode = bool(
            is_draft and server_args.speculative_algorithm in ("DFLASH", "DSPARK")
        )
        if draft_block_decode and server_args.drafter_attention_backend != "trtllm":
            kv_cache_dtype = "bfloat16"
        # "auto" means the model's own dtype, per --kv-cache-dtype's help. A
        # fp16 checkpoint (Whisper-large-v3) must therefore get a fp16 cache:
        # the paged/dense MHA kernels are registered for either dtype but only
        # for an all-same mix, so a fp16 query against a hardcoded-bf16 pool
        # matches no kernel and the engine dies at the first decode step.
        resolved_kv_cache_dtype = (
            model_config.dtype
            if kv_cache_dtype == "auto"
            else resolve_dtype(kv_cache_dtype)
        )

        hf_config = getattr(model_config, "hf_config", None)
        # Gemma 3/4 ship as *ForConditionalGeneration: the decoder's labels and
        # window live on text_config and the top level carries NEITHER, so
        # reading only the top level published one full-history group for all
        # 60 layers -- 960 KiB/token on gemma-4-31B-it where the windowed
        # layers are worth 80.
        text_config = getattr(hf_config, "text_config", None) or hf_config
        # paged_cache_layer_types wins: it can carry labels outside transformers' ALLOWED_LAYER_TYPES
        published_types = getattr(
            hf_config, "paged_cache_layer_types", None
        ) or getattr(text_config, "paged_cache_layer_types", None)
        raw_layer_types = getattr(hf_config, "layer_types", None) or getattr(
            text_config, "layer_types", None
        )
        layer_types = tuple(published_types or ())
        if not layer_types and raw_layer_types:
            # Same derivation the model side assigns group_id from
            # (models/gemma.py), so a layer's group id and the pool's published
            # groups cannot drift apart. A no-op for equal-depth stacks, which
            # is every other MHA model that labels its layers (gpt-oss is 1:1).
            layer_types = split_groups_to_equal_depth(tuple(raw_layer_types))
        if (
            is_draft
            and layer_types
            and len(layer_types) != model_config.num_attention_layers
        ):
            # Target-stack labels don't fit the draft depth; drop so the pool falls back to full attn
            layer_types = ()
        kv_head_counts, kv_head_dims = _per_layer_kv_geometry(
            text_config,
            tuple(raw_layer_types or ()),
            model_config.num_attention_layers,
            model_config.num_key_value_heads,
            model_config.head_dim,
        )
        if is_draft and kv_head_dims is not None:
            # Same reason layer_types is dropped for a draft above: the target
            # stack's per-layer geometry does not describe the draft depth.
            kv_head_counts = kv_head_dims = None
        sliding_window_tokens = getattr(hf_config, "sliding_window", None)
        if sliding_window_tokens is None and layer_types:
            # Only ever read through with labels in hand: a scalar window
            # broadcasts to every sliding layer, and an unlabelled stack would
            # take it on ALL of them.
            sliding_window_tokens = getattr(text_config, "sliding_window", None)
        return cls(
            device=server_args.device,
            context_len=model_config.context_len + server_args.spec_context_pad,
            backend_name=(
                server_args.attention_backend
                if not is_draft
                else server_args.drafter_attention_backend
            ),
            num_attention_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            attn_tp_size=server_args.attn_tp_size or server_args.mapping.attn.tp_size,
            dtype=model_config.dtype,
            kv_cache_dtype=resolved_kv_cache_dtype,
            kv_cache_mxfp8=kv_cache_dtype == "mxfp8",
            page_size=server_args.block_size,
            max_bs=server_args.max_num_seqs
            // (server_args.data_parallel_size or server_args.mapping.attn.dp_size),
            max_graph_bs=server_args.max_cudagraph_capture_size,
            kv_cache_quant_method=server_args.kv_cache_quant_method,
            is_draft=is_draft,
            draft_block_decode=draft_block_decode,
            layer_types=layer_types,
            layer_kv_head_counts=kv_head_counts,
            layer_kv_head_dims=kv_head_dims,
            sliding_window_tokens=sliding_window_tokens,
            max_scheduled_tokens=getattr(server_args, "chunked_prefill_size", 8192),
            pd_disaggregation_enabled=getattr(
                server_args, "disaggregation_mode", "null"
            )
            != "null",
            **kwargs,
        )

    def cache_cell_size(self) -> int:
        # The WIDEST layer, not the top-level one. Every layer is allocated at
        # this width and a narrower layer reinterprets the same bytes as more
        # rows (MHATokenToKVPool._layer_row_view), so a model whose global
        # layers were wider than its top level would otherwise be allocated
        # short. gemma-4 is the other way round -- 4 x 512 against 16 x 256 --
        # so this is unchanged for it, and for every uniform model.
        width = max(self._per_layer_kv_widths(), default=0) or (
            max(self.num_kv_heads // self.attn_tp_size, 1) * self.head_dim
        )
        cell = width * 2 * torch._utils._element_size(self.kv_cache_dtype)
        if self.kv_cache_mxfp8:
            # One UE8M0 byte per 32 fp8 data bytes.
            cell += cell // 32
        return cell

    def _per_layer_kv_widths(self) -> list[int]:
        """Per-layer KV elements per token (post-TP), empty when uniform."""
        if not self.layer_kv_head_dims or not self.layer_kv_head_counts:
            return []
        return [
            max(heads // self.attn_tp_size, 1) * dim
            for heads, dim in zip(self.layer_kv_head_counts, self.layer_kv_head_dims)
        ]
