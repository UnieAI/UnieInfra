# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
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
from tokenspeed.runtime.configs.paged_cache_spec import (
    STATE_LAYER_TYPES,
    PagedCacheGroupSpec,
    resolve_layer_types_and_window,
    scheduler_ext_flat_kvcache,
    split_groups_to_equal_depth,
)
from tokenspeed.runtime.layers.attention.configs.base import (
    BaseAttnConfig,
    resolve_dtype,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.utils.hf_transformers_utils import resolve_architecture
from tokenspeed.runtime.utils.server_args import ServerArgs

# Architectures whose cache groups are relabelled to equal depth before the
# KV profile; see split_groups_to_equal_depth.
_EQUAL_DEPTH_GROUP_ARCHITECTURES = frozenset(
    {"Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"}
)


@dataclass
class MHAConfig(BaseAttnConfig):
    # Per-layer attention-type labels + window, forwarded to the KV pool for
    # paged_cache_group_specs publication (empty -> single full-history group).
    layer_types: tuple[str, ...] = ()
    sliding_window_tokens: int | tuple[int | None, ...] | None = None
    max_scheduled_tokens: int = 0
    # True iff server_args.disaggregation_mode != "null"; the pool's slab
    # guards consume it.
    pd_disaggregation_enabled: bool = False
    # Mamba2/GDN per-state-layer shapes and dtypes (the configs'
    # mamba2_cache_params), forwarded to the pool's state slabs. Populated
    # only on a flat-built scheduler ext — the radix path keeps its
    # SimpleMambaPool state ownership byte-identical (None here means the
    # pool neither allocates state slabs nor runs the page-geometry check).
    conv_state_shape: tuple[int, ...] | None = None
    # Extra model-declared paged-cache groups (e.g. Inkling paged sconv); forwarded to publication
    extra_paged_groups: tuple[PagedCacheGroupSpec, ...] = ()
    # Slot span in tokens (largest group's block)
    slot_tokens: int | None = None
    # Per-group page sizes (hetero zero-padding slots)
    group_page_sizes: dict[str, int] | None = None
    layer_kv_head_counts: tuple[int, ...] | None = None
    temporal_state_shape: tuple[int, ...] | None = None
    conv_dtype: torch.dtype | None = None
    ssm_dtype: torch.dtype | None = None

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ):
        kwargs = {}
        if server_args.speculative_algorithm is not None:
            kwargs.update(
                speculative_num_steps=server_args.speculative_num_steps,
                speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
            )
        kv_cache_dtype = server_args.kv_cache_dtype
        draft_block_decode = bool(
            is_draft and server_args.speculative_algorithm == "DFLASH"
        )
        if draft_block_decode and server_args.drafter_attention_backend != "trtllm":
            kv_cache_dtype = "bfloat16"

        hf_config = getattr(model_config, "hf_config", None)
        # Composite configs (Gemma 3/4) carry these on the text sub-config, so
        # resolve against the same sub-config the rest of this method already
        # reads num_attention_heads and head_dim from.
        layer_types, sliding_window_tokens = resolve_layer_types_and_window(
            hf_config, getattr(model_config, "hf_text_config", None)
        )
        if (
            is_draft
            and layer_types
            and len(layer_types) != model_config.num_attention_layers
        ):
            # Target-stack labels don't fit the draft depth; drop so the pool falls back to full attn
            layer_types = ()
        # Equal-depth cache groups, before the KV profile reads the slab count
        # off these labels. Gated on Gemma 4: every other hybrid here either is
        # already equal-depth (Inkling splits its own) or pairs attention with
        # a recurrent group, where the slab arithmetic is different and this
        # has not been measured.
        if (
            scheduler_ext_flat_kvcache()
            and resolve_architecture(hf_config) in _EQUAL_DEPTH_GROUP_ARCHITECTURES
        ):
            layer_types = split_groups_to_equal_depth(layer_types)
        conv_state_shape = temporal_state_shape = None
        conv_dtype = ssm_dtype = None
        if (
            any(label in STATE_LAYER_TYPES for label in layer_types)
            and scheduler_ext_flat_kvcache()
        ):
            # GDN hybrid on the flat ext: the KV pool owns the recurrent
            # state (state slabs), so it needs the mamba2 shapes/dtypes.
            # Radix branch untouched: SimpleMambaPool owns the state there.
            text_config = getattr(hf_config, "text_config", hf_config)
            (
                conv_state_shape,
                temporal_state_shape,
                conv_dtype,
                ssm_dtype,
                _,
            ) = text_config.mamba2_cache_params
        return cls(
            device=server_args.device,
            context_len=model_config.context_len,
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
            kv_cache_dtype=resolve_dtype(kv_cache_dtype),
            kv_cache_mxfp8=kv_cache_dtype == "mxfp8",
            page_size=server_args.block_size,
            max_bs=server_args.max_num_seqs
            // (server_args.data_parallel_size or server_args.mapping.attn.dp_size),
            max_graph_bs=server_args.max_cudagraph_capture_size,
            kv_cache_quant_method=server_args.kv_cache_quant_method,
            is_draft=is_draft,
            draft_block_decode=draft_block_decode,
            layer_types=layer_types,
            sliding_window_tokens=sliding_window_tokens,
            max_scheduled_tokens=getattr(server_args, "chunked_prefill_size", 8192),
            pd_disaggregation_enabled=getattr(
                server_args, "disaggregation_mode", "null"
            )
            != "null",
            conv_state_shape=conv_state_shape,
            temporal_state_shape=temporal_state_shape,
            conv_dtype=conv_dtype,
            ssm_dtype=ssm_dtype,
            **kwargs,
        )

    def cache_cell_size(self) -> int:
        cell = (
            max(self.num_kv_heads // self.attn_tp_size, 1)
            * self.head_dim
            * 2
            * torch._utils._element_size(self.kv_cache_dtype)
        )
        if self.kv_cache_mxfp8:
            # One UE8M0 byte per 32 fp8 data bytes.
            cell += cell // 32
        return cell

    def create_pool(
        self,
        num_layers: int,
        max_total_num_tokens: int,
        rank: int,
        enable_memory_saver: bool,
    ) -> BaseTokenToKVPool:
        from tokenspeed.runtime.layers.attention.kv_cache.mha import (
            MHATokenToKVPool,
            MHATokenToKVPoolMXFP8,
        )

        pool_cls = MHATokenToKVPool
        if self.kv_cache_mxfp8:
            assert self.page_size == 128, (
                "mxfp8 KV cache requires --block-size 128 (the attention "
                "kernel consumes the interleaved paged scale layout)"
            )
            pool_cls = MHATokenToKVPoolMXFP8

        return pool_cls(
            size=max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            head_num=max(self.num_kv_heads // self.attn_tp_size, 1),
            head_dim=self.head_dim,
            layer_num=num_layers,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            max_batch_size=self.max_bs,
            max_context_len=self.context_len,
            page_size=self.page_size,
            rank=rank,
            layer_types=self.layer_types,
            sliding_window_tokens=self.sliding_window_tokens,
            max_scheduled_tokens=self.max_scheduled_tokens,
            pd_disaggregation_enabled=self.pd_disaggregation_enabled,
            conv_state_shape=self.conv_state_shape,
            extra_paged_groups=self.extra_paged_groups,
            slot_tokens=self.slot_tokens,
            group_page_sizes=self.group_page_sizes,
            layer_kv_head_counts=self.layer_kv_head_counts,
            # Pre-TP width the slab rows are allocated at (head_num is its
            # per-rank shard) — the per-layer view normalization base.
            kv_alloc_head_count=self.num_kv_heads,
            temporal_state_shape=self.temporal_state_shape,
            conv_dtype=self.conv_dtype,
            ssm_dtype=self.ssm_dtype,
        )
