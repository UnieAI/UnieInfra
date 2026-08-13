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

import math
import os
from collections.abc import Sequence

import tokenspeed_kernel.ops.attention.flashinfer.gated_delta_rule  # noqa: F401
import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import ErrorClass, Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

platform = current_platform()

BatchDecodeWithPagedKVCacheWrapper = ErrorClass
BatchMLAPagedAttentionWrapper = ErrorClass
BatchPrefillWithPagedKVCacheWrapper = ErrorClass
BatchPrefillWithRaggedKVCacheWrapper = ErrorClass
cudnn_batch_prefill_with_kv_cache = error_fn
trtllm_batch_context_with_kv_cache = error_fn
trtllm_batch_decode_with_kv_cache = error_fn
trtllm_batch_decode_with_kv_cache_mla = error_fn
trtllm_ragged_attention_deepseek = error_fn

if platform.is_nvidia:
    from flashinfer.decode import (
        BatchDecodeWithPagedKVCacheWrapper,
        trtllm_batch_decode_with_kv_cache,
        trtllm_batch_decode_with_kv_cache_mla,
    )
    from flashinfer.prefill import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        cudnn_batch_prefill_with_kv_cache,
        trtllm_batch_context_with_kv_cache,
        trtllm_ragged_attention_deepseek,
    )

if platform.is_blackwell:
    from flashinfer.mla import (
        BatchMLAPagedAttentionWrapper,
        trtllm_batch_decode_with_kv_cache_mla,
    )


# ------------------------------------------------------------------------------
# Kernel registration
# ------------------------------------------------------------------------------

_workspace_buffer: torch.Tensor | None = None
_dsa_sparse_workspace_buffers: dict[torch.device, torch.Tensor] = {}
_DSA_SPARSE_WORKSPACE_BYTES = 384 * 1024 * 1024


def _get_dsa_sparse_workspace(device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    workspace = _dsa_sparse_workspace_buffers.get(device)
    if workspace is None:
        workspace = torch.zeros(
            _DSA_SPARSE_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        _dsa_sparse_workspace_buffers[device] = workspace
    return workspace


def _flashinfer_trtllm_mla_kv_cache(
    kv_cache: torch.Tensor,
    page_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if kv_cache.dtype != dtype:
        kv_cache = kv_cache.to(dtype)
    if kv_cache.dim() == 2:
        return kv_cache.view(-1, int(page_size), kv_cache.shape[-1]).unsqueeze(1)
    if kv_cache.dim() == 3 and kv_cache.shape[1] == 1:
        return (
            kv_cache.squeeze(1)
            .view(-1, int(page_size), kv_cache.shape[-1])
            .unsqueeze(1)
        )
    if kv_cache.dim() == 4:
        if kv_cache.shape[1] == int(page_size) and kv_cache.shape[2] == 1:
            return kv_cache.permute(0, 2, 1, 3).contiguous()
        if kv_cache.shape[1] == 1 and kv_cache.shape[2] == int(page_size):
            return kv_cache.contiguous()
    raise ValueError(
        "kv_cache must be flat [slots, dim], flat [slots, 1, dim], or paged "
        f"[pages, page_size, 1, dim] for FlashInfer/TRTLLM sparse MLA, got {tuple(kv_cache.shape)}"
    )


def _topk_lens_or_count(
    topk_slots: torch.Tensor, topk_lens: torch.Tensor | None
) -> torch.Tensor:
    if topk_lens is not None:
        return topk_lens.to(device=topk_slots.device, dtype=torch.int32).contiguous()
    return (topk_slots >= 0).sum(dim=-1, dtype=torch.int32).contiguous()


if platform.is_nvidia and platform.is_hopper_plus:

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="flashinfer_trtllm_mha_extend_with_kvcache",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            # Upper bound, because `min_arch_version` alone is an open-ended
            # version comparison that every newer part satisfies. This one is
            # backed by TllmGenFmhaRunner, whose kernels are prebuilt for
            # datacenter Blackwell; on sm_120 it passes selection and then dies
            # with "Unsupported architecture" from
            # flashinfer/trtllm/fmha/fmhaRunner.cuh at the first extend.
            #
            # A capability would be better than a version range -- what these
            # kernels actually need is tcgen05/TMEM, which sm_120 does not have
            # -- but `_get_cuda_sm_features` has no feature for it, so the bound
            # is the honest expression of what is known today.
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16, torch.float8_e4m3fn},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "is_causal": frozenset({False, True}),
            "head_dim": frozenset({64, 128, 256}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def flashinfer_trtllm_mha_extend_with_kvcache(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        is_causal: bool = False,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        global _workspace_buffer
        if _workspace_buffer is None:
            _workspace_buffer = torch.zeros(
                512 * 1024 * 1024,
                dtype=torch.uint8,
                device=q.device,
            )
        # TRTLLM kernels require fp32 sinks.
        if sinks is not None and sinks.dtype != torch.float32:
            sinks = sinks.to(torch.float32)

        return trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=(
                k_cache.permute(0, 2, 1, 3),
                v_cache.permute(0, 2, 1, 3),
            ),
            workspace_buffer=_workspace_buffer,
            block_tables=page_table,
            seq_lens=cache_seqlens,
            max_q_len=max_seqlen_q,
            max_kv_len=max_seqlen_k,
            bmm1_scale=softmax_scale,
            bmm2_scale=1.0,
            batch_size=cache_seqlens.shape[0],
            cum_seq_lens_q=cu_seqlens_q,
            cum_seq_lens_kv=cu_seqlens_kv,
            window_left=window_left,
            sinks=sinks,
            out_dtype=(torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype),
            causal=is_causal,
        )

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="flashinfer_trtllm_mha_decode_with_kvcache",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16, torch.float8_e4m3fn},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            # XQA is compiled with TOKENS_PER_PAGE as a template parameter and
            # rejects anything else outright (flashinfer/jit/xqa.py). Callers
            # already pass page_size to select_kernel, so declaring it here is
            # what keeps this kernel from winning selection for a page size it
            # cannot serve and then raising from inside JIT compilation.
            #
            # Hybrid attention/linear-attention models are the case that hits
            # this: their attention page size is derived from the linear-state
            # row rather than chosen, so it is routinely none of these four.
            "page_size": frozenset({16, 32, 64, 128}),
            # The same file requires head_dim to be a multiple of 16 in
            # [16, 256]. This used to be left undeclared, on the grounds that
            # a finite trait set would go stale against the rule -- but the
            # rule IS a finite set, so generate it rather than transcribe it,
            # and the sibling extend kernel above already declares its own.
            #
            # Gemma 4 is the model the old note said had not turned up yet:
            # its 10 full-attention layers are 512 wide, so selection handed
            # them this kernel and JIT raised `Invalid head_dim: 512`, taking
            # the server down at capture. Declaring it lets selection route
            # per layer instead -- the 50 sliding layers (256) keep XQA, the
            # 10 full ones fall through to Triton. vLLM cannot do that: it
            # decides once per model and forces every layer onto Triton
            # (`Gemma4 model has heterogeneous head dimensions ... forcing
            # TRITON_ATTN`).
            "head_dim": frozenset(range(16, 257, 16)),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def flashinfer_trtllm_mha_decode_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        max_seqlen_q: int = 1,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        global _workspace_buffer
        if _workspace_buffer is None:
            _workspace_buffer = torch.zeros(
                512 * 1024 * 1024,
                dtype=torch.uint8,
                device=q.device,
            )

        # TRTLLM kernels require fp32 sinks
        if sinks is not None and sinks.dtype != torch.float32:
            sinks = sinks.to(torch.float32)

        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(
                k_cache.permute(0, 2, 1, 3),
                v_cache.permute(0, 2, 1, 3),
            ),
            workspace_buffer=_workspace_buffer,
            block_tables=page_table,
            seq_lens=cache_seqlens,
            max_seq_len=max_seqlen_k,
            bmm1_scale=softmax_scale,
            bmm2_scale=1.0,
            window_left=window_left,
            sinks=sinks,
            out_dtype=(torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype),
            q_len_per_req=max_seqlen_q,
        )

    @register_kernel(
        "attention",
        "dsa_decode",
        name="flashinfer_trtllm_dsa_decode",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "kv_cache_available": frozenset({True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def flashinfer_trtllm_dsa_decode(
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        sparse_kv_cache: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor | None,
        max_seqlen_k: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        page_size: int,
        q_len_per_req: int = 1,
        logit_cap: float = 0.0,
        k_scale: float = 1.0,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kv_cache is None:
            raise RuntimeError("FlashInfer/TRTLLM sparse MLA requires kv_cache")
        if return_lse:
            raise RuntimeError(
                "FlashInfer/TRTLLM sparse MLA does not support return_lse"
            )
        if logit_cap != 0.0:
            raise RuntimeError(
                "FlashInfer/TRTLLM sparse MLA does not support logit_cap"
            )
        if q.dim() == 3:
            num_tokens = q.shape[0]
            q_kernel = q.view(num_tokens, 1, q.shape[1], q.shape[2])
        elif q.dim() == 4:
            num_tokens = q.shape[0] * q.shape[1]
            q_kernel = q.reshape(num_tokens, 1, q.shape[2], q.shape[3])
        else:
            raise ValueError(f"unsupported q shape {tuple(q.shape)}")
        kv_dtype = q.dtype if q.dtype == torch.float8_e4m3fn else kv_cache.dtype
        kv = _flashinfer_trtllm_mla_kv_cache(kv_cache, page_size, kv_dtype)
        seq_lens = _topk_lens_or_count(topk_slots, topk_lens)
        result = trtllm_batch_decode_with_kv_cache_mla(
            query=q_kernel,
            kv_cache=kv,
            workspace_buffer=_get_dsa_sparse_workspace(q.device),
            qk_nope_head_dim=int(qk_nope_head_dim),
            kv_lora_rank=int(kv_lora_rank),
            qk_rope_head_dim=int(qk_rope_head_dim),
            block_tables=topk_slots.view(num_tokens, 1, -1),
            seq_lens=seq_lens,
            max_seq_len=int(max_seqlen_k),
            sparse_mla_top_k=topk_slots.shape[-1],
            bmm1_scale=float(k_scale) * float(softmax_scale),
            backend="trtllm-gen",
        )
        result = result.reshape(num_tokens, q_kernel.shape[2], int(kv_lora_rank))
        if out is not None:
            out.reshape_as(result).copy_(result)
            return out
        return result

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="flashinfer_trtllm_dsa_prefill",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "kv_cache_available": frozenset({True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def flashinfer_trtllm_dsa_prefill(
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        sparse_kv_cache: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor | None,
        max_seqlen_k: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        page_size: int,
        q_len_per_req: int = 1,
        logit_cap: float = 0.0,
        k_scale: float = 1.0,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return flashinfer_trtllm_dsa_decode(
            q=q,
            kv_cache=kv_cache,
            sparse_kv_cache=sparse_kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            page_size=page_size,
            q_len_per_req=q_len_per_req,
            logit_cap=logit_cap,
            k_scale=k_scale,
            return_lse=return_lse,
            out=out,
        )


# ------------------------------------------------------------------------------
# Direct export
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Ragged (uncached-KV) prefill attention
# ------------------------------------------------------------------------------
#
# The trtllm-gen FMHA registrations above are bounded to sm10x because they need
# tcgen05. This one is the portable FA2-style path, which is what both SGLang
# (`srt/layers/attention/flashinfer_backend.py`) and TRT-LLM
# (`_torch/attention_backend/flashinfer.py`) call for ragged prefill -- and both
# cache the wrapper rather than constructing one per call, because `plan()` does
# device->host syncs. We do the same here.
#
# Why it matters on sm_120: our own FlashAttention registrations are bounded to
# the arch they were written for (fa3 -> sm90 exactly, fa4 -> sm100 exactly), so
# this part falls back to the Triton kernel. Measured on RTX PRO 6000, Qwen3-8B
# shapes (32 q heads, 8 kv heads, head_dim 128), device time:
#
#     tokens |  triton  |  flashinfer ragged |  vLLM FlashAttention
#        256 |  0.0594  |       0.0228       |        0.0174
#       1024 |  0.1304  |       0.0701       |        0.0644
#       4096 |  1.7335  |       0.5294       |        0.5341
#
# i.e. ~3.3x over the Triton fallback at 4096 and level with vLLM.

_ragged_prefill_wrappers: dict[tuple, "BatchPrefillWithRaggedKVCacheWrapper"] = {}
# wrapper id -> the plan params it currently holds, so a layer that asks for
# the same batch shape reuses the plan instead of re-syncing.
_ragged_prefill_plans: dict[int, tuple] = {}
_RAGGED_PREFILL_WORKSPACE_BYTES = 128 * 1024 * 1024


def _ragged_prefill_wrapper(device: torch.device):
    """One wrapper per device, reused across calls.

    Constructing a wrapper allocates its workspace; SGLang and TRT-LLM both keep
    them in a dict for this reason. `plan()` still runs per call because the
    ragged offsets change, but the workspace does not.
    """
    key = (device.type, device.index)
    wrapper = _ragged_prefill_wrappers.get(key)
    if wrapper is None:
        workspace = torch.empty(
            _RAGGED_PREFILL_WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        _ragged_prefill_wrappers[key] = wrapper
    return wrapper


def _resolve_ragged_run():
    """`BatchPrefillWithRaggedKVCacheWrapper.run` minus its API decorator.

    `run` is wrapped in `@flashinfer_api(trace=gqa_ragged_prefill_trace)`, whose
    per-call Python cost is what keeps this path behind vLLM's FlashAttention at
    short sequences -- the same overhead that made the fused GDN kernel look
    slower than the Triton scan (see gated_delta_rule.py). The decorator exposes
    the undecorated function, so use it and fall back if that ever changes.
    """
    run = getattr(BatchPrefillWithRaggedKVCacheWrapper, "run", None)
    if run is None:
        return None
    return getattr(run, "__wrapped__", run)


_ragged_run = _resolve_ragged_run()


def _ragged_prefill_available() -> bool:
    """Off by default until an end-to-end run backs the kernel-level numbers."""
    return (
        BatchPrefillWithRaggedKVCacheWrapper is not ErrorClass
        and os.getenv("TOKENSPEED_ENABLE_FI_RAGGED_PREFILL") == "1"
    )


if _ragged_prefill_available():

    @register_kernel(
        "attention",
        "mha_prefill",
        name="flashinfer_ragged_mha_prefill",
        solution="flashinfer",
        capability=CapabilityRequirement(
            # An open lower bound is correct *here* -- unlike the trtllm-gen
            # kernels above, this path is the portable FA2-style one and does not
            # depend on an arch-specific feature. Verified on sm_120; sm80..sm90
            # are flashinfer's own documented floor, not something measured here.
            min_arch_version=ArchVersion(8, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.PERFORMANT,
        traits={
            # plan() takes a window and a soft cap, but neither is exercised or
            # verified yet -- declare only what has been checked.
            "sliding_window": frozenset({False}),
            "support_sinks": frozenset({False}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def flashinfer_ragged_mha_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: list[int],
        max_seqlen: int,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        wrapper = _ragged_prefill_wrapper(q.device)

        # plan() is per *batch*, not per layer, and it does device->host syncs
        # (SGLang carries a whole `fast_prefill_plan` to avoid them; TRT-LLM
        # keeps a plan-params -> wrapper dict and plans outside stream capture).
        # Every layer of one forward shares the same ragged offsets and shapes,
        # so re-planning per layer serialises the host against the GPU for
        # nothing: measured on Qwen3-8B at 4096 tokens, host time went from
        # 2.2 ms (Triton, launches only) to 20.4 ms with a plan per layer, while
        # device time was 3x *better*. Key on the host-side offsets we are
        # already given, so the check itself costs no sync.
        plan_key = (
            tuple(cu_seqlens_cpu),
            q.shape[1],
            k.shape[1],
            q.shape[-1],
            v.shape[-1],
            softmax_scale,
            q.dtype,
            k.dtype,
        )
        if _ragged_prefill_plans.get(id(wrapper)) != plan_key:
            wrapper.plan(
                cu_seqlens,
                cu_seqlens,
                num_qo_heads=q.shape[1],
                num_kv_heads=k.shape[1],
                head_dim_qk=q.shape[-1],
                head_dim_vo=v.shape[-1],
                causal=True,
                sm_scale=softmax_scale,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
            )
            _ragged_prefill_plans[id(wrapper)] = plan_key

        # Allocate the output here and hand it over. Letting `run` allocate
        # costs 26% more per call at short sequences (0.556 -> 0.413 ms over 36
        # calls at 256 tokens), which is the whole margin at that size.
        out = torch.empty_like(q)
        return _ragged_run(wrapper, q, k, v, out=out, return_lse=return_lse)


# ------------------------------------------------------------------------------
# Paged extend (FA2) -- the kernel family vLLM prefills with
# ------------------------------------------------------------------------------
#
# Only two kernels were registered for `mha_extend_with_kvcache`: the trtllm-gen
# one, which is capability-bounded to <= sm_103 because it needs tcgen05, and a
# Triton fallback. So on sm_120 every prefill ran the Triton kernel, while vLLM
# ran FlashAttention. End to end on Qwen3-8B that is a 1.39x TTFT loss on a cold
# 5892-token prompt (381 vs 273 ms) -- and vLLM's 273 ms is within 4% of the
# pure GEMM device time for that prompt, i.e. it is at the compute roofline
# while we are 117 ms above it.
#
# This is SGLang's paged prefill path, ported as-is: the same
# `BatchPrefillWithPagedKVCacheWrapper` with `backend="fa2"`, planned per batch
# shape and run over the paged KV cache
# (`flashinfer_backend.py:1302 prefill_wrapper_paged.forward`). The plain
# `flash_attn` package that vLLM calls into is not installed here -- it is a
# namespace with only `cute` in it -- so flashinfer's wrapper around the same
# FA2 kernels is the closest available equivalent.

_paged_prefill_wrappers: dict[tuple, "BatchPrefillWithPagedKVCacheWrapper"] = {}
# 512 MB, matching the trtllm path above rather than the 128 MB the ragged
# sibling uses. flashinfer sizes batch_prefill_tmp_v from the batch and head
# geometry, and 128 MB is enough for an 8B at one request but overflows on
# Qwen3.5-27B at four -- "Buffer overflow when allocating memory for
# batch_prefill_tmp_v", raised from plan() rather than from the kernel, so the
# forward dies rather than degrading. A bug that only appears at both a larger
# model and a higher concurrency is exactly the kind an 8B single-request
# benchmark cannot find.
_PAGED_PREFILL_WORKSPACE_BYTES = 512 * 1024 * 1024

# Batches installed by the runtime for the forward that is about to run, each
# with a wrapper of its own, so the 60 layers of that forward do not each
# rebuild a plan. A list rather than a single entry because the flat KV path
# gives each cache group its own page table and a layer is served by its
# group's -- so there is one entry per group, not per forward. Its wrapper is
# planned on first use and re-planned only when the LAYER's geometry or scale
# changes; sharing a wrapper between two groups is the whole bug this
# arrangement exists to prevent. Matched by tensor identity, scanned linearly
# (groups number in the low single digits). See `install_paged_extend_plan`.
_installed_paged_plans: list[dict] = []

# Wrapper slot for calls that arrive with NO installed plan. It must never
# collide with an installed group's slot (0, 1, 2, ...): planning a wrapper
# overwrites whatever plan it held, so a fallback call that borrowed slot 0
# would silently re-point group 0's installed plan at its own pages, and the
# next group-0 layer would then attend over them. Negative slots are unreachable
# from `len(_installed_paged_plans)`.
_PAGED_FALLBACK_SLOT = -1


def _paged_prefill_wrapper(device: torch.device, slot: int = 0):
    """One wrapper (and one workspace) per device and plan slot.

    A wrapper holds one plan at a time, so the flat KV path -- which plans one
    page table per state group -- needs one wrapper per group. `slot` is the
    group's position in the install order, which is stable across forwards.
    """
    key = (device.type, device.index, slot)
    wrapper = _paged_prefill_wrappers.get(key)
    if wrapper is None:
        workspace = torch.empty(
            _PAGED_PREFILL_WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
        _paged_prefill_wrappers[key] = wrapper
    return wrapper


def _resolve_paged_run():
    """`BatchPrefillWithPagedKVCacheWrapper.run` minus its API decorator.

    Same reason as `_resolve_ragged_run`: the `@flashinfer_api` wrapper's
    per-call Python cost is a large fraction of the kernel at short sequences.
    """
    run = getattr(BatchPrefillWithPagedKVCacheWrapper, "run", None)
    if run is None:
        return None
    return getattr(run, "__wrapped__", run)


_paged_run = _resolve_paged_run()


def _paged_prefill_available() -> bool:
    """Off by default until an end-to-end run backs the kernel-level numbers."""
    return (
        BatchPrefillWithPagedKVCacheWrapper is not ErrorClass
        and os.getenv("TOKENSPEED_ENABLE_FI_PAGED_EXTEND") == "1"
    )


def _paged_plan_key(
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    is_causal,
    window_left,
    q_dtype,
    kv_dtype,
):
    """The exactly-comparable part of a plan's identity.

    Floats live outside this on purpose -- see the tolerance check at the match
    site.
    """
    return (
        int(num_qo_heads),
        int(num_kv_heads),
        int(head_dim),
        int(page_size),
        bool(is_causal),
        int(window_left),
        q_dtype,
        kv_dtype,
    )


def _plan_paged_extend(
    wrapper,
    *,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens_cpu: Sequence[int],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    is_causal: bool,
    softmax_scale: float,
    window_left: int,
    logit_cap: float,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> None:
    """Plan `wrapper` for one page table at one layer geometry.

    `page_size` is the LAYER'S, not the pool's: a heterogeneous-KV model (Gemma
    4) pages its full-attention group at a different granularity from its
    sliding groups, so the page count per request is a function of the caller's
    cache view and cannot be precomputed from a pool-wide constant.

    Everything here is derived from `seq_lens_cpu` (host-side) and `page_table`
    (device-side, gathered without a read-back), so no call syncs.
    """
    pages_per_seq = [(int(n) + page_size - 1) // page_size for n in seq_lens_cpu]
    kv_indptr = [0]
    for n in pages_per_seq:
        kv_indptr.append(kv_indptr[-1] + n)
    last_page_len = [
        max(1, int(n) - (p - 1) * page_size) if p else 1
        for n, p in zip(seq_lens_cpu, pages_per_seq)
    ]
    device = page_table.device
    kv_indices = torch.cat([page_table[i, :p] for i, p in enumerate(pages_per_seq)]).to(
        torch.int32
    )
    # The wrapper's internal buffers are allocated on first use. When the caller
    # plans from `init_forward_metadata` that happens outside inference mode,
    # and writing to an inference tensor from normal mode raises; entering it
    # here makes both call sites agree.
    with torch.inference_mode():
        wrapper.plan(
            cu_seqlens_q.to(torch.int32),
            torch.tensor(kv_indptr, dtype=torch.int32, device=device),
            kv_indices,
            torch.tensor(last_page_len, dtype=torch.int32, device=device),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=is_causal,
            sm_scale=softmax_scale,
            window_left=window_left,
            logits_soft_cap=logit_cap,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
        )


def _run_paged_extend(
    wrapper,
    q: torch.Tensor,
    total_q: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    return_lse: bool,
    k_scale,
    v_scale,
):
    """Run a planned wrapper over `q`, respecting the caller's padded shape.

    Preallocating the output rather than letting `run` do it is worth 26% per
    call at short sequences on the ragged sibling; same reason here.
    """
    out = torch.empty_like(q)
    result = _paged_run(
        wrapper,
        q[:total_q] if total_q != q.shape[0] else q,
        (k_cache, v_cache),
        out=out[:total_q],
        return_lse=return_lse,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    if total_q == q.shape[0]:
        return result
    # `out` is the padded-size buffer the caller expects; `result` is the view
    # `run` wrote into, which aliases it.
    if return_lse:
        return out, result[1]
    return out


def paged_extend_planning_enabled() -> bool:
    """Whether a caller should bother installing a per-forward plan."""
    return _paged_prefill_available()


def clear_paged_extend_plan() -> None:
    """Drop any installed plan.

    The runtime calls this at the start of every extend forward, so a plan can
    never outlive the forward it was built for -- which is what makes the
    identity check in the kernel sufficient.
    """
    _installed_paged_plans.clear()


def install_paged_extend_plan(
    *,
    cu_seqlens_q: torch.Tensor,
    cache_seqlens: torch.Tensor,
    seq_lens_cpu: Sequence[int],
    page_table: torch.Tensor,
) -> None:
    """Reserve a wrapper for one page table of the forward about to run.

    Without this the kernel plans per call, and the expensive part is the
    `cache_seqlens.to("cpu")` needed to describe the batch. On CUDA that
    synchronises, so every layer drains the queue that the previous layers just
    filled, serialising exactly the CPU/GPU overlap the engine is built around.
    SGLang avoids it by planning in `init_forward_metadata`
    (`flashinfer_backend.py`), which is what this exposes.

    `seq_lens_cpu` must be the host-side total KV length per request. The runtime
    already has it (prefix + extend, both CPU-side in the extend metadata), so
    computing it here costs no sync.

    What is installed is the batch description and a wrapper of its own, NOT a
    finished plan. The per-layer geometry (head count, head dim, page size) and
    the softmax scale, window and soft cap are the LAYER'S, and a caller that
    plans a forward cannot know them: a heterogeneous-KV model has more than one
    geometry across the layers sharing this forward (Gemma 4: 16 x 256 at page
    16 on its sliding groups, 4 x 512 at page 32 on its full-attention one), and
    the scale is a model constant the backend does not carry (Gemma 4 declares
    1.0, not `head_dim ** -0.5`). So the first layer to use a page table plans
    its wrapper with the values it was actually handed, and the remaining layers
    of that group reuse it. Planning from `seq_lens_cpu` costs no sync wherever
    it happens, which is what makes deferring it free.

    The kernel accepts the installed entry only when the tensors it was handed
    are the very objects named here (`is`, not equality), so an entry can never
    be applied to a different batch -- and, critically, never to a different
    page table: one wrapper per table is what keeps one cache group's pages out
    of another's attention.
    """
    if not _paged_prefill_available():
        _installed_paged_plans.clear()
        return

    _installed_paged_plans.append(
        {
            "wrapper": _paged_prefill_wrapper(
                page_table.device, len(_installed_paged_plans)
            ),
            "cu_seqlens_q": cu_seqlens_q,
            "cache_seqlens": cache_seqlens,
            "page_table": page_table,
            "seq_lens_cpu": [int(n) for n in seq_lens_cpu],
            "total_q": int(cu_seqlens_q[-1]) if cu_seqlens_q.numel() else 0,
            # Filled in by the first layer that uses this table; see above.
            "planned": None,
        }
    )


if _paged_prefill_available():

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="flashinfer_paged_mha_extend_with_kvcache",
        solution="flashinfer_fa2",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(8, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.PERFORMANT,
        traits={
            "is_causal": frozenset({False, True}),
            "head_dim": frozenset({64, 128, 256}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    )
    def flashinfer_paged_mha_extend_with_kvcache(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        is_causal: bool = False,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sinks is not None:
            raise NotImplementedError("fa2 paged extend does not support sinks")
        if q_scale is not None:
            raise NotImplementedError("fa2 paged extend does not support q_scale")

        page_size = k_cache.shape[1]
        num_qo_heads = q.shape[1]
        num_kv_heads = k_cache.shape[2]
        head_dim = q.shape[-1]
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        shape_key = _paged_plan_key(
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            is_causal,
            window_left,
            q.dtype,
            k_cache.dtype,
        )
        # Fast path: the runtime reserved a wrapper for this batch and this page
        # table. Identity checks only -- no `.tolist()`, no `.to("cpu")`, so no
        # sync. The runtime installs or clears on every extend forward, so a
        # match here cannot be a leftover from an earlier one.
        #
        # The match is on the BATCH (which rows, which pages), never on the
        # layer's geometry or scale: those belong to the caller and are re-planned
        # below whenever they change. Folding them into the match instead sends
        # any layer that disagrees with the installed values down the per-call
        # path -- which is where the pages got crossed (see there).
        for installed in _installed_paged_plans:
            if not (
                installed["cu_seqlens_q"] is cu_seqlens_q
                and installed["cache_seqlens"] is cache_seqlens
                and installed["page_table"] is page_table
            ):
                continue
            want = (shape_key, float(softmax_scale), float(logit_cap))
            if installed["planned"] != want:
                _plan_paged_extend(
                    installed["wrapper"],
                    cu_seqlens_q=cu_seqlens_q,
                    page_table=page_table,
                    seq_lens_cpu=installed["seq_lens_cpu"],
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    is_causal=is_causal,
                    softmax_scale=softmax_scale,
                    window_left=window_left,
                    logit_cap=logit_cap,
                    q_dtype=q.dtype,
                    kv_dtype=k_cache.dtype,
                )
                installed["planned"] = want
            return _run_paged_extend(
                installed["wrapper"],
                q,
                installed["total_q"],
                k_cache,
                v_cache,
                return_lse,
                k_scale,
                v_scale,
            )

        # No reservation for this batch: describe it here instead. `cache_seqlens`
        # is on device, so this costs one host copy -- a sync per layer, which is
        # why SGLang builds it once per forward in `init_forward_metadata` and why
        # `install_paged_extend_plan` exists.
        #
        # This path plans EVERY call, and it must. A plan is a page list, and the
        # only cheap description of a page list is the table it came from -- which
        # is a device tensor, so nothing that can be compared here without another
        # sync. A cache keyed on the batch's SHAPE instead (page counts, last-page
        # fills, query offsets, geometry) collides between any two page tables of
        # equal shape, and the flat KV path hands out one table per cache group:
        # gemma-4-31B publishes six (five sliding sub-groups plus full_attention),
        # all with identical page counts for a given request. Under that key the
        # first group planned the shared wrapper and the other four ran on ITS
        # pages -- fluent output, entirely the wrong sequence, nothing raised.
        # Measured directly: two tables over disjoint pages, same shape, gave
        # bit-identical outputs where the Triton kernel differed by 2.78.
        #
        # The runtime hands over a q padded up to a graph bucket and relies on
        # the kernel ignoring the tail (see `_scrub_extend_padding` in the MHA
        # backend); the trtllm and Triton kernels do. This one asserts
        # `q.shape[0] == qo_indptr[-1]` instead, so trim to the real tokens and
        # give the padded rows back untouched.
        total_q = int(cu_seqlens_q[-1]) if cu_seqlens_q.numel() else 0
        seqlens_cpu = cache_seqlens.to("cpu", non_blocking=False)
        # A dedicated wrapper: planning one overwrites the plan it held, so
        # borrowing an installed group's slot would re-point that group at this
        # call's pages.
        wrapper = _paged_prefill_wrapper(q.device, _PAGED_FALLBACK_SLOT)
        _plan_paged_extend(
            wrapper,
            cu_seqlens_q=cu_seqlens_q,
            page_table=page_table,
            seq_lens_cpu=seqlens_cpu.tolist(),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            window_left=window_left,
            logit_cap=logit_cap,
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
        )
        return _run_paged_extend(
            wrapper, q, total_q, k_cache, v_cache, return_lse, k_scale, v_scale
        )


__all__ = [
    "BatchDecodeWithPagedKVCacheWrapper",
    "BatchMLAPagedAttentionWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchPrefillWithRaggedKVCacheWrapper",
    "cudnn_batch_prefill_with_kv_cache",
    "trtllm_batch_context_with_kv_cache",
    "trtllm_batch_decode_with_kv_cache",
    "trtllm_batch_decode_with_kv_cache_mla",
    "trtllm_ragged_attention_deepseek",
]
