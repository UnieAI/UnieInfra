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

"""FlashInfer FA2 paged attention for MHA extend/decode.

The ``flashinfer_trtllm_*`` kernels in this package use the TRT-LLM-gen fused
MHA runner, which only supports datacenter Blackwell (sm_100 / sm_103) and hard
errors on other architectures (notably sm_120 / RTX PRO 6000). This module wires
the same ``mha_extend_with_kvcache`` / ``mha_decode_with_kvcache`` operators to
FlashInfer's FA2 paged wrappers, which JIT-compile for the running architecture
(sm_80+, including sm_120) and therefore provide a working FlashInfer attention
path on consumer / workstation Blackwell.

Both kernels share the same paged-KV contract as the rest of the attention
family:

* ``k_cache`` / ``v_cache``: ``[num_pages, page_size, num_kv_heads, head_dim]``
  (this is exactly FlashInfer's ``NHD`` paged layout).
* ``page_table``: ``[batch, max_pages_per_seq]`` of physical page ids.
* ``cache_seqlens``: ``[batch]`` total visible KV length per request.

FlashInfer's wrappers instead take the CSR-style ``(paged_kv_indptr,
paged_kv_indices, paged_kv_last_page_len)`` description, which we derive from
``page_table`` + ``cache_seqlens`` on each call.
"""

from __future__ import annotations

import math
import os

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()

if platform.is_nvidia:
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    # FlashInfer sub-allocates its prefill temporaries out of this buffer via an
    # AlignedAllocator. At 128 MiB, Qwen3-scale prefill asked for 136,052,736 B
    # against 134,217,728 B available -- short by 1.75 MiB -- and aborted startup
    # with "Increase the workspace buffer size". That made the flashinfer
    # attention backend look unsupported on sm_120 when it was merely
    # unconfigured; raising it lets the backend serve. This is the sm_120 path
    # specifically: these fa2 kernels register with min_arch_version=(8, 0) and
    # are the only solution="flashinfer" MHA kernels sm_120 can reach, since the
    # trtllm-gen ones are capped at 10.3.
    #
    # 512 MiB matches the siblings in this package (see __init__.py, which uses
    # 512 MiB for its own wrappers). The env var mirrors
    # tokenspeed.runtime.utils.flashinfer_config, but is read directly rather
    # than imported: tokenspeed-kernel is the kernel-package boundary and must
    # not depend back on tokenspeed.runtime (AGENTS.md, "Dependency boundaries").
    _FA2_WORKSPACE_BYTES = int(
        os.environ.get("FLASHINFER_WORKSPACE_SIZE", 512 * 1024 * 1024)
    )

    # Keyed by (device, window_left), not device alone.
    #
    # A FlashInfer wrapper records ``window_left`` at plan() time and asserts the
    # value passed to run() matches:
    #
    #   File flashinfer/decode.py, in run
    #     assert window_left == self._window_left
    #   AssertionError
    #
    # One wrapper per device is fine while every layer shares a window. gemma-4
    # does not: 50 sliding layers at 1024 and 10 global layers with no window, so
    # whichever planned last loses. Each distinct window needs its own wrapper,
    # and its own workspace with it, since the wrapper keeps plan state there.
    # Cost is 128 MiB per (device, window) rather than per device.
    #
    # NOT SUFFICIENT ON ITS OWN, and gemma-4 still fails with that assertion.
    # These factories only serve the eager path. Under CUDA graphs the wrapper is
    # created and owned by the caller -- see
    # runtime/layers/attention/backends/mha.py, `_fa2_decode_wrappers`, which is
    # keyed by batch size and shared across every layer.
    #
    # The machinery to fix it already exists there: the "flat" path builds one
    # wrapper per attention group (`_fa2_flat_decode_wrappers[(bs, gid)]`), which
    # is exactly the right shape for heterogeneous layers. gemma-4 simply does
    # not reach it -- `_fa2_flat_enabled` is only set when `att_gids` is
    # non-empty. Routing gemma-4 through the flat path, or keying the non-flat
    # wrapper by window as well, is where the real fix belongs.
    #
    # Left in place because it is independently correct for the eager path, but
    # it fixes nothing observable today. gemma-4 is not blocked by this: it runs
    # correctly on vLLM's TRITON_ATTN backend.
    _WrapperKey = tuple[torch.device, int]

    _prefill_workspace: dict[_WrapperKey, torch.Tensor] = {}
    _decode_workspace: dict[_WrapperKey, torch.Tensor] = {}
    _prefill_wrappers: dict[_WrapperKey, BatchPrefillWithPagedKVCacheWrapper] = {}
    _decode_wrappers: dict[_WrapperKey, BatchDecodeWithPagedKVCacheWrapper] = {}

    def _workspace(cache: dict, key: _WrapperKey) -> torch.Tensor:
        ws = cache.get(key)
        if ws is None:
            ws = torch.empty(
                _FA2_WORKSPACE_BYTES, dtype=torch.uint8, device=key[0]
            )
            cache[key] = ws
        return ws

    def _prefill_wrapper(
        device: torch.device, window_left: int = -1
    ) -> BatchPrefillWithPagedKVCacheWrapper:
        key = (device, window_left)
        wrapper = _prefill_wrappers.get(key)
        if wrapper is None:
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                _workspace(_prefill_workspace, key),
                kv_layout="NHD",
                backend="fa2",
            )
            _prefill_wrappers[key] = wrapper
        return wrapper

    def _decode_wrapper(
        device: torch.device, window_left: int = -1
    ) -> BatchDecodeWithPagedKVCacheWrapper:
        key = (device, window_left)
        wrapper = _decode_wrappers.get(key)
        if wrapper is None:
            try:
                wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    _workspace(_decode_workspace, key),
                    kv_layout="NHD",
                    use_tensor_cores=True,
                    backend="fa2",
                )
            except (TypeError, ValueError):
                # Older/newer signatures may not accept ``backend`` for decode;
                # "auto" resolves to the FA2 decode kernel on sm_120.
                wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    _workspace(_decode_workspace, key),
                    kv_layout="NHD",
                    use_tensor_cores=True,
                )
            _decode_wrappers[key] = wrapper
        return wrapper

    def _paged_kv_csr(
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert (page_table, cache_seqlens) to FlashInfer CSR paged layout.

        Returns ``(indptr, indices, last_page_len)`` where ``indptr`` has shape
        ``[batch + 1]``, ``indices`` is the flattened list of physical page ids
        (concatenated per request), and ``last_page_len`` is the number of valid
        tokens in each request's final page, in ``[1, page_size]``.
        """
        cache_seqlens = cache_seqlens.to(torch.int32)
        num_pages = (cache_seqlens + (page_size - 1)) // page_size
        # Build indptr = [0, cumsum(num_pages)...] entirely on-device via a pad,
        # avoiding the scalar `indptr[0] = 0` host->device copy that is illegal
        # under CUDA-graph capture. This function still uses a data-dependent
        # boolean gather below, so it must only be called OUTSIDE a captured
        # region (from the plan/replay hooks); the device-native indptr keeps it
        # from failing on the specific scalar-copy op if it is ever reached.
        indptr = torch.nn.functional.pad(
            torch.cumsum(num_pages, dim=0, dtype=torch.int32), (1, 0)
        )
        batch, max_pages = page_table.shape
        cols = torch.arange(max_pages, device=page_table.device).unsqueeze(0)
        mask = cols < num_pages.unsqueeze(1)
        indices = page_table[mask].to(torch.int32)
        last_page_len = (cache_seqlens - (num_pages - 1) * page_size).to(torch.int32)
        return indptr, indices, last_page_len

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="flashinfer_fa2_mha_extend_with_kvcache",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(8, 0),
            required_features=frozenset({"memory:async_copy"}),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        # Below trtllm-gen (SPECIALIZED) so datacenter Blackwell keeps trtllm,
        # above Triton (PORTABLE) so this wins on sm_120 where trtllm is gated out.
        priority=Priority.PERFORMANT,
        traits={
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    )
    def flashinfer_fa2_mha_extend_with_kvcache(
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        page_size = k_cache.shape[1]
        num_qo_heads = q.shape[1]
        num_kv_heads = k_cache.shape[2]
        head_dim = q.shape[-1]

        indptr, indices, last_page_len = _paged_kv_csr(
            page_table, cache_seqlens, page_size
        )
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        # Under graph mode the caller pads q with tail rows beyond the real
        # token count (cu_seqlens_q[-1]); Triton/trtllm bound work by cu_seqlens
        # and ignore them, but FlashInfer's prefill wrapper requires
        # q.shape[0] == qo_indptr[-1]. Extend is never cuda-graph captured
        # (the capture hook rejects extend/mixed), so slicing off the padding
        # with a host sync here is safe, and we scatter the result back.
        total_real = int(cu_seqlens_q[-1].item())
        q_run = q[:total_real] if total_real != q.shape[0] else q
        wrapper = _prefill_wrapper(q.device, window_left)
        wrapper.plan(
            qo_indptr=cu_seqlens_q,
            paged_kv_indptr=indptr,
            paged_kv_indices=indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=is_causal,
            sm_scale=softmax_scale,
            window_left=window_left,
            logits_soft_cap=(logit_cap if logit_cap != 0.0 else None),
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
        )
        out_run = wrapper.run(
            q_run,
            (k_cache, v_cache),
            sinks=sinks,
            return_lse=return_lse,
        )
        if q_run is q:
            return out_run
        # Scatter back into a full-length (padded) output; tail rows are
        # discarded downstream but are zeroed to avoid propagating NaNs.
        if return_lse:
            out_run, lse_run = out_run
        out = q.new_zeros((q.shape[0], out_run.shape[1], out_run.shape[2]))
        out[:total_real] = out_run
        if return_lse:
            lse = q.new_zeros((q.shape[0], lse_run.shape[1]), dtype=lse_run.dtype)
            lse[:total_real] = lse_run
            return out, lse
        return out

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="flashinfer_fa2_mha_decode_with_kvcache",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(8, 0),
            required_features=frozenset({"memory:async_copy"}),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.PERFORMANT,
        traits={
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    )
    def flashinfer_fa2_mha_decode_with_kvcache(
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        page_size = k_cache.shape[1]
        num_qo_heads = q.shape[1]
        num_kv_heads = k_cache.shape[2]
        head_dim = q.shape[-1]

        indptr, indices, last_page_len = _paged_kv_csr(
            page_table, cache_seqlens, page_size
        )

        if max_seqlen_q > 1:
            # Speculative-decode / MTP verify: several uniformly packed query
            # tokens per request, causal among themselves. The dedicated decode
            # wrapper's q_len_per_req path uses different masking semantics than
            # this operator, so route multi-token decode through the prefill
            # wrapper with a uniform qo_indptr, which matches exactly.
            batch = cache_seqlens.shape[0]
            qo_indptr = torch.arange(
                0,
                (batch + 1) * max_seqlen_q,
                max_seqlen_q,
                dtype=torch.int32,
                device=q.device,
            )
            wrapper = _prefill_wrapper(q.device, window_left)
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=indptr,
                paged_kv_indices=indices,
                paged_kv_last_page_len=last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                causal=True,
                sm_scale=softmax_scale,
                window_left=window_left,
                logits_soft_cap=(logit_cap if logit_cap != 0.0 else None),
                q_data_type=q.dtype,
                kv_data_type=k_cache.dtype,
            )
            return wrapper.run(
                q,
                (k_cache, v_cache),
                sinks=sinks,
                return_lse=return_lse,
            )

        wrapper = _decode_wrapper(q.device, window_left)
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            sm_scale=softmax_scale,
            window_left=window_left,
            logits_soft_cap=(logit_cap if logit_cap != 0.0 else None),
        )
        return wrapper.run(
            q,
            (k_cache, v_cache),
            sinks=sinks,
            return_lse=return_lse,
        )

    # ------------------------------------------------------------------
    # CUDA-graph plan/run split for FA2 decode.
    #
    # FlashInfer's wrappers require plan() (host-side split-KV scheduling +
    # data-dependent CSR index building) to run OUTSIDE the captured region,
    # once per capture and once per replay, with run() captured inside the
    # graph reading pre-planned fixed buffers. These helpers let the MHA
    # attention backend drive that lifecycle for the decode-graph hot path
    # without pulling FlashInfer specifics into the backend. They are only
    # engaged when FA2 is the selected flashinfer decode kernel (i.e. sm_120);
    # on sm_100/sm_103 the trtllm-gen decode kernel is monolithic and stays on
    # the normal single-call op path, so nothing here touches it.
    # ------------------------------------------------------------------

    def fa2_decode_is_selected(dtype: torch.dtype) -> bool:
        """Whether flashinfer_fa2_mha_decode_with_kvcache wins selection for
        the flashinfer backend at this platform + dtype (True on sm_120,
        False on sm_100/sm_103 where trtllm-gen wins)."""
        if dtype not in (torch.float16, torch.bfloat16):
            return False
        try:
            sig = next(
                iter(format_signatures(("q", "k_cache", "v_cache"), "dense", {dtype}))
            )
            kernel = select_kernel(
                "attention",
                "mha_decode_with_kvcache",
                sig,
                traits={
                    "sliding_window": False,
                    "support_logit_cap": False,
                    "support_sinks": False,
                    "return_lse": False,
                },
                solution="flashinfer",
            )
            return kernel.name == "flashinfer_fa2_mha_decode_with_kvcache"
        except Exception:
            return False

    def make_fa2_decode_cuda_graph_wrapper(
        workspace: torch.Tensor,
        indptr_buffer: torch.Tensor,
        indices_buffer: torch.Tensor,
        last_page_len_buffer: torch.Tensor,
        use_tensor_cores: bool = True,
    ) -> BatchDecodeWithPagedKVCacheWrapper:
        """Construct a cuda-graph-capturable FA2 decode wrapper bound to fixed
        (graph-stable) CSR buffers."""
        kwargs = dict(
            kv_layout="NHD",
            use_cuda_graph=True,
            use_tensor_cores=use_tensor_cores,
            paged_kv_indptr_buffer=indptr_buffer,
            paged_kv_indices_buffer=indices_buffer,
            paged_kv_last_page_len_buffer=last_page_len_buffer,
        )
        try:
            return BatchDecodeWithPagedKVCacheWrapper(
                workspace, backend="fa2", **kwargs
            )
        except (TypeError, ValueError):
            return BatchDecodeWithPagedKVCacheWrapper(workspace, **kwargs)

    def fa2_decode_cuda_graph_plan(
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        indptr_buffer: torch.Tensor,
        indices_buffer: torch.Tensor,
        last_page_len_buffer: torch.Tensor,
        window_left: int = -1,
        logit_cap: float = 0.0,
    ) -> None:
        """Build CSR from (page_table, cache_seqlens), copy it into the fixed
        buffers, and (re)plan. Runs OUTSIDE the captured region."""
        bs = cache_seqlens.shape[0]
        indptr, indices, last_page_len = _paged_kv_csr(
            page_table, cache_seqlens.clamp_min(1), page_size
        )
        n = indices.shape[0]
        indptr_buffer[: bs + 1].copy_(indptr)
        indices_buffer[:n].copy_(indices)
        last_page_len_buffer[:bs].copy_(last_page_len)
        wrapper.plan(
            indptr_buffer[: bs + 1],
            indices_buffer[:n],
            last_page_len_buffer[:bs],
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            window_left=window_left,
            logits_soft_cap=(logit_cap if logit_cap != 0.0 else None),
        )

    def fa2_decode_cuda_graph_run(
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        sinks: torch.Tensor | None = None,
        window_left: int = -1,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Captured inside the graph; reads the pre-planned fixed buffers."""
        return wrapper.run(
            q,
            (k_cache, v_cache),
            sinks=sinks,
            window_left=(window_left if window_left >= 0 else None),
            out=out,
        )
