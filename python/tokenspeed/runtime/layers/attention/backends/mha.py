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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_plan,
    mha_prefill,
)
from tokenspeed_kernel.ops.attention.flashinfer import (
    clear_paged_extend_plan,
    install_paged_extend_plan,
    paged_extend_planning_enabled,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    fused_fp8_set_kv_buffer,
    gather_page_table_with_padding,
)
from tokenspeed_kernel.ops.quantization import quantize_mxfp8

# FA2 cuda-graph plan/run split helpers. Present only on NVIDIA (the helpers
# live inside flashinfer.fa2's `if platform.is_nvidia` block); import guarded so
# non-NVIDIA builds and any import failure leave the split cleanly disabled.
try:
    from tokenspeed_kernel.ops.attention.flashinfer.fa2 import (
        fa2_decode_cuda_graph_plan,
        fa2_decode_cuda_graph_run,
        fa2_decode_is_selected,
        make_fa2_decode_cuda_graph_wrapper,
    )

    _HAS_FA2_GRAPH = True
except Exception:
    _HAS_FA2_GRAPH = False

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import scrub_padding_tail
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.flat_groups import (
    FlatCacheGroupsMixin,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import build_page_table
from tokenspeed.runtime.utils.common import ceil_div
from tokenspeed.runtime.utils.flashinfer_config import get_flashinfer_workspace_size

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


_KERNEL_SOLUTION_BY_BACKEND = {
    # None means "no solution constraint": selection ranks every registered
    # kernel by priority, which is what you want unless you are pinning one for
    # a comparison. On sm_120 that lands on the fa2 paged extend kernel.
    "mha": None,
    "flashinfer_fa2": "flashinfer_fa2",
    "fa3": "fa3",
    "fa4": "fa4",
    "triton": "triton",
    # Solution names denote lineage, not algorithm: "fa3"/"fa4" are Tri Dao's
    # flash_attn package (ops/attention/flash_attn/, importing flash_attn.cute),
    # while everything flashinfer-backed is prefixed "flashinfer". So the FA2
    # *algorithm* as implemented by flashinfer is "flashinfer_fa2", not "fa2" --
    # asking for "fa2" should not hand you a different library's kernel.
    #
    # Bare "flashinfer" pins the solution of that name, which for
    # mha_extend_with_kvcache is only the trtllm-gen kernel -- capability-bounded
    # to <= sm_103 because it needs tcgen05, so it fails selection on sm_120.
    "flashinfer": "flashinfer",
}


def mixed_batch_supported(
    spec_num_tokens: int, is_draft: bool, draft_block_decode: bool
) -> bool:
    """Whether an MHA backend with this spec configuration can serve MIXED.

    The MIXED decode tail is served as exactly one query token per row (see
    ``MHAAttnBackend._forward_mixed``). That holds for the plain
    decode plan, whose ``decode_input_tokens`` is 1, but not for a speculative
    build: there each decode row is a ``spec_num_tokens``-wide verify window, and
    a draft rank additionally re-uses the extend metadata as its own decode
    metadata. Those combinations stay unsupported so the startup guard rejects
    ``--enable-mixed-batch`` instead of a step failing mid-serve.

    Args:
        spec_num_tokens: Speculative draft tokens per step (1 when off).
        is_draft: Whether this backend instance belongs to a draft rank.
        draft_block_decode: Whether DFLASH block decode is active.

    Returns:
        ``True`` when MIXED batches are safe for this configuration.
    """
    return not (spec_num_tokens > 1 or is_draft or draft_block_decode)


def _scrub_extend_padding(metadata, q, k, v) -> None:
    """Zero the q/k/v rows beyond the real (unpadded) token count under a prefill graph.

    Reads the count from the pinned CPU cu-seqlens mirror (sync-free) and delegates the
    zeroing to the shared prefill-graph padding helper. No-op on normal unpadded forwards.
    """
    scrub_padding_tail(metadata.cu_extend_seq_lens_cpu[-1], q, k, v)


@dataclass(kw_only=True)
class MHAExtendMetadata:
    # Device-side metadata:
    # - seq_lens: total length after this step
    # - extend_seq_lens: length of new tokens
    #   cu_extend_seq_lens: the cumsum version of extend_seq_lens
    #   cu_seqlens_kv: the cumsum version of seq_lens
    # - extend_prefix_lens: length of the cached prefix tokens
    # seq_lens[i] = extend_prefix_lens[i] + extend_seq_lens[i]
    # page_table is None on the flat path (per-group page_tables route reads).
    page_table: torch.Tensor | None
    seq_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cu_extend_seq_lens: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens_cpu: list[int]
    cu_extend_seq_lens_cpu: list[int]
    max_extend_seq_len: int
    max_extend_prefix_len: int = 0
    # Flat per-group page tables (group_id -> [num_reqs, max_pages]); None on
    # the single-table path. TODO(radix-removal): drop the single page_table.
    page_tables: dict[str, torch.Tensor] | None = None
    # Flat per-group KV write locations (group_id -> [num_tokens] int32),
    # built with page_tables — same groups, same lifecycle.
    out_cache_locs: dict[str, torch.Tensor] | None = None
    # MIXED batches only (all three are 0 on EXTEND). Every field above then
    # describes the PREFILL segment alone -- rows [0, num_extends) -- and the
    # decode tail lives in the sibling MHADecodeMetadata, because the two
    # segments are served by different kernels. See _forward_mixed for why.
    # mixed_prefill_tokens is the query-token count of the prefill segment,
    # i.e. where the decode tail starts in the packed q/k/v.
    mixed_prefill_tokens: int = 0
    mixed_decode_rows: int = 0
    # Real (unpadded) token count of the WHOLE batch, for the padding scrub --
    # cu_extend_seq_lens_cpu[-1] only covers the prefill segment here.
    mixed_total_tokens: int = 0


@dataclass(kw_only=True)
class MHADecodeMetadata:
    # page_table is None on the flat path (per-group page_tables route reads).
    page_table: torch.Tensor | None
    seq_lens: torch.Tensor
    # Flat per-group tables/write-locs; see MHAExtendMetadata.
    page_tables: dict[str, torch.Tensor] | None = None
    out_cache_locs: dict[str, torch.Tensor] | None = None
    # Pre-planned FlashInfer FA2 decode wrapper, attached only when the FA2
    # cuda-graph plan/run split is active (sm_120 flashinfer decode). None on
    # every other path (eager, Triton, trtllm-gen), so those are unaffected.
    fa2_decode_wrapper: object | None = None


class MHAAttnBackend(FlatCacheGroupsMixin, AttentionBackend):
    """Standard MHA backend that routes through tokenspeed_kernel attention APIs."""

    # Unconditional: safety comes from the publication rule
    # (paged_cache_spec.publish_paged_cache_groups) plus the replay
    # stale-table guard. TODO(radix-removal): drop the flag.
    uses_flat_cache_groups: bool = True
    # MIXED is served by splitting the batch at the segment boundary and giving
    # each half the kernel built for it -- see _forward_mixed. __init__ turns
    # this back OFF for spec / draft ranks, whose decode rows are verify windows
    # rather than single tokens; that path is not wired and must not be
    # reachable from a flag.
    supports_mixed_batch: bool = True

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return forward_mode is not None and forward_mode.is_decode()

    def __init__(self, config: MHAConfig):
        super().__init__(config)
        # Map the selected backend to the corresponding kernel solution string.
        backend_name = config.backend_name or "mha"
        self.kernel_solution = _KERNEL_SOLUTION_BY_BACKEND[backend_name]

        # Static information needed for metadata construction and kernel dispatch
        self.num_q_heads_local = max(
            config.num_attention_heads // config.attn_tp_size, 1
        )
        self.num_kv_heads_local = max(config.num_kv_heads // config.attn_tp_size, 1)
        self.max_context_len = config.context_len
        self.page_size = config.page_size
        self.max_num_pages = ceil_div(self.max_context_len, self.page_size)
        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_kv_heads
        self.tp_q_head_num = max(num_q_heads // config.attn_tp_size, 1)
        self.tp_kv_head_num = max(num_kv_heads // config.attn_tp_size, 1)
        self.head_dim = config.head_dim
        self.qkv_dtype = config.dtype
        self.kv_cache_dtype = config.kv_cache_dtype
        self.is_mxfp8 = bool(getattr(config, "kv_cache_mxfp8", False))
        # mxfp8 shares the fp8 storage dtype but uses block scales; keep it off the per-tensor casts
        self.is_fp8 = (
            self.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and not self.is_mxfp8
        )
        # The Triton kernels read the fp8 cache directly and convert it up to
        # the query dtype, so they want the query left in the activation dtype:
        # casting it down quantizes the query for no benefit, and because the
        # output buffer is allocated `empty_like(q)` it also hands the attention
        # output back in e4m3. The fa3/fa4 fp8 kernels do need an fp8 query --
        # their signatures declare all three roles fp8 -- so the cast stays for
        # them.
        self.cast_query_to_fp8 = self.kernel_solution != "triton"
        self.plan = partial(
            mha_plan,
            dtype=(
                torch.float8_e4m3fn
                if self.is_mxfp8
                else (
                    self.kv_cache_dtype
                    if self.is_fp8 and self.cast_query_to_fp8
                    else self.qkv_dtype
                )
            ),
            head_dim=self.head_dim,
            return_lse=False,
            solution=self.kernel_solution,
        )
        # DFLASH draft: expand decode metadata to spec_num_tokens rows/request
        # (whole block in one decode forward), with uniform non-causal seq_lens.
        self.draft_block_decode = bool(getattr(config, "draft_block_decode", False))

        # MIXED support is per-instance, not per-class; see mixed_batch_supported.
        self.supports_mixed_batch = mixed_batch_supported(
            self.spec_num_tokens, self.is_draft, self.draft_block_decode
        )

        # Forward metadata is initialized in the runner per forward call
        self.forward_decode_metadata: MHADecodeMetadata | None = None
        self.forward_extend_metadata: MHAExtendMetadata | None = None

        # family="state" ids (GDN/mamba) learned in init_cuda_graph_state; this backend sheds them
        self.flat_state_group_ids: frozenset[str] = frozenset()
        # Per-group page sizes (heterogeneous block sizes), learned with them.
        self.flat_group_page_sizes: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Metadata initialization
    # ------------------------------------------------------------------

    def _install_paged_extend_plan(self, metadata, extend_prefix_lens_cpu) -> None:
        """Plan the FA2 paged extend kernel once for this forward.

        The kernel can plan itself, but only by copying ``cache_seqlens`` to the
        host to key the plan -- a sync per layer, which drains the queue the
        previous layers just filled. Everything the plan needs is already on the
        host here (prefix + extend lengths), so doing it once costs no sync at
        all. This is where SGLang plans too (``init_forward_metadata``).

        Installs on every extend forward, or clears; a plan is never allowed to
        outlive the forward it was built for, which is what lets the kernel
        match it by tensor identity alone.
        """
        if not paged_extend_planning_enabled():
            return
        clear_paged_extend_plan()
        # The flat KV path hands each state group its own page table and
        # `_select_page_table` gives a layer its group's, so plan one per group;
        # the single-table path has exactly one. Planning the table the kernel
        # will not be handed is what made the first version of this inert.
        if metadata.page_tables:
            tables = list(metadata.page_tables.values())
        elif metadata.page_table is not None:
            tables = [metadata.page_table]
        else:
            return
        prefix_cpu = [int(x) for x in extend_prefix_lens_cpu.tolist()]
        seq_lens_cpu = [p + e for p, e in zip(prefix_cpu, metadata.extend_seq_lens_cpu)]
        # Only the BATCH is installed -- which rows, which pages, how long each
        # is. The geometry (head count, head dim, page size) and the scale,
        # window and soft cap are per LAYER, and this backend has only pool-wide
        # values for them: a heterogeneous-KV model runs several geometries
        # through one forward (Gemma 4: 16 x 256 at page 16 on its sliding
        # groups, 4 x 512 at page 32 on full_attention) and declares a scale of
        # 1.0 rather than `head_dim ** -0.5`. The kernel plans each group's
        # wrapper from what the layer actually hands it; passing guesses from
        # here would only be something for it to disagree with.
        for page_table in tables:
            install_paged_extend_plan(
                cu_seqlens_q=metadata.cu_extend_seq_lens,
                cache_seqlens=metadata.seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                page_table=page_table,
            )

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        # Only consumed on the extend/mixed path; decode callers (e.g. the
        # DFLASH draft and the cuda-graph wrapper's draft decode init) omit
        # them, so they must be optional.
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        flat_block_tables: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        # NOT an assert: under ``python -O`` a skipped check would run the
        # extend metadata path over decode rows with out-of-range
        # extend_seq_lens/extend_prefix_lens slices, i.e. silent corruption.
        # ``supports_mixed_batch`` (set in __init__) gates the same condition at
        # startup; this is the last line of defence if a caller bypasses it.
        is_mixed = forward_mode.is_mixed()
        if is_mixed and not self.supports_mixed_batch:
            raise NotImplementedError(
                "this MHA backend instance cannot serve mixed prefill+decode "
                "batches (speculative decoding / draft rank: the decode tail is "
                "a verify window, not one token per row); run without "
                "--enable-mixed-batch (see AttentionBackend.supports_mixed_batch)"
            )

        seq_lens = seq_lens[:bs]

        flat_page_tables = self._shed_state_groups(flat_block_tables)
        flat_out_cache_locs = None
        if flat_page_tables:
            # Verify keeps [bs]-row tables; only DFLASH expands rows. TODO(flat+dflash).
            assert not (
                self.draft_block_decode and self.spec_num_tokens > 1
            ), "flat cache groups are unsupported with DFLASH block decode"
            # The flat path routes every read/write through the per-group
            # tables; the radix single table would be dead work.
            page_table = None
            if is_mixed:
                # Two segments, two builders. The prefill rows' write positions
                # come from the CPU prefix/extend mirrors as usual; the decode
                # tail's come from seq_lens on device (position seq_len - 1),
                # because a decode row's prefix exists only there and reading it
                # on the host would be a per-step D2H sync.
                assert extend_prefix_lens_cpu is not None
                assert extend_seq_lens_cpu is not None
                flat_out_cache_locs = self._compute_flat_mixed_out_cache_locs(
                    flat_page_tables,
                    bs,
                    num_extends,
                    extend_prefix_lens_cpu,
                    extend_seq_lens_cpu,
                    seq_lens,
                )
            elif forward_mode.is_extend_or_mixed():
                assert extend_prefix_lens_cpu is not None
                assert extend_seq_lens_cpu is not None
                flat_out_cache_locs = self._compute_flat_extend_out_cache_locs(
                    flat_page_tables,
                    extend_prefix_lens_cpu[:bs],
                    extend_seq_lens_cpu[:bs],
                    self.page_size,
                )
            else:
                # Spec rounds write multi-token windows on BOTH sides: the
                # target's verify chunk, and a window-mode draft's catch-up
                # plus per-depth window passes, which all cover the same
                # verify-window rows (positions seq-N..seq-1) every step.
                # Sizing the draft's locs at 1 token/request used to make
                # _save_kv_cache trim the bs*N-row draft writes down to bs —
                # silently corrupting every decode-round draft KV write on
                # the flat arm. (A chaining draft ignores these locs via
                # prefer_caller in the base forward_decode.)
                verify_tokens = self.spec_num_tokens if self.spec_num_tokens > 1 else 1
                flat_out_cache_locs = self._compute_flat_decode_out_cache_locs(
                    flat_page_tables,
                    seq_lens,
                    self.page_size,
                    verify_tokens,
                )
            self._maybe_check_flat_write_locs(
                flat_page_tables, flat_out_cache_locs, self.page_size
            )
        else:
            page_table = build_page_table(
                req_pool_indices[:bs],
                req_to_page,
                self.page_size,
                self.max_context_len,
            )

        if forward_mode.is_extend_or_mixed():
            assert extend_seq_lens is not None
            assert extend_seq_lens_cpu is not None
            assert extend_prefix_lens is not None
            assert extend_prefix_lens_cpu is not None

            # A MIXED plan holds prefill rows [0, num_extends) followed by
            # decode rows [num_extends, bs) (the scheduler's
            # FlatForwardOperation stable-partitions prefill first), and carries
            # the extend vectors for the PREFILL rows only -- model_executor
            # slices extend_*_buf[:num_extends]. Rather than synthesising a
            # decode tail for those vectors and serving all bs rows through one
            # varlen extend call, the extend metadata is built over the prefill
            # rows alone and the decode tail gets its own decode metadata; the
            # forward then dispatches each segment to the kernel built for it.
            # Measured on the B64 shape (1 x 3072 prefill + 63 decode rows over
            # 3k of cache, Qwen3-8B/TP1/sm_120, triton): 5.37 ms fused into one
            # extend call vs 2.91 ms split, i.e. the decode tail costs 3.3x more
            # inside the prefill kernel (3.74 ms vs 1.12 ms) because that kernel
            # tiles queries at BLOCK_M=128 and has no split-K over the KV, so a
            # seqlen_q == 1 row occupies a whole tile and scans its cache with
            # one block. Over 36 layers that is +89 ms per mixed step.
            rows = num_extends if is_mixed else bs
            extend_seq_lens = extend_seq_lens[:rows]
            extend_seq_lens_cpu = [int(x) for x in extend_seq_lens_cpu[:rows].tolist()]
            extend_prefix_lens = extend_prefix_lens[:rows]
            max_extend_prefix_len = int(extend_prefix_lens_cpu[:rows].max().item())
            extend_seq_lens_all = seq_lens[:rows]

            # Create cumulative sum of the sequence lengths for Q and KV.
            cu_extend_seq_lens = torch.nn.functional.pad(
                torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32),
                (1, 0),
            )
            cu_extend_seq_lens_cpu = [0]
            for length in extend_seq_lens_cpu:
                cu_extend_seq_lens_cpu.append(cu_extend_seq_lens_cpu[-1] + length)
            cu_seqlens_kv = torch.nn.functional.pad(
                torch.cumsum(extend_seq_lens_all, dim=0, dtype=torch.int32),
                (1, 0),
            )
            max_extend_seq_len = max(extend_seq_lens_cpu)

            self.forward_extend_metadata = MHAExtendMetadata(
                page_table=None if page_table is None else page_table[:rows],
                seq_lens=extend_seq_lens_all,
                extend_seq_lens=extend_seq_lens,
                cu_extend_seq_lens=cu_extend_seq_lens,
                cu_seqlens_kv=cu_seqlens_kv,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                cu_extend_seq_lens_cpu=cu_extend_seq_lens_cpu,
                max_extend_seq_len=max_extend_seq_len,
                max_extend_prefix_len=max_extend_prefix_len,
                page_tables=(
                    None
                    if flat_page_tables is None
                    else {gid: t[:rows] for gid, t in flat_page_tables.items()}
                ),
                # The whole batch's write locations: the mixed forward writes KV
                # for both segments in ONE call before either attention runs.
                out_cache_locs=flat_out_cache_locs,
                mixed_prefill_tokens=(cu_extend_seq_lens_cpu[-1] if is_mixed else 0),
                mixed_decode_rows=(bs - num_extends if is_mixed else 0),
                mixed_total_tokens=(
                    cu_extend_seq_lens_cpu[-1] + (bs - num_extends) if is_mixed else 0
                ),
            )
            self._install_paged_extend_plan(
                self.forward_extend_metadata, extend_prefix_lens_cpu[:rows]
            )

            if is_mixed:
                # Decode tail: rows [num_extends, bs), one query token each.
                self.forward_decode_metadata = MHADecodeMetadata(
                    page_table=(
                        None if page_table is None else page_table[num_extends:bs]
                    ),
                    seq_lens=seq_lens[num_extends:bs],
                    page_tables=(
                        None
                        if flat_page_tables is None
                        else {
                            gid: t[num_extends:bs]
                            for gid, t in flat_page_tables.items()
                        }
                    ),
                    # KV for this segment is written by the mixed forward's
                    # single whole-batch save, so the decode call needs no locs.
                    out_cache_locs=None,
                )

            # Drafter step 1+ decodes under an EXTEND/MIXED target; seq_lens
            # aliases the drafter's live buffer (pre-written by the wrapper).
            if self.is_draft:
                self.forward_decode_metadata = MHADecodeMetadata(
                    page_table=page_table,
                    seq_lens=seq_lens,
                    page_tables=flat_page_tables,
                    out_cache_locs=flat_out_cache_locs,
                )
        else:
            if self.draft_block_decode and self.spec_num_tokens > 1:
                # DFLASH drafts a whole block in one decode forward; the decode
                # kernel keys masking off max_seqlen_q, so expand each request
                # into spec_num_tokens rows with the SAME full seq_len. That
                # makes max_seqlen_q == 1 per row, so every block query attends
                # over the entire block (non-causal block-diffusion drafting).
                # Target verify keeps the unexpanded multi-query decode path.
                expanded_page_table, expanded_seq_lens = (
                    self._make_spec_metadata_buffers(
                        bs,
                        page_table.device,
                    )
                )
                self._fill_spec_metadata_uniform(
                    expanded_page_table,
                    expanded_seq_lens,
                    page_table,
                    seq_lens,
                )
                self.forward_decode_metadata = MHADecodeMetadata(
                    page_table=expanded_page_table,
                    seq_lens=expanded_seq_lens,
                    page_tables=flat_page_tables,
                    out_cache_locs=flat_out_cache_locs,
                )
            else:
                self.forward_decode_metadata = MHADecodeMetadata(
                    page_table=page_table,
                    seq_lens=seq_lens,
                    page_tables=flat_page_tables,
                    out_cache_locs=flat_out_cache_locs,
                )

    def _compute_flat_mixed_out_cache_locs(
        self,
        page_tables: dict[str, torch.Tensor],
        bs: int,
        num_extends: int,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Per-group flat write locations for a MIXED batch.

        The KV write order follows ``cu_extend_seq_lens``: every prefill row's
        new tokens first, then one token per decode row. So the locs are the
        extend builder's output over rows ``[0, num_extends)`` concatenated with
        the decode builder's output over rows ``[num_extends, bs)`` -- the two
        halves of the same vector, in that order. Getting the order wrong (or
        sizing either half from the whole batch) writes each row's K/V into
        another request's page, which is exactly the failure the MIXED guard
        used to prevent.

        Args:
            page_tables: ``group_id -> [bs, max_pages]`` read tables.
            bs: Total rows.
            num_extends: Leading prefill rows.
            extend_prefix_lens_cpu: ``[num_extends]`` CPU prefix lengths.
            extend_seq_lens_cpu: ``[num_extends]`` CPU new-token counts.
            seq_lens: ``[bs]`` post-step lengths, on device.

        Returns:
            ``group_id -> [total_tokens]`` write locations.
        """
        prefill_locs = self._compute_flat_extend_out_cache_locs(
            page_tables,
            extend_prefix_lens_cpu[:num_extends],
            extend_seq_lens_cpu[:num_extends],
            self.page_size,
        )
        if bs == num_extends:
            return prefill_locs
        decode_locs = self._compute_flat_decode_out_cache_locs(
            {gid: table[num_extends:bs] for gid, table in page_tables.items()},
            seq_lens[num_extends:bs],
            self.page_size,
            1,
        )
        return {
            gid: torch.cat([prefill_locs[gid], decode_locs[gid]]) for gid in page_tables
        }

    def init_cuda_graph_state(
        self,
        max_bs: int,
        seq_lens_buf: torch.Tensor,
        paged_cache_group_specs: Sequence = (),
        **kwargs,
    ):
        # State-family groups (GDN/mamba pages) belong to the mamba backend;
        # learn their ids from the pool's specs so every flat table/loc path
        # here (eager, capture, replay) sheds them.
        self._learn_flat_state_groups(paged_cache_group_specs)
        assert (
            seq_lens_buf.dtype == torch.int32
            and seq_lens_buf.dim() == 1
            and seq_lens_buf.shape[0] >= max_bs
        ), (
            f"seq_lens_buf must be int32 with shape[0] >= {max_bs}, "
            f"got {seq_lens_buf.dtype} {tuple(seq_lens_buf.shape)}"
        )

        self.cuda_graph_decode_metadata = {}
        # FA2 cuda-graph plan/run split state. Disabled by default; enabled only
        # on the normal decode path below when FA2 is the selected flashinfer
        # decode kernel. Initialized before the DFLASH early return so the
        # capture/replay hooks can read it unconditionally.
        self._fa2_graph_enabled = False
        # Single-table path: bs -> wrapper. Flat path: (bs, group_id) -> wrapper.
        self._fa2_decode_wrappers: dict[int, object] = {}
        self._fa2_flat_enabled = False
        self._fa2_flat_decode_wrappers: dict[tuple[int, str], object] = {}
        # Per-group graph-stable CSR buffers: group_id -> (indptr, indices, last).
        self._fa2_flat_buffers: dict[str, tuple] = {}
        # Flat per-group persistent buffers. TODO(radix-removal): parallels
        # cuda_graph_page_table.
        # Initialized before the DFLASH early return: replay reads the dict
        # unconditionally for the stale-table guard.
        self._init_flat_graph_buffers(max_bs)
        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block: expand to spec_num_tokens decode rows per
            # request (one row per block position), so max_seqlen_q == 1 per row
            # and every block query attends over the whole block (non-causal).
            self.cuda_graph_page_table, self.cuda_graph_seq_lens = (
                self._make_spec_metadata_buffers(max_bs, self.device)
            )
            self.cuda_graph_page_table.zero_()
            # seq_lens are filled from the live draft length inside the captured
            # graph; seed a valid baseline so any pre-broadcast read stays in range.
            self.cuda_graph_seq_lens.fill_(self.spec_num_tokens)
            return
        self.cuda_graph_page_table = torch.zeros(
            (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        if self.spec_num_tokens > 1 and not self.is_draft:
            self.cuda_graph_seq_lens = torch.empty(
                (max_bs,), dtype=torch.int32, device=self.device
            )
        else:
            # Alias controller's seq_lens_buf — backend never mutates it.
            self.cuda_graph_seq_lens = seq_lens_buf

        # Enable the FA2 decode plan/run split only for the plain single-query
        # decode path (no fp8/mxfp8, no spec/DFLASH, target rank) AND only when
        # FA2 is the selected flashinfer decode kernel (sm_120). On sm_100/103
        # this stays False so trtllm-gen keeps the normal op path untouched.
        self._fa2_graph_enabled = (
            _HAS_FA2_GRAPH
            and self.kernel_solution == "flashinfer"
            and not self.is_fp8
            and not self.is_mxfp8
            and self.spec_num_tokens <= 1
            and not self.draft_block_decode
            and not self.is_draft
            and fa2_decode_is_selected(self.qkv_dtype)
        )
        if self._fa2_graph_enabled:
            # Attention cache groups this backend consumes on the flat path
            # (exclude state/mamba and engine-owned conv groups). Non-empty
            # only under a flat scheduler build that publishes groups; empty on
            # the radix/single-table default, so single-table stays unchanged.
            att_gids = [
                gid
                for gid in self.flat_group_page_sizes
                if gid not in self.flat_state_group_ids
                and gid not in self.flat_engine_owned_group_ids
            ]
            # Was a hardcoded 128 MiB with no override. FlashInfer sub-allocates
            # its prefill temporaries out of this buffer through an
            # AlignedAllocator, and on Qwen3-scale prefill it asks for slightly
            # more than that -- the observed failure was batch_prefill_tmp_v
            # wanting 136,052,736 B against 134,217,728 B available, i.e. short
            # by 1.75 MiB, which aborted startup with "Increase the workspace
            # buffer size". That made the flashinfer backend look unsupported on
            # sm_120 when it was only unconfigured, and this is the sm_120 path
            # specifically (see _fa2_graph_enabled above).
            #
            # Every other backend here is already larger and configurable:
            # flashmla.py uses get_flashinfer_workspace_size() (384 MiB default,
            # FLASHINFER_WORKSPACE_SIZE to override) and trtllm_mla.py uses
            # 256 MiB. Reuse the project's own helper rather than inventing a
            # second knob.
            self._fa2_workspace = torch.empty(
                get_flashinfer_workspace_size(), dtype=torch.uint8, device=self.device
            )
            if att_gids:
                # Flat path: one CSR buffer set per attention group, sized by
                # that group's page-size granularity.
                self._fa2_flat_enabled = True
                for gid in att_gids:
                    group_width = ceil_div(
                        self.max_num_pages * self.page_size,
                        self._group_page_size(gid),
                    )
                    self._fa2_flat_buffers[gid] = (
                        torch.zeros(max_bs + 1, dtype=torch.int32, device=self.device),
                        torch.zeros(
                            max_bs * group_width,
                            dtype=torch.int32,
                            device=self.device,
                        ),
                        torch.zeros(max_bs, dtype=torch.int32, device=self.device),
                    )
            else:
                # Single-table path.
                self._fa2_indptr_buf = torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                )
                self._fa2_indices_buf = torch.zeros(
                    max_bs * self.max_num_pages, dtype=torch.int32, device=self.device
                )
                self._fa2_last_page_len_buf = torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        flat_cache_group_ids: tuple[str, ...] = (),
        **kwargs,
    ):
        assert not forward_mode.is_extend_or_mixed()

        # Real tables only arrive at replay; capture records metadata views
        # into the persistent per-group buffers so replay can copy_ fresh data
        # to the graph-recorded addresses.
        if flat_cache_group_ids:
            # Verify keeps [bs]-row tables + [bs*N] loc views. TODO(flat+dflash).
            assert not (
                self.draft_block_decode and self.spec_num_tokens > 1
            ), "flat_cache_group_ids is unsupported with DFLASH block decode"
        page_tables, out_cache_locs = self._flat_capture_group_views(
            bs,
            flat_cache_group_ids,
            # Multi-token window locs for target verify AND window-mode
            # drafts alike; see the eager decode branch.
            tokens_per_req=(self.spec_num_tokens if self.spec_num_tokens > 1 else 1),
        )

        if self.draft_block_decode and self.spec_num_tokens > 1:
            # DFLASH draft block: spec_num_tokens decode rows per request.
            expanded_bs = bs * self.spec_num_tokens
            metadata = MHADecodeMetadata(
                page_table=self.cuda_graph_page_table[:expanded_bs, :],
                seq_lens=self.cuda_graph_seq_lens[:expanded_bs],
                page_tables=page_tables,
                out_cache_locs=out_cache_locs,
            )
            # Uniform non-causal seq_lens are written by the drafter inside the
            # captured graph (see fill_block_decode_seq_lens); seed a safe
            # baseline for the capture run before that op records.
            metadata.seq_lens.fill_(self.spec_num_tokens)
        else:
            metadata = MHADecodeMetadata(
                # Flat captures route reads through the per-group tables and
                # replay never fills the radix single table, so mirror the
                # eager flat path: page_table=None instead of a slice of the
                # never-filled zero buffer.
                page_table=(
                    None
                    if page_tables is not None
                    else self.cuda_graph_page_table[:bs, :]
                ),
                seq_lens=self.cuda_graph_seq_lens[:bs],
                page_tables=page_tables,
                out_cache_locs=out_cache_locs,
            )
            if self.spec_num_tokens > 1 and not self.is_draft:
                metadata.seq_lens.copy_(seq_lens[:bs].clamp_min(self.spec_num_tokens))

        # FA2 decode plan/run split: build graph-capturable wrapper(s) for this
        # bs and pre-plan them (outside the captured region) so the forward that
        # is about to be captured records only wrapper.run().
        if self._fa2_graph_enabled:
            if self._fa2_flat_enabled and metadata.page_tables is not None:
                # Flat path: one wrapper per attention group, planned from that
                # group's own graph-stable page table + page-size granularity.
                # (Real tables arrive at replay; capture plans over the zeroed
                # buffers, which is fine because the schedule keys off seq_lens.)
                group_wrappers: dict[str, object] = {}
                for gid, group_table in metadata.page_tables.items():
                    bufs = self._fa2_flat_buffers.get(gid)
                    if bufs is None:
                        continue
                    indptr_buf, indices_buf, last_buf = bufs
                    wrapper = make_fa2_decode_cuda_graph_wrapper(
                        self._fa2_workspace,
                        indptr_buf[: bs + 1],
                        indices_buf,
                        last_buf[:bs],
                        use_tensor_cores=True,
                    )
                    fa2_decode_cuda_graph_plan(
                        wrapper,
                        group_table,
                        metadata.seq_lens,
                        self._group_page_size(gid),
                        self.tp_q_head_num,
                        self.tp_kv_head_num,
                        self.head_dim,
                        self.qkv_dtype,
                        self.kv_cache_dtype,
                        indptr_buf,
                        indices_buf,
                        last_buf,
                    )
                    self._fa2_flat_decode_wrappers[(bs, gid)] = wrapper
                    group_wrappers[gid] = wrapper
                if group_wrappers:
                    metadata.fa2_decode_wrapper = group_wrappers
            elif metadata.page_table is not None:
                # Single-table path.
                wrapper = make_fa2_decode_cuda_graph_wrapper(
                    self._fa2_workspace,
                    self._fa2_indptr_buf[: bs + 1],
                    self._fa2_indices_buf,
                    self._fa2_last_page_len_buf[:bs],
                    use_tensor_cores=True,
                )
                fa2_decode_cuda_graph_plan(
                    wrapper,
                    metadata.page_table,
                    metadata.seq_lens,
                    self.page_size,
                    self.tp_q_head_num,
                    self.tp_kv_head_num,
                    self.head_dim,
                    self.qkv_dtype,
                    self.kv_cache_dtype,
                    self._fa2_indptr_buf,
                    self._fa2_indices_buf,
                    self._fa2_last_page_len_buf,
                )
                self._fa2_decode_wrappers[bs] = wrapper
                metadata.fa2_decode_wrapper = wrapper

        self.cuda_graph_decode_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        flat_block_tables: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        assert not forward_mode.is_extend_or_mixed()

        # Fail loudly instead of replaying over stale/zero page tables.
        self._flat_replay_stale_guard(bs, flat_block_tables)

        # Flat captures read only the per-group buffers; the radix single
        # table (cuda_graph_page_table) would be dead work there.
        if not self.cuda_graph_flat_page_tables:
            gather_page_table_with_padding(
                req_to_page=req_to_page,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out=self.cuda_graph_page_table,
                bs=bs,
                max_num_pages=self.max_num_pages,
                page_size=self.page_size,
                dummy_slot=0,
            )
        if self.spec_num_tokens > 1 and not self.is_draft:
            # Clamp padded rows (seq_len 1) to N: verify derives row lengths as seq - N + t + 1, which must stay positive.
            torch.clamp_min(
                seq_lens[:bs],
                self.spec_num_tokens,
                out=self.cuda_graph_seq_lens[:bs],
            )
        elif self.draft_block_decode:
            # DFLASH draft: replicate each request's page table to its
            # spec_num_tokens block rows. The block-end seq_lens are filled by
            # the drafter inside the captured graph, so they are not touched
            # here (they re-derive from the live draft length on every replay).
            base_page_table = req_to_page[req_pool_indices[:bs], : self.max_num_pages]
            self.cuda_graph_page_table[: bs * self.spec_num_tokens, :].view(
                bs, self.spec_num_tokens, self.max_num_pages
            ).copy_(base_page_table[:, None, :])

        # cuda_graph_seq_lens is filled by input prep / the spec copy above.
        if flat_block_tables:
            self._flat_replay_fill(
                bs,
                flat_block_tables,
                self.cuda_graph_seq_lens,
                tokens_per_req=(
                    self.spec_num_tokens if self.spec_num_tokens > 1 else 1
                ),
            )

        # FA2 decode plan/run split: re-plan this bs's wrapper(s) with the live
        # page table + seq_lens BEFORE graph.replay() (outside the captured
        # region). The captured run() then reads the refreshed fixed buffers.
        if self._fa2_graph_enabled and self._fa2_flat_enabled:
            # Flat path: re-plan each attention group's wrapper from its live
            # graph-stable table (filled by _flat_replay_fill above).
            for gid, group_table in self.cuda_graph_flat_page_tables.items():
                wrapper = self._fa2_flat_decode_wrappers.get((bs, gid))
                bufs = self._fa2_flat_buffers.get(gid)
                if wrapper is None or bufs is None:
                    continue
                indptr_buf, indices_buf, last_buf = bufs
                fa2_decode_cuda_graph_plan(
                    wrapper,
                    group_table[:bs],
                    self.cuda_graph_seq_lens[:bs],
                    self._group_page_size(gid),
                    self.tp_q_head_num,
                    self.tp_kv_head_num,
                    self.head_dim,
                    self.qkv_dtype,
                    self.kv_cache_dtype,
                    indptr_buf,
                    indices_buf,
                    last_buf,
                )
        elif (
            self._fa2_graph_enabled
            and not self.cuda_graph_flat_page_tables
            and bs in self._fa2_decode_wrappers
        ):
            fa2_decode_cuda_graph_plan(
                self._fa2_decode_wrappers[bs],
                self.cuda_graph_page_table[:bs],
                self.cuda_graph_seq_lens[:bs],
                self.page_size,
                self.tp_q_head_num,
                self.tp_kv_head_num,
                self.head_dim,
                self.qkv_dtype,
                self.kv_cache_dtype,
                self._fa2_indptr_buf,
                self._fa2_indices_buf,
                self._fa2_last_page_len_buf,
            )

        if bs in self.cuda_graph_decode_metadata:
            self.forward_decode_metadata = self.cuda_graph_decode_metadata[bs]

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """DFLASH: broadcast each request's block-end length to its
        spec_num_tokens cuda-graph decode rows (uniform, non-causal).

        Called by the drafter inside the captured graph so that on every replay
        the expanded seq_lens re-derive from the live draft length (which is
        recomputed in-graph from the target's accept lengths).

        Args:
            bs: Number of draft requests.
            block_seq_lens: ``[bs]`` per-request block-end lengths
                (prefix + spec_num_tokens).
        """
        spec = self.spec_num_tokens
        self.cuda_graph_seq_lens[: bs * spec].view(bs, spec).copy_(
            block_seq_lens[:bs]
            .clamp(self.spec_num_tokens, self.max_context_len)
            .unsqueeze(1)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim
        assert (k is None) == (v is None)
        has_kv = k is not None

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if has_kv:
            k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
            v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        sinks = kwargs.get("sinks")

        out_cache_loc = self._select_out_cache_loc(
            layer,
            self.forward_decode_metadata,
            out_cache_loc,
            prefer_caller=self.is_draft,
        )

        return self._forward_decode(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            self.forward_decode_metadata,
            save_kv_cache=save_kv_cache,
            sinks=sinks,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim
        assert (k is None) == (v is None)
        assert k is not None

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        metadata = self.forward_extend_metadata
        sinks = kwargs.get("sinks")
        out_cache_loc = self._select_out_cache_loc(layer, metadata, out_cache_loc)
        plan = self.plan(
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
        )
        extend_mode = plan.get("extend_mode", "prewrite")

        if metadata.mixed_decode_rows:
            return self._forward_mixed(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                save_kv_cache,
                sinks,
                extend_mode,
            )

        if metadata.max_extend_prefix_len == 0 and extend_mode == "postwrite":
            return self._forward_prefill(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                save_kv_cache,
                sinks,
            )
        else:
            return self._forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                save_kv_cache,
                sinks,
            )

    def _forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAExtendMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
        extend_mode: str,
    ) -> torch.Tensor:
        """Serve a MIXED batch: prefill rows and decode rows, one forward, two kernels.

        The batch is packed prefill-first, so the query rows split cleanly at
        ``mixed_prefill_tokens`` and each half goes to the kernel built for it --
        the varlen extend/prefill kernel for the chunks, the decode kernel for
        the one-token rows. Serving all of it through a single varlen extend call
        also works and is numerically identical, but is far slower: that kernel
        tiles queries at ``BLOCK_M`` (128 on this arch) and parallelises only
        over (request, head, query-block), so a ``seqlen_q == 1`` row occupies a
        whole tile and walks its entire KV with a single block, with no split-K.
        Measured on the B64 shape (1 x 3072-token chunk + 63 decode rows over 3k
        of cache, Qwen3-8B/TP1/sm_120/triton): the decode tail costs 3.74 ms
        fused versus 1.12 ms on the decode kernel, making the layer 5.37 ms
        instead of 2.91 ms -- +89 ms per step across 36 layers.

        The KV write happens ONCE for the whole batch before either attention
        call, which is what lets both halves read their own new tokens: the
        prefill segment needs its chunk in cache for the paged extend path, and
        every decode row needs the token it is about to attend over.

        Args:
            q: Packed queries ``[total_tokens, num_q_heads, head_dim]``; may
                carry a padded tail under a prefill-graph replay.
            k: Packed keys for the same rows.
            v: Packed values for the same rows.
            layer: The attention layer (page size, scales, cache group).
            out_cache_loc: Whole-batch KV write locations.
            token_to_kv_pool: The paged KV pool.
            metadata: Extend metadata for the PREFILL segment, carrying the
                ``mixed_*`` split points.
            save_kv_cache: Whether this backend owns the KV write.
            sinks: Optional attention sinks.
            extend_mode: ``"prewrite"``/``"postwrite"`` from the kernel plan.

        Returns:
            ``[total_tokens, num_q_heads * v_head_dim]`` in the input row order;
            a padded tail (graph replay) is returned zeroed.
        """
        npt = metadata.mixed_prefill_tokens
        nd = metadata.mixed_decode_rows
        total = metadata.mixed_total_tokens

        # Scrub before the write so a padded tail cannot reach the cache, then
        # write both segments' KV in one call (see the docstring).
        scrub_padding_tail(total, q, k, v)
        if save_kv_cache:
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)

        # Prefill segment. A fresh chunk (no cached prefix) can still take the
        # no-cache prefill kernel; anything with a prefix must read the pages.
        if metadata.max_extend_prefix_len == 0 and extend_mode == "postwrite":
            prefill_out = self._forward_prefill(
                q[:npt],
                k[:npt],
                v[:npt],
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                False,
                sinks,
            )
        else:
            prefill_out = self._forward_extend(
                q[:npt],
                None,
                None,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                False,
                sinks,
            )

        # Decode segment: one query token per row, over the pages written above.
        decode_out = self._forward_decode(
            q[npt:total],
            None,
            None,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            self.forward_decode_metadata,
            False,
            sinks,
        )

        if q.shape[0] == total:
            return torch.cat([prefill_out, decode_out])
        # Padded prefill-graph replay: keep the caller's row count, zero the tail.
        out = prefill_out.new_zeros(q.shape[0], prefill_out.shape[1])
        out[:npt] = prefill_out
        out[npt:total] = decode_out
        return out

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAExtendMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        _scrub_extend_padding(metadata, q, k, v)
        # TODO: use a custom kernel to do downcast
        # Uncached prefill: nothing is read from the fp8 cache here, so the
        # downcast buys no memory -- it only matches what the store will write.
        # `_save_kv_cache` quantizes on store anyway (fused_fp8_set_kv_buffer),
        # so on the Triton path this is skipped and prefill runs at activation
        # precision.
        if self.is_fp8 and self.cast_query_to_fp8:
            q = q.to(self.kv_cache_dtype)
            k = k.to(self.kv_cache_dtype)
            v = v.to(self.kv_cache_dtype)

        output = mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens=metadata.cu_extend_seq_lens,
            cu_seqlens_cpu=metadata.cu_extend_seq_lens_cpu,
            max_seqlen=metadata.max_extend_seq_len,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            softmax_scale=layer.scaling,
            sinks=sinks,
            solution=self.kernel_solution,
        )
        output = output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
        if save_kv_cache:
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)
        return output

    def _forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAExtendMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        _scrub_extend_padding(metadata, q, k, v)
        if save_kv_cache:
            # KV store (incl. the mxfp8 quantize-on-store path) lives solely
            # in _save_kv_cache.
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)

        scale_kwargs = {}
        if self.is_mxfp8:
            q, q_sf = self._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif self.is_fp8 and self.cast_query_to_fp8:
            q = q.to(self.kv_cache_dtype)

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        output = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=metadata.cu_extend_seq_lens,
            cu_seqlens_kv=metadata.cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=self._select_page_table(layer, metadata),
            cache_seqlens=metadata.seq_lens,
            max_seqlen_q=metadata.max_extend_seq_len,
            max_seqlen_k=self.max_context_len,
            # DFLASH marks its draft attention non-causal so the draft block's
            # query positions attend bidirectionally. Every other layer leaves
            # the attribute unset, so this stays causal by default.
            is_causal=not bool(getattr(layer, "non_causal", False)),
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            softmax_scale=layer.scaling,
            sinks=sinks,
            solution=self.kernel_solution,
            **scale_kwargs,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHADecodeMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        if save_kv_cache:
            # KV store (incl. the mxfp8 quantize-on-store path) lives solely
            # in _save_kv_cache.
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)

        scale_kwargs = {}
        if self.is_mxfp8:
            q, q_sf = self._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif self.is_fp8 and self.cast_query_to_fp8:
            q = q.to(self.kv_cache_dtype)

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)

        # FA2 cuda-graph plan/run split: when a pre-planned wrapper is attached
        # (sm_120 flashinfer decode under a captured graph), run() only — plan
        # already happened in the capture/replay hook outside the graph. The
        # shared decode plan carries no per-layer logit_cap, so guard it.
        fa2_wrapper = getattr(metadata, "fa2_decode_wrapper", None)
        if isinstance(fa2_wrapper, dict):
            # Flat path: select this layer's cache-group wrapper (matches the
            # per-group k/v view + page-size that wrapper was planned with).
            fa2_wrapper = self._select_group_entry(
                layer, fa2_wrapper, "fa2 decode wrapper"
            )
        if fa2_wrapper is not None:
            if layer.logit_cap:
                raise RuntimeError(
                    "FA2 cuda-graph decode split does not support logit_cap; "
                    "disable the flashinfer backend for this model."
                )
            # The wrapper was planned with the kernel's default scale, and run()
            # takes no override -- a model whose scale is not that default (e.g.
            # Gemma 4's 1.0) would be silently attended at the wrong temperature.
            if not math.isclose(
                layer.scaling, 1.0 / math.sqrt(layer.qk_head_dim), rel_tol=1e-12
            ):
                raise RuntimeError(
                    "FA2 cuda-graph decode split carries no per-layer "
                    f"softmax_scale; layer {layer.layer_id} declares "
                    f"scaling={layer.scaling}, not the planned "
                    f"{1.0 / math.sqrt(layer.qk_head_dim)}. Disable the "
                    "flashinfer backend for this model."
                )
            output = fa2_decode_cuda_graph_run(
                fa2_wrapper,
                q,
                k_cache,
                v_cache,
                sinks=sinks,
                window_left=layer.sliding_window_size,
            )
            return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

        max_seqlen_q = q.shape[0] // metadata.seq_lens.shape[0]
        output = mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=self._select_page_table(layer, metadata),
            cache_seqlens=metadata.seq_lens,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            softmax_scale=layer.scaling,
            sinks=sinks,
            max_seqlen_k=self.max_context_len,
            max_seqlen_q=max_seqlen_q,
            solution=self.kernel_solution,
            **scale_kwargs,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _store_kv_mxfp8(self, layer, loc, token_to_kv_pool, k, v) -> None:
        """MXFP8 quantize-on-store: one fused launch when the pool supports
        it (bit-identical, PDL-chained), else the split 5-launch path.
        The sole caller (_save_kv_cache) has already trimmed k/v to loc."""
        fused = getattr(token_to_kv_pool, "quantize_and_set_kv_buffer", None)
        if fused is not None and fused(layer, loc, k, v):
            return
        k_q, k_sf = self._quantize_mxfp8_tokens(k)
        v_q, v_sf = self._quantize_mxfp8_tokens(v)
        token_to_kv_pool.set_kv_buffer(layer, loc, k_q, v_q, k_scale=k_sf, v_scale=v_sf)

    def _quantize_mxfp8_tokens(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token MXFP8: (fp8-e4m3 [T, H, D], UE8M0 scales [T, H, D // 32]).

        Accepts [T, H, D] or [T, H * D]; H is inferred so the same helper
        serves q (tp_q_head_num) and k/v (tp_kv_head_num).
        """
        t, d = x.shape[0], self.head_dim
        h = x.numel() // (t * d)
        # (A PDL triton variant measured 0.07 ms slower e2e at decode Q shapes; flashinfer stays)
        data, sf = quantize_mxfp8(x.reshape(t * h, d))
        return (
            data.view(t, h, d),
            sf.view(torch.float8_e8m0fnu).view(t, h, d // 32),
        )

    def _save_kv_cache(
        self,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
    ) -> None:
        if k is None:
            return
        k, v = self._trim_kv_to_locs(out_cache_loc, k, v)

        if self.is_mxfp8:
            # Quantize-on-store: fp8 data + per-token e8m0 scales into the paged interleaved layout
            self._store_kv_mxfp8(layer, out_cache_loc, token_to_kv_pool, k, v)
        elif self.is_fp8:
            # Re-view per layer, exactly as the read path does (_get_kv_cache).
            # The pool allocates every row at the widest layer's geometry and a
            # narrower layer's row is a re-count of those bytes, so the raw
            # buffer is the wrong shape for anything but the widest layer.
            # Taking it raw handed Gemma 4's 4x512 full-attention layers to an
            # 8x256 row and the server died at startup with
            # `num_kv_heads mismatch: k.shape[1]=4 vs cache=8`. The kernel
            # accepts the paged layout directly, so the read path's view is
            # already the shape it wants -- and the page size must be the
            # layer's group's, not the backend's global one, for the same
            # reason the read path uses _layer_page_size.
            k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
            fused_fp8_set_kv_buffer(
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_loc=out_cache_loc,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
                page_size=self._layer_page_size(layer),
            )
        else:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

    def _get_kv_cache(self, layer: PagedAttention, token_to_kv_pool):
        # Hetero block sizes: page ids are in the LAYER'S group page-size units, so view at that size
        page_size = self._layer_page_size(layer)
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1,
            page_size,
            layer.tp_k_head_num,
            layer.qk_head_dim,
        )
        v_cache = token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1,
            page_size,
            layer.tp_v_head_num,
            layer.v_head_dim,
        )
        return k_cache, v_cache

    def _make_spec_metadata_buffers(
        self,
        bs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_bs = bs * self.spec_num_tokens
        cuda_graph_page_table = torch.empty(
            (expanded_bs, self.max_num_pages),
            dtype=torch.int32,
            device=device,
        )
        cuda_graph_seq_lens = torch.empty(
            (expanded_bs,),
            dtype=torch.int32,
            device=device,
        )
        return (cuda_graph_page_table, cuda_graph_seq_lens)

    def _fill_spec_metadata_uniform(
        self,
        expanded_page_table: torch.Tensor,
        expanded_seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """Expand spec metadata with a uniform (non-causal) seq_len per row.

        Replicates the full seq_len to all spec_num_tokens rows of a request so
        each row decodes with max_seqlen_q == 1 over the whole block. Used by the
        DFLASH drafter so every block query attends over the entire block
        (non-causal block-diffusion drafting), as opposed to the target's
        unexpanded causal multi-query verify path.
        """
        bs = seq_lens.shape[0]
        spec_num_tokens = self.spec_num_tokens
        expanded_page_table = expanded_page_table.view(
            bs, spec_num_tokens, self.max_num_pages
        )
        expanded_page_table.copy_(page_table[:, None, :])
        # Clamp to max_context_len so the draft decode never asks the attention
        # kernel for more than max_num_pages worth of page-table columns. The
        # block-end length is prefix + spec_num_tokens, which can exceed
        # max_context_len for a request near the context limit; without the
        # clamp the kernel reads page_table[:, >= max_num_pages] out of bounds
        # (CUDA illegal memory access). Mirrors fill_block_decode_seq_lens on the
        # cuda-graph path (this eager path is taken by mixed prefill+decode
        # batches even when cuda graphs are enabled).
        expanded_seq_lens.view(bs, spec_num_tokens).copy_(
            seq_lens.clamp(spec_num_tokens, self.max_context_len)[:, None]
        )


for _backend_name in _KERNEL_SOLUTION_BY_BACKEND:
    register_backend(_backend_name, {AttentionArch.MHA}, MHAAttnBackend)
