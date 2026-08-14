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
)
from tokenspeed_kernel.ops.quantization import quantize_mxfp8

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    scrub_padding_tail,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.cache_groups import (
    CacheGroupsMixin,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import build_page_table
from tokenspeed.runtime.utils.common import ceil_div

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


_KERNEL_SOLUTION_BY_BACKEND = {
    "mha": None,
    "fa3": "fa3",
    "fa4": "fa4",
    "triton": "triton",
    "flashinfer": "flashinfer",
}


def _slice_extend_inputs(metadata, q, k, v):
    """Remove prefill-graph padding rows before calling an attention kernel.

    The live cu-seqlens describe only real tokens. Some kernels tolerate extra
    zero rows, but others still derive work from the tensor shape. Use the
    pinned CPU mirror (sync-free) so every solution receives the same exact-row
    contract. No-op on normal unpadded forwards.
    """
    return slice_to_real_tokens(metadata.cu_extend_seq_lens_cpu[-1], q, k, v)


def mixed_batch_supported(
    spec_num_tokens: int, is_draft: bool, draft_block_decode: bool
) -> bool:
    """Whether an MHA backend with this spec configuration can serve MIXED.

    The MIXED decode tail is served as exactly one query token per row (see
    :meth:`MHAAttnBackend._forward_mixed`). That holds for the plain decode
    plan, whose ``decode_input_tokens`` is 1, but not for a speculative build:
    there each decode row is a ``spec_num_tokens``-wide verify window, and a
    draft rank additionally re-uses the extend metadata as its own decode
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


@dataclass(kw_only=True)
class MHAExtendMetadata:
    # Device-side metadata:
    # - seq_lens: total length after this step
    # - extend_seq_lens: length of new tokens
    #   cu_extend_seq_lens: the cumsum version of extend_seq_lens
    #   cu_seqlens_kv: the cumsum version of seq_lens
    # - extend_prefix_lens: length of the cached prefix tokens
    # seq_lens[i] = extend_prefix_lens[i] + extend_seq_lens[i]
    # page_table is None on the cache path (per-group page_tables route reads).
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
    # Per-group page tables (group_id -> [num_reqs, max_pages]); None on
    # the cache path (DFLASH block-decode drafts still use it).
    page_tables: dict[str, torch.Tensor] | None = None
    # Per-group KV write locations (group_id -> [num_tokens] int32),
    # built with page_tables — same groups, same lifecycle.
    out_cache_locs: dict[str, torch.Tensor] | None = None
    # MIXED split points. The batch is packed prefill-first, so the query rows
    # divide at `mixed_prefill_tokens` and each segment is served by the kernel
    # built for it -- see `_forward_mixed` for why that is worth doing.
    # `mixed_prefill_tokens` is the query-token count of the prefill segment,
    # `mixed_decode_rows` the number of one-token decode rows after it, and
    # `mixed_total_tokens` their sum, which is the real row count before any
    # prefill-graph padding. All zero on a non-mixed batch.
    mixed_prefill_tokens: int = 0
    mixed_decode_rows: int = 0
    mixed_total_tokens: int = 0


@dataclass(kw_only=True)
class MHADecodeMetadata:
    # page_table is None on the cache path (per-group page_tables route reads).
    page_table: torch.Tensor | None
    seq_lens: torch.Tensor
    # Per-group tables/write-locs; see MHAExtendMetadata.
    page_tables: dict[str, torch.Tensor] | None = None
    out_cache_locs: dict[str, torch.Tensor] | None = None


class MHAAttnBackend(CacheGroupsMixin, AttentionBackend):
    """Standard MHA backend that routes through tokenspeed_kernel attention APIs."""

    # Unconditional: safety comes from the publication rule
    # (kv_cache.recipes.publish) plus the replay
    # stale-table guard. drop the flag.
    uses_cache_groups: bool = True

    # This backend has a mixed path (`_forward_mixed`), but whether a given
    # INSTANCE can use it depends on its speculative configuration, so
    # `__init__` narrows this per instance via `mixed_batch_supported`.
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
        #
        # Without this, --kv-cache-dtype fp8_e4m3 does not start at all:
        # `triton_mha_extend_with_kvcache` and (since this change)
        # `flashinfer_paged_mha_extend_with_kvcache` declare fp16/bf16 query
        # over fp8 KV, and nothing declares fp8 query over fp8 KV, so selection
        # raises NoKernelFoundError.
        #
        # Only fa3/fa4 want the cast: their fp8 signatures declare all three
        # roles fp8. The flashinfer paged wrapper plans `q_data_type` and
        # `kv_data_type` separately, so it takes the mixed pair like Triton
        # does. Anything else (including the auto "mha" backend, which resolves
        # to one of them at selection time) is better off leaving the query at
        # activation precision.
        self.cast_query_to_fp8 = self.kernel_solution in ("fa3", "fa4")
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
        self.draft_block_decode = bool(config.draft_block_decode)

        # MIXED support is per-instance, not per-class: the class-level default
        # says only that this backend has a mixed path at all, while a
        # speculative or draft configuration cannot use it. See
        # `mixed_batch_supported`. The startup guard in
        # `validate_mixed_batch_backend` reads this off the constructed
        # backend, so a build that cannot mix refuses --enable-mixed-batch
        # instead of asserting on the first mixed step.
        self.supports_mixed_batch = mixed_batch_supported(
            self.spec_num_tokens, self.is_draft, self.draft_block_decode
        )

        # Forward metadata is initialized in the runner per forward call
        self.forward_decode_metadata: MHADecodeMetadata | None = None
        self.forward_extend_metadata: MHAExtendMetadata | None = None

        # family="state" ids (GDN/mamba) learned in init_cuda_graph_state; this backend sheds them
        self.state_group_ids: frozenset[str] = frozenset()
        # Group geometry is available to eager metadata before graph setup.
        self.group_page_sizes = dict(config.group_page_sizes or {})

    # ------------------------------------------------------------------
    # Metadata initialization
    # ------------------------------------------------------------------

    def _install_paged_extend_plan(self, metadata, extend_prefix_lens_cpu) -> None:
        """Plan the FA2 paged extend kernel once for this forward.

        The kernel can plan itself, but only by copying ``cache_seqlens`` to the
        host to key the plan -- a sync per layer, which drains the queue the
        previous layers just filled. On a 64-layer model that is 64 pipeline
        stalls per prefill forward. Everything the plan needs is already on the
        host here (prefix + extend lengths), so doing it once costs no sync at
        all. This is where SGLang plans too (``init_forward_metadata``).

        Installs on every extend forward, or clears; a plan is never allowed to
        outlive the forward it was built for, which is what lets the kernel
        match it by tensor identity alone.
        """
        if not paged_extend_planning_enabled():
            return
        clear_paged_extend_plan()
        # Per-group page tables route the reads, and `_select_page_table` gives
        # a layer its own group's, so plan one per group; the single-table path
        # has exactly one. Planning the table the kernel will not be handed is
        # what made the first version of this inert.
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
        # values for them. The kernel plans each group's wrapper from what the
        # layer actually hands it; passing guesses from here would only be
        # something for it to disagree with.
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
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        # Only consumed on the extend/mixed path; decode callers (e.g. the
        # DFLASH draft and the cuda-graph wrapper's draft decode init) omit
        # them, so they must be optional.
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        block_tables: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        # `supports_mixed_batch` (narrowed per instance in __init__) gates the
        # same condition at startup, so reaching here with an unsupported
        # configuration is an engine bug rather than a user one.
        is_mixed = forward_mode.is_mixed()
        if is_mixed and not self.supports_mixed_batch:
            raise AssertionError(
                "this MHA backend instance cannot serve mixed prefill+decode "
                "batches (speculative or draft configuration); the startup "
                "guard should have refused --enable-mixed-batch (see "
                "AttentionBackend.supports_mixed_batch)"
            )

        seq_lens = seq_lens[:bs]

        group_page_tables = self._shed_state_groups(block_tables)
        group_out_cache_locs = None
        if group_page_tables:
            # Verify keeps [bs]-row tables; only DFLASH expands rows.
            assert not (
                self.draft_block_decode and self.spec_num_tokens > 1
            ), "paged cache groups are unsupported with DFLASH block decode"
            # The cache path routes every read/write through the per-group
            # tables; a shared single page_table would be dead work.
            page_table = None
            if forward_mode.is_extend_or_mixed():
                assert extend_prefix_lens_cpu is not None
                assert extend_seq_lens_cpu is not None
                group_out_cache_locs = self._compute_extend_group_out_cache_locs(
                    group_page_tables,
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
                # the grouped-cache arm. (A chaining draft ignores these locs via
                # prefer_caller in the base forward_decode.)
                verify_tokens = self.spec_num_tokens if self.spec_num_tokens > 1 else 1
                group_out_cache_locs = self._compute_decode_group_out_cache_locs(
                    group_page_tables,
                    seq_lens,
                    self.page_size,
                    verify_tokens,
                )
            self._maybe_check_group_write_locs(
                group_page_tables, group_out_cache_locs, self.page_size
            )
            group_page_tables = self._kernel_page_tables(group_page_tables)
        else:
            page_table = build_page_table(
                req_pool_indices[:bs],
                page_table,
                self.page_size,
                self.max_context_len,
            )

        if forward_mode.is_extend_or_mixed():
            assert extend_seq_lens is not None
            assert extend_seq_lens_cpu is not None
            assert extend_prefix_lens is not None
            assert extend_prefix_lens_cpu is not None

            # A MIXED plan holds prefill rows [0, num_extends) followed by
            # decode rows [num_extends, bs) -- the scheduler partitions prefill
            # first -- and carries the extend vectors for the PREFILL rows only,
            # because model_executor slices extend_*_buf[:num_extends]. Rather
            # than synthesising a decode tail for those vectors and serving all
            # bs rows through one varlen extend call, the extend metadata is
            # built over the prefill rows alone and the decode tail gets its own
            # decode metadata; the forward then dispatches each segment to the
            # kernel built for it. See `_forward_mixed` for the measurement
            # behind that split.
            rows = num_extends if is_mixed else bs
            extend_seq_lens_all = seq_lens[:rows]

            # Create cumulative sum of the sequence lengths for Q and KV.
            extend_seq_lens = extend_seq_lens[:rows]
            extend_seq_lens_cpu = [int(x) for x in extend_seq_lens_cpu[:rows].tolist()]
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
            extend_prefix_lens = extend_prefix_lens[:rows]
            max_extend_seq_len = max(extend_seq_lens_cpu)
            max_extend_prefix_len = int(extend_prefix_lens_cpu[:rows].max().item())

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
                    if group_page_tables is None
                    else {gid: t[:rows] for gid, t in group_page_tables.items()}
                ),
                # The whole batch's write locations: the mixed forward writes KV
                # for both segments in ONE call before either attention runs.
                out_cache_locs=group_out_cache_locs,
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
                        if group_page_tables is None
                        else {
                            gid: t[num_extends:bs]
                            for gid, t in group_page_tables.items()
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
                    page_tables=group_page_tables,
                    out_cache_locs=group_out_cache_locs,
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
                    page_tables=group_page_tables,
                    out_cache_locs=group_out_cache_locs,
                )
            else:
                self.forward_decode_metadata = MHADecodeMetadata(
                    page_table=page_table,
                    seq_lens=seq_lens,
                    page_tables=group_page_tables,
                    out_cache_locs=group_out_cache_locs,
                )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        paged_cache_group_specs: Sequence = (),
        **kwargs,
    ):
        # State-family groups (GDN/mamba pages) belong to the mamba backend;
        # learn their ids from the pool's specs so every table/location path
        # here (eager, capture, replay) sheds them.
        self._learn_cache_groups(paged_cache_group_specs)

        self.cuda_graph_decode_metadata = {}
        # Per-group persistent buffers. parallels
        # cuda_graph_page_table.
        # Initialized before the DFLASH early return: replay reads the dict
        # unconditionally for the stale-table guard.
        self._init_group_graph_buffers(max_bs)
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
        # Own the cache-seqlens buffer in all non-DFLASH cases; replay copies the
        # live lengths in (verify clamps, plain decode/draft copies), so graph
        # state does not depend on the controller mutating a shared tensor.
        self.cuda_graph_seq_lens = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        cache_group_ids: tuple[str, ...] = (),
        **kwargs,
    ):
        assert not forward_mode.is_extend_or_mixed()

        # Real tables only arrive at replay; capture records metadata views
        # into the persistent per-group buffers so replay can copy_ fresh data
        # to the graph-recorded addresses.
        if cache_group_ids:
            # Verify keeps [bs]-row tables plus [bs*N] location views.
            assert not (
                self.draft_block_decode and self.spec_num_tokens > 1
            ), "cache_group_ids is unsupported with DFLASH block decode"
        page_tables, out_cache_locs = self._capture_group_views(
            bs,
            cache_group_ids,
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
                # Cache-group captures route reads through the per-group tables and
                # replay never fills a shared single page_table, so mirror the
                # eager cache path: page_table=None instead of a slice of the
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
            else:
                # Seed the owned buffer: the capture run reads it before replay.
                metadata.seq_lens.copy_(seq_lens[:bs])
        self.cuda_graph_decode_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        block_tables: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        assert not forward_mode.is_extend_or_mixed()

        # Fail loudly instead of replaying over stale/zero page tables.
        self._replay_stale_guard(bs, block_tables)

        # Every pool publishes at least one history group now, so the
        # per-group capture buffers always exist; the pre-LCM single-table
        # gather has no remaining producer.
        if not self.cuda_graph_page_tables:
            raise RuntimeError(
                "MHA replay without per-group capture buffers: the pool "
                "published no cache groups, which the LCM contract forbids"
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
            # spec_num_tokens block rows. The drafter's page table is
            # batch-ordered (row i == batch position i). The block-end seq_lens
            # are filled by the drafter inside the captured graph, so they are
            # not touched here (they re-derive from the live draft length on
            # every replay).
            base_page_table = page_table[:bs, : self.max_num_pages]
            self.cuda_graph_page_table[: bs * self.spec_num_tokens, :].view(
                bs, self.spec_num_tokens, self.max_num_pages
            ).copy_(base_page_table[:, None, :])
        else:
            # Plain decode / plain draft: copy the live lengths into our own
            # buffer (previously aliased the controller's seq_lens_buf).
            self.cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs])
        if block_tables:
            self._fill_group_graph_buffers(
                bs,
                block_tables,
                self.cuda_graph_seq_lens,
                tokens_per_req=(
                    self.spec_num_tokens if self.spec_num_tokens > 1 else 1
                ),
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
        if metadata.mixed_total_tokens > 0:
            # Prefill chunks and decode rows in one batch; each half wants a
            # different kernel, so it cannot go down either branch below.
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
        q, k, v = _slice_extend_inputs(metadata, q, k, v)
        # TODO: use a custom kernel to do downcast
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
        q, k, v = _slice_extend_inputs(metadata, q, k, v)
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
        max_seqlen_q = q.shape[0] // metadata.seq_lens.shape[0]
        output = mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=self._select_page_table(layer, metadata),
            cache_seqlens=metadata.seq_lens,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
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
            k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
            fused_fp8_set_kv_buffer(
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_loc=out_cache_loc,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
                page_size=self.page_size,
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

    def _consumer_page_size(self, group_id: str) -> int:
        return self._group_page_size(group_id)

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
