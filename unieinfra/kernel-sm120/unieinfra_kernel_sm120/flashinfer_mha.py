# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""FlashAttention-class MHA kernels for sm_120, via FlashInfer's paged wrappers.

Why this exists
---------------
On sm_120 TokenSpeed has no FlashAttention path for MHA at all:

* ``ops/attention/flash_attn`` registers FA4 only, every entry pinned to
  ``min_arch == max_arch == (10, 0)`` — datacenter Blackwell exclusively.
* The image's ``flash_attn`` distribution is an empty namespace directory with
  no modules in it.
* FlashInfer's trtllm XQA decode rejects any page size outside
  {16, 32, 64, 128}, and Qwen3.5's GDN state row forces the attention block to
  832.
* FlashInfer's AOT paged *decode* kernel rejects ``group_size 6`` at head_dim
  256 (24 query heads over 4 KV heads).

So the engine falls back to the Triton kernel in the PORTABLE band, while vLLM
runs FlashAttention-2 on the same GPU for the same model. That gap is wiring,
not hardware: FlashInfer's *general* paged prefill wrapper — the same family
vLLM's FA2 backend is built on — runs at exactly this model's geometry
(page_size 832, head_dim 256, GQA group 6), and decode is extend with a
uniform short query.

CUDA graphs
-----------
TokenSpeed captures decode in CUDA graphs, and FlashInfer's ``plan()`` cannot
run under capture: it does ``qo_indptr.to("cpu")`` plus two more device-to-host
copies, then runs a host-side C++ scheduler that ends in a ``cudaMemcpyAsync``
out of pinned memory. Any of those raises ``operation not permitted when
stream is capturing``.

The fix is the three-phase split every other engine uses — SGLang's
``init_cuda_graph_state`` / capture / replay hooks, vLLM's metadata ``build()``
ahead of ``graph.replay()``:

1. **Allocate once.** One wrapper per (batch, tokens-per-request, geometry)
   bucket, constructed with ``use_cuda_graph=True`` over caller-owned
   ``qo_indptr`` / ``paged_kv_indptr`` / ``paged_kv_indices`` /
   ``paged_kv_last_page_len`` buffers. The captured graph bakes those
   addresses.
2. **Plan outside the graph.** :func:`prepare_paged_mha_decode` runs the host
   scheduler and refreshes those buffers plus FlashInfer's device int
   workspace. It runs once before capture and once before every replay.
3. **Run inside the graph.** The kernel entry point calls only
   ``wrapper.run()``, which is pure device work over the persistent buffers.

This is sound because of what ``enable_cuda_graph`` does to FlashInfer's
scheduler (``include/flashinfer/attention/scheduler.cuh``,
``PrefillSplitQOKVIndptr`` and ``PrefillPlanImpl``). With it set,
``padded_batch_size``, ``cta_tile_q`` and every workspace offset in the
returned ``PrefillPlanInfo`` become functions of *fixed* quantities only — SM
count, head counts, head_dim, page_size, batch size, total query rows — and
never of the KV lengths; ``split_kv`` is forced true. Everything that does vary
with sequence length (``kv_chunk_size``, ``request_indices``,
``kv_tile_indices``, ``merge_indptr``, ``block_valid_mask``) is written into
the *device* int workspace, whose pointer the graph baked and whose contents
each re-plan overwrites. So the host-side ``plan_info`` frozen at capture stays
valid for every later replay, and the per-step schedule reaches the kernel
through memory instead of through the launch parameters.

``block_valid_mask`` is why ``split_kv`` must stay on: it is the only early
exit for the padded tiles of the fixed grid. Passing ``disable_split_kv=True``
would make every replay do capture-sized work.

The consequence is a hard requirement rather than a nicety: a bucket's batch
size and per-request query-token count may never change after its first plan.
Both are properties of a CUDA-graph bucket, so that holds by construction.

Without the hook
----------------
If the runtime never calls :func:`prepare_paged_mha_decode` this module still
works — outside capture the kernel plans inline on every call, exactly as a
plain eager path would. Only the captured-graph path needs the hook, and there
the kernel raises a directed error instead of a raw CUDA fault.

Known cost
----------
``plan()`` still copies three small int32 tensors to the host, which is a
device-to-host sync per *step*. vLLM and SGLang avoid it by handing FlashInfer
a host-side mirror of the sequence lengths; TokenSpeed's decode metadata has no
such mirror on the graph path, so it is left in place. It is one sync per step,
not per layer: the epoch guard below makes all attention layers of a step share
a single plan.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

# FlashInfer sizes its split-KV scratch (tmp_v/tmp_s) out of this. The scratch
# is live only within one kernel launch, so a single buffer per device serves
# every bucket. Split-KV is mandatory under CUDA graphs (block_valid_mask is
# the only early exit for the padded grid), so the scratch is not optional and
# it is large: at head_dim 256 it is tens to hundreds of MiB.
_WORKSPACE_MIN_BYTES = 128 * 1024 * 1024
_workspaces: dict[torch.device, torch.Tensor] = {}
# Buffers a growth step replaced. Already-captured graphs baked the old pointer
# as their scratch address, so those allocations must outlive the swap.
_retired_workspaces: list[torch.Tensor] = []

# Each wrapper privately owns FlashInfer's 8 MiB int workspace, and it has to:
# a captured graph reads its schedule out of that buffer, so two buckets cannot
# share one. This bound only warns — evicting a slot whose graph is still
# captured would leave that graph reading a plan nobody refreshes.
_MAX_PLANS = 48

# Bumped by prepare_paged_mha_decode(). A slot planned in the current epoch
# against the same metadata tensors is reused rather than re-planned, so the
# full-attention layers of one step share a single plan. Epoch 0 means nobody
# ever called prepare, which keeps the un-hooked path re-planning per call.
_epoch = 0


def _align16(n: int) -> int:
    return (n + 15) & ~15


def _fa2_cta_tile_q(avg_packed_qo_len: int, head_dim: int) -> int:
    """Mirror of FlashInfer's ``FA2DetermineCtaTileQ`` for Ampere and newer.

    Only used to size the workspace ahead of ``plan()``. Guessing high wastes
    memory; guessing low makes ``plan()`` raise a buffer-overflow error rather
    than corrupt anything, so this is a sizing hint, not a correctness input.
    """
    if head_dim >= 512:
        return 16 if avg_packed_qo_len <= 32 else 32
    if avg_packed_qo_len > 64 and head_dim < 256:
        return 128
    return 64 if avg_packed_qo_len > 16 else 16


def _required_workspace_bytes(
    device: torch.device,
    batch: int,
    max_seqlen_q: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Split-KV scratch one bucket will ask for, from the graph-mode schedule.

    Reproduces ``PrefillSplitQOKVIndptr`` / ``PrefillPlanImpl`` under
    ``enable_cuda_graph``: the query tile size and the padded batch size are
    fixed by the bucket's shape and the GPU's SM count, never by the KV
    lengths, which is exactly what makes the plan replay-invariant — and also
    what makes this computable before the first plan.

    Args:
        device: CUDA device the bucket runs on.
        batch: Requests in the bucket.
        max_seqlen_q: Query tokens per request.
        num_qo_heads: Query heads.
        num_kv_heads: KV heads.
        head_dim: Attention head dimension.

    Returns:
        Bytes of float workspace ``tmp_v`` plus ``tmp_s`` need.
    """
    gqa = max(num_qo_heads // num_kv_heads, 1)
    total_rows = batch * max_seqlen_q
    max_qo_len = (total_rows - batch + 1) * gqa
    cta_tile_q = _fa2_cta_tile_q(max_qo_len, head_dim)
    total_num_tiles_q = -(-(total_rows * gqa) // cta_tile_q) + batch - 1
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    # num_blocks_per_sm is hard-coded to 2 in PrefillPlanImpl.
    max_batch_size_if_split = (2 * sm_count) // num_kv_heads
    padded_batch_size = max(max_batch_size_if_split, total_num_tiles_q)
    tmp_v = num_qo_heads * padded_batch_size * cta_tile_q * head_dim * 4
    tmp_s = num_qo_heads * padded_batch_size * cta_tile_q * 4
    return _align16(tmp_v) + _align16(tmp_s)


def _workspace(device: torch.device, need_bytes: int = 0) -> torch.Tensor:
    """Per-device split-KV scratch, grown on demand and never shrunk.

    Growth keeps the displaced buffer alive in ``_retired_workspaces``: a graph
    captured before the swap recorded the old address as its scratch pointer.
    That is safe because the buffer holds nothing across launches — it is pure
    per-kernel scratch — so an old graph writing into a retired allocation is
    correct, merely wasteful of address space.

    Args:
        device: CUDA device.
        need_bytes: Minimum size this caller requires.

    Returns:
        The device's current workspace tensor.
    """
    buf = _workspaces.get(device)
    if buf is not None and buf.numel() >= need_bytes:
        return buf

    size = max(need_bytes, _WORKSPACE_MIN_BYTES)
    if buf is not None:
        size = max(size, buf.numel() * 2)
        _retired_workspaces.append(buf)
        logger.info(
            "unieinfra-kernel-sm120: growing FlashInfer workspace on %s to %d MiB; "
            "the previous %d MiB stays resident for graphs already captured "
            "against it",
            device,
            size >> 20,
            buf.numel() >> 20,
        )
    # Outside inference mode for the same reason as the slot buffers: FlashInfer
    # writes its schedule into this workspace on every re-plan, and the re-plan
    # happens outside the captured region.
    with torch.inference_mode(False):
        grown = torch.empty(size, dtype=torch.uint8, device=device)
    _workspaces[device] = grown
    for slot in _plans.values():
        if slot.qo_indptr_buf.device == device:
            slot.wrapper.reset_workspace_buffer(
                grown, slot.wrapper._int_workspace_buffer
            )
    return grown


@dataclass
class _PlanSlot:
    """One FlashInfer graph-mode wrapper plus the buffers a captured graph bakes.

    Attributes:
        wrapper: ``BatchPrefillWithPagedKVCacheWrapper`` built with
            ``use_cuda_graph=True``. Its ``run()`` reads only the four ``*_buf``
            tensors below, the shared float workspace, and its own device int
            workspace.
        qo_indptr_buf: Query offsets, ``[batch + 1]``. Decode packs a uniform
            number of tokens per request, so this is filled at construction and
            never changes.
        kv_indptr_buf: Page offsets per request, ``[batch + 1]``.
        kv_indices_buf: Flattened page ids, ``[batch * max_pages_per_seq]``.
        last_page_len_buf: Tokens used in each request's final page, ``[batch]``.
        scratch_indptr: Staging for ``kv_indptr_buf``; ``plan()`` copies staging
            into the wrapper's buffers, so keeping them distinct avoids handing
            it a tensor that aliases its own destination.
        scratch_indices: Staging for ``kv_indices_buf`` with one extra trailing
            slot that absorbs the scatter writes of out-of-range page columns.
        scratch_last_page_len: Staging for ``last_page_len_buf``.
        causal: Mask mode frozen at construction. ``run()`` bakes it into the
            captured launch, so it may not vary across replays.
        max_pages_per_seq: Page-table width this slot was built for.
        epoch: ``_epoch`` value at the last successful plan.
        fingerprint: Identity of the metadata tensors that plan read.
    """

    wrapper: object
    qo_indptr_buf: torch.Tensor
    kv_indptr_buf: torch.Tensor
    kv_indices_buf: torch.Tensor
    last_page_len_buf: torch.Tensor
    scratch_indptr: torch.Tensor
    scratch_indices: torch.Tensor
    scratch_last_page_len: torch.Tensor
    causal: bool
    max_pages_per_seq: int
    epoch: int = 0
    fingerprint: tuple = field(default_factory=tuple)


_plans: dict[tuple, _PlanSlot] = {}


def _plan_key(
    device: torch.device,
    batch: int,
    max_seqlen_q: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_pages_per_seq: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> tuple:
    """Identity of a plan slot.

    Every field is frozen for the lifetime of a CUDA-graph bucket.
    ``max_seqlen_q`` is in the key because it fixes both the mask mode and the
    total query-row count, which the scheduler's graph-mode upper bounds are
    derived from.
    """
    return (
        device.type,
        device.index,
        batch,
        max_seqlen_q,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_pages_per_seq,
        q_dtype,
        kv_dtype,
    )


def _geometry_prefix(key: tuple) -> tuple:
    """The part of a plan key a caller can match without having seen ``q``.

    Drops ``max_seqlen_q`` (index 3). The runtime knows the batch and the cache
    geometry at metadata time but not necessarily how many query tokens per
    request this step carries, so prepare re-plans every ``max_seqlen_q``
    variant already registered for the bucket.
    """
    return key[:3] + key[4:]


def _fill_paged_kv_layout(
    slot: _PlanSlot,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
) -> None:
    """Write TokenSpeed's page table into FlashInfer's CSR page index.

    Builds ``kv_indptr`` / ``kv_indices`` / ``last_page_len`` entirely on
    device. The obvious formulation — ``page_table[mask]`` — needs the kept
    element count on the host, which costs a device-to-host sync every call.
    Instead the valid page ids are scattered to their CSR destinations and
    every out-of-range column is aimed at one trailing scratch slot that
    nothing reads.

    Args:
        slot: Destination staging buffers.
        page_table: Page ids per request, ``[batch, max_pages_per_seq]``.
        cache_seqlens: Visible KV length per request, ``[batch]``.
        page_size: Tokens per page.
    """
    batch = slot.last_page_len_buf.shape[0]
    max_pages = slot.max_pages_per_seq
    device = page_table.device

    # CUDA-graph padding rows can carry length 0. Clamping to 1 keeps the CSR
    # well formed — FlashInfer wants last_page_len in [1, page_size] — at the
    # cost of one dummy page read whose output is discarded along with the rest
    # of the padding tail. That is the same bargain the Triton kernel makes.
    lens = cache_seqlens[:batch].to(torch.int32).clamp_min(1)
    pages = ((lens + (page_size - 1)) // page_size).clamp(1, max_pages)

    slot.scratch_indptr[:1].zero_()
    slot.scratch_indptr[1:] = torch.cumsum(pages, dim=0, dtype=torch.int32)

    cols = torch.arange(max_pages, device=device, dtype=torch.int32)
    valid = cols.unsqueeze(0) < pages.unsqueeze(1)
    dest = slot.scratch_indptr[:batch].unsqueeze(1) + cols.unsqueeze(0)
    trash = slot.scratch_indices.numel() - 1
    dest = torch.where(valid, dest, torch.full_like(dest, trash))
    slot.scratch_indices.scatter_(
        0,
        dest.reshape(-1).to(torch.int64),
        page_table[:batch, :max_pages].reshape(-1).to(torch.int32),
    )

    last = lens - (pages - 1) * page_size
    slot.scratch_last_page_len.copy_(last.clamp(1, page_size))


def _make_slot(
    device: torch.device,
    batch: int,
    max_seqlen_q: int,
    max_pages_per_seq: int,
    causal: bool,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> _PlanSlot:
    """Allocate a graph-mode wrapper and its persistent buffers for one bucket.

    Allocation is forced out of inference mode. TokenSpeed builds and captures
    the model under ``torch.inference_mode``, and a tensor created there is an
    *inference tensor* for its whole life: any later in-place write from outside
    that mode raises ``Inplace update to inference tensor outside InferenceMode``.
    These buffers exist precisely to be rewritten before every replay, and the
    re-plan runs from the runtime's metadata hooks — outside the captured
    region, and not necessarily under inference mode. Creating them as ordinary
    tensors is what makes the three-phase split legal.
    """
    import flashinfer

    with torch.inference_mode(False):
        return _make_slot_buffers(
            flashinfer,
            device,
            batch,
            max_seqlen_q,
            max_pages_per_seq,
            causal,
            num_qo_heads,
            num_kv_heads,
            head_dim,
        )


def _make_slot_buffers(
    flashinfer,
    device: torch.device,
    batch: int,
    max_seqlen_q: int,
    max_pages_per_seq: int,
    causal: bool,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> _PlanSlot:
    """Body of :func:`_make_slot`, split out so the mode guard wraps all of it."""
    qo_indptr = torch.arange(
        0,
        (batch + 1) * max_seqlen_q,
        max_seqlen_q,
        dtype=torch.int32,
        device=device,
    )
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(
        batch * max_pages_per_seq, dtype=torch.int32, device=device
    )
    last_page_len = torch.ones(batch, dtype=torch.int32, device=device)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace(
            device,
            _required_workspace_bytes(
                device, batch, max_seqlen_q, num_qo_heads, num_kv_heads, head_dim
            ),
        ),
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr,
        paged_kv_indptr_buf=kv_indptr,
        paged_kv_indices_buf=kv_indices,
        paged_kv_last_page_len_buf=last_page_len,
    )

    if len(_plans) >= _MAX_PLANS:
        logger.warning(
            "unieinfra-kernel-sm120: %d FlashInfer decode plan slots live, each "
            "holding a private 8 MiB scheduling workspace; expected roughly one "
            "per CUDA-graph bucket",
            len(_plans),
        )

    return _PlanSlot(
        wrapper=wrapper,
        qo_indptr_buf=qo_indptr,
        kv_indptr_buf=kv_indptr,
        kv_indices_buf=kv_indices,
        last_page_len_buf=last_page_len,
        scratch_indptr=torch.zeros(batch + 1, dtype=torch.int32, device=device),
        # +1 trailing slot: the scatter aims every out-of-range page column here.
        scratch_indices=torch.zeros(
            batch * max_pages_per_seq + 1, dtype=torch.int32, device=device
        ),
        scratch_last_page_len=torch.ones(batch, dtype=torch.int32, device=device),
        causal=causal,
        max_pages_per_seq=max_pages_per_seq,
    )


def _plan_slot(
    slot: _PlanSlot,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_scale: float,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> None:
    """Run FlashInfer's host-side scheduler for one slot.

    Never legal while the stream is capturing; callers gate on that.
    """
    global _epoch

    batch = slot.last_page_len_buf.shape[0]
    _fill_paged_kv_layout(slot, page_table, cache_seqlens, page_size)
    slot.wrapper.plan(
        slot.qo_indptr_buf,
        slot.scratch_indptr,
        slot.scratch_indices[: batch * slot.max_pages_per_seq],
        slot.scratch_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=slot.causal,
        sm_scale=softmax_scale,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
    slot.epoch = _epoch
    slot.fingerprint = (
        page_table.data_ptr(),
        cache_seqlens.data_ptr(),
        int(page_table.shape[1]),
    )


def prepare_paged_mha_decode(
    *,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    max_seqlen_q: int | None = None,
    softmax_scale: float | None = None,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> None:
    """Run FlashInfer's host-side scheduling for the decode step about to run.

    Call this from outside any CUDA-graph capture or replay: immediately before
    ``torch.cuda.graph(...)`` at capture, and before ``graph.replay()`` on
    every step. It refreshes the persistent metadata buffers and the device int
    workspace that the captured ``run()`` reads, which is the entire reason a
    FlashInfer wrapper can live inside a graph.

    Calling it is optional. When nothing calls it the kernel plans inline on
    every invocation, which is correct everywhere except inside a captured
    graph.

    Args:
        page_table: Page ids per request, ``[batch, max_pages_per_seq]``. Must
            be the same tensor the kernel will be handed — the graph baked
            nothing about it, but the schedule is derived from its contents.
        cache_seqlens: Visible KV length per request, ``[batch]``, including
            the tokens being decoded now. Also the tensor the kernel will see.
        page_size: Tokens per page.
        num_q_heads: Query heads on this rank.
        num_kv_heads: KV heads on this rank.
        head_dim: Attention head dimension.
        q_dtype: Query dtype.
        kv_dtype: KV cache dtype.
        max_seqlen_q: Query tokens per request. ``None`` re-plans every
            per-request token count already registered for this bucket, which
            is what a caller that does not separately track its speculative
            width should pass.
        softmax_scale: QK scale; defaults to ``1/sqrt(head_dim)``.
        window_left: Sliding window. Anything but -1 means selection will not
            pick this kernel, so preparation is skipped.
        logit_cap: Logit soft cap. Non-zero is likewise not this kernel's.
        sinks: Attention sinks. Not None is likewise not this kernel's.
        return_lse: Accepted and ignored; the schedule does not depend on it.

    Returns:
        None. A bucket with no registered slot is left alone unless
        ``max_seqlen_q`` names the one to build; otherwise the first eager or
        warm-up call creates it.

    Raises:
        RuntimeError: If the stream is capturing. Planning there is precisely
            the thing this function exists to hoist out.
    """
    del return_lse
    if window_left >= 0 or logit_cap != 0.0 or sinks is not None:
        # Selection routes these to Triton, so there is nothing here to plan.
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "prepare_paged_mha_decode() ran while the stream was capturing. "
            "FlashInfer's plan() does host-side scheduling and a pinned-memory "
            "copy, so it must run before torch.cuda.graph(...) and before each "
            "graph.replay(), never inside either."
        )

    global _epoch
    _epoch += 1

    device = page_table.device
    batch = int(cache_seqlens.shape[0])
    max_pages_per_seq = int(page_table.shape[1])
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    def key_for(seqlen_q: int) -> tuple:
        return _plan_key(
            device,
            batch,
            seqlen_q,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_pages_per_seq,
            q_dtype,
            kv_dtype,
        )

    if max_seqlen_q is not None:
        keys = [key_for(int(max_seqlen_q))]
        if keys[0] not in _plans:
            _plans[keys[0]] = _make_slot(
                device,
                batch,
                int(max_seqlen_q),
                max_pages_per_seq,
                causal=int(max_seqlen_q) > 1,
                num_qo_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
    else:
        wanted = _geometry_prefix(key_for(0))
        keys = [key for key in _plans if _geometry_prefix(key) == wanted]

    for key in keys:
        _plan_slot(
            _plans[key],
            page_table,
            cache_seqlens,
            page_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            softmax_scale,
            q_dtype,
            kv_dtype,
        )


def _paged_kv_layout(
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert TokenSpeed's page table into FlashInfer's ragged page index form.

    Used by the extend path, which is never captured and can therefore afford
    the compacting gather.

    Args:
        page_table: Page ids per sequence, shape [batch, max_pages_per_seq].
        cache_seqlens: Visible KV length per sequence, shape [batch].
        page_size: Tokens per page.

    Returns:
        ``(kv_indptr, kv_indices, last_page_len)`` — the CSR-style page index
        FlashInfer plans against.
    """
    pages_per_seq = (cache_seqlens + page_size - 1) // page_size
    kv_indptr = torch.zeros(
        cache_seqlens.shape[0] + 1, dtype=torch.int32, device=page_table.device
    )
    torch.cumsum(pages_per_seq, dim=0, out=kv_indptr[1:])

    max_pages = page_table.shape[1]
    keep = torch.arange(max_pages, device=page_table.device).unsqueeze(
        0
    ) < pages_per_seq.unsqueeze(1)
    kv_indices = page_table[keep].to(torch.int32)

    # A sequence occupying a whole number of pages ends on a full last page,
    # not an empty one; clamp rather than let 0 through.
    last_page_len = cache_seqlens - (pages_per_seq - 1) * page_size
    last_page_len = last_page_len.clamp(min=1, max=page_size).to(torch.int32)
    return kv_indptr, kv_indices, last_page_len


def _absorb_query_padding(
    q: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    last_page_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extend the ragged index so it covers padding rows at the end of ``q``.

    TokenSpeed pads ``q`` up to a CUDA-graph bucket, so ``q.shape[0]`` can exceed
    ``qo_indptr[-1]``. The in-tree Triton kernel ignores the surplus through its
    grid bounds; FlashInfer validates the two against each other and refuses the
    call with ``q.shape[0] (N) does not match qo_indptr[-1] (M)``. That surfaces
    as a server crash during CUDA-graph capture, not as a unit-test failure,
    which is how it reached the engine the first time and how it came back after
    a rewrite dropped it.

    Slicing ``q`` down to the real token count would need ``qo_indptr[-1]`` as a
    Python int, and that device-to-host read is exactly what this path must not
    do. Instead the ragged index gains one more sequence covering the padding
    rows, pointed at the null page so FlashInfer has something well-formed to
    attend over; its output lands in rows the caller discards. With no padding
    the extra segment is empty and costs nothing.

    Args:
        q: Query tensor, possibly with trailing padding rows.
        qo_indptr: Query cumulative lengths, ``[batch + 1]``, int32.
        kv_indptr: Page-index offsets per sequence, ``[batch + 1]``, int32.
        kv_indices: Flat page ids, int32.
        last_page_len: Tokens used in each sequence's final page, ``[batch]``.

    Returns:
        The four tensors, each extended by one sequence.
    """
    dev = q.device
    qo_indptr = torch.cat(
        [qo_indptr, torch.tensor([q.shape[0]], dtype=torch.int32, device=dev)]
    )
    kv_indptr = torch.cat([kv_indptr, (kv_indptr[-1] + 1).reshape(1)])
    kv_indices = torch.cat([kv_indices, torch.zeros(1, dtype=torch.int32, device=dev)])
    last_page_len = torch.cat(
        [last_page_len, torch.ones(1, dtype=torch.int32, device=dev)]
    )
    return qo_indptr, kv_indptr, kv_indices, last_page_len


def flashinfer_paged_mha_extend(
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
    """MHA extend against a paged KV cache, on FlashInfer's paged prefill kernel.

    Args:
        q: Queries, [total_q, num_q_heads, head_dim].
        cu_seqlens_q: Query cumulative lengths, [batch + 1]. Used directly as
            FlashInfer's ``qo_indptr``.
        cu_seqlens_kv: Unused here; the page index is derived from
            ``page_table`` and ``cache_seqlens`` instead.
        k_cache: Paged keys, [num_pages, page_size, num_kv_heads, head_dim].
        v_cache: Paged values, same shape as ``k_cache``.
        page_table: Page ids per sequence, [batch, max_pages_per_seq].
        cache_seqlens: Visible KV length per sequence, [batch].
        max_seqlen_q: Longest query in the batch.
        max_seqlen_k: Longest visible KV length in the batch.
        is_causal: Whether queries are a causal suffix of the cached KV.
        window_left: Sliding window. Only -1 (full attention) is registered.
        logit_cap: Logit soft cap. Only 0.0 (off) is registered.
        sinks: Attention sinks. Only None is registered.
        return_lse: Whether to also return log-sum-exp, [total_q, num_q_heads].
        softmax_scale: QK scale; defaults to ``1/sqrt(head_dim)``.
        q_scale: Unused — the block-scaled path is not registered.
        k_scale: Unused.
        v_scale: Unused.

    Returns:
        The attention output shaped like ``q``, or ``(out, lse)`` when
        ``return_lse``.
    """
    import flashinfer

    page_size = k_cache.shape[1]
    num_qo_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # A plan built by prepare() for exactly this step is reused; every
    # full-attention layer of a step sees identical shapes, so re-planning per
    # layer recomputes one answer 16 times at 5.9-8.1 ms each. The key carries
    # every argument the schedule depends on, so a mismatch falls through to
    # planning inline rather than reusing a schedule built for other shapes.
    plan_key = (
        cu_seqlens_q.data_ptr(),
        int(cu_seqlens_q.shape[0]),
        page_table.data_ptr(),
        tuple(page_table.shape),
        cache_seqlens.data_ptr(),
        page_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        int(q.shape[0]),
        bool(is_causal),
        float(softmax_scale),
        q.dtype,
        k_cache.dtype,
        bool(return_lse),
    )
    wrapper = _planned_extend_wrapper(plan_key)

    if wrapper is None:
        kv_indptr, kv_indices, last_page_len = _paged_kv_layout(
            page_table, cache_seqlens, page_size
        )
        qo_indptr, kv_indptr, kv_indices, last_page_len = _absorb_query_padding(
            q, cu_seqlens_q.to(torch.int32), kv_indptr, kv_indices, last_page_len
        )
        batch = max(int(cu_seqlens_q.shape[0]) - 1, 1)
        need = _required_workspace_bytes(
            q.device,
            batch,
            max(int(q.shape[0]) // batch, 1),
            num_qo_heads,
            num_kv_heads,
            head_dim,
        )
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _workspace(q.device, need), "NHD"
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=is_causal,
            sm_scale=softmax_scale,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
        )
    out = wrapper.run(q, (k_cache, v_cache), return_lse=return_lse)
    if return_lse:
        return out[0], out[1]
    return out


def flashinfer_paged_mha_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MHA decode against a paged KV cache, on the same paged prefill kernel.

    Decode is extend with a uniform, usually single-token query, so the prefill
    wrapper serves it directly — the same unification vLLM's FA2 backend uses.
    Going through the prefill wrapper also sidesteps both FlashInfer decode
    limitations that block this model on sm_120: the trtllm XQA path's page-size
    whitelist, and the AOT decode kernel's ``Unsupported group_size: 6`` at
    head_dim 256.

    Only ``wrapper.run()`` happens here. Host-side scheduling lives in
    :func:`prepare_paged_mha_decode`, which the runtime calls outside the
    captured region; see the module docstring. When nothing calls it, this
    falls back to planning inline, which is correct outside a graph.

    Args:
        q: Queries, [batch * max_seqlen_q, num_q_heads, head_dim]. Rows past
            the live request count are CUDA-graph padding; they are computed
            and discarded rather than detected, because detecting them would
            need a device-to-host sync.
        k_cache: Paged keys, [num_pages, page_size, num_kv_heads, head_dim].
        v_cache: Paged values, same shape as ``k_cache``.
        page_table: Page ids per sequence, [batch, max_pages_per_seq].
        cache_seqlens: Visible KV length per sequence including the tokens
            being decoded now, [batch].
        max_seqlen_k: Longest visible KV length.
        max_seqlen_q: Query tokens per request — 1 for ordinary decode, the
            draft length under speculative decoding.
        window_left: Sliding window. Only -1 is registered.
        logit_cap: Logit soft cap. Only 0.0 is registered.
        sinks: Attention sinks. Only None is registered.
        return_lse: Whether to also return log-sum-exp.
        softmax_scale: QK scale; defaults to ``1/sqrt(head_dim)``.
        q_scale: Unused — the block-scaled path is not registered.
        k_scale: Unused.
        v_scale: Unused.

    Returns:
        The attention output shaped like ``q``, or ``(out, lse)``.

    Raises:
        RuntimeError: If reached under CUDA-graph capture with no fresh plan
            for this bucket. That means the runtime is missing the prepare
            hook; the alternative is an opaque ``operation not permitted when
            stream is capturing`` from inside FlashInfer, or worse, a silently
            stale schedule.
    """
    page_size = k_cache.shape[1]
    batch = int(cache_seqlens.shape[0])
    num_qo_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[-1]
    max_pages_per_seq = int(page_table.shape[1])
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    key = _plan_key(
        q.device,
        batch,
        max_seqlen_q,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_pages_per_seq,
        q.dtype,
        k_cache.dtype,
    )
    slot = _plans.get(key)
    capturing = torch.cuda.is_current_stream_capturing()

    if slot is None:
        if capturing:
            raise RuntimeError(
                "FlashInfer paged MHA decode was reached under CUDA-graph "
                f"capture with no plan for bucket {key}. Call "
                "unieinfra_kernel_sm120.flashinfer_mha.prepare_paged_mha_decode()"
                " before capture and before every replay — FlashInfer's plan() "
                "cannot run while the stream is capturing."
            )
        slot = _make_slot(
            q.device,
            batch,
            max_seqlen_q,
            max_pages_per_seq,
            causal=max_seqlen_q > 1,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        _plans[key] = slot

    fresh = (
        slot.epoch != 0
        and slot.epoch == _epoch
        and slot.fingerprint
        == (page_table.data_ptr(), cache_seqlens.data_ptr(), max_pages_per_seq)
    )
    if not fresh:
        if capturing:
            # Running on a stale schedule would be worse than faulting, because
            # it would look like it worked.
            raise RuntimeError(
                "FlashInfer paged MHA decode is capturing against a stale plan "
                f"for bucket {key}. prepare_paged_mha_decode() must run "
                "immediately before torch.cuda.graph(...), over the same "
                "page_table and cache_seqlens tensors the captured kernel reads."
            )
        _plan_slot(
            slot,
            page_table,
            cache_seqlens,
            page_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            softmax_scale,
            q.dtype,
            k_cache.dtype,
        )

    out = slot.wrapper.run(q, (k_cache, v_cache), return_lse=return_lse)
    if return_lse:
        return out[0], out[1]
    return out


# The dispatch layer reaches the hook through the selected kernel object, so it
# hangs off the kernel itself rather than needing its own registry family.
flashinfer_paged_mha_decode.prepare = prepare_paged_mha_decode


# --- extend: one plan per step, not one per layer -------------------------
#
# The decode hoist above exists because CUDA graph capture forbids a host pass.
# This one exists because the host pass is expensive: FlashInfer's plan() costs
# 5.9-8.1 ms per call on sm_120 against a run() of 0.67-7.1 ms, and the extend
# path called it inside all 16 full-attention layers of every step. The plan is
# a function of the batch's shapes alone, so every one of those calls recomputed
# the same answer.
#
# Extend is not captured (TokenSpeed breaks its prefill graph around attention),
# so no persistent-buffer machinery is needed here — a plain memo keyed on the
# step is enough. What makes the memo safe is that it is only ever populated by
# prepare(), which the runtime calls once per step from outside the layer loop;
# the kernel never caches on its own initiative. If the runtime does not call
# prepare, the kernel plans inline exactly as before and stays correct, just
# slow.
_extend_plan: dict | None = None


def prepare_paged_mha_extend(
    *,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    total_q: int,
    is_causal: bool = False,
    softmax_scale: float | None = None,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> None:
    """Plan the upcoming extend once, for every layer of this step to reuse.

    Called by the runtime from ``mha_extend_prepare`` before the layers run. It
    builds the ragged page index and runs FlashInfer's host scheduling pass,
    then stores the planned wrapper together with the identity of the tensors it
    was planned against.

    Args:
        cu_seqlens_q: Query cumulative lengths, [batch + 1].
        page_table: Page table for this step, [batch, max_pages_per_seq].
        cache_seqlens: Visible KV lengths, [batch].
        page_size: Tokens per page.
        num_q_heads: Query heads on this rank.
        num_kv_heads: KV heads on this rank.
        head_dim: Attention head dimension.
        q_dtype: Query dtype the layers will pass.
        kv_dtype: KV cache dtype.
        total_q: Query rows including CUDA-graph padding.
        is_causal: Whether the extend is causal.
        softmax_scale: QK scale, or None for ``1/sqrt(head_dim)``.
        window_left: Unsupported here; a non-default value clears the plan so
            the kernel falls back to planning inline.
        logit_cap: Unsupported here, as ``window_left``.
        sinks: Unsupported here, as ``window_left``.
        return_lse: Whether the layers will ask for LSE.
    """
    global _extend_plan
    import flashinfer

    # Features this kernel does not implement are declared unsupported at
    # registration, so selection should never route them here. Clearing rather
    # than raising keeps a surprise routing correct at the cost of speed.
    if window_left >= 0 or logit_cap != 0.0 or sinks is not None:
        _extend_plan = None
        return

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    kv_indptr, kv_indices, last_page_len = _paged_kv_layout(
        page_table, cache_seqlens, page_size
    )
    qo_indptr = cu_seqlens_q.to(torch.int32)
    if total_q > 0:
        pad = torch.tensor([total_q], dtype=torch.int32, device=qo_indptr.device)
        qo_indptr = torch.cat([qo_indptr, pad])
        kv_indptr = torch.cat([kv_indptr, (kv_indptr[-1] + 1).reshape(1)])
        kv_indices = torch.cat(
            [kv_indices, torch.zeros(1, dtype=torch.int32, device=kv_indices.device)]
        )
        last_page_len = torch.cat(
            [
                last_page_len,
                torch.ones(1, dtype=torch.int32, device=last_page_len.device),
            ]
        )

    # Size the split-KV scratch for THIS shape before planning. Extend carries
    # far longer queries than decode, so the shared minimum is not enough:
    # planning a 4096-token prefill against the 128 MiB floor raises
    # "Buffer overflow ... batch_prefill_tmp_v" and takes the engine down.
    batch = max(int(cu_seqlens_q.shape[0]) - 1, 1)
    need = _required_workspace_bytes(
        page_table.device,
        batch,
        max(total_q // batch, 1),
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace(page_table.device, need), "NHD"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=is_causal,
        sm_scale=softmax_scale,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
    _extend_plan = {
        "wrapper": wrapper,
        # Identity of what was planned. A layer whose arguments disagree with
        # any of these is not the step that was prepared, and must not silently
        # reuse a schedule built for different shapes.
        "key": (
            cu_seqlens_q.data_ptr(),
            int(cu_seqlens_q.shape[0]),
            page_table.data_ptr(),
            tuple(page_table.shape),
            cache_seqlens.data_ptr(),
            page_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            total_q,
            bool(is_causal),
            float(softmax_scale),
            q_dtype,
            kv_dtype,
            bool(return_lse),
        ),
    }


def _planned_extend_wrapper(key: tuple):
    """Return the wrapper prepared for exactly this call, or None."""
    plan = _extend_plan
    if plan is None or plan["key"] != key:
        return None
    return plan["wrapper"]


prepare_paged_mha_extend.__module__ = __name__
flashinfer_paged_mha_extend.prepare = prepare_paged_mha_extend
