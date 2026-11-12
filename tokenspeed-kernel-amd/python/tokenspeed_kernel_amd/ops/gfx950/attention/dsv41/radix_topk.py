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

"""GFX950 one-CTA-per-row radix top-k for DeepSeek V4.1 CSA2 selection.

One kernel serves both CSA2 selections of the indexer: the top-``topk`` rows
of each query's logits (``launch_gluon_dsv41_index_topk_radix_gfx950``) and the
top-``candidate_topk`` 8-row blocks by block maximum
(``launch_gluon_dsv41_index_block_radix_gfx950``). Both write position-sorted
ids and keep the lowest positions among equal-score boundary ties, the same
contracts as ``launch_gluon_dsv41_index_topk_select_gfx950`` and
``index_block_select``.

The algorithm follows the one-block radix select of AITER's
``radix_topk_one_block`` (long-row variant, digits 14 + 10 + 8 on gfx950):

1. Pass 1 reads the row once and histograms the high 14 bits of the order key
   into a 16K-bin LDS histogram (the narrow buckets keep the next step small).
2. Pass 2 reads the row again (back to front, so the tail is still in cache):
   keys above the K-th key's bucket go straight to the pick list; keys in that
   bucket are compacted into LDS with their positions.
3. The low 10 + 8 bits are resolved over the compacted bucket in LDS; equal
   boundary keys keep the lowest positions (a radix select on ``~position``).
4. A position bitmap turns the unordered picks into ascending order.

If the selected bucket overflows the LDS buffer, the low digits and a
position-order compaction re-scan the row instead (one row pass each).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41.indexer import (
    _select_order_key,
    _select_pick,
    _select_popcount,
)

__all__ = [
    "launch_gluon_dsv41_index_block_radix_gfx950",
    "launch_gluon_dsv41_index_topk_radix_gfx950",
]

# 14 + 10 + 8 key bits, the gfx950 split of AITER's long-row radix top-k.
_HIGH_BITS = gl.constexpr(14)
_MID_BITS = gl.constexpr(10)
_LOW_BITS = gl.constexpr(8)
_HIGH_SHIFT = gl.constexpr(_MID_BITS.value + _LOW_BITS.value)
# Bucket keys compacted in LDS (keys plus positions: 64 KiB, aliasing the
# 16K-bin pass-1 histogram), so a CTA holds about 77 KiB and two fit a CU.
_CAPACITY = gl.constexpr(8192)
_CAPACITY_CHUNK = gl.constexpr(4096)
# Rows (or block rows) up to 32 * _BITMAP_WORDS order their picks with a
# position bitmap; wider rows use an all-pairs rank sort.
_BITMAP_WORDS = gl.constexpr(4096)
_LATE_BINS = gl.constexpr(1 << _MID_BITS.value)
_NUM_WARPS = 16
# 16-byte vectors in flight per thread per row-scan step.
_REPEAT = 2
assert _LOW_BITS.value <= _MID_BITS.value
assert 2 * _CAPACITY.value == 1 << _HIGH_BITS.value
assert 2 * _BITMAP_WORDS.value + 2048 <= 1 << _HIGH_BITS.value


def _radix_launch_metadata(grid, kernel, args):
    """Two row passes; the picks add their id loads and stores."""
    queries = grid[0]
    return {
        "name": kernel.name,
        "bytes": queries * (2 * args["width"] * 4 + args["take"] * (4 + 8)),
    }


@gluon.jit
def _radix_tile(
    row,
    start,
    n,
    limit,
    exact,
    latest,
    ELEMS: gl.constexpr,
    BLOCK_MAX: gl.constexpr,
    ALIGNED: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Element ids, order keys and finite flags of elements [start, start + ELEMS).

    Without ``BLOCK_MAX`` an element is one logits column. With it, element e
    is the maximum of columns [8e, 8e + 8) below ``exact`` (the visible
    length), and the newest visible block ``latest`` ranks above every other
    finite block.
    """
    if BLOCK_MAX:
        LB: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [NUM_WARPS, 1], [1, 0])
        elem = start + gl.arange(0, ELEMS, layout=gl.SliceLayout(1, LB))
        lane = gl.arange(0, 8, layout=gl.SliceLayout(0, LB))
        col = gl.expand_dims(elem * 8, 1) + gl.expand_dims(lane, 0)
        values = gl.amd.cdna4.buffer_load(
            row, col, mask=col < limit, other=-float("inf")
        )
        if ALIGNED:
            values = gl.where(col < exact, values, -float("inf"))
        x = gl.max(values, axis=1)
        x = gl.where((elem == latest) & (x > -float("inf")), float("inf"), x)
        valid = (elem < n) & (x > -float("inf"))
        # The same element-per-thread distribution as a plain 1-D layout, which
        # the position-order scan of the overflow path supports.
        L1: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
        elem = gl.convert_layout(elem, L1)
        x = gl.convert_layout(x, L1)
        valid = gl.convert_layout(valid, L1)
    else:
        L: gl.constexpr = gl.BlockedLayout([4], [64], [NUM_WARPS], [0])
        elem = start + gl.arange(0, ELEMS, layout=L)
        x = gl.amd.cdna4.buffer_load(row, elem, mask=elem < limit, other=-float("inf"))
        if ALIGNED:
            x = gl.where(elem < exact, x, -float("inf"))
        valid = x > -float("inf")
    return elem, _select_order_key(x), valid


@gluon.jit
def _radix_row_level(
    row,
    n,
    limit,
    exact,
    latest,
    prefix,
    remaining,
    hist_smem,
    SHIFT: gl.constexpr,
    BITS: gl.constexpr,
    ELEMS: gl.constexpr,
    BLOCK_MAX: gl.constexpr,
    ALIGNED: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """One later digit from a row scan (overflow path): the next prefix and rank."""
    LH: gl.constexpr = gl.BlockedLayout(
        [_LATE_BINS // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    hist_smem.store(gl.zeros([_LATE_BINS], gl.int32, layout=LH))
    gl.barrier()
    for c0 in range(0, n, ELEMS):
        _, key, valid = _radix_tile(
            row, c0, n, limit, exact, latest, ELEMS, BLOCK_MAX, ALIGNED, NUM_WARPS
        )
        match = valid & ((key >> (SHIFT + BITS)) == prefix)
        digit = ((key >> SHIFT) & ((1 << BITS) - 1)).to(gl.int32)
        hist_smem.atomic_scatter_add(
            gl.full(digit.shape, 1, gl.int32, digit.type.layout),
            digit,
            axis=0,
            mask=match,
        )
    gl.barrier()
    pick, higher, _ = _select_pick(
        hist_smem.load(LH), gl.arange(0, _LATE_BINS, layout=LH), remaining
    )
    return (prefix << BITS) | pick.to(gl.uint32), remaining - higher


@gluon.jit
def _radix_bucket_level(
    keys_smem,
    cols_smem,
    bucket,
    match_key,
    prefix,
    remaining,
    hist_smem,
    SHIFT: gl.constexpr,
    BITS: gl.constexpr,
    TIES: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """One digit over the ``bucket`` keys compacted in LDS.

    Without ``TIES`` the digit is of the order key; with it, of ``~position``
    among the keys equal to ``match_key`` (lowest positions rank highest).
    ``prefix`` holds the digit-key bits above this digit.
    """
    LK: gl.constexpr = gl.BlockedLayout(
        [_CAPACITY_CHUNK // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    LH: gl.constexpr = gl.BlockedLayout(
        [_LATE_BINS // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    hist_smem.store(gl.zeros([_LATE_BINS], gl.int32, layout=LH))
    gl.barrier()
    ones = gl.full([_CAPACITY_CHUNK], 1, gl.int32, layout=LK)
    for ch in gl.static_range(_CAPACITY // _CAPACITY_CHUNK):
        if ch * _CAPACITY_CHUNK < bucket:
            i = ch * _CAPACITY_CHUNK + gl.arange(0, _CAPACITY_CHUNK, layout=LK)
            keys = (
                keys_smem.slice(ch * _CAPACITY_CHUNK, _CAPACITY_CHUNK)
                .load(LK)
                .to(gl.uint32, bitcast=True)
            )
            match = i < bucket
            if TIES:
                cols = cols_smem.slice(ch * _CAPACITY_CHUNK, _CAPACITY_CHUNK).load(LK)
                match = match & (keys == match_key)
                keys = ~cols.to(gl.uint32, bitcast=True)
            if SHIFT + BITS < 32:
                match = match & ((keys >> (SHIFT + BITS)) == prefix)
            digit = ((keys >> SHIFT) & ((1 << BITS) - 1)).to(gl.int32)
            hist_smem.atomic_scatter_add(ones, digit, axis=0, mask=match)
    gl.barrier()
    pick, higher, _ = _select_pick(
        hist_smem.load(LH), gl.arange(0, _LATE_BINS, layout=LH), remaining
    )
    if SHIFT + BITS < 32:
        prefix = (prefix << BITS) | pick.to(gl.uint32)
    else:
        prefix = pick.to(gl.uint32)
    return prefix, remaining - higher


@gluon.jit(
    launch_metadata=_radix_launch_metadata,
    do_not_specialize=(
        "logits_stride",
        "cand_stride",
        "out_stride",
        "width",
        "out_width",
        "take",
    ),
)
def gluon_dsv41_index_topk_radix_gfx950(
    logits,  # [Q, width] fp32, -inf = unscored
    candidates,  # [Q, blocks] int32/int64 block ids, or logits when unused
    limits,  # [Q] live widths (rows) or visible rows (blocks), or logits
    out,  # [Q, out_width] int32
    out_lens,  # [Q] int32
    logits_stride,
    cand_stride,
    out_stride,
    width,
    out_width,
    take,  # at most this many picks per row
    TOPK: gl.constexpr,  # power-of-two capacity >= take, at least 512
    BLOCK_MAX: gl.constexpr,  # select 8-column blocks by their maximum
    HAS_CANDIDATES: gl.constexpr,
    HAS_LIMITS: gl.constexpr,
    ALIGNED: gl.constexpr,  # 16-byte aligned rows whose width is a multiple of 4
    REPEAT: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Top-``take`` finite elements of one logits row as position-sorted ids.

    Elements are logits columns (mapped through ``candidates`` when given), or
    with ``BLOCK_MAX`` the 8-column blocks of the visible rows ranked by their
    maximum, the newest visible block first. Positions >= the pick count are
    written as -1 up to ``out_width``.
    """
    BINS: gl.constexpr = 1 << _HIGH_BITS
    if BLOCK_MAX:
        ELEMS: gl.constexpr = 64 * NUM_WARPS * REPEAT
    else:
        ELEMS: gl.constexpr = 4 * 64 * NUM_WARPS * REPEAT
    LBIN: gl.constexpr = gl.BlockedLayout(
        [BINS // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    LK: gl.constexpr = gl.BlockedLayout(
        [_CAPACITY_CHUNK // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    LP: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    LC64: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    SL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    q = gl.program_id(0).to(gl.int64)
    if ALIGNED:
        # Spelled as a multiple of 4 so the compiler sees 16-byte alignment.
        row = logits + q * (logits_stride // 4) * 4
    else:
        row = logits + q * logits_stride
    latest = -1
    if BLOCK_MAX:
        # Only blocks holding visible rows are read; columns past the visible
        # length are masked, so they need not hold -inf.
        seen = gl.minimum(gl.maximum(gl.load(limits + q).to(gl.int32), 0), width)
        latest = gl.where(seen > 0, (seen - 1) // 8, -1)
        n = gl.minimum(width // 8, (seen + 7) // 8)
        exact = seen
    else:
        n = width
        if HAS_LIMITS:
            # Columns at or past the row's live width are -inf: skip them.
            n = gl.minimum(width, gl.maximum(gl.load(limits + q).to(gl.int32), 0))
        exact = n
    limit = exact
    if ALIGNED:
        limit = (exact + 3) // 4 * 4

    # LDS: the pass-1 histogram, later the compacted bucket (keys, then
    # positions) and finally the ordering bitmap; plus the late-digit
    # histogram, the pick list and two counters.
    region = gl.allocate_shared_memory(gl.int32, [BINS], SL)
    late_smem = gl.allocate_shared_memory(gl.int32, [_LATE_BINS], SL)
    picked_smem = gl.allocate_shared_memory(gl.int32, [TOPK], SL)
    count_smem = gl.allocate_shared_memory(gl.int32, [64], SL)
    keys_smem = region.slice(0, _CAPACITY)
    cols_smem = region.slice(_CAPACITY, _CAPACITY)

    # Pass 1: histogram the high digit of every finite key.
    region.store(gl.zeros([BINS], gl.int32, layout=LBIN))
    count_smem.store(gl.zeros([64], gl.int32, layout=LC64))
    gl.barrier()
    for c0 in range(0, n, ELEMS):
        _, key, valid = _radix_tile(
            row, c0, n, limit, exact, latest, ELEMS, BLOCK_MAX, ALIGNED, NUM_WARPS
        )
        digit = (key >> _HIGH_SHIFT).to(gl.int32)
        region.atomic_scatter_add(
            gl.full(digit.shape, 1, gl.int32, digit.type.layout),
            digit,
            axis=0,
            mask=valid,
        )
    gl.barrier()
    hist = region.load(LBIN)
    total = gl.sum(hist, 0)
    count = gl.minimum(take, total)
    pick, above, kept = _select_pick(hist, gl.arange(0, BINS, layout=LBIN), count)
    # With at most ``take`` finite keys every one is picked: no bucket digit
    # matches and every key counts as above it.
    take_all = total <= take
    high = gl.where(take_all, -1, pick.to(gl.int32))
    bucket = gl.where(take_all, 0, kept - above)
    remaining = gl.where(take_all, 0, count - above)
    fits = bucket <= _CAPACITY
    gl.barrier()

    # Pass 2 (back to front): keys above the bucket join the pick list; keys
    # in it are compacted with their positions while they fit.
    chunks = (n + ELEMS - 1) // ELEMS
    for t in range(0, chunks):
        elem, key, valid = _radix_tile(
            row,
            (chunks - 1 - t) * ELEMS,
            n,
            limit,
            exact,
            latest,
            ELEMS,
            BLOCK_MAX,
            ALIGNED,
            NUM_WARPS,
        )
        digit = (key >> _HIGH_SHIFT).to(gl.int32)
        is_above = valid & (digit > high)
        in_bucket = valid & (digit == high) & fits
        ones = gl.full(digit.shape, 1, gl.int32, digit.type.layout)
        zero = gl.zeros(digit.shape, gl.int32, digit.type.layout)
        slot = count_smem.atomic_scatter_add(ones, zero, axis=0, mask=is_above)
        picked_smem.atomic_scatter_xchg(elem, slot, axis=0, mask=is_above)
        slot = count_smem.atomic_scatter_add(ones, zero + 1, axis=0, mask=in_bucket)
        keys_smem.atomic_scatter_xchg(
            key.to(gl.int32, bitcast=True), slot, axis=0, mask=in_bucket
        )
        cols_smem.atomic_scatter_xchg(elem, slot, axis=0, mask=in_bucket)
    gl.barrier()

    if bucket > 0:
        if fits:
            # The low digits over the compacted bucket.
            prefix, remaining = _radix_bucket_level(
                keys_smem,
                cols_smem,
                bucket,
                0,
                high.to(gl.uint32),
                remaining,
                late_smem,
                _LOW_BITS,
                _MID_BITS,
                False,
                NUM_WARPS,
            )
            threshold, remaining = _radix_bucket_level(
                keys_smem,
                cols_smem,
                bucket,
                0,
                prefix,
                remaining,
                late_smem,
                0,
                _LOW_BITS,
                False,
                NUM_WARPS,
            )
            # ``remaining`` keys equal to ``threshold`` are still to pick. When
            # more tie, keep the lowest positions: the ``remaining`` largest
            # ``~position`` among the ties, resolved 8 bits at a time.
            ties = 0
            for ch in gl.static_range(_CAPACITY // _CAPACITY_CHUNK):
                i = ch * _CAPACITY_CHUNK + gl.arange(0, _CAPACITY_CHUNK, layout=LK)
                keys = (
                    keys_smem.slice(ch * _CAPACITY_CHUNK, _CAPACITY_CHUNK)
                    .load(LK)
                    .to(gl.uint32, bitcast=True)
                )
                ties += gl.sum(((i < bucket) & (keys == threshold)).to(gl.int32), 0)
            tie_floor = gl.to_tensor(0).to(gl.uint32)
            if ties > remaining:
                tie_rank = remaining
                for level in gl.static_range(4):
                    tie_floor, tie_rank = _radix_bucket_level(
                        keys_smem,
                        cols_smem,
                        bucket,
                        threshold,
                        tie_floor,
                        tie_rank,
                        late_smem,
                        24 - 8 * level,
                        8,
                        True,
                        NUM_WARPS,
                    )
            ones_k = gl.full([_CAPACITY_CHUNK], 1, gl.int32, layout=LK)
            zero_k = gl.zeros([_CAPACITY_CHUNK], gl.int32, layout=LK)
            for ch in gl.static_range(_CAPACITY // _CAPACITY_CHUNK):
                if ch * _CAPACITY_CHUNK < bucket:
                    i = ch * _CAPACITY_CHUNK + gl.arange(0, _CAPACITY_CHUNK, layout=LK)
                    keys = (
                        keys_smem.slice(ch * _CAPACITY_CHUNK, _CAPACITY_CHUNK)
                        .load(LK)
                        .to(gl.uint32, bitcast=True)
                    )
                    cols = cols_smem.slice(ch * _CAPACITY_CHUNK, _CAPACITY_CHUNK).load(
                        LK
                    )
                    take_it = (i < bucket) & (
                        (keys > threshold)
                        | (
                            (keys == threshold)
                            & (~cols.to(gl.uint32, bitcast=True) >= tie_floor)
                        )
                    )
                    slot = count_smem.atomic_scatter_add(
                        ones_k, zero_k, axis=0, mask=take_it
                    )
                    picked_smem.atomic_scatter_xchg(cols, slot, axis=0, mask=take_it)
        else:
            # Crowded bucket: resolve the low digits and compact the picks in
            # position order with further row scans.
            prefix, remaining = _radix_row_level(
                row,
                n,
                limit,
                exact,
                latest,
                high.to(gl.uint32),
                remaining,
                late_smem,
                _LOW_BITS,
                _MID_BITS,
                ELEMS,
                BLOCK_MAX,
                ALIGNED,
                NUM_WARPS,
            )
            threshold, remaining = _radix_row_level(
                row,
                n,
                limit,
                exact,
                latest,
                prefix,
                remaining,
                late_smem,
                0,
                _LOW_BITS,
                ELEMS,
                BLOCK_MAX,
                ALIGNED,
                NUM_WARPS,
            )
            n_above = 0
            n_ties = 0
            for c0 in range(0, n, ELEMS):
                elem, key, valid = _radix_tile(
                    row,
                    c0,
                    n,
                    limit,
                    exact,
                    latest,
                    ELEMS,
                    BLOCK_MAX,
                    ALIGNED,
                    NUM_WARPS,
                )
                is_above = valid & (key > threshold)
                is_tie = valid & (key == threshold)
                # One packed scan counts both: low 16 bits the keys above the
                # threshold, high 16 bits the ties (a tile holds < 2**16).
                packed = is_above.to(gl.int32) | (is_tie.to(gl.int32) << 16)
                before = gl.cumsum(packed, 0) - packed
                ties_before = n_ties + (before >> 16)
                take_it = is_above | (is_tie & (ties_before < remaining))
                slot = n_above + (before & 0xFFFF) + gl.minimum(ties_before, remaining)
                picked_smem.atomic_scatter_xchg(elem, slot, axis=0, mask=take_it)
                packed_total = gl.sum(packed, 0)
                n_above += packed_total & 0xFFFF
                n_ties += packed_total >> 16
    gl.barrier()

    # Order the picks by position: a bitmap with one bit per position gives
    # each pick the number of picked positions before it.
    i = gl.arange(0, TOPK, layout=LP)
    live = i < count
    picks = gl.where(live, picked_smem.load(LP), 0)
    slot = i
    if n <= 32 * _BITMAP_WORDS:
        LW: gl.constexpr = gl.BlockedLayout(
            [_BITMAP_WORDS // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
        )
        bits_smem = region.slice(0, _BITMAP_WORDS)
        before_smem = region.slice(_BITMAP_WORDS, _BITMAP_WORDS)
        bits_smem.store(gl.zeros([_BITMAP_WORDS], gl.int32, layout=LW))
        gl.barrier()
        word = picks >> 5
        bit = gl.full([TOPK], 1, gl.int32, layout=LP) << (picks & 31)
        bits_smem.atomic_scatter_or(bit, word, axis=0, mask=live)
        gl.barrier()
        set_bits = _select_popcount(bits_smem.load(LW))
        before_smem.store(gl.cumsum(set_bits, 0) - set_bits)
        gl.barrier()
        slot = before_smem.gather(word, axis=0) + _select_popcount(
            bits_smem.gather(word, axis=0) & (bit - 1)
        )
    if HAS_CANDIDATES:
        block = gl.load(
            candidates + q * cand_stride + picks // 8, mask=live, other=-1
        ).to(gl.int32)
        logical = block * 8 + picks % 8
    else:
        logical = picks
    # Unpicked slots hold the int32 max and rank after every id.
    ordered_smem = region.slice(2 * _BITMAP_WORDS, TOPK)
    ordered_smem.store(gl.full([TOPK], 0x7FFFFFFF, gl.int32, layout=LP))
    gl.barrier()
    ordered_smem.atomic_scatter_xchg(logical, slot, axis=0, mask=live)
    gl.barrier()

    # Position order is id order unless shuffled candidates or a row too wide
    # for the bitmap leave descents: rank-sort those (ids are unique).
    LT: gl.constexpr = gl.BlockedLayout([1, 16], [64, 1], [NUM_WARPS, 1], [1, 0])
    LR: gl.constexpr = gl.SliceLayout(1, LT)
    LC: gl.constexpr = gl.SliceLayout(0, LT)
    mine = ordered_smem.load(LR)
    j = gl.arange(0, TOPK, layout=LR)
    previous = gl.gather(mine, gl.maximum(j - 1, 0), axis=0)
    descents = gl.sum(((j > 0) & (j < count) & (mine < previous)).to(gl.int32), 0)
    rank = j
    if descents > 0:
        rank = gl.zeros([TOPK], gl.int32, layout=LR)
        for j0 in gl.static_range(0, TOPK, 16):
            other = ordered_smem.slice(j0, 16).load(LC)
            rank += gl.sum(
                (gl.expand_dims(other, 0) < gl.expand_dims(mine, 1)).to(gl.int32),
                axis=1,
            )
    dst = out + q * out_stride
    gl.store(dst + rank, mine, mask=j < count)
    for p0 in range(0, out_width, TOPK):
        pad = p0 + j
        gl.store(
            dst + pad,
            gl.full([TOPK], -1, gl.int32, layout=LR),
            mask=(pad >= count) & (pad < out_width),
        )
    gl.store(out_lens + q, count)


def _aligned(logits: torch.Tensor, width: int, multiple: int) -> bool:
    return (
        logits.stride(0) % 4 == 0
        and width % multiple == 0
        and logits.data_ptr() % 16 == 0
    )


def launch_gluon_dsv41_index_topk_radix_gfx950(
    logits, candidates, live_widths, topk, row_out, row_lens
):
    """Select CSA2 rows from scored logits, position-sorted.

    Same contract as ``launch_gluon_dsv41_index_topk_select_gfx950``.

    Args:
        logits: FP32 [Q, width] scores; -inf marks unscored rows.
        candidates: None, or int32/int64 [Q, blocks] block ids whose 8-row
            blocks map logits column c to logical row
            ``candidates[q, c // 8] * 8 + c % 8``.
        live_widths: None, or int32/int64 [Q] counts of leading columns that
            may be finite; row q's columns at or past ``live_widths[q]`` must be
            -inf and are not read.
        topk: Selection capacity in [1, 2048]; V4.1 uses 512.
        row_out: Int32 [Q, >= topk] destination. Row q receives the logical ids
            of its ``min(topk, width)`` largest finite logits in ascending
            order, then -1 padding through column ``topk``. Equal-score
            boundary ties keep the lowest columns.
        row_lens: Int32 [Q] destination for the number of ids written.

    Returns:
        None.
    """
    queries, width = logits.shape
    if not queries:
        return
    topk = int(topk)
    if not 1 <= topk <= 2048:
        raise ValueError(f"radix select topk must be in [1, 2048], got {topk}")
    if (
        logits.stride(1) != 1
        or row_out.stride(1) != 1
        or row_lens.stride(0) != 1
        or (candidates is not None and candidates.stride(1) != 1)
    ):
        raise ValueError("radix select requires unit inner strides")
    if row_out.shape[0] != queries or row_out.shape[1] < topk:
        raise ValueError(f"row_out must be [{queries}, >= {topk}]")
    if row_lens.shape != (queries,):
        raise ValueError(f"row_lens must be [{queries}]")
    if candidates is not None and (
        candidates.shape[0] != queries or candidates.shape[1] * 8 < width
    ):
        raise ValueError(f"candidates must be [{queries}, >= {-(-width // 8)}]")
    if live_widths is not None and (
        live_widths.shape != (queries,)
        or live_widths.stride(0) != 1
        or live_widths.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(f"live_widths must be contiguous int32/int64 [{queries}]")
    gluon_dsv41_index_topk_radix_gfx950[(queries,)](
        logits,
        logits if candidates is None else candidates,
        logits if live_widths is None else live_widths,
        row_out,
        row_lens,
        logits.stride(0),
        0 if candidates is None else candidates.stride(0),
        row_out.stride(0),
        width,
        topk,
        min(topk, width),
        TOPK=max(512, triton.next_power_of_2(topk)),
        BLOCK_MAX=False,
        HAS_CANDIDATES=candidates is not None,
        HAS_LIMITS=live_widths is not None,
        ALIGNED=_aligned(logits, width, 4),
        REPEAT=_REPEAT,
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
        # Two 16-wave CTAs per CU (8 waves per SIMD) need <= 64 VGPRs; the
        # row scans are latency bound, so occupancy beats registers.
        waves_per_eu=8,
    )


def launch_gluon_dsv41_index_block_radix_gfx950(
    logits, visible, candidate_topk, block_out, block_lens
):
    """Select each query's best 8-row blocks from scored logits, position-sorted.

    Same contract as ``index_block_select``.

    Args:
        logits: FP32 [Q, width] full-selection scores with unit inner stride;
            columns at or past ``visible[q]`` are not read.
        visible: Int32/int64 [Q] visible row counts.
        candidate_topk: Number of blocks to keep, at most 2048.
        block_out: Int32 [Q, cap] destination with unit inner stride. Row q
            receives the ids of its ``min(candidate_topk, width // 8)`` best
            blocks by maximum logit in ascending order, then -1. The block
            holding row ``visible[q] - 1`` is forced in when it has a finite
            score; equal-score boundary ties keep the lowest blocks.
        block_lens: Int32 [Q] destination for the number of ids written.

    Returns:
        None.
    """
    queries, width = logits.shape
    if not queries:
        return
    k = min(int(candidate_topk), width // 8)
    if k < 1:
        block_out.fill_(-1)
        block_lens.zero_()
        return
    if int(candidate_topk) > 2048:
        raise ValueError(f"radix block select keeps at most 2048, got {candidate_topk}")
    if logits.stride(1) != 1 or block_out.stride(1) != 1:
        raise ValueError("block select requires unit inner strides")
    if (
        visible.shape != (queries,)
        or visible.stride(0) != 1
        or visible.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(f"visible must be contiguous int32/int64 [{queries}]")
    if block_out.shape[0] != queries or block_lens.shape != (queries,):
        raise ValueError(f"block_out and block_lens must have {queries} rows")
    gluon_dsv41_index_topk_radix_gfx950[(queries,)](
        logits,
        logits,
        visible,
        block_out,
        block_lens,
        logits.stride(0),
        0,
        block_out.stride(0),
        width,
        int(block_out.shape[1]),
        k,
        TOPK=max(512, triton.next_power_of_2(int(candidate_topk))),
        BLOCK_MAX=True,
        HAS_CANDIDATES=False,
        HAS_LIMITS=True,
        ALIGNED=_aligned(logits, width, 8),
        REPEAT=_REPEAT,
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
        # Two 16-wave CTAs per CU (8 waves per SIMD) need <= 64 VGPRs; the
        # row scans are latency bound, so occupancy beats registers.
        waves_per_eu=8,
    )
