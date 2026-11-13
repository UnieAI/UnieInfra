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

"""GFX950 DeepSeek V4.1 CSA2 indexer.

Scores page-planar MXFP4 index-K with the V4 ``mfma_scaled`` e2m1 tile.
Full mode uses a row-block scorer that shares key loads across query rows with
matching page tables (speculative decode, prefill); candidate (Reindex) mode
gathers each row's candidate blocks into the same 32x32x64 MFMA tiles. The tokenspeed-kernel adapter owns query packing, bounded query
tiling, and selection.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._device import num_cus
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4.indexer import (
    _HEADS_PER_MFMA,
    _PACKED_DIM,
    _PAGE_SIZE,
    _SCALE_DIM,
    _load_query_group,
)

__all__ = [
    "dsv41_index_logits_gfx950",
    "launch_gluon_dsv41_index_topk_select_gfx950",
]

_MFMA_HEADS = 32
# One wave scores a 16-row key tile per step: on MI350P (128 CUs) this beats
# 32-row tiles over two waves at every decode and prefill shape measured
# (96 rows x 75K history: 617 -> 434 us at a 256-row score chunk).
_BLOCK_N = gl.constexpr(16)
_NUM_WARPS = 1
# Score-chunk buckets (rows per CTA). Longer chunks amortize the per-CTA query
# load over more key tiles; shorter ones keep enough CTAs when the work is
# small. Picked by scored rows (queries x width) so only three variants compile.
_SCORE_CHUNKS = ((1 << 19, 256), (1 << 21, 512))
_MAX_SCORE_CHUNK = 1024


def _score_chunk(queries: int, width: int, bound: int) -> int:
    """Rows per CTA for a [queries, width] logits launch, at most ``bound``."""
    work = queries * width
    chunk = next(
        (size for limit, size in _SCORE_CHUNKS if work < limit), _MAX_SCORE_CHUNK
    )
    return min(chunk, bound)


@gluon.constexpr_function
def _indexer_mfma_layouts(NUM_WARPS):
    mfma = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_a = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    dot_b = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    a_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_a, [_HEADS_PER_MFMA, _SCALE_DIM])
    b_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_b, [_BLOCK_N, _SCALE_DIM])
    return mfma, dot_a, dot_b, a_scale, b_scale


@gluon.jit
def _csa2_page_rows(
    positions,
    valid,
    page_table,
    candidates,
    query,
    table_stride,
    cand_stride,
    num_pages,
    visible,
    PAGE_SIZE: gl.constexpr,
    table_width,
    CANDIDATES: gl.constexpr,
):
    if CANDIDATES >= 0:
        block = gl.amd.cdna4.buffer_load(
            ptr=candidates,
            offsets=(query * cand_stride + positions // 8).to(gl.int32),
            mask=valid,
            other=-1,
        ).to(gl.int64)
        logical = gl.where(block >= 0, block * 8 + positions % 8, -1)
        valid = valid & (logical >= 0) & (logical < visible)
    else:
        logical = positions.to(gl.int64)
        valid = valid & (logical < visible)
    logical_page = logical // PAGE_SIZE
    safe_page = gl.minimum(gl.maximum(logical_page, 0), table_width - 1)
    physical = gl.amd.cdna4.buffer_load(
        ptr=page_table,
        offsets=(query * table_stride + safe_page).to(gl.int32),
        mask=valid,
        other=-1,
    ).to(gl.int64)
    valid = valid & (physical >= 0) & (physical < num_pages)
    return gl.where(valid, physical, 0), logical % PAGE_SIZE, valid


@gluon.jit
def _load_csa2_key_tile(
    index_k_cache,
    page_table,
    candidates,
    token,
    tile_start,
    candidate_end,
    table_stride,
    cand_stride,
    page_stride_bytes,
    num_pages,
    visible,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    table_width,
    CANDIDATES: gl.constexpr,
):
    """Load one BLOCK_N-row MXFP4 key tile and its E8M0 scales."""
    packed_dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(1, dot_b_layout))[
        :, None
    ]
    columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    positions = tile_start + columns
    valid = positions < candidate_end
    pages, page_rows, valid = _csa2_page_rows(
        positions,
        valid,
        page_table,
        candidates,
        token,
        table_stride,
        cand_stride,
        num_pages,
        visible,
        PAGE_SIZE,
        table_width,
        CANDIDATES,
    )
    key_offsets = (
        pages * page_stride_bytes + page_rows.to(gl.int64) * _PACKED_DIM + packed_dims
    )
    key = gl.load(index_k_cache + key_offsets, mask=valid, other=0)

    scale_columns = gl.arange(0, _BLOCK_N, layout=gl.SliceLayout(1, b_scale_layout))[
        :, None
    ]
    scale_groups = gl.arange(0, _SCALE_DIM, layout=gl.SliceLayout(0, b_scale_layout))[
        None, :
    ]
    scale_positions = tile_start + scale_columns
    scale_valid = scale_positions < candidate_end
    scale_pages, scale_page_rows, scale_valid = _csa2_page_rows(
        scale_positions,
        scale_valid,
        page_table,
        candidates,
        token,
        table_stride,
        cand_stride,
        num_pages,
        visible,
        PAGE_SIZE,
        table_width,
        CANDIDATES,
    )
    key_scale_offsets = (
        scale_pages * page_stride_bytes
        + PAGE_SIZE * _PACKED_DIM
        + scale_page_rows.to(gl.int64) * _SCALE_DIM
        + scale_groups
    )
    key_scales = gl.load(
        index_k_cache + key_scale_offsets,
        mask=scale_valid,
        other=127,
    )
    return key, key_scales


@gluon.jit
def _score_csa2_group(
    query,
    query_scales,
    head_weights,
    key,
    key_scales,
    mfma_layout: gl.constexpr,
):
    """Score one 16-head query group against a loaded key tile."""
    accumulator = gl.zeros(
        [_HEADS_PER_MFMA, _BLOCK_N], dtype=gl.float32, layout=mfma_layout
    )
    head_scores = gl.amd.cdna4.mfma_scaled(
        a=query,
        a_scale=query_scales,
        a_format="e2m1",
        b=key,
        b_scale=key_scales,
        b_format="e2m1",
        acc=accumulator,
    )
    head_scores = gl.maximum(
        head_scores,
        0.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    return gl.sum(head_scores * head_weights[:, None], axis=0)


def _index_launch_metadata(grid, kernel, args):
    """Describe the score capacity without reading device-resident lengths."""
    queries, width = args["logits"].shape
    heads = args["NUM_HEADS"]
    return {
        "name": kernel.name,
        "flops4": 2 * queries * heads * width * 128,
        "bytes": queries * width * 68
        + args["q"].numel() * args["q"].element_size() * grid[1]
        + args["logits"].numel() * args["logits"].element_size(),
    }


@gluon.jit(
    launch_metadata=_index_launch_metadata,
    # page_stride_bytes stays specializable: its divisible-by-16 attribute lets
    # the key-tile loads vectorize (runtime arenas align index pages to 256 B,
    # and the stride is fixed per server, so it never recompiles while serving).
    # Without it every key byte is a separate load (about 3x slower).
    do_not_specialize=(
        "stride_q_token",
        "stride_q_head",
        "stride_q_scale_token",
        "stride_q_scale_head",
        "stride_w_token",
        "stride_w_head",
        "table_stride",
        "table_width",
        "cand_stride",
        "logits_stride",
        "num_pages",
    ),
)
def gluon_dsv41_index_topk_gfx950(
    q,
    q_scales,
    weights,
    index_k_cache,
    visible,
    page_table,
    candidates,
    logits,
    stride_q_token,
    stride_q_head,
    stride_q_scale_token,
    stride_q_scale_head,
    stride_w_token,
    stride_w_head,
    table_stride,
    cand_stride,
    logits_stride,
    page_stride_bytes,
    num_pages,
    max_candidates,
    NUM_HEADS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    table_width,
    CANDIDATES: gl.constexpr,
    SCORE_CHUNK: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    split = gl.program_id(1)
    vis = gl.minimum(
        gl.maximum(gl.load(visible + token).to(gl.int32), 0),
        table_width * PAGE_SIZE,
    )
    if CANDIDATES >= 0:
        width = gl.where(vis > 0, CANDIDATES * 8, 0)
    else:
        width = vis
    candidate_start = split * SCORE_CHUNK
    candidate_end = gl.minimum(width, candidate_start + SCORE_CHUNK)
    candidate_end = gl.minimum(candidate_end, max_candidates)
    if candidate_start >= candidate_end:
        return

    layouts: gl.constexpr = _indexer_mfma_layouts(NUM_WARPS)
    mfma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]
    query_0, query_scale_0, weight_0 = _load_query_group(
        q,
        q_scales,
        weights,
        token,
        0,
        stride_q_token,
        stride_q_head,
        stride_q_scale_token,
        stride_q_scale_head,
        stride_w_token,
        stride_w_head,
        mfma_layout,
        dot_a_layout,
        a_scale_layout,
    )
    query_1, query_scale_1, weight_1 = _load_query_group(
        q,
        q_scales,
        weights,
        token,
        16,
        stride_q_token,
        stride_q_head,
        stride_q_scale_token,
        stride_q_scale_head,
        stride_w_token,
        stride_w_head,
        mfma_layout,
        dot_a_layout,
        a_scale_layout,
    )
    output_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    output_columns = gl.arange(0, BLOCK_N, layout=output_layout)
    for tile_offset in range(0, CHUNK_N, BLOCK_N):
        tile_start = candidate_start + tile_offset
        # Both 16-head groups score the same key rows: load the tile (page
        # lookups, packed keys, scales) once and feed it to both MFMAs.
        key, key_scales = _load_csa2_key_tile(
            index_k_cache,
            page_table,
            candidates,
            token,
            tile_start,
            candidate_end,
            table_stride,
            cand_stride,
            page_stride_bytes,
            num_pages,
            vis,
            dot_b_layout,
            b_scale_layout,
            PAGE_SIZE,
            table_width,
            CANDIDATES,
        )
        scores = _score_csa2_group(
            query_0, query_scale_0, weight_0, key, key_scales, mfma_layout
        )
        scores += _score_csa2_group(
            query_1, query_scale_1, weight_1, key, key_scales, mfma_layout
        )
        positions = tile_start + output_columns
        live = positions < candidate_end
        _, _, live = _csa2_page_rows(
            positions,
            live,
            page_table,
            candidates,
            token,
            table_stride,
            cand_stride,
            num_pages,
            vis,
            PAGE_SIZE,
            table_width,
            CANDIDATES,
        )
        gl.store(
            logits + token * logits_stride + positions,
            scores,
            mask=(positions < max_candidates) & live,
        )


# Row-block scorer: one CTA scores BLOCK_M consecutive query rows over one KV
# split. A speculative-decode request's next_n rows (and a prefill chunk's
# rows) share one page table, so each 64-key page is loaded into registers once
# and reused by every row of the block. Uses the 32x32x64 scaled MFMA with
# heads on M (not transposed): a lane holds 16 of the 32 heads of one key in
# registers, so the head sum is in-register FMAs plus one cross-lane step.
_ROWS_HEADS = gl.constexpr(32)
_ROWS_PAGE_SIZE = gl.constexpr(_PAGE_SIZE)
_ROWS_BLOCK_KV = 64
_ROWS_BLOCK_M = 6
_ROWS_WAVES_PER_EU = 2
# Splits fill the machine (4 SIMDs per CU x resident waves) but stay at least
# this many tiles long so the per-CTA query load is amortized, and at most the
# cap so a long history still spreads over many CTAs.
_ROWS_MIN_TILES = 4
_ROWS_MAX_TILES = 64


@gluon.constexpr_function
def _rows_mfma_layouts(BLOCK_KV):
    mfma = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=False,
        warps_per_cta=[1, 1],
    )
    dot_a = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    dot_b = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    a_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_a, [_MFMA_HEADS, _SCALE_DIM])
    b_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_b, [BLOCK_KV, _SCALE_DIM])
    return mfma, dot_a, dot_b, a_scale, b_scale


@gluon.jit
def _rows_scale_bytes(words, scale_layout: gl.constexpr):
    """Expand per-row int32 words into the [rows, 4] E8M0 byte operand."""
    shifts = gl.arange(0, _SCALE_DIM, layout=gl.SliceLayout(0, scale_layout)) * 8
    return ((words[:, None] >> shifts[None, :]) & 0xFF).to(gl.uint8)


@gluon.jit
def _rows_load_query(
    q,
    q_scales,
    weights,
    row,
    mfma_layout: gl.constexpr,
    dot_a_layout: gl.constexpr,
    a_scale_layout: gl.constexpr,
):
    heads = gl.arange(0, _ROWS_HEADS, layout=gl.SliceLayout(1, dot_a_layout))[:, None]
    dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(0, dot_a_layout))[None, :]
    values = gl.amd.cdna4.buffer_load(
        ptr=q + row.to(gl.int64) * (_ROWS_HEADS * _PACKED_DIM),
        offsets=heads * _PACKED_DIM + dims,
    )
    # A lane needs two of a head's four E8M0 bytes: load the head's int32 word
    # once and pick the bytes in registers instead of four byte loads.
    scale_heads = gl.arange(0, _ROWS_HEADS, layout=gl.SliceLayout(1, a_scale_layout))
    words = gl.amd.cdna4.buffer_load(
        ptr=q_scales + row.to(gl.int64) * _ROWS_HEADS, offsets=scale_heads
    )
    scales = _rows_scale_bytes(words, a_scale_layout)
    weight_heads = gl.arange(0, _ROWS_HEADS, layout=gl.SliceLayout(1, mfma_layout))
    head_weights = gl.amd.cdna4.buffer_load(
        ptr=weights + row.to(gl.int64) * _ROWS_HEADS,
        offsets=weight_heads,
    ).to(gl.float32)
    return values, scales, head_weights


@gluon.jit
def _rows_load_page_tile(
    index_k_cache,
    page,
    page_row,
    page_stride_bytes,
    enabled,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
    BLOCK_KV: gl.constexpr,
):
    """Load BLOCK_KV keys starting at ``page_row`` of physical ``page``.

    ``enabled`` False masks every lane: no memory traffic, zero operands.
    """
    base = index_k_cache + page.to(gl.int64) * page_stride_bytes
    dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(1, dot_b_layout))[:, None]
    keys = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, dot_b_layout))[None, :]
    key = gl.amd.cdna4.buffer_load(
        ptr=base,
        offsets=(page_row + keys) * _PACKED_DIM + dims,
        mask=(keys >= 0) & enabled,
        other=0,
    )
    scale_keys = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(1, b_scale_layout))
    words = gl.amd.cdna4.buffer_load(
        ptr=base.to(gl.pointer_type(gl.int32)),
        offsets=_ROWS_PAGE_SIZE * _PACKED_DIM // 4 + page_row + scale_keys,
        mask=(scale_keys >= 0) & enabled,
        other=0x7F7F7F7F,
    )
    key_scales = _rows_scale_bytes(words, b_scale_layout)
    return key, key_scales


@gluon.constexpr_function
def _rows_fold_plan(linear_layout, num_heads, block_kv):
    """Reshape plan that moves the in-register head bits to the last axes.

    Head bits a lane holds in registers fold with FMAs; the rest live across
    lanes and are summed once at the end (after aiter's pa_mqa_logits fold).
    Returns (split shape, permutation, folded shape, folded bit count).
    """
    head_bits = num_heads.bit_length() - 1
    reg_bases = [tuple(b) for b in linear_layout.reg_bases]
    summed, folded = [], []
    for bit in range(head_bits):
        stride = 1 << (head_bits - 1 - bit)
        (folded if (stride, 0) in reg_bases else summed).append(bit)
    return (
        tuple([2] * head_bits + [block_kv]),
        tuple(summed + [head_bits] + folded),
        tuple([1 << len(summed), block_kv] + [2] * len(folded)),
        len(folded),
    )


@gluon.jit
def _rows_leaf(x, INDEX: gl.constexpr, DEPTH: gl.constexpr):
    for bit in gl.static_range(0, DEPTH):
        low, high = x.split()
        x = low if (INDEX // (2**bit)) % 2 == 0 else high
    return x


@gluon.jit
def _rows_fma(a, b, c):
    # One v_fma_f32: packed FP32 math does not co-issue with MFMA on gfx950,
    # and the packed add would also lose the free |x| source modifier.
    return gl.inline_asm_elementwise(
        "v_fma_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        [a, b, c],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _rows_score(
    query,
    query_scales,
    head_weights,
    key,
    key_scales,
    mfma_layout: gl.constexpr,
    BLOCK_KV: gl.constexpr,
):
    accumulator = gl.zeros(
        [_ROWS_HEADS, BLOCK_KV], dtype=gl.float32, layout=mfma_layout
    )
    scores = gl.amd.cdna4.mfma_scaled(
        a=query,
        a_scale=query_scales,
        a_format="e2m1",
        b=key,
        b_scale=key_scales,
        b_format="e2m1",
        acc=accumulator,
    )
    # One v_maximum3_f32 per score (no packed FP32 max exists to vectorize
    # into); NaN propagates as in the per-row scorer.
    scores = gl.maximum(scores, 0.0, propagate_nan=tl.PropagateNan.ALL)
    linear: gl.constexpr = gl.to_linear_layout(mfma_layout, [_ROWS_HEADS, BLOCK_KV])
    plan: gl.constexpr = _rows_fold_plan(linear, _ROWS_HEADS.value, BLOCK_KV)
    split_shape: gl.constexpr = plan[0]
    order: gl.constexpr = plan[1]
    folded_shape: gl.constexpr = plan[2]
    DEPTH: gl.constexpr = plan[3]
    weights = head_weights[:, None].broadcast_to([_ROWS_HEADS, BLOCK_KV])
    scores = scores.reshape(split_shape).permute(order).reshape(folded_shape)
    weights = weights.reshape(split_shape).permute(order).reshape(folded_shape)
    total = _rows_leaf(scores, 0, DEPTH) * _rows_leaf(weights, 0, DEPTH)
    for i in gl.static_range(1, 2**DEPTH):
        total = _rows_fma(
            _rows_leaf(scores, i, DEPTH), _rows_leaf(weights, i, DEPTH), total
        )
    return gl.convert_layout(gl.sum(total, axis=0), gl.SliceLayout(0, mfma_layout))


@gluon.jit
def _rows_lane_page(lane_pages, lanes, tile, page_lo, TILES_PER_PAGE: gl.constexpr):
    """The page id lane ``tile // TILES_PER_PAGE - page_lo`` holds."""
    pick = lanes == tile // TILES_PER_PAGE - page_lo
    return gl.max(gl.where(pick, lane_pages, -1), axis=0)


@gluon.jit
def _rows_safe_page(page, num_pages):
    return gl.where((page >= 0) & (page < num_pages), page, 0)


@gluon.jit
def _rows_store(logits, row, logits_stride, scores, positions, end, page, num_pages):
    # Every visible column is written (-inf off the resident pages), so the
    # caller need not pre-fill the logits.
    live = (page >= 0) & (page < num_pages)
    gl.amd.cdna4.buffer_store(
        gl.where(live, scores, -float("inf")),
        ptr=logits + row.to(gl.int64) * logits_stride,
        offsets=positions,
        mask=positions < end,
    )


@gluon.jit
def _rows_walk_shared(
    index_k_cache,
    logits,
    logits_stride,
    page_stride_bytes,
    num_pages,
    rows,
    ends,
    qs,
    qss,
    ws,
    pages,
    lanes,
    page_lo,
    tile_lo,
    tile_hi,
    columns,
    mfma_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    TILES_PER_PAGE: gl.constexpr,
):
    """Every row of the block reads the same pages: one key tile per step.

    Software pipeline: the next tile's keys load while this tile is scored;
    the tail re-reads the last tile instead of branching.
    """
    last = tile_hi - 1
    page = _rows_lane_page(pages, lanes, tile_lo, page_lo, TILES_PER_PAGE)
    key, key_scales = _rows_load_page_tile(
        index_k_cache,
        _rows_safe_page(page, num_pages),
        (tile_lo % TILES_PER_PAGE) * BLOCK_KV,
        page_stride_bytes,
        True,
        dot_b_layout,
        b_scale_layout,
        BLOCK_KV,
    )
    for tile in range(tile_lo, tile_hi):
        following = gl.minimum(tile + 1, last)
        next_page = _rows_lane_page(pages, lanes, following, page_lo, TILES_PER_PAGE)
        next_key, next_key_scales = _rows_load_page_tile(
            index_k_cache,
            _rows_safe_page(next_page, num_pages),
            (following % TILES_PER_PAGE) * BLOCK_KV,
            page_stride_bytes,
            True,
            dot_b_layout,
            b_scale_layout,
            BLOCK_KV,
        )
        positions = tile * BLOCK_KV + columns
        for r in gl.static_range(0, BLOCK_M):
            scores = _rows_score(
                qs[r], qss[r], ws[r], key, key_scales, mfma_layout, BLOCK_KV
            )
            _rows_store(
                logits,
                rows[r],
                logits_stride,
                scores,
                positions,
                ends[r],
                page,
                num_pages,
            )
        page = next_page
        key = next_key
        key_scales = next_key_scales


@gluon.jit
def _rows_walk_each(
    index_k_cache,
    logits,
    logits_stride,
    page_stride_bytes,
    num_pages,
    rows,
    ends,
    qs,
    qss,
    ws,
    row_pages,
    lanes,
    page_lo,
    tile_lo,
    tile_hi,
    columns,
    mfma_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    TILES_PER_PAGE: gl.constexpr,
):
    """Fallback for blocks spanning three or more page tables: per-row keys."""
    for tile in range(tile_lo, tile_hi):
        positions = tile * BLOCK_KV + columns
        page_row = (tile % TILES_PER_PAGE) * BLOCK_KV
        for r in gl.static_range(0, BLOCK_M):
            page = _rows_lane_page(row_pages[r], lanes, tile, page_lo, TILES_PER_PAGE)
            key, key_scales = _rows_load_page_tile(
                index_k_cache,
                _rows_safe_page(page, num_pages),
                page_row,
                page_stride_bytes,
                True,
                dot_b_layout,
                b_scale_layout,
                BLOCK_KV,
            )
            scores = _rows_score(
                qs[r], qss[r], ws[r], key, key_scales, mfma_layout, BLOCK_KV
            )
            _rows_store(
                logits,
                rows[r],
                logits_stride,
                scores,
                positions,
                ends[r],
                page,
                num_pages,
            )


def _rows_launch_metadata(grid, kernel, args):
    queries, width = args["logits"].shape
    block_m = args["BLOCK_M"]
    return {
        "name": kernel.name,
        "flops4": 2 * queries * _MFMA_HEADS * width * 128,
        "bytes": triton.cdiv(queries, block_m) * width * 68
        + args["q"].numel() * args["q"].element_size() * grid[1]
        + args["logits"].numel() * args["logits"].element_size(),
    }


@gluon.jit(
    launch_metadata=_rows_launch_metadata,
    # page_stride_bytes stays specializable (see gluon_dsv41_index_topk_gfx950).
    do_not_specialize=(
        "queries",
        "table_stride",
        "table_width",
        "logits_stride",
        "num_pages",
        "max_candidates",
        "tiles_per_split",
    ),
)
def gluon_dsv41_index_topk_rows_gfx950(
    q,
    q_scales,
    weights,
    index_k_cache,
    visible,
    page_table,
    logits,
    queries,
    table_stride,
    table_width,
    logits_stride,
    page_stride_bytes,
    num_pages,
    max_candidates,
    tiles_per_split,
    BLOCK_M: gl.constexpr,
    BLOCK_KV: gl.constexpr,
):
    gl.static_assert(_ROWS_PAGE_SIZE % BLOCK_KV == 0, "a key tile must not span pages")
    layouts: gl.constexpr = _rows_mfma_layouts(BLOCK_KV)
    mfma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]
    TILES_PER_PAGE: gl.constexpr = _ROWS_PAGE_SIZE // BLOCK_KV

    row0 = gl.program_id(0) * BLOCK_M
    split = gl.program_id(1)
    capacity = gl.minimum(table_width * _ROWS_PAGE_SIZE, max_candidates)

    # Rows past the launch repeat the last row's query and store nothing.
    rows = ()
    ends = ()
    block_end = 0
    for r in gl.static_range(0, BLOCK_M):
        row = gl.minimum(row0 + r, queries - 1)
        end = gl.minimum(gl.maximum(gl.load(visible + row).to(gl.int32), 0), capacity)
        end = gl.where(row0 + r < queries, end, 0)
        rows = rows + (row,)
        ends = ends + (end,)
        block_end = gl.maximum(block_end, end)

    tile_lo = split * tiles_per_split
    tile_hi = gl.minimum(
        tile_lo + tiles_per_split, (block_end + BLOCK_KV - 1) // BLOCK_KV
    )
    if tile_lo >= tile_hi:
        return

    qs = ()
    qss = ()
    ws = ()
    for r in gl.static_range(0, BLOCK_M):
        value, scale, weight = _rows_load_query(
            q,
            q_scales,
            weights,
            rows[r],
            mfma_layout,
            dot_a_layout,
            a_scale_layout,
        )
        qs = qs + (value,)
        qss = qss + (scale,)
        ws = ws + (weight,)

    columns = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, mfma_layout))
    # Page ids of the split, one logical page per lane, read once up front:
    # the walk extracts them with a cross-lane reduce instead of a dependent
    # global load per tile.
    lane_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    lanes = gl.arange(0, 64, layout=lane_layout)
    page_lo = tile_lo // TILES_PER_PAGE
    page_live = page_lo + lanes <= (tile_hi - 1) // TILES_PER_PAGE
    logical_pages = gl.minimum(page_lo + lanes, table_width - 1)
    row_pages = ()
    for r in gl.static_range(0, BLOCK_M):
        row_pages = row_pages + (
            gl.load(
                page_table + rows[r].to(gl.int64) * table_stride + logical_pages,
                mask=page_live,
                other=-1,
            ),
        )
    # Rows of one request share its page table: when the whole block does,
    # each key tile is loaded once for all its rows.
    shared = True
    for r in gl.static_range(1, BLOCK_M):
        shared = shared & (
            gl.min((row_pages[r] == row_pages[0]).to(gl.int32), axis=0) > 0
        )
    if shared:
        _rows_walk_shared(
            index_k_cache,
            logits,
            logits_stride,
            page_stride_bytes,
            num_pages,
            rows,
            ends,
            qs,
            qss,
            ws,
            row_pages[0],
            lanes,
            page_lo,
            tile_lo,
            tile_hi,
            columns,
            mfma_layout,
            dot_b_layout,
            b_scale_layout,
            BLOCK_M,
            BLOCK_KV,
            TILES_PER_PAGE,
        )
    else:
        _rows_walk_each(
            index_k_cache,
            logits,
            logits_stride,
            page_stride_bytes,
            num_pages,
            rows,
            ends,
            qs,
            qss,
            ws,
            row_pages,
            lanes,
            page_lo,
            tile_lo,
            tile_hi,
            columns,
            mfma_layout,
            dot_b_layout,
            b_scale_layout,
            BLOCK_M,
            BLOCK_KV,
            TILES_PER_PAGE,
        )


def _rows_splits(
    groups: int, width: int, max_tiles: int, waves_per_eu: int
) -> tuple[int, int]:
    """(splits, tiles per split) for a launch of ``groups`` CTAs per split."""
    tiles = triton.cdiv(width, _ROWS_BLOCK_KV)
    target = 4 * num_cus() * waves_per_eu
    splits = max(triton.cdiv(target, groups), triton.cdiv(tiles, max_tiles))
    per = max(triton.cdiv(tiles, splits), _ROWS_MIN_TILES)
    per = min(per, max_tiles)
    return triton.cdiv(tiles, per), per


def _launch_rows(values, scales, w, cache_2d, table, visible, logits, max_tiles):
    """Score Full-mode logits with the row-block kernel, ``max_tiles`` per CTA."""
    queries, width = logits.shape
    groups = triton.cdiv(queries, _ROWS_BLOCK_M)
    splits, per = _rows_splits(groups, width, max_tiles, _ROWS_WAVES_PER_EU)
    gluon_dsv41_index_topk_rows_gfx950[(groups, splits)](
        values.contiguous(),
        scales.contiguous(),
        w.float().contiguous(),
        cache_2d,
        visible,
        table,
        logits,
        queries,
        table.stride(0),
        int(table.shape[1]),
        logits.stride(0),
        int(cache_2d.stride(0)),
        int(cache_2d.shape[0]),
        width,
        per,
        BLOCK_M=_ROWS_BLOCK_M,
        BLOCK_KV=_ROWS_BLOCK_KV,
        num_warps=1,
        # Six rows of queries and weights stay in registers; two waves per
        # SIMD is the most that fits them without spilling.
        waves_per_eu=_ROWS_WAVES_PER_EU,
    )


# Candidate (Reindex) scorer: one row per CTA; rows of a request pick
# different candidate blocks, so there is no key tile to share.
_CAND_BLOCK_M = 1
_CAND_WAVES_PER_EU = 4


@gluon.jit
def _cand_blocks(candidates, row, cand_stride, columns, end):
    """Candidate block id of each logits column (-1 past the row's width)."""
    return gl.load(
        candidates + row.to(gl.int64) * cand_stride + columns // 8,
        mask=columns < end,
        other=-1,
    ).to(gl.int32)


@gluon.jit
def _cand_resolve(
    block,
    page_table,
    row,
    table_stride,
    table_width,
    num_pages,
    columns,
    visible,
    page_stride_bytes,
):
    """Candidate blocks -> (key byte offset, scale byte offset, live)."""
    logical = gl.where(block >= 0, block * 8 + columns % 8, -1)
    live = (logical >= 0) & (logical < visible)
    logical = gl.where(live, logical, 0)
    page = gl.load(
        page_table
        + row.to(gl.int64) * table_stride
        + gl.minimum(logical // _ROWS_PAGE_SIZE, table_width - 1),
        mask=live,
        other=-1,
    )
    live = live & (page >= 0) & (page < num_pages)
    base = gl.where(live, page, 0).to(gl.int64) * page_stride_bytes
    page_row = (logical % _ROWS_PAGE_SIZE).to(gl.int64)
    offset = base + page_row * _PACKED_DIM
    scale_offset = base + _ROWS_PAGE_SIZE * _PACKED_DIM + page_row * _SCALE_DIM
    return offset, scale_offset, live


@gluon.jit
def _cand_load_keys(
    index_k_cache,
    offset,
    scale_offset,
    dot_b_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
):
    dims = gl.arange(0, _PACKED_DIM, layout=gl.SliceLayout(1, dot_b_layout))[:, None]
    key_offset = gl.convert_layout(offset, gl.SliceLayout(0, dot_b_layout))
    key = gl.load(index_k_cache + key_offset[None, :] + dims)
    words = gl.load(
        (
            index_k_cache
            + gl.convert_layout(scale_offset, gl.SliceLayout(1, b_scale_layout))
        ).to(gl.pointer_type(gl.int32))
    )
    return key, _rows_scale_bytes(words, b_scale_layout)


def _cand_launch_metadata(grid, kernel, args):
    queries, width = args["logits"].shape
    return {
        "name": kernel.name,
        "flops4": 2 * queries * _MFMA_HEADS * width * 128,
        "bytes": queries * width * (68 + 4 + 4)
        + args["q"].numel() * args["q"].element_size() * grid[1]
        + args["logits"].numel() * args["logits"].element_size(),
    }


@gluon.jit(
    launch_metadata=_cand_launch_metadata,
    # page_stride_bytes stays specializable (see gluon_dsv41_index_topk_gfx950).
    do_not_specialize=(
        "queries",
        "table_stride",
        "table_width",
        "cand_stride",
        "candidate_blocks",
        "logits_stride",
        "num_pages",
        "max_candidates",
        "tiles_per_split",
    ),
)
def gluon_dsv41_index_topk_candidates_gfx950(
    q,
    q_scales,
    weights,
    index_k_cache,
    visible,
    page_table,
    candidates,
    logits,
    queries,
    table_stride,
    table_width,
    cand_stride,
    candidate_blocks,
    logits_stride,
    page_stride_bytes,
    num_pages,
    max_candidates,
    tiles_per_split,
    BLOCK_M: gl.constexpr,
    BLOCK_KV: gl.constexpr,
):
    """Reindex scoring: column c of row q is key candidates[q, c // 8] * 8 + c % 8.

    Each row gathers its own candidate keys (rows of one request pick
    different blocks); BLOCK_M rows per CTA keep several independent gather
    chains in flight, and each row's MFMA tile covers eight 8-key blocks.
    """
    layouts: gl.constexpr = _rows_mfma_layouts(BLOCK_KV)
    mfma_layout: gl.constexpr = layouts[0]
    dot_a_layout: gl.constexpr = layouts[1]
    dot_b_layout: gl.constexpr = layouts[2]
    a_scale_layout: gl.constexpr = layouts[3]
    b_scale_layout: gl.constexpr = layouts[4]

    row0 = gl.program_id(0) * BLOCK_M
    split = gl.program_id(1)
    width = gl.minimum(candidate_blocks * 8, max_candidates)
    rows = ()
    ends = ()
    visibles = ()
    for r in gl.static_range(0, BLOCK_M):
        row = gl.minimum(row0 + r, queries - 1)
        vis = gl.minimum(
            gl.maximum(gl.load(visible + row).to(gl.int32), 0),
            table_width * _ROWS_PAGE_SIZE,
        )
        end = gl.where((row0 + r < queries) & (vis > 0), width, 0)
        rows = rows + (row,)
        ends = ends + (end,)
        visibles = visibles + (vis,)
    block_end = ends[0]
    for r in gl.static_range(1, BLOCK_M):
        block_end = gl.maximum(block_end, ends[r])

    tile_lo = split * tiles_per_split
    tile_hi = gl.minimum(
        tile_lo + tiles_per_split, (block_end + BLOCK_KV - 1) // BLOCK_KV
    )
    if tile_lo >= tile_hi:
        return

    qs = ()
    qss = ()
    ws = ()
    for r in gl.static_range(0, BLOCK_M):
        value, scale, weight = _rows_load_query(
            q, q_scales, weights, rows[r], mfma_layout, dot_a_layout, a_scale_layout
        )
        qs = qs + (value,)
        qss = qss + (scale,)
        ws = ws + (weight,)

    columns = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, mfma_layout))
    # Three-stage software pipeline per row: block ids two tiles ahead, their
    # page ids one tile ahead, then the keys, so no dependent load waits in
    # the tile that consumes it. The tail re-reads the last tile.
    last = tile_hi - 1
    first = tile_lo * BLOCK_KV + columns
    second = gl.minimum(tile_lo + 1, last) * BLOCK_KV + columns
    resolved = ()
    loaded = ()
    blocks = ()
    for r in gl.static_range(0, BLOCK_M):
        here = _cand_resolve(
            _cand_blocks(candidates, rows[r], cand_stride, first, ends[r]),
            page_table,
            rows[r],
            table_stride,
            table_width,
            num_pages,
            first,
            visibles[r],
            page_stride_bytes,
        )
        loaded = loaded + (
            _cand_load_keys(
                index_k_cache, here[0], here[1], dot_b_layout, b_scale_layout
            ),
        )
        resolved = resolved + (
            _cand_resolve(
                _cand_blocks(candidates, rows[r], cand_stride, second, ends[r]),
                page_table,
                rows[r],
                table_stride,
                table_width,
                num_pages,
                second,
                visibles[r],
                page_stride_bytes,
            ),
        )
        blocks = blocks + (here[2],)
    lives = blocks
    blocks = ()
    for r in gl.static_range(0, BLOCK_M):
        third = gl.minimum(tile_lo + 2, last) * BLOCK_KV + columns
        blocks = blocks + (
            _cand_blocks(candidates, rows[r], cand_stride, third, ends[r]),
        )
    for tile in range(tile_lo, tile_hi):
        positions = tile * BLOCK_KV + columns
        next_positions = gl.minimum(tile + 1, last) * BLOCK_KV + columns
        after_positions = gl.minimum(tile + 2, last) * BLOCK_KV + columns
        later_positions = gl.minimum(tile + 3, last) * BLOCK_KV + columns
        next_loaded = ()
        next_resolved = ()
        next_blocks = ()
        for r in gl.static_range(0, BLOCK_M):
            next_loaded = next_loaded + (
                _cand_load_keys(
                    index_k_cache,
                    resolved[r][0],
                    resolved[r][1],
                    dot_b_layout,
                    b_scale_layout,
                ),
            )
            next_resolved = next_resolved + (
                _cand_resolve(
                    blocks[r],
                    page_table,
                    rows[r],
                    table_stride,
                    table_width,
                    num_pages,
                    after_positions,
                    visibles[r],
                    page_stride_bytes,
                ),
            )
            next_blocks = next_blocks + (
                _cand_blocks(
                    candidates, rows[r], cand_stride, later_positions, ends[r]
                ),
            )
        next_lives = ()
        for r in gl.static_range(0, BLOCK_M):
            scores = _rows_score(
                qs[r], qss[r], ws[r], loaded[r][0], loaded[r][1], mfma_layout, BLOCK_KV
            )
            gl.amd.cdna4.buffer_store(
                scores,
                ptr=logits + rows[r].to(gl.int64) * logits_stride,
                offsets=positions,
                mask=lives[r] & (positions < ends[r]),
            )
            next_lives = next_lives + (resolved[r][2],)
        loaded = next_loaded
        lives = next_lives
        resolved = next_resolved
        blocks = next_blocks


def _launch_candidates(
    values, scales, w, cache_2d, table, visible, candidates, logits, max_tiles
):
    """Score Reindex logits with the candidate kernel, ``max_tiles`` per CTA."""
    queries, width = logits.shape
    groups = triton.cdiv(queries, _CAND_BLOCK_M)
    splits, per = _rows_splits(groups, width, max_tiles, _CAND_WAVES_PER_EU)
    gluon_dsv41_index_topk_candidates_gfx950[(groups, splits)](
        values.contiguous(),
        scales.contiguous(),
        w.float().contiguous(),
        cache_2d,
        visible,
        table,
        candidates,
        logits,
        queries,
        table.stride(0),
        int(table.shape[1]),
        candidates.stride(0),
        int(candidates.shape[1]),
        logits.stride(0),
        int(cache_2d.stride(0)),
        int(cache_2d.shape[0]),
        width,
        per,
        BLOCK_M=_CAND_BLOCK_M,
        BLOCK_KV=_ROWS_BLOCK_KV,
        num_warps=1,
        # One row's query and two key tiles leave room for four waves per
        # SIMD, which hide the candidate -> page -> key gather chain.
        waves_per_eu=_CAND_WAVES_PER_EU,
    )


def dsv41_index_logits_gfx950(
    values,
    scales,
    w,
    cache_2d,
    table,
    visible,
    candidates,
    logits,
    score_chunk_size,
):
    """Score prepared 32-head MXFP4 queries into caller-owned CSA2 logits.

    Args:
        values: Packed E2M1 query values shaped [T, 32, 64].
        scales: E8M0 query scales as int32 words shaped [T, 32].
        w: FP32 head weights shaped [T, 32].
        cache_2d: Page-planar MXFP4 bytes shaped [pages, 64 * 68].
        table: Physical page IDs shaped [T, logical_pages].
        visible: Visible logical row counts shaped [T].
        candidates: Optional candidate block IDs shaped [T, blocks].
        logits: FP32 destination shaped [T, scored_rows], initialized to -inf.
        score_chunk_size: Positive multiple-of-eight upper bound on rows per
            CTA; the launch picks a chunk up to it from the logits shape.

    With a chunk of at least 64 rows, Full mode (no candidates) uses the
    row-block kernel: each CTA scores six consecutive query rows and, when
    their page-table rows match over its split (a request's speculative rows,
    or a prefill chunk), loads every key page once for all six. Reindex mode
    (candidates) uses the candidate kernel: one row per CTA, its 8-row
    candidate blocks gathered in 64-column MFMA tiles through a pipelined
    candidate -> page -> key chain. Smaller chunks use the per-row scorer.

    Returns:
        None. ``logits`` is mutated in place.
    """
    score_chunk_size = int(score_chunk_size)
    if score_chunk_size < 8 or score_chunk_size % 8:
        raise ValueError("score_chunk_size must be a positive multiple of 8")
    if score_chunk_size >= _ROWS_BLOCK_KV:
        queries, width = logits.shape
        if not (queries and width and cache_2d.shape[0]):
            return
        max_tiles = min(_ROWS_MAX_TILES, score_chunk_size // _ROWS_BLOCK_KV)
        if candidates is None:
            _launch_rows(values, scales, w, cache_2d, table, visible, logits, max_tiles)
        else:
            _launch_candidates(
                values,
                scales,
                w,
                cache_2d,
                table,
                visible,
                candidates,
                logits,
                max_tiles,
            )
        return
    _launch_per_row(
        values,
        scales,
        w,
        cache_2d,
        table,
        visible,
        candidates,
        logits,
        score_chunk_size,
    )


def writes_visible_rows(candidates, score_chunk_size) -> bool:
    """Whether a launch writes every column below each row's visible length.

    The full-mode rows kernel does, so its caller may skip the -inf pre-fill
    of the logits when the consumers read only visible columns.
    """
    return candidates is None and int(score_chunk_size) >= _ROWS_BLOCK_KV


def _launch_per_row(
    values, scales, w, cache_2d, table, visible, candidates, logits, score_chunk_size
):
    """One CTA per (query row, score chunk); handles candidates and any chunk."""
    queries, width = logits.shape
    score_chunk_size = _score_chunk(queries, width, score_chunk_size)
    chunk_n = triton.cdiv(score_chunk_size, _BLOCK_N) * _BLOCK_N
    cand = table if candidates is None else candidates
    scale_dim = 4
    gluon_dsv41_index_topk_gfx950[(queries, triton.cdiv(width, score_chunk_size))](
        values,
        scales.view(torch.uint8).reshape(queries, _MFMA_HEADS, scale_dim),
        w,
        cache_2d,
        visible,
        table,
        cand,
        logits,
        values.stride(0),
        values.stride(1),
        scale_dim * _MFMA_HEADS,
        scale_dim,
        w.stride(0),
        w.stride(1),
        table.stride(0),
        cand.stride(0),
        logits.stride(0),
        int(cache_2d.stride(0)),
        int(cache_2d.shape[0]),
        width,
        NUM_HEADS=_MFMA_HEADS,
        PAGE_SIZE=_PAGE_SIZE,
        table_width=int(table.shape[1]),
        CANDIDATES=-1 if candidates is None else int(candidates.shape[1]),
        SCORE_CHUNK=score_chunk_size,
        BLOCK_N=_BLOCK_N,
        CHUNK_N=chunk_n,
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
        # The scorer is latency-bound on gathered key loads; more resident
        # waves hide it (96 decode rows x 75K history: 2 -> 8 cut 926 -> 617 us).
        waves_per_eu=8,
    )


# Radix-select digit split of the 32-bit order key: 12 + 10 + 10 bits. The
# first digit is histogrammed over the whole row; the selected bucket and every
# key above it are then compacted into LDS, so the later digits normally cost
# no extra global passes.
_SELECT_HIGH_BITS = 12
_SELECT_LOW_BITS = 10
# 16-wave CTAs while every row has its own CTA slot (two per CU); beyond
# that, 8-wave CTAs keep more rows in flight per CU.
_SELECT_WARPS = 16
_SELECT_WARPS_MANY_ROWS = 8
# Elements of the per-pass global scan per CTA: each thread loads
# _SELECT_REPEAT 16-byte vectors per step so several loads are in flight.
_SELECT_REPEAT = 2
# LDS-compacted keys (the first-digit bucket plus the keys above it). Rows whose
# compacted set is larger re-scan the logits for the later digits.
_SELECT_CAPACITY = 4096
# Rows up to 32 * _SELECT_BITMAP_WORDS columns order the compacted picks with a
# column bitmap instead of an all-pairs rank sort.
_SELECT_BITMAP_WORDS = 4096
assert 2 * _SELECT_BITMAP_WORDS <= 2 * _SELECT_CAPACITY  # shares the key LDS


def _select_launch_metadata(grid, kernel, args):
    """Two row passes on the compacted path; candidates add one id load per pick."""
    queries = grid[0]
    width = args["width"]
    topk = args["topk"]
    return {
        "name": kernel.name,
        "bytes": queries * (2 * width * 4 + topk * (4 + 8)),
    }


@gluon.jit
def _select_order_key(values):
    # Monotone map from FP32 to uint32: larger value -> larger key.
    bits = values.to(gl.uint32, bitcast=True)
    flip = gl.where(
        (bits >> 31) != 0,
        gl.full(bits.shape, 0xFFFFFFFF, gl.uint32, bits.type.layout),
        gl.full(bits.shape, 0x80000000, gl.uint32, bits.type.layout),
    )
    return bits ^ flip


@gluon.jit
def _select_load(row, col, width, limit, ALIGNED: gl.constexpr):
    """Row values at ``col``; columns at or past ``width`` read as -inf.

    With ``ALIGNED`` the mask is ``limit`` (``width`` rounded up to a multiple
    of 4 within the row), so the loads stay 16-byte vectors.
    """
    # Buffer loads predicate out-of-range lanes without a branch, so the
    # loads of one chunk issue back to back.
    values = gl.amd.cdna4.buffer_load(row, col, mask=col < limit, other=-float("inf"))
    if ALIGNED:
        values = gl.where(col < width, values, -float("inf"))
    return values


@gluon.jit
def _select_popcount(x):
    # Set bits per int32 lane (SWAR).
    x = x.to(gl.uint32, bitcast=True)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24).to(gl.int32)


@gluon.jit
def _select_pick_combine(count_a, kept_a, higher_a, count_b, kept_b, higher_b):
    return (
        count_a + count_b,
        gl.minimum(kept_a, kept_b),
        gl.maximum(higher_a, higher_b),
    )


@gluon.jit
def _select_pick(hist, d, remaining):
    """Digit of the ``remaining``-th largest key in ``hist`` and the count above it."""
    # at_or_above[d] counts keys whose digit is >= d; the K-th key's digit is
    # the largest d with at_or_above[d] >= remaining.
    at_or_above = gl.cumsum(hist, 0, reverse=True)
    # at_or_above falls as d grows, so in one fused reduction: the digits at
    # or below the pick count it, the smallest at_or_above[d] >= remaining is
    # the keys kept at the pick and the largest below it the keys above it.
    reaches = at_or_above >= remaining
    count, kept, higher = gl.reduce(
        (
            reaches.to(gl.int32),
            gl.where(reaches, at_or_above, 0x7FFFFFFF),
            gl.where(reaches, 0, at_or_above),
        ),
        0,
        _select_pick_combine,
    )
    # With nothing to select (remaining == 0) every digit reaches it.
    kept = gl.where(count == 0, 0, kept)
    return gl.maximum(count - 1, 0), higher, kept


@gluon.jit
def _select_radix_pass(
    row,
    width,
    limit,
    prefix,
    above,
    remaining,
    hist_smem,
    ones,
    d,
    SHIFT: gl.constexpr,
    DIGIT_BITS: gl.constexpr,
    BLOCK: gl.constexpr,
    BINS: gl.constexpr,
    L: gl.constexpr,
    LBIN: gl.constexpr,
    ALIGNED: gl.constexpr,
):
    """Histogram one later key digit of the live row keys matching ``prefix``."""
    hist_smem.store(gl.zeros([BINS], gl.int32, layout=LBIN))
    for c0 in range(0, width, BLOCK):
        col = c0 + gl.arange(0, BLOCK, layout=L)
        values = _select_load(row, col, width, limit, ALIGNED)
        key = _select_order_key(values)
        live = (values > -float("inf")) & ((key >> (SHIFT + DIGIT_BITS)) == prefix)
        digit = ((key >> SHIFT) & ((1 << DIGIT_BITS) - 1)).to(gl.int32)
        hist_smem.atomic_scatter_add(ones, digit, axis=0, mask=live)
    pick, higher, _ = _select_pick(hist_smem.load(LBIN), d, remaining)
    prefix = (prefix << DIGIT_BITS) | pick.to(gl.uint32)
    return prefix, above + higher, remaining - higher


@gluon.jit
def _select_kept_pass(
    keys,
    kept,
    prefix,
    remaining,
    hist_smem,
    ones,
    d,
    SHIFT: gl.constexpr,
    DIGIT_BITS: gl.constexpr,
    BINS: gl.constexpr,
    LBIN: gl.constexpr,
):
    """Histogram one key digit of the LDS-resident keys matching ``prefix``.

    ``prefix`` holds the key bits above this digit (ignored for the top digit).
    """
    hist_smem.store(gl.zeros([BINS], gl.int32, layout=LBIN))
    match = kept
    if SHIFT + DIGIT_BITS < 32:
        match = match & ((keys >> (SHIFT + DIGIT_BITS)) == prefix)
    digit = ((keys >> SHIFT) & ((1 << DIGIT_BITS) - 1)).to(gl.int32)
    hist_smem.atomic_scatter_add(ones, digit, axis=0, mask=match)
    pick, higher, _ = _select_pick(hist_smem.load(LBIN), d, remaining)
    if SHIFT + DIGIT_BITS < 32:
        prefix = (prefix << DIGIT_BITS) | pick.to(gl.uint32)
    else:
        prefix = pick.to(gl.uint32)
    return prefix, remaining - higher


@gluon.jit
def _select_kept_key(
    keys,
    kept,
    remaining,
    hist_smem,
    ones,
    d,
    HIGH_BITS: gl.constexpr,
    LOW_BITS: gl.constexpr,
    BINS: gl.constexpr,
    LBIN: gl.constexpr,
):
    """Order key of the ``remaining``-th largest LDS-resident ``kept`` key."""
    prefix, remaining = _select_kept_pass(
        keys,
        kept,
        0,
        remaining,
        hist_smem,
        ones,
        d,
        2 * LOW_BITS,
        HIGH_BITS,
        BINS,
        LBIN,
    )
    prefix, remaining = _select_kept_pass(
        keys,
        kept,
        prefix,
        remaining,
        hist_smem,
        ones,
        d,
        LOW_BITS,
        LOW_BITS,
        BINS,
        LBIN,
    )
    prefix, remaining = _select_kept_pass(
        keys,
        kept,
        prefix,
        remaining,
        hist_smem,
        ones,
        d,
        0,
        LOW_BITS,
        BINS,
        LBIN,
    )
    return prefix, remaining


@gluon.jit
def _select_scatter_picks(
    picked_smem,
    key,
    col,
    live,
    threshold,
    remaining,
    n_above,
    n_ties,
    candidates,
    q,
    cand_stride,
    TOPK: gl.constexpr,
    HAS_CANDIDATES: gl.constexpr,
):
    """Column-order compaction of one chunk into ``picked_smem``.

    Keeps keys above ``threshold`` plus the first ``remaining`` ties; returns
    the updated running counts.
    """
    is_above = live & (key > threshold)
    is_tie = live & (key == threshold)
    # One packed scan counts both: low 16 bits the keys above the threshold,
    # high 16 bits the ties (a chunk holds fewer than 2**16).
    packed = is_above.to(gl.int32) | (is_tie.to(gl.int32) << 16)
    before = gl.cumsum(packed, 0) - packed
    ties_before = n_ties + (before >> 16)
    take_it = is_above | (is_tie & (ties_before < remaining))
    slot = n_above + (before & 0xFFFF) + gl.minimum(ties_before, remaining)
    if HAS_CANDIDATES:
        block = gl.load(
            candidates + q * cand_stride + col // 8, mask=take_it, other=-1
        ).to(gl.int32)
        logical = block * 8 + col % 8
    else:
        logical = col
    # A masked exchange is a scatter of the picked lanes only.
    picked_smem.atomic_scatter_xchg(logical, slot, axis=0, mask=take_it)
    total = gl.sum(packed, 0)
    return n_above + (total & 0xFFFF), n_ties + (total >> 16)


@gluon.jit(
    launch_metadata=_select_launch_metadata,
    do_not_specialize=(
        "logits_stride",
        "cand_stride",
        "out_stride",
        "width",
        "topk",
        "take",
    ),
)
def gluon_dsv41_index_topk_select_gfx950(
    logits,  # [Q, width] fp32, -inf = unscored
    candidates,  # [Q, blocks] int32/int64 block ids, or logits when unused
    row_limits,  # [Q] live widths, or logits when unused
    row_out,  # [Q, TOPK] int32
    row_lens,  # [Q] int32
    logits_stride,
    cand_stride,
    out_stride,
    width,
    topk,  # row_out width
    take,  # min(topk, width)
    TOPK: gl.constexpr,  # power-of-two capacity >= topk, at least 512
    HAS_CANDIDATES: gl.constexpr,
    HAS_LIMITS: gl.constexpr,
    HIGH_BITS: gl.constexpr,
    LOW_BITS: gl.constexpr,
    REPEAT: gl.constexpr,
    CAPACITY: gl.constexpr,
    ALIGNED: gl.constexpr,  # 16-byte aligned rows whose width is a multiple of 4
    BITMAP_WORDS: gl.constexpr,  # >0: column bitmap words (32 * words >= width)
    NUM_WARPS: gl.constexpr,
):
    """Top-``take`` finite logits of one query row as position-sorted ids.

    An MSD radix select over the 32-bit order key finds the K-th largest finite
    logit (K = min(take, finite count)). Pass 1 histograms the high digit over
    the row; pass 2 appends every key in or above the picked bucket to LDS,
    where the two low digits (and, if needed, the lowest-column boundary ties)
    are resolved without touching global memory again. A column bitmap orders
    the picks by position. If the compacted set would overflow ``CAPACITY``,
    the low digits and a column-order compaction re-scan the row instead.
    Picked logical ids that are not ascending (shuffled candidates, or rows too
    wide for the bitmap) are ordered by an all-pairs rank sort. Positions >= K
    are written as -1.
    """
    VEC_BLOCK: gl.constexpr = 4 * 64 * NUM_WARPS
    BLOCK: gl.constexpr = VEC_BLOCK * REPEAT
    BINS: gl.constexpr = 1 << HIGH_BITS
    MID_SHIFT: gl.constexpr = LOW_BITS
    HIGH_SHIFT: gl.constexpr = 2 * LOW_BITS
    L: gl.constexpr = gl.BlockedLayout([4], [64], [NUM_WARPS], [0])
    LBIN: gl.constexpr = gl.BlockedLayout(
        [BINS // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    LK: gl.constexpr = gl.BlockedLayout(
        [CAPACITY // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    SL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    q = gl.program_id(0).to(gl.int64)
    if ALIGNED:
        # Spelled as a multiple of 4 so the compiler sees 16-byte alignment.
        row = logits + q * (logits_stride // 4) * 4
    else:
        row = logits + q * logits_stride
    if HAS_LIMITS:
        # Columns at or past the row's live width are -inf: skip them.
        width = gl.minimum(width, gl.maximum(gl.load(row_limits + q).to(gl.int32), 0))
    limit = width
    if ALIGNED:
        limit = (width + 3) // 4 * 4
    hist_smem = gl.allocate_shared_memory(gl.int32, [BINS], SL)
    ones = gl.full([BLOCK], 1, gl.int32, layout=L)
    d = gl.arange(0, BINS, layout=LBIN)

    # Pass 1: histogram the high digit of every live key.
    hist_smem.store(gl.zeros([BINS], gl.int32, layout=LBIN))
    for c0 in range(0, width, BLOCK):
        col = c0 + gl.arange(0, BLOCK, layout=L)
        values = _select_load(row, col, width, limit, ALIGNED)
        digit = (_select_order_key(values) >> HIGH_SHIFT).to(gl.int32)
        hist_smem.atomic_scatter_add(ones, digit, axis=0, mask=values > -float("inf"))
    hist = hist_smem.load(LBIN)
    count = gl.minimum(take, gl.sum(hist, 0))
    # ``prefix`` holds the fixed high bits of the K-th key, ``remaining`` its
    # rank among keys sharing them and ``above`` the count of larger keys.
    pick, above, kept = _select_pick(hist, d, count)
    prefix = pick.to(gl.uint32)
    remaining = count - above

    # Output ids are int32, so the int32 max marks an unpicked slot.
    picked_smem = gl.allocate_shared_memory(gl.int32, [TOPK], SL)
    LP: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    picked_smem.store(gl.full([TOPK], 0x7FFFFFFF, gl.int32, layout=LP))
    # One allocation: compacted keys (as int32 bits), then their columns.
    kept_smem = gl.allocate_shared_memory(gl.int32, [2 * CAPACITY], SL)
    # LDS slot counters: [0] for the pass-2 append, [1] for the picks.
    count_smem = gl.allocate_shared_memory(gl.int32, [64], SL)
    LC64: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    key_smem = kept_smem.slice(0, CAPACITY)
    col_smem = kept_smem.slice(CAPACITY, CAPACITY)
    if kept <= CAPACITY - 64:
        # Pass 2: append every live key in or above the picked bucket to LDS.
        # Slots come from an LDS counter, so the order is arbitrary; only the
        # few kept lanes touch LDS.
        count_smem.store(gl.zeros([64], gl.int32, layout=LC64))
        # Each thread reserves slots for its 4-column vector with one atomic.
        L2: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [NUM_WARPS, 1], [1, 0])
        VECS: gl.constexpr = BLOCK // 4
        vec = gl.arange(0, VECS, layout=gl.SliceLayout(1, L2))
        lane = gl.arange(0, 4, layout=gl.SliceLayout(0, L2))
        offs = gl.expand_dims(vec * 4, 1) + gl.expand_dims(lane, 0)
        zero = gl.zeros([VECS], gl.int32, layout=gl.SliceLayout(1, L2))
        scratch = gl.reshape(
            CAPACITY - 64 + gl.expand_dims(vec % 64, 1) + gl.expand_dims(lane * 0, 0),
            [BLOCK],
        )
        for c0 in range(0, width, BLOCK):
            col = c0 + offs
            values = _select_load(row, col, width, limit, ALIGNED)
            key = _select_order_key(values)
            keep = (values > -float("inf")) & ((key >> HIGH_SHIFT) >= prefix)
            flag = keep.to(gl.int32)
            n_vec = gl.sum(flag, 1)
            base = count_smem.atomic_scatter_add(n_vec, zero, axis=0, mask=n_vec > 0)
            slot = gl.reshape(
                gl.expand_dims(base, 1) + gl.cumsum(flag, 1) - flag, [BLOCK]
            )
            # Unkept lanes write one of the last 64 (per-lane scratch) slots: plain
            # stores need no masked atomics or waits.
            slot = gl.where(gl.reshape(flag, [BLOCK]) != 0, slot, scratch)
            key_smem.scatter(
                gl.reshape(key.to(gl.int32, bitcast=True), [BLOCK]), slot, axis=0
            )
            col_smem.scatter(gl.reshape(col, [BLOCK]), slot, axis=0)
        gl.barrier()
        i = gl.arange(0, CAPACITY, layout=LK)
        live = i < kept
        keys = key_smem.load(LK).to(gl.uint32, bitcast=True)
        cols = col_smem.load(LK)
        ones_k = gl.full([CAPACITY], 1, gl.int32, layout=LK)
        LOW_BINS: gl.constexpr = 1 << LOW_BITS
        LLOW: gl.constexpr = gl.BlockedLayout(
            [LOW_BINS // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
        )
        low_smem = hist_smem.slice(0, LOW_BINS)
        d_low = gl.arange(0, LOW_BINS, layout=LLOW)
        # The low digits only see keys sharing the picked high digit.
        prefix, remaining = _select_kept_pass(
            keys,
            live,
            prefix,
            remaining,
            low_smem,
            ones_k,
            d_low,
            MID_SHIFT,
            LOW_BITS,
            LOW_BINS,
            LLOW,
        )
        threshold, remaining = _select_kept_pass(
            keys,
            live,
            prefix,
            remaining,
            low_smem,
            ones_k,
            d_low,
            0,
            LOW_BITS,
            LOW_BINS,
            LLOW,
        )
        is_above = live & (keys > threshold)
        is_tie = live & (keys == threshold)
        # Boundary ties keep the lowest columns: when more keys tie than are
        # taken, select the ``remaining`` largest of ~column among the ties.
        tie_key = gl.where(is_tie, ~cols.to(gl.uint32, bitcast=True), 0)
        tie_floor = gl.to_tensor(0).to(gl.uint32)
        if gl.sum(is_tie.to(gl.int32), 0) > remaining:
            tie_floor, _ = _select_kept_key(
                tie_key,
                is_tie,
                remaining,
                hist_smem,
                ones_k,
                d,
                HIGH_BITS,
                LOW_BITS,
                BINS,
                LBIN,
            )
        take_it = is_above | (is_tie & (tie_key >= tie_floor))
        if HAS_CANDIDATES:
            block = gl.load(
                candidates + q * cand_stride + cols // 8, mask=take_it, other=-1
            ).to(gl.int32)
            logical = block * 8 + cols % 8
        else:
            logical = cols
        if BITMAP_WORDS > 0:
            # Column-order slots: set one bit per picked column, then a pick's
            # slot is the number of set bits before it.
            LW: gl.constexpr = gl.BlockedLayout(
                [BITMAP_WORDS // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
            )
            # The compacted keys are in registers now: reuse their LDS.
            bits_smem = kept_smem.slice(0, BITMAP_WORDS)
            before_smem = kept_smem.slice(BITMAP_WORDS, BITMAP_WORDS)
            bits_smem.store(gl.zeros([BITMAP_WORDS], gl.int32, layout=LW))
            word = cols >> 5
            bit = gl.full([CAPACITY], 1, gl.int32, layout=LK) << (cols & 31)
            bits_smem.atomic_scatter_or(bit, word, axis=0, mask=take_it)
            words = bits_smem.load(LW)
            set_bits = _select_popcount(words)
            before_smem.store(gl.cumsum(set_bits, 0) - set_bits)
            slot = gl.where(
                take_it,
                before_smem.gather(gl.where(take_it, word, 0), axis=0)
                + _select_popcount(
                    bits_smem.gather(gl.where(take_it, word, 0), axis=0) & (bit - 1)
                ),
                0,
            )
        else:
            slot = count_smem.atomic_scatter_add(
                ones_k,
                gl.full([CAPACITY], 1, gl.int32, layout=LK),
                axis=0,
                mask=take_it,
            )
        picked_smem.atomic_scatter_xchg(logical, slot, axis=0, mask=take_it)
    else:
        # Fallback for a crowded high bucket: resolve the low digits and
        # compact the picks with further passes over the row.
        prefix, above, remaining = _select_radix_pass(
            row,
            width,
            limit,
            prefix,
            above,
            remaining,
            hist_smem,
            ones,
            d,
            MID_SHIFT,
            LOW_BITS,
            BLOCK,
            BINS,
            L,
            LBIN,
            ALIGNED,
        )
        prefix, above, remaining = _select_radix_pass(
            row,
            width,
            limit,
            prefix,
            above,
            remaining,
            hist_smem,
            ones,
            d,
            0,
            LOW_BITS,
            BLOCK,
            BINS,
            L,
            LBIN,
            ALIGNED,
        )
        n_above = 0
        n_ties = 0
        for c0 in range(0, width, BLOCK):
            col = c0 + gl.arange(0, BLOCK, layout=L)
            values = _select_load(row, col, width, limit, ALIGNED)
            n_above, n_ties = _select_scatter_picks(
                picked_smem,
                _select_order_key(values),
                col,
                values > -float("inf"),
                prefix,
                remaining,
                n_above,
                n_ties,
                candidates,
                q,
                cand_stride,
                TOPK,
                HAS_CANDIDATES,
            )

    # Rank sort of the picked ids: ids are unique, unpicked slots hold the
    # int32 max and rank after them.
    LT: gl.constexpr = gl.BlockedLayout([1, 16], [64, 1], [NUM_WARPS, 1], [1, 0])
    LR: gl.constexpr = gl.SliceLayout(1, LT)
    LC: gl.constexpr = gl.SliceLayout(0, LT)
    mine = picked_smem.load(LR)
    i = gl.arange(0, TOPK, layout=LR)
    rank = i
    # Both paths write column-ordered picks when the bitmap covers the row.
    # Column order already is position order without candidates, and with
    # candidate blocks listed in ascending order; sort only otherwise.
    descents = 0
    if HAS_CANDIDATES or BITMAP_WORDS == 0:
        previous = gl.gather(mine, gl.maximum(i - 1, 0), axis=0)
        descents = gl.sum(((i > 0) & (i < count) & (mine < previous)).to(gl.int32), 0)
    if descents > 0:
        rank = gl.zeros([TOPK], gl.int32, layout=LR)
        for j0 in gl.static_range(0, TOPK, 16):
            other = picked_smem.slice(j0, 16).load(LC)
            rank += gl.sum(
                (gl.expand_dims(other, 0) < gl.expand_dims(mine, 1)).to(gl.int32),
                axis=1,
            )
    out = row_out + q * out_stride
    gl.store(out + rank, mine, mask=i < count)
    gl.store(
        out + i,
        gl.full([TOPK], -1, gl.int32, layout=LR),
        mask=(i >= count) & (i < topk),
    )
    gl.store(row_lens + q, count)


def launch_gluon_dsv41_index_topk_select_gfx950(
    logits, candidates, live_widths, topk, row_out, row_lens
):
    """Select CSA2 rows from scored logits, position-sorted.

    Args:
        logits: FP32 [Q, width] scores; -inf marks unscored rows.
        candidates: None, or int32/int64 [Q, blocks] block ids whose 8-row
            blocks map logits column c to logical row
            ``candidates[q, c // 8] * 8 + c % 8``.
        live_widths: None, or int32/int64 [Q] counts of leading columns that
            may be finite; row q's columns at or past ``live_widths[q]`` must be
            -inf and are not read. Full selection passes the visible lengths, so
            a page-table-capacity row costs its context, not its capacity.
        topk: Selection capacity in [1, 1024]; V4.1 uses 512.
        row_out: Int32 [Q, topk] destination. Row q receives the logical ids of
            its ``min(topk, width)`` largest finite logits in ascending order,
            then -1 padding. Equal-score boundary ties keep the lowest columns.
        row_lens: Int32 [Q] destination for the number of ids written.

    Returns:
        None.
    """
    queries, width = logits.shape
    if not queries:
        return
    topk = int(topk)
    if not 1 <= topk <= 1024:
        raise ValueError(f"select topk must be in [1, 1024], got {topk}")
    if (
        logits.stride(1) != 1
        or row_out.stride(1) != 1
        or row_lens.stride(0) != 1
        or (candidates is not None and candidates.stride(1) != 1)
    ):
        raise ValueError("select requires unit inner strides")
    # The kernel writes row_out[q, :topk] and row_lens[q], and reads
    # candidates[q, col // 8] for every logits column.
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
    cand = logits if candidates is None else candidates
    warps = _SELECT_WARPS if queries <= 2 * num_cus() else _SELECT_WARPS_MANY_ROWS
    gluon_dsv41_index_topk_select_gfx950[(queries,)](
        logits,
        cand,
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
        HAS_CANDIDATES=candidates is not None,
        HAS_LIMITS=live_widths is not None,
        HIGH_BITS=_SELECT_HIGH_BITS,
        LOW_BITS=_SELECT_LOW_BITS,
        REPEAT=_SELECT_REPEAT,
        CAPACITY=_SELECT_CAPACITY,
        BITMAP_WORDS=_SELECT_BITMAP_WORDS if width <= 32 * _SELECT_BITMAP_WORDS else 0,
        ALIGNED=logits.stride(0) % 4 == 0
        and width % 4 == 0
        and logits.data_ptr() % 16 == 0,
        NUM_WARPS=warps,
        num_warps=warps,
        # Up to 8 waves per SIMD (two 16-wave CTAs per CU): the row scans are
        # latency bound, so occupancy beats registers.
        waves_per_eu=8,
    )
