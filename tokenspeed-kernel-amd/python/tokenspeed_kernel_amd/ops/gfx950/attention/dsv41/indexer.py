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
matching page tables (speculative decode, prefill); candidate mode scores one
row per CTA. The tokenspeed-kernel adapter owns query packing, bounded query
tiling, and selection.
"""

from __future__ import annotations

import torch
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
    live = (positions < end) & (page >= 0) & (page < num_pages)
    gl.amd.cdna4.buffer_store(
        scores,
        ptr=logits + row.to(gl.int64) * logits_stride,
        offsets=positions,
        mask=live,
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


def _rows_splits(groups: int, width: int, max_tiles: int, device) -> tuple[int, int]:
    """(splits, tiles per split) for a row-block launch over ``width`` keys."""
    tiles = triton.cdiv(width, _ROWS_BLOCK_KV)
    target = 4 * _num_cus(device) * _ROWS_WAVES_PER_EU
    splits = max(triton.cdiv(target, groups), triton.cdiv(tiles, max_tiles))
    per = max(triton.cdiv(tiles, splits), _ROWS_MIN_TILES)
    per = min(per, max_tiles)
    return triton.cdiv(tiles, per), per


_CU_COUNTS: dict[int, int] = {}


def _num_cus(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _CU_COUNTS:
        _CU_COUNTS[index] = torch.cuda.get_device_properties(
            index
        ).multi_processor_count
    return _CU_COUNTS[index]


def _launch_rows(values, scales, w, cache_2d, table, visible, logits, max_tiles):
    """Score Full-mode logits with the row-block kernel, ``max_tiles`` per CTA."""
    queries, width = logits.shape
    groups = triton.cdiv(queries, _ROWS_BLOCK_M)
    splits, per = _rows_splits(groups, width, max_tiles, logits.device)
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

    Full mode (no candidates) with a chunk of at least 64 rows uses the
    row-block kernel: each CTA scores six consecutive query rows and, when
    their page-table rows match over its split (a request's speculative rows,
    or a prefill chunk), loads every key page once for all six.

    Returns:
        None. ``logits`` is mutated in place.
    """
    score_chunk_size = int(score_chunk_size)
    if score_chunk_size < 8 or score_chunk_size % 8:
        raise ValueError("score_chunk_size must be a positive multiple of 8")
    if candidates is None and score_chunk_size >= _ROWS_BLOCK_KV:
        queries, width = logits.shape
        if queries and width and cache_2d.shape[0]:
            _launch_rows(
                values,
                scales,
                w,
                cache_2d,
                table,
                visible,
                logits,
                min(_ROWS_MAX_TILES, score_chunk_size // _ROWS_BLOCK_KV),
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


# Radix-select digit split of the 32-bit order key: 11 + 11 + 10 bits.
_SELECT_RADIX_BITS = 11
_SELECT_WARPS = 8


def _select_launch_metadata(grid, kernel, args):
    """Each pass re-reads the logits row; candidates add one id load per pick."""
    queries = grid[0]
    width = args["width"]
    topk = args["topk"]
    return {
        "name": kernel.name,
        "bytes": queries * (4 * width * 4 + topk * (4 + 8)),
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
def _select_radix_pass(
    row,
    width,
    take,
    prefix,
    above,
    remaining,
    hist_smem,
    ones,
    d,
    SHIFT: gl.constexpr,
    DIGIT_BITS: gl.constexpr,
    FIRST: gl.constexpr,
    BLOCK: gl.constexpr,
    BINS: gl.constexpr,
    L: gl.constexpr,
    LBIN: gl.constexpr,
):
    """Histogram one key digit of the live keys matching ``prefix``."""
    hist_smem.store(gl.zeros([BINS], gl.int32, layout=LBIN))
    for c0 in range(0, width, BLOCK):
        col = c0 + gl.arange(0, BLOCK, layout=L)
        values = gl.load(row + col, mask=col < width, other=-float("inf"))
        key = _select_order_key(values)
        live = values > -float("inf")
        if not FIRST:
            live = live & ((key >> (SHIFT + DIGIT_BITS)) == prefix)
        digit = ((key >> SHIFT) & ((1 << DIGIT_BITS) - 1)).to(gl.int32)
        hist_smem.atomic_scatter_add(ones, digit, axis=0, mask=live)
    hist = hist_smem.load(LBIN)
    if FIRST:
        remaining = gl.minimum(take, gl.sum(hist, 0))
    # at_or_above[d] counts keys whose digit is >= d; the K-th key's digit is
    # the largest d with at_or_above[d] >= remaining.
    at_or_above = gl.cumsum(hist, 0, reverse=True)
    pick = gl.maximum(gl.sum((at_or_above >= remaining).to(gl.int32), 0) - 1, 0)
    higher = gl.sum(gl.where(d > pick, hist, 0), 0)
    if FIRST:
        prefix = pick.to(gl.uint32)
    else:
        prefix = (prefix << DIGIT_BITS) | pick.to(gl.uint32)
    return prefix, above + higher, remaining - higher


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
    TOPK: gl.constexpr,  # power-of-two capacity >= topk, at least 256
    HAS_CANDIDATES: gl.constexpr,
    HAS_LIMITS: gl.constexpr,
    RADIX_BITS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Top-``take`` finite logits of one query row as position-sorted ids.

    A three-pass MSD radix select finds the order key of the K-th largest
    finite logit (K = min(take, finite count)); a column-order compaction then
    keeps every larger key plus the lowest-column ties, and an all-pairs rank
    sorts the picked logical ids. Positions >= K are written as -1.
    """
    BLOCK: gl.constexpr = 4 * 64 * NUM_WARPS
    BINS: gl.constexpr = 1 << RADIX_BITS
    L: gl.constexpr = gl.BlockedLayout([4], [64], [NUM_WARPS], [0])
    LBIN: gl.constexpr = gl.BlockedLayout(
        [BINS // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    SL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    q = gl.program_id(0).to(gl.int64)
    row = logits + q * logits_stride
    if HAS_LIMITS:
        # Columns at or past the row's live width are -inf: skip them.
        width = gl.minimum(width, gl.maximum(gl.load(row_limits + q).to(gl.int32), 0))
    hist_smem = gl.allocate_shared_memory(gl.int32, [BINS], SL)
    ones = gl.full([BLOCK], 1, gl.int32, layout=L)
    d = gl.arange(0, BINS, layout=LBIN)

    # MSD radix select: after each pass, ``prefix`` holds the fixed high bits
    # of the K-th key, ``remaining`` its rank among keys sharing them and
    # ``above`` the count of larger keys.
    LOW_BITS: gl.constexpr = 32 - 2 * RADIX_BITS
    prefix, above, remaining = _select_radix_pass(
        row,
        width,
        take,
        0,
        0,
        0,
        hist_smem,
        ones,
        d,
        32 - RADIX_BITS,
        RADIX_BITS,
        True,
        BLOCK,
        BINS,
        L,
        LBIN,
    )
    prefix, above, remaining = _select_radix_pass(
        row,
        width,
        take,
        prefix,
        above,
        remaining,
        hist_smem,
        ones,
        d,
        LOW_BITS,
        RADIX_BITS,
        False,
        BLOCK,
        BINS,
        L,
        LBIN,
    )
    prefix, above, remaining = _select_radix_pass(
        row,
        width,
        take,
        prefix,
        above,
        remaining,
        hist_smem,
        ones,
        d,
        0,
        LOW_BITS,
        False,
        BLOCK,
        BINS,
        L,
        LBIN,
    )
    threshold = prefix
    count = above + remaining

    # Column-order compaction: keys above the threshold plus the first
    # ``remaining`` ties (equal-score ties may pick any equivalent subset).
    picked_smem = gl.allocate_shared_memory(gl.int64, [2 * TOPK], SL)
    LP: gl.constexpr = gl.BlockedLayout(
        [2 * TOPK // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    picked_smem.store(gl.full([2 * TOPK], 0x7FFFFFFFFFFFFFFF, gl.int64, layout=LP))
    # One packed scan per chunk counts both: low 16 bits the keys above the
    # threshold, high 16 bits the ties (a chunk holds fewer than 2**16).
    n_above = 0
    n_ties = 0
    for c0 in range(0, width, BLOCK):
        col = c0 + gl.arange(0, BLOCK, layout=L)
        values = gl.load(row + col, mask=col < width, other=-float("inf"))
        key = _select_order_key(values)
        live = values > -float("inf")
        is_above = live & (key > threshold)
        is_tie = live & (key == threshold)
        packed = is_above.to(gl.int32) | (is_tie.to(gl.int32) << 16)
        before = gl.cumsum(packed, 0) - packed
        ties_before = n_ties + (before >> 16)
        take_it = is_above | (is_tie & (ties_before < remaining))
        slot = n_above + (before & 0xFFFF) + gl.minimum(ties_before, remaining)
        if HAS_CANDIDATES:
            block = gl.load(
                candidates + q * cand_stride + col // 8, mask=take_it, other=-1
            ).to(gl.int64)
            logical = block * 8 + (col % 8).to(gl.int64)
        else:
            logical = col.to(gl.int64)
        # Unpicked lanes write the scratch half [TOPK, 2*TOPK).
        picked_smem.scatter(logical, gl.where(take_it, slot, TOPK + col % TOPK), axis=0)
        total = gl.sum(packed, 0)
        n_above += total & 0xFFFF
        n_ties += total >> 16

    # Rank sort of the picked ids: ids are unique, unpicked slots hold the
    # int64 max and rank after them.
    LT: gl.constexpr = gl.BlockedLayout([1, 64], [64, 1], [NUM_WARPS, 1], [1, 0])
    LR: gl.constexpr = gl.SliceLayout(1, LT)
    LC: gl.constexpr = gl.SliceLayout(0, LT)
    mine = picked_smem.slice(0, TOPK).load(LR)
    i = gl.arange(0, TOPK, layout=LR)
    rank = i
    # Column order already is position order without candidates, and with
    # candidate blocks listed in ascending order; sort only otherwise.
    previous = gl.gather(mine, gl.maximum(i - 1, 0), axis=0)
    descents = gl.sum(((i > 0) & (i < count) & (mine < previous)).to(gl.int32), 0)
    if descents > 0:
        rank = gl.zeros([TOPK], gl.int32, layout=LR)
        for j0 in gl.static_range(0, TOPK, 64):
            other = picked_smem.slice(j0, 64).load(LC)
            rank += gl.sum(
                (gl.expand_dims(other, 0) < gl.expand_dims(mine, 1)).to(gl.int32),
                axis=1,
            )
    out = row_out + q * out_stride
    gl.store(out + rank, mine.to(gl.int32), mask=i < count)
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
        TOPK=max(256, triton.next_power_of_2(topk)),
        HAS_CANDIDATES=candidates is not None,
        HAS_LIMITS=live_widths is not None,
        RADIX_BITS=_SELECT_RADIX_BITS,
        NUM_WARPS=_SELECT_WARPS,
        num_warps=_SELECT_WARPS,
    )
