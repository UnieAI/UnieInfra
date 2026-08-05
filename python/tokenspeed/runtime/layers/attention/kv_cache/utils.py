# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the FluentLLM project
#
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

import torch
import triton
import triton.language as tl


def normalize_layer_kv_geometry(
    layer_kv_geometry: Sequence[tuple[int, int]],
) -> tuple[int, tuple[int, ...]]:
    """Fold per-layer ``(kv_heads, head_dim)`` onto ONE common head dim.

    The heterogeneous-KV path in :class:`MHATokenToKVPool` describes a layer
    by a head COUNT alone: every layer shares the pool's ``head_dim`` and a
    narrower layer reinterprets the byte-uniform slab row as more, shorter
    rows. Models whose layers also differ in head dim (Gemma 4: 16 x 256 on
    sliding layers, 4 x 512 on full layers) are expressible in that path
    without any new pool geometry, because ``4 x 512`` occupies exactly the
    bytes of ``8 x 256`` -- the same row, counted in the common dim.

    This is a pure re-count of the SAME bytes in the same order: head h of a
    512-dim layer is heads 2h and 2h+1 of the 256-dim view, contiguous and
    in order, so the attention backend's
    ``get_key_buffer(l).view(-1, page, real_heads, real_dim)`` recovers the
    checkpoint geometry byte-for-byte.

    Args:
        layer_kv_geometry: Per-layer ``(kv_heads, head_dim)`` PRE-TP, in
            layer order (i.e. straight off the checkpoint config).

    Returns:
        ``(alloc_head_count, layer_kv_head_counts)`` -- the pre-TP head count
        the slab rows must be allocated at (``kv_alloc_head_count``), and the
        per-layer counts to pass as ``layer_kv_head_counts``. Both are
        expressed in the common head dim, which is ``min(head_dim)``.

    Raises:
        ValueError: when the geometries do not share a common head dim (some
            layer's per-token element count is not a multiple of the smallest
            head dim). Such a model cannot ride this path and must not be
            silently approximated.
    """
    if not layer_kv_geometry:
        raise ValueError("layer_kv_geometry is empty")
    dims = [int(d) for _, d in layer_kv_geometry]
    common_dim = min(dims)
    counts: list[int] = []
    for layer_id, (heads, dim) in enumerate(layer_kv_geometry):
        elems = int(heads) * int(dim)
        if elems % common_dim:
            raise ValueError(
                f"layer {layer_id} serves {heads} x {dim} = {elems} KV "
                f"elements per token, which is not a multiple of the common "
                f"head dim {common_dim} (the smallest per-layer head dim). "
                "This geometry cannot be normalized onto the heterogeneous "
                "KV path; it needs a genuinely per-layer head_dim in the "
                "pool, not a re-count."
            )
        counts.append(elems // common_dim)
    return max(counts), tuple(counts)


def move_kv_cache_native(
    k_buffer: list[torch.Tensor],
    v_buffer: list[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
        v_cache[tgt_loc_flat] = v_cache[src_loc_flat]


@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel. Safe for in-place copy."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    tl.multiple_of(byte_off, 16)

    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)

    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)
