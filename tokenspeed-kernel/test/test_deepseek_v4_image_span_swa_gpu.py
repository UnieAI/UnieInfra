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

"""GPU tests for the image-span bidirectional SWA prefill index combiners.

Covers both the top-k CSA/SWA combine kernel and the dense-compressed/SWA
combine kernel. Inside an image span the SWA window is widened to the reference
``get_window_topk_idxs_visible`` window (``window_size - 1`` to the left,
``max_image_tokens`` to the right), while every non-span token keeps the exact
causal ``[pos - window_size + 1, pos]`` slice. A span wider than the window
exercises the ``max_image_tokens`` cap so the patch cannot silently fall back
to "attend to the whole span" (which would be wrong *and* could touch KV
positions past the request).
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton import deepseek_v4 as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA GPU"
)

DEVICE = "cuda:0"
WINDOW = 64
TOPK = 8
MAX_IMAGE = 384
ALIGN = ops.DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
# A narrow span that fits inside the widened window (span [30, 49]).
NARROW_LO, NARROW_HI = 30, 49
# A span wider than the window, so the max_image_tokens cap must bound the
# right extent (span [30, 500]; a 520-token request).
WIDE_LO, WIDE_HI, WIDE_SEQ = 30, 500, 520


def _span_arrays(num_tokens: int, lo: int, hi: int):
    span_lo = torch.full((num_tokens,), -1, dtype=torch.int32, device=DEVICE)
    span_hi = torch.full((num_tokens,), -1, dtype=torch.int32, device=DEVICE)
    span_lo[lo : hi + 1] = lo
    span_hi[lo : hi + 1] = hi
    return span_lo, span_hi


def _expected_swa(pos: int, lo: int, hi: int, window: int = WINDOW) -> tuple[int, int, list[int]]:
    """Reference window (topk_len, cur_len, swa_source_positions) for a token."""
    topk_len = min(pos + 1, TOPK)
    if lo <= pos <= hi:
        left = min(pos - lo, MAX_IMAGE - 1)
        right = min(hi - pos, MAX_IMAGE)
        start = max(max(pos - window + 1, pos - left), 0)
        cur_len = pos + right - start + 1
        return topk_len, cur_len, list(range(start, start + cur_len))
    cur_len = min(pos + 1, window)
    start = pos - cur_len + 1
    return topk_len, cur_len, list(range(start, start + cur_len))


def _assert_token_row(
    row: torch.Tensor, topk_len: int, cur_len: int, swa: list[int]
):
    total_len = topk_len + cur_len
    assert row[topk_len : topk_len + cur_len].cpu().tolist() == swa
    assert bool((row[total_len:] == -1).all().item())
    assert row[:topk_len].shape == (topk_len,)


def _topk_combined_width(seq_len: int):
    # topk + swa_capacity, aligned up to the sparse prefill block.
    n = WINDOW + MAX_IMAGE
    swa_capacity = 1 << (n - 1).bit_length()
    return (TOPK + swa_capacity + ALIGN - 1) // ALIGN * ALIGN


def _dense_combined_width(seq_len: int):
    n = max(WINDOW + MAX_IMAGE, 1)
    swa_capacity = 1 << (n - 1).bit_length()
    return (max(swa_capacity, 1) + ALIGN - 1) // ALIGN * ALIGN


def test_combine_topk_swa_image_span_bidirectional_narrow():
    dev = DEVICE
    seq_len = 130
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    gather_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=dev)
    topk_indices = (torch.arange(seq_len * TOPK, device=dev) % 1000).reshape(
        seq_len, TOPK
    ).to(torch.int32).contiguous()
    span_lo, span_hi = _span_arrays(seq_len, NARROW_LO, NARROW_HI)

    combined, lens = ops.deepseek_v4_combine_topk_swa_indices(
        topk_indices=topk_indices,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        window_size=WINDOW,
        compress_ratio=1,
        topk=TOPK,
        workspace_width=seq_len,
        compressed_base=0,
        span_lo=span_lo,
        span_hi=span_hi,
        max_image_tokens=MAX_IMAGE,
    )

    assert combined.shape[1] == _topk_combined_width(seq_len)
    for pos in range(seq_len):
        topk_len, cur_len, swa = _expected_swa(pos, NARROW_LO, NARROW_HI)
        row = combined[pos]
        assert row[:topk_len].cpu().tolist() == topk_indices[pos, :topk_len].cpu().tolist()
        _assert_token_row(row, topk_len, cur_len, swa)
        assert int(lens[pos]) == topk_len + cur_len

    _, _, swa_mid = _expected_swa(40, NARROW_LO, NARROW_HI)
    assert swa_mid == list(range(NARROW_LO, NARROW_HI + 1))
    _, _, swa_edge = _expected_swa(NARROW_HI, NARROW_LO, NARROW_HI)
    assert swa_edge == list(range(NARROW_LO, NARROW_HI + 1))


def test_combine_topk_swa_image_span_wider_than_window_is_capped():
    dev = DEVICE
    seq_len = WIDE_SEQ
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    gather_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=dev)
    topk_indices = torch.zeros(seq_len, TOPK, dtype=torch.int32, device=dev)
    span_lo, span_hi = _span_arrays(seq_len, WIDE_LO, WIDE_HI)

    combined, lens = ops.deepseek_v4_combine_topk_swa_indices(
        topk_indices=topk_indices,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        window_size=WINDOW,
        compress_ratio=1,
        topk=TOPK,
        workspace_width=seq_len,
        compressed_base=0,
        span_lo=span_lo,
        span_hi=span_hi,
        max_image_tokens=MAX_IMAGE,
    )

    # Token 40 is near the span start: right extent is capped at
    # max_image_tokens (384), so the window ends at 40+384 == 424, NOT at the
    # far span edge (500). Left extent is capped at window_size-1 (63) so it
    # starts at max(40-63, 40-10) == 30.
    topk_len, cur_len, swa = _expected_swa(40, WIDE_LO, WIDE_HI)
    assert swa == list(range(30, 425))
    _assert_token_row(combined[40], topk_len, cur_len, swa)
    assert int(lens[40]) == topk_len + cur_len


def test_combine_topk_swa_without_span_keeps_causal():
    # Regression guard: with no span arrays the SWA window stays causal.
    dev = DEVICE
    seq_len = 130
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    gather_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=dev)
    topk_indices = torch.zeros(seq_len, TOPK, dtype=torch.int32, device=dev)
    combined, lens = ops.deepseek_v4_combine_topk_swa_indices(
        topk_indices=topk_indices,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        window_size=WINDOW,
        compress_ratio=1,
        topk=TOPK,
        workspace_width=seq_len,
        compressed_base=0,
    )
    pos = 80
    topk_len, cur_len, swa = _expected_swa(pos, -1, -1)
    _assert_token_row(combined[pos], topk_len, cur_len, swa)
    assert int(lens[pos]) == topk_len + cur_len
    assert swa == list(range(pos - WINDOW + 1, pos + 1))


def test_combine_dense_swa_image_span_bidirectional():
    dev = DEVICE
    seq_len = 130
    positions = torch.arange(seq_len, dtype=torch.int32, device=dev)
    token_to_req_indices = torch.zeros(seq_len, dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    compressed_lens = torch.zeros(1, dtype=torch.int32, device=dev)
    gather_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    span_lo, span_hi = _span_arrays(seq_len, NARROW_LO, NARROW_HI)

    combined, lens = ops.deepseek_v4_combine_dense_swa_indices(
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        seq_lens=seq_lens,
        compressed_lens=compressed_lens,
        gather_lens=gather_lens,
        window_size=WINDOW,
        compress_ratio=1,
        workspace_width=seq_len,
        compressed_base=0,
        span_lo=span_lo,
        span_hi=span_hi,
        max_image_tokens=MAX_IMAGE,
    )

    assert combined.shape[1] == _dense_combined_width(seq_len)
    for pos in range(seq_len):
        _, cur_len, swa = _expected_swa(pos, NARROW_LO, NARROW_HI)
        row = combined[pos]
        # compress_ratio == 1 -> no compressed region, SWA starts at column 0.
        assert row[:cur_len].cpu().tolist() == swa
        assert bool((row[cur_len:] == -1).all().item())
        assert int(lens[pos]) == cur_len
