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

"""Every output row of the prefill convolution must be defined.

A continuous-batch layout can carry padded request slots whose state page is
``pad_slot_id``. Those slots own no cache line and must not update any state,
but their token rows still sit inside ``query_start_loc`` -- a mixed batch's
recurrent scan reads every row up to the last boundary -- so a program that
returns early without storing leaves the caller reading the allocation's
leftovers. For a bf16 activation that leftover is frequently NaN, and one NaN
row poisons a whole GDN chunk and then the persisted conv window: a silent
correctness failure that no throughput number would show.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed_kernel.ops.conv import PAD_SLOT_ID, causal_conv1d_fn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="prefill convolution runs on CUDA."
)

WIDTH = 4
DIM = 512
DEVICE = "cuda"
DTYPE = torch.bfloat16


def _reference(x, weight, window):
    """Causal FIR: out[t] = sum_w weight[w] * [window ++ x][t + w]."""
    padded = torch.cat([window, x], dim=1).float()
    return sum(
        padded[:, w : w + x.shape[1]] * weight[:, w].float()[:, None]
        for w in range(weight.shape[1])
    ).to(x.dtype)


def _poison_allocator(shape, dtype):
    """Hand the next same-sized allocation a block full of NaN.

    Without this the test compares the padded rows against whatever the caching
    allocator happened to leave there, which is often already zero on a fresh
    process -- the defect would slip through.
    """
    junk = torch.full(shape, float("nan"), dtype=dtype, device=DEVICE)
    torch.cuda.synchronize()
    del junk


def _run(lengths, cache_indices, has_initial_state, block_m, dtype=DTYPE):
    torch.manual_seed(0)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE
    )
    total = int(query_start_loc[-1])
    x = (torch.randn(DIM, total, dtype=dtype, device=DEVICE) * 4.0).to(dtype)
    weight = torch.randn(DIM, WIDTH, dtype=dtype, device=DEVICE)
    cache = torch.randn(8, DIM, WIDTH - 1, dtype=dtype, device=DEVICE)
    original = cache.clone()
    indices = torch.tensor(cache_indices, dtype=torch.int32, device=DEVICE)
    initial = torch.tensor(has_initial_state, dtype=torch.bool, device=DEVICE)
    metadata = build_causal_conv1d_prefill_metadata(
        query_start_loc, torch.tensor(lengths, dtype=torch.int32), block_m
    )
    _poison_allocator(x.shape, dtype)
    out = causal_conv1d_fn(
        x,
        weight,
        None,
        cache,
        query_start_loc,
        indices,
        initial,
        activation=None,
        pad_slot_id=PAD_SLOT_ID,
        prefill_metadata=metadata,
    )
    return x, weight, original, cache, query_start_loc, out, metadata


def test_padded_slot_rows_are_defined_and_zero():
    """A pad-slot row reads as zeros, and its cache line is left untouched."""
    lengths = [5, 1, 1]
    indices = [3, PAD_SLOT_ID, 5]
    x, _, original, cache, query_start_loc, out, metadata = _run(
        lengths, indices, [True, False, False], block_m=4
    )

    assert out.shape == x.shape
    assert torch.isfinite(out).all(), "a padded row reached the caller as a NaN"

    pad_rows = slice(int(query_start_loc[1]), int(query_start_loc[2]))
    assert (
        out[:, pad_rows].abs().max().item() == 0.0
    ), "a padded slot must report zeros, not the allocation's leftovers"
    # Two real chunks of four, plus one chunk for each single-token row.
    assert metadata.batch_indices.numel() == 2 + 1 + 1
    # The pad slot is never a write target.
    assert torch.equal(cache[PAD_SLOT_ID], original[PAD_SLOT_ID])


@pytest.mark.parametrize("block_m", [1, 4, 8])
def test_padded_slots_anywhere_in_the_batch(block_m):
    """Pad slots first, middle and last all stay defined and zero."""
    lengths = [3, 1, 11, 1]
    indices = [PAD_SLOT_ID, 2, PAD_SLOT_ID, 7]
    _, _, _, _, query_start_loc, out, _ = _run(
        lengths, indices, [False, True, False, True], block_m=block_m
    )
    for seq, slot in enumerate(indices):
        if slot != PAD_SLOT_ID:
            continue
        rows = slice(int(query_start_loc[seq]), int(query_start_loc[seq + 1]))
        assert out[:, rows].abs().max().item() == 0.0
    assert torch.isfinite(out).all()


def test_real_rows_match_the_fir_reference():
    """Defining the padded rows must not perturb the real ones."""
    lengths = [6, 3, 5]
    indices = [1, PAD_SLOT_ID, 4]
    initial_state = [True, False, False]
    x, weight, original, _, query_start_loc, out, _ = _run(
        lengths, indices, initial_state, block_m=4, dtype=torch.float32
    )

    for seq in range(len(lengths)):
        if indices[seq] == PAD_SLOT_ID:
            continue
        rows = slice(int(query_start_loc[seq]), int(query_start_loc[seq + 1]))
        window = (
            original[indices[seq]]
            if initial_state[seq]
            else torch.zeros(DIM, WIDTH - 1, dtype=torch.float32, device=DEVICE)
        )
        expected = _reference(x[:, rows], weight, window)
        torch.testing.assert_close(
            out[:, rows].float(), expected.float(), atol=1e-3, rtol=1e-3
        )
