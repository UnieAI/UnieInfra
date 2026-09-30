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

"""Tests for the DeepSeek V4 image-span prefill metadata builder."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from tokenspeed.runtime.models.deepseek_v4 import build_image_spans_from_context


def _ctx(seq_lens, prefix_lens, offsets_per_req, num_reqs):
    mm_inputs = []
    for offsets in offsets_per_req:
        items = []
        for lo, hi in offsets:
            items.append(
                SimpleNamespace(
                    offsets=[(lo, hi)],
                    encoded=object(),
                    hash=None,
                    modality="image",
                )
            )
        mm_inputs.append(SimpleNamespace(mm_items=items))
    return SimpleNamespace(
        extend_seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64),
        mm_inputs=mm_inputs,
    )


def test_single_request_image_span():
    ctx = _ctx([130], [0], [[(30, 49)]], 1)
    lo, hi = build_image_spans_from_context(ctx, 130, torch.device("cpu"))
    assert lo.shape == (130,) and hi.shape == (130,)
    assert torch.equal(lo[:30], torch.full((30,), -1, dtype=torch.int32))
    assert torch.equal(hi[:30], torch.full((30,), -1, dtype=torch.int32))
    assert torch.equal(lo[30:50], torch.full((20,), 30, dtype=torch.int32))
    assert torch.equal(hi[30:50], torch.full((20,), 49, dtype=torch.int32))
    assert torch.equal(lo[50:], torch.full((80,), -1, dtype=torch.int32))
    assert torch.equal(hi[50:], torch.full((80,), -1, dtype=torch.int32))


def test_multi_request_base_accumulation():
    # req 0: seq 60 with span [10, 20] -> flat indices [10, 20]
    # req 1: seq 70 with span [5, 15]  -> flat indices 60 + [5, 15] = [65, 75]
    ctx = _ctx([60, 70], [0, 0], [[(10, 20)], [(5, 15)]], 2)
    lo, hi = build_image_spans_from_context(ctx, 130, torch.device("cpu"))
    assert torch.equal(lo[10:21], torch.full((11,), 10, dtype=torch.int32))
    assert torch.equal(hi[10:21], torch.full((11,), 20, dtype=torch.int32))
    assert torch.equal(lo[65:76], torch.full((11,), 5, dtype=torch.int32))
    assert torch.equal(hi[65:76], torch.full((11,), 15, dtype=torch.int32))
    assert torch.equal(lo[21:65], torch.full((44,), -1, dtype=torch.int32))
    assert torch.equal(lo[76:], torch.full((54,), -1, dtype=torch.int32))


def test_prefix_offsets_shift_flat_indices():
    # prefix = 10, seq = 50; span request-absolute [16, 24] maps to flat
    # indices [base + (16 - 10), base + (24 - 10)] = [6, 14].
    ctx = _ctx([50], [10], [[(16, 24)]], 1)
    lo, hi = build_image_spans_from_context(ctx, 50, torch.device("cpu"))
    assert torch.equal(lo[6:15], torch.full((9,), 16, dtype=torch.int32))
    assert torch.equal(hi[6:15], torch.full((9,), 24, dtype=torch.int32))
    assert torch.equal(lo[:6], torch.full((6,), -1, dtype=torch.int32))
    assert torch.equal(hi[15:], torch.full((35,), -1, dtype=torch.int32))


def test_spans_clipped_to_num_tokens():
    # span [115, 135] in a 130-token slice: tokens beyond 129 must stay -1 and
    # the builder must not write out of bounds.
    ctx = _ctx([130], [0], [[(115, 135)]], 1)
    lo, hi = build_image_spans_from_context(ctx, 130, torch.device("cpu"))
    assert torch.equal(lo[115:130], torch.full((15,), 115, dtype=torch.int32))
    assert torch.equal(hi[115:130], torch.full((15,), 129, dtype=torch.int32))
    assert torch.equal(lo[:115], torch.full((115,), -1, dtype=torch.int32))


def test_decode_and_empty_requests_do_not_advance_or_set_spans():
    # A decode request (seq 0) and a text-only request must not move the base
    # or flag any token as an image span.
    ctx = _ctx([60, 0, 40], [0, 0, 0], [[(10, 20)], [], []], 3)
    lo, hi = build_image_spans_from_context(ctx, 100, torch.device("cpu"))
    assert torch.equal(lo[:10], torch.full((10,), -1, dtype=torch.int32))
    assert torch.equal(lo[10:21], torch.full((11,), 10, dtype=torch.int32))
    assert torch.equal(lo[21:100], torch.full((79,), -1, dtype=torch.int32))