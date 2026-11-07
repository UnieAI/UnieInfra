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


"""GFX950 CSA2 row selection at decode widths matches torch top-k.

Covers the LDS-compacted path (pass 1 histograms the high key digit, pass 2
compacts that bucket and every key above it), its boundary-tie and candidate
handling, the column-bitmap ordering limit, and the full-scan fallback taken
when the compacted set would overflow LDS.
"""

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsv41._gluon.indexer import (  # noqa: E402
    select_rows_torch,
)
from tokenspeed_kernel_amd._device import num_cus  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41.indexer import (  # noqa: E402
    launch_gluon_dsv41_index_topk_select_gfx950 as launch_select,
)

DEVICE = "cuda"


def _select(fn, logits, candidates, topk, live_widths=None):
    rows = torch.empty(logits.shape[0], topk, dtype=torch.int32, device=DEVICE)
    lens = torch.empty(logits.shape[0], dtype=torch.int32, device=DEVICE)
    fn(logits, candidates, live_widths, topk, rows, lens)
    return rows, lens


def _check(logits, candidates, topk, live_widths=None):
    rows, lens = _select(launch_select, logits, candidates, topk, live_widths)
    ref_rows, ref_lens = _select(select_rows_torch, logits, candidates, topk)
    assert torch.equal(lens, ref_lens)
    assert torch.equal(rows, ref_rows)


def _live_logits(queries, width, generator, live):
    logits = torch.randn(queries, width, device=DEVICE, generator=generator)
    columns = torch.arange(width, device=DEVICE)
    return logits.masked_fill(columns[None] >= live[:, None], -torch.inf)


@pytest.mark.parametrize("width", [16384, 65536, 131072 + 512])
def test_decode_widths_match_torch(width):
    # 131072 + 512 columns exceed the column bitmap and use the rank sort.
    generator = torch.Generator(device=DEVICE).manual_seed(width)
    queries = 24
    live = torch.randint(
        width // 2, width + 1, (queries,), device=DEVICE, generator=generator
    )
    live[0], live[1], live[2] = width, 0, 300
    logits = _live_logits(queries, width, generator, live)
    # A recency-skewed row: the largest scores cluster at the end.
    logits[3, -2048:] += 4.0
    _check(logits, None, 512, live.to(torch.int32))
    _check(logits, None, 512)


def test_wide_boundary_ties_keep_lowest_columns():
    # Few distinct high scores: the K-th score is shared by many columns, but
    # the compacted set still fits in LDS.
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    queries, width = 8, 65536
    logits = torch.randn(queries, width, device=DEVICE, generator=generator)
    hot = torch.rand(queries, width, device=DEVICE, generator=generator) < 0.02
    logits = torch.where(hot, torch.full_like(logits, 7.0), logits)
    _check(logits, None, 512)


def test_crowded_bucket_falls_back_to_row_scans():
    # Every finite score is equal: the picked bucket holds the whole row.
    generator = torch.Generator(device=DEVICE).manual_seed(4)
    queries, width = 4, 65536
    logits = torch.ones(queries, width, device=DEVICE)
    logits[1] = (torch.randn(width, device=DEVICE, generator=generator) * 2).round()
    _check(logits, None, 512)


@pytest.mark.parametrize("ordered", [True, False])
def test_wide_candidates_map_to_logical_ids(ordered):
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    queries, width = 6, 32768
    blocks = torch.stack(
        [
            torch.randperm(1 << 20, device=DEVICE, generator=generator)[: width // 8]
            for _ in range(queries)
        ]
    )
    if ordered:
        blocks = blocks.sort(dim=1).values
    live = torch.full((queries,), width, device=DEVICE)
    logits = _live_logits(queries, width, generator, live)
    _check(logits, blocks.to(torch.int32), 512)


def test_unaligned_rows_match_torch():
    # Odd strides and widths take the scalar-load variant.
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    base = torch.randn(5, 20001, device=DEVICE, generator=generator)
    _check(base[:, 1:], None, 512)
    _check(base[:, :19999], None, 300)


def test_many_rows_match_torch():
    # More rows than two per CU select the 8-wave CTA variant.
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    queries, width = 2 * num_cus() + 5, 8192
    live = torch.randint(0, width + 1, (queries,), device=DEVICE, generator=generator)
    logits = _live_logits(queries, width, generator, live)
    _check(logits, None, 512, live.to(torch.int32))
