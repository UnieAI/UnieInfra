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


"""GFX950 one-CTA-per-row radix top-k matches the CSA2 selection contracts.

Row selection must equal the exact top-k set of torch (position-sorted, the
lowest columns kept among equal boundary scores); block selection must equal
``index_block_select`` bit for bit. Covers the LDS-compacted bucket, the
row-rescan overflow path, boundary ties, candidates, the rank-sort ordering of
rows wider than the position bitmap, and JIT specialization.
"""

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsv41.triton import (  # noqa: E402
    index_block_select,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41.radix_topk import (  # noqa: E402
    gluon_dsv41_index_topk_radix_gfx950,
    launch_gluon_dsv41_index_block_radix_gfx950,
    launch_gluon_dsv41_index_topk_radix_gfx950,
)

DEVICE = "cuda"


def _reference_rows(logits, candidates, topk):
    """Exact top-k: larger scores first, equal scores by lower column."""
    width = logits.shape[1]
    take = min(topk, width)
    # A stable sort keeps equal scores in column order.
    order = torch.sort(-logits, dim=1, stable=True).indices[:, :take]
    finite = torch.gather(logits, 1, order) > -torch.inf
    if candidates is not None:
        logical = torch.gather(candidates.long(), 1, order // 8) * 8 + order % 8
    else:
        logical = order
    big = torch.iinfo(torch.int64).max
    ids = torch.where(finite, logical, big).sort(dim=1).values
    ids = ids.masked_fill(ids == big, -1).to(torch.int32)
    rows = torch.full((logits.shape[0], topk), -1, dtype=torch.int32, device=DEVICE)
    rows[:, :take] = ids
    return rows, finite.sum(dim=1).to(torch.int32)


def _radix_rows(logits, candidates, topk, live_widths=None):
    rows = torch.empty(logits.shape[0], topk, dtype=torch.int32, device=DEVICE)
    lens = torch.empty(logits.shape[0], dtype=torch.int32, device=DEVICE)
    launch_gluon_dsv41_index_topk_radix_gfx950(
        logits, candidates, live_widths, topk, rows, lens
    )
    return rows, lens


def _check_rows(logits, candidates, topk, live_widths=None):
    rows, lens = _radix_rows(logits, candidates, topk, live_widths)
    ref_rows, ref_lens = _reference_rows(logits, candidates, topk)
    assert torch.equal(lens, ref_lens)
    assert torch.equal(rows, ref_rows)


def _live_logits(queries, width, generator, live, scale=1.0, offset=0.0):
    logits = (
        torch.randn(queries, width, device=DEVICE, generator=generator) * scale + offset
    )
    columns = torch.arange(width, device=DEVICE)
    return logits.masked_fill(columns[None] >= live[:, None], -torch.inf)


@pytest.mark.parametrize("width", [8192, 65536, 131072 + 4096])
@pytest.mark.parametrize("topk", [512, 2048])
def test_rows_match_torch(width, topk):
    # 131072 + 4096 columns exceed the position bitmap and use the rank sort.
    generator = torch.Generator(device=DEVICE).manual_seed(width + topk)
    queries = 12
    live = torch.randint(
        width // 2, width + 1, (queries,), device=DEVICE, generator=generator
    )
    live[0], live[1], live[2], live[3] = width, 0, 300, topk
    logits = _live_logits(queries, width, generator, live)
    # A recency-skewed row: the largest scores cluster at the end.
    logits[4, -2048:] += 4.0
    _check_rows(logits, None, topk, live.to(torch.int32))
    _check_rows(logits, None, topk)


def test_narrow_scores_overflow_the_bucket():
    # Scores of one narrow band put many keys in the K-th key's high bucket:
    # rows with more than 8192 there take the row-rescan path.
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    queries, width = 6, 1 << 20
    live = torch.full((queries,), width, device=DEVICE)
    live[1] = 200_000
    logits = _live_logits(queries, width, generator, live, scale=0.05, offset=20.0)
    _check_rows(logits, None, 512, live.to(torch.int32))


def test_boundary_ties_keep_lowest_columns():
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    queries, width = 8, 65536
    logits = torch.randn(queries, width, device=DEVICE, generator=generator)
    # Few hot columns share the K-th score, but fewer than the bucket capacity.
    hot = torch.rand(queries, width, device=DEVICE, generator=generator) < 0.02
    logits = torch.where(hot, torch.full_like(logits, 7.0), logits)
    # Every finite score is equal (overflow path), or of a few rounded values.
    logits[1] = 1.0
    logits[2] = (torch.randn(width, device=DEVICE, generator=generator) * 2).round()
    logits[3, ::3] = -torch.inf
    logits[3, 1::3] = 0.5
    _check_rows(logits, None, 512)


@pytest.mark.parametrize("ordered", [True, False])
def test_candidates_map_to_logical_ids(ordered):
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    queries, width = 6, 16384
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
    logits[1, 4096:] = -torch.inf
    _check_rows(logits, blocks.to(torch.int32), 512)
    _check_rows(logits, blocks.to(torch.int64), 512)


def test_unaligned_rows_match_torch():
    # Odd strides and widths take the scalar-load variant.
    generator = torch.Generator(device=DEVICE).manual_seed(6)
    base = torch.randn(5, 20001, device=DEVICE, generator=generator)
    _check_rows(base[:, 1:], None, 512)
    _check_rows(base[:, :19999], None, 300)


def _blocks(fn, logits, visible, k, cap):
    out = torch.full((logits.shape[0], cap), 7, dtype=torch.int32, device=DEVICE)
    lens = torch.full((logits.shape[0],), 7, dtype=torch.int32, device=DEVICE)
    fn(logits, visible, k, out, lens)
    return out, lens


@pytest.mark.parametrize("width", [65536, 1 << 20])
def test_blocks_match_index_block_select(width):
    generator = torch.Generator(device=DEVICE).manual_seed(width)
    queries = 10
    visible = torch.randint(
        width // 4, width + 1, (queries,), device=DEVICE, generator=generator
    )
    # Empty, partial-block, at-capacity and fewer-blocks-than-k rows.
    visible[0], visible[1], visible[2], visible[3] = 0, 13, width, 2048 * 8 - 5
    logits = torch.randn(queries, width, device=DEVICE, generator=generator)
    # Columns past the visible length need not be -inf: they must be ignored.
    logits[5, 100:] = -torch.inf
    logits[6] = 2.0  # every block ties: lowest blocks win, newest forced in
    logits[7] = (logits[7] * 2).round()
    for k, cap in ((2048, 2048), (1000, 1024)):
        got = _blocks(
            launch_gluon_dsv41_index_block_radix_gfx950, logits, visible, k, cap
        )
        ref = _blocks(index_block_select, logits, visible, k, cap)
        assert torch.equal(got[1], ref[1])
        assert torch.equal(got[0], ref[0])
    # Int64 visible lengths select the same blocks.
    got = _blocks(
        launch_gluon_dsv41_index_block_radix_gfx950, logits, visible.long(), 2048, 2048
    )
    ref = _blocks(index_block_select, logits, visible, 2048, 2048)
    assert torch.equal(got[0], ref[0])


def test_no_recompile_across_batch_shapes():
    generator = torch.Generator(device=DEVICE).manual_seed(9)

    def run(queries, width, live_n):
        live = torch.full((queries,), live_n, device=DEVICE, dtype=torch.int32)
        logits = _live_logits(queries, width, generator, live)
        _radix_rows(logits, None, 512, live)
        _blocks(launch_gluon_dsv41_index_block_radix_gfx950, logits, live, 2048, 2048)

    run(16, 65536, 40000)
    run(7, 65536 + 64, 40001)
    with assert_no_triton_compile(gluon_dsv41_index_topk_radix_gfx950):
        run(33, 131072, 99999)
        run(5, 32768 + 128, 3)
        run(64, 1 << 18, 1 << 17)
