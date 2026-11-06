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

"""GFX950 DeepSeek V4.1 row-block indexer logits (Full mode).

``dsv41_index_logits_gfx950`` without candidates scores six query rows per CTA
and shares key loads when their page tables match. It must match the per-row
scorer bit-for-bit in which logits it writes and to FP32 rounding in their
values, whether rows share tables (speculative decode, prefill) or not.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsv41 import (
    torch_dsv41_cache_quantize,
)
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41._gluon import indexer as gluon_indexer
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41 import (  # noqa: E402
    indexer as module,
)

DEVICE = "cuda"
SCORE_CHUNK = 1024


def _random_inputs(requests, next_n, keys, seed, page_pad=0):
    """Random MXFP4 queries and cache; a request's rows share its table."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    pages = (keys + 63) // 64
    num_pages = requests * pages + 3
    storage = torch.randint(
        0,
        256,
        (num_pages, 64 * 68 + page_pad),
        dtype=torch.uint8,
        device=DEVICE,
        generator=g,
    )
    cache = storage[:, : 64 * 68]
    # E8M0 scales near 1 keep scores finite and well conditioned.
    cache[:, 64 * 64 :] = torch.randint(
        118, 128, (num_pages, 64 * 4), dtype=torch.uint8, device=DEVICE, generator=g
    )
    order = torch.randperm(num_pages, device=DEVICE, generator=g)
    table = order[: requests * pages].view(requests, pages).int()
    table = table.repeat_interleave(next_n, 0).contiguous()
    rows = requests * next_n
    # Speculative rows see one more key each, the last row the whole history.
    visible = torch.tensor(
        [keys - (next_n - 1 - i) for _ in range(requests) for i in range(next_n)],
        dtype=torch.int32,
        device=DEVICE,
    )
    values = torch.randint(
        0, 256, (rows, 32, 64), dtype=torch.uint8, device=DEVICE, generator=g
    )
    scales = torch.randint(
        118, 128, (rows, 32, 4), dtype=torch.uint8, device=DEVICE, generator=g
    )
    scales = scales.view(torch.int32).reshape(rows, 32)
    weights = torch.randn(rows, 32, device=DEVICE, generator=g)
    return values, scales, weights, cache, table, visible, pages * 64


def _both(values, scales, weights, cache, table, visible, width, candidates=None):
    new = torch.full((values.shape[0], width), -torch.inf, device=DEVICE)
    old = torch.full_like(new, -torch.inf)
    module.dsv41_index_logits_gfx950(
        values, scales, weights, cache, table, visible, candidates, new, SCORE_CHUNK
    )
    module._launch_per_row(
        values, scales, weights, cache, table, visible, candidates, old, SCORE_CHUNK
    )
    return new, old


def _random_candidates(rows, next_n, keys, blocks, shared, seed):
    """Ascending candidate block ids per row, as the block selector emits."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    total = keys // 8
    picks = rows // next_n if shared else rows
    ids = torch.stack(
        [
            torch.randperm(total, device=DEVICE, generator=g)[:blocks]
            for _ in range(picks)
        ]
    )
    if shared:
        ids = ids.repeat_interleave(next_n, 0)
    return ids.sort(1).values.int().contiguous()


def _assert_matches_per_row(new, old):
    assert torch.equal(torch.isfinite(new), torch.isfinite(old))
    assert torch.isfinite(old).any()
    # Same products, different FP32 summation order over the 32 heads.
    torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize(
    ("requests", "next_n", "keys"),
    [
        (16, 6, 8192),  # DSpark decode, short history
        (64, 6, 65536),  # DSpark decode, long history
        (16, 6, 65536 - 37),  # partial last page
        (1, 512, 32768),  # prefill chunk: one request, causal rows
        (64, 1, 16384),  # plain decode: no two rows share a table
        (24, 4, 4096),  # blocks straddle requests
    ],
)
def test_rows_match_per_row_scorer(requests, next_n, keys):
    *inputs, width = _random_inputs(requests, next_n, keys, seed=requests + keys)
    _assert_matches_per_row(*_both(*inputs, width))


@pytest.mark.parametrize(
    ("requests", "next_n", "keys", "blocks", "shared"),
    [
        (16, 6, 131072, 2048, False),  # DSpark Reindex decode at 130K
        (64, 6, 131072, 2048, False),
        (16, 6, 131072, 2048, True),  # rows of a request with one list
        (1, 512, 32768, 2048, False),  # prefill chunk
        (8, 6, 4096, 300, False),  # partial last tile
    ],
)
def test_candidates_match_per_row_scorer(requests, next_n, keys, blocks, shared):
    *inputs, _ = _random_inputs(requests, next_n, keys, seed=blocks + requests)
    candidates = _random_candidates(
        requests * next_n, next_n, keys, blocks, shared, seed=keys
    )
    _assert_matches_per_row(*_both(*inputs, blocks * 8, candidates))


def test_candidates_mask_like_per_row_scorer():
    values, scales, weights, cache, table, visible, _ = _random_inputs(
        5, 6, 4096, seed=3, page_pad=48
    )
    table = table.clone()
    visible = visible.clone()
    candidates = _random_candidates(30, 6, 4096, 96, False, seed=4)
    # Padding, blocks past the visible history, a missing page, unsorted and
    # repeated ids, an empty row, and a logits width narrower than the list.
    candidates[0, 50:] = -1
    candidates[1, 10] = 4096 // 8 + 7
    candidates[2] = candidates[2].flip(0)
    candidates[3, 5:9] = candidates[3, 4]
    table[6:12, 2] = -1
    visible[7] = 0
    visible[8] = 100
    new, old = _both(
        values, scales, weights, cache, table, visible, 96 * 8 - 40, candidates
    )
    _assert_matches_per_row(new, old)
    assert torch.isneginf(new[7]).all()
    assert torch.isneginf(new[0, 50 * 8 :]).all()


def test_rows_masks_like_per_row_scorer():
    values, scales, weights, cache, table, visible, width = _random_inputs(
        7, 6, 2048, seed=5, page_pad=48
    )
    table = table.clone()
    visible = visible.clone()
    # Missing and out-of-pool pages, an empty row, a row past table capacity,
    # and a row count that leaves a partial last block.
    table[0:6, 3] = -1
    table[6:12, 5] = cache.shape[0] + 9
    table[13, 1] = -1  # one row diverges from its request's table
    visible[20] = 0
    visible[25] = 10**6
    rows = 40
    new, old = _both(
        values[:rows],
        scales[:rows],
        weights[:rows],
        cache,
        table[:rows],
        visible[:rows],
        width,
    )
    _assert_matches_per_row(new, old)
    assert torch.isneginf(new[20]).all()
    assert torch.isneginf(new[0:6, 3 * 64 : 4 * 64]).all()


def test_rows_match_dequantized_reference():
    g = torch.Generator(device=DEVICE).manual_seed(41)
    keys = torch.randn(512, 128, device=DEVICE, dtype=torch.bfloat16, generator=g)
    keys *= 0.1
    _, key_reference = torch_dsv41_cache_quantize(keys, "index")
    cache = torch.zeros(8, 64, 68, device=DEVICE, dtype=torch.uint8)
    dsv41.cache_scatter(keys, cache, torch.arange(512, device=DEVICE), "index")
    q = torch.randn(12, 32, 128, device=DEVICE, dtype=torch.bfloat16, generator=g)
    q *= 0.1
    weights = torch.randn(12, 32, device=DEVICE, generator=g)
    # Two requests of six speculative rows over pages in shuffled order.
    table = torch.tensor(
        [[3, 1, 0, 2]] * 6 + [[7, 5, 4, 6]] * 6, device=DEVICE, dtype=torch.int32
    )
    visible = torch.tensor(
        [188 + i for i in range(6)] + [250 + i for i in range(6)],
        device=DEVICE,
        dtype=torch.int32,
    )
    out = torch.full((12, 256), -torch.inf, device=DEVICE)
    values, scales, packed_weights = gluon_indexer._pack_index_q(q, weights)
    module.dsv41_index_logits_gfx950(
        values,
        scales,
        packed_weights,
        cache.flatten(1),
        table,
        visible,
        None,
        out,
        SCORE_CHUNK,
    )

    _, q_reference = torch_dsv41_cache_quantize(q.reshape(-1, 128), "index")
    q_reference = q_reference.reshape(12, 32, 128)
    expected = torch.full((12, 256), -torch.inf)
    for row in range(12):
        logical = torch.arange(int(visible[row]))
        slots = table[row].cpu()[logical // 64].long() * 64 + logical % 64
        dots = q_reference[row] @ key_reference[slots].float().T
        expected[row, logical] = (dots.relu() * weights[row].cpu()[:, None]).sum(0)
    torch.testing.assert_close(out.cpu(), expected, atol=2e-4, rtol=2e-4)


def test_rows_batch_shapes_do_not_recompile():
    *inputs, _ = _random_inputs(40, 6, 16384, seed=9)
    values, scales, weights, cache, table, visible = inputs

    def run(rows, keys, table_width):
        logits = torch.full((rows, keys), -torch.inf, device=DEVICE)
        module.dsv41_index_logits_gfx950(
            values[:rows],
            scales[:rows],
            weights[:rows],
            cache,
            table[:rows, :table_width].contiguous(),
            visible[:rows].clamp(max=keys),
            None,
            logits,
            SCORE_CHUNK,
        )

    def run_candidates(rows, blocks):
        candidates = _random_candidates(rows, 1, 16384, blocks, False, seed=rows)
        logits = torch.full((rows, blocks * 8), -torch.inf, device=DEVICE)
        module.dsv41_index_logits_gfx950(
            values[:rows],
            scales[:rows],
            weights[:rows],
            cache,
            table[:rows],
            visible[:rows],
            candidates,
            logits,
            SCORE_CHUNK,
        )

    run(96, 8192, 128)
    run_candidates(96, 256)
    with assert_no_triton_compile(
        module.gluon_dsv41_index_topk_rows_gfx950,
        module.gluon_dsv41_index_topk_candidates_gfx950,
    ):
        for rows, keys, table_width in [
            (1, 64, 1),
            (17, 4096, 65),
            (240, 16384, 256),
            (6, 1000, 16),
            (37, 12345, 200),
        ]:
            run(rows, keys, table_width)
        for rows, blocks in [(1, 1), (17, 2048), (240, 100), (37, 33)]:
            run_candidates(rows, blocks)


def test_rows_launch_metadata():
    args = {
        "q": torch.empty((12, 32, 64), dtype=torch.uint8),
        "logits": torch.empty((12, 512), dtype=torch.float32),
        "BLOCK_M": 6,
    }

    class Kernel:
        name = "index_rows"

    metadata = module._rows_launch_metadata((2, 3, 1), Kernel(), args)
    assert metadata["name"] == "index_rows"
    assert metadata["flops4"] == 2 * 12 * 32 * 512 * 128
    # Keys are read once per six-row block, not once per row.
    assert metadata["bytes"] == 2 * 512 * 68 + 12 * 32 * 64 * 3 + 12 * 512 * 4
    assert (
        module.gluon_dsv41_index_topk_rows_gfx950.launch_metadata
        is module._rows_launch_metadata
    )


def test_candidates_launch_metadata():
    args = {
        "q": torch.empty((12, 32, 64), dtype=torch.uint8),
        "logits": torch.empty((12, 512), dtype=torch.float32),
    }

    class Kernel:
        name = "index_candidates"

    metadata = module._cand_launch_metadata((12, 2, 1), Kernel(), args)
    assert metadata["flops4"] == 2 * 12 * 32 * 512 * 128
    # Each column gathers its key row plus a block id and a page id.
    assert metadata["bytes"] == 12 * 512 * 76 + 12 * 32 * 64 * 2 + 12 * 512 * 4
    assert (
        module.gluon_dsv41_index_topk_candidates_gfx950.launch_metadata
        is module._cand_launch_metadata
    )
