# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Verification for the sm_120 FlashInfer paged MHA decode kernel.

Runs standalone (``python3 tests/test_paged_mha_decode_cuda_graph.py``) or under
pytest. Needs an sm_120 device and the plugin installed.

The geometry is the live one: 24 query heads over 4 KV heads, head_dim 256,
page_size 832. Round numbers hide the two failures this kernel exists to work
around (``Unsupported group_size: 6``, and the trtllm page-size whitelist).

The negative control in :func:`test_stale_plan_is_wrong` is the important one.
A stale FlashInfer schedule does not fault and does not produce NaNs — it
produces plausible output. Without that check, a capture/replay test would pass
on a build where the re-plan does nothing.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention import mha_decode_with_kvcache
from tokenspeed_kernel.plugins import discover_plugins
from unieinfra_kernel_sm120 import flashinfer_mha as fim

discover_plugins()

DTYPE = torch.bfloat16
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 24, 4, 256
PAGE_SIZE, MAX_PAGES, NUM_PAGES = 832, 6, 256
MAX_SEQ = PAGE_SIZE * MAX_PAGES
SOLUTION = "flashinfer_paged"
# bf16 through a different reduction order; cosine is the bar, per the porting
# skill. Max relative error on near-zero elements is uninformative here.
COS_BAR = 0.9999


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _caches(device: torch.device, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    shape = (NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k = torch.randn(shape, generator=g, dtype=torch.float32).to(device, DTYPE)
    v = torch.randn(shape, generator=g, dtype=torch.float32).to(device, DTYPE)
    return k, v


def _batch(device: torch.device, batch: int, max_seqlen_q: int, seed: int, min_len=64):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(
        (batch * max_seqlen_q, NUM_Q_HEADS, HEAD_DIM), generator=g, dtype=torch.float32
    ).to(device, DTYPE)
    page_table = torch.randint(
        0, NUM_PAGES, (batch, MAX_PAGES), generator=g, dtype=torch.int32
    ).to(device)
    seq_lens = (
        torch.randint(min_len, MAX_SEQ, (batch,), generator=g, dtype=torch.int32)
        .clamp_min(max_seqlen_q)
        .to(device)
    )
    return q, page_table, seq_lens


def _decode(solution, q, k, v, page_table, seq_lens, max_seqlen_q):
    return mha_decode_with_kvcache(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table=page_table,
        cache_seqlens=seq_lens,
        max_seqlen_k=MAX_SEQ,
        max_seqlen_q=max_seqlen_q,
        solution=solution,
    )


def _prepare(page_table, seq_lens, max_seqlen_q):
    fim.prepare_paged_mha_decode(
        page_table=page_table,
        cache_seqlens=seq_lens,
        page_size=PAGE_SIZE,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        q_dtype=DTYPE,
        kv_dtype=DTYPE,
        max_seqlen_q=max_seqlen_q,
    )


def _capture(q, k, v, page_table, seq_lens, max_seqlen_q):
    """Warm up on a side stream, plan outside, then capture only ``run()``."""
    _prepare(page_table, seq_lens, max_seqlen_q)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _decode(SOLUTION, q, k, v, page_table, seq_lens, max_seqlen_q)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    _prepare(page_table, seq_lens, max_seqlen_q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _decode(SOLUTION, q, k, v, page_table, seq_lens, max_seqlen_q)
    return graph, out


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0),
    reason="requires an sm_120 device",
)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0")


@pytest.mark.parametrize("batch", [1, 8, 33])
@pytest.mark.parametrize("max_seqlen_q", [1, 4])
def test_matches_triton_eager(device, batch, max_seqlen_q):
    """Agreement with the in-tree kernel this displaces, through the dispatch."""
    k, v = _caches(device)
    q, page_table, seq_lens = _batch(
        device, batch, max_seqlen_q, batch * 10 + max_seqlen_q
    )
    ref = _decode("triton", q, k, v, page_table, seq_lens, max_seqlen_q)
    got = _decode(SOLUTION, q, k, v, page_table, seq_lens, max_seqlen_q)
    assert _cosine(ref, got) > COS_BAR


@pytest.mark.parametrize("batch", [1, 8, 33])
@pytest.mark.parametrize("max_seqlen_q", [1, 4])
def test_matches_triton_with_hoisted_plan(device, batch, max_seqlen_q):
    """Same, with the schedule built by the out-of-graph hook rather than inline."""
    k, v = _caches(device)
    q, page_table, seq_lens = _batch(device, batch, max_seqlen_q, 900 + batch)
    ref = _decode("triton", q, k, v, page_table, seq_lens, max_seqlen_q)
    _prepare(page_table, seq_lens, max_seqlen_q)
    got = _decode(SOLUTION, q, k, v, page_table, seq_lens, max_seqlen_q)
    assert _cosine(ref, got) > COS_BAR


@pytest.mark.parametrize("batch,max_seqlen_q", [(8, 1), (8, 4), (33, 1)])
def test_capture_and_replay(device, batch, max_seqlen_q):
    """Capture, then replay against page tables and lengths the capture never saw."""
    k, v = _caches(device)
    q, page_table, seq_lens = _batch(device, batch, max_seqlen_q, 5000 + batch)
    graph, out = _capture(q, k, v, page_table, seq_lens, max_seqlen_q)
    try:
        for rep in range(3):
            fresh_q, fresh_pt, fresh_len = _batch(
                device, batch, max_seqlen_q, 7000 + rep * 31 + batch, min_len=97
            )
            q.copy_(fresh_q)
            page_table.copy_(fresh_pt)
            seq_lens.copy_(fresh_len)

            ref = _decode("triton", q, k, v, page_table, seq_lens, max_seqlen_q)
            _prepare(page_table, seq_lens, max_seqlen_q)
            graph.replay()
            torch.cuda.synchronize()
            assert _cosine(ref, out) > COS_BAR, f"replay {rep}"
    finally:
        del graph, out


def test_stale_plan_is_wrong(device):
    """Negative control: skipping the re-plan must give a wrong answer.

    If this ever starts passing with a high cosine, the hoisted plan has stopped
    carrying the step and every other test here is vacuous.
    """
    k, v = _caches(device)
    batch, max_seqlen_q = 16, 1
    q, page_table, seq_lens = _batch(device, batch, max_seqlen_q, 1234)
    seq_lens.fill_(300)  # one page
    graph, out = _capture(q, k, v, page_table, seq_lens, max_seqlen_q)
    try:
        for new_len in (4900, 900, 4160):  # 6, 2, 5 pages
            seq_lens.fill_(new_len)
            ref = _decode("triton", q, k, v, page_table, seq_lens, max_seqlen_q)

            graph.replay()  # plan still describes the previous length
            torch.cuda.synchronize()
            stale = _cosine(ref, out)

            _prepare(page_table, seq_lens, max_seqlen_q)
            graph.replay()
            torch.cuda.synchronize()
            fresh = _cosine(ref, out)

            assert fresh > COS_BAR, f"re-planned replay wrong at {new_len}"
            assert stale < 0.99, (
                f"stale replay agreed at {new_len} (cos={stale}); the hoisted "
                "plan is no longer load-bearing and these tests prove nothing"
            )
    finally:
        del graph, out


def test_cuda_graph_padding_rows_stay_finite(device):
    """Padded bucket rows carry seq_len 0 or 1 and must not fault or poison."""
    k, v = _caches(device)
    batch, max_seqlen_q = 16, 1
    q, page_table, seq_lens = _batch(device, batch, max_seqlen_q, 4321)
    seq_lens.fill_(2000)
    graph, out = _capture(q, k, v, page_table, seq_lens, max_seqlen_q)
    try:
        seq_lens[batch // 2 :] = 0
        seq_lens[batch // 2] = 1
        _prepare(page_table, seq_lens, max_seqlen_q)
        graph.replay()
        torch.cuda.synchronize()

        ref = _decode("triton", q, k, v, page_table, seq_lens, max_seqlen_q)
        live = slice(0, batch // 2)
        assert torch.isfinite(out).all()
        assert _cosine(ref[live], out[live]) > COS_BAR
    finally:
        del graph, out


def test_capture_without_hook_raises_directed_error(device):
    """A missing hook must name the hook, not surface a raw CUDA capture fault."""
    k, v = _caches(device)
    q, page_table, seq_lens = _batch(device, 16, 1, 555)
    fim._plans.clear()
    fim._epoch = 0
    graph = torch.cuda.CUDAGraph()
    try:
        with pytest.raises(RuntimeError, match="prepare_paged_mha_decode"):
            with torch.cuda.graph(graph):
                _decode(SOLUTION, q, k, v, page_table, seq_lens, 1)
    finally:
        del graph
        torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
