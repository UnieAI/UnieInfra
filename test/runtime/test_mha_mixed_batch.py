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

"""A MHA leaf serves a MIXED round as two kernels over one packed batch.

The router packs the extending requests first and the decoding requests after
them, one token each, and the prologue has already written every row's KV. The
leaf then runs its extend kernel over the prefill span and its decode kernel
over the trailing rows -- the same split MLA does in the model, done in the leaf
because only the leaf knows its kernels.

The kernel calls are the only thing faked here: they need a real GPU, while the
metadata, the row accounting and the slicing this test pins are ordinary host
arithmetic.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged import mha as mha_module
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend

HEAD_DIM = 4
Q_HEADS = 2
MAX_NUM_PAGES = 3


class _Layer:
    """The handful of fields the leaf reads off a ``PagedAttention`` layer."""

    def __init__(self) -> None:
        self.layer_id = 0
        self.qk_head_dim = HEAD_DIM
        self.v_head_dim = HEAD_DIM
        self.tp_q_head_num = Q_HEADS
        self.tp_k_head_num = Q_HEADS
        self.tp_v_head_num = Q_HEADS
        self.sliding_window_size = -1
        self.logit_cap = 0.0
        self.scaling = 1.0


class _Pool:
    """A KV pool holding one page per layer, all zeros."""

    def __init__(self) -> None:
        self.page_size = 4

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return torch.zeros(self.page_size, Q_HEADS, HEAD_DIM, dtype=torch.float32)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return torch.zeros(self.page_size, Q_HEADS, HEAD_DIM, dtype=torch.float32)


class _KernelLog:
    """Records the kernel calls the split makes, and what each one was given."""

    def __init__(self) -> None:
        self.extend: list[dict] = []
        self.prefill: list[dict] = []
        self.decode: list[dict] = []

    def extend_kernel(
        self, *, q, cu_seqlens_q, cu_seqlens_kv, page_table, cache_seqlens, **kwargs
    ):
        self.extend.append(
            dict(
                rows=q.shape[0],
                cu_seqlens_q=cu_seqlens_q.tolist(),
                cu_seqlens_kv=cu_seqlens_kv.tolist(),
                page_table=page_table.tolist(),
                cache_seqlens=cache_seqlens.tolist(),
            )
        )
        return torch.ones_like(q)

    def prefill_kernel(self, *, q, cu_seqlens, cu_seqlens_cpu, **kwargs):
        self.prefill.append(dict(rows=q.shape[0], cu_seqlens=cu_seqlens_cpu))
        return torch.ones_like(q)

    def decode_kernel(self, *, q, page_table, cache_seqlens, max_seqlen_q, **kwargs):
        self.decode.append(
            dict(
                rows=q.shape[0],
                page_table=page_table.tolist(),
                cache_seqlens=cache_seqlens.tolist(),
                max_seqlen_q=max_seqlen_q,
            )
        )
        return torch.full_like(q, 2.0)


@pytest.fixture
def log(monkeypatch):
    log = _KernelLog()
    monkeypatch.setattr(mha_module, "mha_extend_with_kvcache", log.extend_kernel)
    monkeypatch.setattr(mha_module, "mha_prefill", log.prefill_kernel)
    monkeypatch.setattr(mha_module, "mha_decode_with_kvcache", log.decode_kernel)
    return log


def _backend(
    *,
    spec_num_tokens: int = 1,
    draft_block_decode: bool = False,
    is_draft: bool = False,
):
    """A CPU ``MHAAttnBackend`` with its persistent decode buffers allocated.

    The real constructor wants a model config and a device, neither of which
    any of the arithmetic under test reads.
    """
    backend = MHAAttnBackend.__new__(MHAAttnBackend)
    backend.device = "cpu"
    backend.is_draft = is_draft
    backend.spec_num_tokens = spec_num_tokens
    backend.draft_block_decode = draft_block_decode
    backend.kernel_page_size = 4
    backend.max_num_pages = MAX_NUM_PAGES
    backend.max_context_len = 512
    backend.head_dim = HEAD_DIM
    backend.qkv_dtype = torch.float32
    backend.kv_cache_dtype = torch.float32
    backend.is_fp8 = False
    backend.is_mxfp8 = False
    backend.kernel_solution = "flashinfer"
    backend.skip_softmax_threshold = None
    backend._decode_views_by_bs = {}
    backend.forward_decode_metadata = None
    backend.forward_extend_metadata = None
    backend.forward_num_extends = 0
    backend.tree_prefix_lens_buf = None
    backend.init_cuda_graph_state(4)
    # The kernel plan picks the extend arm; a cached prefix (max prefix > 0)
    # always runs the cache-reading extend kernel, which is the arm a MIXED
    # round with history takes.
    backend.plan = lambda **kwargs: {"extend_mode": "postwrite"}
    return backend


def _mixed_round(
    backend,
    *,
    num_extends: int,
    decode_seqlens: list[int],
    extend_lens: list[int],
    prefixes: list[int],
):
    """Init the leaf for a MIXED round and return the tables it was handed."""
    bs = num_extends + len(decode_seqlens)
    seq_lens = torch.tensor(
        [p + n for p, n in zip(prefixes, extend_lens)] + decode_seqlens,
        dtype=torch.int32,
    )
    page_table = torch.arange(bs, dtype=torch.int32)[:, None].repeat(1, MAX_NUM_PAGES)
    backend.init_forward_metadata(
        bs,
        num_extends,
        seq_lens,
        page_table,
        ForwardMode.MIXED,
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor(extend_lens, dtype=torch.int32),
        extend_prefix_lens=torch.tensor(prefixes, dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor(prefixes, dtype=torch.int32),
        extend_with_prefix=True,
        query_shard=None,
        page_table_cpu=None,
    )
    return seq_lens, page_table


def _q(rows: int) -> torch.Tensor:
    return torch.zeros(rows, Q_HEADS, HEAD_DIM, dtype=torch.float32)


def test_extend_metadata_covers_only_the_prefill_span():
    """The extend half must not read a decode request's rows.

    Its page table, its cache lengths and its token span are all over the
    leading requests; a decode row's history would otherwise be attended as if
    it were part of the prompt.
    """
    backend = _backend()

    _mixed_round(
        backend,
        num_extends=2,
        decode_seqlens=[11],
        extend_lens=[4, 3],
        prefixes=[100, 200],
    )
    md = backend.forward_extend_metadata

    assert backend.forward_num_extends == 2
    assert md.cu_extend_seq_lens_cpu == [0, 4, 7]
    assert md.page_table.tolist() == [[0, 0, 0], [1, 1, 1]]
    assert md.seq_lens.tolist() == [104, 203]
    assert md.max_extend_prefix_len == 200


def test_the_same_round_publishes_a_decode_window_past_the_extend_rows():
    backend = _backend()

    seq_lens, page_table = _mixed_round(
        backend,
        num_extends=2,
        decode_seqlens=[11],
        extend_lens=[4, 3],
        prefixes=[100, 200],
    )
    md = backend.forward_decode_metadata

    assert md.num_extends == 2
    assert md.seq_lens[:3].tolist() == seq_lens.tolist()
    assert md.page_table[2].tolist() == page_table[2].tolist()


def test_forward_splits_the_packed_batch_between_two_kernels(log):
    """Prefill rows to the extend kernel, decode rows to the decode kernel."""
    backend = _backend()
    seq_lens, _ = _mixed_round(
        backend,
        num_extends=2,
        decode_seqlens=[11],
        extend_lens=[4, 3],
        prefixes=[100, 200],
    )

    out = backend.forward_extend(
        _q(8),
        _q(8),
        _q(8),
        _Layer(),
        torch.zeros(8, dtype=torch.int64),
        _Pool(),
        3,
        forward_mode=ForwardMode.MIXED,
    )

    assert [call["rows"] for call in log.extend] == [7]
    assert [call["rows"] for call in log.decode] == [1]
    assert log.prefill == []

    # The extend kernel sees the prefill requests' own cache geometry.
    assert log.extend[0]["cu_seqlens_q"] == [0, 4, 7]
    assert log.extend[0]["cu_seqlens_kv"] == [0, 104, 307]
    assert log.extend[0]["cache_seqlens"] == [104, 203]
    # The decode kernel sees one row and one request, starting after the
    # prefill rows rather than at row 0.
    assert log.decode[0]["cache_seqlens"] == [seq_lens[2].item()]
    assert log.decode[0]["page_table"] == [[2, 2, 2]]
    assert log.decode[0]["max_seqlen_q"] == 1

    # Each half lands in its own rows of the packed output.
    assert torch.all(out[:7] == 1.0)
    assert torch.all(out[7:] == 2.0)


def test_graph_padding_past_the_real_rows_does_not_move_the_split(log):
    """A prefill-graph replay pads the activations, not the batch.

    Row counts come from the live batch, so padding rows after the decode rows
    are left alone instead of being mistaken for more decode tokens.
    """
    backend = _backend()
    _mixed_round(
        backend,
        num_extends=1,
        decode_seqlens=[11, 12],
        extend_lens=[5],
        prefixes=[100],
    )

    backend.forward_extend(
        _q(5 + 2 + 4),
        _q(5 + 2 + 4),
        _q(5 + 2 + 4),
        _Layer(),
        torch.zeros(5 + 2 + 4, dtype=torch.int64),
        _Pool(),
        3,
        forward_mode=ForwardMode.MIXED,
    )

    assert [call["rows"] for call in log.extend] == [5]
    assert [call["rows"] for call in log.decode] == [2]
    assert log.decode[0]["cache_seqlens"] == [11, 12]


def test_a_pure_extend_round_keeps_its_one_kernel_path(log):
    """The split is a MIXED-round property, invisible to an extend round."""
    backend = _backend()
    page_table = torch.zeros(3, MAX_NUM_PAGES, dtype=torch.int32)
    backend.init_forward_metadata(
        3,
        3,
        torch.tensor([10, 20, 30], dtype=torch.int32),
        page_table,
        ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor([4, 3, 2], dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor([4, 3, 2], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([6, 17, 28], dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor([6, 17, 28], dtype=torch.int32),
        extend_with_prefix=True,
        query_shard=None,
        page_table_cpu=None,
    )

    backend.forward_extend(
        _q(9),
        _q(9),
        _q(9),
        _Layer(),
        torch.zeros(9, dtype=torch.int64),
        _Pool(),
        3,
        forward_mode=ForwardMode.EXTEND,
    )

    assert backend.forward_num_extends == 0
    assert [call["rows"] for call in log.extend] == [9]
    assert log.decode == []


def test_a_decode_refresh_clears_the_mixed_split(log):
    """The next pure decode round reads every row again.

    The split is stamped onto the cached per-batch decode view, so a refresh
    that forgets to clear it would keep skipping this round's leading rows.
    """
    backend = _backend()
    seq_lens, page_table = _mixed_round(
        backend,
        num_extends=2,
        decode_seqlens=[11],
        extend_lens=[4, 3],
        prefixes=[100, 200],
    )
    backend.refresh_decode_metadata(3, 3, seq_lens, page_table)

    out = backend.forward_decode(
        _q(3),
        None,
        None,
        _Layer(),
        torch.zeros(3, dtype=torch.int64),
        _Pool(),
        3,
    )

    assert backend.forward_decode_metadata.num_extends == 0
    assert log.decode[0]["rows"] == 3
    assert log.decode[0]["cache_seqlens"] == seq_lens.tolist()
    assert torch.all(out == 2.0)


def test_block_decode_refuses_a_mixed_round():
    """A DFLASH-style block draft replicates a request's pages per position.

    It has no leading-span offset, so a MIXED round must fail loudly here
    rather than read the first request's pages for the last one.
    """
    backend = _backend(spec_num_tokens=2, draft_block_decode=True)
    seq_lens = torch.tensor([10, 20], dtype=torch.int32)
    page_table = torch.zeros(2, MAX_NUM_PAGES, dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="MIXED round"):
        backend.refresh_decode_metadata(2, 2, seq_lens, page_table, num_extends=1)


def test_a_drafter_window_after_an_extend_round_reads_the_whole_batch(log):
    """An MTP drafter's decode steps after a prefill read one row per request.

    The executor refreshes the draft window with the round's extend count. A
    target uses it to skip a MIXED round's prompt rows; a drafter must not, or
    a round of only new requests leaves it an empty window (the step divides
    its rows by zero requests).
    """
    backend = _backend(is_draft=True)
    seq_lens = torch.tensor([104, 203], dtype=torch.int32)
    page_table = torch.zeros(2, MAX_NUM_PAGES, dtype=torch.int32)
    backend.refresh_decode_metadata(2, 2, seq_lens, page_table, num_extends=2)

    backend.forward_decode(
        _q(2),
        None,
        None,
        _Layer(),
        torch.zeros(2, dtype=torch.int64),
        _Pool(),
        2,
    )

    assert backend.forward_decode_metadata.num_extends == 0
    assert log.decode[0]["rows"] == 2
    assert log.decode[0]["cache_seqlens"] == seq_lens.tolist()


@pytest.mark.parametrize(
    "mode", [ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.DECODE]
)
def test_only_a_drafter_narrows_outside_decode_rounds(mode):
    """A drafter has a decode window in every round; a target only in decode."""
    assert _backend(is_draft=True).supports_narrowed_draft_decode(mode)
    assert _backend().supports_narrowed_draft_decode(mode) == mode.is_decode()
