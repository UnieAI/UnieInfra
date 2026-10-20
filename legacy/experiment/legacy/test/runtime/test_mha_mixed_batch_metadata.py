"""MHA mixed prefill+decode batch metadata.

A ``ForwardMode.MIXED`` plan holds prefill rows ``[0, num_extends)`` followed by
decode rows ``[num_extends, bs)``, but carries ``extend_seq_lens`` /
``extend_prefix_lens`` for the prefill rows ONLY -- the runtime slices
``extend_*_buf[:num_extends]``. The backend therefore builds extend metadata
over the prefill segment alone and a sibling decode metadata over the tail, and
``_forward_mixed`` dispatches each segment to its own kernel.

These tests pin the split, because getting it wrong is silent: a decode row that
reads a prefill row's page table, or a segment boundary off by one token, still
produces fluent output -- from the wrong KV.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.mha import (
    MHAAttnBackend,
    MHADecodeMetadata,
)


def _stub_backend(page_size=64, max_context_len=8192):
    """An MHAAttnBackend shell carrying only what metadata init reads."""
    b = MHAAttnBackend.__new__(MHAAttnBackend)
    b.page_size = page_size
    b.max_context_len = max_context_len
    b.max_num_pages = max_context_len // page_size
    b.spec_num_tokens = 1
    b.is_draft = False
    b.draft_block_decode = False
    b.supports_mixed_batch = True
    b.flat_state_group_ids = frozenset()
    b.flat_engine_owned_group_ids = frozenset()
    b.flat_group_page_sizes = {}
    b.forward_extend_metadata = None
    b.forward_decode_metadata = None
    return b


class MixedMetadataSplitTest(unittest.TestCase):
    """The single-table (radix) path: one page table sliced per segment."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prefill_extend = [3072, 777]
        self.prefill_prefix = [0, 1024]
        self.decode_seq = [1000, 2001, 65]
        self.ne = len(self.prefill_extend)
        self.bs = self.ne + len(self.decode_seq)

    def _init(self):
        b = _stub_backend()
        dev = self.device
        seq_lens = torch.tensor(
            [e + p for e, p in zip(self.prefill_extend, self.prefill_prefix)]
            + self.decode_seq,
            dtype=torch.int32,
            device=dev,
        )
        # req_to_page rows are distinct and non-contiguous so a wrong row index
        # cannot look right by accident.
        req_to_page = (
            torch.arange(
                1, (self.bs + 1) * b.max_num_pages + 1, dtype=torch.int32, device=dev
            )
            .view(self.bs + 1, b.max_num_pages)
            .flip(0)
        )
        rpi = torch.arange(self.bs, dtype=torch.int64, device=dev)
        b.init_forward_metadata(
            bs=self.bs,
            num_extends=self.ne,
            req_pool_indices=rpi,
            seq_lens=seq_lens,
            req_to_page=req_to_page,
            forward_mode=ForwardMode.MIXED,
            extend_seq_lens=torch.tensor(
                self.prefill_extend, dtype=torch.int32, device=dev
            ),
            extend_seq_lens_cpu=torch.tensor(self.prefill_extend, dtype=torch.int32),
            extend_prefix_lens=torch.tensor(
                self.prefill_prefix, dtype=torch.int32, device=dev
            ),
            extend_prefix_lens_cpu=torch.tensor(self.prefill_prefix, dtype=torch.int32),
        )
        return b, seq_lens, req_to_page, rpi

    def test_extend_metadata_covers_only_the_prefill_segment(self):
        b, seq_lens, _, _ = self._init()
        md = b.forward_extend_metadata
        self.assertEqual(md.extend_seq_lens.tolist(), self.prefill_extend)
        self.assertEqual(md.extend_prefix_lens.tolist(), self.prefill_prefix)
        self.assertEqual(md.extend_seq_lens_cpu, self.prefill_extend)
        # cu_seqlens must be [num_extends + 1] long, NOT [bs + 1]: the extend
        # kernel is given the prefill rows only.
        self.assertEqual(md.cu_extend_seq_lens.shape[0], self.ne + 1)
        self.assertEqual(md.cu_seqlens_kv.shape[0], self.ne + 1)
        self.assertEqual(md.seq_lens.tolist(), seq_lens[: self.ne].tolist())
        self.assertEqual(md.max_extend_seq_len, max(self.prefill_extend))
        self.assertEqual(md.page_table.shape[0], self.ne)

    def test_split_points_locate_the_decode_tail_in_the_packed_q(self):
        b, _, _, _ = self._init()
        md = b.forward_extend_metadata
        self.assertEqual(md.mixed_prefill_tokens, sum(self.prefill_extend))
        self.assertEqual(md.mixed_decode_rows, len(self.decode_seq))
        # One query token per decode row -- the boundary the forward slices at.
        self.assertEqual(
            md.mixed_total_tokens,
            sum(self.prefill_extend) + len(self.decode_seq),
        )
        self.assertEqual(md.cu_extend_seq_lens_cpu[-1], md.mixed_prefill_tokens)

    def test_decode_metadata_covers_only_the_decode_tail(self):
        b, seq_lens, req_to_page, rpi = self._init()
        dm = b.forward_decode_metadata
        self.assertIsInstance(dm, MHADecodeMetadata)
        self.assertEqual(dm.seq_lens.tolist(), self.decode_seq)
        self.assertEqual(dm.page_table.shape[0], len(self.decode_seq))
        # Each decode row must carry ITS OWN request's pages, not row 0's.
        for j in range(len(self.decode_seq)):
            self.assertEqual(
                dm.page_table[j].tolist(),
                req_to_page[rpi[self.ne + j], : b.max_num_pages].tolist(),
            )

    def test_pure_extend_is_untouched(self):
        # The EXTEND path must keep covering all bs rows and declare no split,
        # so nothing about this change can reach a non-mixed forward.
        b = _stub_backend()
        dev = self.device
        ext, pre = [512, 300], [0, 128]
        seq_lens = torch.tensor(
            [e + p for e, p in zip(ext, pre)], dtype=torch.int32, device=dev
        )
        req_to_page = torch.arange(
            1, 3 * b.max_num_pages + 1, dtype=torch.int32, device=dev
        ).view(3, b.max_num_pages)
        b.init_forward_metadata(
            bs=2,
            num_extends=2,
            req_pool_indices=torch.arange(2, dtype=torch.int64, device=dev),
            seq_lens=seq_lens,
            req_to_page=req_to_page,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor(ext, dtype=torch.int32, device=dev),
            extend_seq_lens_cpu=torch.tensor(ext, dtype=torch.int32),
            extend_prefix_lens=torch.tensor(pre, dtype=torch.int32, device=dev),
            extend_prefix_lens_cpu=torch.tensor(pre, dtype=torch.int32),
        )
        md = b.forward_extend_metadata
        self.assertEqual(md.mixed_decode_rows, 0)
        self.assertEqual(md.mixed_prefill_tokens, 0)
        self.assertEqual(md.cu_extend_seq_lens.shape[0], 3)
        self.assertEqual(md.extend_seq_lens.tolist(), ext)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class MixedFlatWriteLocsTest(unittest.TestCase):
    """Flat per-group KV write locations for a MIXED batch.

    The mixed forward writes KV for BOTH segments in one call, in packed q
    order: every prefill row's new tokens, then one token per decode row. A tail
    sized from the whole batch, or concatenated the other way round, writes each
    row's K/V into another request's pages.
    """

    def test_locs_are_prefill_tokens_then_one_per_decode_row(self):
        device = "cuda"
        page_size = 64
        backend = _stub_backend(page_size)
        backend.flat_group_page_sizes = {"full": page_size}

        prefill_extend, prefill_prefix = [130, 70], [0, 128]
        decode_seq = [200, 65]
        bs, ne, max_pages = 4, 2, 8
        table = (
            torch.arange(1, bs * max_pages + 1, dtype=torch.int32, device=device)
            .view(bs, max_pages)
            .flip(0)
        )
        seq_lens = torch.tensor(
            [e + p for e, p in zip(prefill_extend, prefill_prefix)] + decode_seq,
            dtype=torch.int32,
            device=device,
        )
        locs = backend._compute_flat_mixed_out_cache_locs(
            {"full": table},
            bs,
            ne,
            torch.tensor(prefill_prefix, dtype=torch.int32),
            torch.tensor(prefill_extend, dtype=torch.int32),
            seq_lens,
        )["full"]

        self.assertEqual(locs.shape[0], sum(prefill_extend) + len(decode_seq))

        def expect(row, pos):
            return int(table[row, pos // page_size]) * page_size + pos % page_size

        want = []
        for row, (start, num) in enumerate(zip(prefill_prefix, prefill_extend)):
            want += [expect(row, p) for p in range(start, start + num)]
        for j, s in enumerate(decode_seq):
            want.append(expect(ne + j, s - 1))
        self.assertEqual(locs.tolist(), want)


if __name__ == "__main__":
    unittest.main()
