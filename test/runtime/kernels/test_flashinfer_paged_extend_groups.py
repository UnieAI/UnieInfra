# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""One cache group's pages must never answer another group's attention.

``flashinfer_paged_mha_extend_with_kvcache`` is planned per page table. The
flat KV path hands out ONE TABLE PER CACHE GROUP, and gemma-4-31B publishes six
of them (five ``sliding_attention_<k>`` sub-groups plus ``full_attention``),
every one of which covers the same request with the same page COUNT over
different pages.

The plan cache used to be keyed on the batch's shape -- page counts, last-page
fills, query offsets, head geometry, scale -- and on a single shared wrapper.
Every one of those fields is equal across the five sliding groups, so the first
group planned the wrapper and the other four ran on ITS page list: gemma-4-31B
answered fluently, at length, about a different subject, and nothing raised.
That is what these tests pin.

Both entry paths are covered, because they failed for different reasons:

* the per-call path (no plan installed), which is where the shape-keyed cache
  lived;
* the installed-plan path, which gemma reached only to be rejected by it --
  the plan recorded ``1 / sqrt(head_dim)`` while Gemma 4 declares a softmax
  scale of 1.0 -- and so fell through to the per-call path on every layer.
"""

from __future__ import annotations

import os
import sys
import unittest

# The kernel registers itself at import time, behind this flag; set it before
# anything drags tokenspeed_kernel in.
os.environ.setdefault("TOKENSPEED_ENABLE_FI_PAGED_EXTEND", "1")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

# gemma-4-31B's sliding layers, per rank at TP1: 32 q heads over 16 KV heads of
# 256, paged 16 tokens to a page, sliding_window 1024, and a softmax scale of
# 1.0 (Gemma 4 has no query_pre_attn_scalar).
NUM_Q_HEADS = 32
NUM_KV_HEADS = 16
HEAD_DIM = 256
PAGE_SIZE = 16
WINDOW_LEFT = 1024
SOFTMAX_SCALE = 1.0
SEQ_LEN = 512
NUM_GROUPS = 5
# bf16 attention over bf16 pages: the two kernels agree to one ulp of the
# output magnitude, while a crossed page list is O(1) wrong (measured 2.8).
TOL = 0.05


class FlashInferPagedExtendGroupsTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"needs torch: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        try:
            from tokenspeed_kernel.ops.attention.flashinfer import (
                clear_paged_extend_plan,
                install_paged_extend_plan,
                paged_extend_planning_enabled,
            )
            from tokenspeed_kernel.ops.attention.triton.mha_prefill import (
                triton_mha_extend_with_kvcache,
            )
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"needs tokenspeed_kernel: {exc}")
        if not paged_extend_planning_enabled():
            self.skipTest("flashinfer paged extend is not registered on this build")
        from tokenspeed_kernel.ops.attention.flashinfer import (
            flashinfer_paged_mha_extend_with_kvcache,
        )

        self.torch = torch
        self.fi = flashinfer_paged_mha_extend_with_kvcache
        self.triton = triton_mha_extend_with_kvcache
        self.install = install_paged_extend_plan
        self.clear = clear_paged_extend_plan

        torch.manual_seed(0)
        dev, dt = "cuda", torch.bfloat16
        pages_per_group = SEQ_LEN // PAGE_SIZE
        pool_pages = NUM_GROUPS * pages_per_group
        # Disjoint page ranges, one per group -- the flat pool hands each group
        # ids from one free list, so no id is ever shared.
        self.tables = [
            torch.arange(
                g * pages_per_group,
                (g + 1) * pages_per_group,
                dtype=torch.int32,
                device=dev,
            ).view(1, -1)
            for g in range(NUM_GROUPS)
        ]
        self.k_cache = (
            torch.randn(
                pool_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=dev, dtype=dt
            )
            * 0.5
        )
        self.v_cache = (
            torch.randn(
                pool_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=dev, dtype=dt
            )
            * 0.5
        )
        self.q = torch.randn(SEQ_LEN, NUM_Q_HEADS, HEAD_DIM, device=dev, dtype=dt) * 0.5
        self.cu_q = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=dev)
        self.cu_kv = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=dev)
        self.seq_lens = torch.tensor([SEQ_LEN], dtype=torch.int32, device=dev)
        self.addCleanup(self.clear)

    def _call(self, fn, table):
        return fn(
            q=self.q,
            cu_seqlens_q=self.cu_q,
            cu_seqlens_kv=self.cu_kv,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            page_table=table,
            cache_seqlens=self.seq_lens,
            max_seqlen_q=SEQ_LEN,
            max_seqlen_k=SEQ_LEN,
            is_causal=True,
            window_left=WINDOW_LEFT,
            logit_cap=0.0,
            softmax_scale=SOFTMAX_SCALE,
        )

    def _reference(self):
        """Per-group answers from the Triton kernel, which takes the table
        directly and holds no plan."""
        return [self._call(self.triton, t) for t in self.tables]

    def _assert_matches_own_group(self, out, group, reference):
        """`out` is group `group`'s answer, and nobody else's."""
        own = (out.float() - reference[group].float()).abs().max().item()
        self.assertLess(
            own,
            TOL,
            f"group {group} does not match its own reference (max abs {own:.4f})",
        )
        for other in range(NUM_GROUPS):
            if other == group:
                continue
            cross = (out.float() - reference[other].float()).abs().max().item()
            self.assertGreater(
                cross,
                TOL,
                f"group {group}'s output is indistinguishable from group "
                f"{other}'s (max abs {cross:.4f}) -- the plan crossed page tables",
            )

    def test_per_call_path_replans_for_each_page_table(self):
        """No installed plan: every call must describe its own page table."""
        reference = self._reference()
        self.clear()
        for rep in range(2):
            for group, table in enumerate(self.tables):
                with self.subTest(rep=rep, group=group):
                    self._assert_matches_own_group(
                        self._call(self.fi, table), group, reference
                    )

    def test_installed_plans_do_not_share_a_wrapper(self):
        """One installed entry per table, exercised the way a 60-layer forward
        interleaves its groups. The scale here is Gemma 4's 1.0, i.e. not the
        kernel's ``1 / sqrt(head_dim)`` default, so the entry has to be planned
        from what the LAYER hands over rather than from anything the installer
        could have guessed."""
        reference = self._reference()
        self.clear()
        for table in self.tables:
            self.install(
                cu_seqlens_q=self.cu_q,
                cache_seqlens=self.seq_lens,
                seq_lens_cpu=[SEQ_LEN],
                page_table=table,
            )
        for rep in range(3):
            for group, table in enumerate(self.tables):
                with self.subTest(rep=rep, group=group):
                    self._assert_matches_own_group(
                        self._call(self.fi, table), group, reference
                    )

    def test_installed_plan_is_not_clobbered_by_an_unplanned_call(self):
        """A call with no entry of its own must not borrow an installed
        wrapper: planning one overwrites the plan it held, and the next layer
        of that group would then attend over the borrower's pages."""
        reference = self._reference()
        self.clear()
        planned, unplanned = self.tables[0], self.tables[1]
        self.install(
            cu_seqlens_q=self.cu_q,
            cache_seqlens=self.seq_lens,
            seq_lens_cpu=[SEQ_LEN],
            page_table=planned,
        )
        self._assert_matches_own_group(self._call(self.fi, planned), 0, reference)
        # `unplanned` shares neither an entry nor a shape distinction with it.
        self._assert_matches_own_group(self._call(self.fi, unplanned), 1, reference)
        # ... and group 0 still answers for group 0.
        self._assert_matches_own_group(self._call(self.fi, planned), 0, reference)


if __name__ == "__main__":
    unittest.main()
