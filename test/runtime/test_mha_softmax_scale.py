# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
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

"""The MHA backend must attend at the layer's own softmax scale.

Every model that reached this backend before Gemma 4 declares
``scaling == head_dim ** -0.5``, which is also what the kernels use when
``softmax_scale`` is omitted -- so the backend never passing it was invisible.
Gemma 4 declares ``scaling = 1.0`` (it has no ``query_pre_attn_scalar``), and
with the argument dropped its 256-wide sliding layers were attended at
``1/sqrt(256) = 0.0625``. E33 measured cosine **0.439** against HF for this
alone, on an otherwise correct model.

No GPU: the kernel entrypoints are replaced with recorders, so what is under
test is the plumbing, not the attention.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

NUM_Q_HEADS = 2
HEAD_DIM = 4
DEFAULT_SCALE = 1.0 / math.sqrt(HEAD_DIM)
# Gemma 4's value, and deliberately not the default for any head_dim.
GEMMA4_SCALE = 1.0

MODULE = "tokenspeed.runtime.layers.attention.backends.mha"


class _BackendHarness:
    """Helpers shared by the two cases below.

    Deliberately not a ``TestCase``: subclassing one would make every test in
    it run a second time under the subclass's name.
    """

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.mha import (
                MHAAttnBackend,
                MHADecodeMetadata,
                MHAExtendMetadata,
            )
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")

        self.torch = torch
        self.MHADecodeMetadata = MHADecodeMetadata
        self.MHAExtendMetadata = MHAExtendMetadata

        # Bypass __init__: these forwards read a handful of flags, and building
        # a real MHAConfig would drag in a model config for no added coverage.
        self.backend = MHAAttnBackend.__new__(MHAAttnBackend)
        self.backend.is_mxfp8 = False
        self.backend.is_fp8 = False
        self.backend.cast_query_to_fp8 = False
        self.backend.kernel_solution = "triton"
        self.backend.max_context_len = 64
        # Reads that have nothing to do with the scale.
        self.backend._get_kv_cache = lambda layer, pool: (None, None)
        self.backend._select_page_table = lambda layer, meta: meta.page_table

    def _layer(self, scaling):
        return SimpleNamespace(
            scaling=scaling,
            sliding_window_size=-1,
            logit_cap=0.0,
            tp_q_head_num=NUM_Q_HEADS,
            v_head_dim=HEAD_DIM,
            qk_head_dim=HEAD_DIM,
            layer_id=0,
            group_id="full_attention",
        )

    def _qkv(self, tokens):
        return [self.torch.zeros(tokens, NUM_Q_HEADS, HEAD_DIM) for _ in range(3)]

    def _extend_meta(self, tokens):
        t = self.torch
        return self.MHAExtendMetadata(
            page_table=t.zeros((1, 1), dtype=t.int32),
            seq_lens=t.tensor([tokens], dtype=t.int32),
            extend_seq_lens=t.tensor([tokens], dtype=t.int32),
            cu_extend_seq_lens=t.tensor([0, tokens], dtype=t.int32),
            cu_seqlens_kv=t.tensor([0, tokens], dtype=t.int32),
            extend_prefix_lens=t.tensor([0], dtype=t.int32),
            extend_seq_lens_cpu=[tokens],
            cu_extend_seq_lens_cpu=[0, tokens],
            max_extend_seq_len=tokens,
        )

    def _decode_meta(self, batch):
        t = self.torch
        return self.MHADecodeMetadata(
            page_table=t.zeros((batch, 1), dtype=t.int32),
            seq_lens=t.ones(batch, dtype=t.int32),
            page_tables=None,
        )

    def _recorder(self, tokens):
        """Stand-in kernel that captures its kwargs and returns a usable shape."""
        seen = {}

        def kernel(**kwargs):
            seen.update(kwargs)
            return self.torch.zeros(tokens, NUM_Q_HEADS, HEAD_DIM)

        return seen, kernel

    # -- the three kernel entrypoints ---------------------------------------

    def _run_prefill(self, scaling, tokens=3):
        seen, kernel = self._recorder(tokens)
        q, k, v = self._qkv(tokens)
        with mock.patch(f"{MODULE}.mha_prefill", kernel):
            self.backend._forward_prefill(
                q,
                k,
                v,
                self._layer(scaling),
                None,
                None,
                self._extend_meta(tokens),
                False,
                None,
            )
        return seen

    def _run_extend(self, scaling, tokens=3):
        seen, kernel = self._recorder(tokens)
        q, k, v = self._qkv(tokens)
        with mock.patch(f"{MODULE}.mha_extend_with_kvcache", kernel):
            self.backend._forward_extend(
                q,
                k,
                v,
                self._layer(scaling),
                None,
                None,
                self._extend_meta(tokens),
                False,
                None,
            )
        return seen

    def _run_decode(self, scaling, batch=2):
        seen, kernel = self._recorder(batch)
        q, k, v = self._qkv(batch)
        with mock.patch(f"{MODULE}.mha_decode_with_kvcache", kernel):
            self.backend._forward_decode(
                q,
                k,
                v,
                self._layer(scaling),
                None,
                None,
                self._decode_meta(batch),
                False,
                None,
            )
        return seen


class ScaleForwardingTest(_BackendHarness, unittest.TestCase):
    def test_prefill_forwards_the_layer_scale(self):
        self.assertEqual(self._run_prefill(GEMMA4_SCALE)["softmax_scale"], GEMMA4_SCALE)

    def test_extend_forwards_the_layer_scale(self):
        self.assertEqual(self._run_extend(GEMMA4_SCALE)["softmax_scale"], GEMMA4_SCALE)

    def test_decode_forwards_the_layer_scale(self):
        self.assertEqual(self._run_decode(GEMMA4_SCALE)["softmax_scale"], GEMMA4_SCALE)

    def test_scale_is_passed_explicitly_even_when_it_is_the_default(self):
        """Not `None`, and not omitted.

        Leaving it out happens to be numerically right for every pre-Gemma-4
        model, which is exactly why the omission survived. Pin the explicit
        pass so it cannot be "simplified" back to the kernel default.
        """
        for name, run in (
            ("prefill", self._run_prefill),
            ("extend", self._run_extend),
            ("decode", self._run_decode),
        ):
            with self.subTest(path=name):
                seen = run(DEFAULT_SCALE)
                self.assertIn("softmax_scale", seen)
                self.assertEqual(seen["softmax_scale"], DEFAULT_SCALE)


class Fa2GraphSplitScaleGuardTest(_BackendHarness, unittest.TestCase):
    """The FA2 cuda-graph decode split plans the scale and `run()` cannot
    override it, so a non-default scale must be refused rather than silently
    attended at the planned temperature."""

    def _run_decode_with_wrapper(self, scaling, batch=2):
        q, k, v = self._qkv(batch)
        meta = self._decode_meta(batch)
        meta.fa2_decode_wrapper = object()
        return self.backend._forward_decode(
            q, k, v, self._layer(scaling), None, None, meta, False, None
        )

    def test_non_default_scale_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "softmax_scale"):
            self._run_decode_with_wrapper(GEMMA4_SCALE)

    def test_default_scale_is_not_refused_by_the_scale_guard(self):
        # Reaches the wrapper run and fails there instead (the stand-in is not
        # a real wrapper) -- what matters is that it is not the scale guard.
        with self.assertRaises(Exception) as ctx:
            self._run_decode_with_wrapper(DEFAULT_SCALE)
        self.assertNotIn("softmax_scale", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
