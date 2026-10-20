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

"""The prefill graph gathers embeddings through the model's own seam.

The graph runs the embedding lookup OUTSIDE the captured region, so anything
the model does to the lookup afterwards has to happen there too. Gemma scales
by ``sqrt(hidden_size)``; qwen2/qwen3 scale by ``config.scale_emb`` when the
checkpoint declares one. Gathering straight from ``embed_tokens`` skipped that
and changed nothing observable about shape or dtype: gemma-4-31B replayed the
graph and emitted the same token for every prompt, while eager prefill was
correct.

``get_input_embeddings`` is overloaded in this tree -- the HuggingFace form
takes no arguments and returns the module -- so the seam has to be chosen by
signature, which is what these tests pin.
"""

from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

SCALE = 73.32  # gemma-4-31B's sqrt(hidden_size), near enough for a test


def _import():
    from tokenspeed.runtime.execution.prefill_graph import (
        _resolve_input_embeds_fn,
        _takes_input_ids,
    )

    return _resolve_input_embeds_fn, _takes_input_ids


class ResolveInputEmbedsFnTest(unittest.TestCase):
    def setUp(self):
        try:
            self.resolve, self.takes_ids = _import()
            import torch
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch

    def _embed_tokens(self):
        """Stand-in for the raw lookup: one row per id, value == id."""

        def lookup(input_ids):
            return input_ids.unsqueeze(-1).to(self.torch.float32).repeat(1, 4)

        return lookup

    def _ids(self):
        return self.torch.tensor([1, 2, 3])

    def test_compute_form_seam_is_used(self):
        """A Gemma-shaped model: the seam scales, and the seam must win."""
        embed = self._embed_tokens()

        class Model:
            embed_tokens = staticmethod(embed)

            @staticmethod
            def get_input_embeddings(input_ids):
                return embed(input_ids) * SCALE

        fn = self.resolve(Model(), embed)
        out = fn(self._ids())
        raw = embed(self._ids())
        self.assertTrue(self.torch.allclose(out, raw * SCALE))
        # The bug this pins: the raw lookup is not the answer.
        self.assertFalse(self.torch.allclose(out, raw))

    def test_huggingface_form_seam_is_refused(self):
        """``get_input_embeddings()`` returning the module must not be called
        with ids -- it would hand the graph an nn.Module, not embeddings."""
        embed = self._embed_tokens()

        class Model:
            embed_tokens = staticmethod(embed)

            @staticmethod
            def get_input_embeddings():
                return embed

        fn = self.resolve(Model(), embed)
        self.assertIs(fn, embed)

    def test_no_seam_falls_back_to_embed_tokens(self):
        embed = self._embed_tokens()

        class Model:
            embed_tokens = staticmethod(embed)

        fn = self.resolve(Model(), embed)
        self.assertIs(fn, embed)

    def test_unscaled_seam_is_equivalent_to_the_lookup(self):
        """No regression for families whose seam just forwards (qwen3 without
        ``scale_emb``): resolving to the seam must not change the values."""
        embed = self._embed_tokens()

        class Model:
            embed_tokens = staticmethod(embed)

            @staticmethod
            def get_input_embeddings(input_ids):
                return embed(input_ids)

        fn = self.resolve(Model(), embed)
        self.assertTrue(self.torch.allclose(fn(self._ids()), embed(self._ids())))


class TakesInputIdsTest(unittest.TestCase):
    def setUp(self):
        try:
            _, self.takes_ids = _import()
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")

    def test_bound_method_with_ids(self):
        class M:
            def get_input_embeddings(self, input_ids):
                return input_ids

        self.assertTrue(self.takes_ids(M().get_input_embeddings))

    def test_bound_method_without_args(self):
        class M:
            def get_input_embeddings(self):
                return None

        self.assertFalse(self.takes_ids(M().get_input_embeddings))

    def test_keyword_only_does_not_count(self):
        def fn(*, input_ids=None):
            return input_ids

        self.assertFalse(self.takes_ids(fn))

    def test_unintrospectable_callable_is_refused(self):
        # Some C-implemented callables have no signature; refusing keeps the
        # fallback rather than raising during graph setup.
        self.assertFalse(self.takes_ids(print if False else object()))


if __name__ == "__main__":
    unittest.main()
