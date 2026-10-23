"""A cache row must be one token of the layer that owns it.

gemma-4 is the first model here whose layers differ in head_dim as well as
head count:

    its sliding layers   16 heads x 256 = 4,096 elements per token
    its ten global ones   4 heads x 512 = 2,048   (half)
    head count alone      4 heads x 256 = 1,024   (a quarter)

Reading a global layer's row at the sliding head_dim puts every global-layer
token at half stride while the 50 sliding layers stay correct -- the engine
answered "What is the capital of France?" with '額LLLLLLL額額額額...' and every
latency number in the harness looked healthy. Layer-by-layer against HF put
the first divergence at layer 5 and every later one at 11, 17, 23, ... 59:
exactly the ten full_attention layers.

The recipe now plans each layer's plane at the layer's own (kv_heads,
head_dim) -- gemma4_layer_kv_geometry, covered in models/test_gemma4.py --
and the pool's row view takes the width from the plane. These are the pieces
of that which need no GPU; test_mha_per_layer_geometry.py checks the bytes.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, suite="runtime-1gpu")

# gemma-4-31B-it's real stack: 50 sliding and 10 full in a 5:1 interleave, so
# the full layers land on 5, 11, 17, ... 59 -- the indices the layer-by-layer
# comparison against HF flagged.
GEMMA4_TYPES = (("sliding_attention",) * 5 + ("full_attention",)) * 10


def _gemma4_config(*, global_heads=4, global_dim=512):
    return SimpleNamespace(
        model_type="gemma4",
        text_config=SimpleNamespace(
            model_type="gemma4_text",
            num_attention_heads=32,
            head_dim=256,
            global_head_dim=global_dim,
            num_key_value_heads=16,
            num_global_key_value_heads=global_heads,
            layer_types=list(GEMMA4_TYPES),
        ),
    )


class PerLayerKvGeometryTest(unittest.TestCase):
    """MHAConfig has to learn the geometry from the checkpoint."""

    def setUp(self):
        try:
            from tokenspeed.runtime.layers.attention.configs.mha import (
                _per_layer_kv_geometry,
            )
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"needs the runtime package: {exc}")
        self.fn = lambda hf: _per_layer_kv_geometry(SimpleNamespace(hf_config=hf))

    def test_gemma4_global_layers_are_four_by_512(self):
        geometry = self.fn(_gemma4_config())
        self.assertEqual(len(geometry), 60)
        for i, label in enumerate(GEMMA4_TYPES):
            want = (16, 256) if label == "sliding_attention" else (4, 512)
            self.assertEqual(geometry[i], want, f"layer {i} ({label})")

    def test_the_global_layers_are_exactly_half_the_sliding_width(self):
        """The number this whole test exists for. If it is not 2, the row
        view and the plan disagree by something else."""
        geometry = self.fn(_gemma4_config())
        (sliding_heads, sliding_dim), (full_heads, full_dim) = geometry[0], geometry[5]
        self.assertEqual((sliding_heads * sliding_dim) // (full_heads * full_dim), 2)

    def test_a_uniform_model_stays_on_the_uniform_path(self):
        """None, not a tuple of identical values: the pool's fast path checks
        for None, and a model that never had this problem must not change."""
        self.assertIsNone(self.fn(SimpleNamespace(model_type="llama")))

    def test_declared_but_identical_is_still_uniform(self):
        self.assertIsNone(self.fn(_gemma4_config(global_heads=16, global_dim=256)))


class RowViewWidthTest(unittest.TestCase):
    """The pool's row view must produce the layer's own width."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"needs torch + the runtime package: {exc}")
        self.torch = torch
        self.cls = MHATokenToKVPool

    def _pool(self, counts, *, head_dim=256):
        """A bare instance: __init__ allocates a real arena, and the row view
        needs none of it."""
        pool = self.cls.__new__(self.cls)
        pool.head_num = 16
        pool.head_dim = head_dim
        pool._layer_kv_head_counts = counts
        pool._kv_alloc_head_count = 16
        return pool

    def test_a_global_row_is_one_global_token(self):
        """The plan shapes the global plane at 4 x 512; the view keeps it there
        rather than splitting it at the pool's sliding head_dim."""
        pool = self._pool((16,) * 5 + (4,))
        glob = pool._layer_row_view(self.torch.zeros(8, 4, 512), 5)
        self.assertEqual(tuple(glob.shape), (8, 4, 512))
        self.assertEqual(glob.shape[1] * glob.shape[2], 2048)

    def test_a_sliding_row_is_returned_untouched(self):
        pool = self._pool((16,) * 5 + (4,))
        buf = self.torch.zeros(8, 16, 256)
        self.assertIs(pool._layer_row_view(buf, 0), buf)

    def test_a_fewer_head_layer_at_one_head_dim_splits_into_more_rows(self):
        """A head-count-only model (one head_dim) still reinterprets the
        allocation-wide plane as more rows of its own head count."""
        pool = self._pool((16,) * 5 + (4,), head_dim=128)
        rows = pool._layer_row_view(self.torch.zeros(8, 16, 128), 5)
        self.assertEqual(tuple(rows.shape), (32, 4, 128))

    def test_a_uniform_pool_is_returned_untouched(self):
        pool = self._pool(None)
        buf = self.torch.zeros(8, 16, 256)
        self.assertIs(pool._layer_row_view(buf, 5), buf)


if __name__ == "__main__":
    unittest.main()
