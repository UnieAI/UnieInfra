"""A cache row must be one token of the layer that owns it.

The KV pool allocates every layer's field at one width and lets a narrower
layer reinterpret the same bytes as more rows. That reinterpretation was
computed from the layer's head COUNT against the allocation's head_dim, which
is only the layer's width while every layer shares a head_dim.

gemma-4 is the first model here where they do not:

    allocation (its sliding half)   16 heads x 256 = 4,096 elements per token
    its ten global layers            4 heads x 512 = 2,048   (half)
    what the head count alone said   4 heads x 256 = 1,024   (a quarter)

Off by two, so every global-layer token was written and read at half stride
while the 50 sliding layers stayed correct -- the engine answered "What is the
capital of France?" with '額LLLLLLL額額額額...' and every latency number in the
harness looked healthy. Layer-by-layer against HF put the first divergence at
layer 5 and every later one at 11, 17, 23, 29, 35, 41, 47, 53, 59: exactly the
ten full_attention layers.

The end-to-end check is `.attn_cmp.py` (layer 5 against SDPA on the q/k/v the
engine itself passed in, with layer 0 as the control). These are the pieces of
it that need no GPU.
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


class PerLayerKvGeometryTest(unittest.TestCase):
    """MHAConfig has to learn the geometry from the checkpoint."""

    def setUp(self):
        try:
            from tokenspeed.runtime.layers.attention.configs.mha import (
                _per_layer_kv_geometry,
            )
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"needs the runtime package: {exc}")
        self.fn = _per_layer_kv_geometry

    def test_gemma4_global_layers_are_four_by_512(self):
        heads, dims = self.fn(
            SimpleNamespace(num_global_key_value_heads=4, global_head_dim=512),
            GEMMA4_TYPES,
            60,
            16,
            256,
        )
        self.assertEqual(len(heads), 60)
        for i, label in enumerate(GEMMA4_TYPES):
            want = (16, 256) if label == "sliding_attention" else (4, 512)
            self.assertEqual((heads[i], dims[i]), want, f"layer {i} ({label})")

    def test_the_global_layers_are_exactly_half_the_allocation(self):
        """The number this whole change exists for. If it is not 2, the row
        view and the allocation disagree by something else."""
        heads, dims = self.fn(
            SimpleNamespace(num_global_key_value_heads=4, global_head_dim=512),
            GEMMA4_TYPES,
            60,
            16,
            256,
        )
        alloc = 16 * 256
        self.assertEqual(alloc // (heads[5] * dims[5]), 2)
        self.assertEqual(heads[0] * dims[0], alloc)

    def test_a_uniform_model_stays_on_the_uniform_path(self):
        """None, not a tuple of identical values: the pool's fast path checks
        for None, and a model that never had this problem must not change."""
        self.assertEqual(
            self.fn(SimpleNamespace(), GEMMA4_TYPES, 60, 16, 128), (None, None)
        )

    def test_declared_but_identical_is_still_uniform(self):
        self.assertEqual(
            self.fn(
                SimpleNamespace(num_global_key_value_heads=16, global_head_dim=256),
                GEMMA4_TYPES,
                60,
                16,
                256,
            ),
            (None, None),
        )

    def test_labels_that_do_not_cover_the_stack_are_refused(self):
        """Without per-layer labels there is no way to say WHICH layers are
        global, and guessing would be wrong on every layer instead of some."""
        with self.assertRaises(ValueError):
            self.fn(
                SimpleNamespace(num_global_key_value_heads=4, global_head_dim=512),
                ("sliding_attention",),
                60,
                16,
                256,
            )


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

    def _pool(self, dims):
        """A bare instance: __init__ allocates a real arena, and the row view
        needs none of it."""
        pool = self.cls.__new__(self.cls)
        pool.head_num = 16
        pool.head_dim = 256
        pool._layer_kv_head_counts = (16,) * 5 + (4,)
        pool._layer_kv_head_dims = dims
        pool._kv_alloc_head_count = 16
        return pool

    def test_global_layer_row_is_half_the_allocation(self):
        pool = self._pool((256,) * 5 + (512,))
        buf = self.torch.zeros(8, 16, 256)
        sliding = pool._layer_row_view(buf, 0)
        glob = pool._layer_row_view(buf, 5)
        self.assertEqual(tuple(sliding.shape), (8, 16, 256))
        self.assertEqual(tuple(glob.shape), (16, 4, 512))
        self.assertEqual(glob.shape[1] * glob.shape[2], 2048)

    def test_without_dims_it_is_the_old_quarter_width(self):
        """Pins the defect this change fixes, so a revert cannot pass."""
        pool = self._pool(None)
        glob = pool._layer_row_view(self.torch.zeros(8, 16, 256), 5)
        self.assertEqual(glob.shape[1] * glob.shape[2], 1024)
        self.assertNotEqual(
            glob.shape[1] * glob.shape[2],
            self._pool((256,) * 5 + (512,))
            ._layer_row_view(self.torch.zeros(8, 16, 256), 5)
            .shape[1:]
            .numel(),
        )

    def test_a_uniform_layer_is_returned_untouched(self):
        pool = self._pool((256,) * 6)
        buf = self.torch.zeros(8, 16, 256)
        self.assertIs(pool._layer_row_view(buf, 0), buf)

    def test_dims_without_counts_is_refused(self):
        pool = self.cls.__new__(self.cls)
        pool.head_num = 16
        pool.head_dim = 256
        pool._layer_kv_head_counts = None
        pool._layer_kv_head_dims = (512,) * 6
        # The guard lives in __init__; the view degrades to the uniform path
        # rather than overrunning, which is the safe half of the same rule.
        buf = self.torch.zeros(8, 16, 256)
        self.assertIs(pool._layer_row_view(buf, 0), buf)


if __name__ == "__main__":
    unittest.main()
