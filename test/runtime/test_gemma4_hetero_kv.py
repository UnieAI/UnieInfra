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

"""Gemma 4's two KV geometries on the MHA pool.

Gemma 4's layers are not one shape: sliding-attention layers serve
``num_key_value_heads x head_dim`` and full-attention layers
``num_global_key_value_heads x global_head_dim`` -- HALF the bytes per token
(31B: 4 x 512 = 2048 against 16 x 256 = 4096; 26B-A4B: 2 x 512 = 1024 against
8 x 256 = 2048).

The KV pool allocates one row shape for every layer. Before the fix, writing
a full layer's K into a sliding-sized row SUCCEEDED: ``store_kv_cache``
copies ``src.numel() // n_tokens`` elements per token at the DESTINATION row
stride and never compares the two, so the leading half landed and the tail
kept whatever the row held -- zeros on a fresh pool, another layer's KV once
paired layers alias a slab. No exception, no shape error, and the model then
generated confidently wrong output.

These tests pin, at both checkpoints' geometry:

1. the guard: a store whose per-token width is not the row it lands in
   raises, in BOTH directions (truncation and overrun);
2. the fix: with the geometries normalized onto a common head dim
   (``normalize_layer_kv_geometry``), store/load round-trips every element of
   both geometries, through the same view the attention backend re-pages;
3. the aliased slab: a full layer's write does not corrupt the sliding layer
   sharing its slab.
"""

from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")


# (name, kv_heads x head_dim on sliding layers, on full layers, num layers,
#  full-attention stride) -- read off the checkpoints' config.json
# text_config, NOT invented: 31B has 60 layers with full attention every 6th
# from index 5; 26B-A4B has 30 on the same stride.
GEMMA4_CHECKPOINTS = (
    ("gemma-4-31B-it", (16, 256), (4, 512), 60),
    ("gemma-4-26B-A4B-it", (8, 256), (2, 512), 30),
)
BASE_PAGE = 128


def _layer_types(num_layers: int) -> tuple[str, ...]:
    return tuple(
        "full_attention" if (i % 6) == 5 else "sliding_attention"
        for i in range(num_layers)
    )


class Gemma4HeteroKVTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"needs torch: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        try:
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
            from tokenspeed.runtime.layers.paged_attention import PagedAttention
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"needs tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MHATokenToKVPool = MHATokenToKVPool
        self.PagedAttention = PagedAttention

    @property
    def normalize(self):
        # Imported lazily and NOT in setUp: a build without the normalizer
        # must make the guard tests FAIL, not silently skip the whole class.
        from tokenspeed.runtime.layers.attention.kv_cache.utils import (
            normalize_layer_kv_geometry,
        )

        return normalize_layer_kv_geometry

    # -- helpers ------------------------------------------------------

    def _geometry(self, swa, full, num_layers):
        return [
            full if t == "full_attention" else swa for t in _layer_types(num_layers)
        ]

    def _pool(self, swa, num_layers, *, hetero, counts=None, alloc=None, size=512):
        page_sizes = {"full_attention": 2 * BASE_PAGE}
        return self.MHATokenToKVPool(
            size=size,
            dtype=self.torch.bfloat16,
            head_num=swa[0],
            head_dim=swa[1],
            layer_num=num_layers,
            device="cuda",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=size,
            page_size=BASE_PAGE,
            rank=0,
            layer_types=_layer_types(num_layers),
            sliding_window_tokens=1024,
            max_scheduled_tokens=size,
            slot_tokens=2 * BASE_PAGE if hetero else None,
            group_page_sizes=page_sizes if hetero else None,
            layer_kv_head_counts=counts if hetero else None,
            kv_alloc_head_count=alloc if hetero else None,
        )

    def _store(self, pool, layer_id, group_id, heads, dim, loc, k, v):
        layer = self.PagedAttention(
            num_heads=heads,
            head_dim=dim,
            scaling=1.0,
            num_kv_heads=heads,
            layer_id=layer_id,
            group_id=group_id,
        )
        pool.set_kv_buffer(layer, loc, k, v)

    def _rand(self, heads, dim, tokens=8):
        return self.torch.randn(
            tokens, heads, dim, dtype=self.torch.bfloat16, device="cuda"
        )

    # -- 1. the guard -------------------------------------------------

    def test_narrow_store_into_wide_row_raises(self):
        """The exact silent case: a full layer's K into a sliding-sized row."""
        torch = self.torch
        for name, swa, full, nl in GEMMA4_CHECKPOINTS:
            with self.subTest(name):
                pool = self._pool(swa, nl, hetero=False, size=256)
                full_layer = _layer_types(nl).index("full_attention")
                loc = torch.arange(1, 9, dtype=torch.int64, device="cuda")
                k = self._rand(*full)
                with self.assertRaises(ValueError) as cm:
                    self._store(
                        pool, full_layer, "full_attention", *full, loc, k, k.clone()
                    )
                msg = str(cm.exception)
                self.assertIn(str(full[0] * full[1]), msg)
                self.assertIn(str(swa[0] * swa[1]), msg)
                del pool
                torch.cuda.empty_cache()

    def test_wide_store_into_narrow_row_raises(self):
        """The other direction: an overrun must not be silent either."""
        torch = self.torch
        for name, swa, full, nl in GEMMA4_CHECKPOINTS:
            with self.subTest(name):
                pool = self._pool(swa, nl, hetero=False, size=256)
                loc = torch.arange(1, 5, dtype=torch.int64, device="cuda")
                k = self._rand(swa[0] * 2, swa[1], tokens=4)
                with self.assertRaises(ValueError):
                    self._store(
                        pool,
                        0,
                        "sliding_attention",
                        swa[0] * 2,
                        swa[1],
                        loc,
                        k,
                        k.clone(),
                    )
                del pool
                torch.cuda.empty_cache()

    def test_uniform_store_is_not_falsely_rejected(self):
        """The guard must not fire on the shape the pool was built for."""
        torch = self.torch
        for name, swa, _full, nl in GEMMA4_CHECKPOINTS:
            with self.subTest(name):
                pool = self._pool(swa, nl, hetero=False, size=256)
                loc = torch.arange(1, 9, dtype=torch.int64, device="cuda")
                k = self._rand(*swa)
                v = self._rand(*swa)
                self._store(pool, 0, "sliding_attention", *swa, loc, k, v)
                torch.cuda.synchronize()
                got = pool.get_key_buffer(0)[loc]
                self.assertTrue(torch.equal(got, k))
                del pool
                torch.cuda.empty_cache()

    # -- 2. normalization ---------------------------------------------

    def test_normalize_layer_kv_geometry(self):
        for name, swa, full, nl in GEMMA4_CHECKPOINTS:
            with self.subTest(name):
                geo = self._geometry(swa, full, nl)
                alloc, counts = self.normalize(geo)
                self.assertEqual(len(counts), nl)
                # The allocation width is the pool's own head_num: the widest
                # layer must not exceed the row the pool allocates.
                self.assertEqual(alloc, swa[0])
                for (heads, dim), served in zip(geo, counts):
                    # Same bytes, re-counted in the common head dim.
                    self.assertEqual(heads * dim, served * swa[1])

    def test_normalize_rejects_incommensurable_geometry(self):
        with self.assertRaises(ValueError):
            # 3 x 100 = 300 is not a multiple of the common dim 128.
            self.normalize([(4, 128), (3, 100)])

    def test_pool_rejects_inexact_recount(self):
        """A head count that does not divide the slab row is refused at
        construction, not discovered as a straddled row in the first
        forward."""
        _name, swa, _full, nl = GEMMA4_CHECKPOINTS[0]
        counts = [swa[0]] * nl
        counts[5] = 5  # 16 / 5 is not an exact re-count
        with self.assertRaises(ValueError):
            self._pool(swa, nl, hetero=True, counts=tuple(counts), alloc=swa[0])

    # -- 3. round-trip at both geometries -----------------------------

    def test_round_trip_preserves_every_element(self):
        torch = self.torch
        for name, swa, full, nl in GEMMA4_CHECKPOINTS:
            with self.subTest(name):
                alloc, counts = self.normalize(self._geometry(swa, full, nl))
                pool = self._pool(swa, nl, hetero=True, counts=counts, alloc=alloc)
                lts = _layer_types(nl)
                swa_l, full_l = lts.index("sliding_attention"), lts.index(
                    "full_attention"
                )

                written = {}
                for label, lid, geo in (
                    ("sliding", swa_l, swa),
                    ("full", full_l, full),
                ):
                    heads, dim = geo
                    # Locs are per-layer VIEW rows: page id x the GROUP's page
                    # size, exactly as backends/flat_groups.py computes them.
                    # Distinct page ids so the aliased slab is exercised
                    # without the two groups overlapping.
                    page_id, page = (
                        (1, BASE_PAGE)
                        if label == "sliding"
                        else (
                            2,
                            2 * BASE_PAGE,
                        )
                    )
                    tokens = 64
                    loc = torch.arange(
                        page_id * page,
                        page_id * page + tokens,
                        dtype=torch.int64,
                        device="cuda",
                    )
                    k = self._rand(heads, dim, tokens)
                    v = self._rand(heads, dim, tokens)
                    self._store(pool, lid, lts[lid], heads, dim, loc, k, v)
                    torch.cuda.synchronize()
                    written[label] = (lid, heads, dim, page, loc, k, v)

                for label, (lid, heads, dim, page, loc, k, v) in written.items():
                    # The row the pool serves must be exactly the checkpoint's
                    # per-token KV width.
                    self.assertEqual(pool._layer_row_elems(lid), heads * dim, label)
                    # Read back through the SAME re-view the attention backend
                    # applies (backends/mha.py::_get_kv_cache).
                    kb = pool.get_key_buffer(lid).view(-1, page, heads, dim)
                    vb = pool.get_value_buffer(lid).view(-1, page, heads, dim)
                    got_k = kb.reshape(-1, heads, dim)[loc]
                    got_v = vb.reshape(-1, heads, dim)[loc]
                    self.assertTrue(torch.equal(got_k, k), f"{label} K")
                    self.assertTrue(torch.equal(got_v, v), f"{label} V")

                # The aliased-slab case only exists when the hybrid slab
                # layout is active, which needs a scheduler ext built with
                # TOKENSPEED_FLAT_KVCACHE=ON. On a radix build the pool falls
                # back to one buffer per layer -- the round-trip above still
                # has to hold, but there is no shared slab to corrupt.
                if pool._slab_group_size is None:
                    self.assertNotEqual(
                        pool.k_buffer[swa_l].data_ptr(),
                        pool.k_buffer[full_l].data_ptr(),
                        "per-layer layout must not alias two layers",
                    )
                else:
                    # The two layers really do share one slab, and neither
                    # clobbered the other.
                    self.assertEqual(
                        pool.k_buffer[swa_l].data_ptr(),
                        pool.k_buffer[full_l].data_ptr(),
                    )
                lid, heads, dim, _page, loc, k, _v = written["sliding"]
                still = pool.get_key_buffer(lid).reshape(-1, heads, dim)[loc]
                self.assertTrue(torch.equal(still, k))

                del pool
                torch.cuda.empty_cache()

    def test_hetero_store_is_not_falsely_rejected(self):
        """With hetero on, BOTH geometries store without raising."""
        torch = self.torch
        for name, swa, full, nl in GEMMA4_CHECKPOINTS:
            with self.subTest(name):
                alloc, counts = self.normalize(self._geometry(swa, full, nl))
                pool = self._pool(swa, nl, hetero=True, counts=counts, alloc=alloc)
                lts = _layer_types(nl)
                for lid, geo, page_id, page in (
                    (lts.index("sliding_attention"), swa, 1, BASE_PAGE),
                    (lts.index("full_attention"), full, 2, 2 * BASE_PAGE),
                ):
                    loc = torch.arange(
                        page_id * page,
                        page_id * page + 8,
                        dtype=torch.int64,
                        device="cuda",
                    )
                    k = self._rand(*geo)
                    self._store(pool, lid, lts[lid], *geo, loc, k, k.clone())
                torch.cuda.synchronize()
                del pool
                torch.cuda.empty_cache()


class Gemma4RegistryGateTest(unittest.TestCase):
    """The registry must publish the hetero layout for Gemma 4; without it the
    pool is uniform and (now) refuses every full-attention store."""

    def setUp(self):
        try:
            import tokenspeed.runtime.layers.attention.registry  # noqa: F401
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"needs the runtime: {exc}")

    @property
    def apply(self):
        # Missing gate => FAIL, not skip (see the note in the pool test).
        from tokenspeed.runtime.layers.attention.registry import (
            _apply_gemma4_hetero_kv,
        )

        return _apply_gemma4_hetero_kv

    @property
    def geometry(self):
        from tokenspeed.runtime.layers.attention.registry import (
            gemma4_layer_kv_geometry,
        )

        return gemma4_layer_kv_geometry

    def _text_config(self, swa, full, num_layers):
        from types import SimpleNamespace

        return SimpleNamespace(
            num_key_value_heads=swa[0],
            head_dim=swa[1],
            num_global_key_value_heads=full[0],
            global_head_dim=full[1],
            layer_types=list(_layer_types(num_layers)),
        )

    def _config(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            page_size=BASE_PAGE,
            slot_tokens=None,
            group_page_sizes=None,
            layer_kv_head_counts=None,
        )

    def _model_config(self, tc):
        from types import SimpleNamespace

        return SimpleNamespace(hf_config=SimpleNamespace(get_text_config=lambda: tc))

    def _flat_ext(self, value):
        from unittest import mock

        from tokenspeed.runtime.layers.attention import registry

        # The gate resolves the probe from its own module globals at call
        # time; the installed ext's real build flavor must not decide the test.
        return mock.patch.object(
            registry, "scheduler_ext_flat_kvcache", return_value=value
        )

    def test_gate_publishes_hetero_layout(self):
        for name, swa, full, nl in GEMMA4_CHECKPOINTS:
            with self.subTest(name), self._flat_ext(True):
                tc = self._text_config(swa, full, nl)
                cfg = self._config()
                self.assertTrue(self.apply(cfg, self._model_config(tc)))
                self.assertEqual(cfg.slot_tokens, 2 * BASE_PAGE)
                self.assertEqual(
                    cfg.group_page_sizes, {"full_attention": 2 * BASE_PAGE}
                )
                counts = cfg.layer_kv_head_counts
                self.assertEqual(len(counts), nl)
                for (heads, dim), served in zip(self.geometry(tc), counts):
                    self.assertEqual(heads * dim, served * swa[1])

    def test_gate_is_a_noop_for_uniform_geometry(self):
        # A Gemma 4 variant whose full layers match the sliding geometry needs
        # no hetero layout; the gate must leave the plain pool alone.
        tc = self._text_config((8, 256), (8, 256), 12)
        cfg = self._config()
        with self._flat_ext(True):
            self.assertFalse(self.apply(cfg, self._model_config(tc)))
        self.assertIsNone(cfg.layer_kv_head_counts)
        self.assertIsNone(cfg.slot_tokens)

    def test_gate_refuses_on_a_radix_built_ext(self):
        """Per-group page sizes are only honoured by the flat ext.

        Publishing them on a radix build would emit write locs in the wrong
        units; publishing nothing and continuing would leave a uniform pool
        that refuses the first full-attention write, i.e. the server starts and
        dies on the first request. So the gate raises here, while the message
        can still name the rebuild flag, and nothing is published.
        """
        _name, swa, full, nl = GEMMA4_CHECKPOINTS[0]
        tc = self._text_config(swa, full, nl)
        cfg = self._config()
        with self._flat_ext(False):
            with self.assertRaises(RuntimeError) as cm:
                self.apply(cfg, self._model_config(tc))
        self.assertIn("TOKENSPEED_FLAT_KVCACHE", str(cm.exception))
        self.assertIsNone(cfg.layer_kv_head_counts)
        self.assertIsNone(cfg.group_page_sizes)

    def test_gate_stays_silent_for_a_uniform_model_on_a_radix_ext(self):
        """The refusal is Gemma-4-hetero only; a uniform stack must not raise.

        Every other model on a radix build reaches the same code path, so the
        early return for uniform geometry has to come first.
        """
        tc = self._text_config((8, 256), (8, 256), 12)
        cfg = self._config()
        with self._flat_ext(False):
            self.assertFalse(self.apply(cfg, self._model_config(tc)))


if __name__ == "__main__":
    unittest.main()
