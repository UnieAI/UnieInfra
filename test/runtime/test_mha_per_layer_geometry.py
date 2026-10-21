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

"""Per-layer-type KV cache geometry in the ordinary MHA cache path.

Gemma-4 is the one MHA model whose head_dim and kv-head count differ by layer
type (full_attention 4 x 512, sliding_attention 16 x 256). These tests assert
that the ordinary cache recipe resolves each layer's page at its own geometry
when a per-layer ``layer_kv_geometry`` is present, AND that the uniform default
every other model takes (``layer_kv_geometry is None``) is byte-for-byte
unchanged -- the same field shapes and the same per-token byte budget as the
single-cell path produced before this change.

CPU-only: these build the config objects and the pure field/byte helpers
directly, with no GPU allocation or model construction.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The shared pool/arena builders live beside this file in test/runtime; make
# them importable whatever directory the test runner starts from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    FLAT_KV_PLANE,
    _config_bytes_per_token,
    _config_kv_head_counts,
    _config_layer_fields,
    _mha_layer_fields,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    group as group_declarations,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    layer_group_ids,
)

# gemma-4-31B's per-layer-type geometry (num_kv_heads, head_dim), PRE-TP.
_FULL = (4, 512)
_SLIDING = (16, 256)
# 5 sliding : 1 full, the gemma-4 layout: full at indices 5, 11.
_LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"]
_GEMMA4_GEOMETRY = tuple(
    _FULL if lt == "full_attention" else _SLIDING for lt in _LAYER_TYPES * 2
)
_PREFIX_GRANULARITY = 16


def _mha_config(
    *,
    layer_kv_geometry: tuple[tuple[int, int], ...] | None,
    num_kv_heads: int,
    head_dim: int,
    attn_tp_size: int,
    cache_layer_types: tuple[str, ...],
) -> AttnConfig:
    """A bare AttnConfig wrapping one MHAConfig, no server/model config.

    Carries only what the cache field + byte helpers read (geometry, cache
    dtype, prefix granularity); every scheduler/spec field takes a neutral
    value because these tests never run the capacity search.
    """
    spec = MHAConfig(
        backend_name="triton",
        num_attention_heads=max(num_kv_heads, 1),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        attn_tp_size=attn_tp_size,
        cache_layer_types=cache_layer_types,
        sliding_window_tokens=1024,
        layer_kv_geometry=layer_kv_geometry,
    )
    return AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="auto",
        kv_cache_mxfp8=False,
        prefix_granularity=_PREFIX_GRANULARITY,
        context_len=4096,
        max_bs=1,
        components=(spec,),
    )


def _kv_shapes(config: AttnConfig, num_layers: int) -> list[tuple[int, ...]]:
    """The K page shape each layer's MHA fields resolve to."""
    shapes = []
    for layer_id in range(num_layers):
        fields = _mha_layer_fields(config, layer_id, layer_id, 0)
        # fields[0] is the K page; .shape is (prefix_granularity, kv_heads, head_dim)
        shapes.append(tuple(fields[0].shape))
    return shapes


class PerLayerGeometryFieldTest(unittest.TestCase):
    """A per-layer gemma-4 config pages every layer at its own geometry."""

    def test_full_and_sliding_layers_resolve_distinct_head_dims(self):
        config = _mha_config(
            layer_kv_geometry=_GEMMA4_GEOMETRY,
            num_kv_heads=16,
            head_dim=256,
            attn_tp_size=1,
            cache_layer_types=tuple(_LAYER_TYPES * 2),
        )
        shapes = _kv_shapes(config, len(_GEMMA4_GEOMETRY))
        # Sliding layers (indices 0-4, 6-10): 16 kv heads x 256.
        self.assertEqual(shapes[0], (_PREFIX_GRANULARITY, 16, 256))
        self.assertEqual(shapes[4], (_PREFIX_GRANULARITY, 16, 256))
        # Full layers (indices 5, 11): 4 kv heads x 512.
        self.assertEqual(shapes[5], (_PREFIX_GRANULARITY, 4, 512))
        self.assertEqual(shapes[11], (_PREFIX_GRANULARITY, 4, 512))

    def test_tp_shards_the_per_layer_kv_head_count(self):
        """TP=2 halves each layer's kv-head count; head_dim is untouched."""
        config = _mha_config(
            layer_kv_geometry=_GEMMA4_GEOMETRY,
            num_kv_heads=16,
            head_dim=256,
            attn_tp_size=2,
            cache_layer_types=tuple(_LAYER_TYPES * 2),
        )
        shapes = _kv_shapes(config, len(_GEMMA4_GEOMETRY))
        self.assertEqual(shapes[0], (_PREFIX_GRANULARITY, 8, 256))  # 16 // 2
        self.assertEqual(shapes[5], (_PREFIX_GRANULARITY, 2, 512))  # 4 // 2

    def test_per_layer_bytes_sum_the_actual_layer_cells(self):
        """The byte budget sums each layer's real cell, not one cell x N."""
        config = _mha_config(
            layer_kv_geometry=_GEMMA4_GEOMETRY,
            num_kv_heads=16,
            head_dim=256,
            attn_tp_size=1,
            cache_layer_types=tuple(_LAYER_TYPES * 2),
        )
        num_layers = len(_GEMMA4_GEOMETRY)
        elem = torch._utils._element_size(torch.bfloat16)
        sliding_cell = 16 * 256 * 2 * elem
        full_cell = 4 * 512 * 2 * elem
        expected = (10 * sliding_cell) + (2 * full_cell)
        self.assertEqual(_config_bytes_per_token(config, num_layers), expected)


class PerLayerPlaneTest(unittest.TestCase):
    """Per-layer-geometry layers share one flat K/V plane.

    Gemma-4's full (4 x 512) and sliding (16 x 256) layers cannot pair
    byte-equal planes the way a uniform hybrid model does, and planes of their
    own would leave the other group's planes of every parent unused. Every
    field instead lands on one flat plane, strided by whole payloads. The
    uniform default keeps the group-agnostic ``unit.{occurrence}`` plane every
    current model relies on so byte-equal full/sliding layers still pair in
    one slab.
    """

    def _fields(self, config, layer_id, occurrence):
        return _mha_layer_fields(config, layer_id, layer_id, occurrence)

    def test_every_per_layer_field_shares_the_flat_plane(self):
        config = _mha_config(
            layer_kv_geometry=_GEMMA4_GEOMETRY,
            num_kv_heads=16,
            head_dim=256,
            attn_tp_size=1,
            cache_layer_types=tuple(_LAYER_TYPES * 2),
        )
        for layer_id, occurrence in ((0, 0), (1, 1), (5, 0), (11, 1)):
            for field in self._fields(config, layer_id, occurrence):
                self.assertEqual(field.plane_id, FLAT_KV_PLANE)
                self.assertFalse(field.exact_page_stride)
                # The stride is whole payloads, so a page starts on a kernel
                # page of the field's row view.
                self.assertEqual(field.page_stride_alignment_bytes, field.payload_bytes)

    def test_uniform_model_keeps_group_agnostic_unit_plane(self):
        config = _mha_config(
            layer_kv_geometry=None,
            num_kv_heads=8,
            head_dim=128,
            attn_tp_size=1,
            cache_layer_types=("full_attention",) * 6,
        )
        fields = self._fields(config, 0, 0)
        self.assertEqual({f.plane_id for f in fields}, {"unit.0.k", "unit.0.v"})
        self.assertTrue(all(f.exact_page_stride for f in fields))


class UniformDefaultInvariantTest(unittest.TestCase):
    """A model with one geometry for every layer is unchanged by this change.

    ``layer_kv_geometry is None`` is the path every current model (gemma-3,
    llama, qwen, ...) takes; the shapes and the byte budget must equal the
    single-cell path exactly.
    """

    def test_uniform_field_shapes_are_the_single_cell_shapes(self):
        config = _mha_config(
            layer_kv_geometry=None,
            num_kv_heads=8,
            head_dim=128,
            attn_tp_size=1,
            cache_layer_types=("full_attention",) * 6,
        )
        shapes = _kv_shapes(config, 6)
        for shape in shapes:
            self.assertEqual(shape, (_PREFIX_GRANULARITY, 8, 128))

    def test_uniform_bytes_equal_cell_size_times_layers(self):
        """The uniform byte budget is exactly cache_cell_size x layer count."""
        config = _mha_config(
            layer_kv_geometry=None,
            num_kv_heads=8,
            head_dim=128,
            attn_tp_size=1,
            cache_layer_types=("full_attention",) * 6,
        )
        cell = config.cache_cell_size()
        self.assertEqual(_config_bytes_per_token(config, 6), cell * 6)

    def test_uniform_tp_shards_the_single_kv_head_count(self):
        config = _mha_config(
            layer_kv_geometry=None,
            num_kv_heads=8,
            head_dim=128,
            attn_tp_size=4,
            cache_layer_types=("full_attention",) * 6,
        )
        shapes = _kv_shapes(config, 6)
        for shape in shapes:
            self.assertEqual(shape, (_PREFIX_GRANULARITY, 2, 128))  # 8 // 4


class TotalCacheBytesPerTokenTest(unittest.TestCase):
    """``MHAConfig.total_cache_bytes_per_token`` is the per-group byte budget.

    The per-layer path sums each layer's own K+V cell (full and sliding cells
    differ); the uniform path is a byte-for-byte identity with one cell times
    the layer count.
    """

    def test_per_layer_sum_charges_each_layer_its_own_cell(self):
        config = _mha_config(
            layer_kv_geometry=_GEMMA4_GEOMETRY,
            num_kv_heads=16,
            head_dim=256,
            attn_tp_size=1,
            cache_layer_types=tuple(_LAYER_TYPES * 2),
        )
        spec = config.component(SoftmaxAttnConfig)
        num_layers = len(_GEMMA4_GEOMETRY)
        elem = torch._utils._element_size(torch.bfloat16)
        sliding_cell = 16 * 256 * 2 * elem
        full_cell = 4 * 512 * 2 * elem
        expected = (10 * sliding_cell) + (2 * full_cell)
        self.assertEqual(spec.total_cache_bytes_per_token(config, num_layers), expected)

    def test_per_layer_sum_shards_kv_heads_under_tp(self):
        """TP halves the kv-head count in every per-layer cell; head_dim is
        untouched, so the summed budget halves exactly."""
        config = _mha_config(
            layer_kv_geometry=_GEMMA4_GEOMETRY,
            num_kv_heads=16,
            head_dim=256,
            attn_tp_size=2,
            cache_layer_types=tuple(_LAYER_TYPES * 2),
        )
        spec = config.component(SoftmaxAttnConfig)
        num_layers = len(_GEMMA4_GEOMETRY)
        elem = torch._utils._element_size(torch.bfloat16)
        sliding_cell = (16 // 2) * 256 * 2 * elem
        full_cell = (4 // 2) * 512 * 2 * elem
        expected = (10 * sliding_cell) + (2 * full_cell)
        self.assertEqual(spec.total_cache_bytes_per_token(config, num_layers), expected)

    def test_uniform_total_is_cell_size_times_layers(self):
        """``layer_kv_geometry is None`` sums to exactly cache_cell_size x N,
        the identity the single-geometry path has always produced."""
        config = _mha_config(
            layer_kv_geometry=None,
            num_kv_heads=8,
            head_dim=128,
            attn_tp_size=1,
            cache_layer_types=("full_attention",) * 6,
        )
        spec = config.component(SoftmaxAttnConfig)
        self.assertEqual(
            spec.total_cache_bytes_per_token(config, 6),
            spec.cache_cell_size(config) * 6,
        )

    def test_geometry_length_mismatch_raises(self):
        """A geometry that does not cover every layer is a configuration bug,
        not a silently truncated budget."""
        config = _mha_config(
            layer_kv_geometry=_GEMMA4_GEOMETRY,
            num_kv_heads=16,
            head_dim=256,
            attn_tp_size=1,
            cache_layer_types=tuple(_LAYER_TYPES * 2),
        )
        spec = config.component(SoftmaxAttnConfig)
        with self.assertRaises(ValueError):
            spec.total_cache_bytes_per_token(config, len(_GEMMA4_GEOMETRY) + 1)


class PerLayerKvHeadCountsTest(unittest.TestCase):
    """The recipe publishes per-layer KV head counts only for a per-layer model.

    ``_config_kv_head_counts`` is the pure helper behind the recipe's
    ``layer_kv_head_counts`` property: it drops the head_dim axis of the
    per-layer geometry and keeps the kv-head count the pool and scheduler read.
    """

    def test_per_layer_counts_are_the_geometry_kv_heads_pre_tp(self):
        config = _mha_config(
            layer_kv_geometry=_GEMMA4_GEOMETRY,
            num_kv_heads=16,
            head_dim=256,
            attn_tp_size=1,
            cache_layer_types=tuple(_LAYER_TYPES * 2),
        )
        counts = _config_kv_head_counts(config, len(_GEMMA4_GEOMETRY))
        expected = tuple(kv_heads for kv_heads, _ in _GEMMA4_GEOMETRY)
        self.assertEqual(counts, expected)
        # The two cache groups show up as the two distinct counts, PRE-TP.
        self.assertEqual(set(counts), {4, 16})

    def test_uniform_model_publishes_no_per_layer_counts(self):
        """A model with one geometry everywhere keeps the None uniform path, so
        the cache spec carries no per-layer vector."""
        config = _mha_config(
            layer_kv_geometry=None,
            num_kv_heads=8,
            head_dim=128,
            attn_tp_size=1,
            cache_layer_types=("full_attention",) * 6,
        )
        self.assertIsNone(_config_kv_head_counts(config, 6))

    def test_none_config_has_no_counts(self):
        """A missing (draft) side contributes no per-layer counts."""
        self.assertIsNone(_config_kv_head_counts(None, 0))

    def test_geometry_length_mismatch_raises(self):
        config = _mha_config(
            layer_kv_geometry=_GEMMA4_GEOMETRY,
            num_kv_heads=16,
            head_dim=256,
            attn_tp_size=1,
            cache_layer_types=tuple(_LAYER_TYPES * 2),
        )
        with self.assertRaises(ValueError):
            _config_kv_head_counts(config, len(_GEMMA4_GEOMETRY) + 1)


# The equal-depth-split gemma-4 layout the recipe actually packs: every cache
# group holds the same number of layers (split_groups_to_equal_depth turns the
# 5:1 sliding:full ratio into five sliding sub-groups, each paired 1:1 with the
# full group), so for a two-block stack each group has exactly two occurrences.
# full_attention layers are 4 x 512, every sliding sub-group layer is 16 x 256.
_SPLIT_LAYER_TYPES = (
    "sliding_attention_0",
    "sliding_attention_1",
    "sliding_attention_2",
    "sliding_attention_3",
    "sliding_attention_4",
    "full_attention",
) * 2
_SPLIT_GEOMETRY = tuple(
    _FULL if lt == "full_attention" else _SLIDING for lt in _SPLIT_LAYER_TYPES
)


def _gemma4_split_config(*, attn_tp_size: int) -> AttnConfig:
    """A per-layer gemma-4 config over the equal-depth-split labels."""
    return _mha_config(
        layer_kv_geometry=_SPLIT_GEOMETRY,
        num_kv_heads=16,
        head_dim=256,
        attn_tp_size=attn_tp_size,
        cache_layer_types=_SPLIT_LAYER_TYPES,
    )


def _pack_gemma4(config: AttnConfig):
    """Pack the gemma-4 two-geometry groups the way the recipe does.

    Walks the layers once through the shared ``group`` builder (deriving one
    cache group per distinct id, each field carrying its layer's own geometry)
    and packs with ``cache_blocks_per_lcm_block=None`` -- the per-layer branch
    of ``OrdinaryRecipe.packing`` -- so ``pack`` derives per-group packing from
    the groups' byte ratios. ``alignment`` and ``max_padding_fraction`` match
    the ordinary recipe's seams (1 and 1.0).
    """
    spec = config.component(SoftmaxAttnConfig)
    group_ids = tuple(
        layer_group_ids(
            layer_types=tuple(spec.cache_layer_types),
            sliding_window_tokens=spec.sliding_window_tokens,
        )
    )

    def fields_for_layer(layer_id, group_id, occurrence):
        return _config_layer_fields(
            config,
            layer_id=layer_id,
            local_layer_id=layer_id,
            occurrence=occurrence,
        )

    groups = group_declarations(
        layer_types=tuple(spec.cache_layer_types),
        group_ids=group_ids,
        sliding_window_tokens=spec.sliding_window_tokens,
        prefix_granularity=config.prefix_granularity,
        fields_for_layer=fields_for_layer,
        pd_disaggregation_enabled=False,
    )
    layout = pack(
        groups,
        prefix_granularity=config.prefix_granularity,
        cache_blocks_per_lcm_block=None,
        alignment=1,
        max_padding_fraction=1.0,
    )
    return layout


class PerLayerGeometryPackingTest(unittest.TestCase):
    """The two-geometry gemma-4 cache packs on the general (derived) path.

    This is the regression for the padding blowup: pinning every group to one
    CacheBlock per parent (the uniform shared-slab policy) sized each group's
    stride at the whole LCM block while its payload filled only its own planes,
    so the full group's padding fraction hit 10.0 and ``pack`` rejected it.
    Returning ``None`` from the per-layer branch lets ``pack`` derive per-group
    packing from byte ratios, which fits inside the padding budget.
    """

    def test_two_geometry_cache_packs_without_padding_blowup(self):
        # Would have raised ValueError (padding fraction exceeds limit) under
        # the old shared-slab pin; the derived path packs cleanly.
        layout = _pack_gemma4(_gemma4_split_config(attn_tp_size=1))
        packing = dict(layout.group_packing)
        # The derived packing is the byte-ratio one.
        self.assertEqual(packing["full_attention"], 2)
        for k in range(5):
            self.assertEqual(packing[f"sliding_attention_{k}"], 1)

    def test_derived_packing_is_the_byte_ratio_full_gets_twice_sliding(self):
        """Full page 131072 B is half the sliding page 262144 B, so the full
        group gets twice the sliding group's CacheBlocks per parent.

        The two groups hold equal layer counts after the equal-depth split, so
        the per-group raw totals keep the single-layer page ratio: full
        4 x 512 x 2B x 16 tokens x 2 planes = 131072 B, sliding 16 x 256 x 2B
        x 16 tokens x 2 planes = 262144 B.
        """
        layout = _pack_gemma4(_gemma4_split_config(attn_tp_size=1))
        packing = dict(layout.group_packing)
        for k in range(5):
            self.assertEqual(
                packing["full_attention"], 2 * packing[f"sliding_attention_{k}"]
            )

    def test_tp_preserves_the_byte_ratio_packing(self):
        """TP shards both geometries' kv heads by the same factor, so the full/
        sliding page ratio -- and therefore the derived packing -- is TP
        invariant."""
        layout = _pack_gemma4(_gemma4_split_config(attn_tp_size=2))
        packing = dict(layout.group_packing)
        self.assertEqual(packing["full_attention"], 2)
        for k in range(5):
            self.assertEqual(packing[f"sliding_attention_{k}"], 1)


def _pack_gemma4_two_groups(config: AttnConfig):
    """Pack the two cache groups the runtime serves gemma-4 with (one sliding,
    one full) from the recipe's own fields."""
    spec = config.component(SoftmaxAttnConfig)
    layer_types = tuple(spec.cache_layer_types)
    groups = group_declarations(
        layer_types=layer_types,
        group_ids=layer_types,
        sliding_window_tokens=spec.sliding_window_tokens,
        prefix_granularity=config.prefix_granularity,
        fields_for_layer=lambda layer_id, group_id, occurrence: _config_layer_fields(
            config, layer_id=layer_id, local_layer_id=layer_id, occurrence=occurrence
        ),
        pd_disaggregation_enabled=False,
    )
    return pack(
        groups,
        prefix_granularity=config.prefix_granularity,
        cache_blocks_per_lcm_block=None,
        alignment=1,
        max_padding_fraction=1.0,
    )


class FlatPlaneTest(unittest.TestCase):
    """Both groups fill whole parents, and the spaced row views address the
    planned bytes.

    Ten sliding layers (16 x 256) and two full layers (4 x 512): a sliding
    page is 10 x 2 x 131072 B and a full page 2 x 2 x 65536 B, so a parent
    holds one sliding page or ten full pages and no byte of it is idle.
    """

    def _config(self):
        return _mha_config(
            layer_kv_geometry=_GEMMA4_GEOMETRY,
            num_kv_heads=16,
            head_dim=256,
            attn_tp_size=1,
            cache_layer_types=tuple(_LAYER_TYPES * 2),
        )

    def test_a_parent_is_one_sliding_page_or_ten_full_pages(self):
        layout = _pack_gemma4_two_groups(self._config())
        packing = dict(layout.group_packing)
        self.assertEqual(packing, {"full_attention": 10, "sliding_attention": 1})
        sliding_page = 10 * 2 * _PREFIX_GRANULARITY * 16 * 256 * 2
        self.assertEqual(layout.lcm_block_bytes, sliding_page)
        full_fields = [f for f in layout.fields if f.group_id == "full_attention"]
        self.assertTrue(
            all(f.page_stride_bytes == sliding_page // 10 for f in full_fields)
        )

    def test_spaced_rows_land_in_the_planned_page_bytes(self):
        if not torch.cuda.is_available():
            self.skipTest("set_kv_buffer runs a CUDA triton scatter")
        from cache_pool_test_utils import make_arena

        from tokenspeed.runtime.layers.attention.kv_cache.mha import (
            MHATokenToKVPool,
        )
        from tokenspeed.runtime.layers.paged_attention import PagedAttention

        plan = _pack_gemma4_two_groups(self._config()).bind(4)
        arena = make_arena(plan, "cuda")
        # Page stride over payload: 20 sliding payloads, 4 full payloads.
        self.assertEqual(arena.page_spacing("sliding_attention"), 20)
        self.assertEqual(arena.page_spacing("full_attention"), 4)
        pool = MHATokenToKVPool(
            arena=arena,
            dtype=torch.bfloat16,
            head_num=16,
            head_dim=256,
            layer_num=len(_GEMMA4_GEOMETRY),
            rank=0,
            layer_kv_head_counts=tuple(h for h, _ in _GEMMA4_GEOMETRY),
            kv_alloc_head_count=16,
            field_layer_offset=0,
        )
        n = _PREFIX_GRANULARITY
        written = {}
        # Write page 2 of every layer of both groups through its row view.
        for layer_id, (kv_heads, head_dim) in enumerate(_GEMMA4_GEOMETRY):
            gid = _LAYER_TYPES[layer_id % len(_LAYER_TYPES)]
            first_row = 2 * arena.page_spacing(gid) * n
            loc = torch.arange(first_row, first_row + n, device="cuda")
            k = torch.randn(n, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            layer = PagedAttention(
                num_heads=kv_heads,
                head_dim=head_dim,
                scaling=1.0,
                num_kv_heads=kv_heads,
                layer_id=layer_id,
                logit_cap=0.0,
                sliding_window_size=-1,
                rotary_emb=None,
                qk_norm=None,
            )
            pool.set_kv_buffer(layer, loc, k, v)
            written[f"layer.{layer_id}.k"] = k
            written[f"layer.{layer_id}.v"] = v
        torch.cuda.synchronize()
        raw = arena.buffer
        for field_id, rows in written.items():
            start = plan.field_page_byte_offset(field_id, 2)
            payload = rows.numel() * rows.element_size()
            stored = raw[start : start + payload].view(torch.bfloat16)
            # Bit-exact at the planned offset, so no write overlapped another.
            self.assertTrue(torch.equal(stored, rows.reshape(-1)), field_id)

    def test_the_page_table_expands_blocks_by_their_spacing(self):
        from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
            GroupTableSpec,
            GroupTableStacks,
        )

        stacks = GroupTableStacks(
            [
                GroupTableSpec("spaced", 16, 16, 4, page_spacing=20),
                GroupTableSpec("split", 16, 8, 8, page_spacing=4),
            ],
            max_bs=1,
            max_tokens_per_req=1,
            max_extend_tokens=0,
            device="cpu",
        )
        blocks = torch.tensor([[3, 0, 1]], dtype=torch.int32)
        stacks.fill(1, 1, {"spaced": blocks, "split": blocks})
        self.assertEqual(stacks.table("spaced", 1).tolist(), [[60, 0, 20, 0]])
        # Two kernel pages per block, spaced by four blocks' worth of pages.
        self.assertEqual(
            stacks.table("split", 1).tolist(), [[24, 25, 0, 0, 8, 9, 0, 0]]
        )
        self.assertEqual(
            stacks.host_table("split", blocks, 1).tolist(),
            [[24, 25, 0, 0, 8, 9]],
        )


class PoolRowViewWidthTest(unittest.TestCase):
    """The MHA pool serves each layer's KV plane at that layer's OWN head_dim.

    Regression for the TP=2 illegal memory access: the pool is built with one
    model-wide ``head_dim`` (gemma-4's sliding 256), but its full-attention
    planes are allocated at head_dim 512. ``_layer_row_view`` must read the
    width from each plane, not from the pool scalar -- otherwise it reshapes a
    512-wide full plane as 256-wide, halving the row stride, and
    ``store_kv_cache`` writes 1024-element K rows at a 512-element stride
    (overlapping, out-of-bounds).

    GPU: ``set_kv_buffer`` runs the triton scatter, so the round-trip actually
    exercises the write path the autotune prefill crashed in.
    """

    _PREFIX = 16
    # Two layers, TP=2 post-shard: sliding 8 x 256 (layer 0), full 2 x 512
    # (layer 1). The pool scalar head_dim is the sliding 256; the full plane
    # is the 512-wide one the old view mis-measured.
    _LAYER_TYPES = ("sliding_attention", "full_attention")
    _GEOMETRY_POST_TP = ((8, 256), (2, 512))

    def _build_pool(self, device: str):
        from cache_pool_test_utils import make_arena

        from tokenspeed.runtime.layers.attention.kv_cache.mha import (
            MHATokenToKVPool,
        )

        kv_dtype = torch.bfloat16
        # Group each layer at its own geometry, exactly as the recipe does,
        # scoping planes per group so the two geometries never share a plane.
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
            CacheFieldSpec,
            scatter_stored_dtype_name,
        )

        stored = scatter_stored_dtype_name(kv_dtype)

        def fields_for_layer(layer_id, group_id, occurrence):
            kv_heads, head_dim = self._GEOMETRY_POST_TP[layer_id]
            shape = (self._PREFIX, kv_heads, head_dim)
            prefix = f"unit.{group_id}"
            return (
                CacheFieldSpec(
                    f"layer.{layer_id}.k", f"{prefix}.{occurrence}.k", shape, stored
                ),
                CacheFieldSpec(
                    f"layer.{layer_id}.v", f"{prefix}.{occurrence}.v", shape, stored
                ),
            )

        groups = group_declarations(
            layer_types=self._LAYER_TYPES,
            group_ids=self._LAYER_TYPES,
            sliding_window_tokens=1024,
            prefix_granularity=self._PREFIX,
            fields_for_layer=fields_for_layer,
            pd_disaggregation_enabled=False,
        )
        layout = pack(
            groups,
            prefix_granularity=self._PREFIX,
            cache_blocks_per_lcm_block=None,
            alignment=1,
            max_padding_fraction=1.0,
        )
        plan = layout.bind(8)
        arena = make_arena(plan, device)
        # Pool scalar geometry is the SLIDING layer's (what MHAConfig publishes
        # as the model-wide head_dim / num_kv_heads); the per-layer counts are
        # PRE-TP (16 sliding, 4 full) and kv_alloc_head_count is the pre-TP max.
        pool = MHATokenToKVPool(
            arena=arena,
            dtype=kv_dtype,
            head_num=8,
            head_dim=256,
            layer_num=2,
            rank=0,
            layer_kv_head_counts=(16, 4),
            kv_alloc_head_count=16,
            field_layer_offset=0,
        )
        return pool

    def test_each_layer_view_keeps_its_own_head_dim(self):
        if not torch.cuda.is_available():
            self.skipTest("per-layer KV pool views require a CUDA arena")
        pool = self._build_pool("cuda")
        sliding_k = pool.get_key_buffer(0)
        full_k = pool.get_key_buffer(1)
        # Sliding plane served as-is: 8 heads x 256.
        self.assertEqual(sliding_k.shape[-2:], (8, 256))
        # Full plane served at its OWN 512 width, NOT reshaped to 256.
        self.assertEqual(full_k.shape[-2:], (2, 512))

    def test_full_layer_write_read_roundtrips_at_512(self):
        if not torch.cuda.is_available():
            self.skipTest("set_kv_buffer runs a CUDA triton scatter")
        from tokenspeed.runtime.layers.paged_attention import PagedAttention

        pool = self._build_pool("cuda")
        layer = PagedAttention(
            num_heads=16,
            head_dim=512,
            scaling=1.0,
            num_kv_heads=2,
            layer_id=1,
            logit_cap=0.0,
            sliding_window_size=-1,
            rotary_emb=None,
            qk_norm=None,
        )
        n = self._PREFIX
        k = torch.randn(n, 2, 512, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(n, 2, 512, device="cuda", dtype=torch.bfloat16)
        loc = torch.arange(n, device="cuda", dtype=torch.int64)
        pool.set_kv_buffer(layer, loc, k, v)
        torch.cuda.synchronize()
        k_cache = pool.get_key_buffer(1)
        v_cache = pool.get_value_buffer(1)
        # Every written slot comes back bit-exact at the true 512 width; a
        # half-width stride would have overlapped neighbouring rows.
        self.assertTrue(torch.equal(k_cache[loc].reshape(n, 2, 512), k))
        self.assertTrue(torch.equal(v_cache[loc].reshape(n, 2, 512), v))


if __name__ == "__main__":
    raise SystemExit(unittest.main())
