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

"""Page-zeroing byte ranges must be unchanged by the fast lookup path.

``CachePool._block_byte_segments`` expands one ``(byte_offset, size)`` range per
(block, field) pair, and DeepSeek-V4 expands ~80k pairs per prefill step. Each
pair used to resolve a field, its group and its plane through the memory plan,
whose accessors were linear scans over ~258 fields -- 726,502 calls per profile,
1.7us each for ``field()`` alone.

The plan accessors are now dictionary-backed and ``_block_byte_segments``
resolves each group's affine ``base + block * page_stride`` once instead of once
per pair. Both changes are supposed to be pure, so this module pins them:

* every accessor returns the same object a linear scan returned, first match
  wins when an id is duplicated, and unknown ids still raise ``KeyError``;
* the expanded ranges are byte-identical to the previous per-pair expansion;
* the bounds ``IndexError`` and the empty-group no-op are preserved.

These tests need no GPU.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_KV_CACHE_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "layers"
    / "attention"
    / "kv_cache"
)

# Mirrors the DeepSeek-V4 arena advertised at boot: 43 layers sharing one plane
# each, over these six groups.
_DSV4_GROUPS = (
    "v4.c128a.compressed_kv",
    "v4.c128a.compressor_state",
    "v4.c4a.compressed_kv",
    "v4.c4a.compressor_state",
    "v4.c4a.indexer_compressor_state",
    "v4.swa_kv",
)
_DSV4_LAYERS = 43


def _load_plan_module():
    """Load ``recipes/plan.py`` by path, the way the cache tests do.

    ``plan`` imports nothing from the package, so loading it directly keeps this
    module independent of the runtime import graph.
    """
    spec = importlib.util.spec_from_file_location(
        "_zero_seg_plan", _KV_CACHE_DIR / "recipes" / "plan.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations via sys.modules[cls.__module__].
    sys.modules["_zero_seg_plan"] = module
    spec.loader.exec_module(module)
    return module


def _build_plan(
    plan_module,
    *,
    groups=_DSV4_GROUPS,
    planes=_DSV4_LAYERS,
    fields_per_group=_DSV4_LAYERS,
    page_count=4096,
    field_groups=None,
):
    """A DeepSeek-V4-shaped plan: one plane per layer, fields in plan order.

    ``field_groups`` selects which groups carry fields, so a group with none can
    be built on purpose.
    """
    if field_groups is None:
        field_groups = groups
    plane_layouts = tuple(
        plan_module.CachePlaneLayout(
            plane_id=f"unit.{index}",
            bytes_per_lcm_block=32768,
            arena_offset_bytes=index * 32768,
        )
        for index in range(planes)
    )
    group_layouts = tuple(
        plan_module.CacheGroupLayout(
            group_id=group_id,
            cache_blocks_per_lcm_block=1,
            page_count=page_count,
        )
        for group_id in groups
    )
    fields = tuple(
        plan_module.CacheFieldLayout(
            group_id=group_id,
            field_id=f"{group_id}.layer.{layer}",
            plane_id=f"unit.{layer % planes}",
            shape=(64, 8),
            element_size=2,
            # Vary with the group as well as the layer: in a real arena every
            # group packs its own stride and field offset into the shared plane.
            field_offset_bytes=(group_index * 8 + layer % 8) * 1024,
            page_stride_bytes=1024 * (group_index * 8 + layer % 8 + 1),
        )
        for group_index, group_id in enumerate(field_groups)
        for layer in range(fields_per_group)
    )
    return plan_module.CacheMemoryPlan(
        logical_block_tokens=256,
        lcm_block_bytes=32768,
        num_lcm_blocks=4096,
        groups=group_layouts,
        planes=plane_layouts,
        fields=fields,
    )


def _pool_stub(pool_class, plan):
    """A real pool instance holding only what the segment builder touches.

    ``object.__new__`` skips the real ``__init__``, which allocates the arena and
    binds a device, but keeps ``self`` a genuine ``CachePool`` so the method under
    test can still reach its own helpers through the class.
    """
    pool = object.__new__(pool_class)
    pool.plan = plan
    pool._group_field_specs = {}
    return pool


def _linear_scan(items, attr, key):
    """The lookup the plan accessors used to perform."""
    for item in items:
        if getattr(item, attr) == key:
            return item
    raise KeyError(key)


def _reference_segments(pool_class, pool, group_id, block_ids):
    """The previous per-pair expansion, kept verbatim as the oracle."""
    pool.plan.group(group_id)
    fields = [field for field in pool.plan.fields if field.group_id == group_id]
    return [
        (
            pool_class._field_block_byte_offset(pool, field.field_id, block_id),
            field.payload_bytes,
        )
        for block_id in block_ids
        for field in fields
    ]


def _load_pool_base():
    from tokenspeed.runtime.layers.attention.kv_cache import base

    return base.CachePool


class CachePlanAccessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan_module = _load_plan_module()

    def test_accessors_return_what_a_linear_scan_returned(self):
        plan = _build_plan(self.plan_module)
        for field in plan.fields:
            self.assertIs(
                plan.field(field.field_id),
                _linear_scan(plan.fields, "field_id", field.field_id),
            )
            self.assertIs(
                plan.group(field.group_id),
                _linear_scan(plan.groups, "group_id", field.group_id),
            )
            self.assertIs(
                plan.plane(field.plane_id),
                _linear_scan(plan.planes, "plane_id", field.plane_id),
            )

    def test_accessor_keeps_the_first_of_duplicate_ids(self):
        plan = _build_plan(
            self.plan_module, groups=("g",), planes=1, fields_per_group=3
        )
        duplicate = self.plan_module.CacheFieldLayout(
            group_id="g",
            field_id="g.layer.0",  # same id as the first field
            plane_id="unit.0",
            shape=(1,),
            element_size=4,
            field_offset_bytes=999,
            page_stride_bytes=999,
        )
        plan = self.plan_module.CacheMemoryPlan(
            logical_block_tokens=256,
            lcm_block_bytes=plan.lcm_block_bytes,
            num_lcm_blocks=plan.num_lcm_blocks,
            groups=plan.groups,
            planes=plan.planes,
            fields=(plan.fields[0], duplicate, *plan.fields[1:]),
        )
        self.assertIs(plan.field("g.layer.0"), plan.fields[0])
        self.assertIsNot(plan.field("g.layer.0"), duplicate)

    def test_unknown_ids_still_raise_keyerror(self):
        plan = _build_plan(
            self.plan_module, groups=("g",), planes=1, fields_per_group=2
        )
        with self.assertRaises(KeyError):
            plan.field("nope")
        with self.assertRaises(KeyError):
            plan.group("nope")
        with self.assertRaises(KeyError):
            plan.plane("nope")

    def test_payload_bytes_is_unchanged_by_caching(self):
        plan = _build_plan(
            self.plan_module, groups=("g",), planes=1, fields_per_group=4
        )
        for field in plan.fields:
            expected = field.shape[0] * field.shape[1] * field.element_size
            self.assertEqual(field.payload_bytes, expected)
            # Cached: a second read must return the same value, not a stale copy
            # of a different field's geometry.
            self.assertEqual(field.payload_bytes, expected)

    def test_field_scan_is_not_rebuilt_per_call(self):
        plan = _build_plan(self.plan_module)
        self.assertIs(plan._fields_by_id, plan._fields_by_id)
        self.assertEqual(len(plan._fields_by_id), len(plan.fields))


class BlockByteSegmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan_module = _load_plan_module()
        cls.pool_base = _load_pool_base()

    def _segments(self, pool, group_id, block_ids):
        return self.pool_base._block_byte_segments(pool, group_id, block_ids)

    def test_segments_are_byte_identical_to_the_pair_expansion(self):
        plan = _build_plan(self.plan_module)
        pool = _pool_stub(self.pool_base, plan)
        for group_id in _DSV4_GROUPS:
            for block_ids in ([0], [1, 2, 3], list(range(0, 256, 7)), [4095]):
                self.assertEqual(
                    self._segments(pool, group_id, block_ids),
                    _reference_segments(self.pool_base, pool, group_id, block_ids),
                    f"{group_id} {block_ids[:3]}...",
                )

    def test_segments_are_unchanged_on_repeat_and_across_groups(self):
        plan = _build_plan(self.plan_module)
        pool = _pool_stub(self.pool_base, plan)
        first = self._segments(pool, _DSV4_GROUPS[0], [0, 1, 2])
        self.assertEqual(self._segments(pool, _DSV4_GROUPS[0], [0, 1, 2]), first)
        # A second group must not be served the first group's cached specs.
        self.assertNotEqual(self._segments(pool, _DSV4_GROUPS[5], [0, 1, 2]), first)
        self.assertEqual(
            self._segments(pool, _DSV4_GROUPS[5], [0, 1, 2]),
            _reference_segments(self.pool_base, pool, _DSV4_GROUPS[5], [0, 1, 2]),
        )

    def test_segments_cover_disjoint_blocks_without_overlap(self):
        """Distinct blocks of one field must not alias once the affine form is used."""
        plan = _build_plan(
            self.plan_module, groups=("g",), planes=1, fields_per_group=1
        )
        pool = _pool_stub(self.pool_base, plan)
        segments = dict(self._segments(pool, "g", [0, 1, 2, 3]))
        self.assertEqual(len(segments), 4)

    def test_out_of_range_block_raises_indexerror(self):
        plan = _build_plan(
            self.plan_module, groups=("g",), planes=1, fields_per_group=2, page_count=4
        )
        pool = _pool_stub(self.pool_base, plan)
        for bad in (-1, 4, 99):
            with self.assertRaises(IndexError):
                self._segments(pool, "g", [bad])

    def test_empty_group_yields_no_segments(self):
        """A group with no fields produced no segments before, bounds check included."""
        plan = _build_plan(
            self.plan_module,
            groups=("g", "empty"),
            field_groups=("g",),
            planes=1,
            fields_per_group=2,
        )
        pool = _pool_stub(self.pool_base, plan)
        self.assertEqual(self._segments(pool, "empty", [0, 10_000]), [])
        self.assertEqual(self._segments(pool, "empty", []), [])

    def test_empty_block_list_yields_no_segments(self):
        plan = _build_plan(
            self.plan_module, groups=("g",), planes=1, fields_per_group=2
        )
        pool = _pool_stub(self.pool_base, plan)
        self.assertEqual(self._segments(pool, "g", []), [])

    def test_unknown_group_raises_keyerror(self):
        plan = _build_plan(
            self.plan_module, groups=("g",), planes=1, fields_per_group=2
        )
        pool = _pool_stub(self.pool_base, plan)
        with self.assertRaises(KeyError):
            self._segments(pool, "nope", [0])


if __name__ == "__main__":
    unittest.main()
