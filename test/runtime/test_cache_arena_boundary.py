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

"""Byte geometry of the movable boundary between the two cache regions.

The attention parents and the recurrent-state slots share one physical buffer:
parents grow up from offset 0, slots grow down from the end, and the boundary
between them is a pair of quotas rather than a resize. That arrangement exists
for one reason -- a CUDA graph bakes a tensor's ``data_ptr()`` into its kernel
arguments at capture time and never re-reads it, so any layout that relocated a
page when the boundary moved would leave every replay reading freed memory, and
a wrong recurrent state produces no latency signal whatsoever.

So the properties under test are addresses, not sizes: the same page must land
at the same byte offset no matter where the boundary currently sits, and the
one inequality that keeps the regions apart must reject the states it exists to
prevent -- with a test that fails without it.
"""

import importlib.util
import pathlib
import sys
import types
import unittest

_KV_CACHE_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "layers"
    / "attention"
    / "kv_cache"
)


def _load_plan():
    for package_name in (
        "tokenspeed",
        "tokenspeed.runtime",
        "tokenspeed.runtime.layers",
        "tokenspeed.runtime.layers.attention",
        "tokenspeed.runtime.layers.attention.kv_cache",
        "tokenspeed.runtime.layers.attention.kv_cache.recipes",
    ):
        sys.modules.setdefault(package_name, types.ModuleType(package_name))
    name = "tokenspeed.runtime.layers.attention.kv_cache.recipes.plan"
    spec = importlib.util.spec_from_file_location(
        name, _KV_CACHE_DIR / "recipes" / "plan.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


plan_mod = _load_plan()

LCM_BLOCK_BYTES = 8 << 20
NUM_PARENTS = 32
# Three recurrent groups, conv + ssm each -- the shape Qwen3.8's GDN stack has,
# at a size a test can reason about.
_GROUPS = ("gdn_0", "gdn_1", "gdn_2")
_CONV_BYTES = 60 << 10
_SSM_BYTES = 3 << 20


def _fields():
    fields = [
        plan_mod.CacheFieldLayout(
            group_id="full_attention",
            field_id="full_attention.k",
            plane_id="attn",
            shape=(64, 8, 128),
            element_size=2,
            field_offset_bytes=0,
            page_stride_bytes=1 << 20,
        )
    ]
    for group_id in _GROUPS:
        fields.append(
            plan_mod.CacheFieldLayout(
                group_id=group_id,
                field_id=f"{group_id}.conv",
                plane_id=group_id,
                shape=(_CONV_BYTES // 2,),
                element_size=2,
                field_offset_bytes=0,
                page_stride_bytes=_CONV_BYTES,
                external=True,
            )
        )
        fields.append(
            plan_mod.CacheFieldLayout(
                group_id=group_id,
                field_id=f"{group_id}.ssm",
                plane_id=group_id,
                shape=(_SSM_BYTES // 4,),
                element_size=4,
                field_offset_bytes=0,
                page_stride_bytes=_SSM_BYTES,
                external=True,
            )
        )
    return tuple(fields)


def _plan(with_state=True, num_parents=NUM_PARENTS):
    """A buffer both regions can fill alone -- so they genuinely trade.

    Sizing it for the two maxima added together would make the boundary
    unmovable: neither side would ever hold a byte the other could take.
    """
    buffer_bytes = (num_parents + 1) * LCM_BLOCK_BYTES
    slot_bytes = len(_GROUPS) * (_CONV_BYTES + _SSM_BYTES)
    if not with_state:
        return plan_mod.CacheMemoryPlan(
            logical_block_tokens=64,
            lcm_block_bytes=LCM_BLOCK_BYTES,
            num_lcm_blocks=num_parents,
            groups=(),
            planes=(),
            fields=_fields(),
        )
    return plan_mod.CacheMemoryPlan(
        logical_block_tokens=64,
        lcm_block_bytes=LCM_BLOCK_BYTES,
        num_lcm_blocks=num_parents,
        groups=(),
        planes=(),
        fields=_fields(),
        max_state_slots=buffer_bytes // slot_bytes,
        buffer_bytes=buffer_bytes,
    )


class StateSlotGeometryTest(unittest.TestCase):
    def test_slot_bytes_counts_every_external_field_once(self):
        p = _plan()
        self.assertEqual(p.state_slot_bytes, len(_GROUPS) * (_CONV_BYTES + _SSM_BYTES))

    def test_the_two_regions_together_overcommit_the_buffer(self):
        p = _plan()
        # 兩邊各自都填得滿整塊，加起來遠超過它 —— 這才有得交換
        self.assertEqual(p.unified_buffer_bytes, p.arena_bytes)
        self.assertGreater(
            p.arena_bytes + p.max_state_slots * p.state_slot_bytes,
            p.unified_buffer_bytes,
        )

    def test_the_last_row_sits_against_the_end(self):
        """Rows ascend with the address; the pool is what runs backwards.

        torch has no negative stride, so a region indexed downward could not be
        expressed as an ``as_strided`` view at all. The occupied rows end up at
        the buffer's end because the state pool hands out its highest ids
        first, not because the layout is reversed.
        """
        p = _plan()
        last = p.state_field_offset_bytes(f"{_GROUPS[0]}.conv", p.max_state_slots - 1)
        self.assertEqual(last, p.unified_buffer_bytes - p.state_slot_bytes)

    def test_row_zero_is_the_first_row_after_the_gap(self):
        p = _plan()
        first = p.state_field_offset_bytes(f"{_GROUPS[0]}.conv", 0)
        self.assertEqual(
            first,
            p.unified_buffer_bytes - p.max_state_slots * p.state_slot_bytes,
        )

    def test_rows_ascend_by_exactly_one_slot(self):
        p = _plan()
        for slot in range(p.max_state_slots - 1):
            here = p.state_field_offset_bytes(f"{_GROUPS[0]}.conv", slot)
            above = p.state_field_offset_bytes(f"{_GROUPS[0]}.conv", slot + 1)
            self.assertEqual(above - here, p.state_slot_bytes)

    def test_the_free_rows_are_the_ones_next_to_the_parents(self):
        """The gap a moved boundary reaches has to be the LOW rows.

        With S slots in use the pool holds the top S, so rows 0..max-S-1 are
        free and they sit immediately above the parents. If the free rows were
        at the buffer end instead, the attention region could never reach them
        and the boundary would be decorative.
        """
        p = _plan()
        in_use = 4
        lowest_used = p.state_field_offset_bytes(
            f"{_GROUPS[0]}.conv", p.max_state_slots - in_use
        )
        self.assertEqual(
            lowest_used,
            p.unified_buffer_bytes - in_use * p.state_slot_bytes,
        )
        p.check_boundary((lowest_used // p.lcm_block_bytes) - 1, in_use)

    def test_one_slot_is_contiguous_across_its_fields(self):
        """Every group's page for a slot sits together, with no hole.

        Interleaving by slot is what keeps the free bytes in one contiguous gap
        beside the parents; stacking whole groups would strand a gap inside the
        stack where no parent could ever use it.
        """
        p = _plan()
        external = [f for f in p.fields if f.external]
        offsets = [
            (p.state_field_offset_bytes(f.field_id, 3), f.payload_bytes)
            for f in external
        ]
        offsets.sort()
        for (off, size), (nxt, _) in zip(offsets, offsets[1:]):
            self.assertEqual(off + size, nxt)
        first_off, _ = offsets[0]
        last_off, last_size = offsets[-1]
        self.assertEqual(last_off + last_size - first_off, p.state_slot_bytes)

    def test_addresses_do_not_depend_on_where_the_boundary_sits(self):
        """The reason the whole layout exists: a captured pointer stays valid.

        The plan is sized for the maximum, so the offsets are a function of the
        slot alone. Nothing here consults a current quota, and that is the
        property a moved boundary must not break.
        """
        p = _plan()
        want = {
            (f.field_id, slot): p.state_field_offset_bytes(f.field_id, slot)
            for f in p.fields
            if f.external
            for slot in range(p.max_state_slots)
        }
        for parents in (1, 7, NUM_PARENTS):
            for slots in (0, 1, p.max_state_slots):
                try:
                    p.check_boundary(parents, slots)
                except ValueError:
                    continue
                for (field_id, slot), offset in want.items():
                    self.assertEqual(p.state_field_offset_bytes(field_id, slot), offset)

    def test_a_plan_without_a_state_region_keeps_the_old_geometry(self):
        p = _plan(with_state=False)
        self.assertEqual(p.unified_buffer_bytes, p.arena_bytes)
        with self.assertRaises(ValueError):
            p.state_field_offset_bytes(f"{_GROUPS[0]}.conv", 0)

    def test_a_slot_outside_the_sized_region_is_refused(self):
        p = _plan()
        with self.assertRaises(ValueError):
            p.state_field_offset_bytes(f"{_GROUPS[0]}.conv", p.max_state_slots)
        with self.assertRaises(ValueError):
            p.state_field_offset_bytes(f"{_GROUPS[0]}.conv", -1)

    def test_an_unknown_field_is_not_silently_given_an_offset(self):
        p = _plan()
        with self.assertRaises(KeyError):
            p.state_field_offset_bytes("full_attention.k", 0)


class BoundaryInvariantTest(unittest.TestCase):
    def test_every_legal_boundary_keeps_the_regions_apart(self):
        p = _plan()
        legal = 0
        for parents in range(p.num_lcm_blocks + 1):
            for slots in range(p.max_state_slots + 1):
                parent_end = (parents + 1) * p.lcm_block_bytes
                state_start = p.unified_buffer_bytes - slots * p.state_slot_bytes
                if parent_end > state_start:
                    with self.assertRaises(ValueError):
                        p.check_boundary(parents, slots)
                else:
                    p.check_boundary(parents, slots)
                    legal += 1
        self.assertGreater(legal, 0, "the sweep never exercised a legal pair")

    def test_an_overlapping_boundary_is_rejected(self):
        """Fails without the check -- which is the point of having it.

        Sized so the parents alone fill the buffer, leaving nothing for a
        single slot.
        """
        p = _plan()
        with self.assertRaises(ValueError) as caught:
            p.check_boundary(p.num_lcm_blocks, p.max_state_slots)
        self.assertIn("overlaps", str(caught.exception))

    def test_quotas_outside_their_pools_are_refused(self):
        p = _plan()
        with self.assertRaises(ValueError):
            p.check_boundary(p.num_lcm_blocks + 1, 0)
        with self.assertRaises(ValueError):
            p.check_boundary(-1, 0)
        with self.assertRaises(ValueError):
            p.check_boundary(1, p.max_state_slots + 1)
        with self.assertRaises(ValueError):
            p.check_boundary(1, -1)


if __name__ == "__main__":
    unittest.main()
