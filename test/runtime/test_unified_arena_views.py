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

"""Byte-level aliasing check for the two regions sharing one buffer.

Two views into the same allocation that overlap by one byte produce no error,
no slowdown, and no signal of any kind -- just a recurrent state that is
quietly someone else's. Three KV-transfer defects on this engine were found
only by a byte comparison. So the sharing gets one before anything is wired to
it, and it runs on CPU tensors, where `as_strided` behaves exactly as it does
on the device.
"""

import importlib.util
import pathlib
import sys
import types
import unittest

import torch

_KV_CACHE_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "layers"
    / "attention"
    / "kv_cache"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in (
    "tokenspeed",
    "tokenspeed.runtime",
    "tokenspeed.runtime.layers",
    "tokenspeed.runtime.layers.attention",
    "tokenspeed.runtime.layers.attention.kv_cache",
    "tokenspeed.runtime.layers.attention.kv_cache.recipes",
):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))

plan_mod = _load(
    "tokenspeed.runtime.layers.attention.kv_cache.recipes.plan",
    _KV_CACHE_DIR / "recipes" / "plan.py",
)

# base.py pulls in triton kernels and the paged-attention layer, none of which
# this helper touches. Import the one function by source rather than dragging
# the device stack into a CPU test.
_src = (_KV_CACHE_DIR / "base.py").read_text(encoding="utf-8")
_start = _src.index("def external_field_view(")
_end = _src.index("\nclass CachePool:")
_ns = {"torch": torch, "CacheMemoryPlan": plan_mod.CacheMemoryPlan}
exec(compile(_src[_start:_end], "base.py", "exec"), _ns)  # noqa: S102
external_field_view = _ns["external_field_view"]

LCM = 4096
NUM_PARENTS = 16
GROUPS = ("gdn_0", "gdn_1", "gdn_2")
ROW_ELEMS = 8  # float32 -> 32 bytes per group per slot


def _plan(max_state_slots=6):
    fields = [
        plan_mod.CacheFieldLayout(
            group_id="full_attention",
            field_id="full_attention.k",
            plane_id="attn",
            shape=(64,),
            element_size=2,
            field_offset_bytes=0,
            page_stride_bytes=LCM,
        )
    ]
    for group_id in GROUPS:
        fields.append(
            plan_mod.CacheFieldLayout(
                group_id=group_id,
                field_id=f"{group_id}.ssm",
                plane_id=group_id,
                shape=(ROW_ELEMS,),
                element_size=4,
                field_offset_bytes=0,
                page_stride_bytes=ROW_ELEMS * 4,
                external=True,
            )
        )
    return plan_mod.CacheMemoryPlan(
        logical_block_tokens=64,
        lcm_block_bytes=LCM,
        num_lcm_blocks=NUM_PARENTS,
        groups=(),
        planes=(),
        fields=tuple(fields),
        max_state_slots=max_state_slots,
        buffer_bytes=(NUM_PARENTS + 1) * LCM,
    )


class ExternalViewTest(unittest.TestCase):
    def setUp(self):
        self.plan = _plan()
        self.buffer = torch.zeros(self.plan.unified_buffer_bytes, dtype=torch.uint8)

    def _views(self):
        return {
            g: external_field_view(self.buffer, self.plan, f"{g}.ssm", torch.float32)
            for g in GROUPS
        }

    def test_shape_and_row_length(self):
        views = self._views()
        for view in views.values():
            self.assertEqual(tuple(view.shape), (self.plan.max_state_slots, ROW_ELEMS))

    def test_no_two_rows_alias(self):
        """Every (group, slot) row owns bytes no other row owns."""
        views = self._views()
        n = 0
        for group in GROUPS:
            for slot in range(self.plan.max_state_slots):
                n += 1
                views[group][slot] = float(n)
        seen = {}
        for group in GROUPS:
            for slot in range(self.plan.max_state_slots):
                value = float(views[group][slot][0])
                self.assertNotIn(
                    value, seen, f"{group} slot {slot} aliases {seen.get(value)}"
                )
                seen[value] = (group, slot)
        self.assertEqual(len(seen), len(GROUPS) * self.plan.max_state_slots)

    def test_writing_a_slot_leaves_the_parents_untouched(self):
        """The regions must not overlap for any boundary the plan accepts."""
        views = self._views()
        # Fill every parent byte the attention side may legally use.
        usable_parents = NUM_PARENTS
        while True:
            try:
                self.plan.check_boundary(usable_parents, self.plan.max_state_slots)
                break
            except ValueError:
                usable_parents -= 1
                self.assertGreater(usable_parents, 0, "no legal boundary at all")
        parent_bytes = (usable_parents + 1) * LCM
        self.buffer[:parent_bytes] = 0xAB

        for group in GROUPS:
            for slot in range(self.plan.max_state_slots):
                views[group][slot] = -1.0

        self.assertTrue(
            bool((self.buffer[:parent_bytes] == 0xAB).all()),
            "a state write reached into the attention region",
        )

    def test_a_parent_write_leaves_the_slots_untouched(self):
        views = self._views()
        for group in GROUPS:
            for slot in range(self.plan.max_state_slots):
                views[group][slot] = 7.0
        usable_parents = NUM_PARENTS
        while True:
            try:
                self.plan.check_boundary(usable_parents, self.plan.max_state_slots)
                break
            except ValueError:
                usable_parents -= 1
        self.buffer[: (usable_parents + 1) * LCM] = 0xCD
        for group in GROUPS:
            for slot in range(self.plan.max_state_slots):
                self.assertTrue(
                    bool((views[group][slot] == 7.0).all()),
                    f"a parent write reached {group} slot {slot}",
                )

    def test_the_last_slot_ends_exactly_at_the_buffer_end(self):
        views = self._views()
        last = views[GROUPS[-1]][self.plan.max_state_slots - 1]
        end = last.storage_offset() * 4 + last.numel() * 4
        self.assertEqual(end, self.plan.unified_buffer_bytes)

    def test_a_plan_without_a_state_region_is_refused(self):
        plan = _plan(max_state_slots=0)
        buffer = torch.zeros(plan.unified_buffer_bytes, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            external_field_view(buffer, plan, f"{GROUPS[0]}.ssm", torch.float32)

    def test_a_non_external_field_is_refused(self):
        with self.assertRaises(ValueError):
            external_field_view(
                self.buffer, self.plan, "full_attention.k", torch.float16
            )

    def test_a_dtype_that_disagrees_with_the_plan_is_refused(self):
        with self.assertRaises(ValueError):
            external_field_view(
                self.buffer, self.plan, f"{GROUPS[0]}.ssm", torch.float16
            )

    def test_a_short_buffer_is_refused(self):
        small = torch.zeros(self.plan.unified_buffer_bytes - 1, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            external_field_view(small, self.plan, f"{GROUPS[0]}.ssm", torch.float32)


if __name__ == "__main__":
    unittest.main()
