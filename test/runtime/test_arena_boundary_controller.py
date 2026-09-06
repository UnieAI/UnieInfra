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

"""What the boundary controller must not do.

A step is expensive -- one slot is 2,240 tokens of prefix cache on the measured
Qwen3.8-27B geometry -- so the interesting properties are all about restraint:
it must not move when nothing is starved, must not move when *both* sides are,
must not oscillate, and must never propose quotas the geometry cannot hold.
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_modules():
    for package_name in (
        "tokenspeed",
        "tokenspeed.runtime",
        "tokenspeed.runtime.layers",
        "tokenspeed.runtime.layers.attention",
        "tokenspeed.runtime.layers.attention.kv_cache",
        "tokenspeed.runtime.layers.attention.kv_cache.recipes",
    ):
        sys.modules.setdefault(package_name, types.ModuleType(package_name))
    plan = _load(
        "tokenspeed.runtime.layers.attention.kv_cache.recipes.plan",
        _KV_CACHE_DIR / "recipes" / "plan.py",
    )
    boundary = _load(
        "tokenspeed.runtime.layers.attention.kv_cache.arena_boundary",
        _KV_CACHE_DIR / "arena_boundary.py",
    )
    return plan, boundary


plan_mod, boundary_mod = _load_modules()

# The measured Qwen3.8-27B two-arena geometry, scaled down so a test can hold
# it in its head: 8 MiB parents, three recurrent groups whose page is 48.94 MiB
# each, so one slot is 146.8 MiB and worth 17 parents (17.5, floored -- a step
# must not promise bytes the buffer does not have).
LCM = 8 << 20
SLOT_GROUPS = 3
GROUP_PAGE = 48 << 20
PARENTS_PER_SLOT = (SLOT_GROUPS * GROUP_PAGE) // LCM
NUM_PARENTS = 400
MAX_SLOTS = 20


def _plan():
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
    for i in range(SLOT_GROUPS):
        fields.append(
            plan_mod.CacheFieldLayout(
                group_id=f"gdn_{i}",
                field_id=f"gdn_{i}.ssm",
                plane_id=f"gdn_{i}",
                shape=(GROUP_PAGE // 4,),
                element_size=4,
                field_offset_bytes=0,
                page_stride_bytes=GROUP_PAGE,
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
        max_state_slots=MAX_SLOTS,
        buffer_bytes=(NUM_PARENTS + 1) * LCM,
    )


def _controller(**kw):
    kw.setdefault("trigger", 100)
    kw.setdefault("cooldown", 2)
    return boundary_mod.BoundaryController(_plan(), PARENTS_PER_SLOT, **kw)


class RestraintTest(unittest.TestCase):
    def test_quiet_windows_move_nothing(self):
        c = _controller()
        self.assertIsNone(c.propose(200, 10, 0, 0))

    def test_shortage_below_the_trigger_is_noise(self):
        c = _controller(trigger=100)
        self.assertIsNone(c.propose(200, 10, 99, 0))

    def test_both_sides_starved_moves_nothing(self):
        """Robbing a starved side to feed a starved side helps nobody.

        This is the memory-is-simply-short case, and it is the one where a
        naive "whoever is shorter wins" rule would thrash hardest.
        """
        c = _controller()
        self.assertIsNone(c.propose(200, 10, 5000, 5000))

    def test_it_will_not_step_back_within_the_cooldown(self):
        c = _controller(cooldown=2)
        first = c.propose(200, 10, 1000, 0)
        self.assertIsNotNone(first)
        self.assertIsNone(c.propose(first.parents, first.slots, 0, 1000))
        self.assertIsNone(c.propose(first.parents, first.slots, 0, 1000))
        third = c.propose(first.parents, first.slots, 0, 1000)
        self.assertIsNotNone(third)

    def test_alternating_pressure_does_not_oscillate_every_window(self):
        c = _controller(cooldown=3)
        parents, slots = 200, 10
        moves = 0
        for window in range(20):
            attention, state = (1000, 0) if window % 2 else (0, 1000)
            step = c.propose(parents, slots, attention, state)
            if step is not None:
                parents, slots = step.parents, step.slots
                moves += 1
        self.assertLessEqual(moves, 20 // (3 + 1) + 1)


class DirectionTest(unittest.TestCase):
    def test_attention_pressure_takes_a_slot(self):
        c = _controller()
        step = c.propose(200, 10, 1000, 0)
        self.assertEqual(step.slots, 9)
        self.assertEqual(step.parents, 200 + PARENTS_PER_SLOT)
        self.assertIn("attention", step.reason)

    def test_state_pressure_gives_a_slot(self):
        c = _controller()
        step = c.propose(200, 10, 0, 1000)
        self.assertEqual(step.slots, 11)
        self.assertEqual(step.parents, 200 - PARENTS_PER_SLOT)
        self.assertIn("recurrent-state", step.reason)

    def test_one_step_at_a_time_however_large_the_shortage(self):
        c = _controller(cooldown=0)
        step = c.propose(200, 10, 10**6, 0)
        self.assertEqual(step.slots, 9)


class FloorTest(unittest.TestCase):
    def test_the_state_region_is_not_emptied(self):
        """Zero slots retains no snapshot at all, so every hybrid hit dies."""
        c = _controller(min_slots=2, cooldown=0)
        self.assertIsNone(c.propose(200, 2, 10**6, 0))

    def test_the_attention_region_keeps_its_floor(self):
        c = _controller(min_parents=100, cooldown=0)
        self.assertIsNone(c.propose(100 + PARENTS_PER_SLOT - 1, 5, 0, 10**6))

    def test_it_will_not_exceed_the_sized_state_region(self):
        c = _controller(cooldown=0)
        self.assertIsNone(c.propose(50, MAX_SLOTS, 0, 10**6))

    def test_every_proposal_is_representable(self):
        """Nothing that fails check_boundary ever leaves the controller."""
        plan = _plan()
        c = _controller(cooldown=0)
        for parents in range(1, NUM_PARENTS, 37):
            for slots in range(0, MAX_SLOTS + 1, 3):
                try:
                    plan.check_boundary(parents, slots)
                except ValueError:
                    continue
                for attention, state in ((10**6, 0), (0, 10**6)):
                    step = c.propose(parents, slots, attention, state)
                    if step is None:
                        continue
                    plan.check_boundary(step.parents, step.slots)


class ConstructionTest(unittest.TestCase):
    def test_a_slot_worth_no_parents_is_refused(self):
        with self.assertRaises(ValueError):
            boundary_mod.BoundaryController(_plan(), 0)

    def test_a_trigger_of_zero_is_refused(self):
        with self.assertRaises(ValueError):
            boundary_mod.BoundaryController(_plan(), PARENTS_PER_SLOT, trigger=0)


if __name__ == "__main__":
    unittest.main()
