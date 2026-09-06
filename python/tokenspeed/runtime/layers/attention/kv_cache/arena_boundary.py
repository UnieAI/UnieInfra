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

"""Where to put the boundary between the attention and recurrent-state regions.

There is no right constant. Measured on Qwen3.8-27B, one RTX PRO 6000: the
agent shape wants 112 state slots and reads 0.55x at 180; the RAG shape wants
180 and reads 0.69x at 112. The two regions are also very nearly the same size
-- 16.38 GiB of parents against 16.20 GiB of slots out of a 32.58 GiB budget --
so the boundary is not a trim, it decides half the cache.

The exchange rate is steep and that is what makes a policy necessary rather
than obvious: one slot is 146.8 MiB, which is 17.5 parents, which is 2,240
tokens of prefix cache. A single step is a real amount of cache, so a
controller that chases every transient shortage would spend the cache moving it
around. Hence: move only on sustained pressure, one step at a time, and never
back within the same cooldown.

This module decides *whether and where*; it does not move anything. Applying a
proposal is `CacheMemoryPlan.check_boundary` followed by `SetUsableParents` on
the attention pool and on each state pool, and the give-up side must report
`DrainedAboveUsable` first.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BoundaryProposal", "BoundaryController"]


@dataclass(frozen=True)
class BoundaryProposal:
    """One step. ``parents``/``slots`` are the quotas to install."""

    parents: int
    slots: int
    reason: str


class BoundaryController:
    """Turns per-arena admission shortages into at most one step at a time.

    Args:
        plan: The ``CacheMemoryPlan`` whose ``check_boundary`` defines what is
            representable. Every proposal is validated against it, so an
            unrepresentable step is never returned rather than being rejected
            downstream.
        parents_per_slot: Parents one slot is worth, from the byte geometry.
        min_slots: Floor for the state region. Zero slots means no snapshot can
            be retained at all, which costs every hybrid prefix hit.
        min_parents: Floor for the attention region.
        trigger: Shortage count within one window that counts as pressure.
            Admission tallies refusals per arena; a handful is noise, and the
            measured runs show thousands when a side is genuinely starved
            (1,345 on one arm, 1,435 on the other).
        cooldown: Windows to wait after a move before considering another. A
            move only pays off once the freed pages have been reused, which
            takes at least one working set's worth of traffic.
    """

    def __init__(
        self,
        plan,
        parents_per_slot: int,
        *,
        min_slots: int = 1,
        min_parents: int = 1,
        trigger: int = 100,
        cooldown: int = 2,
    ) -> None:
        if parents_per_slot < 1:
            raise ValueError("a slot must be worth at least one parent")
        if min_slots < 0 or min_parents < 1:
            raise ValueError("floors must be non-negative, parents at least 1")
        if trigger < 1:
            raise ValueError("trigger must be at least one shortage")
        if cooldown < 0:
            raise ValueError("cooldown cannot be negative")
        self.plan = plan
        self.parents_per_slot = parents_per_slot
        self.min_slots = min_slots
        self.min_parents = min_parents
        self.trigger = trigger
        self.cooldown = cooldown
        self._wait = 0

    def propose(
        self,
        parents: int,
        slots: int,
        attention_shortage: int,
        state_shortage: int,
    ) -> BoundaryProposal | None:
        """Decide one step, or ``None`` to leave the boundary alone.

        Args:
            parents: Parents the attention pool may currently hand out.
            slots: Slots each state pool may currently hand out.
            attention_shortage: Admission refusals charged to the attention
                arena in this window.
            state_shortage: Refusals charged to the recurrent-state arenas.

        Returns:
            The quotas to install, or ``None``.
        """
        if self._wait > 0:
            self._wait -= 1
            return None

        attention_short = attention_shortage >= self.trigger
        state_short = state_shortage >= self.trigger
        if attention_short == state_short:
            # Neither side is under pressure, or both are: moving bytes from
            # one starved side to the other starved side helps nobody, and the
            # engine is simply short of memory. Say nothing.
            return None

        if attention_short:
            step = self._give_to_attention(parents, slots)
        else:
            step = self._give_to_state(parents, slots)
        if step is not None:
            self._wait = self.cooldown
        return step

    def _give_to_attention(self, parents, slots):
        if slots - 1 < self.min_slots:
            return None
        want_slots = slots - 1
        want_parents = parents + self.parents_per_slot
        want_parents = min(want_parents, self.plan.num_lcm_blocks)
        try:
            self.plan.check_boundary(want_parents, want_slots)
        except ValueError:
            return None
        return BoundaryProposal(
            want_parents,
            want_slots,
            f"attention arena short; one slot ({self.parents_per_slot} parents) moved to it",
        )

    def _give_to_state(self, parents, slots):
        if slots + 1 > self.plan.max_state_slots:
            return None
        want_slots = slots + 1
        want_parents = parents - self.parents_per_slot
        if want_parents < self.min_parents:
            return None
        try:
            self.plan.check_boundary(want_parents, want_slots)
        except ValueError:
            return None
        return BoundaryProposal(
            want_parents,
            want_slots,
            f"recurrent-state arena short; {self.parents_per_slot} parents moved to one slot",
        )
