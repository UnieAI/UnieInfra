"""Candidate memory.

KDA's evidence-records convention: `candidates.jsonl`, one record per candidate,
carrying parent links and status so a future reader can reconstruct what was
tried and why it was kept or dropped.

Two design choices worth stating, both from experience rather than taste:

**Rejections are stored, with reasons.** Of this session's largest findings,
three were rejected on re-examination, and the reasons -- a non-converged arm, a
chunk-vs-token miscount, correctness judged against expectation -- turned out to
be more valuable than the original claims. A memory that only keeps winners
teaches nothing about how the losses happened, and the same loss recurs.

**A rejected MEASUREMENT is not a rejected CANDIDATE.** They are different
statuses here. Conflating them is how a good candidate gets discarded because
the harness was misconfigured that afternoon.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Candidate:
    """One point in the search space, with its evidence."""

    id: str
    strategy: str                     # which strategy proposed it
    parent: str | None = None         # lineage, for evolve/refine chains
    description: str = ""

    # what it is
    target: str = ""                  # kernel / subsystem being changed
    change: str = ""                  # one sentence: what differs from parent

    # what happened
    status: str = "proposed"          # proposed | built | measured |
    #                                   promoted | rejected | unmeasurable
    reason: str = ""                  # REQUIRED when status is not promoted

    # evidence
    metric: float | None = None       # the target metric, e.g. tok/s
    baseline: float | None = None
    delta: float | None = None        # fractional, vs baseline
    sigma: float | None = None
    gates: dict[str, bool] = field(default_factory=dict)

    # attribution -- a number without these is a different claim, not a weaker
    # one. See gate_attribution.
    model: str | None = None
    concurrency: int | None = None
    shape: str | None = None

    # profiler evidence backing the diagnosis, if any
    profile: str | None = None        # path under profile/
    diagnosis: str = ""               # e.g. "bandwidth-bound, 78% of peak"

    extra: dict[str, Any] = field(default_factory=dict)

    #: Statuses that close a candidate negatively. These require a reason;
    #: ``promoted`` does not, because its justification is the evidence fields.
    NEGATIVE = ("rejected", "unmeasurable")

    def __post_init__(self) -> None:
        if self.status in self.NEGATIVE and not self.reason:
            raise ValueError(
                f"candidate {self.id}: status={self.status} requires a reason. "
                "Silent rejection is how the same mistake gets made twice."
            )


class CandidateLog:
    """Append-only JSONL store."""

    def __init__(self, path: str = "candidates.jsonl") -> None:
        self.path = path

    def append(self, c: Candidate) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    def all(self) -> list[Candidate]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(Candidate(**json.loads(line)))
        return out

    def promoted(self) -> list[Candidate]:
        return [c for c in self.all() if c.status == "promoted"]

    def yield_per_strategy(self) -> dict[str, dict[str, float]]:
        """Admissible yield per strategy, for the bandit.

        A measurement rejected by a gate counts as cost with no reward. That is
        deliberate: a strategy whose candidates are hard to measure reliably is
        genuinely worth less than its raw hit rate suggests, and the accounting
        should say so rather than flattering it.
        """
        acc: dict[str, dict[str, float]] = {}
        for c in self.all():
            s = acc.setdefault(c.strategy, {"trials": 0, "promoted": 0,
                                            "unmeasurable": 0, "gain": 0.0})
            s["trials"] += 1
            if c.status == "promoted":
                s["promoted"] += 1
                s["gain"] += (c.delta or 0.0)
            elif c.status == "unmeasurable":
                s["unmeasurable"] += 1
        for s in acc.values():
            s["hit_rate"] = s["promoted"] / s["trials"] if s["trials"] else 0.0
            s["mean_gain"] = s["gain"] / s["trials"] if s["trials"] else 0.0
        return acc
