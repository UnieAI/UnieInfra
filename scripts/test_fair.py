#!/usr/bin/env python3
"""Unit tests for the fair-comparison model in scripts/fair.py.

The model decides every number the scoreboard reports, so its judgement calls are
tested directly: interpolation between our own cells, the tolerance tiers that
decide whether a comparison is "tight", "near" or too wide to call matched, the
lower bound when the reference never gets as interactive as we do, and the gate
that keeps a scheduling variant (mixed batch, non-default chunked prefill) out of
the frontier until it beats the plain cell at the same concurrency and cap.

Run: python3 scripts/test_fair.py
"""
import sys
import unittest

sys.path.insert(0, "/home/smc/workspace/roy/scripts")
import fair  # noqa: E402


def cell(conc, cap, rpm, inter, engine="unieinfra", variant=None, ep=False, dur=900):
    return {
        "conc": conc,
        "cap": cap,
        "rpm": rpm,
        "inter": inter,
        "engine": engine,
        "dur": dur,
        "variant": variant,
        "ep": ep,
        "quant": False,
        "valid": True,
        "noisy": False,
        "ttft90": 10.0,
        "outs": 100.0,
        "itl50": 1000.0 / inter,
        "itl90": 1000.0 / inter,
        "stamp": "20260101T000000Z",
    }


class FairModelTest(unittest.TestCase):
    def setUp(self):
        self._cells = fair.cells
        self._ref = fair.reference

    def tearDown(self):
        fair.cells = self._cells
        fair.reference = self._ref

    def use(self, cells, ref=None):
        fair.cells = lambda: list(cells)
        fair.reference = ref or (lambda: {})

    def test_interp_hits_a_cell_exactly(self):
        pts = [cell(16, 2, 21.7, 149.0), cell(16, 8, 60.0, 50.0)]
        rpm, status = fair.interp(pts, 50.0)
        self.assertEqual((rpm, status), (60.0, "exact"))

    def test_interp_is_log_spaced_between_two_cells(self):
        # Halfway in log-interactivity must land halfway in RPM, not on a lerp of
        # the raw interactivity: the tolerance tiers are defined in log space too.
        pts = [cell(16, 4, 40.0, 100.0), cell(16, 8, 80.0, 400.0)]
        rpm, status = fair.interp(pts, 200.0)
        self.assertEqual(status, "interp")
        self.assertAlmostEqual(rpm, 60.0, places=6)

    def test_interp_reports_which_side_the_missing_cell_is_on(self):
        # Our most interactive cell slower than the target => a smaller cap exists.
        slow = [cell(16, 8, 80.0, 40.0), cell(16, 16, 120.0, 20.0)]
        self.assertEqual(fair.interp(slow, 60.0), (None, "need-smaller-cap"))
        fast = [cell(16, 8, 80.0, 400.0), cell(16, 16, 120.0, 200.0)]
        self.assertEqual(fair.interp(fast, 100.0), (None, "need-bigger-cap"))
        self.assertEqual(fair.interp([], 100.0), (None, "no-cells"))

    def test_matched_calls_a_sub_two_percent_gap_tight(self):
        # c4: four users mean the cap cannot be tuned finer than four, so 145 vs
        # 146 must be reported as tight rather than silently reusing the cell.
        self.use([cell(4, 4, 8.3, 145.0)])
        fair.reference = lambda: {4: cell(4, None, 8.3, 146.0, engine="vllm")}
        (row,) = fair.matched()
        self.assertEqual(row["status"], "tight")
        self.assertAlmostEqual(row["ratio"], 1.0, places=6)

    def test_matched_prefers_a_real_interpolation_over_a_near_cell(self):
        # Two cells bracket the reference on both axes (bigger cap: more RPM, less
        # interactivity, so neither dominates); the interpolated value is what an
        # honest matched comparison uses.
        self.use([cell(16, 4, 20.0, 400.0), cell(16, 8, 80.0, 100.0)])
        fair.reference = lambda: {16: cell(16, None, 70.0, 200.0, engine="vllm")}
        (row,) = fair.matched()
        self.assertEqual(row["status"], "interp")
        self.assertAlmostEqual(row["rpm"], 50.0, places=6)  # 200 is log-midway in 100..400
        self.assertAlmostEqual(row["ratio"], 50.0 / 70.0, places=6)

    def test_floor_states_a_lower_bound_only_when_we_are_both_faster_and_further(self):
        # Reference tops out at 8.7 interactivity; every cell of ours is more
        # interactive than that, and the cap-ceiling one is faster in RPM too, so the
        # honest claim is a lower bound rather than a matched ratio.
        self.use([cell(128, 64, 127.9, 14.0)])
        fair.reference = lambda: {128: cell(128, None, 33.3, 8.7, engine="vllm")}
        (row,) = fair.matched()
        self.assertIsNone(row["ratio"])
        self.assertEqual(row["status"], "need-bigger-cap")
        b = fair.floor(row)
        self.assertIsNotNone(b)
        self.assertAlmostEqual(b["ratio"], 127.9 / 33.3, places=6)
        # If our cap-ceiling cell were the *slower* per-user one, no bound is claimable.
        self.use([cell(128, 64, 127.9, 5.0)])
        fair.reference = lambda: {128: cell(128, None, 33.3, 8.7, engine="vllm")}
        self.assertIsNone(fair.floor(fair.matched()[0]))

    def test_variant_must_beat_the_plain_cell_at_the_same_cap(self):
        base = cell(64, 16, 60.3, 49.0)
        better = cell(64, 16, 65.0, 55.0, variant="chunk2")
        worse = cell(64, 16, 70.0, 44.0, variant="chunk8")
        self.use([base, better, worse])
        verdicts = {v["cell"]["variant"]: v["accepted"] for v in fair.variants()}
        self.assertTrue(verdicts["chunk2"])
        self.assertFalse(verdicts["chunk8"])  # faster but less interactive: not a fair win
        front = fair.ours()[64]
        self.assertIn(better, front)
        self.assertNotIn(worse, front)

    def test_variant_without_a_plain_cell_cannot_be_accepted(self):
        lone = cell(64, 16, 90.0, 60.0, variant="mix")
        self.use([lone])
        self.assertFalse(fair.variants()[0]["accepted"])
        self.assertEqual(fair.ours().get(64, []), [])

    def test_frontier_drops_a_cell_dominated_at_the_same_concurrency(self):
        a = cell(64, 16, 60.0, 49.0)
        b = cell(64, 8, 32.4, 39.7)
        self.assertEqual(fair.frontier([a, b]), [a])

    def test_no_quantized_or_noisy_cell_is_ever_eligible(self):
        bad = cell(16, 4, 999.0, 150.0)
        bad["quant"] = True
        noisy = cell(16, 8, 998.0, 140.0)
        noisy["noisy"] = True
        good = cell(16, 4, 40.0, 100.0)
        self.use([bad, noisy, good])
        self.assertEqual(fair.eligible(), [good])


if __name__ == "__main__":
    unittest.main(verbosity=2)
