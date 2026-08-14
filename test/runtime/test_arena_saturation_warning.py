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

"""Tests for the KV-arena saturation warning.

A saturated arena is invisible in every existing signal: requests still succeed,
output is still correct, and only aggregate throughput moves. Measured on
Qwen3.5-27B RAG at 128 concurrent, saturation cost 38% of throughput against the
same engine capped just below the knee. These tests pin the warning's trigger,
its one-shot behaviour, and that it recovers if occupancy falls back.
"""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import unittest
from unittest.mock import patch

from tokenspeed.runtime.execution.model_executor import ModelExecutor


class _Probe:
    """Bare carrier for the warning state, avoiding ModelExecutor construction."""

    def __init__(self):
        self._saturated_decode_steps = 0
        self._warned_arena_saturated = False

    _ARENA_SATURATION_RATIO = ModelExecutor._ARENA_SATURATION_RATIO
    _ARENA_SATURATION_STEPS = ModelExecutor._ARENA_SATURATION_STEPS
    _warn_if_arena_saturated = ModelExecutor._warn_if_arena_saturated


class TestArenaSaturationWarning(unittest.TestCase):
    def setUp(self):
        self.probe = _Probe()

    def _feed(self, ratios, running=100):
        with patch("tokenspeed.runtime.execution.model_executor.logger") as mock_logger:
            for ratio in ratios:
                self.probe._warn_if_arena_saturated(ratio, running)
            return mock_logger.warning.call_args_list

    def test_silent_below_threshold(self):
        self.assertEqual(self._feed([0.89] * 10), [])

    def test_silent_until_sustained(self):
        """One saturated step is a spike, not a regime."""
        n = ModelExecutor._ARENA_SATURATION_STEPS
        self.assertEqual(self._feed([1.0] * (n - 1)), [])

    def test_warns_once_sustained(self):
        n = ModelExecutor._ARENA_SATURATION_STEPS
        calls = self._feed([1.0] * (n + 5))
        self.assertEqual(len(calls), 1)

    def test_names_the_knobs_without_inventing_a_cap(self):
        """Two earlier versions computed a recommended --max-num-seqs.

        Both were wrong. Saturation is first reached during ramp-up, when long
        prompts already fill the arena while only a handful of requests have
        reached decode: the batch size at that instant suggested 26 on a run
        whose measured optimum was 88, and the largest batch carried while
        unsaturated was 1. The regime is worth reporting; a cap derived from it
        is not, so the message must not contain a running-request count.
        """
        n = ModelExecutor._ARENA_SATURATION_STEPS
        self._feed([0.90], running=88)
        calls = self._feed([1.0] * n, running=29)
        self.assertEqual(len(calls), 1)
        template, *args = calls[0].args
        self.assertIn("--max-num-seqs", template)
        self.assertIn("--gpu-memory-utilization", template)
        self.assertNotIn(29, args)
        self.assertNotIn(88, args)

    def test_streak_resets_when_occupancy_recovers(self):
        """A dip below the threshold means the regime ended; do not accumulate."""
        n = ModelExecutor._ARENA_SATURATION_STEPS
        self.assertEqual(self._feed(([1.0] * (n - 1) + [0.5]) * 3), [])

    def test_fires_when_saturated_from_the_first_step(self):
        """The ramp never has an unsaturated window on some workloads."""
        n = ModelExecutor._ARENA_SATURATION_STEPS
        self.assertEqual(len(self._feed([1.0] * n, running=40)), 1)


if __name__ == "__main__":
    unittest.main()
