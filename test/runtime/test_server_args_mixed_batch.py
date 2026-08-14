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

"""Regression tests for --enable-mixed-batch / --disable-mixed-batch.

Guards against the state this repository actually shipped in: ``enable_mixed_batch``
defaults to True, but its only CLI form was ``--enable-mixed-batch`` with
``action="store_true"``. Passing the flag was a no-op and there was no way to turn
mixing off at all -- so the feature could neither be bisected when it misbehaved
nor A/B-measured to find out what it was worth.
"""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import argparse
import unittest

from tokenspeed.runtime.utils.server_args import ServerArgs


class TestMixedBatchFlag(unittest.TestCase):
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        return parser.parse_args([*argv, "--model-path", "/nonexistent"])

    def test_defaults_on(self):
        self.assertTrue(self._parse([]).enable_mixed_batch)

    def test_enable_flag_keeps_it_on(self):
        self.assertTrue(self._parse(["--enable-mixed-batch"]).enable_mixed_batch)

    def test_disable_flag_turns_it_off(self):
        self.assertFalse(self._parse(["--disable-mixed-batch"]).enable_mixed_batch)

    def test_both_flags_last_one_wins(self):
        """argparse applies actions in order, so the trailing flag decides."""
        self.assertFalse(
            self._parse(
                ["--enable-mixed-batch", "--disable-mixed-batch"]
            ).enable_mixed_batch
        )
        self.assertTrue(
            self._parse(
                ["--disable-mixed-batch", "--enable-mixed-batch"]
            ).enable_mixed_batch
        )

    def test_both_forms_share_one_dest(self):
        """Two actions, one dest -- otherwise the disable form silently does nothing."""
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        options = {
            option
            for action in parser._actions
            if action.dest == "enable_mixed_batch"
            for option in action.option_strings
        }
        self.assertEqual(options, {"--enable-mixed-batch", "--disable-mixed-batch"})


if __name__ == "__main__":
    unittest.main()
