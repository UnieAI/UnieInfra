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


"""TOKENSPEED_IRIS_TWOSHOT_MIN_BYTES pins the producer-direct two-shot cutover."""

import pytest
from tokenspeed_kernel.platform import current_platform

if not current_platform().is_amd:
    pytest.skip("Iris all-reduce is AMD-only", allow_module_level=True)

import tokenspeed_kernel.ops.communication.iris as iris


@pytest.fixture
def default_thresholds():
    saved = dict(iris._PRODUCER_TWO_SHOT_MIN_BYTES)
    yield saved
    iris._PRODUCER_TWO_SHOT_MIN_BYTES = saved


@pytest.mark.parametrize("raw, expected", [("0", 0), ("1048576", 1 << 20)])
def test_override_applies_to_every_world_size(
    monkeypatch, default_thresholds, raw, expected
):
    monkeypatch.setenv("TOKENSPEED_IRIS_TWOSHOT_MIN_BYTES", raw)
    iris._apply_two_shot_threshold_override()
    assert iris._PRODUCER_TWO_SHOT_MIN_BYTES == dict.fromkeys(
        default_thresholds, expected
    )


@pytest.mark.parametrize("raw", ["", "-1", "1MB"])
def test_unset_or_malformed_value_keeps_the_defaults(
    monkeypatch, default_thresholds, raw
):
    monkeypatch.setenv("TOKENSPEED_IRIS_TWOSHOT_MIN_BYTES", raw)
    iris._apply_two_shot_threshold_override()
    assert iris._PRODUCER_TWO_SHOT_MIN_BYTES == default_thresholds
