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

from __future__ import annotations

import pytest
import torch

from tokenspeed_kernel.ops.attention.gdn import flashinfer


@pytest.mark.parametrize(
    ("head_dim", "dtype", "num_q_heads", "num_v_heads", "expected"),
    [
        (128, torch.bfloat16, 16, 48, True),
        (128, torch.bfloat16, 16, 16, True),
        (128, torch.float16, 16, 48, False),
        (64, torch.bfloat16, 16, 48, False),
        (128, torch.bfloat16, 48, 16, False),
    ],
)
def test_gdn_flashinfer_shape_gate(
    monkeypatch, head_dim, dtype, num_q_heads, num_v_heads, expected
):
    monkeypatch.setattr(flashinfer, "is_available", lambda: True)

    assert (
        flashinfer.is_supported(head_dim, dtype, num_q_heads, num_v_heads)
        is expected
    )


def test_gdn_flashinfer_requires_available_platform(monkeypatch):
    monkeypatch.setattr(flashinfer, "is_available", lambda: False)

    assert not flashinfer.is_supported(128, torch.bfloat16, 16, 48)
