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

"""ROCm decode queries pad only to the gfx950 16-head block."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41

pytest.importorskip("tokenspeed_triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm GPU required",
)


@pytest.mark.parametrize("heads", [1, 16, 17, 32, 64, 100])
def test_rope_pad_query_pads_to_sixteen_on_rocm(heads):
    torch.manual_seed(heads)
    tokens, rope_dim = 7, 64
    values = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32) * 37
    angles = torch.randn(4096, rope_dim // 2, device="cuda")
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=-1).float()
    padded = dsv41.rope_pad_query(values, positions, cos_sin, None)
    assert padded.shape == (tokens, -(-heads // 16) * 16, 512)
    assert padded.is_contiguous()
    expected = values.clone()
    dsv41.rope_inplace(expected, positions, cos_sin, None)
    torch.testing.assert_close(padded[:, :heads], expected, rtol=0, atol=0)
    assert not padded[:, heads:].any()
