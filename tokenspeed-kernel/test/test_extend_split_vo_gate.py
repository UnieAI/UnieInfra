# Copyright (c) 2026 UnieAI
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
"""The output-dimension split must fire only where it was measured to win.

Splitting head_dim 512 attention into two head_dim_vo 256 passes costs an extra
QK (1.5x the FLOPs) and buys back a halved fp32 accumulator. That trade is
profitable only for the widest heads. Measured on an idle sm_120 at gemma-4's
production prefill shape (8192 new tokens against 35,326 cached, 32 q / 4 kv):

    head_dim   one pass   two passes
       512     115.99 ms    90.46 ms   1.28x   split wins
       256      26.83 ms    38.58 ms   0.70x   split loses
       128      11.76 ms    18.14 ms   0.65x   split loses

So the gate is a measurement, not a guess, and narrowing or widening it changes
real numbers. head_dim 256 covers gemma-4's fifty sliding layers and every
attention layer of both Qwen models, whose output must stay bit-identical.

No GPU needed: this checks the predicate, not the kernel.
"""

import torch
from tokenspeed_kernel.ops.attention.flashinfer import _split_vo_width


class _Cache:
    """Just enough of a tensor for the predicate: a dtype and a last dim."""

    def __init__(self, head_dim: int, dtype=torch.bfloat16):
        self.shape = (4, 16, 4, head_dim)
        self.dtype = dtype


def _width(head_dim, k_dtype=torch.bfloat16, v_dtype=torch.bfloat16, total_q=8192):
    return _split_vo_width(
        head_dim, _Cache(head_dim, k_dtype), _Cache(head_dim, v_dtype), total_q
    )


def test_splits_at_512():
    """gemma-4's ten global layers, the only measured win."""
    assert _width(512) == 256


def test_does_not_split_at_256_or_below():
    """Measured 0.70x and 0.65x; splitting here would be a regression."""
    assert _width(256) is None
    assert _width(128) is None
    assert _width(64) is None


def test_does_not_split_tiny_batches():
    """Below three query rows the asymmetric kernel does not fit in shared
    memory on this GPU and flashinfer raises. Measured boundary: 1 and 2 rows
    raise, 3 upward run, at both 2,048 and 35,326 cached tokens. A
    `#new-token: 1` batch inside the prefill graph is what took the server down
    before this guard existed."""
    assert _width(512, total_q=1) is None
    assert _width(512, total_q=2) is None
    assert _width(512, total_q=3) == 256


def test_does_not_split_fp8_kv():
    """fp8 carries per-tensor scales that were never checked against a split."""
    assert _width(512, k_dtype=torch.float8_e4m3fn) is None
    assert _width(512, v_dtype=torch.float8_e5m2) is None


def test_env_disables():
    """The escape hatch has to actually reach the predicate."""
    import os

    previous = os.environ.get("TOKENSPEED_FI_EXTEND_SPLIT_VO")
    os.environ["TOKENSPEED_FI_EXTEND_SPLIT_VO"] = "0"
    try:
        assert _width(512) is None
    finally:
        if previous is None:
            os.environ.pop("TOKENSPEED_FI_EXTEND_SPLIT_VO", None)
        else:
            os.environ["TOKENSPEED_FI_EXTEND_SPLIT_VO"] = previous
