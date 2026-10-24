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

"""Vendor-neutral PyTorch implementations of the activation family.

They cover the modes no vendor kernel serves on a platform (GeLU-gated MLPs
on AMD, say): FP32 math, one rounding to the input dtype.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FLOATS = format_signatures(
    "x", "dense", {torch.bfloat16, torch.float16, torch.float32}
)


def _gated(x: torch.Tensor, activation, out: torch.Tensor | None) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    result = (activation(gate) * up).to(x.dtype)
    return result if out is None else out.copy_(result)


@register_kernel(
    "activation",
    "gelu_and_mul",
    name="native_gelu_and_mul",
    solution="torch",
    signatures=_FLOATS,
    priority=Priority.PORTABLE,
)
def native_gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    return _gated(x, F.gelu, out)


@register_kernel(
    "activation",
    "gelu_tanh_and_mul",
    name="native_gelu_tanh_and_mul",
    solution="torch",
    signatures=_FLOATS,
    priority=Priority.PORTABLE,
)
def native_gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    return _gated(x, lambda gate: F.gelu(gate, approximate="tanh"), out)
