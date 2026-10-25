# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CUDA activation kernels."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel.thirdparty.cuda.activation import (
        silu_and_mul_fuse_block_quant,
        silu_and_mul_fuse_nvfp4_quant,
    )
except ImportError:
    silu_and_mul_fuse_block_quant = error_fn
    silu_and_mul_fuse_nvfp4_quant = error_fn

__all__ = ["silu_and_mul_fuse_block_quant", "silu_and_mul_fuse_nvfp4_quant"]

if silu_and_mul_fuse_block_quant is not error_fn:

    @register_kernel(
        "activation",
        "silu_and_mul_fp8_block_quant",
        name="cuda_silu_and_mul_fp8_block_quant",
        solution="cuda",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
        signatures=format_signatures("x", "dense", {torch.bfloat16, torch.float16}),
        priority=Priority.PERFORMANT,
    )
    def cuda_silu_and_mul_fp8_block_quant(x, scale, out):
        return silu_and_mul_fuse_block_quant(x, scale, out)
