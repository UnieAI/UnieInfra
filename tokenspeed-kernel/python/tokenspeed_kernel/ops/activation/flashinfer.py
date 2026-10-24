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

"""FlashInfer activation kernels."""

import torch
from tokenspeed_kernel.platform import (
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

gelu_and_mul = error_fn
gelu_tanh_and_mul = error_fn
silu_and_mul = error_fn

if current_platform().is_nvidia:
    try:
        from flashinfer import (
            gelu_and_mul,
            gelu_tanh_and_mul,
            silu_and_mul,
        )
    except ImportError:
        pass

__all__ = ["gelu_and_mul", "gelu_tanh_and_mul", "silu_and_mul"]

_NVIDIA = CapabilityRequirement(vendors=frozenset({"nvidia"}))
_HALF = format_signatures("x", "dense", {torch.bfloat16, torch.float16})


def _gated_out(x: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    if out is not None:
        return out
    return torch.empty(
        x.shape[:-1] + (x.shape[-1] // 2,), dtype=x.dtype, device=x.device
    )


if silu_and_mul is not error_fn:

    @register_kernel(
        "activation",
        "silu_and_mul",
        name="flashinfer_silu_and_mul",
        solution="flashinfer",
        capability=_NVIDIA,
        signatures=_HALF,
        # FlashInfer has no clamp: a limited SwiGLU takes the Triton kernel.
        traits={"limited": frozenset({False})},
        priority=Priority.PERFORMANT,
    )
    def flashinfer_silu_and_mul(x, limit, out=None):
        return silu_and_mul(x, _gated_out(x, out), enable_pdl=pdl_enabled())


if gelu_and_mul is not error_fn:

    @register_kernel(
        "activation",
        "gelu_and_mul",
        name="flashinfer_gelu_and_mul",
        solution="flashinfer",
        capability=_NVIDIA,
        signatures=_HALF,
        priority=Priority.PERFORMANT,
    )
    def flashinfer_gelu_and_mul(x, out=None):
        return gelu_and_mul(x, _gated_out(x, out), enable_pdl=pdl_enabled())


if gelu_tanh_and_mul is not error_fn:

    @register_kernel(
        "activation",
        "gelu_tanh_and_mul",
        name="flashinfer_gelu_tanh_and_mul",
        solution="flashinfer",
        capability=_NVIDIA,
        signatures=_HALF,
        priority=Priority.PERFORMANT,
    )
    def flashinfer_gelu_tanh_and_mul(x, out=None):
        return gelu_tanh_and_mul(x, _gated_out(x, out), enable_pdl=pdl_enabled())
