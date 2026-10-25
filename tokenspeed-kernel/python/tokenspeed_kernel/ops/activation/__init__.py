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

"""Activation kernel entry points."""

from __future__ import annotations

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.activation.cuda  # noqa: F401
import tokenspeed_kernel.ops.activation.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.activation.native  # noqa: F401
import tokenspeed_kernel.ops.activation.triton  # noqa: F401
import torch
from tokenspeed_kernel.ops.activation.triton import add3, relu2
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_npu:
    import tokenspeed_kernel.ops.activation.ascend  # noqa: F401

# Every gated activation below dispatches through the kernel registry: a
# vendor (or an out-of-tree plugin) adds an implementation by registering
# the same family and mode, and TOKENSPEED_KERNEL_OVERRIDE_ACTIVATION_<MODE>
# forces one.


def _select(mode: str, x: torch.Tensor, role: str = "x", **traits):
    return select_kernel(
        "activation",
        mode,
        format_signature(**{role: dense_tensor_format(x.dtype)}),
        traits=traits or None,
    )


def silu_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    limit: float | None = None,
) -> torch.Tensor:
    """``silu(x[..., :D]) * x[..., D:]``; a ``limit`` clamps both halves."""
    return _select("silu_and_mul", x, limited=limit is not None)(x, limit, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``gelu(x[..., :D]) * x[..., D:]`` (exact erf GeLU)."""
    return _select("gelu_and_mul", x)(x, out=out)


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``gelu(x[..., :D], approximate="tanh") * x[..., D:]``."""
    return _select("gelu_tanh_and_mul", x)(x, out=out)


def situ_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """SiTU gate times up, optionally soft-capping up with ``linear_beta``."""
    return _select("situ_and_mul", x)(x, beta, linear_beta, out=out)


def silu_and_mul_fp8_block_quant(
    x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU quantized to FP8 in 128-column blocks, written to ``out``/``scale``."""
    return _select("silu_and_mul_fp8_block_quant", x)(x, scale, out)


def sigmoid_mul(
    x: torch.Tensor, gate: torch.Tensor, head_bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``x * sigmoid(gate + head_bias)``."""
    kernel = select_kernel(
        "activation",
        "sigmoid_mul",
        format_signature(
            x=dense_tensor_format(x.dtype), gate=dense_tensor_format(gate.dtype)
        ),
    )
    return kernel(x, gate, head_bias)


def swiglu_oai(
    gate_up: torch.Tensor, *, alpha: float = 1.702, limit: float = 7.0
) -> torch.Tensor:
    """The gpt-oss clamped SwiGLU over interleaved ``gate_up``."""
    return _select("swiglu_oai", gate_up, role="gate_up")(gate_up, alpha, limit)


def fused_swiglu_fp8_ue8m0(
    gate_up: torch.Tensor,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU quantized to FP8 with UE8M0 scales per 128 columns."""
    kernel = _select("fused_swiglu_fp8_ue8m0", gate_up, role="gate_up")
    return kernel(gate_up, swiglu_limit, swiglu_alpha, swiglu_beta)


__all__ = [
    "add3",
    "fused_swiglu_fp8_ue8m0",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "relu2",
    "sigmoid_mul",
    "silu_and_mul",
    "silu_and_mul_fp8_block_quant",
    "situ_and_mul",
    "swiglu_oai",
]
