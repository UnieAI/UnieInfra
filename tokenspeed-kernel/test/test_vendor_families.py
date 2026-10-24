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

"""Every kernel family the runtime calls resolves on each vendor.

Selection is checked against fixed NVIDIA (H100) and AMD (MI350) platforms,
so an AMD path is exercised without AMD hardware; the device tests below run
the selected kernels where a GPU is present.
"""

from __future__ import annotations

import importlib

import pytest
import tokenspeed_kernel.ops.activation  # noqa: F401
import tokenspeed_kernel.ops.copy  # noqa: F401
import tokenspeed_kernel.ops.kvcache  # noqa: F401
import tokenspeed_kernel.ops.kvcache.host_dma  # noqa: F401
import torch
from tokenspeed_kernel.ops.activation.flashinfer import silu_and_mul as _fi_silu
from tokenspeed_kernel.registry import error_fn
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_FLASHINFER = "flashinfer" if _fi_silu is not error_fn else None


def _bytes(role: str):
    return format_signature(**{role: dense_tensor_format(torch.uint8)})


def _bf16(role: str = "x"):
    return format_signature(**{role: dense_tensor_format(torch.bfloat16)})


# (family, mode, signature, traits, H100 kernel, MI350 kernel); None for the
# H100 column means "the FlashInfer kernel when FlashInfer is installed,
# otherwise the portable one".
CASES = [
    (
        "kvcache",
        "zero_device_byte_ranges",
        _bytes("backing"),
        None,
        "triton_zero_device_byte_ranges",
        "triton_zero_device_byte_ranges",
    ),
    (
        "kvcache",
        "zero_host_byte_ranges",
        _bytes("backing"),
        None,
        "cute_dsl_zero_host_byte_ranges",
        "sync_zero_host_byte_ranges",
    ),
    (
        "kvcache",
        "host_dma",
        _bytes("host"),
        None,
        "host_dma_cuda_batch",
        "host_dma_copy",
    ),
    ("copy", "memcpy_2d", _bytes("bytes"), None, "cuda_memcpy_2d", "hip_memcpy_2d"),
    (
        "activation",
        "silu_and_mul",
        _bf16(),
        {"limited": False},
        "flashinfer_silu_and_mul" if _FLASHINFER else "triton_silu_and_mul",
        "triton_silu_and_mul",
    ),
    (
        "activation",
        "silu_and_mul",
        _bf16(),
        {"limited": True},
        "triton_silu_and_mul",
        "triton_silu_and_mul",
    ),
    (
        "activation",
        "gelu_tanh_and_mul",
        _bf16(),
        None,
        "flashinfer_gelu_tanh_and_mul" if _FLASHINFER else "native_gelu_tanh_and_mul",
        "native_gelu_tanh_and_mul",
    ),
    (
        "activation",
        "situ_and_mul",
        _bf16(),
        None,
        "triton_situ_and_mul",
        "triton_situ_and_mul",
    ),
    (
        "activation",
        "swiglu_oai",
        _bf16("gate_up"),
        None,
        "triton_swiglu_oai",
        "triton_swiglu_oai",
    ),
]


@pytest.mark.parametrize(
    ("family", "mode", "signature", "traits", "nvidia", "amd"),
    CASES,
    ids=[f"{c[0]}.{c[1]}{'.' + str(c[3]) if c[3] else ''}" for c in CASES],
)
def test_each_vendor_selects_its_kernel(
    h100_platform, mi350_platform, family, mode, signature, traits, nvidia, amd
):
    for platform, expected in ((h100_platform, nvidia), (mi350_platform, amd)):
        kernel = select_kernel(
            family, mode, signature, platform=platform, traits=traits
        )
        assert kernel.name == expected, (platform.vendor, family, mode)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@requires_cuda
@pytest.mark.parametrize("limit", [None, 7.0])
def test_gated_activations_match_their_references(limit):
    from tokenspeed_kernel.numerics.torch.activation import torch_silu_and_mul
    from tokenspeed_kernel.ops.activation import gelu_tanh_and_mul, silu_and_mul
    from tokenspeed_kernel.ops.activation.native import native_gelu_tanh_and_mul

    x = torch.randn(33, 2 * 256, dtype=torch.bfloat16, device="cuda") * 4
    out = torch.empty(33, 256, dtype=torch.bfloat16, device="cuda")
    assert silu_and_mul(x, out, limit=limit) is out
    torch.testing.assert_close(out, torch_silu_and_mul(x, limit), atol=2e-2, rtol=2e-2)
    if limit is None:
        torch.testing.assert_close(
            gelu_tanh_and_mul(x), native_gelu_tanh_and_mul(x), atol=2e-2, rtol=2e-2
        )


@requires_cuda
@pytest.mark.parametrize(
    "kernel", ["cute_dsl_zero_host_byte_ranges", "sync_zero_host_byte_ranges"]
)
def test_host_byte_ranges_zero_under_either_kernel(monkeypatch, kernel):
    from tokenspeed_kernel.ops.kvcache import zero_byte_ranges

    monkeypatch.setenv(
        "TOKENSPEED_KERNEL_OVERRIDE_KVCACHE_ZERO_HOST_BYTE_RANGES", kernel
    )
    if kernel.startswith("cute_dsl") and importlib.util.find_spec("cutlass") is None:
        pytest.skip("CuTe DSL is not installed")
    backing = torch.full((4096,), 7, dtype=torch.uint8, pin_memory=True)
    zero_byte_ranges(backing, [(16, 100), (1000, 3000)], device="cuda")
    torch.cuda.synchronize()
    expected = torch.full((4096,), 7, dtype=torch.uint8)
    expected[16:116] = 0
    expected[1000:4000] = 0
    assert torch.equal(backing, expected)


@requires_cuda
def test_memcpy_2d_copies_a_pitched_rectangle():
    from tokenspeed_kernel.ops.copy import memcpy_2d_async, memcpy_2d_supported

    assert memcpy_2d_supported()
    src = torch.arange(8 * 64, dtype=torch.int32, device="cuda").to(torch.uint8)
    dst = torch.zeros(8 * 16, dtype=torch.uint8, device="cuda")
    memcpy_2d_async(
        dst=dst.data_ptr(),
        dst_pitch=16,
        src=src.data_ptr(),
        src_pitch=64,
        width=16,
        height=8,
        stream_ptr=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    assert torch.equal(dst.view(8, 16), src.view(8, 64)[:, :16])
