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

"""AMD 2D device-to-device copy."""

from __future__ import annotations

import ctypes
from functools import lru_cache

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# hipMemcpyKind, the same values as cudaMemcpyKind.
_HIP_MEMCPY_DEVICE_TO_DEVICE = 3


@lru_cache(maxsize=1)
def _hip_memcpy_2d_async():
    from tokenspeed_kernel.platform import _get_hip_runtime

    function = _get_hip_runtime().hipMemcpy2DAsync
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    return function


@register_kernel(
    "copy",
    "memcpy_2d",
    name="hip_memcpy_2d",
    solution="hip",
    capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    signatures=format_signatures("bytes", "dense", {torch.uint8}),
    priority=Priority.PERFORMANT,
)
def hip_memcpy_2d_async(
    *,
    dst: int,
    dst_pitch: int,
    src: int,
    src_pitch: int,
    width: int,
    height: int,
    stream_ptr: int,
) -> None:
    """``hipMemcpy2DAsync`` of a device rectangle on a HIP stream."""
    error = _hip_memcpy_2d_async()(
        dst,
        dst_pitch,
        src,
        src_pitch,
        width,
        height,
        _HIP_MEMCPY_DEVICE_TO_DEVICE,
        stream_ptr,
    )
    if error != 0:
        raise RuntimeError(f"hipMemcpy2DAsync failed: {error}")
