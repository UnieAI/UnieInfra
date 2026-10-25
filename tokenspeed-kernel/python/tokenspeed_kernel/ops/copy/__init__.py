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

"""Device copy helpers.

``memcpy_2d_async`` copies a pitched device rectangle through the kernel
registry (family ``copy``, mode ``memcpy_2d``): each vendor registers its
runtime's 2D copy, and ``memcpy_2d_supported`` tells a caller whether this
platform has one before it plans around it.
"""

from __future__ import annotations

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.copy.cuda  # noqa: F401
import tokenspeed_kernel.ops.copy.hip  # noqa: F401
import torch
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

# The op takes raw pointers; the signature names the bytes it moves.
_SIGNATURE = format_signature(bytes=dense_tensor_format(torch.uint8))


def memcpy_2d_supported() -> bool:
    """Whether some registered backend serves this platform."""
    try:
        select_kernel("copy", "memcpy_2d", _SIGNATURE)
    except Exception:
        return False
    return True


def memcpy_2d_async(
    *,
    dst: int,
    dst_pitch: int,
    src: int,
    src_pitch: int,
    width: int,
    height: int,
    stream_ptr: int,
) -> None:
    """Copy a ``height`` x ``width``-byte device rectangle on ``stream_ptr``."""
    select_kernel("copy", "memcpy_2d", _SIGNATURE)(
        dst=dst,
        dst_pitch=dst_pitch,
        src=src,
        src_pitch=src_pitch,
        width=width,
        height=height,
        stream_ptr=stream_ptr,
    )


__all__ = ["memcpy_2d_async", "memcpy_2d_supported"]
