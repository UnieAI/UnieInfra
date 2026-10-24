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

"""Registered implementations of the kvcache family's byte-range zeroing.

``zero_device_byte_ranges`` clears ranges of a Device allocation on the
current stream; ``zero_host_byte_ranges`` clears ranges of a pinned Host
allocation in stream order on an execution device. The implementations import
their backends on first call, so registering them costs nothing at import. An
out-of-tree backend (another vendor's copy engine, say) registers under the
same family and mode with ``Priority.PLUGIN``.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_BYTES = format_signatures("backing", "dense", {torch.uint8})


@register_kernel(
    "kvcache",
    "zero_device_byte_ranges",
    name="triton_zero_device_byte_ranges",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=_BYTES,
    priority=Priority.PORTABLE,
)
def triton_zero_device_byte_ranges(backing, ranges, *, device):
    from tokenspeed_kernel.ops.kvcache.triton import zero_byte_ranges

    zero_byte_ranges(backing, ranges)


@register_kernel(
    "kvcache",
    "zero_device_byte_ranges",
    name="torch_zero_device_byte_ranges",
    solution="torch",
    signatures=_BYTES,
    priority=Priority.REFERENCE,
)
def torch_zero_device_byte_ranges(backing, ranges, *, device):
    for offset, size in ranges:
        backing[offset : offset + size].zero_()


@register_kernel(
    "kvcache",
    "zero_host_byte_ranges",
    name="cute_dsl_zero_host_byte_ranges",
    solution="cute_dsl",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_BYTES,
    priority=Priority.PERFORMANT,
)
def cute_dsl_zero_host_byte_ranges(backing, ranges, *, device):
    from tokenspeed_kernel.ops.kvcache.cute_dsl import (
        cute_dsl_zero_host_byte_ranges as zero,
    )

    zero(backing, ranges, device=device)


@register_kernel(
    "kvcache",
    "zero_host_byte_ranges",
    name="sync_zero_host_byte_ranges",
    solution="torch",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=_BYTES,
    priority=Priority.PORTABLE,
)
def sync_zero_host_byte_ranges(backing, ranges, *, device):
    # Any vendor: drain the stream so earlier copies out of these bytes have
    # landed, then clear them on the CPU before later work can be enqueued.
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("host range clearing must run outside graph capture")
    torch.cuda.current_stream(device).synchronize()
    for offset, size in ranges:
        backing[offset : offset + size].zero_()
