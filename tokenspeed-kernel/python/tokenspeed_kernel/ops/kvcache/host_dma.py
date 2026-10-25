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

"""Submitters for the Host cache's DMA byte ranges: kvcache family, mode
``host_dma`` in the kernel registry.

A submitter enqueues every ``(device_buffer_index, device_offset,
host_offset, num_bytes)`` range of one transfer on the current stream. One
``copy_`` per range costs ~20 us of CPU on the submitting (forward) thread --
as long as the copy itself for a 256 KiB range -- so each vendor registers the
cheapest batch submission its runtime has:

  ``host_dma_cuda_batch``  NVIDIA, ``cudaMemcpyBatchAsync`` (CUDA 12.8+)
  ``host_dma_copy``        any device, one ``Tensor.copy_`` per range

Another backend (e.g. a HIP batch for AMD) registers the same family and mode
with ``vendors={"amd"}`` -- from an out-of-tree plugin with
``Priority.PLUGIN``. Selection is the registry's, so
``TOKENSPEED_KERNEL_OVERRIDE_KVCACHE_HOST_DMA=<name>`` forces one. A submitter
returns False, having enqueued nothing, when its runtime turns out not to
support it (say, a driver without the API); it is then refused for the
process and ``host_dma_copy`` carries the transfer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

Direction = Literal["d2h", "h2d"]
Range = tuple[int, int, int, int]

FALLBACK = "host_dma_copy"
_SIGNATURE = format_signature(host=dense_tensor_format(torch.uint8))
_SIGNATURES = format_signatures("host", "dense", {torch.uint8})
# Submitters whose runtime refused them; process-wide.
_refused: set[str] = set()


def submit_host_dma(
    direction: Direction,
    device_buffers: Sequence[torch.Tensor],
    host_buffer: torch.Tensor,
    ranges: Sequence[Range],
) -> str:
    """Enqueue ``ranges`` on the current stream; return the submitter used."""
    kernel = select_kernel("kvcache", "host_dma", _SIGNATURE)
    if kernel.name not in _refused and kernel(
        direction, device_buffers, host_buffer, ranges
    ):
        return kernel.name
    # Nothing was enqueued: refuse it for the process.
    _refused.add(kernel.name)
    kernel = select_kernel("kvcache", "host_dma", _SIGNATURE, override=FALLBACK)
    kernel(direction, device_buffers, host_buffer, ranges)
    return kernel.name


# --- copy: one Tensor.copy_ per range -----------------------------------------


@register_kernel(
    "kvcache",
    "host_dma",
    name=FALLBACK,
    solution="torch",
    signatures=_SIGNATURES,
    priority=Priority.PORTABLE,
)
def _submit_copy(direction, device_buffers, host_buffer, ranges) -> bool:
    byte_buffers = tuple(
        buffer.view(torch.uint8).reshape(-1) for buffer in device_buffers
    )
    for device_buffer_index, device_offset, host_offset, num_bytes in ranges:
        device_slice = byte_buffers[device_buffer_index][
            device_offset : device_offset + num_bytes
        ]
        host_slice = host_buffer[host_offset : host_offset + num_bytes]
        destination, source = (
            (host_slice, device_slice)
            if direction == "d2h"
            else (device_slice, host_slice)
        )
        destination.copy_(source, non_blocking=True)
    return True


# --- cuda-batch: cudaMemcpyBatchAsync ------------------------------------------

# Copies per call; the driver's cost per copy grows past a few thousand in one
# call (it waits for room in the stream's queue).
CUDA_MEMCPY_BATCH = 2048


def _cuda_runtime():
    from cuda.bindings import runtime

    return runtime


@register_kernel(
    "kvcache",
    "host_dma",
    name="host_dma_cuda_batch",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_SIGNATURES,
    priority=Priority.PERFORMANT,
)
def _submit_cuda_batch(direction, device_buffers, host_buffer, ranges) -> bool:
    if not ranges:
        return True
    try:
        runtime = _cuda_runtime()
    except ImportError:
        return False
    if not hasattr(runtime, "cudaMemcpyBatchAsync"):
        return False
    device = device_buffers[0].device
    attributes = runtime.cudaMemcpyAttributes()
    attributes.srcAccessOrder = (
        runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
    )
    attributes.flags = runtime.cudaMemcpyFlags.cudaMemcpyFlagPreferOverlapWithCompute
    device_location = runtime.cudaMemLocation()
    device_location.type = runtime.cudaMemLocationType.cudaMemLocationTypeDevice
    device_location.id = device.index if device.index is not None else 0
    host_location = runtime.cudaMemLocation()
    host_location.type = runtime.cudaMemLocationType.cudaMemLocationTypeHost
    if direction == "d2h":
        attributes.srcLocHint, attributes.dstLocHint = device_location, host_location
    else:
        attributes.srcLocHint, attributes.dstLocHint = host_location, device_location
    buffer_addresses = [buffer.data_ptr() for buffer in device_buffers]
    host_address = host_buffer.data_ptr()
    stream = torch.cuda.current_stream(device).cuda_stream
    for start in range(0, len(ranges), CUDA_MEMCPY_BATCH):
        batch = ranges[start : start + CUDA_MEMCPY_BATCH]
        device_addresses = [
            buffer_addresses[index] + offset for index, offset, _, _ in batch
        ]
        host_addresses = [host_address + offset for _, _, offset, _ in batch]
        sizes = [num_bytes for *_, num_bytes in batch]
        destinations, sources = (
            (host_addresses, device_addresses)
            if direction == "d2h"
            else (device_addresses, host_addresses)
        )
        result = runtime.cudaMemcpyBatchAsync(
            destinations, sources, sizes, len(sizes), [attributes], [0], 1, stream
        )
        error = result[0] if isinstance(result, tuple) else result
        if error != runtime.cudaError_t.cudaSuccess:
            if start == 0:
                # A driver without batch support refuses the first call.
                return False
            raise RuntimeError(f"cudaMemcpyBatchAsync failed: {error}")
    return True
