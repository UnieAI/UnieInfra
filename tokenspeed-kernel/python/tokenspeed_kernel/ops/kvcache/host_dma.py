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

"""Pluggable submitters for the Host cache's DMA byte ranges.

A submitter enqueues every ``(device_buffer_index, device_offset,
host_offset, num_bytes)`` range of one transfer on the current stream. One
``copy_`` per range costs ~20 us of CPU on the submitting (forward) thread --
as long as the copy itself for a 256 KiB range -- so each vendor plugs in the
cheapest batch submission its runtime has:

  ``cuda-batch``  NVIDIA, ``cudaMemcpyBatchAsync`` (CUDA 12.8+), ~1 us a range
  ``copy``        any device, one ``Tensor.copy_`` per range (the fallback)

Another backend (e.g. a HIP batch for AMD) registers itself with
``register_host_dma_submitter``. The highest-priority available submitter is
used unless ``TOKENSPEED_HOST_DMA_SUBMITTER`` names one. A submitter that
returns False before submitting anything (say, a driver without the API) is
retired for the process and the next one takes over.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

Direction = Literal["d2h", "h2d"]
Range = tuple[int, int, int, int]
SubmitFn = Callable[
    [Direction, Sequence[torch.Tensor], torch.Tensor, Sequence[Range]], bool
]


@dataclass(frozen=True)
class HostDmaSubmitter:
    """One way to enqueue a transfer's ranges.

    Attributes:
        name: The name ``TOKENSPEED_HOST_DMA_SUBMITTER`` selects it by.
        priority: Higher is tried first.
        available: Whether this process can use it at all (checked once).
        submit: Enqueues every range on the current stream and returns True,
            or returns False without enqueuing anything when it cannot.
    """

    name: str
    priority: int
    available: Callable[[], bool]
    submit: SubmitFn


_registry: dict[str, HostDmaSubmitter] = {}
# Submitters in the order they are tried, resolved on first use.
_active: list[HostDmaSubmitter] | None = None


def register_host_dma_submitter(submitter: HostDmaSubmitter) -> None:
    """Add (or replace) a submitter; it competes from the next transfer on."""
    global _active
    _registry[submitter.name] = submitter
    _active = None


def host_dma_submitters() -> list[str]:
    """The names of the usable submitters, in the order they are tried."""
    return [submitter.name for submitter in _resolve()]


def _resolve() -> list[HostDmaSubmitter]:
    global _active
    if _active is None:
        forced = os.environ.get("TOKENSPEED_HOST_DMA_SUBMITTER")
        if forced:
            if forced not in _registry:
                raise ValueError(
                    f"unknown TOKENSPEED_HOST_DMA_SUBMITTER {forced!r}; "
                    f"registered: {sorted(_registry)}"
                )
            candidates = [_registry[forced]]
        else:
            candidates = sorted(
                _registry.values(), key=lambda submitter: -submitter.priority
            )
        _active = [submitter for submitter in candidates if submitter.available()]
        if _registry["copy"] not in _active:
            _active.append(_registry["copy"])
    return _active


def submit_host_dma(
    direction: Direction,
    device_buffers: Sequence[torch.Tensor],
    host_buffer: torch.Tensor,
    ranges: Sequence[Range],
) -> str:
    """Enqueue ``ranges`` on the current stream; return the submitter used."""
    active = _resolve()
    while True:
        submitter = active[0]
        if submitter.submit(direction, device_buffers, host_buffer, ranges):
            return submitter.name
        # Nothing was enqueued: retire it for the process.
        active.pop(0)


# --- copy: one Tensor.copy_ per range -----------------------------------------


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


register_host_dma_submitter(
    HostDmaSubmitter(
        name="copy", priority=0, available=lambda: True, submit=_submit_copy
    )
)


# --- cuda-batch: cudaMemcpyBatchAsync ------------------------------------------

# Copies per call; the driver's cost per copy grows past a few thousand in one
# call (it waits for room in the stream's queue).
CUDA_MEMCPY_BATCH = 2048


def _cuda_runtime():
    from cuda.bindings import runtime

    return runtime


def _cuda_batch_available() -> bool:
    # ROCm builds report "cuda" devices too; cuda.bindings drives NVIDIA only.
    if torch.version.hip is not None or not torch.cuda.is_available():
        return False
    try:
        return hasattr(_cuda_runtime(), "cudaMemcpyBatchAsync")
    except ImportError:
        return False


def _submit_cuda_batch(direction, device_buffers, host_buffer, ranges) -> bool:
    device = device_buffers[0].device
    if device.type != "cuda":
        return False
    runtime = _cuda_runtime()
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


register_host_dma_submitter(
    HostDmaSubmitter(
        name="cuda-batch",
        priority=10,
        available=_cuda_batch_available,
        submit=_submit_cuda_batch,
    )
)
