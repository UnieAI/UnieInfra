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

"""Compact Host storage for cache transfer entries.

Backed either by pinned RAM or by a file on local NVMe. Both are one flat
uint8 buffer addressed by the same byte arithmetic, so only the allocation
differs -- `host_block_offset` is unchanged and the executor cannot tell them
apart.""" 

from __future__ import annotations

import logging
import os

from tokenspeed.runtime.cache.transfer.layout import CacheTransferLayout

logger = logging.getLogger(__name__)


def _compute_host_layout(
    layout: CacheTransferLayout,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int]:
    host_cache_block_bytes = []
    host_field_offsets = []
    host_lcm_block_bytes = 0
    for group in layout.groups:
        offsets = []
        cursor = 0
        for field in group.fields:
            offsets.append(cursor)
            cursor += field.payload_bytes
        host_cache_block_bytes.append(cursor)
        host_field_offsets.append(tuple(offsets))
        host_lcm_block_bytes = max(
            host_lcm_block_bytes,
            group.cache_blocks_per_lcm_block * cursor,
        )
    return (
        tuple(host_cache_block_bytes),
        tuple(host_field_offsets),
        host_lcm_block_bytes,
    )


def compute_host_lcm_block_bytes(layout: CacheTransferLayout) -> int:
    """Return compact Host bytes required by one LCMBlock."""

    return _compute_host_layout(layout)[2]


class HostCacheStorage:
    """One compact Host allocation indexed by scheduler CacheBlock IDs.

    `backing_path` swaps pinned RAM for a shared mapping of a file on disk.
    RAM is the fast tier; a file on NVMe trades bandwidth for capacity that is
    not bounded by how much memory the box has, which is the tier a deployment
    buys when DRAM is the expensive part. Copies out of a mapped file are
    pageable, so the DMA transfer backend cannot be used with it.
    """

    def __init__(
        self,
        layout: CacheTransferLayout,
        *,
        num_host_lcm_blocks: int,
        backing_path: str | None = None,
    ):
        if num_host_lcm_blocks <= 0:
            raise ValueError("num_host_lcm_blocks must be > 0")
        self.layout = layout
        (
            self.host_cache_block_bytes,
            self.host_field_offsets,
            self.host_lcm_block_bytes,
        ) = _compute_host_layout(layout)
        self.num_host_lcm_blocks = int(num_host_lcm_blocks)
        self.host_buffer_bytes = self.num_host_lcm_blocks * self.host_lcm_block_bytes
        self.backing_path = backing_path
        import torch

        if backing_path is None:
            self.host_buffer = torch.empty(
                self.host_buffer_bytes,
                dtype=torch.uint8,
                pin_memory=True,
            )
        else:
            os.makedirs(os.path.dirname(backing_path) or ".", exist_ok=True)
            # Preallocate: a sparse file turns every first write into an
            # allocating fault under the transfer stream.
            with open(backing_path, "wb") as fh:
                os.posix_fallocate(fh.fileno(), 0, self.host_buffer_bytes)
            self.host_buffer = None
            # O_DIRECT rather than a mapping. Every offset and length here is a
            # multiple of the 524,288-byte plane, so the alignment O_DIRECT
            # demands is free -- and it buys the two things the mapped version
            # could not have: reads and writes that never enter the page cache
            # (so the tier is genuinely the disk, with no per-range msync,
            # madvise and fadvise to evict what should never have been cached),
            # and I/O that can be issued from a worker thread.
            flags = os.O_RDWR
            self._direct = hasattr(os, "O_DIRECT")
            if self._direct:
                flags |= os.O_DIRECT
            try:
                self._fd = os.open(backing_path, flags)
            except OSError:  # filesystem without O_DIRECT
                self._direct = False
                self._fd = os.open(backing_path, os.O_RDWR)
            logger.info(
                "Host L2 backed by file %s (%.2f GB, O_DIRECT=%s)",
                backing_path,
                self.host_buffer_bytes / 1e9,
                self._direct,
            )

    _fd: int | None = None
    _direct: bool = False

    # Linux caps one read or write syscall at MAX_RW_COUNT, INT_MAX rounded down
    # to a page: 2 GB - 4096. Coalescing coarse ranges reaches past that, and the
    # kernel then transfers a prefix and returns -- which without the loop below
    # is silent corruption, because the rest of the file keeps whatever it held
    # while every latency number stays healthy. Chunked well under the cap so a
    # partial transfer stays the exception rather than the rule; the boundary is
    # page-aligned, which O_DIRECT requires.
    _IO_CHUNK = 1 << 30

    def read_into(self, view: memoryview, file_offset: int) -> None:
        """Fill `view` from the backing file at `file_offset`."""

        done = 0
        total = len(view)
        while done < total:
            want = min(self._IO_CHUNK, total - done)
            got = os.preadv(self._fd, [view[done : done + want]], file_offset + done)
            if got <= 0:
                raise OSError(
                    f"read from cache file stalled at {done} of {total} bytes "
                    f"from offset {file_offset}"
                )
            done += got

    def write_from(self, view: memoryview, file_offset: int) -> None:
        """Write `view` to the backing file at `file_offset`."""

        done = 0
        total = len(view)
        while done < total:
            want = min(self._IO_CHUNK, total - done)
            put = os.pwritev(self._fd, [view[done : done + want]], file_offset + done)
            if put <= 0:
                raise OSError(
                    f"write to cache file stalled at {done} of {total} bytes "
                    f"to offset {file_offset}"
                )
            done += put

    def host_block_offset(self, group_index: int, block_id: int) -> int:
        """Return the packed byte offset for one non-null CacheBlock."""

        try:
            group = self.layout.groups[group_index]
            host_cache_block_bytes = self.host_cache_block_bytes[group_index]
        except IndexError as exc:
            raise IndexError(f"unknown cache group index {group_index}") from exc
        packing = group.cache_blocks_per_lcm_block
        max_block_id = self.num_host_lcm_blocks * packing
        if block_id <= 0 or block_id > max_block_id:
            raise IndexError(
                f"block_id {block_id} outside [1, {max_block_id}] for "
                f"group {group.group_id!r}"
            )
        zero_based = block_id - 1
        parent_index, child_index = divmod(zero_based, packing)
        return (
            parent_index * self.host_lcm_block_bytes
            + child_index * host_cache_block_bytes
        )

    def host_field_offset(
        self, group_index: int, block_id: int, field_index: int
    ) -> int:
        try:
            host_field_offset = self.host_field_offsets[group_index][field_index]
        except IndexError as exc:
            raise IndexError(
                f"unknown field {field_index} for cache group {group_index}"
            ) from exc
        return self.host_block_offset(group_index, block_id) + host_field_offset
