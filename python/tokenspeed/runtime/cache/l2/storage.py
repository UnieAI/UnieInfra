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
            # Preallocate so the mapping is backed by real extents; a sparse
            # file turns every first write into an allocating fault under the
            # transfer stream.
            with open(backing_path, "wb") as fh:
                os.posix_fallocate(fh.fileno(), 0, self.host_buffer_bytes)
                # fallocate warms the whole file through the page cache, so a
                # residency measurement afterwards reads 100% before the engine
                # has touched anything. Start from cold or the number means
                # nothing.
                os.posix_fadvise(fh.fileno(), 0, self.host_buffer_bytes,
                                 os.POSIX_FADV_DONTNEED)
            self.host_buffer = torch.from_file(
                backing_path,
                shared=True,
                size=self.host_buffer_bytes,
                dtype=torch.uint8,
            )
            logger.info(
                "Host L2 backed by file %s (%.2f GB)",
                backing_path,
                self.host_buffer_bytes / 1e9,
            )
            # A mapped file is served out of the page cache, so on a box with
            # spare RAM the "disk" tier silently measures DRAM -- residency sat
            # at 100% of 226 GB. posix_fadvise from another process cannot fix
            # that: mapped pages are pinned by the mapping. Dropping them has to
            # come from inside this process, over the range just used.
            self._drop_after_use = (
                os.getenv("TOKENSPEED_KVSTORE_NO_PAGECACHE") == "1"
            )
            if self._drop_after_use:
                import ctypes
                import ctypes.util

                self._libc = ctypes.CDLL(ctypes.util.find_library("c"),
                                         use_errno=True)
                self._base_addr = self.host_buffer.data_ptr()
                self._backing_fd = os.open(backing_path, os.O_RDONLY)
                self._page_size = os.sysconf("SC_PAGESIZE")
                logger.info("Host L2 will drop page-cache pages after each use")

    _drop_after_use = False

    def drop_range(self, offset: int, num_bytes: int, *, dirty: bool) -> None:
        """Return one byte range to the file, so the next read hits the disk.

        No-op unless TOKENSPEED_KVSTORE_NO_PAGECACHE=1.

        MADV_DONTNEED alone does not do this. On a MAP_SHARED file mapping it
        only tears down this process's page-table entries; the page cache keeps
        the pages, so the next touch is a minor fault and the "disk" tier still
        serves out of RAM -- measured, residency stayed at 100% of 226 GB and
        the arm was slower purely from the msync. posix_fadvise is what frees
        them, and it skips any page still mapped, so it has to come second.

        `dirty` says whether this range was just written. Only then is a
        writeback needed before the pages can go; a read path that syncs pays a
        per-range flush for nothing.
        """

        if not self._drop_after_use:
            return
        MS_SYNC, MADV_DONTNEED, POSIX_FADV_DONTNEED = 4, 4, 4
        start = (self._base_addr + offset) & ~(self._page_size - 1)
        end = (self._base_addr + offset + num_bytes + self._page_size - 1) & ~(
            self._page_size - 1
        )
        length = end - start
        if length <= 0:
            return
        import ctypes

        if dirty:
            self._libc.msync(ctypes.c_void_p(start), ctypes.c_size_t(length),
                             MS_SYNC)
        self._libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(length),
                           MADV_DONTNEED)
        file_offset = start - self._base_addr
        self._libc.posix_fadvise(self._backing_fd, ctypes.c_long(file_offset),
                                 ctypes.c_long(length), POSIX_FADV_DONTNEED)

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
