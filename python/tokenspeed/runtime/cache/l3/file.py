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

"""Local-disk L3 store: one file per packed Host CacheBlock on an NVMe path.

Each object is a 4 KiB header (magic, payload size, CRC32, the full key) and
the payload padded to 4 KiB, so the file can be read and written with
O_DIRECT. Reads and writes of one ``batch_*`` call run in parallel on a
thread pool (``os.preadv`` / ``os.pwritev`` release the GIL). When a Host
slice is 4 KiB aligned in address and size, the transfer goes straight
between the file and the pinned Host buffer; otherwise it goes through an
aligned per-thread bounce buffer.

Puts are atomic (written to a temporary name, then renamed), so a reader
never sees a partial object; a torn file left by a crash fails the size or
CRC check and reads as a miss. The key → file index lives in memory and is
rebuilt from the headers at start-up, so objects survive a restart.
Capacity is bounded: a put that does not fit evicts least-recently-used
objects first.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import logging
import mmap
import os
import struct
import threading
import zlib
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from tokenspeed.runtime.cache.l3.backend import host_buffer_ptr

logger = logging.getLogger(__name__)

ALIGN = 4096
HEADER_BYTES = ALIGN
_MAGIC = b"TSL3FIL1"
# magic, payload bytes, CRC32 of the payload (0 when not verified), key bytes
_HEADER = struct.Struct("<8sQII")
_MAX_KEY_BYTES = HEADER_BYTES - _HEADER.size
_SUFFIX = ".kv"


def _round_up(value: int, multiple: int = ALIGN) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class FileStoreConfig:
    """Settings of the local-disk L3 store; every field is required.

    ``path``: directory for the objects (a local NVMe file system).
    ``capacity_gb``: bytes on disk each rank may use, in GiB, headers and
    padding included; least-recently-used objects are evicted beyond it.
    ``io_threads``: parallel reads/writes per ``batch_*`` call.
    ``o_direct``: bypass the page cache (start-up fails if the file system
    refuses O_DIRECT).
    ``verify_crc``: store a CRC32 per object and reject reads that mismatch.
    ``fsync``: flush each object to the device before it becomes visible.
    """

    path: str
    capacity_gb: float
    io_threads: int
    o_direct: bool
    verify_crc: bool
    fsync: bool

    @property
    def capacity_bytes(self) -> int:
        return int(self.capacity_gb * (1 << 30))


def parse_file_store_config(extra_config: str | None) -> FileStoreConfig:
    """Parse ``--kvstore-storage-backend-extra-config`` for the file store."""

    if not extra_config:
        raise ValueError(
            "the file L3 store needs --kvstore-storage-backend-extra-config with "
            '"path", "capacity_gb", "io_threads", "o_direct", "verify_crc", "fsync"'
        )
    raw = json.loads(extra_config)
    if not isinstance(raw, dict):
        raise ValueError("file L3 extra config must be a JSON object")
    fields = {
        "path": str,
        "capacity_gb": (int, float),
        "io_threads": int,
        "o_direct": bool,
        "verify_crc": bool,
        "fsync": bool,
    }
    missing = [name for name in fields if name not in raw]
    unknown = sorted(set(raw) - set(fields))
    if missing or unknown:
        raise ValueError(f"file L3 extra config: missing {missing}, unknown {unknown}")
    for name, kind in fields.items():
        value = raw[name]
        # bool is an int subclass: keep the numeric fields numeric.
        if not isinstance(value, kind) or (
            kind is not bool and isinstance(value, bool)
        ):
            raise ValueError(f"file L3 extra config: {name!r} has the wrong type")
    if raw["capacity_gb"] <= 0 or raw["io_threads"] <= 0:
        raise ValueError("file L3 extra config: capacity_gb and io_threads must be > 0")
    return FileStoreConfig(**raw)


class _AlignedBuffer:
    """A page-aligned anonymous mapping that grows on demand (one per thread)."""

    def __init__(self) -> None:
        self._map: mmap.mmap | None = None
        self.address = 0
        self.size = 0

    def ensure(self, size: int) -> None:
        if size <= self.size:
            return
        self._map = mmap.mmap(-1, _round_up(size))
        self.size = len(self._map)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self._map))

    def view(self, size: int, offset: int = 0) -> memoryview:
        return memoryview(self._map)[offset : offset + size]


def _host_view(address: int, size: int) -> memoryview:
    return memoryview((ctypes.c_char * size).from_address(address)).cast("B")


def _read_full(fd: int, views: list[memoryview], offset: int) -> int:
    """preadv until every view is filled or EOF; return bytes read."""

    total = 0
    want = sum(len(v) for v in views)
    while total < want:
        remaining = []
        skip = total
        for view in views:
            if skip >= len(view):
                skip -= len(view)
                continue
            remaining.append(view[skip:])
            skip = 0
        got = os.preadv(fd, remaining, offset + total)
        if got <= 0:
            break
        total += got
    return total


def _write_full(fd: int, views: list[memoryview]) -> None:
    total = 0
    want = sum(len(v) for v in views)
    while total < want:
        remaining = []
        skip = total
        for view in views:
            if skip >= len(view):
                skip -= len(view)
                continue
            remaining.append(view[skip:])
            skip = 0
        written = os.pwritev(fd, remaining, total)
        if written <= 0:
            raise OSError(errno.EIO, "short write to L3 file")
        total += written


class FileKvStore:
    """L3 byte store on a local directory (see the module docstring)."""

    def __init__(self, config: FileStoreConfig, *, rank: int) -> None:
        self.config = config
        # One subdirectory per rank: each rank indexes, sizes and evicts only
        # its own objects even when ranks share the configured path.
        self.root = os.path.join(config.path, f"rank{int(rank)}")
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        self._flags = os.O_DIRECT if config.o_direct else 0
        if config.o_direct:
            self._check_o_direct()
        self._lock = threading.Lock()
        # key -> on-disk bytes, least recently used first
        self._index: OrderedDict[str, int] = OrderedDict()
        self._used = 0
        self._local = threading.local()
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=config.io_threads, thread_name_prefix="l3-file"
        )
        self._rebuild_index()

    # -- layout -------------------------------------------------------------

    def _path(self, key: str) -> str:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.root, digest[:2], digest + _SUFFIX)

    def _buffer(self) -> _AlignedBuffer:
        buf = getattr(self._local, "buffer", None)
        if buf is None:
            buf = _AlignedBuffer()
            self._local.buffer = buf
        return buf

    def _check_o_direct(self) -> None:
        probe = os.path.join(self.root, ".o_direct_probe")
        try:
            fd = os.open(
                probe, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o600
            )
        except OSError as exc:
            raise ValueError(
                f"file L3: {self.config.path!r} does not support O_DIRECT "
                f'({exc}); set "o_direct": false'
            ) from exc
        os.close(fd)
        os.unlink(probe)

    def _rebuild_index(self) -> None:
        """Index the objects already on disk; drop temporaries and torn files."""

        found = []
        for root, _dirs, files in os.walk(self.root):
            for name in files:
                path = os.path.join(root, name)
                if not name.endswith(_SUFFIX):
                    if ".tmp." in name:
                        os.unlink(path)
                    continue
                header = self._read_header_file(path)
                if header is None:
                    os.unlink(path)
                    continue
                key, payload = header
                found.append(
                    (os.stat(path).st_mtime, key, HEADER_BYTES + _round_up(payload))
                )
        for _mtime, key, disk in sorted(found):
            self._index[key] = disk
            self._used += disk
        if found:
            logger.info(
                f"[L3 file] indexed {len(found)} objects, "
                f"{self._used / (1 << 30):.2f} GiB under {self.root}"
            )

    def _read_header_file(self, path: str) -> tuple[str, int] | None:
        try:
            fd = os.open(path, os.O_RDONLY | self._flags)
        except OSError:
            return None
        try:
            buf = self._buffer()
            buf.ensure(HEADER_BYTES)
            if _read_full(fd, [buf.view(HEADER_BYTES)], 0) != HEADER_BYTES:
                return None
            return self._parse_header(buf.view(HEADER_BYTES))
        finally:
            os.close(fd)

    @staticmethod
    def _parse_header(view: memoryview) -> tuple[str, int] | None:
        magic, payload, _crc, key_len = _HEADER.unpack_from(view, 0)
        if magic != _MAGIC or key_len > _MAX_KEY_BYTES:
            return None
        key = bytes(view[_HEADER.size : _HEADER.size + key_len]).decode(
            errors="replace"
        )
        return key, payload

    # -- KvStoreStorage ---------------------------------------------------------

    def batch_exists(self, keys: Sequence[str]) -> list[bool]:
        with self._lock:
            results = []
            for key in keys:
                present = key in self._index
                if present:
                    # A probe precedes a prefetch: keep the object from being
                    # evicted between the two.
                    self._index.move_to_end(key)
                results.append(present)
            return results

    def batch_get_into(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        if not (len(keys) == len(offsets) == len(sizes)):
            raise ValueError("ragged L3 get")
        base = host_buffer_ptr(host_buffer)
        return list(
            self._run(
                self._get_one,
                [
                    (key, base + int(off), int(size))
                    for key, off, size in zip(keys, offsets, sizes)
                ],
            )
        )

    def batch_put_from(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        if not (len(keys) == len(offsets) == len(sizes)):
            raise ValueError("ragged L3 put")
        base = host_buffer_ptr(host_buffer)
        results = [False] * len(keys)
        todo = []
        with self._lock:
            batch = set(keys)
            for i, (key, off, size) in enumerate(zip(keys, offsets, sizes)):
                if key in self._index:
                    self._index.move_to_end(key)
                    results[i] = True
                    continue
                if len(key.encode()) > _MAX_KEY_BYTES:
                    continue
                disk = HEADER_BYTES + _round_up(int(size))
                if not self._make_room(disk, keep=batch):
                    continue
                # Reserve the bytes now so concurrent puts cannot overcommit.
                self._used += disk
                todo.append((i, key, base + int(off), int(size), disk))
        outcomes = self._run(
            self._put_one, [(key, addr, size) for _, key, addr, size, _ in todo]
        )
        with self._lock:
            for (i, key, _addr, _size, disk), ok in zip(todo, outcomes):
                if ok:
                    self._index[key] = disk
                    results[i] = True
                else:
                    self._used -= disk
        return results

    def remove_by_prefix(self, prefix: str) -> bool:
        with self._lock:
            victims = [key for key in self._index if key.startswith(prefix)]
            for key in victims:
                self._used -= self._index.pop(key)
        ok = True
        for key in victims:
            try:
                os.unlink(self._path(key))
            except FileNotFoundError:
                pass
            except OSError:
                logger.exception(f"[L3 file] failed to delete {key!r}")
                ok = False
        return ok

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def used_bytes(self) -> int:
        with self._lock:
            return self._used

    # -- internals --------------------------------------------------------------

    def _run(self, fn, items: list[tuple]) -> list[bool]:
        if not items:
            return []
        if self._pool is None:
            raise RuntimeError("file L3 store is closed")
        if len(items) == 1:
            return [fn(*items[0])]
        return list(self._pool.map(lambda args: fn(*args), items))

    def _make_room(self, disk: int, *, keep: set[str]) -> bool:
        """Evict LRU objects (not in ``keep``) until ``disk`` more bytes fit.

        Called with the lock held. Files are unlinked here too: an evicted
        key is gone from the index first, so a concurrent get misses cleanly.
        """

        capacity = self.config.capacity_bytes
        if disk > capacity:
            return False
        if self._used + disk <= capacity:
            return True
        for key in list(self._index):
            if self._used + disk <= capacity:
                break
            if key in keep:
                continue
            self._used -= self._index.pop(key)
            try:
                os.unlink(self._path(key))
            except FileNotFoundError:
                pass
        return self._used + disk <= capacity

    def _get_one(self, key: str, address: int, size: int) -> bool:
        with self._lock:
            disk = self._index.get(key)
            if disk is None or disk != HEADER_BYTES + _round_up(size):
                return False
            self._index.move_to_end(key)
        try:
            fd = os.open(self._path(key), os.O_RDONLY | self._flags)
        except FileNotFoundError:
            return False
        try:
            buf = self._buffer()
            padded = _round_up(size)
            direct = address % ALIGN == 0 and size % ALIGN == 0
            if direct:
                buf.ensure(HEADER_BYTES)
                views = [buf.view(HEADER_BYTES), _host_view(address, size)]
            else:
                buf.ensure(HEADER_BYTES + padded)
                views = [buf.view(HEADER_BYTES + padded)]
            if _read_full(fd, views, 0) != HEADER_BYTES + padded:
                return False
            magic, payload, crc, key_len = _HEADER.unpack_from(
                buf.view(HEADER_BYTES), 0
            )
            stored_key = (
                bytes(buf.view(key_len, _HEADER.size))
                if key_len <= _MAX_KEY_BYTES
                else b""
            )
            if magic != _MAGIC or payload != size or stored_key != key.encode():
                return False
            data = views[1] if direct else buf.view(size, HEADER_BYTES)
            if self.config.verify_crc and zlib.crc32(data) != crc:
                return False
            if not direct:
                ctypes.memmove(address, buf.address + HEADER_BYTES, size)
            return True
        except OSError:
            logger.exception(f"[L3 file] read failed for {key!r}")
            return False
        finally:
            os.close(fd)

    def _put_one(self, key: str, address: int, size: int) -> bool:
        path = self._path(key)
        tmp = f"{path}.tmp.{threading.get_ident()}"
        source = _host_view(address, size)
        crc = zlib.crc32(source) if self.config.verify_crc else 0
        key_bytes = key.encode()
        buf = self._buffer()
        padded = _round_up(size)
        direct = address % ALIGN == 0 and size % ALIGN == 0
        buf.ensure(HEADER_BYTES if direct else HEADER_BYTES + padded)
        header = buf.view(HEADER_BYTES)
        header[:] = bytes(HEADER_BYTES)
        _HEADER.pack_into(header, 0, _MAGIC, size, crc, len(key_bytes))
        header[_HEADER.size : _HEADER.size + len(key_bytes)] = key_bytes
        if direct:
            views = [header, source]
        else:
            body = buf.view(padded, HEADER_BYTES)
            ctypes.memmove(buf.address + HEADER_BYTES, address, size)
            if padded > size:
                body[size:] = bytes(padded - size)
            views = [buf.view(HEADER_BYTES + padded)]
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        try:
            fd = os.open(
                tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | self._flags, 0o600
            )
            try:
                _write_full(fd, views)
                if self.config.fsync:
                    os.fdatasync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path)
            return True
        except OSError:
            logger.exception(f"[L3 file] write failed for {key!r}")
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            return False
