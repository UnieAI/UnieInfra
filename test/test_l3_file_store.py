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

"""Local-disk L3 store: round trips, restart, integrity, capacity and config."""

from __future__ import annotations

import json
import os

import pytest
import torch

from tokenspeed.runtime.cache.l3.factory import create_kvstore_storage_backend
from tokenspeed.runtime.cache.l3.file import (
    ALIGN,
    HEADER_BYTES,
    FileKvStore,
    FileStoreConfig,
    parse_file_store_config,
)


def _config(
    path, *, capacity_gb=1.0, o_direct=True, verify_crc=True
) -> FileStoreConfig:
    return FileStoreConfig(
        path=str(path),
        capacity_gb=capacity_gb,
        io_threads=4,
        o_direct=o_direct,
        verify_crc=verify_crc,
        fsync=False,
    )


def _aligned_host(nbytes: int) -> tuple[torch.Tensor, int]:
    """A uint8 host buffer and the first offset whose address is 4 KiB aligned."""

    buf = torch.empty(nbytes + ALIGN, dtype=torch.uint8)
    return buf, (-buf.data_ptr()) % ALIGN


def _pattern(n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n,), dtype=torch.uint8, generator=gen)


@pytest.mark.parametrize("o_direct", [True, False])
@pytest.mark.parametrize(
    "skew, size",
    [(0, 3 * ALIGN), (0, 3 * ALIGN + 100), (64, 3 * ALIGN), (7, 1000)],
)
def test_round_trip_into_aligned_and_unaligned_host_slices(
    tmp_path, o_direct, skew, size
):
    store = FileKvStore(_config(tmp_path, o_direct=o_direct), rank=0)
    src, base = _aligned_host(4 * size + 4 * ALIGN)
    keys = [f"k{i}" for i in range(3)]
    offsets = [base + skew + i * _round(size) for i in range(3)]
    for i, off in enumerate(offsets):
        src[off : off + size] = _pattern(size, i)
    assert store.batch_put_from(keys, src, offsets, [size] * 3) == [True] * 3
    assert store.batch_exists(keys + ["missing"]) == [True, True, True, False]

    dst, dbase = _aligned_host(4 * size + 4 * ALIGN)
    dst.zero_()
    doffs = [dbase + skew + i * _round(size) for i in range(3)]
    assert store.batch_get_into(keys, dst, doffs, [size] * 3) == [True] * 3
    for i, off in enumerate(doffs):
        assert torch.equal(dst[off : off + size], _pattern(size, i))
    # A get for the wrong size is a miss and leaves the slice untouched.
    dst.zero_()
    assert store.batch_get_into(["k0"], dst, [doffs[0]], [size + ALIGN]) == [False]
    assert int(dst.sum()) == 0
    store.close()


def _round(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def test_objects_survive_a_restart_and_temporaries_are_dropped(tmp_path):
    store = FileKvStore(_config(tmp_path), rank=0)
    src, base = _aligned_host(2 * ALIGN)
    src[base : base + 5000] = _pattern(5000, 7)
    assert store.batch_put_from(["persist"], src, [base], [5000]) == [True]
    store.close()
    stray = os.path.join(str(tmp_path), "rank0", "ab", "x.kv.tmp.1")
    os.makedirs(os.path.dirname(stray), exist_ok=True)
    open(stray, "wb").write(b"partial")

    again = FileKvStore(_config(tmp_path), rank=0)
    assert again.batch_exists(["persist"]) == [True]
    assert not os.path.exists(stray)
    dst, dbase = _aligned_host(2 * ALIGN)
    assert again.batch_get_into(["persist"], dst, [dbase], [5000]) == [True]
    assert torch.equal(dst[dbase : dbase + 5000], _pattern(5000, 7))
    assert again.used_bytes() == HEADER_BYTES + _round(5000)
    again.close()


def _object_file(store: FileKvStore, key: str) -> str:
    return store._path(key)


def test_corrupted_payload_reads_as_a_miss(tmp_path):
    store = FileKvStore(_config(tmp_path), rank=0)
    src, base = _aligned_host(2 * ALIGN)
    src[base : base + ALIGN] = _pattern(ALIGN, 1)
    assert store.batch_put_from(["bad"], src, [base], [ALIGN]) == [True]
    path = _object_file(store, "bad")
    with open(path, "r+b") as f:
        f.seek(HEADER_BYTES + 10)
        byte = f.read(1)
        f.seek(HEADER_BYTES + 10)
        f.write(bytes([byte[0] ^ 0xFF]))
    dst, dbase = _aligned_host(2 * ALIGN)
    assert store.batch_get_into(["bad"], dst, [dbase], [ALIGN]) == [False]
    store.close()


def test_truncated_file_is_dropped_at_restart(tmp_path):
    store = FileKvStore(_config(tmp_path), rank=0)
    src, base = _aligned_host(2 * ALIGN)
    assert store.batch_put_from(["torn"], src, [base], [ALIGN]) == [True]
    path = _object_file(store, "torn")
    store.close()
    with open(path, "r+b") as f:
        f.truncate(100)
    again = FileKvStore(_config(tmp_path), rank=0)
    assert again.batch_exists(["torn"]) == [False]
    assert not os.path.exists(path)
    again.close()


def test_capacity_evicts_least_recently_used(tmp_path):
    per_object = HEADER_BYTES + ALIGN
    capacity_gb = 3 * per_object / (1 << 30)
    store = FileKvStore(_config(tmp_path, capacity_gb=capacity_gb), rank=0)
    src, base = _aligned_host(2 * ALIGN)
    for key in ("a", "b", "c"):
        assert store.batch_put_from([key], src, [base], [ALIGN]) == [True]
    store.batch_exists(["a"])  # touch: "b" is now the oldest
    assert store.batch_put_from(["d"], src, [base], [ALIGN]) == [True]
    assert store.batch_exists(["a", "b", "c", "d"]) == [True, False, True, True]
    assert not os.path.exists(_object_file(store, "b"))
    assert store.used_bytes() == 3 * per_object
    # An object larger than the whole capacity is refused, nothing evicted.
    big, bbase = _aligned_host(4 * per_object)
    assert store.batch_put_from(["huge"], big, [bbase], [4 * per_object]) == [False]
    assert store.batch_exists(["a", "c", "d"]) == [True, True, True]
    store.close()


def test_remove_by_prefix_and_rank_isolation(tmp_path):
    rank0 = FileKvStore(_config(tmp_path), rank=0)
    rank1 = FileKvStore(_config(tmp_path), rank=1)
    src, base = _aligned_host(2 * ALIGN)
    assert rank0.batch_put_from(["ns1_x", "ns2_y"], src, [base, base], [64, 64]) == [
        True,
        True,
    ]
    assert rank1.batch_put_from(["ns1_x"], src, [base], [64]) == [True]
    assert rank0.remove_by_prefix("ns1_") is True
    assert rank0.batch_exists(["ns1_x", "ns2_y"]) == [False, True]
    assert rank1.batch_exists(["ns1_x"]) == [True]
    rank0.close()
    rank1.close()
    assert FileKvStore(_config(tmp_path), rank=1).used_bytes() == HEADER_BYTES + ALIGN


def test_config_requires_every_field(tmp_path):
    full = {
        "path": str(tmp_path),
        "capacity_gb": 2,
        "io_threads": 8,
        "o_direct": True,
        "verify_crc": True,
        "fsync": False,
    }
    assert parse_file_store_config(json.dumps(full)).io_threads == 8
    for name in full:
        partial = {k: v for k, v in full.items() if k != name}
        with pytest.raises(ValueError, match="missing"):
            parse_file_store_config(json.dumps(partial))
    with pytest.raises(ValueError, match="unknown"):
        parse_file_store_config(json.dumps({**full, "threads": 4}))
    with pytest.raises(ValueError, match="wrong type"):
        parse_file_store_config(json.dumps({**full, "o_direct": "yes"}))
    with pytest.raises(ValueError, match="wrong type"):
        parse_file_store_config(json.dumps({**full, "io_threads": True}))
    with pytest.raises(ValueError, match="> 0"):
        parse_file_store_config(json.dumps({**full, "capacity_gb": 0}))
    with pytest.raises(ValueError):
        parse_file_store_config(None)


def test_factory_builds_the_file_store(tmp_path):
    extra = json.dumps(
        {
            "path": str(tmp_path),
            "capacity_gb": 1,
            "io_threads": 2,
            "o_direct": False,
            "verify_crc": True,
            "fsync": True,
        }
    )
    backend = create_kvstore_storage_backend(
        "file", extra, host_buffer=object(), tp_size=1, pp_size=1, rank=3
    )
    assert isinstance(backend, FileKvStore)
    assert backend.root == os.path.join(str(tmp_path), "rank3")
    backend.close()
