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

"""Free device memory on an integrated GPU counts reclaimable page cache."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.utils import common

GIB = 1 << 30


def _fake_cuda(*, free_bytes: int, is_integrated: bool) -> SimpleNamespace:
    return SimpleNamespace(
        device_count=lambda: 1,
        current_device=lambda: 0,
        empty_cache=lambda: None,
        mem_get_info=lambda gpu_id: (free_bytes, 128 * GIB),
        get_device_properties=lambda gpu_id: SimpleNamespace(
            is_integrated=is_integrated
        ),
    )


@pytest.mark.parametrize(
    "is_integrated, expected_gib",
    [
        # GB10 after loading a checkpoint: MemFree 7 GiB, 55 GiB of page cache.
        (True, 62.0),
        # A discrete GPU's free memory has nothing to do with host page cache.
        (False, 7.0),
    ],
)
def test_integrated_gpu_counts_reclaimable_page_cache(
    monkeypatch, is_integrated, expected_gib
):
    fake = _fake_cuda(free_bytes=7 * GIB, is_integrated=is_integrated)
    monkeypatch.setattr(torch, "get_device_module", lambda device: fake)
    monkeypatch.setattr(
        common.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=62 * GIB),
    )

    assert common.get_available_gpu_memory("cuda", 0) == pytest.approx(expected_gib)
