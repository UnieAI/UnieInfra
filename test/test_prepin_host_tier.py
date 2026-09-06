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

"""Which configurations may take the pin before the KV budget is profiled.

The pin has to land inside the memory profile or it is spent out of headroom
the budget already handed to the cache -- that is how gemma-4-31B died on
5 of 10 tiered attempts. But it can only move earlier when its size is known
without the device pool, which is exactly the explicit `--kvstore-size` case.
Getting that predicate wrong in either direction is silent: too eager and
ratio mode pins the wrong number of bytes, too shy and the crash comes back.
"""

from __future__ import annotations

import pathlib
import types

import pytest

# Importing executor.py pulls in tokenspeed_kernel, whose platform detection
# refuses to load without a visible GPU -- and making one visible to a unit
# test risks a third CUDA context on a box whose two cards are running
# measurements. Load the one function by source instead, the way
# test/runtime/test_unified_arena_views.py loads `external_field_view`.
_SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "cache"
    / "l2"
    / "executor.py"
).read_text(encoding="utf-8")


def _load(namespace):
    start = _SRC.index("def prepin_host_tier(")
    end = _SRC.index("\n# One load carries up to a few GB of ranges.")
    exec(compile(_SRC[start:end], "executor.py", "exec"), namespace)  # noqa: S102
    return namespace["prepin_host_tier"]


class _Recorder(types.SimpleNamespace):
    """The module globals prepin_host_tier reads, with the pins recorded."""

    def __init__(self, available=1 << 60):
        self.taken = []
        torch = types.SimpleNamespace(
            uint8=object(),
            cuda=types.SimpleNamespace(is_available=lambda: False),
            empty=self._empty,
        )
        psutil = types.SimpleNamespace(
            virtual_memory=lambda: types.SimpleNamespace(available=available)
        )
        super().__init__(
            torch=torch,
            psutil=psutil,
            logger=types.SimpleNamespace(info=lambda *a, **k: None),
            _HOST_MEM_HEADROOM_BYTES=0,
        )

    def _empty(self, n, *, dtype, pin_memory):
        assert pin_memory is True
        self.taken.append(n)
        return types.SimpleNamespace(numel=lambda: n)


@pytest.fixture
def env():
    return _Recorder()


def test_an_explicit_size_is_pinned_up_front(env):
    buffer = _load(vars(env))(
        enable_kvstore=True, host_size_gb=227, backing_path=None
    )
    assert buffer is not None
    assert env.taken == [int(227 * 1e9)]


def test_ratio_mode_is_left_alone(env):
    """Ratio sizes off the device pool's block count, which does not exist yet."""
    assert (
        _load(vars(env))(enable_kvstore=True, host_size_gb=0, backing_path=None)
        is None
    )
    assert env.taken == []


def test_a_file_backed_tier_is_left_alone(env):
    """It pins only its staging buffer; 6 of 6 file-backed cells already pass."""
    assert (
        _load(vars(env))(
            enable_kvstore=True, host_size_gb=227, backing_path="/tmp/kv.bin"
        )
        is None
    )
    assert env.taken == []


def test_a_disabled_tier_is_left_alone(env):
    assert (
        _load(vars(env))(
            enable_kvstore=False, host_size_gb=227, backing_path=None
        )
        is None
    )
    assert env.taken == []


def test_a_host_shortage_is_refused_before_anything_is_pinned():
    """Fail with the size in the message, not by page-locking the box to death."""
    env = _Recorder(available=1 << 30)
    env.torch.empty = lambda *a, **k: pytest.fail("pinned despite the shortage")
    with pytest.raises(ValueError, match="Not enough Host memory"):
        _load(vars(env))(
            enable_kvstore=True, host_size_gb=227, backing_path=None
        )
