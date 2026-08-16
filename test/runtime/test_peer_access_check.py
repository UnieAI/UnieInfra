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

"""A multi-GPU launch on non-peer devices must be refused at startup.

Without the check the run loads weights, initialises collectives, and dies
inside CUDA-graph capture with ``CUDART error: peer access is not supported
between these two devices`` -- minutes later, naming neither the offending
devices nor a pairing that would work. On a PCIe box only devices behind the
same switch peer, so the operator has to choose; the engine's job is to say so
before the load, and to say which pairs are usable.
"""

import types

import pytest

from tokenspeed.runtime.entrypoints.engine import _check_peer_access


def _args(world_size):
    return types.SimpleNamespace(mapping=types.SimpleNamespace(world_size=world_size))


class _FakeCuda:
    """Stands in for torch.cuda with a peer matrix we control."""

    def __init__(self, count, peers):
        self._count = count
        self._peers = peers

    def is_available(self):
        return True

    def device_count(self):
        return self._count

    def can_device_access_peer(self, i, j):
        return (i, j) in self._peers or (j, i) in self._peers


@pytest.fixture
def fake_torch(monkeypatch):
    def install(count, peers):
        fake = types.ModuleType("torch")
        fake.cuda = _FakeCuda(count, peers)
        monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    return install


def test_single_gpu_is_never_refused(fake_torch):
    """world_size 1 must not consult the topology at all."""
    fake_torch(4, peers=set())  # nothing peers, and it still must not raise
    _check_peer_access(_args(1))


def test_all_peers_passes(fake_torch):
    fake_torch(2, peers={(0, 1)})
    _check_peer_access(_args(2))


def test_non_peer_pair_is_refused_before_the_load(fake_torch):
    fake_torch(2, peers=set())
    with pytest.raises(ValueError) as excinfo:
        _check_peer_access(_args(2))
    assert "(0, 1)" in str(excinfo.value)


def test_the_error_names_a_pairing_that_would_work(fake_torch, monkeypatch):
    """The point of failing early is telling the operator what to do instead.

    Mirrors this box: 0-1 and 2-3 sit behind the same switch and peer, the
    cross pairs do not. An error that only says "not supported" sends someone
    back to trial and error.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    fake_torch(4, peers={(0, 1), (2, 3)})
    with pytest.raises(ValueError) as excinfo:
        _check_peer_access(_args(4))
    message = str(excinfo.value)
    assert "(0, 1)" in message and "(2, 3)" in message
    assert "0,1,2,3" in message


def test_a_broken_topology_query_does_not_block_startup(monkeypatch):
    """The check is a diagnostic; it must never be the reason a launch fails."""
    fake = types.ModuleType("torch")

    class _Boom:
        def is_available(self):
            raise RuntimeError("no driver")

    fake.cuda = _Boom()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    _check_peer_access(_args(2))
