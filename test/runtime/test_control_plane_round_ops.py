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


"""Per-round control-plane helpers that must stay free of torch ops."""

from __future__ import annotations

import os
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._python_dispatch import TorchDispatchMode

from tokenspeed.runtime.engine.scheduler_utils import profiler_range
from tokenspeed.runtime.utils.common import PipelinedPyobjBroadcaster

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")


class _CountOps(TorchDispatchMode):
    """Counts the torch ops dispatched on this thread, like the
    control-plane guard sees them."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def test_profiler_range_runs_no_torch_op_outside_a_profile():
    with _CountOps() as counter:
        with profiler_range("scheduler.advance"):
            pass
    assert counter.ops == []


def test_profiler_range_labels_the_span_while_a_profile_records():
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        with profiler_range("scheduler.advance"):
            pass
    assert "scheduler.advance" in {event.key for event in prof.key_averages()}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _broadcast_rounds(rank: int, world_size: int, port: int, results) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        broadcaster = PipelinedPyobjBroadcaster(rank, dist.group.WORLD, src=0)
        received = []
        for payload in ([{"rid": "a"}, 7], [], None, ["b"]):
            broadcaster.start(payload if rank == 0 else None)
            assert broadcaster.in_flight
            received.append(broadcaster.finish())
            assert not broadcaster.in_flight
        results[rank] = received
    finally:
        dist.destroy_process_group()


def test_pipelined_broadcaster_delivers_each_round_to_every_rank():
    """Non-empty rounds reach the peer whole; empty and None rounds are []
    on the peer (the source keeps its own empty value)."""
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_broadcast_rounds, args=(2, _free_port(), results), nprocs=2)

    assert results[0] == [[{"rid": "a"}, 7], [], [], ["b"]]
    assert results[1] == [[{"rid": "a"}, 7], [], [], ["b"]]
