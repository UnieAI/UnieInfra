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

"""ForwardThread (the data plane) and PendingExecution contract tests."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_thread import ForwardThread, LaunchGate
from tokenspeed.runtime.execution.types import (
    ModelExecutionResult,
    PendingExecution,
)


@pytest.fixture
def thread():
    ft = ForwardThread(torch.device("cpu"))
    yield ft
    ft.shutdown()


def test_fifo_order_and_results(thread):
    order = []
    futures = [thread.submit(lambda i=i: (order.append(i), i)[1]) for i in range(100)]
    assert [f.result() for f in futures] == list(range(100))
    assert order == list(range(100))


def test_exception_relayed_and_thread_survives(thread):
    with pytest.raises(ZeroDivisionError):
        thread.submit(lambda: 1 / 0).result()
    # The thread keeps serving after a failed item.
    assert thread.run(lambda: "alive") == "alive"


def test_submit_does_not_block_on_slow_work(thread):
    release = threading.Event()
    slow = thread.submit(release.wait)
    t0 = time.monotonic()
    fast = thread.submit(lambda: "queued")
    # Submission is O(queue put) even with the thread busy.
    assert time.monotonic() - t0 < 0.1
    assert not fast.done()
    release.set()
    assert slow.result() is True
    assert fast.result() == "queued"


def test_pending_execution_joins_future_then_copy_event():
    calls = []
    results = SimpleNamespace(sync=lambda: calls.append("sync"))
    ft = ForwardThread(torch.device("cpu"))
    try:
        pending = PendingExecution(ft.submit(lambda: (calls.append("run"), results)[1]))
        assert pending.result() is results
        assert calls == ["run", "sync"]
        # Memoized: a second commit-side call joins nothing.
        assert pending.result() is results
        assert calls == ["run", "sync"]
    finally:
        ft.shutdown()


def test_sync_is_rejected_after_the_pending_execution_synced():
    event = SimpleNamespace(synchronize=lambda: None)
    results = ModelExecutionResult(
        output_tokens=torch.tensor([7], dtype=torch.int32),
        copy_event=event,
    )
    ft = ForwardThread(torch.device("cpu"))
    try:
        assert PendingExecution(ft.submit(lambda: results)).result() is results
        with pytest.raises(RuntimeError, match="exactly once"):
            results.sync()
    finally:
        ft.shutdown()


def test_sync_requires_a_copy_event():
    results = ModelExecutionResult(output_tokens=torch.tensor([7], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="copy_event is required"):
        results.sync()


def test_launch_gate_releases_waiter_at_replay_not_at_forward_end(thread):
    gate = LaunchGate()
    first, second = gate.next_seq(), gate.next_seq()
    replay_entered = threading.Event()
    finish = threading.Event()

    def forward(seq):
        gate.begin(seq)
        try:
            gate.mark_launching()  # the replay site
            replay_entered.set()
            finish.wait(timeout=5)  # the blocking launch
        finally:
            gate.end(seq)

    thread.submit(lambda: forward(first))
    assert gate.wait_launched(first, timeout_s=5)
    assert replay_entered.is_set() and not finish.is_set()
    # The second forward has not started: its waiter times out.
    assert not gate.wait_launched(second, timeout_s=0.05)
    finish.set()
    thread.submit(lambda: forward(second))
    assert gate.wait_launched(second, timeout_s=5)


def test_launch_gate_marks_a_forward_that_raises_before_launch(thread):
    gate = LaunchGate()
    seq = gate.next_seq()

    def failing_forward():
        gate.begin(seq)
        try:
            raise RuntimeError("before launch")
        finally:
            gate.end(seq)

    future = thread.submit(failing_forward)
    assert gate.wait_launched(seq, timeout_s=5)
    with pytest.raises(RuntimeError, match="before launch"):
        future.result()
    # Outside a gated forward (capture, warmup) the replay site is a no-op.
    gate.mark_launching()
    assert not gate.wait_launched(gate.next_seq(), timeout_s=0.01)
