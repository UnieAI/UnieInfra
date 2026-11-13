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

from tokenspeed.runtime.execution.forward_thread import ForwardThread, mark_launch
from tokenspeed.runtime.execution.types import (
    ModelExecutionResult,
    PendingExecution,
)


@pytest.fixture
def thread():
    ft = ForwardThread(torch.device("cpu"), hold_results_until_next_launch=False)
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
    ft = ForwardThread(torch.device("cpu"), hold_results_until_next_launch=False)
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
    ft = ForwardThread(torch.device("cpu"), hold_results_until_next_launch=False)
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


# Holding a forward's result until the next forward launches.


@pytest.fixture
def holding():
    ft = ForwardThread(torch.device("cpu"), hold_results_until_next_launch=True)
    yield ft
    ft.shutdown()


def _queue_behind(thread, gate):
    """Block the thread on ``gate`` so later submissions queue up together."""
    return thread.submit(gate.wait)


def test_held_result_released_when_next_forward_launches(holding):
    gate = threading.Event()
    _queue_behind(holding, gate)
    seen = {}
    first = holding.submit_forward(lambda: "first")

    def prep():
        # The next round's zeroing runs while the first result is held.
        seen["during_prep"] = first.done()

    def second():
        seen["before_launch"] = first.done()
        mark_launch()
        seen["after_launch"] = first.done()
        return "second"

    holding.submit_prep(prep)
    last = holding.submit_forward(second)
    gate.set()
    assert last.result(timeout=5) == "second"
    assert first.result(timeout=5) == "first"
    assert seen == {"during_prep": False, "before_launch": False, "after_launch": True}


def test_held_result_released_when_queue_drains(holding):
    # Nothing queued behind: the control plane may need this result to plan.
    assert holding.submit_forward(lambda: 7).result(timeout=5) == 7


def test_held_result_released_by_queue_drain_after_prep(holding):
    gate = threading.Event()
    _queue_behind(holding, gate)
    first = holding.submit_forward(lambda: "first")
    prep = holding.submit_prep(lambda: "zeroed")
    gate.set()
    assert prep.result(timeout=5) == "zeroed"
    assert first.result(timeout=5) == "first"


def test_plain_task_releases_held_result_before_it_runs(holding):
    gate = threading.Event()
    _queue_behind(holding, gate)
    first = holding.submit_forward(lambda: "first")
    # A plain task may wait on the control plane, which waits on ``first``.
    plain = holding.submit(lambda: first.result(timeout=5))
    gate.set()
    assert plain.result(timeout=5) == "first"


def test_results_stay_fifo_when_a_forward_never_launches(holding):
    gate = threading.Event()
    _queue_behind(holding, gate)
    order = []
    first = holding.submit_forward(lambda: "first")
    second = holding.submit_forward(lambda: "second")
    first.add_done_callback(lambda f: order.append(f.result()))
    second.add_done_callback(lambda f: order.append(f.result()))
    gate.set()
    assert second.result(timeout=5) == "second"
    assert order == ["first", "second"]


def test_failing_task_releases_held_result(holding):
    gate = threading.Event()
    _queue_behind(holding, gate)
    first = holding.submit_forward(lambda: "first")
    bad = holding.submit_prep(lambda: 1 / 0)
    blocker = threading.Event()
    # Keeps the queue non-empty until the failure has been observed.
    holding.submit_prep(lambda: blocker.wait(5))
    gate.set()
    with pytest.raises(ZeroDivisionError):
        bad.result(timeout=5)
    assert first.result(timeout=5) == "first"
    blocker.set()


def test_without_holding_a_forward_result_resolves_immediately(thread):
    gate = threading.Event()
    first = thread.submit_forward(lambda: "first")
    thread.submit_prep(gate.wait)
    assert first.result(timeout=5) == "first"
    gate.set()


def test_mark_launch_off_the_forward_thread_is_a_noop():
    mark_launch()
