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

"""The per-rank forward thread: the engine's single data plane.

The event loop is the CONTROL plane: ZMQ input, gloo collectives, the C++
scheduler, and commit post-processing. Dispatching a round never waits on
the GPU — it submits and moves on, and only commit joins — so a round is
microseconds and the cross-rank collectives that keep the redundant
schedulers aligned (request broadcast, DP sync) always find every rank
promptly, no matter how deep the GPUs are in queued work. (``DeviceHandle``
does have blocking ``run_*`` operations, but only on rounds that dispatch
nothing — the DP idle forward — or off the round path entirely.)

Everything that touches CUDA is therefore submitted here and runs on ONE
thread per rank, in submission order:

- model forwards (eager launches, graph replays, idle forwards),
- pipeline-stage NCCL recv/send,
- KV page zeroing, and KV buffer repair after a memory-saver wake,
- the PD receive trigger and the writes a completed remote prefill lands,
- in-place weight updates from an RL trainer.

One thread, one ordering: NCCL collectives on a shared communicator are
issued in the same order on every rank because every rank enqueues the same
work sequence (the redundant schedulers plan identically), and a stage's
launch-queue backpressure or a blocking stage recv stalls only this thread,
never the control plane. This is the FluentLLM ForwardThread design point,
generalized to every mode: depth 0/1/pp_size only changes how long a result
future may stay unresolved, not where work runs.

Why a thread and not a process: the forward work reads the executor's device
buffers, CUDA graphs, and NCCL communicators in place. A process would need
its own CUDA context and communicator clones plus an IPC copy of every
batch's metadata — the cost and failure modes of a second engine. The GIL is
not a bottleneck here: the thread spends its time inside CUDA launches and
NCCL waits, which release the GIL.

The capture contract
--------------------

Sharing an address space is a convenience, not a licence: the two planes
exchange information ONLY through the two channels below, so that the day
this becomes a process the interface is already the whole interface.

1. Outbound is the closure, and only the closure. The control plane hands
   work over as ``submit(fn)`` / ``run(fn)`` arguments captured in ``fn``.
   Passing something by parking it on a shared object — a field on the
   executor, on an attention backend, on a request state — is not a channel;
   it is a race waiting for a deep enough queue.

2. Captured means frozen. Submission is fire-and-forget: once ``fn`` is
   queued, the control plane must not mutate what it captured — no attribute
   rebinding, no in-place edit, no releasing a resource the object holds.
   Capture CPU-plain values, or a snapshot the control plane will only read
   afterwards. Bind at capture time (``functools.partial``, or a default
   argument) rather than closing over a variable the caller will rebind — a
   bare lambda reads it at execution time, possibly a round later.

3. Inbound is the future, and only the future. Results reach the control
   plane through ``PendingExecution.result()``; nothing else crosses back.

4. Device-side state is not reachable from the control plane at all. The
   execution stream, ``runtime_states``, ``input_buffers``, and every
   attention backend's forward metadata live behind ``DeviceHandle``
   (``device.py``), which is what the event loop holds instead of the
   executor and which hands back no device object at all. Rules 1 and 2 are
   enforced by that shape rather than by anyone remembering them: what the
   loop cannot see, it cannot pass implicitly or mutate afterwards.

5. Two registered exceptions — and new ones belong in this list, not in a
   comment at the call site:

   - Grammar matchers. ``GrammarStepInputs.grammars`` holds the control
     plane's live matcher objects, and ownership is split by path. Under
     capturable grammar the side-stream hostfunc advances them and the
     commit path deliberately does not; under eager grammar the data plane
     reads them during the fill and the commit path advances them, so the
     loop drains its in-flight queue before dispatching a grammar batch
     (``EventLoop._dispatch_depends_on_pending_commit`` is the registry of
     those drain rules). Copying a matcher per round is not free, which is
     why this stays an exception rather than becoming rule 2.

   - Multimodal items. The gather's snapshot is shallow on purpose: the
     ``MultimodalDataItem`` objects stay shared, and the data plane WRITES
     through them mid-forward — the embedder fills ``item.encoded`` and
     drops the raw ``feature``/``feature_shm`` it no longer needs, and EPD
     publishes ``encoded`` via the handle's embedding slot. The discipline
     is ordering, not freezing: mutations happen on the FIFO (inside the
     forward, or a submitted closure), the control plane only reads them at
     commit time or later, and the release of the shared SHM handles rides
     the FIFO too (``run_multimodal_work`` with ``wait=False``) so it lands
     behind any forward still reading them. What the shallow copy does
     freeze is the OUTER struct — ``mrope_positions`` and its siblings,
     which the control plane rebinds every round.
"""

from __future__ import annotations

import enum
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import torch


class _Kind(enum.Enum):
    """How a queued task interacts with held forward results."""

    PLAIN = "plain"  # releases held results before it runs
    PREP = "prep"  # may run with results held
    FORWARD = "forward"  # may run with results held; its own may be held


class _LaunchOwner(threading.local):
    """The ForwardThread running on the current thread, if any."""

    def __init__(self) -> None:
        self.thread: ForwardThread | None = None


_launch_owner = _LaunchOwner()


def mark_launch() -> None:
    """Release forward results held behind the forward now launching.

    Called by the forward step right before its device launch (a graph
    replay, which releases the GIL for the whole replay, or the start of an
    eager forward). A no-op off the forward thread or with nothing held.
    """
    owner = _launch_owner.thread
    if owner is not None:
        owner._release_held()


class ForwardThread:
    """Single consumer thread executing submitted GPU work in FIFO order.

    ``submit`` returns a ``concurrent.futures.Future`` resolved with the
    callable's return value (or its exception). The callable runs with the
    thread's CUDA device set; stream discipline stays inside the executor
    (``execute_forward_op`` manages ``execution_stream`` itself).

    Holding results until the next launch
    -------------------------------------

    With ``hold_results_until_next_launch``, a forward's result (from
    ``submit_forward``) is not handed to the control plane while the next
    round's work is already queued. Resolving it wakes the commit loop,
    whose ``post_process`` is milliseconds of pure Python that takes the GIL
    the moment this thread releases it (any CUDA call does), so the next
    round's zeroing and input prep — the only host work between two graph
    replays, all on the GPU's critical path because the replay itself
    returns only once the GPU is nearly done — would wait for it. Instead the
    result is held until the next forward reaches its launch
    (``mark_launch``), and commit runs while that replay occupies the GPU.

    A held result is released, in FIFO order, as soon as any of these holds:
    the next forward launches; the queue is empty (nothing to protect, and
    the control plane may be waiting on this result to plan more work); the
    next task is not ``submit_prep``/``submit_forward`` work (it may wait on
    the control plane); a later forward finishes; any task raises.
    """

    def __init__(self, device, *, hold_results_until_next_launch: bool) -> None:
        # Resolve the CUDA index on the CALLER's thread: an index-less
        # "cuda" means the caller's current device (set_device(local_rank)
        # ran during distributed init), which a fresh thread would not
        # inherit — torch defaults new threads to device 0.
        self._device_type: str | None = None
        self._device_index: int | None = None
        resolved = torch.device(device)
        if resolved.type in {"cuda", "npu"}:
            device_module = torch.get_device_module(resolved.type)
            self._device_type = resolved.type
            self._device_index = (
                resolved.index
                if resolved.index is not None
                else device_module.current_device()
            )
        self._hold_results = hold_results_until_next_launch
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        # Forward results finished but not yet handed back; data-plane only.
        self._held: list[tuple[Future, Any]] = []
        self._thread = threading.Thread(
            target=self._run, name="tokenspeed::forward", daemon=True
        )
        self._thread.start()

    def _release_held(self) -> None:
        held, self._held = self._held, []
        for future, value in held:
            future.set_result(value)

    def _run(self) -> None:
        if self._device_index is not None:
            torch.get_device_module(self._device_type).set_device(self._device_index)
        _launch_owner.thread = self
        while True:
            if self._held and self._queue.empty():
                self._release_held()
            item = self._queue.get()
            if item is None:
                self._release_held()
                return
            fn, future, kind = item
            if kind is _Kind.PLAIN:
                self._release_held()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                value = fn()
            except BaseException as exc:  # noqa: BLE001 — relayed to the waiter
                self._release_held()
                future.set_exception(exc)
                continue
            if kind is not _Kind.FORWARD:
                future.set_result(value)
                continue
            # A forward that never reached mark_launch (an early return)
            # still releases its predecessor first: results stay FIFO.
            self._release_held()
            if self._hold_results:
                self._held.append((future, value))
            else:
                future.set_result(value)

    def _put(self, fn: Callable[[], Any], kind: _Kind) -> Future:
        future: Future = Future()
        self._queue.put((fn, future, kind))
        return future

    def submit_forward(self, fn: Callable[[], Any]) -> Future:
        """``submit`` for a model forward, whose result may be held.

        Under ``hold_results_until_next_launch`` the returned future may
        resolve only once the next forward launches (see the class doc);
        ``fn`` must call ``mark_launch`` before its device launch.
        """
        return self._put(fn, _Kind.FORWARD)

    def submit_prep(self, fn: Callable[[], Any]) -> Future:
        """``submit`` for a round's own device prep (page zeroing, copies).

        It never waits on the control plane, so a held forward result may
        stay held while it runs.
        """
        return self._put(fn, _Kind.PREP)

    def submit(self, fn: Callable[[], Any]) -> Future:
        """Enqueue ``fn`` for FIFO execution; resolve the returned future.

        ``fn``'s captures are the whole interface (see the capture contract
        above): everything it needs must be bound at capture time, and the
        caller must not mutate any of it afterwards.

        Args:
            fn: Zero-argument callable to run on the data plane.

        Returns:
            A future resolved with ``fn``'s return value, or its exception.
        """
        return self._put(fn, _Kind.PLAIN)

    def run(self, fn: Callable[[], Any]) -> Any:
        """Submit ``fn`` and block until it completes.

        For the rare submissions whose caller genuinely needs the outcome
        before it can continue — startup and teardown, and the low-rate PD
        events that must see their device write land. The per-round forward
        path uses ``submit`` and waits only at commit.

        Args:
            fn: Zero-argument callable to run on the data plane.

        Returns:
            ``fn``'s return value; its exception is re-raised here.
        """
        return self.submit(fn).result()

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)
