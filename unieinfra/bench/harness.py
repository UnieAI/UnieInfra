# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Measurement primitives, and the reasons they are shaped the way they are.

Every rule here was added after a wrong number had already been believed. Read
the docstrings before writing a new benchmark; the failure modes are not
obvious and several of them invert a result rather than blurring it.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

# This box: RTX PRO 6000 Blackwell. A kernel reporting more than this is reading
# from L2, not from memory, and the row is not a bandwidth comparison.
HBM_PEAK_GB_S = 1792.0


def device_us(fn: Callable[[], object], reps: int = 30, warmup: int = 10):
    """Device time and launch count for one call, from the profiler.

    Not from CUDA events. For anything short the host cannot enqueue as fast as
    the device runs, so the GPU idles between the two events and the delta
    measures the host. RoPE read 16.3 us that way and looked 1.28x behind vLLM;
    the profiler put the kernel at 1.47 us against vLLM's 2.51, i.e. 1.7x ahead.
    That one mistake inverted four kernels at once.

    The profiler's own host-side numbers are inflated ~3.6x here and are not
    used; its kernel names, counts and device durations are exact.

    Returns (device_us_per_call, launches_per_call).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    launches = 0
    for ev in prof.key_averages():
        if ev.device_time_total > 0 and ev.count:
            total += ev.device_time_total
            launches += ev.count
    return total / reps, launches / reps


def host_us(fn: Callable[[], object], reps: int = 400, warmup: int = 50):
    """Host enqueue cost for one call: no events, no sync inside the loop.

    Keep `reps` small enough that the CUDA queue never fills. Once it does the
    loop becomes device-bound and the wrapper cost this is trying to isolate
    disappears -- at reps=3000 a GEMM ablation showed every stage costing the
    same, which was the queue backing up rather than the stages being free.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    elapsed = (time.perf_counter() - t0) * 1e6 / reps
    torch.cuda.synchronize()
    return elapsed


def gbps(num_bytes: int, micros: float) -> float:
    """Achieved bandwidth. Print it next to every kernel timing.

    Above HBM_PEAK_GB_S the working set is L2-resident (128 MB here), which
    makes the number a cache-hit rate rather than a kernel speed.
    """
    return num_bytes / 1e9 / (micros / 1e6) if micros > 0 else 0.0


def row(label: str, dev: float, launches: float, host: float, nbytes: int = 0):
    """One comparable line. Both instruments, plus the L2 tell-tale."""
    band = f"{gbps(nbytes, dev):9.1f}" if nbytes else " " * 9
    flag = " L2" if nbytes and gbps(nbytes, dev) > HBM_PEAK_GB_S else ""
    return f"  {label:>18} {dev:9.2f} {launches:7.1f} {host:9.2f} {band}{flag}"


def header() -> str:
    return (
        f"  {'case':>18} {'device us':>9} {'launch':>7} {'host us':>9} " f"{'GB/s':>9}"
    )


def summarize(samples: list[float]) -> str:
    """Median plus spread, because a median alone hides a bimodal arm.

    An oscillating arm must be scored over the whole post-warmup run rather
    than a tail: a limit cycle makes a tail mean phase-biased.
    """
    if not samples:
        return "no samples"
    lo, hi = min(samples), max(samples)
    med = statistics.median(samples)
    spread = (hi - lo) / med * 100 if med else 0.0
    return f"median {med:8.2f}  min {lo:8.2f}  max {hi:8.2f}  spread {spread:4.1f}%"


def assert_engaged(counter_name: str, count: int, expected: int) -> None:
    """Fail loudly when a fast path did not run as many times as it should.

    A small improvement is not evidence that a mechanism engaged. Two
    consecutive attempts at hoisting a flashinfer plan measured 1-2% better
    while the fast path was never taken even once -- the first planned the wrong
    page table, the second missed on the last bit of a float. Both were only
    caught by making the slow path raise and counting.
    """
    if count != expected:
        raise AssertionError(
            f"{counter_name}: fast path ran {count} times, expected {expected}. "
            "A timing delta is not evidence; verify the mechanism."
        )
