# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Kernel autotune lifecycle.

Autotuning **measures** candidate kernels, which means launching them and
synchronising. That is safe exactly once — during engine startup — and unsafe
forever after, for two independent reasons:

* it blocks the launch thread for the whole profiling run, so a request that
  happens to be the first to hit a new shape pays the entire tuning cost; and
* under tensor parallelism, ranks that begin tuning at different times miss
  their next collective and **deadlock**.

So the runtime calls :func:`freeze_autotuning` once startup (model load, warmup,
CUDA-graph capture) completes. After the freeze, :func:`select_kernel` falls
back to heuristic ranking for shapes it has no cached measurement for — a
possibly-worse kernel, never a stall and never a hang.

Ported from ``tokenspeed_kernel.ops.tuning`` (the retiring tree, now under
``legacy/``); the semantics and the function names are unchanged so an
integration written against either base keeps working.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

__all__ = [
    "autotune_frozen",
    "freeze_autotuning",
    "unfreeze_autotuning",
    "autotuning_unfrozen",
]

_frozen = False


def autotune_frozen() -> bool:
    """Whether autotuning is frozen (i.e. the engine is serving).

    Returns:
        ``True`` once :func:`freeze_autotuning` has been called.
    """
    return _frozen


def freeze_autotuning() -> None:
    """Disallow any further autotune runs. Call once startup completes."""
    global _frozen
    _frozen = True


def unfreeze_autotuning() -> None:
    """Re-allow autotuning.

    Intended for tests and for offline tuning tools. A serving engine should
    never call this: see the module docstring for why tuning during serving
    deadlocks under TP.
    """
    global _frozen
    _frozen = False


@contextmanager
def autotuning_unfrozen() -> Generator[None, None, None]:
    """Scope in which autotuning is permitted, restoring the previous state."""
    global _frozen
    previous = _frozen
    _frozen = False
    try:
        yield
    finally:
        _frozen = previous
