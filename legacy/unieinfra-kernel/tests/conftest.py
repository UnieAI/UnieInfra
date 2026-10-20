# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Test bootstrap for unieinfra-kernel.

Two jobs, both done at import time (before pytest collects any test):

1. Put the ``unieinfra-kernel`` package root on ``sys.path`` so ``import
   unieinfra_kernel`` resolves from the source tree without an install.

2. Make ``tokenspeed_kernel``'s *core* importable on a CPU-only box. The real
   ``tokenspeed_kernel/__init__.py`` eagerly imports every GPU op module (which
   pull in ``tokenspeed_triton``, flashinfer, etc.), so a plain ``import
   tokenspeed_kernel`` cannot succeed without a GPU toolchain. The registry /
   platform / selection / signature modules this package actually depends on
   need only ``torch``. When the full package is unavailable we install a
   lightweight stand-in package object (real file ``__path__``, heavy
   ``__init__`` body skipped) into ``sys.modules`` so submodule imports load the
   genuine core modules and the genuine ``KernelRegistry`` singleton.

On a real GPU box where ``import tokenspeed_kernel`` succeeds, the stub is not
installed and the real package is used unchanged.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _TESTS_DIR.parent  # unieinfra-kernel/
_REPO_ROOT = _PKG_ROOT.parent  # UnieInfra/
# The retiring tokenspeed tree has moved twice (repo root -> ``legacy/`` ->
# ``experiment/legacy/``), and a stale single path here silently degrades the
# whole suite to a collection error. Search every location it has lived in and
# take the first that exists.
_TSK_CORE_CANDIDATES = (
    _REPO_ROOT / "experiment" / "legacy" / "tokenspeed-kernel",
    _REPO_ROOT / "legacy" / "tokenspeed-kernel",
    _REPO_ROOT / "tokenspeed-kernel",
)
_TSK_CORE_DIR = next(
    (
        candidate / "python" / "tokenspeed_kernel"
        for candidate in _TSK_CORE_CANDIDATES
        if (candidate / "python" / "tokenspeed_kernel").is_dir()
    ),
    _TSK_CORE_CANDIDATES[0] / "python" / "tokenspeed_kernel",
)

# 1. unieinfra_kernel importable from source.
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


def _install_tokenspeed_kernel_core_stub() -> None:
    """Register a lightweight ``tokenspeed_kernel`` package if the real one fails.

    The stub is a bare package object whose ``__path__`` points at the real
    source directory, so ``import tokenspeed_kernel.registry`` (and its siblings)
    load the actual core modules — sharing the one true ``KernelRegistry`` — while
    the heavy ``__init__`` body (GPU op imports) never runs.
    """
    if "tokenspeed_kernel" in sys.modules:
        return
    try:
        import tokenspeed_kernel  # noqa: F401  (real package present -> use it)

        return
    except Exception:
        pass

    if not _TSK_CORE_DIR.is_dir():
        return  # nothing to stub against; let the test import fail loudly

    pkg = types.ModuleType("tokenspeed_kernel")
    pkg.__path__ = [str(_TSK_CORE_DIR)]
    pkg.__doc__ = "Lightweight core-only stub installed by unieinfra-kernel tests."
    sys.modules["tokenspeed_kernel"] = pkg


_install_tokenspeed_kernel_core_stub()
