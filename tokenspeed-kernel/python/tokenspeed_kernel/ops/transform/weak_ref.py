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

"""Non-owning tensor view for CUDA-graph break closures.

``weak_ref_tensor(t)`` returns a tensor aliasing ``t``'s memory WITHOUT owning
its storage (``at::from_blob`` with a no-op deleter). A breakable-CUDA-graph
break closure that holds such a view does not pin ``t``'s mempool block, so the
graph pool can recycle blocks across segments and captures -- graph memory
drops from sum-over-buckets of break inputs back to ~peak-live. Correctness
relies on replay order: the captured segment rewrites the aliased address
before the break reads it, every replay (the same discipline that lets the
decode graph share one pool across batch sizes).

Adapted from vLLM ``csrc/ops.h`` / SGLang ``sgl-kernel/csrc/memory/
weak_ref_tensor.cpp``. Compiled lazily via ``torch.utils.cpp_extension``
(pure C++ against ATen -- no nvcc, no wheel rebuild); falls back to identity
(strong ref: correct, more memory) if compilation is unavailable.
"""

from __future__ import annotations

import logging
import os
import tempfile

import torch

logger = logging.getLogger(__name__)

_CPP_SOURCE = r"""
#include <ATen/ATen.h>
#include <vector>

at::Tensor weak_ref_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "weak_ref_tensor expects a CUDA tensor");
  void* data_ptr = tensor.data_ptr();
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();
  auto options = tensor.options();
  return at::from_blob(data_ptr, sizes, strides, options);
}
"""

_module = None
_load_failed = False


def _is_writable_dir(path: str) -> bool:
    """Return whether ``path`` is (or can be made) a writable directory.

    Uses a real create/write/remove probe rather than ``os.access`` so that a
    read-only mount or a container running as root is classified correctly.

    Args:
        path: Directory to test.

    Returns:
        True if a probe file could be created and removed under ``path``.
    """
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".tokenspeed_write_probe")
        with open(probe, "w"):
            pass
        os.remove(probe)
    except OSError:
        return False
    return True


def _resolve_build_directory() -> str | None:
    """Pick a writable build directory for the lazily compiled helper.

    ``torch.utils.cpp_extension`` defaults to ``$TORCH_EXTENSIONS_DIR`` or
    ``$HOME/.cache/torch_extensions``. A serving container frequently has a
    read-only ``$HOME``, which silently degrades this module to the identity
    fallback. That fallback is correct but pins graph-pool blocks, so
    breakable-graph capture memory scales with the bucket sum instead of the
    peak -- which then forces a single-rung prefill ladder, and every prefill
    step replays the largest bucket no matter how few tokens it really has.
    Falling back to a temp directory keeps the climb out of that trap.

    Returns:
        The first writable candidate directory, or ``None`` if none is
        writable, in which case the caller leaves the choice to torch.
    """
    home = os.environ.get("HOME")
    candidates = [
        os.environ.get("TOKENSPEED_EXTENSIONS_DIR"),
        os.environ.get("TORCH_EXTENSIONS_DIR"),
        os.path.join(home, ".cache", "torch_extensions") if home else None,
        os.path.join(tempfile.gettempdir(), "tokenspeed_torch_extensions"),
    ]
    for candidate in candidates:
        if candidate and _is_writable_dir(candidate):
            return candidate
    return None


def _load():
    global _module, _load_failed
    if _module is not None or _load_failed:
        return _module
    try:
        from torch.utils.cpp_extension import load_inline

        build_directory = _resolve_build_directory()
        if build_directory is None:
            logger.warning(
                "weak_ref_tensor: no writable build directory found; leaving "
                "the choice to torch (this is likely to fail).",
            )
            build_directory = ""
        _module = load_inline(
            name="tokenspeed_weak_ref_tensor",
            cpp_sources=_CPP_SOURCE,
            functions=["weak_ref_tensor"],
            with_cuda=False,
            verbose=False,
            build_directory=build_directory,
        )
    except Exception as exc:  # pragma: no cover - host without a C++ toolchain
        _load_failed = True
        logger.warning(
            "weak_ref_tensor extension unavailable (%s: %s); falling back to "
            "identity (strong refs pin graph-pool blocks -- correct, but "
            "breakable-graph capture memory scales with the bucket sum, so the "
            "prefill graph is forced onto a single largest-bucket rung and "
            "every prefill step replays that bucket's padded token count).",
            type(exc).__name__,
            exc,
        )
    return _module


def weak_ref_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return a non-owning view of CUDA tensor ``t`` (alias, same shape/strides).

    Args:
        t: A CUDA tensor. The caller must guarantee the aliased memory is
            rewritten (or still valid) whenever the view is consumed -- for
            breakable-graph closures this holds because the captured segment
            producing ``t`` replays before the break reads the view.

    Returns:
        A tensor sharing ``t``'s memory without owning its storage, or ``t``
        itself if the extension is unavailable (identity fallback).
    """
    mod = _load()
    if mod is None:
        return t
    return mod.weak_ref_tensor(t)
