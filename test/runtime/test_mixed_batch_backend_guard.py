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

"""`--enable-mixed-batch` must be refused by a backend that cannot serve MIXED.

Two things have to hold and they fail independently, so they are pinned
separately: each backend has to *declare* whether it can serve a plan holding
prefill and decode ops together, and the engine has to *act* on that declaration
at startup. Either one alone leaves the flag silently broken -- a declaration
nobody reads, or a refusal that never triggers.

The guard itself is imported, not reproduced. An earlier version of this file
hand-copied the check out of ``EventLoop.__init__``, which passes whatever the
engine does.
"""

from __future__ import annotations

import pytest

from tokenspeed.runtime.execution.model_executor import validate_mixed_batch_backend
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.msa import MSAAttnBackend


class _Incapable:
    supports_mixed_batch = False


class _Capable:
    supports_mixed_batch = True


class _Silent:
    """Declares nothing, e.g. an out-of-tree backend predating the attribute."""


class _Args:
    def __init__(self, enable_mixed_batch):
        self.enable_mixed_batch = enable_mixed_batch


# ---- declarations ----


def test_backends_may_mix_by_default():
    """A backend that says nothing is assumed capable, as before this guard."""
    assert getattr(AttentionBackend, "supports_mixed_batch", True) is True


@pytest.mark.parametrize("leaf", [MHAAttnBackend, MSAAttnBackend])
def test_paged_leaves_that_assert_against_mixed_declare_it(leaf):
    """Both leaves assert ``not forward_mode.is_mixed()`` in prepare()."""
    assert leaf.supports_mixed_batch is False


class _Router:
    """Stands in for CacheGroupRouter: per-group leaves, nothing declared."""

    def __init__(self, *leaves):
        self.leaves = {f"g{i}": leaf for i, leaf in enumerate(leaves)}


class _Hybrid:
    """Stands in for HybridLinearAttnBackend: two halves, nothing declared."""

    def __init__(self, full, linear):
        self.full_attn_backend = full
        self.linear_attn_backend = linear


def test_a_composite_refuses_when_any_leaf_cannot_mix():
    backend = _Hybrid(_Router(_Capable(), _Incapable()), _Capable())
    with pytest.raises(ValueError, match="_Incapable"):
        validate_mixed_batch_backend(_Args(True), backend)


def test_a_composite_of_capable_leaves_is_allowed():
    validate_mixed_batch_backend(_Args(True), _Hybrid(_Router(_Capable()), _Silent()))


# ---- the refusal ----


def test_refuses_an_incapable_backend():
    with pytest.raises(ValueError, match="supports_mixed_batch = False"):
        validate_mixed_batch_backend(_Args(True), _Incapable())


def test_names_the_backend_so_the_message_is_actionable():
    with pytest.raises(ValueError, match="_Incapable"):
        validate_mixed_batch_backend(_Args(True), _Incapable())


@pytest.mark.parametrize("backend", [_Capable(), _Silent()], ids=["capable", "silent"])
def test_allows_a_capable_backend(backend):
    validate_mixed_batch_backend(_Args(True), backend)


def test_says_nothing_when_the_flag_is_off():
    """An incapable backend is fine as long as nobody asked for mixed batches."""
    validate_mixed_batch_backend(_Args(False), _Incapable())
