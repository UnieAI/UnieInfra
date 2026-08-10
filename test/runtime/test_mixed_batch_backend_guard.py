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

from tokenspeed.runtime.engine.event_loop import validate_mixed_batch_backend
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.msa import MSAAttnBackend


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
    assert AttentionBackend.supports_mixed_batch is True


def test_msa_declares_it_cannot_mix():
    """MSA builds extend metadata for every row, so a decode tail is out of range.

    It is a class attribute rather than a property because the constraint is
    structural, not configuration-dependent.
    """
    assert MSAAttnBackend.supports_mixed_batch is False


def test_hybrid_linear_attn_defers_to_its_full_attention_half():
    """The recurrent half serves MIXED without splitting, so it adds no constraint.

    A decode row is an extend row of length 1 whose prefix is its history, which
    makes the whole plan one valid extend batch. The only remaining constraint
    is the full-attention half's (spec width, draft rank), so the answer comes
    from there -- hence a property rather than a class attribute.
    """
    backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)

    backend.full_attn_backend = _Capable()
    assert backend.supports_mixed_batch is True

    backend.full_attn_backend = _Incapable()
    assert backend.supports_mixed_batch is False


def test_hybrid_needs_no_environment_opt_in(monkeypatch):
    """The env gate that kept the unfinished split unreachable is gone.

    Pinned because its removal is the user-visible half of the fix: a hybrid
    model now takes `--enable-mixed-batch` directly.
    """
    backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    backend.full_attn_backend = _Capable()

    monkeypatch.delenv("TOKENSPEED_ENABLE_HYBRID_MIXED_BATCH", raising=False)
    assert backend.supports_mixed_batch is True


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
