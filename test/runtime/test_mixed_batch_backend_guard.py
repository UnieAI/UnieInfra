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

The hybrid linear-attention backend cannot: its state-page gather assumes every
sequence contributes the same number of rows, so a plan holding prefill and
decode together fails inside ``MambaAttnBackend._flat_state_pages`` with a shape
error that reads like a bug rather than a configuration mistake -- and does so
minutes into a run, on the first batch that happens to mix.

These tests pin the declaration and the refusal separately, because either one
alone leaves the flag silently broken.
"""

from __future__ import annotations

import pytest

from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    HybridLinearAttnBackend,
)


def test_hybrid_linear_attn_declares_it_cannot_mix():
    """The class attribute is the whole contract -- the guard reads only this."""
    assert HybridLinearAttnBackend.supports_mixed_batch is False


def test_backends_may_mix_by_default():
    """A backend that says nothing is assumed capable, as before this guard."""
    assert AttentionBackend.supports_mixed_batch is True


class _Incapable:
    supports_mixed_batch = False


class _Capable:
    supports_mixed_batch = True


class _Silent:
    """Declares nothing, e.g. an out-of-tree backend predating the attribute."""


def _check(server_args, attn_backend):
    """The guard as it appears in EventLoop.__init__, isolated.

    Constructing a real EventLoop needs a model, a scheduler and a GPU, none of
    which this decision depends on.
    """
    if server_args.enable_mixed_batch and not getattr(
        attn_backend, "supports_mixed_batch", True
    ):
        raise ValueError(
            "--enable-mixed-batch requires an attention backend that can "
            f"serve mixed batches; {type(attn_backend).__name__} declares "
            "supports_mixed_batch = False. Drop the flag, or run a model "
            "whose backend supports MIXED."
        )


class _Args:
    def __init__(self, enable_mixed_batch):
        self.enable_mixed_batch = enable_mixed_batch


def test_refuses_an_incapable_backend():
    with pytest.raises(ValueError, match="supports_mixed_batch = False"):
        _check(_Args(True), _Incapable())


def test_names_the_backend_so_the_message_is_actionable():
    with pytest.raises(ValueError, match="_Incapable"):
        _check(_Args(True), _Incapable())


@pytest.mark.parametrize("backend", [_Capable(), _Silent()], ids=["capable", "silent"])
def test_allows_a_capable_backend(backend):
    _check(_Args(True), backend)


@pytest.mark.parametrize(
    "backend", [_Capable(), _Silent(), _Incapable()], ids=["capable", "silent", "incapable"]
)
def test_flag_off_never_refuses(backend):
    """Without the flag the backend's capability is irrelevant."""
    _check(_Args(False), backend)
