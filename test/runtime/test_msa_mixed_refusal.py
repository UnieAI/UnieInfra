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

"""The MSA leaf refuses a MIXED round instead of mis-serving it.

The merged engine no longer vets attention backends at startup: every leaf either
serves a MIXED round or says so on the call that hands it one. ``MSAAttnBackend``
does not sling the packed layout yet, so the refusal has to survive any refactor
of the leaf's entry point -- a leaf that accepted the round and then treated the
decode rows as extend rows would produce fluent, wrong output rather than an
error.
"""

from __future__ import annotations

import pytest

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.msa import MSAAttnBackend


def _mixed_call(backend):
    """Hand the leaf a MIXED round, as the cache-group router does.

    The refusal is the entry point's first act, so the per-row arguments a real
    caller passes are never read: the metadata build must stop before it looks at
    any of them. That is what makes a bare instance enough to test, and what a
    leaf that had grown an early read of ``seq_lens`` or ``num_extends`` would
    break.
    """
    return backend.init_forward_metadata(
        2,
        1,
        None,
        None,
        ForwardMode.MIXED,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        extend_prefix_lens=None,
        extend_prefix_lens_cpu=None,
        extend_with_prefix=True,
        query_shard=None,
        page_table_cpu=None,
    )


def test_msa_refuses_a_mixed_round_before_touching_state():
    # A leaf that accepted the round and treated the decode rows as extend rows
    # would produce fluent, wrong output rather than an error, so the refusal has
    # to survive refactors of the entry point.
    backend = MSAAttnBackend.__new__(MSAAttnBackend)

    with pytest.raises(AssertionError, match="MSA backend does not support mixed batch") as caught:
        _mixed_call(backend)

    # The message is the only thing an operator sees at the point of failure.
    assert "MSA" in str(caught.value)
