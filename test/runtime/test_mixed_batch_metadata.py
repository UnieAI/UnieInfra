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

"""A MIXED plan must reach the recurrent path as one whole-batch extend plan.

The recurrent kernels are not split for a MIXED batch: a decode row is an
extend row of length 1 whose prefix is its history, which makes the whole plan
one valid extend batch. That only holds if *every* per-request array the path
indexes describes the same row set -- the whole batch -- because the conv kernel
derives its program count from ``query_start_loc`` while gathering state through
``cache_indices`` and gating it with ``has_initial_state``. vLLM and SGLang both
hold exactly this invariant on their GDN paths, and neither rebases any of it.

Breaking it is not reliably fatal, which is why it has to be tested rather than
observed: a short ``has_initial_state`` raised a clean shape error, but a short
conv input against whole-batch index arrays ran fine at batch 8 and faulted with
an illegal memory access at batch 16.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    MambaAttnBackend,
    _whole_batch_prefix_lens,
)


def _backend():
    """A metadata-only ``MambaAttnBackend``.

    ``init_forward_metadata`` reads a handful of scalars and no GPU state, so
    the real constructor (which wants a model config, a pool and a device) is
    not needed to pin what it produces.
    """
    b = MambaAttnBackend.__new__(MambaAttnBackend)
    b.pool = None  # poolless: mamba_cache_indices is supplied by state pages
    b.device = "cpu"
    b.is_draft = False
    b.spec_num_tokens = 1
    b.speculative_num_draft_tokens = 1
    b.flat_state_active = False
    b._flat_contract_bound = False
    return b


def _mixed_plan(num_extends: int, num_decodes: int, chunk: int = 300):
    """A MIXED plan: extending rows first, then one-token decode rows.

    Mirrors ``FlatForwardOperation``'s layout -- rows ``[0, num_extends)`` are
    extending and own the leading query tokens, rows ``[num_extends, bs)`` are
    decoding and contribute one token each, in order, after them.
    """
    bs = num_extends + num_decodes
    extend_lens = [chunk] * num_extends
    # Extending rows resume mid-document; decode rows carry a long history.
    prefix = [100 * (i + 1) for i in range(num_extends)] + [
        1000 + i for i in range(num_decodes)
    ]
    seq_lens = torch.tensor(
        [prefix[i] + (extend_lens[i] if i < num_extends else 1) for i in range(bs)],
        dtype=torch.int32,
    )
    kwargs = dict(
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
        extend_start_loc=torch.tensor(
            [sum(extend_lens[:i]) for i in range(num_extends)], dtype=torch.int32
        ),
        extend_prefix_lens=torch.tensor(prefix[:num_extends], dtype=torch.int32),
    )
    return bs, seq_lens, kwargs, prefix, extend_lens


@pytest.mark.parametrize(
    ("num_extends", "num_decodes"),
    [(1, 7), (1, 15), (3, 13), (15, 1), (2, 6)],
    ids=["1x7", "1x15", "3x13", "15x1", "2x6"],
)
def test_mixed_metadata_describes_the_whole_batch(num_extends, num_decodes):
    """Every per-request array is `bs` rows long, decode tail included.

    The batch sizes include the pair the fault discriminated between (bs=8
    served, bs=16 faulted), because a row-count bug is batch-size dependent by
    construction and a single size can miss it.
    """
    bs, seq_lens, kwargs, prefix, extend_lens = _mixed_plan(num_extends, num_decodes)
    backend = _backend()

    backend.init_forward_metadata(
        bs,
        torch.arange(bs, dtype=torch.int32),
        seq_lens,
        forward_mode=ForwardMode.MIXED,
        **kwargs,
    )
    md = backend.forward_metadata

    assert md.query_start_loc.numel() == bs + 1
    assert md.extend_prefix_lens.numel() == bs
    assert md.extend_has_initial_states.numel() == bs
    assert md.extend_seq_lens_cpu.numel() == bs

    # The decode tail contributes exactly one query token per row.
    per_row = md.query_start_loc[1:] - md.query_start_loc[:-1]
    assert per_row[:num_extends].tolist() == extend_lens
    assert per_row[num_extends:].tolist() == [1] * num_decodes

    # seq_lens[i] == prefix[i] + extend[i] for every row, so the prefix array
    # the conv kernel gates on is recoverable from the plan alone.
    assert md.extend_prefix_lens.tolist() == prefix

    # A decode row always resumes, so it must never be treated as fresh: a
    # False here zeroes a live recurrent state.
    assert md.extend_has_initial_states[num_extends:].all()

    assert md.extend_total_tokens == sum(extend_lens) + num_decodes


def test_pure_extend_is_unchanged():
    """The whole-batch array reproduces the caller's on a plan with no decodes.

    There is therefore no MIXED-only branch to keep in sync -- the reason the
    prefix is derived rather than special-cased.
    """
    bs, seq_lens, kwargs, prefix, _ = _mixed_plan(4, 0)
    backend = _backend()

    backend.init_forward_metadata(
        bs,
        torch.arange(bs, dtype=torch.int32),
        seq_lens,
        forward_mode=ForwardMode.EXTEND,
        **kwargs,
    )

    assert backend.forward_metadata.extend_prefix_lens.tolist() == prefix
    assert torch.equal(
        backend.forward_metadata.extend_prefix_lens, kwargs["extend_prefix_lens"]
    )


def test_a_fresh_request_is_not_marked_as_resuming():
    """Prefix 0 must stay `has_initial_state=False` even beside decode rows.

    The conv kernel's per-sequence gate is the only thing that stops a fresh
    request from being seeded with whatever the state slab last held.
    """
    extend_lens = torch.tensor([256, 128], dtype=torch.int32)
    prefix = _whole_batch_prefix_lens(
        torch.tensor([256, 628, 900, 901], dtype=torch.int32),
        torch.tensor([256, 128, 1, 1], dtype=torch.int32),
        4,
    )
    assert prefix.tolist() == [0, 500, 899, 900]
    assert (prefix > 0).tolist() == [False, True, True, True]
    assert extend_lens.dtype == prefix.dtype
