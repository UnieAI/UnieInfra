# Copyright (c) 2026 UnieAI
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

"""A request's recurrent-state charge must not grow with its prompt.

At one snapshot per cache block it did, and admission refused on that charge
rather than on the KV: on Qwen3.6-35B-A3B at 16 concurrent, 14 of 16 requests
sat in Submitted while the decode batch ran at 2, for 169.7 tok/s against
290.5 once the charge was bounded.
"""

import types

from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import (
    _MAX_STATE_CHECKPOINTS,
    _state_checkpoint_stride_blocks,
)

BLOCK_TOKENS = 128


def _args(**fields):
    defaults = {"max_model_len": 49152, "state_checkpoint_stride_blocks": 0}
    defaults.update(fields)
    return types.SimpleNamespace(**defaults)


def test_charge_is_bounded_however_long_the_context():
    """The property vLLM gets structurally, expressed as a derivation.

    `MambaSpec.max_memory_usage_bytes` charges one state page per request at
    vLLM's default; the derived stride here caps the count instead of capping
    the length, so a longer `max_model_len` buys a wider stride rather than a
    bigger reservation.
    """
    for max_model_len in (8192, 49152, 131072, 1048576):
        stride = _state_checkpoint_stride_blocks(
            _args(max_model_len=max_model_len), BLOCK_TOKENS
        )
        pages = -(-max_model_len // BLOCK_TOKENS)
        checkpoints = -(-pages // stride)
        assert (
            checkpoints <= _MAX_STATE_CHECKPOINTS
        ), f"max_model_len={max_model_len} charges {checkpoints} snapshots"


def test_an_explicit_stride_still_wins():
    """The derivation is a default, not a policy the operator cannot escape."""
    assert (
        _state_checkpoint_stride_blocks(
            _args(state_checkpoint_stride_blocks=4), BLOCK_TOKENS
        )
        == 4
    )
    assert (
        _state_checkpoint_stride_blocks(
            _args(state_checkpoint_stride_blocks=1), BLOCK_TOKENS
        )
        == 1
    )


def test_an_unknown_context_length_snapshots_every_block():
    """Falling back to the old behaviour beats deriving from a guess.

    A stride wider than the sequence leaves a prefix match resuming from
    further back; without a context length there is nothing to size that
    against, and one-per-block is the conservative end.
    """
    assert _state_checkpoint_stride_blocks(_args(max_model_len=0), BLOCK_TOKENS) == 1
    assert _state_checkpoint_stride_blocks(types.SimpleNamespace(), BLOCK_TOKENS) == 1
