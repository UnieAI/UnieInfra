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

"""Adaptive batch must never admit past the recurrent-state (mamba/GDN) pool.

Regression test for E28: on hybrid linear-attention models the KV-derived
running-batch cap ignored the mamba chunk pool, so the scheduler admitted more
requests than there were state slots and the run died with a device-side
"index out of bounds" inside the mamba metadata gather.
"""

from tokenspeed.runtime.engine.event_loop import derive_adaptive_max_batch

# A KV pool big enough to derive a large cap: 1M tokens / (4096/4) = 976 slots.
_KV_TOKENS = 1_000_000
_CONTEXT_LEN = 4096
_RESERVE_DIVISOR = 4
_CEILING = 1024


def _derive(state_pool_capacity=None, current=160):
    return derive_adaptive_max_batch(
        current_max_num_seqs=current,
        max_total_num_tokens=_KV_TOKENS,
        context_len=_CONTEXT_LEN,
        reserve_divisor=_RESERVE_DIVISOR,
        ceiling=_CEILING,
        state_pool_capacity=state_pool_capacity,
    )


def test_no_state_pool_keeps_kv_derived_cap():
    """Dense models (no recurrent state) are unaffected by the new clamp."""
    assert _derive(state_pool_capacity=None) == _derive(state_pool_capacity=0)
    assert _derive(state_pool_capacity=None) > 160


def test_state_pool_clamps_below_kv_derived_cap():
    """The scarce mamba pool wins over the generous KV-derived cap."""
    assert _derive(state_pool_capacity=152) == 152


def test_state_pool_clamps_below_the_static_floor():
    """A state pool smaller than --max-num-seqs still wins.

    This is the Qwen3.5-27B case (95 chunks vs a 160/256 cap): the floor must
    not be allowed to re-raise the cap past a hard physical pool.
    """
    assert _derive(state_pool_capacity=95, current=160) == 95


def test_generous_state_pool_does_not_raise_the_cap():
    """The clamp only ever lowers; it never invents extra concurrency."""
    unclamped = _derive(state_pool_capacity=None)
    assert _derive(state_pool_capacity=100_000) == unclamped


def test_clamp_never_returns_zero():
    """Degenerate tiny pools still leave a runnable batch of at least one."""
    assert _derive(state_pool_capacity=1) == 1
