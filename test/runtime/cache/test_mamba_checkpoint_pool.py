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

"""The ported int8 checkpoint pool, on CPU so it runs anywhere.

The round-trip error bound is the part worth pinning: a cached state is
quantized once on store and dequantized once on a hit, so the whole quality
argument rests on that single rounding staying small relative to the state's
own scale.
"""

import pytest
import torch

from tokenspeed.runtime.cache.mamba_checkpoint_pool import (
    Int8CheckpointStore,
    MambaCheckpointPool,
    MambaSlotAllocator,
)

L, N, H, DV, DK = 3, 4, 2, 8, 6


def _pool(**over):
    kwargs = dict(
        num_layers=L,
        num_slots=N,
        num_heads=H,
        head_v_dim=DV,
        head_k_dim=DK,
        conv_shapes=[(5, 7)],
        conv_dtype=torch.float32,
        device="cpu",
        temporal_dtype=torch.float32,
    )
    kwargs.update(over)
    return MambaCheckpointPool(**kwargs)


def test_slot_zero_is_reserved():
    """Slot 0 is the padded-token write target, so it is never handed out."""
    alloc = MambaSlotAllocator(size=N, device="cpu")
    got = alloc.alloc(N)
    assert got is not None
    assert got.min().item() == 1
    assert got.max().item() == N


def test_alloc_returns_none_when_short_rather_than_raising():
    """These are cache entries: a short pool costs reuse depth, not a request."""
    alloc = MambaSlotAllocator(size=2, device="cpu")
    assert alloc.alloc(3) is None
    assert alloc.available_size() == 2
    taken = alloc.alloc(2)
    assert alloc.alloc(1) is None
    alloc.free(taken)
    assert alloc.available_size() == 2


def test_quantize_dequantize_round_trip_error_is_bounded():
    """One rounding, per (head, k-channel) scale: within a step of that scale."""
    torch.manual_seed(0)
    state = torch.randn(L, 1, H, DV, DK, dtype=torch.float32)
    q, scale = Int8CheckpointStore.quantize(state)
    back = Int8CheckpointStore.dequantize(q, scale, torch.float32)

    assert q.dtype is torch.int8
    assert scale.shape == (L, 1, H, 1, DK)
    # Symmetric quantization rounds to the nearest multiple of the scale, so the
    # error cannot exceed half a step. Allow a whole step for the float32 divide.
    assert torch.all((back - state).abs() <= scale.expand_as(state))


def test_zero_state_survives_the_round_trip():
    """The amax clamp must not turn an all-zero state into NaN."""
    state = torch.zeros(L, 1, H, DV, DK, dtype=torch.float32)
    q, scale = Int8CheckpointStore.quantize(state)
    back = Int8CheckpointStore.dequantize(q, scale, torch.float32)
    assert torch.isfinite(back).all()
    assert torch.all(back == 0)


def test_store_then_load_moves_state_between_pools():
    torch.manual_seed(1)
    pool = _pool()
    active_temporal = torch.randn(L, 6, H, DV, DK, dtype=torch.float32)
    active_conv = [torch.randn(L, 6, 5, 7, dtype=torch.float32)]

    src = torch.tensor([2, 4], dtype=torch.int64)
    ckpt = pool.alloc(2)
    assert ckpt is not None
    pool.store_from_active(active_temporal, active_conv, src, ckpt)

    dst = torch.tensor([0, 1], dtype=torch.int64)
    pool.load_to_active(active_temporal, active_conv, ckpt, dst)

    # conv is copied verbatim, so it must come back bit for bit.
    assert torch.equal(active_conv[0][:, dst], active_conv[0][:, src])
    # temporal went through int8 and comes back close, not equal.
    err = (active_temporal[:, dst] - active_temporal[:, src]).abs().max()
    ref = active_temporal[:, src].abs().max()
    assert err < 0.02 * ref


def test_estimate_matches_the_allocation_it_predicts():
    """The estimate exists to be checked against free memory before allocating,
    so it has to agree with what construction actually takes."""
    kwargs = dict(
        num_layers=L,
        num_slots=N,
        num_heads=H,
        head_v_dim=DV,
        head_k_dim=DK,
        conv_shapes=[(5, 7)],
        conv_dtype=torch.float32,
        temporal_dtype=torch.float32,
    )
    est = MambaCheckpointPool.estimate_mem_usage_bytes(**kwargs)
    pool = MambaCheckpointPool(device="cpu", **kwargs)
    assert est["total"] == pool.mem_usage_bytes()


def test_int8_is_smaller_than_the_active_state_it_replaces():
    """The whole reason for the pool: a cached slot must cost less than a live
    one, or moving snapshots here buys capacity only by moving the problem."""
    pool = _pool(temporal_dtype=torch.float32)
    active_bytes_per_slot = L * H * DV * DK * 4  # float32 temporal
    ckpt_bytes_per_slot = pool.temporal.bytes_per_slot()
    assert ckpt_bytes_per_slot < active_bytes_per_slot


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scale_follows_the_state_dtype(dtype):
    """Quantize and dequantize must use the identical scale, which is why the
    scale is stored in the state's own dtype rather than always bf16."""
    pool = _pool(temporal_dtype=dtype)
    assert pool.temporal.scale.dtype is dtype
