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

"""The checkpoint pool, on CPU so it runs anywhere.

Two things are worth pinning here. At the default float32 the store must be
*exact*: a snapshot is what a prefix hit resumes from, and a cold prefill
carries that state in an fp32 accumulator, so any rounding on the way through
this pool is a difference between two runs of the same prompt. At int8 -- the
capacity trade -- the state is quantized once on store and dequantized once on
a hit, so the quality argument rests on that single rounding staying small
relative to the state's own scale.
"""

import pytest
import torch

from tokenspeed.runtime.cache.mamba_checkpoint_pool import (
    DenseCheckpointStore,
    Int8CheckpointStore,
    MambaCheckpointPool,
    MambaSlotAllocator,
    resolve_ckpt_dtype,
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
        ckpt_dtype=torch.int8,
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


@pytest.mark.parametrize("ckpt_dtype", [torch.int8, torch.bfloat16, torch.float32])
def test_estimate_matches_the_allocation_it_predicts(ckpt_dtype):
    """The estimate exists to be checked against free memory before allocating,
    so it has to agree with what construction actually takes -- at every dtype,
    since the dtype is what the estimate is used to choose between."""
    kwargs = dict(
        num_layers=L,
        num_slots=N,
        num_heads=H,
        head_v_dim=DV,
        head_k_dim=DK,
        conv_shapes=[(5, 7)],
        conv_dtype=torch.float32,
        temporal_dtype=torch.float32,
        ckpt_dtype=ckpt_dtype,
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


def test_per_layer_round_trip_matches_the_whole_pool_one():
    """The per-layer path is an addition, so it has to agree with the ported
    whole-pool path rather than quietly diverge from it."""
    torch.manual_seed(2)
    pool = _pool()
    ssm = torch.randn(L, 6, H, DV, DK, dtype=torch.float32)
    conv = torch.randn(L, 6, 5, 7, dtype=torch.float32)

    src = torch.tensor([1, 3], dtype=torch.int64)
    whole = pool.alloc(2)
    per_layer = pool.alloc(2)
    assert whole is not None and per_layer is not None

    pool.store_from_active(ssm, [conv], src, whole)
    for layer in range(L):
        pool.store_layer_from_active(layer, ssm[layer], conv[layer], src, per_layer)

    assert torch.equal(pool.temporal.qdata[:, whole], pool.temporal.qdata[:, per_layer])
    assert torch.equal(pool.temporal.scale[:, whole], pool.temporal.scale[:, per_layer])
    assert torch.equal(pool.conv[0][:, whole], pool.conv[0][:, per_layer])

    dst = torch.tensor([0, 5], dtype=torch.int64)
    for layer in range(L):
        pool.load_layer_to_active(layer, ssm[layer], conv[layer], per_layer, dst)
    assert torch.equal(conv[:, dst], conv[:, src])
    err = (ssm[:, dst] - ssm[:, src]).abs().max()
    assert err < 0.02 * ssm[:, src].abs().max()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scale_follows_the_state_dtype(dtype):
    """Quantize and dequantize must use the identical scale, which is why the
    scale is stored in the state's own dtype rather than always bf16."""
    pool = _pool(temporal_dtype=dtype)
    assert pool.temporal.scale.dtype is dtype


# -- The defect this pool's dtype exists to fix -----------------------------
#
# A snapshot is written from a working slot and read back into another one. A
# cold prefill never makes that trip: it carries the state in an fp32
# accumulator for the whole scan. So the store is only faithful if the round
# trip is the identity, and at the working pool's own bf16 it is not -- which
# is what made the same prompt return different text depending on whether it
# hit the cache.


def _round_trip(ckpt_dtype, state):
    """Working slot -> checkpoint slot -> a *different* working slot."""
    pool = _pool(ckpt_dtype=ckpt_dtype, conv_dtype=state.dtype)
    active = torch.zeros(L, 6, H, DV, DK, dtype=state.dtype)
    conv = torch.zeros(L, 6, 5, 7, dtype=state.dtype)
    src = torch.tensor([2], dtype=torch.int64)
    dst = torch.tensor([5], dtype=torch.int64)
    active[:, src[0]] = state

    ckpt = pool.alloc(1)
    assert ckpt is not None
    pool.store_from_active(active, [conv], src, ckpt)
    pool.load_to_active(active, [conv], ckpt, dst)
    return active[:, dst[0]]


def test_float32_checkpoint_reproduces_the_working_state_exactly():
    """The invariant the fix rests on. Not "close": equal."""
    torch.manual_seed(3)
    state = torch.randn(L, H, DV, DK, dtype=torch.float32)
    assert torch.equal(_round_trip(torch.float32, state), state)


def test_bfloat16_checkpoint_does_not():
    """The control. bfloat16 is what the single-pool arrangement stored, and it
    moves the state by ~1e-3 relative -- enough, measured end to end, to flip
    the first generated token's argmax on a 2048-character prompt."""
    torch.manual_seed(3)
    state = torch.randn(L, H, DV, DK, dtype=torch.float32)
    back = _round_trip(torch.bfloat16, state)
    assert not torch.equal(back, state)
    err = (back - state).abs().max() / state.abs().max()
    assert 1e-4 < err < 1e-1


def test_int8_checkpoint_is_lossy_and_says_so():
    """int8 buys slots, not exactness; keep the trade visible in the test."""
    torch.manual_seed(3)
    state = torch.randn(L, H, DV, DK, dtype=torch.float32)
    back = _round_trip(torch.int8, state)
    assert not torch.equal(back, state)
    assert (back - state).abs().max() < 0.02 * state.abs().max()


def test_per_layer_publish_is_exact_at_float32_too():
    """The publish the engine actually runs is one layer at a time, from inside
    that layer's forward, so it needs the same guarantee as the whole-pool one."""
    torch.manual_seed(4)
    pool = _pool(ckpt_dtype=torch.float32)
    ssm = torch.randn(L, 6, H, DV, DK, dtype=torch.float32)
    conv = torch.randn(L, 6, 5, 7, dtype=torch.float32)
    src = torch.tensor([1, 3], dtype=torch.int64)
    dst = torch.tensor([0, 5], dtype=torch.int64)

    slots = pool.alloc(2)
    assert slots is not None
    for layer in range(L):
        pool.store_layer_from_active(layer, ssm[layer], conv[layer], src, slots)
    for layer in range(L):
        pool.load_layer_to_active(layer, ssm[layer], conv[layer], slots, dst)

    assert torch.equal(ssm[:, dst], ssm[:, src])
    assert torch.equal(conv[:, dst], conv[:, src])


def test_a_bfloat16_working_pool_still_reaches_the_checkpoint_at_full_width():
    """The pools differ in dtype by design: the working state is bf16 because it
    sets concurrency, and the snapshot is fp32 because it sets correctness. The
    copy has to convert, not alias."""
    torch.manual_seed(5)
    pool = _pool(ckpt_dtype=torch.float32, conv_dtype=torch.bfloat16)
    active = torch.randn(L, 6, H, DV, DK, dtype=torch.bfloat16)
    conv = torch.randn(L, 6, 5, 7, dtype=torch.bfloat16)
    src = torch.tensor([2], dtype=torch.int64)
    slots = pool.alloc(1)
    assert slots is not None

    pool.store_from_active(active, [conv], src, slots)
    assert pool.temporal.data.dtype is torch.float32
    stored = pool.temporal.data[:, slots[0]]
    assert torch.equal(stored, active[:, src[0]].to(torch.float32))


def test_dense_store_is_not_the_quantized_one():
    """`is_quantized` picks the copy path, so it has to be right on both."""
    assert DenseCheckpointStore.is_quantized is False
    assert Int8CheckpointStore.is_quantized is True
    assert _pool(ckpt_dtype=torch.float32).temporal.is_quantized is False
    assert _pool(ckpt_dtype=torch.int8).temporal.is_quantized is True


@pytest.mark.parametrize(
    "name,expected",
    [("float32", torch.float32), ("bfloat16", torch.bfloat16), ("int8", torch.int8)],
)
def test_dtype_names_resolve(name, expected):
    assert resolve_ckpt_dtype(name) is expected
    assert resolve_ckpt_dtype(expected) is expected


def test_unknown_dtype_name_is_rejected_with_the_choices():
    with pytest.raises(ValueError, match="float32"):
        resolve_ckpt_dtype("fp8_e4m3")


def test_float32_slot_costs_about_four_int8_ones():
    """The capacity trade, in the units the sizing code uses. This is also the
    number a default change has to be argued from: at 27B one float32 slot is
    ~154MB against ~78MB for a working chunk, so it is two working slots, not a
    rounding error."""
    shape = dict(
        num_layers=L,
        num_heads=H,
        head_v_dim=DV,
        head_k_dim=DK,
        conv_shapes=[],  # temporal only: the conv window is the same either way
        conv_dtype=torch.bfloat16,
        temporal_dtype=torch.bfloat16,
    )
    f32 = MambaCheckpointPool.bytes_per_slot_estimate(ckpt_dtype=torch.float32, **shape)
    i8 = MambaCheckpointPool.bytes_per_slot_estimate(ckpt_dtype=torch.int8, **shape)
    # int8 carries a per-(head, k-channel) scale, so it is a little over a
    # quarter rather than exactly one.
    assert 3.0 < f32 / i8 < 4.0
