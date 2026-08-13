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

"""The publish and restore paths the engine's step actually takes.

``test_mamba_checkpoint_pool.py`` pins the store. This pins the two indexed
paths that reach it from a running step -- the decode-time page-boundary
publish and the copy-on-write a cache hit performs -- because those are where
the two pools' *index spaces* meet, and where the previous arrangement wrote a
snapshot into the active pool at a checkpoint pool's index.
"""

import pytest
import torch

from tokenspeed.runtime.cache.mamba_checkpoint_pool import MambaCheckpointPool
from tokenspeed.runtime.layers.attention.linear.mamba_state_scatter_triton import (
    fused_mamba_state_copy,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

L, ACTIVE, CKPT, H, DV, DK = 3, 8, 4, 2, 8, 6
CONV_SHAPE = (5, 7)
PAGE = 16


def _pool(ckpt_dtype):
    return MambaCheckpointPool(
        num_layers=L,
        num_slots=CKPT,
        num_heads=H,
        head_v_dim=DV,
        head_k_dim=DK,
        conv_shapes=[CONV_SHAPE],
        conv_dtype=torch.bfloat16,
        device="cuda",
        temporal_dtype=torch.bfloat16,
        ckpt_dtype=ckpt_dtype,
    )


def _active(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ssm = torch.randn(
        L, ACTIVE, H, DV, DK, device="cuda", dtype=torch.bfloat16, generator=g
    )
    conv = torch.randn(
        L, ACTIVE, *CONV_SHAPE, device="cuda", dtype=torch.bfloat16, generator=g
    )
    return conv, ssm


def _idx(values):
    return torch.tensor(values, device="cuda", dtype=torch.int64)


@pytest.mark.parametrize("ckpt_dtype", [torch.float32, torch.bfloat16, torch.int8])
def test_publish_then_restore_returns_the_working_state(ckpt_dtype):
    """Round trip through the real indexed paths, not the whole-pool helpers."""
    pool = _pool(ckpt_dtype)
    conv, ssm = _active()
    src, dst = _idx([2, 5]), _idx([1, 3])

    pool.store_from_active_indexed(conv, ssm, src, dst)
    back = _idx([6, 7])
    pool.load_to_active_indexed(conv, ssm, dst, back)

    # conv is copied verbatim at every dtype: it is W-1 raw tokens, never
    # quantized and never widened.
    assert torch.equal(conv[:, back], conv[:, src])
    if ckpt_dtype is torch.float32:
        # The invariant the fix rests on, through the path the engine runs.
        assert torch.equal(ssm[:, back], ssm[:, src])
    else:
        err = (ssm[:, back] - ssm[:, src]).abs().max()
        assert err <= 0.05 * ssm[:, src].abs().max()


@pytest.mark.parametrize("ckpt_dtype", [torch.float32, torch.int8])
def test_equal_indices_are_a_copy_not_a_self_copy(ckpt_dtype):
    """`src == dst` names two slots in two pools. Skipping it as a no-op --
    which is right within one pool -- silently drops one snapshot in every
    ``num_slots``."""
    pool = _pool(ckpt_dtype)
    conv, ssm = _active(seed=1)
    same = _idx([3])

    pool.store_from_active_indexed(conv, ssm, same, same)
    back = _idx([7])
    pool.load_to_active_indexed(conv, ssm, same, back)

    assert torch.equal(conv[:, back], conv[:, same])
    if ckpt_dtype is torch.float32:
        assert torch.equal(ssm[:, back], ssm[:, same])


@pytest.mark.parametrize("ckpt_dtype", [torch.float32, torch.int8])
def test_a_negative_slot_publishes_nothing(ckpt_dtype):
    """The scheduler writes -1 when it had no checkpoint slot to give; that is
    a cache miss to come, not an address."""
    pool = _pool(ckpt_dtype)
    conv, ssm = _active(seed=2)
    before_conv = pool.conv[0].clone()

    pool.store_from_active_indexed(conv, ssm, _idx([-1]), _idx([-1]))
    torch.cuda.synchronize()
    assert torch.equal(pool.conv[0][:, 1:], before_conv[:, 1:])


@pytest.mark.parametrize("ckpt_dtype", [torch.float32, torch.int8])
def test_the_page_filter_still_applies_across_pools(ckpt_dtype):
    """A snapshot is only published when the new cache length lands on a page
    boundary; the filter has to survive the move to a second pool."""
    pool = _pool(ckpt_dtype)
    conv, ssm = _active(seed=3)
    src, dst = _idx([2, 5]), _idx([1, 3])
    cache_lengths = torch.tensor([PAGE, PAGE + 1], device="cuda", dtype=torch.int32)

    pool.store_from_active_indexed(
        conv, ssm, src, dst, cache_lengths=cache_lengths, page_size=PAGE
    )
    back = _idx([6, 7])
    ssm[:, back] = 0
    conv[:, back] = 0
    pool.load_to_active_indexed(conv, ssm, dst, back)

    # Row 0 was aligned and published; row 1 was not, so its checkpoint slot
    # still holds whatever it held (here: uninitialized, so only row 0 is
    # asserted -- the point is that row 1's *source* did not reach it).
    assert torch.equal(conv[:, back[0]], conv[:, src[0]])
    assert not torch.equal(conv[:, back[1]], conv[:, src[1]])


def test_fused_copy_rejects_a_destination_with_a_different_slot_size():
    """A shape mismatch between the pools is a configuration bug; it must raise
    rather than reinterpret bytes."""
    src = torch.zeros(L, ACTIVE, H, DV, DK, device="cuda", dtype=torch.bfloat16)
    dst = torch.zeros(L, CKPT, H, DV, DK + 1, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="same number of elements"):
        fused_mamba_state_copy(src, _idx([0]), _idx([0]), dst_pool=dst)


def test_fused_copy_widens_on_the_way_out_and_narrows_on_the_way_back():
    """The one copy that converts. bf16 -> fp32 must be exact, and fp32 -> bf16
    must land back on the value it started from."""
    active = torch.randn(L, ACTIVE, H, DV, DK, device="cuda", dtype=torch.bfloat16)
    wide = torch.zeros(L, CKPT, H, DV, DK, device="cuda", dtype=torch.float32)

    fused_mamba_state_copy(active, _idx([4]), _idx([2]), dst_pool=wide)
    assert torch.equal(wide[:, 2], active[:, 4].to(torch.float32))

    fused_mamba_state_copy(wide, _idx([2]), _idx([0]), dst_pool=active)
    assert torch.equal(active[:, 0], active[:, 4])
