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

"""Load-time AITER MXFP4 weight preprocessing: layout and allocator footprint."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required for AITER MXFP4 MoE", allow_module_level=True)

shuffle = pytest.importorskip("aiter.ops.shuffle")

from tokenspeed_kernel.ops.moe.aiter import (  # noqa: E402
    SHUFFLE_CHUNK_EXPERTS,
    aiter_mxfp4_moe_weights,
)

HIDDEN = 1024
# DeepSeek-V4.1-Flash at TP4: 576 pads to AITER's 640.
ISPP = 576
ISPP_PADDED = 640
E8M0_ONE = 0x7F


def _experts(num_experts: int, seed: int, padded_storage: bool) -> torch.nn.Module:
    """Checkpoint-layout experts; with ``padded_storage`` as the mxfp4 loader
    builds them for AITER (unpadded views of the padded buffers)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(shape, storage_shape, low, high):
        values = torch.randint(
            low, high, shape, dtype=torch.uint8, device="cuda", generator=gen
        )
        if padded_storage:
            buffer = torch.zeros(storage_shape, dtype=torch.uint8, device="cuda")
            view = buffer[tuple(slice(0, size) for size in shape)]
            view.copy_(values)
            values = view
        return torch.nn.Parameter(values, requires_grad=False)

    e, p = num_experts, ISPP_PADDED
    w = torch.nn.Module()
    w.w13_weight = rand((e, 2 * ISPP, HIDDEN // 2), (e, 2 * p, HIDDEN // 2), 0, 256)
    w.w13_weight_scale = rand(
        (e, 2 * ISPP, HIDDEN // 32), (e, 2 * p, HIDDEN // 32), 100, 140
    )
    w.w2_weight = rand((e, HIDDEN, ISPP // 2), (e, HIDDEN, p // 2), 0, 256)
    w.w2_weight_scale = rand((e, HIDDEN, ISPP // 32), (e, HIDDEN, 24), 100, 140)
    return w


def _reference(w: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Pad the whole stack with zero weights / E8M0 1.0 scales, then shuffle it."""
    num_experts = w.w13_weight.shape[0]
    pad = ISPP_PADDED - ISPP
    fp4, e8m0 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu

    weight, scale = w.w13_weight.data, w.w13_weight_scale.data
    zeros = weight.new_zeros((num_experts, pad, weight.shape[-1]))
    ones = scale.new_full((num_experts, pad, scale.shape[-1]), E8M0_ONE)
    gate, up = weight.split(ISPP, dim=1)
    gate_scale, up_scale = scale.split(ISPP, dim=1)
    w13 = torch.cat((gate, zeros, up, zeros), dim=1).contiguous()
    s13 = torch.cat((gate_scale, ones, up_scale, ones), dim=1)
    s13 = s13.reshape(-1, scale.shape[-1])

    weight, scale = w.w2_weight.data, w.w2_weight_scale.data
    w2 = torch.cat((weight, weight.new_zeros((*weight.shape[:-1], pad // 2))), -1)
    s2 = torch.cat(
        (scale, scale.new_full((*scale.shape[:-1], pad // 32), E8M0_ONE)), -1
    )
    s2 = s2.reshape(-1, s2.shape[-1])
    return {
        "w13_weight": shuffle.shuffle_weight(
            w13.view(fp4), is_guinterleave=True, gate_up=True
        ),
        "w13_weight_scale": shuffle.shuffle_scale(
            s13.view(e8m0), num_experts, True, True
        ),
        "w2_weight": shuffle.shuffle_weight(
            w2.contiguous().view(fp4), is_guinterleave=True, gate_up=False
        ),
        "w2_weight_scale": shuffle.shuffle_scale(
            s2.contiguous().view(e8m0), num_experts, True, False
        ),
    }


# One partial chunk, an exact multiple, and a multiple plus a remainder.
@pytest.mark.parametrize(
    "num_experts",
    [
        SHUFFLE_CHUNK_EXPERTS - 7,
        2 * SHUFFLE_CHUNK_EXPERTS,
        2 * SHUFFLE_CHUNK_EXPERTS + 5,
    ],
)
@pytest.mark.parametrize("padded_storage", [True, False])
def test_chunked_preprocessing_matches_whole_stack_shuffle(
    num_experts: int, padded_storage: bool
) -> None:
    w = _experts(num_experts, seed=num_experts, padded_storage=padded_storage)
    expected = _reference(w)

    aiter_mxfp4_moe_weights({}, w)

    for name, ref in expected.items():
        got = getattr(w, name)
        assert got.shape == ref.shape, name
        assert got.dtype == ref.dtype, name
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8)), name
    assert w.w13_weight.is_shuffled and w.w2_weight.is_shuffled
    assert w._aiter_intermediate_pad == ISPP_PADDED - ISPP


def test_preprocessing_padded_storage_allocates_and_frees_no_device_memory() -> None:
    # All layers' stacks are resident before the first one is processed, as in
    # the real loader. A layer that allocates new stacks and frees the old ones
    # either strands the freed blocks in the caching allocator or, released
    # with empty_cache, hands them back to the driver, and on ROCm that memory
    # is not allocatable again right away.
    layers = [
        _experts(3 * SHUFFLE_CHUNK_EXPERTS, seed=i, padded_storage=True)
        for i in range(6)
    ]
    # The first layer warms the cached blocks the chunk temporaries reuse.
    aiter_mxfp4_moe_weights({}, layers[0])
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    allocated = torch.cuda.memory_allocated()

    for w in layers[1:]:
        aiter_mxfp4_moe_weights({}, w)
    torch.cuda.synchronize()

    after = torch.cuda.memory_stats()
    assert after["num_device_alloc"] == stats["num_device_alloc"]
    assert after["num_device_free"] == stats["num_device_free"]
    assert torch.cuda.memory_allocated() == allocated
