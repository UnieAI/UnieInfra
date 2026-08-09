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

"""Fused RoPE+KV-write puts exactly the same bits in the cache as the split path.

Models call the fused form (``fused_set_kv_buffer_arg``) instead of RoPE
followed by ``store_kv_cache``, to save a kernel launch per layer. That swap is
only safe if the two write identical bytes -- a fused write that lands in the
wrong slot, or rounds differently, does not crash: the model keeps generating,
just wrongly. Greedy decode on Qwen3-0.6B did diverge between the two paths
once, and this is the test that placed the blame elsewhere (on the layout of
the Q tensor handed to attention, not on the KV bytes).

Also pins the wrapper behaviour the fast path depends on: ``positions`` given
as int32 must produce the same answer as int64 (the wrapper skips a no-op
``.to()``), and the out-of-place form must match the in-place one exactly.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("tokenspeed_kernel")

from tokenspeed_kernel.ops.kvcache.triton import store_kv_cache  # noqa: E402
from tokenspeed_kernel.thirdparty.cuda.rope import (  # noqa: E402
    apply_rope_with_cos_sin_cache_inplace as rope,
)

HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
MAX_POS = 8192
SLOTS = 4096

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


class _FusedArg:
    """Mirrors the runtime's ``FusedSetKVBufferArg`` fields the kernel reads."""

    def __init__(self, value, k_buffer, v_buffer, cache_loc):
        self.value = value
        self.k_buffer = k_buffer
        self.v_buffer = v_buffer
        self.k_scale = None
        self.v_scale = None
        self.cache_loc = cache_loc


@pytest.fixture(scope="module")
def cos_sin_cache():
    inv = 1.0 / (
        1e6 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    freqs = torch.einsum("i,j->ij", torch.arange(MAX_POS, dtype=torch.float32), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).cuda()


def _inputs(n: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pos = torch.randint(0, MAX_POS, (n,), generator=gen, device="cuda").to(torch.int64)
    q = torch.randn(
        n, NUM_Q_HEADS * HEAD_DIM, generator=gen, dtype=torch.bfloat16, device="cuda"
    )
    k = torch.randn(
        n, NUM_KV_HEADS * HEAD_DIM, generator=gen, dtype=torch.bfloat16, device="cuda"
    )
    v = torch.randn_like(k)
    loc = torch.randperm(SLOTS, generator=gen, device="cuda")[:n].to(torch.int64)
    return pos, q, k, v, loc


@pytest.mark.parametrize("n_tokens", [1, 7, 64, 1024])
@pytest.mark.parametrize("is_neox", [True, False])
def test_fused_kv_write_matches_split_path(cos_sin_cache, n_tokens, is_neox):
    pos, q, k, v, loc = _inputs(n_tokens, seed=n_tokens)

    # Split path: RoPE in place, then the standalone scatter.
    k_split = torch.zeros(
        SLOTS, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    v_split = torch.zeros_like(k_split)
    q_a, k_a = q.clone(), k.clone()
    rope(
        positions=pos,
        query=q_a,
        key=k_a,
        head_size=HEAD_DIM,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )
    store_kv_cache(
        k_a.view(n_tokens, NUM_KV_HEADS, HEAD_DIM),
        v.view(n_tokens, NUM_KV_HEADS, HEAD_DIM),
        k_split,
        v_split,
        loc,
    )

    # Fused path: the RoPE kernel scatters K and V itself.
    k_fused = torch.zeros_like(k_split)
    v_fused = torch.zeros_like(v_split)
    q_b, k_b = q.clone(), k.clone()
    q_rope = torch.empty_like(q)
    rope(
        positions=pos,
        query=q_b,
        key=k_b,
        head_size=HEAD_DIM,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        output_q_rope=q_rope,
        fused_set_kv_buffer_arg=_FusedArg(
            v.view(n_tokens, NUM_KV_HEADS, HEAD_DIM),
            k_fused.view(SLOTS, -1),
            v_fused.view(SLOTS, -1),
            loc,
        ),
    )
    torch.cuda.synchronize()

    # Bitwise, not allclose: these are the same computation, not two
    # approximations of one. Any drift here is a bug, however small.
    assert torch.equal(q_rope, q_a)
    assert torch.equal(k_fused, k_split)
    assert torch.equal(v_fused, v_split)


@pytest.mark.parametrize("is_neox", [True, False])
def test_int32_positions_match_int64(cos_sin_cache, is_neox):
    """The wrapper skips ``.to(int64)`` when it is a no-op; int32 must still work."""
    pos, q, k, _, _ = _inputs(37, seed=11)

    q64, k64 = q.clone(), k.clone()
    rope(
        positions=pos,
        query=q64,
        key=k64,
        head_size=HEAD_DIM,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )
    q32, k32 = q.clone(), k.clone()
    rope(
        positions=pos.to(torch.int32),
        query=q32,
        key=k32,
        head_size=HEAD_DIM,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )
    torch.cuda.synchronize()
    assert torch.equal(q32, q64)
    assert torch.equal(k32, k64)


@pytest.mark.parametrize("is_neox", [True, False])
def test_out_of_place_matches_in_place(cos_sin_cache, is_neox):
    """``output_q_rope``/``output_k_rope`` must not change the result.

    The wrapper reuses the input's 3-D view for the output when they alias, so
    the aliasing and non-aliasing forms take slightly different code paths.
    """
    pos, q, k, _, _ = _inputs(37, seed=12)

    q_ip, k_ip = q.clone(), k.clone()
    rope(
        positions=pos,
        query=q_ip,
        key=k_ip,
        head_size=HEAD_DIM,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    rope(
        positions=pos,
        query=q,
        key=k,
        head_size=HEAD_DIM,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        output_q_rope=q_out,
        output_k_rope=k_out,
    )
    torch.cuda.synchronize()
    assert torch.equal(q_out, q_ip)
    assert torch.equal(k_out, k_ip)
