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
"""Hopper prefills gemma-4's 512-wide global layers on FA4, not Triton.

FA3 stops at 256-wide heads, so those layers used to select the Triton
fallback, which ran them 3-6x slower. The FA4 build in thirdparty/vllm_fa4
serves 512 on SM90; selection must pick it by default, and it must agree with
Triton on paged caches whose pages are spaced apart the way a flat KV plane
lays them out.
"""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha import mha_extend_with_kvcache, mha_prefill
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(not platform.is_hopper, reason="the FA4 512 path is SM90"),
]

PAGE = 64
HEAD_DIM = 512
HEADS_Q = 32
HEADS_KV = 4
SPACING = 20  # gemma-4's global group on the flat plane


def _output(result):
    result = getattr(result, "output", result)
    return result[0] if isinstance(result, tuple) else result


def _paged(q_lens: list[int], prefix_lens: list[int]):
    g = torch.Generator(device="cuda").manual_seed(0)
    kv_lens = [q + p for q, p in zip(q_lens, prefix_lens)]
    pages = [math.ceil(n / PAGE) for n in kv_lens]
    total = sum(pages)
    shape = ((total + 1) * SPACING + 1, PAGE, HEADS_KV, HEAD_DIM)
    k_cache = torch.randn(shape, generator=g, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(shape, generator=g, device="cuda", dtype=torch.bfloat16)
    page_table = torch.zeros(len(q_lens), max(pages), dtype=torch.int32, device="cuda")
    first = 0
    for i, n in enumerate(pages):
        ids = torch.arange(first, first + n, device="cuda", dtype=torch.int32)
        page_table[i, :n] = (ids + 1) * SPACING
        first += n
    q = torch.randn(
        sum(q_lens), HEADS_Q, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16
    )
    cu_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    cu_kv = torch.tensor(
        [0, *torch.tensor(kv_lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    seq = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    return (
        q,
        cu_q,
        cu_kv,
        k_cache,
        v_cache,
        page_table,
        seq,
        max(q_lens),
        max(kv_lens),
    )


@pytest.mark.parametrize(
    "q_lens,prefix_lens",
    [
        ([300], [0]),
        ([200, 77], [1000, 4500]),
        ([64, 1, 1, 1], [3000, 3000, 2000, 9000]),
    ],
)
def test_extend_matches_triton(q_lens, prefix_lens):
    args = _paged(q_lens, prefix_lens)
    scale = 1 / math.sqrt(HEAD_DIM)
    fa4 = _output(mha_extend_with_kvcache(*args, is_causal=True, softmax_scale=scale))
    ref = _output(
        mha_extend_with_kvcache(
            *args, is_causal=True, softmax_scale=scale, solution="triton"
        )
    )
    torch.testing.assert_close(fa4, ref, atol=1e-2, rtol=1e-2)


def test_prefill_matches_triton():
    g = torch.Generator(device="cuda").manual_seed(1)
    lens = [100, 333]
    total = sum(lens)
    q = torch.randn(
        total, HEADS_Q, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        total, HEADS_KV, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        total, HEADS_KV, HEAD_DIM, generator=g, device="cuda", dtype=torch.bfloat16
    )
    cu_cpu = [0, lens[0], total]
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device="cuda")
    scale = 1 / math.sqrt(HEAD_DIM)
    fa4 = _output(mha_prefill(q, k, v, cu, cu_cpu, max(lens), softmax_scale=scale))
    ref = _output(
        mha_prefill(
            q, k, v, cu, cu_cpu, max(lens), softmax_scale=scale, solution="triton"
        )
    )
    torch.testing.assert_close(fa4, ref, atol=1e-2, rtol=1e-2)


def test_selection_prefers_fa4():
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import format_signatures

    (signature,) = format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.bfloat16}
    )
    kernel = select_kernel(
        "attention",
        "mha_extend_with_kvcache",
        signature,
        traits={
            "head_dim": HEAD_DIM,
            "page_size": PAGE,
            "is_causal": True,
            "logit_cap": False,
            "return_lse": False,
            "sinks": False,
            "sliding_window": False,
        },
    )
    assert kernel.name == "fa4_sm90_mha_extend_with_kvcache"
