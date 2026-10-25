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
"""An fp8-e4m3 KV cache is read with an unquantized query where a kernel can.

Decode (Triton, any head) and 512-wide extend on Hopper (FA4's SM90 dequant
path) widen the cache in-kernel, so their only error is the cache's own
rounding: they must match the same read of the cache widened to bf16
beforehand. Where no such kernel exists the query is quantized as before.
"""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_fp8_cache_query_dtype,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(not platform.is_hopper, reason="checked on SM90"),
]

PAGE = 64
HEADS_Q = 32
FP8 = torch.float8_e4m3fn


def _output(result):
    result = getattr(result, "output", result)
    return result[0] if isinstance(result, tuple) else result


def _paged_fp8(batch: int, kv_len: int, heads_kv: int, head_dim: int):
    g = torch.Generator(device="cuda").manual_seed(0)
    pages = math.ceil(kv_len / PAGE)
    shape = (batch * pages + 1, PAGE, heads_kv, head_dim)
    k = torch.randn(shape, generator=g, device="cuda").to(FP8)
    v = torch.randn(shape, generator=g, device="cuda").to(FP8)
    page_table = (
        torch.arange(batch * pages, device="cuda", dtype=torch.int32) + 1
    ).view(batch, pages)
    seq = torch.full((batch,), kv_len, dtype=torch.int32, device="cuda")
    return k, v, page_table, seq


@pytest.mark.parametrize(
    "head_dim,heads_kv,window_left", [(512, 4, -1), (256, 16, 1023)]
)
@pytest.mark.parametrize("batch,kv_len", [(5, 3000), (2, 9000)])
def test_decode_reads_fp8_with_bf16_query(
    head_dim, heads_kv, window_left, batch, kv_len
):
    k, v, page_table, seq = _paged_fp8(batch, kv_len, heads_kv, head_dim)
    q = torch.randn(batch, HEADS_Q, head_dim, device="cuda", dtype=torch.bfloat16)
    common = dict(window_left=window_left, softmax_scale=1 / math.sqrt(head_dim))
    out = _output(
        mha_decode_with_kvcache(q, k, v, page_table, seq, kv_len, 1, **common)
    )
    ref = _output(
        mha_decode_with_kvcache(
            q,
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            page_table,
            seq,
            kv_len,
            1,
            solution="triton",
            **common,
        )
    )
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize(
    "q_lens,prefix_lens", [([300], [0]), ([200, 77], [1000, 4500])]
)
def test_512_extend_reads_fp8_with_bf16_query(q_lens, prefix_lens):
    kv_lens = [q + p for q, p in zip(q_lens, prefix_lens)]
    g = torch.Generator(device="cuda").manual_seed(1)
    pages = [math.ceil(n / PAGE) for n in kv_lens]
    shape = (sum(pages) + 1, PAGE, 4, 512)
    k = torch.randn(shape, generator=g, device="cuda").to(FP8)
    v = torch.randn(shape, generator=g, device="cuda").to(FP8)
    page_table = torch.zeros(len(q_lens), max(pages), dtype=torch.int32, device="cuda")
    first = 1
    for i, n in enumerate(pages):
        page_table[i, :n] = torch.arange(first, first + n, dtype=torch.int32)
        first += n
    q = torch.randn(
        sum(q_lens), HEADS_Q, 512, generator=g, device="cuda", dtype=torch.bfloat16
    )
    cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()]).cuda().int()
    cu_kv = torch.tensor([0, *torch.tensor(kv_lens).cumsum(0).tolist()]).cuda().int()
    seq = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    args = (cu_q, cu_kv)
    rest = (page_table, seq, max(q_lens), max(kv_lens))
    common = dict(is_causal=True, softmax_scale=1 / math.sqrt(512))
    out = _output(mha_extend_with_kvcache(q, *args, k, v, *rest, **common))
    ref = _output(
        mha_extend_with_kvcache(
            q,
            *args,
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            *rest,
            solution="triton",
            **common,
        )
    )
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_query_dtype_follows_the_kernels():
    def pick(operator, head_dim, window_left):
        return mha_fp8_cache_query_dtype(
            operator, torch.bfloat16, head_dim, PAGE, window_left=window_left
        )

    assert pick("mha_decode_with_kvcache", 512, -1) == torch.bfloat16
    assert pick("mha_decode_with_kvcache", 256, 1023) == torch.bfloat16
    assert pick("mha_extend_with_kvcache", 512, -1) == torch.bfloat16
    # FA3 serves the 256-wide sliding extend with an fp8 query only.
    assert pick("mha_extend_with_kvcache", 256, 1023) == FP8
