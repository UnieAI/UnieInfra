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

"""A model-dtype query over an fp8 paged KV cache, on the Triton kernels.

The reference is the same kernel over a bf16 cache upcast from the same fp8
bytes. The kernel upcasts the fp8 cache to the query dtype before any math, so
quantization error is common to both arms and what is compared is the kernel's
handling of the storage dtype. The tl.dot kernels (extend, grouped decode) agree
exactly; the elementwise non-grouped decode is held to one bf16 ulp, which is
how far its fp8 and bf16 variants were measured apart on GB10.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_fp8_cache_query_dtype,
)

_FP8_KV_DTYPES = [
    pytest.param(torch.float8_e4m3fn, id="e4m3"),
    pytest.param(torch.float8_e5m2, id="e5m2"),
]


@pytest.mark.parametrize("kv_dtype", _FP8_KV_DTYPES)
def test_triton_reads_fp8_cache_with_a_model_dtype_query(kv_dtype, h100_platform):
    # Triton has no fp8-query kernel, so the query must stay in the model dtype.
    assert (
        mha_fp8_cache_query_dtype(
            kv_dtype, torch.bfloat16, 128, solution="triton", platform=h100_platform
        )
        == torch.bfloat16
    )


def _paged_cache(
    seqlens: list[int], num_kv_heads: int, head_dim: int, page_size: int, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 K/V pages with random rows for each sequence, and its page table."""
    pages_per_seq = [(n + page_size - 1) // page_size for n in seqlens]
    total_pages = sum(pages_per_seq)
    shape = (total_pages, page_size, num_kv_heads, head_dim)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16)
    page_table = torch.zeros(
        len(seqlens), max(pages_per_seq), device=device, dtype=torch.int32
    )
    first = 0
    for row, n_pages in enumerate(pages_per_seq):
        page_table[row, :n_pages] = torch.arange(
            first, first + n_pages, device=device, dtype=torch.int32
        )
        first += n_pages
    return k, v, page_table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("kv_dtype", _FP8_KV_DTYPES)
@pytest.mark.parametrize("head_dim", [128, 256])
def test_triton_extend_over_fp8_cache_matches_upcast_cache(kv_dtype, head_dim):
    torch.manual_seed(0)
    device = "cuda"
    num_q_heads, num_kv_heads, page_size = 8, 2, 64
    prefix = [63, 48, 17, 80]
    new = [3, 1, 2, 4]
    seqlens = [p + n for p, n in zip(prefix, new)]
    k, v, page_table = _paged_cache(seqlens, num_kv_heads, head_dim, page_size, device)
    k_fp8, v_fp8 = k.to(kv_dtype), v.to(kv_dtype)
    q = torch.randn(
        sum(new), num_q_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    def cumulative(lengths: list[int]) -> torch.Tensor:
        return torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device=device,
            dtype=torch.int32,
        )

    def run(k_cache, v_cache):
        return mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=cumulative(new),
            cu_seqlens_kv=cumulative(seqlens),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=torch.tensor(seqlens, device=device, dtype=torch.int32),
            max_seqlen_q=max(new),
            max_seqlen_k=max(seqlens),
            is_causal=True,
            solution="triton",
        )

    out = run(k_fp8, v_fp8)
    assert out.dtype == torch.bfloat16
    ref = run(k_fp8.to(torch.bfloat16), v_fp8.to(torch.bfloat16))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("kv_dtype", _FP8_KV_DTYPES)
@pytest.mark.parametrize("num_q_heads", [2, 8], ids=["mha", "gqa"])
def test_triton_decode_over_fp8_cache_matches_upcast_cache(kv_dtype, num_q_heads):
    torch.manual_seed(0)
    device = "cuda"
    num_kv_heads, head_dim, page_size = 2, 128, 64
    seqlens = [63, 48, 17, 200]
    k, v, page_table = _paged_cache(seqlens, num_kv_heads, head_dim, page_size, device)
    k_fp8, v_fp8 = k.to(kv_dtype), v.to(kv_dtype)
    q = torch.randn(
        len(seqlens), num_q_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    def run(k_cache, v_cache):
        return mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=torch.tensor(seqlens, device=device, dtype=torch.int32),
            max_seqlen_k=max(seqlens),
            max_seqlen_q=1,
            solution="triton",
        )

    out = run(k_fp8, v_fp8)
    assert out.dtype == torch.bfloat16
    ref = run(k_fp8.to(torch.bfloat16), v_fp8.to(torch.bfloat16))
    ulp = 0.0 if num_q_heads > num_kv_heads else 2.0**-7
    torch.testing.assert_close(out, ref, rtol=ulp, atol=ulp)
