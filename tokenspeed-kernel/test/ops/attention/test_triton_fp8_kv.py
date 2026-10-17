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

"""A high-precision query over an fp8 paged KV cache, on the Triton kernels.

The Triton MHA kernels already convert a loaded fp8 cache up to the query dtype;
what kept fp8 KV unreachable was the registration declaring one dtype across all
three roles, so selection never offered them a mixed signature. These tests pin
the declaration -- for decode *and* extend, since fp8 KV is only usable if both
accept it -- and then check the attention it produces is actually right.

The numerical reference is the same kernel over a bf16 cache built by upcasting
the *same* fp8 bytes, so cache quantization error is common to both arms and
cancels. What the comparison isolates is the kernel's handling of the fp8
storage dtype, which is the thing under test; measuring fp8's inherent accuracy
cost is a model-level question, not a kernel one.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha.triton import (
    triton_mha_decode_with_kvcache,
)
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

# Registration is a side effect of importing the ops package.
import tokenspeed_kernel.ops.attention  # noqa: F401  isort: skip

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

Q_DTYPES = (torch.float16, torch.bfloat16)
KV_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_DECODE_TRAITS = {
    "head_dim": 128,
    "page_size": 64,
    "sliding_window": False,
    "logit_cap": False,
    "sinks": False,
    "return_lse": False,
}
_EXTEND_TRAITS = {**_DECODE_TRAITS, "is_causal": True}


def _signature(q_dtype, kv_dtype):
    """The signature the dispatcher builds from real tensors, per role."""
    return format_signature(
        q=dense_tensor_format(q_dtype),
        k_cache=dense_tensor_format(kv_dtype),
        v_cache=dense_tensor_format(kv_dtype),
    )


@pytest.mark.parametrize("q_dtype", Q_DTYPES)
@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
@pytest.mark.parametrize(
    "mode,traits,expected",
    [
        (
            "mha_decode_with_kvcache",
            _DECODE_TRAITS,
            "triton_mha_decode_with_kvcache",
        ),
        ("mha_extend_with_kvcache", _EXTEND_TRAITS, "triton_mha_extend_with_kvcache"),
    ],
)
def test_fp8_kv_selects_the_triton_kernel(q_dtype, kv_dtype, mode, traits, expected):
    kernel = select_kernel(
        "attention",
        mode,
        _signature(q_dtype, kv_dtype),
        traits=traits,
        solution="triton",
    )

    assert kernel is not None, f"no triton kernel for {q_dtype} q over {kv_dtype} KV"
    assert kernel.name == expected


@pytest.mark.parametrize(
    "mode,traits",
    [
        ("mha_decode_with_kvcache", _DECODE_TRAITS),
        ("mha_extend_with_kvcache", _EXTEND_TRAITS),
    ],
)
def test_uniform_dtypes_still_select(mode, traits):
    """The pre-existing all-bf16 path is unchanged -- the control."""
    kernel = select_kernel(
        "attention",
        mode,
        _signature(torch.bfloat16, torch.bfloat16),
        traits=traits,
        solution="triton",
    )

    assert kernel is not None


@pytest.mark.parametrize(
    "mode,traits",
    [
        ("mha_decode_with_kvcache", _DECODE_TRAITS),
        ("mha_extend_with_kvcache", _EXTEND_TRAITS),
    ],
)
def test_fp8_query_over_fp8_cache_is_not_claimed(mode, traits):
    """An all-fp8 signature is deliberately NOT declared for Triton.

    Casting the query down to e4m3 costs accuracy for no benefit, and the output
    buffer is `empty_like(q)`, so it would also return the attention output in
    fp8. The runtime therefore leaves the query alone on this path; if someone
    reinstates the cast, selection should fail loudly here rather than silently
    degrade.
    """
    with pytest.raises(NoKernelFoundError):
        select_kernel(
            "attention",
            mode,
            _signature(torch.float8_e4m3fn, torch.float8_e4m3fn),
            traits=traits,
            solution="triton",
        )


# --- numerics -------------------------------------------------------------

BATCH, SEQ, PAGE, Q_HEADS, KV_HEADS, DIM = 4, 1024, 64, 32, 8, 128
PAGES_PER_SEQ = SEQ // PAGE
NUM_PAGES = BATCH * PAGES_PER_SEQ


def _page_table():
    return (
        torch.arange(NUM_PAGES, device="cuda", dtype=torch.int32)
        .view(BATCH, PAGES_PER_SEQ)
        .contiguous()
    )


def _cache(kv_dtype, *, kv_heads=KV_HEADS):
    pages = torch.randn(NUM_PAGES, PAGE, kv_heads, DIM, device="cuda") * 0.5
    return pages.to(kv_dtype)


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.to(torch.float32).flatten(), b.to(torch.float32).flatten(), dim=0
    ).item()


# Every e4m3/e5m2 value is exactly representable in bf16, so an ideal kernel
# would score 1.0 against the upcast reference. Measured floors on sm_120 are
# ~0.9995 (e4m3) and ~0.9986 (e5m2, grouped only); the non-grouped elementwise
# path is near-exact for both. The e5m2 shortfall is reproducible but its
# mechanism is NOT established -- the obvious candidate, `p.to(v.dtype)`
# quantizing the softmax probabilities, is ruled out: hand-quantizing p to e4m3
# scores 0.79, which the e4m3 arm plainly does not. Thresholds are therefore set
# from measurement, loose enough not to be flaky and tight enough that a real
# regression in cache handling still trips them.
_COSINE_FLOOR = {torch.float8_e4m3fn: 0.999, torch.float8_e5m2: 0.998}


@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
@pytest.mark.parametrize("q_heads,kv_heads", [(32, 8), (8, 8)])
def test_decode_over_fp8_cache_matches_upcast_reference(kv_dtype, q_heads, kv_heads):
    """Grouped (GQA) and non-grouped decode both read the fp8 cache correctly.

    The two head configurations exercise different inner kernels:
    `_fwd_grouped_kernel_stage1` (tl.dot) and `_fwd_kernel_stage1`
    (elementwise), which convert the loaded cache independently.
    """
    torch.manual_seed(0)
    k_q = _cache(kv_dtype, kv_heads=kv_heads)
    v_q = _cache(kv_dtype, kv_heads=kv_heads)
    k_ref, v_ref = k_q.to(torch.bfloat16), v_q.to(torch.bfloat16)
    q = torch.randn(BATCH, q_heads, DIM, device="cuda", dtype=torch.bfloat16) * 0.5

    common = dict(
        page_table=_page_table(),
        cache_seqlens=torch.full((BATCH,), SEQ, device="cuda", dtype=torch.int32),
        max_seqlen_k=SEQ,
        max_seqlen_q=1,
    )
    ref = triton_mha_decode_with_kvcache(q=q, k_cache=k_ref, v_cache=v_ref, **common)
    out = triton_mha_decode_with_kvcache(q=q, k_cache=k_q, v_cache=v_q, **common)

    # The output must stay at activation precision -- it is not a cache, and
    # `empty_like(q)` is only correct because the query is left in bf16.
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.to(torch.float32)).all()
    floor = _COSINE_FLOOR[kv_dtype]
    cosine = _cosine(out, ref)
    assert cosine > floor, f"cosine {cosine} below {floor} for {kv_dtype}"
