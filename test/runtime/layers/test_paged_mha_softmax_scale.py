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

"""The paged MHA backend attends at the layer's own softmax scale.

The kernels default an omitted scale to ``1/sqrt(head_dim)``, which matches
every model that declares that scale and hides the omission. A model that
declares another one (Gemma 4: 1.0 after qk-norm) is then attended at the
wrong temperature and decodes fluent garbage. Runs the real Triton prefill
kernel through the backend and compares with a reference attention.
"""

import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_HEADS = 2
HEAD_DIM = 64
TOKENS = 16


def _backend():
    from tokenspeed.runtime.layers.attention.backends.paged.mha import (
        MHAAttnBackend,
    )

    # The prefill forward reads only these two fields; a full construction
    # needs a model config and a cache pool that play no part here.
    backend = MHAAttnBackend.__new__(MHAAttnBackend)
    backend.kernel_solution = "triton"
    backend.skip_softmax_threshold = 0.0
    return backend


def _metadata():
    from tokenspeed.runtime.layers.attention.backends.paged.mha import (
        MHAExtendMetadata,
    )

    lens = torch.tensor([TOKENS], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, TOKENS], dtype=torch.int32, device="cuda")
    return MHAExtendMetadata(
        page_table=torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        seq_lens=lens,
        extend_seq_lens=lens,
        cu_extend_seq_lens=cu,
        cu_seqlens_kv=cu,
        extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device="cuda"),
        extend_seq_lens_cpu=[TOKENS],
        cu_extend_seq_lens_cpu=[0, TOKENS],
        max_extend_seq_len=TOKENS,
    )


def _reference(q, k, v, scale):
    # [T, H, D] -> [H, T, D]; causal attention in fp32.
    q, k, v = (x.float().transpose(0, 1) for x in (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=scale
    )
    return out.transpose(0, 1).reshape(TOKENS, NUM_HEADS * HEAD_DIM)


@pytest.mark.parametrize("scale", [1.0, 1.0 / math.sqrt(HEAD_DIM)])
def test_prefill_attends_at_the_layer_scale(scale):
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(TOKENS, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    layer = SimpleNamespace(
        scaling=scale,
        sliding_window_size=-1,
        logit_cap=0.0,
        tp_q_head_num=NUM_HEADS,
        v_head_dim=HEAD_DIM,
    )
    out = _backend()._forward_prefill(q, k, v, layer, _metadata(), None)
    torch.testing.assert_close(
        out.float(), _reference(q, k, v, scale), atol=2e-2, rtol=2e-2
    )
