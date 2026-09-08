"""剪枝在**帶 KV 快取**的路徑上的正確性。

test_prefill_window_prune.py 走的是 triton_mha_prefill(無快取,q 就是整段);
生產走的是 triton_mha_extend_with_kvcache —— query 是 KV 序列的後綴,
`cur_q_start = cur_seq_len - cur_seq_len_extend` 非零,而剪枝的下界
`q_lo - W` 正是由它算出來的。那條路徑沒被前一支測到。
"""
import math

import pytest
import torch

from tokenspeed_kernel.ops.attention.triton.mha_prefill import (
    triton_mha_extend_with_kvcache,
)

PAGE = 64
H = 4
D = 64


def _build_paged(kv_lens, device, dtype):
    pages_per = [(L + PAGE - 1) // PAGE for L in kv_lens]
    k_cache = torch.zeros(sum(pages_per), PAGE, H, D, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    page_table = torch.zeros(len(kv_lens), max(pages_per), device=device, dtype=torch.int32)
    ks, vs, nxt = [], [], 0
    for i, L in enumerate(kv_lens):
        ki = torch.randn(L, H, D, device=device, dtype=dtype) * 0.5
        vi = torch.randn(L, H, D, device=device, dtype=dtype) * 0.5
        ks.append(ki)
        vs.append(vi)
        for p in range(pages_per[i]):
            rows = min(PAGE, L - p * PAGE)
            page_table[i, p] = nxt
            k_cache[nxt, :rows] = ki[p * PAGE : p * PAGE + rows]
            v_cache[nxt, :rows] = vi[p * PAGE : p * PAGE + rows]
            nxt += 1
    return k_cache, v_cache, page_table, ks, vs


def reference(qs, ks, vs, q_lens, kv_lens, window_left, scale):
    outs = []
    for q, k, v, nq, nk in zip(qs, ks, vs, q_lens, kv_lens):
        qf, kf, vf = q.float(), k.float(), v.float()
        s = torch.einsum("qhd,khd->hqk", qf, kf) * scale
        # query 是 KV 的後綴
        qi = torch.arange(nk - nq, nk, device=q.device)[:, None]
        ki = torch.arange(nk, device=q.device)[None, :]
        mask = qi >= ki
        if window_left >= 0:
            mask = mask & (qi <= ki + window_left)
        s = s.masked_fill(~mask[None], float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vf))
    return torch.cat(outs, 0)


@pytest.mark.parametrize(
    "q_lens,kv_lens,window_left",
    [
        ([64], [64], 16),          # 第一個 chunk:q_start = 0
        ([64], [512], 16),         # 後續 chunk:q_start = 448,窗遠在左邊
        ([64], [512], 1023),       # 窗大於整段
        ([128], [1024], 255),      # 窗在 BLOCK 邊界
        ([128], [1024], -1),       # 無窗,純因果 —— 剪的是上界
        ([37, 96], [300, 512], 64),  # 多序列、非對齊
        ([256], [4096], 1023),     # gemma 的窗、長脈絡
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_extend_matches_reference(q_lens, kv_lens, window_left, dtype):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    dev = "cuda"
    k_cache, v_cache, page_table, ks, vs = _build_paged(kv_lens, dev, dtype)
    qs = [torch.randn(n, H, D, device=dev, dtype=dtype) * 0.5 for n in q_lens]
    q = torch.cat(qs, 0)
    cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=dev)
    cu_kv = torch.tensor([0, *torch.tensor(kv_lens).cumsum(0).tolist()], dtype=torch.int32, device=dev)
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(D)
    got = triton_mha_extend_with_kvcache(
        q, cu_q, cu_kv, k_cache, v_cache, page_table, cache_seqlens,
        max(q_lens), max(kv_lens), is_causal=True,
        window_left=window_left, softmax_scale=scale,
    )
    want = reference(qs, ks, vs, q_lens, kv_lens, window_left, scale)
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)
