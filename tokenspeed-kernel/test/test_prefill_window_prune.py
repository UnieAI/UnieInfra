"""triton_mha_prefill 的滑動窗迴圈剪枝:與 PyTorch 參考逐元素比對。

剪枝把迴圈界從 [0, cur_seq_len) 縮到 [q_lo - W, q_hi + 1)。兩個界都來自
AND 進 final_mask 的條件,所以理論上不可能放進本來會被遮掉的 key —— 這支
測試是要證明實作也沒放掉本來該算的,特別是在下列邊界上:
  - W 大於整段序列(不該剪)
  - q_lo < W(下界被 0 夾住)
  - W 恰好落在 BLOCK_N 邊界上、以及差一格
  - 多序列、非對齊長度
"""
import math

import pytest
import torch

from tokenspeed_kernel.ops.attention.triton.mha_prefill import triton_mha_prefill


def reference(q, k, v, seqlens, window_left, scale):
    """因果 + 滑動窗注意力。window_left 是 exclusive 的左界(kernel 的語意)。"""
    outs = []
    off = 0
    for n in seqlens:
        qs, ks, vs = (t[off : off + n].float() for t in (q, k, v))
        s = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        qi = torch.arange(n, device=q.device)[:, None]
        ki = torch.arange(n, device=q.device)[None, :]
        mask = qi >= ki
        if window_left >= 0:
            mask = mask & (qi <= ki + window_left)
        s = s.masked_fill(~mask[None], float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vs))
        off += n
    return torch.cat(outs, 0)


@pytest.mark.parametrize(
    "seqlens,window_left",
    [
        ([64], -1),
        ([64], 4096),
        ([64], 16),
        ([256], 64),
        ([256], 63),
        ([300], 128),
        ([37, 211, 96], 64),
        ([512], 1023),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matches_reference(seqlens, window_left, dtype):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    h, d = 4, 64
    total = sum(seqlens)
    q, k, v = (
        torch.randn(total, h, d, device="cuda", dtype=dtype) for _ in range(3)
    )
    cu = torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    scale = 1.0 / math.sqrt(d)
    got = triton_mha_prefill(
        q, k, v, cu, cu.tolist(), max(seqlens),
        window_left=window_left, softmax_scale=scale,
    )
    want = reference(q, k, v, seqlens, window_left, scale)
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)
