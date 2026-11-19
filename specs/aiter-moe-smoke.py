"""Smoke test: call aiter fused MoE (DeepSeek-V4.1 a8w4 mxfp4 path) from a
non-vLLM process.  Shapes mirror DeepSeek-V4.1-Flash-1017 @ TP4 on gfx950:
E=8 (subset), hidden=5120, inter_dim(per GPU, padded)=640, topk=6.
"""

import os

os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")  # what vLLM's prep sets

import torch  # noqa: E402
import aiter  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter import ActivationType, QuantType  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.ops.shuffle import shuffle_scale, shuffle_weight  # noqa: E402

E, H, I, TOPK = 8, 5120, 640, 6
M = int(os.environ.get("M", "16"))
dev = "cuda"
fp4 = torch.float4_e2m1fn_x2
e8m0 = torch.float8_e8m0fnu
g = torch.Generator(device=dev).manual_seed(0)

# ---- once at weight load: 2x640 rows (576 real + 64 pad), zero pad rows ----
w13 = torch.zeros((E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
w13[:, : 2 * 576, :] = torch.randint(0, 256, (E, 2 * 576, H // 2), dtype=torch.uint8, device=dev, generator=g)
s13 = torch.full((E, 2 * I, H // 32), 0x7F, dtype=torch.uint8, device=dev)  # 2^0
s13[:, : 2 * 576 :] = torch.randint(120, 127, (E, 2 * 576, H // 32), dtype=torch.uint8, device=dev, generator=g)
w2 = torch.zeros((E, H, I // 2), dtype=torch.uint8, device=dev)
w2[:, :, : 576 // 2] = torch.randint(0, 256, (E, H, 576 // 2), dtype=torch.uint8, device=dev, generator=g)
s2 = torch.full((E, H, I // 32), 0x7F, dtype=torch.uint8, device=dev)
s2[:, :, : 576 // 32] = torch.randint(120, 127, (E, H, 576 // 32), dtype=torch.uint8, device=dev, generator=g)

w13s = shuffle_weight(w13.view(fp4), is_guinterleave=True, gate_up=True).contiguous()
s13s = shuffle_scale(s13.view(e8m0).reshape(-1, s13.shape[-1]), E, True, True).contiguous()
w2s = shuffle_weight(w2.view(fp4), is_guinterleave=True, gate_up=False).contiguous()
s2s = shuffle_scale(s2.view(e8m0).reshape(-1, s2.shape[-1]), E, True, False).contiguous()
print("shapes", w13s.shape, s13s.shape, w2s.shape, s2s.shape, w13s.dtype, s13s.dtype, flush=True)

# ---- per token ----
a = torch.randn((M, H), dtype=torch.bfloat16, device=dev) * 0.1
tw = torch.softmax(torch.randn((M, TOPK), dtype=torch.float32, device=dev), -1)
ti = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
print("topk_ids", ti.dtype, ti.shape, flush=True)

out = fused_moe(
    a,
    w13s,
    w2s,
    tw,
    ti,
    expert_mask=None,
    activation=ActivationType.Silu,
    quant_type=QuantType.per_1x32,
    doweight_stage1=False,
    w1_scale=s13s,
    w2_scale=s2s,
    dtype=torch.bfloat16,
    hidden_pad=0,
    intermediate_pad=64,  # (640-576)//64*64
    gate_mode=GateMode.INTERLEAVE.value,
    swiglu_limit=10.0,
)
print("OUT", out.shape, out.dtype, "finite", bool(torch.isfinite(out).all()), "absmean", out.abs().mean().item(), flush=True)
