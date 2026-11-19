"""Numeric check: our AITER MXFP4 W4A16 solution vs a dequantized torch reference.

Builds synthetic MXFP4 experts at the real DeepSeek-V4 TP4 shapes (hidden 5120,
per-partition intermediate 576, padded to 640 by the weight preprocessor), runs
the registered ``aiter`` MoE solution, and compares against a plain PyTorch
reference computed from the *unpadded* weights.

Run inside the engine container:

  PYTHONPATH=/workspace/.vendor:/workspace/.wt-ait/tokenspeed-kernel/python \
    python3 /workspace/../specs/aiter-moe-numeric-check.py
"""

import os
import types

os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")

import torch  # noqa: E402
from tokenspeed_kernel.ops.moe.aiter import aiter_mxfp4_moe_apply  # noqa: E402
from tokenspeed_kernel.ops.moe.aiter import aiter_mxfp4_moe_weights  # noqa: E402

# E=384 is production but needs ~2 GiB of experts; the heuristic FlyDSL pair is
# chosen without a tuned row and does not depend on E (see
# specs/aiter-dispatch-probe.py), so E=8 validates the same kernels when the GPUs
# are busy with a benchmark cell.
E = int(os.environ.get("E", "384"))
H, I, TOPK, M = 5120, 576, 6, 16
LIMIT = 10.0
XSCALE = float(os.environ.get("XSCALE", "0.001"))
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)

# E2M1 codebook, low nibble first (the packed layout the loader produces).
FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
    device=dev,
)


def dequant(weight_u8: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """[E, N, K/2] packed fp4 + [E, N, K/32] e8m0 -> [E, N, K] fp32."""
    lo = (weight_u8 & 0x0F).to(torch.long)
    hi = (weight_u8 >> 4).to(torch.long)
    packed = torch.stack((FP4_LUT[lo], FP4_LUT[hi]), dim=-1)  # [E, N, K/2, 2]
    vals = packed.reshape(*weight_u8.shape[:2], -1)
    scales = torch.pow(2.0, scale_u8.to(torch.float32) - 127.0)  # [E, N, K/32]
    return vals * scales.repeat_interleave(32, dim=-1)


QF8 = os.environ.get("QFP8", "1") == "1"


def q_fp8_1x32(t: torch.Tensor) -> torch.Tensor:
    """Model aiter's per-1x32 e4m3 activation quant with an E8M0 block scale."""
    if not QF8:
        return t
    blocks = t.float().reshape(*t.shape[:-1], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    # E8M0: power-of-two scale, rounded up so the block never overflows e4m3.
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (blocks / scale).to(torch.float8_e4m3fn).float() * scale
    return q.reshape(t.shape)


def reference(w13: torch.Tensor, s13: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor,
              x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor) -> torch.Tensor:
    """Dequantize one expert at a time so 384 experts never sit in fp32 at once."""
    out = torch.zeros_like(x, dtype=torch.float32)
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for t in range(x.shape[0]):
        acc = torch.zeros(H, dtype=torch.float32, device=x.device)
        for k in range(TOPK):
            e = int(topk_ids[t, k])
            if e not in cache:
                cache[e] = (dequant(w13[e: e + 1], s13[e: e + 1])[0],
                            dequant(w2[e: e + 1], s2[e: e + 1])[0])
            gate_up, down = cache[e]
            xq = q_fp8_1x32(x[t].float())
            gate = gate_up[:I] @ xq
            up = gate_up[I:] @ xq
            gate = torch.clamp(gate, max=LIMIT)
            up = torch.clamp(up, -LIMIT, LIMIT)
            act = q_fp8_1x32(torch.sigmoid(gate) * gate * up)  # down is [H, I]
            acc += topk_w[t, k].float() * (down @ act)
        out[t] = acc
    return out


class _W(torch.nn.Module):
    pass


def main() -> None:
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g)
    s13 = torch.randint(120, 128, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev, generator=g)
    s2 = torch.randint(120, 128, (E, H, I // 32), dtype=torch.uint8, device=dev, generator=g)

    # Keep the SwiGLU clamps out of play (swiglu_limit=10): saturated gates make
    # the comparison a test of the clamp threshold, not of the weight layout.
    x = (torch.randn(M, H, device=dev, generator=g) * XSCALE).to(torch.bfloat16)
    topk_w = torch.softmax(torch.randn(M, TOPK, device=dev, generator=g), -1)
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)

    w = _W()
    w.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
    w.w13_weight_scale = torch.nn.Parameter(s13, requires_grad=False)
    w.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    w.w2_weight_scale = torch.nn.Parameter(s2, requires_grad=False)
    w.swiglu_arg = types.SimpleNamespace(alpha=1.0, limit=LIMIT)

    aiter_mxfp4_moe_weights({}, w)
    print("shuffled", tuple(w.w13_weight.shape), w.w13_weight.dtype,
          tuple(w.w2_weight.shape), tuple(w.w2_weight_scale.shape),
          "pads", w._aiter_intermediate_pad, w._aiter_hidden_pad, flush=True)

    got = aiter_mxfp4_moe_apply({}, x, w, None, topk_weights=topk_w, topk_ids=topk_ids)
    want = reference(w13, s13, w2, s2, x, topk_ids, topk_w)

    got_f = got.float()
    err = (got_f - want).abs()
    denom = want.abs().mean().clamp_min(1e-9)
    print(f"out {tuple(got.shape)} {got.dtype} finite={bool(torch.isfinite(got).all())}")
    print(f"ref |x|mean {want.abs().mean().item():.4e}  got |x|mean {got_f.abs().mean().item():.4e}")
    rel_l2 = (err.pow(2).sum().sqrt() / want.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(got_f.flatten(), want.flatten(), dim=0).item()
    print(f"max abs err {err.max().item():.4e}  mean abs err {err.mean().item():.4e} "
          f"(rel {err.mean().item() / denom.item():.2%})")
    print(f"rel L2 {rel_l2:.3%}  cosine {cos:.6f}")


if __name__ == "__main__":
    main()
