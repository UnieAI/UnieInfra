"""Bench one MoE apply solution across token counts at the real TP4 shapes.

Used to explain why the AITER solution helps decode but loses extend-heavy
throughput: the plan-selected kernel is timed for M = 1 ... 4096 rows.

Run inside the engine container, once per solution:

  MOE_BACKEND=aiter PYTHONPATH=/workspace/.vendor:/workspace/.wt-ait/tokenspeed-kernel/python \
    python3 /specs/moe-solution-bench.py
"""

import os
import types

os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")

import torch  # noqa: E402
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights  # noqa: E402

BACKEND = os.environ.get("MOE_BACKEND", "aiter")
E = int(os.environ.get("E", "384"))
H, I, TOPK = 5120, 576, 6
LIMIT = 10.0
MS = [int(m) for m in os.environ.get("MS", "1,16,64,128,256,512,1024,2048,4096").split(",")]
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)


class _W(torch.nn.Module):
    pass


def build() -> _W:
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g)
    s13 = torch.randint(120, 128, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev, generator=g)
    s2 = torch.randint(120, 128, (E, H, I // 32), dtype=torch.uint8, device=dev, generator=g)
    w = _W()
    w.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
    w.w13_weight_scale = torch.nn.Parameter(s13, requires_grad=False)
    w.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    w.w2_weight_scale = torch.nn.Parameter(s2, requires_grad=False)
    w.swiglu_arg = types.SimpleNamespace(alpha=1.0, limit=LIMIT)
    w.hidden_size = H
    w.num_experts = E
    return w


def make_plan():
    return moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ispp=I,
        hidden=H,
        num_experts=E,
        top_k=TOPK,
        swiglu_form="standard",
        activation_clamped=True,
        expert_id_repeats=False,
        internal_activation_dtype="input",
        features=frozenset(),
        solution=BACKEND,
        fast_math=True,
        combine_order="rank",
    )


def bench(plan, w, m: int) -> tuple[float, float]:
    x = (torch.randn(m, H, device=dev, generator=g) * 0.001).to(torch.bfloat16)
    tw = torch.softmax(torch.randn(m, TOPK, device=dev, generator=g), -1)
    ti = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(m)]).to(torch.int32)

    def run() -> None:
        moe_apply(plan, x, w, None, topk_weights=tw, topk_ids=ti)

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    iters = 200 if m <= 128 else 50
    ev0.record()
    for _ in range(iters):
        run()
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / iters
    return ms, m / ms * 1000.0


def main() -> None:
    plan = make_plan()
    w = build()
    moe_process_weights(plan, w)
    print(f"backend={BACKEND} kernel={plan['apply_kernel_name']}", flush=True)
    print(f"{'M':>6} {'ms':>8} {'expert-tok/s':>13}", flush=True)
    for m in MS:
        ms, rate = bench(plan, w, m)
        print(f"{m:6d} {ms:8.3f} {rate:13.0f}", flush=True)


if __name__ == "__main__":
    main()
