"""Probe aiter's MXFP4 MoE metadata selection without touching GPU memory.

Answers: for our production shape (TP4: 5120 hidden, 640 padded intermediate,
384 experts, topk 6) and the weight layout we build, which kernel pair does
`get_2stage_cfgs` return? Runs off tensors-free metadata only.
"""
import functools

import torch

from aiter.fused_moe import ActivationType, GateMode, QuantType, get_2stage_cfgs

BM = 32


def probe(label, *, shuf, gmode, gui, adt=torch.float8_e4m3fn, token=16, inter=640):
    m = get_2stage_cfgs(
        token=token,
        model_dim=5120,
        inter_dim=inter,
        expert=384,
        topk=6,
        dtype=torch.bfloat16,
        q_dtype_a=adt,
        q_dtype_w=torch.float4_e2m1fn_x2,
        q_type=QuantType.per_1x32,
        use_g1u1=True,
        activation=ActivationType.Silu,
        doweight_stage1=False,
        hidden_pad=0,
        intermediate_pad=64,
        is_shuffled=shuf,
        gate_mode=gmode,
        gate_up_interleave=gui,
    )

    def name(p):
        if isinstance(p, functools.partial):
            return p.keywords.get("kernelName") or getattr(p.func, "__name__", "fn")
        return getattr(p, "__name__", type(p).__name__)

    print(
        f"{label:22s} kn1={name(m.stage1)} kn2={name(m.stage2)} "
        f"block_m={m.block_m} ksplit={m.ksplit}"
    )


if __name__ == "__main__":
    probe("shuf=T interleaved/T", shuf=True, gmode=GateMode.INTERLEAVE.value, gui=True)
    probe("shuf=F interleaved/T", shuf=False, gmode=GateMode.INTERLEAVE.value, gui=True)
    probe("shuf=T interleaved/None", shuf=True, gmode=GateMode.INTERLEAVE.value, gui=None)
    probe("shuf=T bf16A", shuf=True, gmode=GateMode.INTERLEAVE.value, gui=True, adt=torch.bfloat16)
    for tok in (1, 64, 128, 2048):
        probe(f"shuf=T tok={tok}", shuf=True, gmode=GateMode.INTERLEAVE.value, gui=True, token=tok)
