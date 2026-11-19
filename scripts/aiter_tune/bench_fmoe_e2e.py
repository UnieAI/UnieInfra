"""End-to-end aiter.fused_moe timing for DeepSeek-V4.1-Flash TP4 MoE shapes.

Builds random MXFP4 experts in exactly the layout UnieInfra's aiter backend
produces (tokenspeed_kernel.ops.moe.aiter.aiter_mxfp4_moe_weights), then times
the same fused_moe call under one or more AITER_CONFIG_FMOE settings, inside a
CUDA graph (as in server decode). Outputs of each config are compared against
the first config (cosine similarity) to catch a wrong tuned kernel.

usage: python bench_fmoe_e2e.py --tokens 1,16,64 --cfg base=default --cfg tuned=/path.csv --out res.json
"""

import argparse
import json
import logging
import os
import time

import torch

E, HIDDEN, ISPP, TOPK = 384, 5120, 576, 6
SWIGLU_LIMIT = 10.0


class Rec(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        m = record.getMessage()
        if "FlyDSL" in m or "tuned" in m or "Fmoe" in m:
            self.msgs.append(m)


def build_weights(seed=0):
    from tokenspeed_kernel.ops.moe.aiter import aiter_mxfp4_moe_weights

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.nn.Module()

    def rnd_u8(shape, lo, hi):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, device="cuda", generator=g)

    w.w13_weight = torch.nn.Parameter(rnd_u8((E, 2 * ISPP, HIDDEN // 2), 0, 256), requires_grad=False)
    w.w13_weight_scale = torch.nn.Parameter(rnd_u8((E, 2 * ISPP, HIDDEN // 32), 118, 124), requires_grad=False)
    w.w2_weight = torch.nn.Parameter(rnd_u8((E, HIDDEN, ISPP // 2), 0, 256), requires_grad=False)
    w.w2_weight_scale = torch.nn.Parameter(rnd_u8((E, HIDDEN, ISPP // 32), 118, 124), requires_grad=False)
    aiter_mxfp4_moe_weights({}, w)
    return w


def make_inputs(M, seed):
    g = torch.Generator(device="cuda").manual_seed(1000 + seed)
    x = torch.randn((M, HIDDEN), dtype=torch.bfloat16, device="cuda", generator=g)
    logits = torch.randn((M, E), dtype=torch.float32, device="cuda", generator=g)
    tw, ti = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    return x, tw.float(), ti.to(torch.int32)


def call(w, x, tw, ti):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    return fused_moe(
        x, w.w13_weight, w.w2_weight, tw, ti,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=w.w13_weight_scale, w2_scale=w.w2_weight_scale,
        a1_scale=None, a2_scale=None, block_size_M=None, num_local_tokens=None,
        hidden_pad=w._aiter_hidden_pad, intermediate_pad=w._aiter_intermediate_pad,
        gate_mode=GateMode.INTERLEAVE.value, swiglu_limit=SWIGLU_LIMIT,
        dtype=torch.bfloat16,
    )


def set_cfg(path):
    import aiter.fused_moe as fm

    if path == "default":
        os.environ.pop("AITER_CONFIG_FMOE", None)
    else:
        os.environ["AITER_CONFIG_FMOE"] = path
    fm.cfg_2stages = None
    fm.get_2stage_cfgs.cache_clear()
    if hasattr(fm, "cfg_2stages_by_file"):
        fm.cfg_2stages_by_file.clear()


def time_graph(fn, iters, graph=True):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        run = gr.replay
    else:
        run = fn
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = []
    for _ in range(3):
        st.record()
        for _ in range(iters):
            run()
        en.record()
        torch.cuda.synchronize()
        best.append(st.elapsed_time(en) * 1000.0 / iters)
    return min(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--cfg", action="append", required=True, help="name=path|default")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    cfgs = [c.split("=", 1) for c in a.cfg]

    rec = Rec()
    logging.getLogger("aiter").addHandler(rec)
    t0 = time.time()
    w = build_weights()
    print(f"weights ready {time.time() - t0:.1f}s", flush=True)
    res = {}
    for M in tokens:
        x, tw, ti = make_inputs(M, M)
        ref = None
        for name, path in cfgs:
            set_cfg(path)
            rec.msgs.clear()
            out = call(w, x, tw, ti).clone()
            torch.cuda.synchronize()
            heur = any("no tuned FlyDSL" in m for m in rec.msgs)
            kern = [m for m in rec.msgs if "fallback" in m]
            us = time_graph(lambda: call(w, x, tw, ti), a.iters, graph=not a.no_graph)
            if ref is None:
                ref = out.float()
                cos = 1.0
            else:
                o = out.float()
                cos = float(torch.nn.functional.cosine_similarity(ref.flatten(), o.flatten(), dim=0))
            finite = bool(torch.isfinite(out).all())
            res.setdefault(str(M), {})[name] = {
                "us": round(us, 2), "cos_vs_first": round(cos, 5),
                "finite": finite, "heuristic": heur,
                "fallback_msg": kern[0][-160:] if kern else "",
            }
            print(f"M={M:5d} {name:10s} {us:9.2f} us cos={cos:.5f} finite={finite} heuristic={heur}", flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
