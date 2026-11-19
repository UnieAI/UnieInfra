"""Synthetic repro: AITER-style MXFP4 MoE preprocessing fragmentation on one GPU.

usage: python frag_repro.py {base,fix} LAYERS
Shapes: DeepSeek-V4.1-Flash TP4 (E=384, H=5120, I/rank=576 -> 640).
"""

import sys

import torch

from tokenspeed.runtime.layers.moe.expert import _unsplit_large_blocks

mode, layers = sys.argv[1], int(sys.argv[2])
E, H, I, IP = 384, 5120, 576, 640
dev = torch.device("cuda", 0)


def shuffle_w(x, gate_up):  # aiter shuffle_weight(is_guinterleave=True), uint8
    e, n, kpk = x.shape
    if gate_up:
        n //= 2
    nl, kp = 16, 16
    kl = 64 // nl
    n0, k0 = n // nl, kpk // (kl * kp)
    if gate_up:
        y = x.view(e, 2, n0, nl, k0, kl, kp).permute(0, 2, 1, 4, 5, 3, 6).contiguous()
    else:
        y = x.view(e, n0, nl, k0, kl, kp).permute(0, 1, 3, 4, 2, 5).contiguous()
    return y.view(*x.shape).contiguous()


def shuffle_s(src, experts, gate_up):  # aiter shuffle_scale(is_guinterleave=True)
    rows, k = src.shape
    if k % 8:
        kp = (k + 7) // 8 * 8
        p = torch.full((rows, kp), 0x7F, dtype=src.dtype, device=src.device)
        p[:, :k] = src
        src, k = p, kp
    n = rows // experts
    k1, n1 = k // 2 // 4, n // 16 // 2
    if gate_up:
        y = src.view(experts, 2, n1, 16, k1, 2, 4).permute(0, 2, 4, 6, 3, 5, 1)
    else:
        y = src.view(experts, n1, 2, 16, k1, 2, 4).permute(0, 1, 4, 6, 3, 5, 2)
    return y.contiguous().view(*src.shape).contiguous()


def preprocess(w):
    pad = IP - I
    wt, sc = w.w13_weight.data, w.w13_weight_scale.data
    z = wt.new_zeros((E, pad, wt.shape[-1]))
    g, u = wt.split(I, 1)
    wt = torch.cat((g, z, u, z), 1).contiguous()
    o = sc.new_full((E, pad, sc.shape[-1]), 0x7F)
    gs, us = sc.split(I, 1)
    sc = torch.cat((gs, o, us, o), 1).contiguous().reshape(-1, sc.shape[-1])
    w13, s13 = shuffle_w(wt, True), shuffle_s(sc, E, True)
    del wt, sc, z, o
    wt, sc = w.w2_weight.data, w.w2_weight_scale.data
    wt = torch.cat((wt, wt.new_zeros((*wt.shape[:-1], pad // 2))), -1).contiguous()
    sc = torch.cat((sc, sc.new_full((*sc.shape[:-1], pad // 32), 0x7F)), -1)
    sc = sc.contiguous().reshape(-1, sc.shape[-1])
    w2, s2 = shuffle_w(wt, False), shuffle_s(sc, E, False)
    del wt, sc
    for n, t in (("w13_weight", w13), ("w2_weight", w2), ("w13_weight_scale", s13), ("w2_weight_scale", s2)):
        setattr(w, n, torch.nn.Parameter(t, requires_grad=False))


mods = []
for _ in range(layers):
    m = torch.nn.Module()
    m.w13_weight = torch.nn.Parameter(torch.zeros((E, 2 * I, H // 2), dtype=torch.uint8, device=dev), requires_grad=False)
    m.w13_weight_scale = torch.nn.Parameter(torch.zeros((E, 2 * I, H // 32), dtype=torch.uint8, device=dev), requires_grad=False)
    m.w2_weight = torch.nn.Parameter(torch.zeros((E, H, I // 2), dtype=torch.uint8, device=dev), requires_grad=False)
    m.w2_weight_scale = torch.nn.Parameter(torch.zeros((E, H, I // 32), dtype=torch.uint8, device=dev), requires_grad=False)
    mods.append(m)
for m in mods:
    if mode == "fix":
        with _unsplit_large_blocks(m):
            preprocess(m)
    else:
        preprocess(m)
torch.cuda.synchronize()
torch.cuda.empty_cache()
GB = 1 << 30
a, r = torch.cuda.memory_allocated() / GB, torch.cuda.memory_reserved() / GB
print(f"{mode} layers={layers} allocated={a:.2f} reserved={r:.2f} stranded={r - a:.2f} GB "
      f"({(r - a) / layers * 1024:.0f} MB/layer)")
