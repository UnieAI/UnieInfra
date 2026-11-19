"""Torch reference of block verification (Sun et al. 2024, as in vLLM's rejection_sampler_utils) + MC losslessness check."""
import torch
torch.manual_seed(0)
V, G = 6, 4  # vocab, draft length

def make_models(noise):
    # tabular context models over the last 2 tokens (+ position-free); target P, draft Q = perturbed P
    P = torch.softmax(torch.randn(V, V, V) * 1.5, -1)
    Q = torch.softmax(torch.log(P) + torch.randn(V, V, V) * noise, -1)
    return P, Q

def dist(M, ctx):  # ctx: list of tokens (len>=2)
    return M[ctx[-2], ctx[-1]]

def draft(Q, ctx, g=G):
    toks = []; c = list(ctx)
    for _ in range(g):
        t = torch.multinomial(dist(Q, c), 1).item(); toks.append(t); c.append(t)
    return toks

def verify_block(P, Q, ctx, toks):
    c = list(ctx); p = 1.0; ps = []; res = []
    # per-position joint ratio and residual mass
    for i, t in enumerate(toks):
        p = min(1.0, p * dist(P, c)[t].item() / dist(Q, c)[t].item()); c.append(t); ps.append(p)
    acc = 0
    c = list(ctx)
    for i, t in enumerate(toks): c.append(t)
    for i in range(len(toks)):
        cc = list(ctx) + toks[: i + 1]
        if i < len(toks) - 1:
            rm = torch.clamp(ps[i] * dist(P, cc) - dist(Q, cc), min=0).sum().item()
            den = rm + 1 - ps[i]; h = rm / den if den > 0 else 1.0
        else: h = ps[i]
        if torch.rand(()).item() <= h: acc = i + 1
    cc = list(ctx) + toks[:acc]
    if acc == len(toks): r = dist(P, cc)
    else:
        pa = 1.0 if acc == 0 else ps[acc - 1]
        r = torch.clamp(pa * dist(P, cc) - dist(Q, cc), min=0)
    r = r / r.sum()
    return toks[:acc] + [torch.multinomial(r, 1).item()], acc

def verify_token(P, Q, ctx, toks):
    c = list(ctx); out = []
    for t in toks:
        pt, qt = dist(P, c)[t].item(), dist(Q, c)[t].item()
        if torch.rand(()).item() < min(1.0, pt / qt): out.append(t); c.append(t)
        else:
            r = torch.clamp(dist(P, c) - dist(Q, c), min=0); r = r / r.sum()
            out.append(torch.multinomial(r, 1).item()); return out, len(out) - 1
    out.append(torch.multinomial(dist(P, c), 1).item()); return out, len(toks)

if __name__ == "__main__":
    for noise in (0.3, 0.8):
        P, Q = make_models(noise); ctx = [0, 1]; N = 30000
        for name, fn in (("block", verify_block), ("token", verify_token)):
            cnt = torch.zeros(V, V); al = 0
            for _ in range(N):
                out, a = fn(P, Q, ctx, draft(Q, ctx)); al += a
                while len(out) < 2:
                    o2, _ = fn(P, Q, ctx + out, draft(Q, ctx + out)); out = out + o2
                cnt[out[0], out[1]] += 1
            exact = dist(P, ctx)[:, None] * P[ctx[-1], :, :]
            tv = 0.5 * (cnt / N - exact).abs().sum().item()
            print(f"noise {noise} {name}: mean accepted {al/N:.3f}  TV(first-two-tokens vs target) {tv:.4f}")
