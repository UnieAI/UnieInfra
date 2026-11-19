import importlib.util, torch
spec = importlib.util.spec_from_file_location("bv", "/w/block_verify.py"); import sys; bv = importlib.util.module_from_spec(spec); sys.modules["bv"] = bv; spec.loader.exec_module(bv)
torch.manual_seed(1); V, G, N = 6, 4, 200000
P = torch.softmax(torch.randn(V, V, V) * 1.5, -1)
for noise in (0.3, 0.8):
    Q = torch.softmax(torch.log(P) + torch.randn(V, V, V) * noise, -1)
    ctx = torch.tensor([0, 1])
    toks = [torch.full((N,), 1)]; prev2 = [torch.full((N,), 0)]
    # sample drafts autoregressively from Q, record Q logits per step
    c2, c1 = torch.full((N,), 0), torch.full((N,), 1); cand = [c1]; qlog = []
    for s in range(G):
        ql = torch.log(Q[c2, c1]); qlog.append(ql)
        t = torch.multinomial(Q[c2, c1], 1)[:, 0]; cand.append(t); c2, c1 = c1, t
    cand = torch.stack(cand, 1)  # [N, G+1]; col0 = anchor (token 1 after ctx [0,1])
    # target logits for each of G+1 rows: context (prev2, cand[:,j])
    prevs = torch.cat([torch.zeros(N, 1, dtype=torch.long), cand[:, :-1]], 1)
    tl_ = torch.log(P[prevs, cand])  # [N, G+1, V]
    qlf = torch.stack(qlog, 1)
    predict = torch.zeros(N * (G + 1), dtype=torch.int32); ai = torch.full((N, G + 1), -1, dtype=torch.int32); al = torch.zeros(N, dtype=torch.int32)
    bv.block_verify_chain(tl_.reshape(N * (G + 1), V), qlf, cand.int(), torch.ones(N), predict, ai, al)
    out = predict.view(N, G + 1)
    # first two generated tokens: out[:,0], and out[:,1] if al>=1 else resample (skip): test token0 marginal and pair when al>=1 via conditional check
    ex0 = P[0, 1]; emp0 = torch.bincount(out[:, 0].long(), minlength=V) / N
    print(f"noise {noise}: mean accepted {al.float().mean():.3f}  TV(token0) {0.5*(emp0-ex0).abs().sum():.4f}")
    # pair test on rows with al>=1: joint of (t0,t1) must equal P(t0)P(t1|t0) restricted to rows where len>=2
    m = al >= 1
    cnt = torch.zeros(V, V); cnt.index_put_((out[m, 0].long(), out[m, 1].long()), torch.ones(int(m.sum())), accumulate=True)
    print("  accepted rows share", m.float().mean().item())
