"""Compare per-layer hidden states against HF and name the first layer that
disagrees.

Pairs with `TOKENSPEED_DUMP_HIDDEN` in models/gemma.py and
`gemma_hf_reference.py`. Both files hold float32 CPU tensors of
[tokens, hidden]; the metric is relative L2 per layer, and a layer where it
jumps from bf16 drift to O(1) is the one that broke.

This is how the per-layer head_dim defect was localized: layers 0-4 at 4.4e-03,
layer5 at 3.16e-01, and every later jump on 11, 17, 23 ... 59 -- exactly
gemma-4's ten full_attention layers. After the fix, layer5 is 5.49e-03 and
final_norm 1.62e-02.

Read the LAST layer row with care: HF's `hidden_states[-1]` has already been
through `model.norm` while the engine's `layer59` has not, so that one row is
expected to disagree and `final_norm` is the row that matters. The two norms
(23.96 against 723.6) say which is which.
"""
import argparse, torch

def rel(a, b):
    n = (a - b).norm().item()
    d = max(b.norm().item(), 1e-12)
    return n / d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="/workspace/roy/.hidden-ours.pt")
    ap.add_argument("--hf", default="/workspace/roy/.hidden-hf.pt")
    args = ap.parse_args()
    a = torch.load(args.ours, map_location="cpu")
    b = torch.load(args.hf, map_location="cpu")

    ia = a["input_ids"].flatten().tolist()
    ib = b["input_ids"].flatten().tolist()
    print(f"input_ids ours={ia} hf={ib}"
          f"{'' if ia == ib else '   <-- DIFFERENT INPUT, nothing below is comparable'}")

    keys = ["embed"] + [f"layer{i}" for i in range(200)] + ["final_norm"]
    prev = None
    for k in keys:
        if k not in a or k not in b:
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"{k:12s} shape {tuple(x.shape)} vs {tuple(y.shape)}  <-- SHAPE")
            continue
        r = rel(x, y)
        flag = ""
        if prev is not None and r > 10 * max(prev, 1e-3) and r > 0.05:
            flag = "  <-- first big jump"
        print(f"{k:12s} rel_l2={r:10.4e}   ours|.|={x.norm():10.3e} "
              f"hf|.|={y.norm():10.3e}{flag}")
        prev = r

if __name__ == "__main__":
    main()
