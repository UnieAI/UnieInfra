"""Split a suspect attention layer into "the cache" and "what we hand it".

This is the regression test for the per-layer head_dim fix (commit 40f2478a).
Run it whenever the KV pool's row view or its per-layer geometry changes.

Feed it the one-shot dump from models/gemma.py:

    TOKENSPEED_DUMP_ATTN=/path/attn.pt \
    TOKENSPEED_DUMP_ATTN_LAYER=5 \
    TOKENSPEED_DUMP_HIDDEN_TOKENS=5,6 \
      python3 -m tokenspeed.cli serve <gemma-4> ...
    # then send a 5-token prompt, then:
    python3 unieinfra/bench/gemma_attn_layer_check.py --dump /path/attn.pt

Two checks on different halves of the layer:

  1. attn_out against scaled_dot_product_attention on the SAME q, k, v the
     engine passed in, at gemma-4's scale of 1.0. The kernels already agree
     with SDPA on synthetic tensors of these geometries
     (kernel_gemma_correctness.py), so a mismatch here is the KV cache round
     trip and nothing else.
  2. q, k, v against a recomputation from `hidden_in` with the checkpoint's own
     weights, which separates the projections and the qk-norms from attention.

Always run it on a sliding layer too (`TOKENSPEED_DUMP_ATTN_LAYER=0`). Both
layers pass the same code; only the geometry differs, so a sliding layer that
also fails means the harness is wrong, not the engine. Measured on
gemma-4-31B-it across the fix:

    layer 0 (32q x 16kv x 256)   1.42e-03 before and after
    layer 5 (32q x  4kv x 512)   1.13 before,  1.48e-03 after
"""
import argparse, json, os, struct

import torch
import torch.nn.functional as F


def load_layer_weights(root, layer_id, names):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for n in names:
        key = f"model.language_model.layers.{layer_id}.self_attn.{n}"
        if key not in idx:
            continue
        path = os.path.join(root, idx[key])
        with open(path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
            meta = hdr[key]
            start, end = meta["data_offsets"]
            fh.seek(8 + hlen + start)
            raw = fh.read(end - start)
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(meta["shape"])
        out[n] = t.float()
    return out


def rms(x, w=None, eps=1e-6):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y * w.float() if w is not None else y


def rel(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/workspace/roy/.attn-ours.pt")
    ap.add_argument("--model", default="/workspace/models/google/gemma-4-31B-it")
    args = ap.parse_args()
    d = torch.load(args.dump, map_location="cpu")
    lid, hd = int(d["layer_id"]), int(d["head_dim"])
    nq, nkv = int(d["num_heads"]), int(d["num_kv_heads"])
    T = d["hidden_in"].shape[0]
    print(f"layer {lid} ({d['layer_type']}), {nq}q x {nkv}kv x {hd}, "
          f"k_eq_v={d['k_eq_v']}, {T} tokens")

    # --- 1. does the attention output match SDPA on the dumped q/k/v? ---
    q = d["q"].view(T, nq, hd)
    k = d["k"].view(T, nkv, hd)
    v = d["v"].view(T, nkv, hd)
    rep = nq // nkv
    qq = q.permute(1, 0, 2).unsqueeze(0)
    kk = k.repeat_interleave(rep, 1).permute(1, 0, 2).unsqueeze(0)
    vv = v.repeat_interleave(rep, 1).permute(1, 0, 2).unsqueeze(0)
    ref = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True, scale=1.0)
    ref = ref.squeeze(0).permute(1, 0, 2).reshape(T, nq * hd)
    r1 = rel(d["attn_out"], ref)
    print(f"  attn_out vs SDPA(dumped q,k,v)   rel={r1:9.3e}  "
          f"{'OK -- the kernel saw these tensors' if r1 < 5e-2 else '<-- WRONG: the cache round trip'}")

    # --- 2. do q/k/v match a recomputation from hidden_in? ---
    w = load_layer_weights(args.model, lid,
                           ["q_proj.weight", "k_proj.weight", "v_proj.weight",
                            "q_norm.weight", "k_norm.weight"])
    h = d["hidden_in"]
    q_raw = h @ w["q_proj.weight"].T
    k_raw = h @ w["k_proj.weight"].T
    print(f"  k_raw vs h @ k_proj.T            rel={rel(d['k_raw'], k_raw):9.3e}")
    qn = rms(q_raw.reshape(-1, hd), w["q_norm.weight"]).reshape(T, nq * hd)
    kn = rms(k_raw.reshape(-1, hd), w["k_norm.weight"]).reshape(T, nkv * hd)
    print(f"  (q,k after norm are compared BEFORE rope, so a nonzero rel here "
          f"is rope, not the norms)")
    print(f"  q vs qnorm(q_proj)  [rope not applied to ref] rel={rel(d['q'], qn):9.3e}")
    print(f"  k vs knorm(k_proj)  [rope not applied to ref] rel={rel(d['k'], kn):9.3e}")
    if d["k_eq_v"]:
        vref = rms(k_raw.reshape(-1, hd)).reshape(T, nkv * hd)
        src = "v_norm(k_proj raw)"
    else:
        v_raw = h @ w["v_proj.weight"].T
        vref = rms(v_raw.reshape(-1, hd)).reshape(T, nkv * hd)
        src = "v_norm(v_proj)"
    r2 = rel(d["v"], vref)
    print(f"  v vs {src:22s}      rel={r2:9.3e}  "
          f"{'OK' if r2 < 5e-2 else '<-- WRONG'}")


if __name__ == "__main__":
    main()
