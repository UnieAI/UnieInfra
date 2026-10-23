"""Gemma-4's RoPE, applied, against HF's own math.

`build_rope_inv_freq` was checked by reading and reproduces HF's
`_compute_proportional_rope_parameters` -- including the part that makes gemma-4
unusual: the global layers return a FULL head_dim//2 = 256 frequencies of which
only the first 64 are non-zero, so RoPE is a full-width rotation whose tail is
the identity, NOT a shortened `rotary_dim`. Those two are different
transformations (neox pairs channel i with i + head_dim//2; a shortened
rotary_dim pairs i with i + rotary_dim//2 and rotates a different set), and
only one of them is gemma-4.

Reading says the frequencies are right. This checks that APPLYING them is:
head_dim 512 with a zero tail is the shape a fused rope kernel is most likely
to get wrong, and it is the last piece of gemma-4's attention path that has not
been compared against a reference.

Reference is HF's own `apply_rotary_pos_emb` math, written out here so the
comparison does not depend on importing a modeling file:

    cos/sin = cat(freqs, freqs) over head_dim
    out     = x * cos + rotate_half(x) * sin
    rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2])

    python3 unieinfer/bench/kernel_gemma_rope.py
"""

import torch
from transformers import AutoConfig

from tokenspeed.runtime.models.gemma import (
    Gemma4RotaryEmbedding,
    build_rope_inv_freq,
    layer_head_dim,
)

MODEL = "/workspace/models/google/gemma-4-31B-it"


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def ref_apply(x, positions, inv_freq):
    """x [T, H, D]; HF's neox rotation with cos/sin = cat(freqs, freqs)."""
    freqs = positions.float()[:, None] * inv_freq[None, :].float()  # [T, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]
    cos = emb.cos()[:, None, :]
    sin = emb.sin()[:, None, :]
    xf = x.float()
    return xf * cos + rotate_half(xf) * sin


def main():
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL)
    tcfg = cfg.text_config
    dev = "cuda"

    for layer_type in ("sliding_attention", "full_attention"):
        d = layer_head_dim(tcfg, layer_type)
        inv = build_rope_inv_freq(layer_type, tcfg)
        nz = int((inv != 0).sum())
        print(
            f"\n## {layer_type}: head_dim {d}, {inv.numel()} angles, "
            f"{nz} non-zero ({'partial' if nz < inv.numel() else 'full'} rotary)"
        )

        rope = Gemma4RotaryEmbedding(tcfg, layer_type, 4096, torch.bfloat16).to(dev)
        nq = tcfg.num_attention_heads
        nkv = (
            tcfg.num_global_key_value_heads
            if layer_type == "full_attention" and tcfg.attention_k_eq_v
            else tcfg.num_key_value_heads
        )
        for T in (1, 25, 1024):
            pos = torch.arange(T, dtype=torch.int64, device=dev)
            q = torch.randn(T, nq * d, dtype=torch.bfloat16, device=dev)
            k = torch.randn(T, nkv * d, dtype=torch.bfloat16, device=dev)
            qo, ko = rope(pos, q.clone(), k.clone())

            rq = ref_apply(q.view(T, nq, d), pos, inv.to(dev)).view(T, nq * d)
            rk = ref_apply(k.view(T, nkv, d), pos, inv.to(dev)).view(T, nkv * d)
            dq = (qo.float() - rq).abs().max().item() / rq.abs().max().item()
            dk = (ko.float() - rk).abs().max().item() / rk.abs().max().item()
            print(
                f"  T={T:5d}  q rel={dq:9.2e}  k rel={dk:9.2e}  "
                f"{'OK' if max(dq, dk) < 5e-2 else '<-- WRONG'}"
            )


if __name__ == "__main__":
    main()
