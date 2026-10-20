"""E37 numerics check: UnieInfra's decode GEMV vs cuBLAS on the shapes it claims.

Run after an A/B that shows the two arms producing *coherent but different*
greedy text. The question that answers is whether the divergence is a bug or a
bf16 tie-break: if our kernel is no further from an fp32 reference than cuBLAS
is, then both arms are equally "correct" and the divergence is a near-tie in the
logits flipping, not an error.
"""

import torch
from unieinfra_kernel.dispatch import linear_op

SHAPES = [
    (1, 6144, 4096),  # qkv_proj, batch 1
    (2, 6144, 4096),  # qkv_proj, batch 2
    (1, 4096, 4096),  # o_proj
    (2, 4096, 4096),
    (1, 4096, 12288),  # down_proj
    (2, 4096, 12288),
]


def main() -> None:
    torch.manual_seed(0)
    hdr = ("shape (M,N,K)", "max_abs", "max_rel", "mean_rel", "bit-exact")
    print(f"{hdr[0]:>22} {hdr[1]:>11} {hdr[2]:>11} {hdr[3]:>11} {hdr[4]:>10}")
    for m, n, k in SHAPES:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        ref = torch.nn.functional.linear(x, w)
        got = linear_op(x, w, None)
        d = (got.float() - ref.float()).abs()
        rel = d / ref.float().abs().clamp_min(1e-6)
        print(
            f"{str((m, n, k)):>22} {d.max():11.3e} {rel.max():11.3e} "
            f"{rel.mean():11.3e} {str(torch.equal(got, ref)):>10}"
        )

    print("\nAccuracy against an fp32 reference (lower is better):")
    for m, n, k in SHAPES:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        exact = torch.nn.functional.linear(x.float(), w.float())
        e_ref = (torch.nn.functional.linear(x, w).float() - exact).abs().mean()
        e_got = (linear_op(x, w, None).float() - exact).abs().mean()
        verdict = "unieinfra closer" if e_got < e_ref else "cuBLAS closer"
        print(
            f"{str((m, n, k)):>22}  cuBLAS={e_ref:.4e}  unieinfra={e_got:.4e}  "
            f"ratio={e_got / e_ref:.3f}  ({verdict})"
        )


if __name__ == "__main__":
    main()
