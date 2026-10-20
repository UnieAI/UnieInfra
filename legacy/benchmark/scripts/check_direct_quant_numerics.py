#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Accuracy of the direct-seam quantized GEMMs against the bf16 answer.

Answers one question and no other: for the projections our decode GEMV claims,
is UnieInfra's weight-only result *at least as close* to the unquantized bf16
reference as the engine's own symmetric-format result? E38 measured this
through the plugin; this reproduces it against `unieinfra-vllm`'s own kernels,
which is what the direct seam now falls back to.

Run inside the `unieinfra-vllm-build` container::

    python benchmark/scripts/check_direct_quant_numerics.py [--json OUT]

Reported per (projection, M): relative error ``||y - y_bf16|| / ||y_bf16||``
for

    ours fp8 W8A16      unieinfra_decode_gemv_fp8_triton, activation stays bf16
    vllm fp8 W8A8       QuantFP8(per-token) + cutlass_scaled_mm
    ours nvfp4 W4A16    unieinfra_decode_gemv_nvfp4_triton
    vllm nvfp4 W4A4     scaled_fp4_quant + cutlass_scaled_fp4_mm

plus ``ours vs its own dequant reference``, which is the kernel-correctness
check rather than the quantization-error one.
"""

from __future__ import annotations

import argparse
import json

import torch

# Qwen3-8B's dense projections. (name, N, K); gate_up (N=24576) is excluded
# because `n_le_16384` means the registry can never claim it.
SHAPES = [
    ("qkv", 6144, 4096),
    ("o", 4096, 4096),
    ("down", 4096, 12288),
]
BATCHES = (1, 2)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item() / b.float().norm().item()


def _fp8_arms(x, w_bf16, dev):
    from unieinfra_kernel.dispatch import linear_fp8
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    w_q, w_s = ops.scaled_fp8_quant(w_bf16, scale=None)
    # vLLM keeps the fp8 weight as [K, N].
    w_kn = w_q.t()

    quant = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN).to(dev)
    x_q, x_s = quant(x, None, None)
    vllm_out = ops.cutlass_scaled_mm(
        x_q, w_kn, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=None
    )
    ours_out = linear_fp8(x, w_kn, w_s, None)
    # Dequantized reference for the *kernel* (not the format).
    ref = torch.nn.functional.linear(x, (w_q.to(torch.float32) * w_s).to(x.dtype))
    return ours_out, vllm_out, ref


def _nvfp4_arms(x, w_bf16, dev):
    from unieinfra_kernel.dispatch import linear_nvfp4
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    fp4_max, fp8_max = 6.0, 448.0
    w_gs = (w_bf16.abs().max().float() / (fp4_max * fp8_max)).to(torch.float32)
    x_gs = (x.abs().max().float() / (fp4_max * fp8_max)).to(torch.float32)
    w_packed, w_scale_linear = ops.scaled_fp4_quant(
        w_bf16, 1.0 / w_gs, is_sf_swizzled_layout=False
    )
    w_scale_swz = swizzle_blockscale(w_scale_linear)
    alpha = (x_gs * w_gs).to(torch.float32)

    x_fp4, x_bs = ops.scaled_fp4_quant(
        x,
        1.0 / x_gs,
        is_sf_swizzled_layout=True,
        backend="cutlass",
        padded_n=x.shape[-1],
    )
    vllm_out = ops.cutlass_scaled_fp4_mm(
        x_fp4, w_packed, x_bs, w_scale_swz, alpha, x.dtype
    )
    ours_out = linear_nvfp4(
        x, w_packed, w_scale_linear, w_scale_swz, w_gs, 1.0 / x_gs, alpha, None
    )
    return ours_out, vllm_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    dev = torch.device(f"cuda:{args.device}")
    torch.manual_seed(0)
    rows = []
    print(
        f"{'shape':<22}{'M':>3}  {'ours fp8':>9}{'vllm fp8':>10}"
        f"{'ours nvfp4':>12}{'vllm nvfp4':>12}{'kern-vs-ref':>13}"
    )
    for name, n, k in SHAPES:
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
        for m in BATCHES:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            ref_bf16 = torch.nn.functional.linear(x, w)

            ours8, vllm8, deq_ref = _fp8_arms(x, w, dev)
            ours4, vllm4 = _nvfp4_arms(x, w, dev)

            row = {
                "shape": f"{name} ({n},{k})",
                "M": m,
                "ours_fp8_w8a16": _rel(ours8, ref_bf16),
                "vllm_fp8_w8a8": _rel(vllm8, ref_bf16),
                "ours_nvfp4_w4a16": _rel(ours4, ref_bf16),
                "vllm_nvfp4_w4a4": _rel(vllm4, ref_bf16),
                "ours_fp8_vs_dequant_ref": _rel(ours8, deq_ref),
            }
            rows.append(row)
            print(
                f"{row['shape']:<22}{m:>3}  {row['ours_fp8_w8a16']:>9.4f}"
                f"{row['vllm_fp8_w8a8']:>10.4f}{row['ours_nvfp4_w4a16']:>12.4f}"
                f"{row['vllm_nvfp4_w4a4']:>12.4f}"
                f"{row['ours_fp8_vs_dequant_ref']:>13.2e}"
            )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
