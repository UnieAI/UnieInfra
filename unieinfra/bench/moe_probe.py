# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Isolate the 122B-A10B-FP8 MoE failure from the model.
#
# The engine dies inside autotune on the first forward, so the traceback names a
# GEMM shape but not which backend could serve it. This calls the fp8 MoE
# solutions directly at the exact shape the model produces -- 256 experts,
# hidden 3072, per-partition intermediate 512 under TP2, so the fused w13 GEMM
# is 1 x 1024 x 3072 -- with no weights to load and one card instead of two.
#
# Only three registered solutions declare weight_dtype fp8: flashinfer_trtllm
# (the one that fails), flashinfer_cutlass, and deep_gemm. deep_gemm sets
# supports_all_to_all_ep and tiles on 128-element blocks, so it is a DeepEP
# path; flashinfer_cutlass sets ispp_alignment 1 and does not require DeepEP,
# which makes it the only candidate for two cards in one box.
#
# Measured answer, 2026-08-14 on sm_120: all three fail at every M. cutlass
# raises NotImplementedError for Blackwell block scaling, trtllm's GEMM runner
# errors on a kernel whose name ends _sm100f, and deep_gemm wants a DeepEP
# group. Re-run this before assuming any of that still holds.
import sys
import traceback

import torch

HIDDEN = 3072
EXPERTS = 256
TOP_K = 8
INTER_PP = 512  # moe_intermediate_size 1024, halved by TP2
BLOCK = 128


class Weights(torch.nn.Module):
    """The attributes the fp8 MoE ops read off the layer, and nothing else."""

    def __init__(self):
        super().__init__()
        self.top_k = TOP_K
        f = torch.float8_e4m3fn
        self.w13_weight = torch.nn.Parameter(
            torch.randn(EXPERTS, 2 * INTER_PP, HIDDEN, device="cuda").to(f),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randn(EXPERTS, HIDDEN, INTER_PP, device="cuda").to(f),
            requires_grad=False,
        )
        self.w13_weight_scale_inv = torch.nn.Parameter(
            torch.rand(
                EXPERTS, 2 * INTER_PP // BLOCK, HIDDEN // BLOCK,
                dtype=torch.float32, device="cuda",
            ) + 0.1,
            requires_grad=False,
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            torch.rand(
                EXPERTS, HIDDEN // BLOCK, INTER_PP // BLOCK,
                dtype=torch.float32, device="cuda",
            ) + 0.1,
            requires_grad=False,
        )
        self.ep_size, self.ep_rank, self.tp_size, self.tp_rank = 1, 0, 1, 0


def probe(name, module_path, fn_name, tokens):
    """Run one solution at `tokens` rows and report what it does."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        fn = getattr(mod, fn_name)
    except Exception as exc:
        print(f"{name:22} M={tokens:<4} IMPORT FAILED: {type(exc).__name__}: {exc}")
        return False

    try:
        w = Weights()
        # the registered preprocessor swaps the gate/up halves in place
        for cand in dir(mod):
            if cand.endswith("_moe_weights"):
                getattr(mod, cand)({}, w)
                break
        x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
        logits = torch.randn(tokens, EXPERTS, dtype=torch.float32, device="cuda")
        tw, ti = torch.topk(torch.softmax(logits, dim=-1), k=TOP_K, dim=-1)
        tw = (tw / tw.sum(dim=-1, keepdim=True)).to(torch.float32)
        out = fn({}, x, w, logits, topk_weights=tw, topk_ids=ti.to(torch.int32))
        torch.cuda.synchronize()
        finite = torch.isfinite(out).all().item()
        print(f"{name:22} M={tokens:<4} OK  out={tuple(out.shape)} "
              f"dtype={out.dtype} finite={finite} absmax={out.abs().max().item():.4f}")
        return True
    except Exception as exc:
        first = " | ".join(str(exc).strip().splitlines())[:400]
        print(f"{name:22} M={tokens:<4} FAILED: {type(exc).__name__}: {first}")
        if "-v" in sys.argv:
            traceback.print_exc()
        return False
    finally:
        torch.cuda.empty_cache()


CASES = [
    ("flashinfer_cutlass", "tokenspeed_kernel.ops.moe.flashinfer.cutlass_fp8",
     "flashinfer_cutlass_fp8_moe_apply"),
    ("flashinfer_trtllm", "tokenspeed_kernel.ops.moe.flashinfer.trtllm_fp8",
     "flashinfer_trtllm_fp8_moe_apply"),
    ("deep_gemm_deepep", "tokenspeed_kernel.ops.moe.deep_gemm.deepep_fp8",
     "deep_gemm_deepep_fp8_moe_apply"),
    ("deep_gemm_contiguous", "tokenspeed_kernel.ops.moe.deep_gemm.contiguous_fp8",
     "deep_gemm_contiguous_fp8_moe_apply"),
]

print(f"shape: experts={EXPERTS} top_k={TOP_K} hidden={HIDDEN} "
      f"inter_per_partition={INTER_PP} (fused w13 N={2 * INTER_PP})")
print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}\n")
for name, path, fn in CASES:
    for m in (1, 8, 128):
        probe(name, path, fn, m)
    print()
