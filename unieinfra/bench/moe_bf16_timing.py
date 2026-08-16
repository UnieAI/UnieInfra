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

"""Time the bf16 MoE layer against the kernel it wraps.

Both engines select `FlashInfer CUTLASS Unquantized` for Qwen3.6-35B-A3B, which
is why the scheduler attribution treated this layer as neutral. That is an
inference, not a measurement: the same kernel can be fed worse shapes, wrapped
in avoidable copies, or autotuned differently. This measures the wrapper
against a direct call to the kernel underneath it, so the overhead we add is
separated from the kernel's own cost.

Reported per call:
  ours   -- the registered op, routing included
  raw    -- cutlass_fused_moe called directly with precomputed routing
  delta  -- what the wrapper costs on top

Shapes come from the model: hidden 2048, 256 experts, top-k 8, per-expert
intermediate 512. Token counts span decode (1), a small mixed batch, and the
prefill chunk width the scheduler actually uses.
"""

import argparse
import sys
import time

import torch

HIDDEN = 2048
EXPERTS = 256
TOP_K = 8
INTER = 512


class Weights(torch.nn.Module):
    """The attributes the unquantized MoE op reads off a layer."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.top_k = TOP_K
        self.w13_weight = torch.nn.Parameter(
            torch.randn(EXPERTS, 2 * INTER, HIDDEN, dtype=dtype, device="cuda") * 0.02,
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randn(EXPERTS, HIDDEN, INTER, dtype=dtype, device="cuda") * 0.02,
            requires_grad=False,
        )
        self.ep_size, self.ep_rank, self.tp_size, self.tp_rank = 1, 0, 1, 0


def timed(fn, warmup=5, iters=20):
    """Median wall time per call, in milliseconds, synchronised each iteration.

    Median rather than mean: the first autotuned launch after a shape change is
    an outlier that a mean would smear across every number.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512, 8192])
    args = ap.parse_args()

    from tokenspeed_kernel.ops.moe.flashinfer import cutlass_unquant

    # The split-K decode candidate, if it is present in the tree. It carries no
    # register_kernel, so it can only be reached by importing it directly.
    try:
        from tokenspeed_kernel.ops.moe.triton import bf16_decode
    except Exception:
        bf16_decode = None

    fn = getattr(cutlass_unquant, "flashinfer_cutlass_unquant_moe_apply", None)
    if fn is None:
        print("unquantized cutlass MoE is not registered on this platform", file=sys.stderr)
        return 1
    raw = getattr(cutlass_unquant, "cutlass_fused_moe", None)

    w = Weights()
    for cand in dir(cutlass_unquant):
        if cand.endswith("_moe_weights"):
            getattr(cutlass_unquant, cand)({}, w)
            break

    print(f"hidden={HIDDEN} experts={EXPERTS} top_k={TOP_K} inter={INTER}  "
          f"device={torch.cuda.get_device_name(0)}")
    print(f"{'tokens':>7} {'ours ms':>9} {'raw ms':>9} {'splitK ms':>10} {'splitK vs ours':>15}")
    print("-" * 56)

    for tokens in args.tokens:
        x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
        logits = torch.randn(tokens, EXPERTS, dtype=torch.float32, device="cuda")
        scores = torch.softmax(logits, dim=-1)
        tw, ti = torch.topk(scores, k=TOP_K, dim=-1)
        tw = (tw / tw.sum(dim=-1, keepdim=True)).to(torch.float32)
        ti = ti.to(torch.int32)

        ours = timed(lambda: fn({}, x, w, logits, topk_weights=tw, topk_ids=ti))

        raw_ms = float("nan")
        if raw is not None:
            out = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")

            def call_raw():
                raw(
                    output=out,
                    input=x,
                    token_selected_experts=ti,
                    token_final_scales=tw,
                    fc1_expert_weights=w.w13_weight,
                    fc2_expert_weights=w.w2_weight,
                    output_dtype=x.dtype,
                    quant_scales=[],
                )

            try:
                raw_ms = timed(call_raw)
            except Exception as exc:
                print(f"{tokens:7d} {ours:9.3f}  raw call unavailable: "
                      f"{type(exc).__name__}: {str(exc).splitlines()[0][:60]}")
                continue

        split_ms = float("nan")
        if bf16_decode is not None:
            try:
                split_ms = timed(lambda: bf16_decode.decode_moe_body(x, w, tw, ti))
            except Exception as exc:
                split_ms = float("nan")
                note = f"{type(exc).__name__}: {str(exc).splitlines()[0][:50]}"
                print(f"{tokens:7d} {ours:9.3f} {raw_ms:9.3f}   split-K failed: {note}")
                continue
        ratio = ours / split_ms if split_ms and split_ms == split_ms else float("nan")
        print(f"{tokens:7d} {ours:9.3f} {raw_ms:9.3f} {split_ms:10.3f} {ratio:14.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
