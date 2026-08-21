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

"""Does the weight layout change which GEMM kernel cuBLAS picks?

A serving profile of Qwen3.6-35B-A3B on RAG C=64 put 9.2% of all device time in
two `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_*_tn_align8` kernels.
Two things stand out: `wmma` is the pre-`mma.sync` instruction path, and `tn`
says cuBLAS saw a transposed B -- which is exactly what `x @ weight.t()` hands
it, the form every linear layer in the runtime uses.

So before writing a kernel, ask whether the layout alone moves it: the same
math with the weight already stored transposed is an `nn` GEMM, and cuBLAS
selects on layout. If it does, this is a load-time change rather than a kernel.

Shapes are the ones the profile recorded, not invented: M is 1024 (a decode
step's batch) or 8176 (a prefill chunk), against the model's four projection
widths.

Reports achieved GB/s next to every timing, because a number above this card's
1792 GB/s HBM peak means the working set sat in L2 and the comparison is
measuring the cache rather than the kernel.
"""

import argparse
import sys

import torch

# (M, K, N) as recorded by the profiler's cpu_op Input Dims.
SHAPES = [
    (1024, 4096, 2048),
    (1024, 2048, 256),
    (1024, 2048, 1024),
    (1024, 512, 2048),
    (8176, 4096, 2048),
    (8176, 2048, 256),
]

HBM_PEAK_GBPS = 1792.0


def timed(fn, warmup=10, iters=50):
    """Median wall time per call in ms, synchronised each iteration.

    Median rather than mean: the first launch after a shape change is an
    outlier a mean would smear across every number.

    Args:
        fn: Callable invoked with no arguments.
        warmup: Untimed iterations before measuring.
        iters: Timed iterations.

    Returns:
        Median milliseconds per call.
    """
    import time

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


def gbps(m, k, n, ms):
    """Bytes moved over the measured time, in GB/s, assuming bf16 operands."""
    move = (m * k + k * n + m * n) * 2
    return move / (ms * 1e-3) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rotate",
        type=int,
        default=4,
        help="distinct weight copies to cycle, to defeat L2 reuse",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)}  HBM peak {HBM_PEAK_GBPS:.0f} GB/s")
    print(
        f"{'M':>6} {'K':>6} {'N':>6}  {'tn ms':>8} {'tn GB/s':>9} "
        f"{'nn ms':>8} {'nn GB/s':>9}  {'nn/tn':>7}"
    )
    print("-" * 72)

    for m, k, n in SHAPES:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
        # Rotate weights so a small shape cannot answer from L2; this card has
        # 128 MB of it, which is enough to hold a whole projection.
        w_t = [
            torch.randn(n, k, dtype=torch.bfloat16, device=dev)
            for _ in range(args.rotate)
        ]
        w_n = [w.t().contiguous() for w in w_t]
        out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)

        step = {"i": 0}

        def run_tn():
            i = step["i"] % args.rotate
            step["i"] += 1
            torch.mm(x, w_t[i].t(), out=out)

        def run_nn():
            i = step["i"] % args.rotate
            step["i"] += 1
            torch.mm(x, w_n[i], out=out)

        ms_tn = timed(run_tn)
        ms_nn = timed(run_nn)
        print(
            f"{m:6d} {k:6d} {n:6d}  {ms_tn:8.3f} {gbps(m,k,n,ms_tn):9.1f} "
            f"{ms_nn:8.3f} {gbps(m,k,n,ms_nn):9.1f}  {ms_tn/ms_nn:6.2f}x"
        )

    print("\nnn/tn > 1 means storing the weight pre-transposed is faster.")
    print("Any GB/s above the HBM peak means the operands were served from L2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
