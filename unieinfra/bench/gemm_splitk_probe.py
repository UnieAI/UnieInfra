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

"""Is the narrow-N projection occupancy-bound, and does split-K fix it?

A serving profile put 9.2% of device time in two
`cutlass_80_wmma_tensorop_bf16_..._32x32_*` kernels, which cuBLAS selects for
the model's narrow projections. On the recorded shapes those run at about
43 TFLOP/s while the wide GEMMs on the same card reach 348 -- and a 32x32 tile
over a 1024x256 output is 256 tiles against 188 SMs, barely one wave. That is
an occupancy ceiling, not an arithmetic one, so the fix is more tiles: split the
reduction and give each K-slice its own program.

Same reasoning as the decode MoE kernel already in the tree, which measured
1.30x graphed on a shape with the same problem.

Reports TFLOP/s rather than GB/s: these shapes are not bandwidth-bound (the
widest reaches 830 GB/s against a 1792 GB/s peak), so bytes moved says nothing
about whether the kernel is good.
"""

import argparse
import sys
import time

import torch
import triton
import triton.language as tl

# (M, K, N) as recorded by the profiler, narrow-N first.
SHAPES = [
    (1024, 2048, 256),
    (8176, 2048, 256),
    (1024, 2048, 1024),
    (1024, 512, 2048),
    (1024, 4096, 2048),
]


@triton.jit
def _splitk_mm(
    a_ptr,
    b_ptr,
    part_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ps,
    stride_pm,
    stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """One K-slice of ``a @ b`` into a per-slice partial buffer."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_per = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per
    for k0 in range(0, k_per, BLOCK_K):
        k = k_start + k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        part_ptr
        + pid_k * stride_ps
        + offs_m[:, None] * stride_pm
        + offs_n[None, :] * stride_pn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _reduce(
    part_ptr,
    out_ptr,
    M,
    N,
    stride_ps,
    stride_pm,
    stride_pn,
    stride_om,
    stride_on,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum the K-slices. A buffer rather than atomics, so the result is
    bit-reproducible: an atomic order that depends on which program finishes
    first cannot validate anything downstream of it."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(
            part_ptr
            + s * stride_ps
            + offs_m[:, None] * stride_pm
            + offs_n[None, :] * stride_pn,
            mask=mask,
            other=0.0,
        )
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def splitk_mm(a, b, split_k, block_m=64, block_n=64, block_k=64):
    """``a @ b`` with the reduction split ``split_k`` ways."""
    m, k = a.shape
    _, n = b.shape
    part = torch.empty((split_k, m, n), dtype=torch.float32, device=a.device)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split_k)
    _splitk_mm[grid](
        a,
        b,
        part,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        part.stride(0),
        part.stride(1),
        part.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
    )
    rgrid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _reduce[rgrid](
        part,
        out,
        m,
        n,
        part.stride(0),
        part.stride(1),
        part.stride(2),
        out.stride(0),
        out.stride(1),
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out


def timed(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        s.append((time.perf_counter() - t0) * 1000)
    s.sort()
    return s[len(s) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[2, 4, 8])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"{torch.cuda.get_device_name(0)}  {sms} SMs")
    head = f"{'M':>6} {'K':>6} {'N':>6} {'tiles':>6} {'torch ms':>9} {'TF/s':>7}"
    for s in args.splits:
        head += f" {'k=' + str(s):>9}"
    print(head + f" {'best':>8}")
    print("-" * len(head + "     best"))

    for m, k, n in SHAPES:
        a = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
        b = torch.randn(k, n, dtype=torch.bfloat16, device=dev)
        ref = a @ b
        flop = 2 * m * k * n
        ms_torch = timed(lambda: torch.mm(a, b))
        tiles = ((m + 31) // 32) * ((n + 31) // 32)
        row = (
            f"{m:6d} {k:6d} {n:6d} {tiles:6d} {ms_torch:9.3f} "
            f"{flop / (ms_torch * 1e-3) / 1e12:7.1f}"
        )
        best = ms_torch
        for s in args.splits:
            try:
                got = splitk_mm(a, b, s)
                err = (got.float() - ref.float()).abs().max().item()
                scale = ref.float().abs().max().item() or 1.0
                ms = timed(lambda s=s: splitk_mm(a, b, s))
                ok = "" if err / scale < 0.02 else "!"
                row += f" {ms_torch / ms:8.2f}x{ok}"
                best = min(best, ms)
            except Exception:
                row += f" {'n/a':>9}"
        print(row + f" {ms_torch / best:7.2f}x")

    print("\n>1.00x means split-K beats cuBLAS on that shape. '!' = wrong answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
