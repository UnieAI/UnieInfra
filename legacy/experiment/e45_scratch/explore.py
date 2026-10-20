#!/usr/bin/env python3
"""E45 exploration: mid-M decode GEMM design space vs cuBLASLt.

Not a deliverable. Survey only: measure the incumbent, then sweep a flexible
Triton NT-GEMM (A [M,K] row-major, B [N,K] row-major = nn.Linear layout) over
tile / split-K / persistent choices on the four Qwen3-8B projections.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

SHAPES: dict[str, tuple[int, int]] = {
    "gate_up_proj": (24576, 4096),
    "down_proj": (4096, 12288),
    "qkv_proj": (6144, 4096),
    "o_proj": (4096, 4096),
}
BATCHES = (32, 64, 128, 256)

L2_ROTATION_BYTES = 640 * 1024 * 1024


def rotation_count(nbytes: int) -> int:
    return max(2, min(16, -(-L2_ROTATION_BYTES // max(nbytes, 1))))


def _bytes_moved(M, N, K, itemsize):
    return (N * K + M * K + M * N) * itemsize


def measure_peak_bandwidth(device):
    n = 1024 * 1024 * 1024 // 2
    src = torch.empty(n, dtype=torch.bfloat16, device=device)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(20):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    secs = start.elapsed_time(end) / 1e3 / 20
    peak = 2 * src.numel() * src.element_size() / secs
    del src, dst
    torch.cuda.empty_cache()
    return peak


def time_us_graph(ops, reps: int = 7) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for fn in ops:
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in ops:
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(5):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / (5 * len(ops)))
    del graph
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# Flexible NT GEMM
# ---------------------------------------------------------------------------


@triton.jit
def _nt_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    NUM_TILES: tl.constexpr,
    NUM_PID_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STREAM_B: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    """C[M,N] = A[M,K] @ B[N,K].T (+bias). Persistent when NUM_PROGRAMS<NUM_TILES."""
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m: tl.constexpr = NUM_TILES // NUM_PID_N

    for tile_id in tl.range(pid, NUM_TILES, NUM_PROGRAMS):
        # Group-M swizzle: consecutive tiles share B columns -> L2 reuse.
        if GROUP_M == 1:
            pid_m = tile_id // NUM_PID_N
            pid_n = tile_id % NUM_PID_N
        else:
            num_in_group = GROUP_M * NUM_PID_N
            group_id = tile_id // num_in_group
            first_m = group_id * GROUP_M
            group_m = min(num_pid_m - first_m, GROUP_M)
            pid_m = first_m + ((tile_id % num_in_group) % group_m)
            pid_n = (tile_id % num_in_group) // group_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        m_mask = offs_m < M
        n_mask = offs_n < N
        # Clamp so out-of-range rows/cols read valid memory (masking the K loop
        # every iteration costs more than a wasted lane).
        offs_am = tl.where(m_mask, offs_m, 0)
        offs_bn = tl.where(n_mask, offs_n, 0)

        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :]
        b_ptrs = b_ptr + offs_bn[:, None] * stride_bn + offs_k[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_start = pid_k * BLOCK_K
        k_step = BLOCK_K * SPLIT_K
        for k0 in tl.range(k_start, K, k_step):
            if EVEN_K:
                a = tl.load(a_ptrs + k0)
                b = tl.load(
                    b_ptrs + k0,
                    eviction_policy="evict_first" if STREAM_B else "",
                )
            else:
                k_ok = (k0 + offs_k) < K
                a = tl.load(a_ptrs + k0, mask=k_ok[None, :], other=0.0)
                b = tl.load(
                    b_ptrs + k0,
                    mask=k_ok[None, :],
                    other=0.0,
                    eviction_policy="evict_first" if STREAM_B else "",
                )
            acc = tl.dot(a, tl.trans(b), acc)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        store_mask = m_mask[:, None] & n_mask[None, :]
        if SPLIT_K == 1:
            if HAS_BIAS:
                bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
                acc += bias.to(tl.float32)[None, :]
            tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=store_mask)
        else:
            tl.atomic_add(c_ptrs, acc, mask=store_mask, sem="relaxed")


@triton.jit
def _cast_epilogue(
    acc_ptr,
    out_ptr,
    bias_ptr,
    M,
    N,
    stride_om,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N
    v = tl.load(acc_ptr + pid_m * N + offs_n, mask=mask, other=0.0)
    if HAS_BIAS:
        v += tl.load(bias_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + pid_m * stride_om + offs_n, v.to(out_ptr.dtype.element_ty), mask=mask
    )


def nt_gemm(a, b, cfg, out=None, workspace=None, bias=None, sms=188):
    M, K = a.shape
    N = b.shape[0]
    bm, bn, bk, gm, sk, warps, stages, persist, stream_b = cfg
    num_pid_m = triton.cdiv(M, bm)
    num_pid_n = triton.cdiv(N, bn)
    num_tiles = num_pid_m * num_pid_n
    if out is None:
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    if persist:
        programs = min(num_tiles, sms * persist)
    else:
        programs = num_tiles

    target = out if sk == 1 else workspace
    if sk > 1:
        workspace.zero_()

    _nt_gemm[(programs, sk)](
        a,
        b,
        target,
        bias,
        M,
        N,
        K,
        a.stride(0),
        b.stride(0),
        target.stride(0),
        NUM_TILES=num_tiles,
        NUM_PID_N=num_pid_n,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=gm,
        SPLIT_K=sk,
        EVEN_K=(K % (bk * sk) == 0),
        EVEN_N=(N % bn == 0),
        HAS_BIAS=(bias is not None and sk == 1),
        STREAM_B=stream_b,
        NUM_PROGRAMS=programs,
        num_warps=warps,
        num_stages=stages,
    )
    if sk > 1:
        _cast_epilogue[(M, triton.cdiv(N, 1024))](
            workspace,
            out,
            bias,
            M,
            N,
            out.stride(0),
            HAS_BIAS=bias is not None,
            BLOCK_N=1024,
        )
    return out


CONFIGS = []
for bm, bn, bk in [
    (64, 64, 128),
    (64, 128, 64),
    (64, 128, 128),
    (64, 256, 64),
    (128, 64, 64),
    (128, 64, 128),
    (128, 128, 64),
    (128, 128, 128),
    (128, 256, 64),
    (256, 64, 64),
    (256, 128, 64),
    (32, 128, 128),
    (32, 256, 64),
    (64, 32, 128),
    (128, 32, 128),
]:
    for warps in (4, 8):
        for stages in (2, 3, 4):
            for sk in (1, 2, 4):
                for persist in (0, 1, 2):
                    CONFIGS.append((bm, bn, bk, 8, sk, warps, stages, persist, True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--batches", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    peak = measure_peak_bandwidth(device)
    print(f"# SMs={sms} peak={peak/1e12:.3f} TB/s", flush=True)

    shapes = {
        k: v
        for k, v in SHAPES.items()
        if not args.shapes or k in args.shapes.split(",")
    }
    batches = (
        [int(x) for x in args.batches.split(",")] if args.batches else list(BATCHES)
    )

    results = []
    for label, (N, K) in shapes.items():
        nrot = rotation_count(N * K * 2)
        bs = [torch.randn(N, K, dtype=dtype, device=device) for _ in range(nrot)]
        for M in batches:
            a = torch.randn(M, K, dtype=dtype, device=device)
            out = torch.empty(M, N, dtype=dtype, device=device)
            ws = torch.zeros(M, N, dtype=torch.float32, device=device)
            nbytes = _bytes_moved(M, N, K, 2)
            ref = torch.nn.functional.linear(a, bs[0])

            t_cublas = time_us_graph(
                [(lambda w=w: torch.nn.functional.linear(a, w)) for w in bs]
            )
            bw_cublas = 100.0 * nbytes / (t_cublas * 1e-6) / peak
            tflops = 2 * M * N * K / (t_cublas * 1e-6) / 1e12
            print(
                f"\n== {label} N={N} K={K} M={M}: cuBLAS {t_cublas:.2f} us  "
                f"BW {bw_cublas:.1f}%  {tflops:.1f} TFLOP/s",
                flush=True,
            )

            scored = []
            for cfg in CONFIGS:
                bm, bn, bk, gm, sk, warps, stages, persist, sb = cfg
                if bm > max(32, 2 * M):
                    continue
                if K % (bk * sk) != 0:
                    continue
                if bm * bn > 32768 or bn * bk * stages * 2 > 200000:
                    continue
                try:
                    nt_gemm(a, bs[0], cfg, out=out, workspace=ws, sms=sms)
                    torch.cuda.synchronize()
                except Exception:
                    continue
                rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if rel > 5e-2:
                    continue
                try:
                    t = time_us_graph(
                        [
                            (
                                lambda w=w: nt_gemm(
                                    a, w, cfg, out=out, workspace=ws, sms=sms
                                )
                            )
                            for w in bs
                        ],
                        reps=3,
                    )
                except Exception:
                    continue
                scored.append((t, cfg, rel))
            scored.sort(key=lambda x: x[0])
            for t, cfg, rel in scored[: args.top]:
                bw = 100.0 * nbytes / (t * 1e-6) / peak
                print(
                    f"   {t:8.2f} us  BW {bw:5.1f}%  {t_cublas/t:5.3f}x  "
                    f"cfg={cfg} rel={rel:.2e}",
                    flush=True,
                )
            if scored:
                best_t, best_cfg, _ = scored[0]
                results.append(
                    {
                        "shape": label,
                        "N": N,
                        "K": K,
                        "M": M,
                        "cublas_us": t_cublas,
                        "cublas_bw_pct": bw_cublas,
                        "best_us": best_t,
                        "best_cfg": list(best_cfg),
                        "speedup": t_cublas / best_t,
                        "best_bw_pct": 100.0 * nbytes / (best_t * 1e-6) / peak,
                        "top": [
                            {"us": t, "cfg": list(c)} for t, c, _ in scored[: args.top]
                        ],
                    }
                )
            del a, out, ws, ref
            torch.cuda.empty_cache()
        del bs
        torch.cuda.empty_cache()

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
