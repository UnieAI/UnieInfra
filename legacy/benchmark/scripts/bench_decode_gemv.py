#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Micro-benchmark for the decode-tuned GEMV (E33).

Three contenders on the same shapes, same clocks, same process:

1. ``cublas``  - ``torch.nn.functional.linear``, i.e. the cuBLAS ``gemvx``
   template the engine runs today (E26 baseline).
2. ``triton``  - ``unieinfra_kernel.ops.gemm.decode_gemv_triton``.
3. ``vllm``    - vLLM's ``bf16_skinny_gemm.cu`` (Apache-2.0), JIT-built from
   ``third_party/vllm`` and instantiated for *our* shapes. Optional: pass
   ``--with-vllm``; it is skipped with a note if the build fails.

Modes::

    --mode sweep    tune the Triton config space per shape
    --mode compare  head-to-head table (time + achieved DRAM BW)
    --mode ncu      one launch of each contender per shape, for `ncu` to profile

Bandwidth is reported as ``bytes / time`` where ``bytes`` counts the weight
stream (N*K*itemsize) plus activations and output; on these shapes the weight is
>99% of it. The peak used as the denominator is measured once by a large
``copy_`` unless ``--peak-gbps`` is given.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO / "unieinfra-kernel", _REPO / "tokenspeed-kernel" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from unieinfra_kernel.ops.gemm.decode_gemv_triton import (  # noqa: E402
    GemvConfig,
    decode_gemv,
    select_gemv_config,
)

# Qwen3-8B (hidden 4096) decode projections. gate_up/down are contrast shapes:
# E26 already measured them at 83-90% of peak, so they are where the new kernel
# must prove it does no harm.
SHAPES: dict[str, tuple[int, int]] = {
    # --- Qwen3-8B (hidden 4096). The shapes this kernel was designed and tuned
    # on; the only ones present in decode_gemv_triton's _TUNED_CONFIGS table.
    "qkv_proj": (6144, 4096),
    "o_proj": (4096, 4096),
    "gate_up_proj": (24576, 4096),
    "down_proj": (4096, 12288),
    # --- Qwen3.6-35B-A3B (hidden 2048, 16 heads x 256, 2 kv heads, MoE).
    # Only the attention projections are dense here; the FFN goes through the
    # MoE path. Note K=2048 is half of anything the tuning sweep saw, and
    # o_proj's N=2048 is narrower than any tuned entry -- the two axes the
    # kernel's own docstring identifies as driving achieved bandwidth.
    "m35_qkv": (5120, 2048),
    "m35_o": (2048, 4096),
    # --- Qwen3-32B (hidden 5120, dense). Larger than anything tuned.
    "q32_qkv": (10240, 5120),
    "q32_o": (5120, 8192),
    "q32_gate_up": (51200, 5120),
    "q32_down": (5120, 25600),
}
BATCHES = (1, 2, 4, 8)


def _bytes_moved(M: int, N: int, K: int, itemsize: int) -> int:
    """Minimum DRAM traffic for one call: weights + activations + output."""
    return (N * K + M * K + M * N) * itemsize


def measure_peak_bandwidth(device: torch.device) -> float:
    """Measure achievable DRAM bandwidth (bytes/s) with a large device copy."""
    n = 1024 * 1024 * 1024 // 2  # 1 GiB of bf16, far past the 128 MiB L2
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


# GB202's L2 is 128 MiB — bigger than every weight matrix under test. Timing a
# loop that re-reads ONE weight matrix therefore measures L2 bandwidth, not DRAM
# (it reported >200% of peak before this was fixed). Rotate over enough distinct
# weight buffers that each call starts cold, as it does in a real decode step
# where 16 GB of weights stream past between two visits to the same layer.
L2_ROTATION_BYTES = 640 * 1024 * 1024


def rotation_count(nbytes: int) -> int:
    """How many distinct weight buffers are needed to defeat the L2."""
    return max(2, min(16, -(-L2_ROTATION_BYTES // max(nbytes, 1))))


def time_us_graph(ops, reps: int = 7) -> float:
    """Time a list of callables by CUDA-graph replay, in microseconds per op.

    A graph is used for two reasons. It removes Python/launch overhead, which on
    a 16 us kernel is otherwise the dominant term and would flatter whichever
    contender has the cheaper Python wrapper. And it is what production does:
    decode runs captured (E7/E29).
    """
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


def time_us(fn, warmup: int = 25, iters: int = 200, reps: int = 5) -> float:
    """Median-of-reps CUDA-event timing, in microseconds per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / iters)
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# vLLM skinny GEMM (optional third baseline)
# ---------------------------------------------------------------------------

_VLLM_SRC = _REPO / "third_party" / "vllm" / "csrc" / "libtorch_stable"


def build_vllm_skinny(shapes, batches, verbose: bool = False):
    """JIT-build vLLM's bf16 skinny GEMM instantiated for ``shapes``.

    The upstream file only instantiates DeepSeek/GLM ``(N, K)`` pairs, none of
    which are Qwen3-8B's, so this wrapper includes the upstream kernel template
    unchanged and adds instantiations for the shapes under test. The upstream
    source is *included*, never copied.

    Returns:
        A callable ``(out, a, b) -> None``, or ``None`` if the build fails.
    """
    from torch.utils.cpp_extension import load_inline

    header = _VLLM_SRC / "bf16_skinny_gemm.cu"
    if not header.is_file():
        print(f"[vllm] source not found at {header}", file=sys.stderr)
        return None

    # Only the kernel template + launcher are wanted; upstream's own explicit
    # instantiations at the bottom of the file are for other shapes and would
    # add ~250 template instantiations to the compile. Cut the file at the
    # instantiation block (a pure build-time concern; no code is modified).
    src = header.read_text()
    marker = "// Explicit instantiations."
    kernel_src = src.split(marker)[0]

    cases = []
    for n, k in sorted({s for s in shapes}):
        npb = 2 if n >= 6144 else 4
        for m in batches:
            cases.append((m, npb, n, k))

    inst = "\n".join(
        f"template void invokeBf16SkinnyGemm<128, {npb}, {m}, {n}, {k}>("
        f"__nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, int64_t,"
        f" cudaStream_t);"
        for (m, npb, n, k) in cases
    )
    dispatch = "\n".join(
        f"  if (m == {m} && n == {n} && k == {k}) {{ "
        f"invokeBf16SkinnyGemm<128, {npb}, {m}, {n}, {k}>(o, ap, bp, out_stride, "
        f"stream); return; }}"
        for (m, npb, n, k) in cases
    )

    cuda_src = f"""
{kernel_src}

{inst}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void vllm_skinny_gemm(torch::Tensor out, torch::Tensor a, torch::Tensor b) {{
  const int m = a.size(0), k = a.size(1), n = b.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto* ap = reinterpret_cast<__nv_bfloat16 const*>(a.data_ptr());
  auto* bp = reinterpret_cast<__nv_bfloat16 const*>(b.data_ptr());
  const int64_t out_stride = out.stride(0);
{dispatch}
  TORCH_CHECK(false, "vllm_skinny_gemm: no instantiation for this shape");
}}
"""
    cpp_src = "void vllm_skinny_gemm(at::Tensor out, at::Tensor a, at::Tensor b);"
    # The container ships several CUDA trees; only cuda-12.9 has an unversioned
    # libcudart.so for the linker to resolve -lcudart against.
    ldflags = [
        f"-L{d}"
        for d in (
            "/usr/local/cuda-12.9/targets/x86_64-linux/lib",
            "/usr/local/cuda/lib64",
        )
        if Path(d, "libcudart.so").exists()
    ]
    try:
        mod = load_inline(
            name="vllm_skinny_gemm_bench",
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            functions=["vllm_skinny_gemm"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode=arch=compute_120,code=sm_120",
            ],
            extra_ldflags=ldflags,
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001 - optional baseline
        print(f"[vllm] build failed, skipping this baseline: {exc}", file=sys.stderr)
        return None
    return mod.vllm_skinny_gemm


# ---------------------------------------------------------------------------
# Config sweep
# ---------------------------------------------------------------------------

SWEEP_BLOCK_N = (1, 2, 4, 8, 16)
SWEEP_BLOCK_K = (512, 1024, 2048)
SWEEP_WARPS = (2, 4, 8)
SWEEP_STAGES = (1, 2)
SWEEP_SPLIT_K = (1, 2, 4)


def sweep(device, dtype, args):
    """Grid-search the Triton config space per (M, N, K).

    Ranking only: plain (non-graph) timing with rotated weight buffers. Python
    launch overhead is an additive constant across Triton configs, so it does
    not reorder them; the winners are then re-timed under CUDA graph in
    ``--mode compare``.
    """
    results: dict[str, dict] = {}
    only = set(args.shapes.split(",")) if args.shapes else set(SHAPES)
    for label, (N, K) in SHAPES.items():
        if label not in only:
            continue
        nrot = rotation_count(N * K * 2)
        bs = [torch.randn(N, K, dtype=dtype, device=device) for _ in range(nrot)]
        for M in BATCHES:
            a = torch.randn(M, K, dtype=dtype, device=device)
            out = torch.empty(M, N, dtype=dtype, device=device)
            ref = torch.nn.functional.linear(a, bs[0])
            base = (
                time_us(
                    lambda: [torch.nn.functional.linear(a, w) for w in bs],
                    warmup=5,
                    iters=10,
                    reps=3,
                )
                / nrot
            )

            best = None
            tried = 0
            for bn, bk, w, st, sk in itertools.product(
                SWEEP_BLOCK_N, SWEEP_BLOCK_K, SWEEP_WARPS, SWEEP_STAGES, SWEEP_SPLIT_K
            ):
                if bn * bk > 16384 or bk * sk > K:
                    continue
                if bk < 32 * w * 2:  # fewer than 2 values/thread: pointless
                    continue
                cfg = GemvConfig(bn, bk, sk, w, st, stream_b=True)
                try:
                    decode_gemv(a, bs[0], out=out, config=cfg)
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001 - config out of resources
                    continue
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if not (err < 5e-2):
                    continue
                t = (
                    time_us(
                        lambda cfg=cfg: [
                            decode_gemv(a, w, out=out, config=cfg) for w in bs
                        ],
                        warmup=3,
                        iters=8,
                        reps=3,
                    )
                    / nrot
                )
                tried += 1
                if best is None or t < best[0]:
                    best = (t, cfg)
            key = f"{label}/M{M}"
            results[key] = {
                "M": M,
                "N": N,
                "K": K,
                "cublas_us": base,
                "best_us": best[0] if best else None,
                "config": best[1].__dict__ if best else None,
                "configs_tried": tried,
            }
            cfgstr = best[1] if best else "none"
            print(
                f"{key:24s} cuBLAS {base:8.2f}us  best "
                f"{best[0] if best else float('nan'):8.2f}us "
                f"({base / best[0] if best else float('nan'):.3f}x)  {cfgstr}",
                flush=True,
            )
            del a, out, ref
        del bs
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return results


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------


def compare(device, dtype, args):
    peak = args.peak_gbps * 1e9 if args.peak_gbps else measure_peak_bandwidth(device)
    print(f"# measured peak DRAM bandwidth: {peak / 1e12:.3f} TB/s")
    vllm_fn = (
        build_vllm_skinny(set(SHAPES.values()), BATCHES) if args.with_vllm else None
    )

    rows = []
    hdr = (
        f"{'shape':14s} {'M':>2s} "
        f"{'cuBLAS us':>10s} {'BW%':>6s} "
        f"{'triton us':>10s} {'BW%':>6s} {'speedup':>8s} "
        f"{'vllm us':>9s} {'BW%':>6s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, (N, K) in SHAPES.items():
        nrot = rotation_count(N * K * 2)
        bs = [torch.randn(N, K, dtype=dtype, device=device) for _ in range(nrot)]
        for M in BATCHES:
            a = torch.randn(M, K, dtype=dtype, device=device)
            out = torch.empty(M, N, dtype=dtype, device=device)
            nbytes = _bytes_moved(M, N, K, a.element_size())
            ref = torch.nn.functional.linear(a, bs[0])

            t_cublas = time_us_graph(
                [(lambda w=w: torch.nn.functional.linear(a, w)) for w in bs]
            )
            cfg = select_gemv_config(M, N, K, device.index or 0)
            decode_gemv(a, bs[0], out=out, config=cfg)
            torch.cuda.synchronize()
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            t_triton = time_us_graph(
                [(lambda w=w: decode_gemv(a, w, out=out, config=cfg)) for w in bs]
            )

            t_vllm = None
            if vllm_fn is not None:
                try:
                    vllm_fn(out, a, bs[0])
                    torch.cuda.synchronize()
                    t_vllm = time_us_graph(
                        [(lambda w=w: vllm_fn(out, a, w)) for w in bs]
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [vllm] {label}/M{M}: {exc}", file=sys.stderr)

            def bw(t):
                return 100.0 * nbytes / (t * 1e-6) / peak if t else float("nan")

            rows.append(
                {
                    "shape": label,
                    "M": M,
                    "N": N,
                    "K": K,
                    "bytes": nbytes,
                    "cublas_us": t_cublas,
                    "cublas_bw_pct": bw(t_cublas),
                    "triton_us": t_triton,
                    "triton_bw_pct": bw(t_triton),
                    "triton_rel_err": rel,
                    "triton_config": cfg.__dict__,
                    "vllm_us": t_vllm,
                    "vllm_bw_pct": bw(t_vllm) if t_vllm else None,
                    "peak_bytes_per_s": peak,
                }
            )
            vs = (
                f"{t_vllm:9.2f} {bw(t_vllm):6.1f}" if t_vllm else f"{'-':>9s} {'-':>6s}"
            )
            print(
                f"{label:14s} {M:2d} "
                f"{t_cublas:10.2f} {bw(t_cublas):6.1f} "
                f"{t_triton:10.2f} {bw(t_triton):6.1f} "
                f"{t_cublas / t_triton:7.3f}x {vs}",
                flush=True,
            )
            del a, out, ref
        del bs
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    return rows


# ---------------------------------------------------------------------------
# ncu mode: one launch of each contender, nothing else
# ---------------------------------------------------------------------------


def ncu_mode(device, dtype, args):
    """Emit exactly one launch per (contender, shape, M) for ncu to capture."""
    only = set(args.shapes.split(",")) if args.shapes else set(SHAPES)
    batches = (
        [int(x) for x in args.batches.split(",")] if args.batches else list(BATCHES)
    )
    vllm_fn = (
        build_vllm_skinny({SHAPES[s] for s in only}, batches, verbose=args.verbose)
        if args.with_vllm
        else None
    )

    for label in SHAPES:
        if label not in only:
            continue
        N, K = SHAPES[label]
        for M in batches:
            a = torch.randn(M, K, dtype=dtype, device=device)
            b = torch.randn(N, K, dtype=dtype, device=device)
            out = torch.empty(M, N, dtype=dtype, device=device)
            cfg = select_gemv_config(M, N, K, device.index or 0)
            # Warm every path (JIT, cuBLAS heuristic pick) OUTSIDE the profiled
            # region; ncu is invoked with --launch-skip to skip these.
            for _ in range(3):
                torch.nn.functional.linear(a, b)
                decode_gemv(a, b, out=out, config=cfg)
                if vllm_fn is not None:
                    vllm_fn(out, a, b)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"cublas/{label}/M{M}")
            torch.nn.functional.linear(a, b)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"triton/{label}/M{M}")
            decode_gemv(a, b, out=out, config=cfg)
            torch.cuda.nvtx.range_pop()
            if vllm_fn is not None:
                torch.cuda.nvtx.range_push(f"vllm/{label}/M{M}")
                vllm_fn(out, a, b)
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            print(f"launched {label} M={M} cfg={cfg}", flush=True)
            del a, b, out
            torch.cuda.empty_cache()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("sweep", "compare", "ncu"), default="compare")
    p.add_argument("--device", type=int, default=None, help="CUDA device index")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=None, help="write JSON results here")
    p.add_argument("--peak-gbps", type=float, default=None)
    p.add_argument("--with-vllm", action="store_true")
    p.add_argument("--shapes", default=None, help="ncu mode: comma-separated subset")
    p.add_argument("--batches", default=None, help="ncu mode: comma-separated M values")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.device is not None:
        torch.cuda.set_device(args.device)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = getattr(torch, args.dtype)
    props = torch.cuda.get_device_properties(device)
    print(
        f"# device {device.index}: {props.name} sm_{props.major}{props.minor} "
        f"SMs={props.multi_processor_count} torch={torch.__version__}"
    )
    torch.manual_seed(0)

    if args.mode == "sweep":
        sweep(device, dtype, args)
    elif args.mode == "compare":
        compare(device, dtype, args)
    else:
        ncu_mode(device, dtype, args)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    raise SystemExit(main())
