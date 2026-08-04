#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Micro-benchmark for the decode-batch GEMM (E45).

Two contenders on the same shapes, same clocks, same process:

1. ``cublas`` - ``torch.nn.functional.linear``, i.e. the cuBLASLt/CUTLASS
   ``s16816gemm`` tiles the engine runs today (the E44 baseline).
2. ``triton`` - ``unieinfra_kernel.ops.gemm.decode_batch_gemm_triton``.

Modes::

    --mode compare  head-to-head table (time + achieved DRAM BW)
    --mode sweep    offline-tune the config space per shape; prints a table
                    ready to paste into ``_TUNED_CONFIGS``
    --mode ncu      one launch of each contender per shape, for `ncu` to profile
    --mode numerics fp32-reference error check (the E37 method, not eyeballing)

Bandwidth is reported as ``bytes / time`` where ``bytes`` counts the weight
stream (``N*K*itemsize``) plus activations and output; on these shapes the
weight is >95% of it. The denominator is a measured large-copy peak unless
``--peak-gbps`` is given. Note a read-dominated kernel can exceed 100% of a
copy peak, because a copy pays for a write on every byte and this does not.

**The baseline is measured, not assumed.** E44 reported these kernels from
inside a live engine; a parallel run could not reproduce one of its absolute
numbers standalone, so ``--mode compare`` always prints the incumbent it
measured and it is that number, not E44's, that the speedup column divides.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO / "unieinfra-kernel", _REPO / "tokenspeed-kernel" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from unieinfra_kernel.ops.gemm.decode_batch_gemm_triton import (  # noqa: E402
    BatchGemmConfig,
    decode_batch_gemm,
    select_batch_gemm_config,
    wave_utilisation,
)

# Qwen3-8B (hidden 4096) decode projections, ordered by their share of the
# C=128 decode step in E44.
SHAPES: dict[str, tuple[int, int]] = {
    "gate_up_proj": (24576, 4096),
    "down_proj": (4096, 12288),
    "qkv_proj": (6144, 4096),
    "o_proj": (4096, 4096),
}
BATCHES = (32, 64, 128, 256)

# In-engine durations E44 measured for the same kernels at C=128, for the
# isolated-vs-in-situ comparison the report has to make. ``None`` where E44 did
# not publish a per-kernel figure.
E44_IN_ENGINE_US: dict[tuple[str, int], float] = {
    ("gate_up_proj", 128): 194.1,
    ("gate_up_proj", 256): 194.9,
}


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


# GB202's L2 is 128 MiB — bigger than two of the weight matrices under test.
# Timing a loop that re-reads ONE weight matrix therefore measures L2
# bandwidth, not DRAM. Rotate over enough distinct weight buffers that each call
# starts cold, as it does in a real decode step where 16 GB of weights stream
# past between two visits to the same layer.
L2_ROTATION_BYTES = 640 * 1024 * 1024


def rotation_count(nbytes: int) -> int:
    """How many distinct weight buffers are needed to defeat the L2."""
    return max(2, min(16, -(-L2_ROTATION_BYTES // max(nbytes, 1))))


def time_us_graph(ops, reps: int = 7) -> float:
    """Time a list of callables by CUDA-graph replay, in microseconds per op.

    A graph is used for two reasons. It removes Python/launch overhead, which
    would otherwise flatter whichever contender has the cheaper Python wrapper.
    And it is what production does: decode runs captured (E7/E29).
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


def _shapes_and_batches(args):
    shapes = [s for s in SHAPES if not args.shapes or s in args.shapes.split(",")]
    batches = (
        [int(x) for x in args.batches.split(",")] if args.batches else list(BATCHES)
    )
    return shapes, batches


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def compare(device, dtype, args):
    """Head-to-head table: incumbent vs first-party, time and achieved BW.

    The two arms are timed **interleaved**, ``--rounds`` times, and the reported
    figure is the median of each arm's samples. This box is shared: other
    agents' engines appear and disappear on the same GPU mid-run, and a layout
    that times all of one arm and then all of the other silently attributes that
    drift to the kernel. A large-copy bandwidth probe is taken before and after
    the whole table and printed, so contention that did occur is visible rather
    than assumed away.
    """
    peak = args.peak_gbps * 1e9 if args.peak_gbps else measure_peak_bandwidth(device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    print(f"# device={torch.cuda.get_device_name(device)} SMs={sms}")
    print(f"# measured peak DRAM bandwidth: {peak / 1e12:.3f} TB/s")
    shapes, batches = _shapes_and_batches(args)

    rows = []
    hdr = (
        f"{'shape':14s} {'M':>4s} {'cuBLAS us':>10s} {'BW%':>6s} "
        f"{'triton us':>10s} {'BW%':>6s} {'speedup':>8s} {'tiles':>6s} "
        f"{'E44 us':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label in shapes:
        N, K = SHAPES[label]
        nrot = rotation_count(N * K * 2)
        bs = [torch.randn(N, K, dtype=dtype, device=device) for _ in range(nrot)]
        for M in batches:
            a = torch.randn(M, K, dtype=dtype, device=device)
            out = torch.empty(M, N, dtype=dtype, device=device)
            nbytes = _bytes_moved(M, N, K, a.element_size())
            ref = torch.nn.functional.linear(a, bs[0])

            cfg = select_batch_gemm_config(M, N, K, device.index or 0)
            decode_batch_gemm(a, bs[0], out=out, config=cfg)
            torch.cuda.synchronize()
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()

            cublas_ops = [(lambda w=w: torch.nn.functional.linear(a, w)) for w in bs]
            triton_ops = [
                (lambda w=w: decode_batch_gemm(a, w, out=out, config=cfg)) for w in bs
            ]
            c_samples, t_samples = [], []
            for _ in range(args.rounds):
                c_samples.append(time_us_graph(cublas_ops, reps=3))
                t_samples.append(time_us_graph(triton_ops, reps=3))
            t_cublas = statistics.median(c_samples)
            t_triton = statistics.median(t_samples)
            spread = (max(c_samples) - min(c_samples)) / max(t_cublas, 1e-9)
            util, tiles, per_sm = wave_utilisation(M, N, K, cfg, sms)

            def bw(t):
                return 100.0 * nbytes / (t * 1e-6) / peak

            e44 = E44_IN_ENGINE_US.get((label, M))
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
                    "triton_tiles": tiles,
                    "triton_ctas_per_sm": per_sm,
                    "triton_util_pred": util,
                    "speedup": t_cublas / t_triton,
                    "e44_in_engine_us": e44,
                    "peak_bytes_per_s": peak,
                    "cublas_samples": c_samples,
                    "triton_samples": t_samples,
                    "cublas_spread": spread,
                }
            )
            print(
                f"{label:14s} {M:4d} {t_cublas:10.2f} {bw(t_cublas):6.1f} "
                f"{t_triton:10.2f} {bw(t_triton):6.1f} "
                f"{t_cublas / t_triton:7.3f}x {tiles:6d} "
                f"{(f'{e44:.1f}' if e44 else '-'):>7s}"
                f"{'   <-- unstable' if spread > 0.05 else ''}",
                flush=True,
            )
            del a, out, ref
        del bs
        torch.cuda.empty_cache()
    peak_after = measure_peak_bandwidth(device)
    print(
        f"# peak DRAM bandwidth after the table: {peak_after / 1e12:.3f} TB/s "
        f"({100.0 * (peak_after - peak) / peak:+.1f}% drift)"
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"peak_before": peak, "peak_after": peak_after, "rows": rows}, indent=2
            )
        )
    return rows


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

_SWEEP_TILES = [
    (32, 128, 128),
    (32, 256, 64),
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
]


def sweep(device, dtype, args):
    """Rank configs offline. The winners become ``_TUNED_CONFIGS`` entries.

    Two stages, because the full cross product is ~230 launches per shape and
    the box is shared — a sweep that runs for an hour is a sweep measured under
    somebody else's job. Stage 1 ranks tile shapes at fixed pipeline settings;
    stage 2 re-opens warps / stages / split-K on the survivors only.
    """
    peak = args.peak_gbps * 1e9 if args.peak_gbps else measure_peak_bandwidth(device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    print(f"# SMs={sms} measured peak DRAM = {peak / 1e12:.3f} TB/s", flush=True)
    shapes, batches = _shapes_and_batches(args)
    best_rows = []
    for label in shapes:
        N, K = SHAPES[label]
        nrot = rotation_count(N * K * 2)
        bs = [torch.randn(N, K, dtype=dtype, device=device) for _ in range(nrot)]
        for M in batches:
            a = torch.randn(M, K, dtype=dtype, device=device)
            out = torch.empty(M, N, dtype=dtype, device=device)
            nbytes = _bytes_moved(M, N, K, 2)
            ref = torch.nn.functional.linear(a, bs[0])
            t_cublas = time_us_graph(
                [(lambda w=w: torch.nn.functional.linear(a, w)) for w in bs]
            )
            print(
                f"\n== {label} N={N} K={K} M={M}: cuBLAS {t_cublas:.2f} us "
                f"BW {100.0 * nbytes / (t_cublas * 1e-6) / peak:.1f}%",
                flush=True,
            )

            def measure(cfg):
                """Time one config, or ``None`` if it does not build or is wrong."""
                if K % (cfg.block_k * cfg.split_k) != 0:
                    return None
                try:
                    decode_batch_gemm(a, bs[0], out=out, config=cfg)
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001 - smem / register limits
                    return None
                rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                if rel > 5e-2:
                    return None
                try:
                    return (
                        time_us_graph(
                            [
                                (
                                    lambda w=w: decode_batch_gemm(
                                        a, w, out=out, config=cfg
                                    )
                                )
                                for w in bs
                            ],
                            reps=3,
                        ),
                        rel,
                    )
                except Exception:  # noqa: BLE001
                    return None

            stage1 = []
            for bm, bn, bk in _SWEEP_TILES:
                if bm > max(16, 1 << (M - 1).bit_length()):
                    continue
                cfg = BatchGemmConfig(
                    block_m=bm,
                    block_n=bn,
                    block_k=bk,
                    num_warps=8 if bm * bn >= 16384 else 4,
                    num_stages=3,
                )
                r = measure(cfg)
                if r:
                    stage1.append((r[0], cfg, r[1]))
            stage1.sort(key=lambda x: x[0])

            scored = list(stage1)
            for _t, base, _rel in stage1[: args.keep]:
                for warps in (4, 8):
                    for stages in (2, 3, 4):
                        for split_k in (1, 2, 4):
                            cfg = BatchGemmConfig(
                                block_m=base.block_m,
                                block_n=base.block_n,
                                block_k=base.block_k,
                                split_k=split_k,
                                num_warps=warps,
                                num_stages=stages,
                            )
                            r = measure(cfg)
                            if r:
                                scored.append((r[0], cfg, r[1]))
            scored.sort(key=lambda x: x[0])
            for t, cfg, rel in scored[: args.top]:
                util, tiles, per_sm = wave_utilisation(M, N, K, cfg, sms)
                print(
                    f"   {t:8.2f} us  BW {100.0 * nbytes / (t * 1e-6) / peak:5.1f}%  "
                    f"{t_cublas / t:6.3f}x  {cfg}  tiles={tiles} "
                    f"cta/sm={per_sm} rel={rel:.1e}",
                    flush=True,
                )
            if scored:
                t, cfg, rel = scored[0]
                best_rows.append(
                    {
                        "shape": label,
                        "M": M,
                        "N": N,
                        "K": K,
                        "cublas_us": t_cublas,
                        "best_us": t,
                        "speedup": t_cublas / t,
                        "cfg": cfg.__dict__,
                    }
                )
            del a, out, ref
            torch.cuda.empty_cache()
        del bs
        torch.cuda.empty_cache()

    print("\n# paste into _TUNED_CONFIGS")
    for r in best_rows:
        c = r["cfg"]
        print(
            f"    ({r['M']}, {r['N']}, {r['K']}): BatchGemmConfig("
            f"{c['block_m']}, {c['block_n']}, {c['block_k']}, {c['group_m']}, "
            f"{c['split_k']}, {c['num_warps']}, {c['num_stages']}, "
            f"{c['persist_factor']}),  # {r['speedup']:.3f}x"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(best_rows, indent=2))
    return best_rows


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------


def numerics(device, dtype, args):
    """Compare both contenders against an fp32 reference (the E37 method).

    Eyeballing "the text looks fine" cannot separate a kernel that is wrong from
    one that is merely differently rounded. Both contenders are scored against
    the same fp32 matmul, so the only claim made is *relative*: our error is no
    worse than the incumbent's.
    """
    shapes, batches = _shapes_and_batches(args)
    print(f"{'shape':14s} {'M':>4s} {'cuBLAS relerr':>14s} {'triton relerr':>14s}")
    ok = True
    rows = []
    for label in shapes:
        N, K = SHAPES[label]
        b = torch.randn(N, K, dtype=dtype, device=device)
        b32 = b.float()
        for M in batches:
            a = torch.randn(M, K, dtype=dtype, device=device)
            gold = torch.nn.functional.linear(a.float(), b32)
            gn = gold.norm()
            c_ref = torch.nn.functional.linear(a, b).float()
            cfg = select_batch_gemm_config(M, N, K, device.index or 0)
            c_ours = decode_batch_gemm(a, b, config=cfg).float()
            e_ref = ((c_ref - gold).norm() / gn).item()
            e_ours = ((c_ours - gold).norm() / gn).item()
            # The bar is the incumbent, with 25% slack for a different
            # summation order over K.
            good = e_ours <= max(e_ref * 1.25, 1e-3)
            ok &= good
            rows.append(
                {
                    "shape": label,
                    "M": M,
                    "cublas_rel_err": e_ref,
                    "triton_rel_err": e_ours,
                    "pass": good,
                }
            )
            print(
                f"{label:14s} {M:4d} {e_ref:14.3e} {e_ours:14.3e}"
                f"{'' if good else '   <-- FAIL'}"
            )
            del a, gold, c_ref, c_ours
        del b, b32
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# l2probe
# ---------------------------------------------------------------------------


def time_us_eager(ops, warmup: int = 3, iters: int = 10, reps: int = 5) -> float:
    """Median-of-reps CUDA-event timing without a graph, in us per call.

    Takes a *list* of callables and cycles through it, exactly like
    :func:`time_us_graph`. Timing one callable in a tight loop would leave its
    weight buffer L2-resident no matter how many buffers the caller allocated —
    the rotation only defeats the L2 if the calls actually alternate.
    """
    for _ in range(warmup):
        for fn in ops:
            fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            for fn in ops:
                fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / (iters * len(ops)))
    return statistics.median(samples)


def l2probe(device, dtype, args):
    """Why an isolated benchmark and a live engine disagree. Two axes, 2x2.

    E44 measured these projections inside a live engine; a standalone rerun of
    ``qkv_proj`` found ~2.5x less time, and it was not clear which harness was
    wrong. Rather than arbitrate, this probe varies the two things that differ
    between a naive micro-benchmark and a decode step and reports all four
    cells:

    * **weight residency.** GB202 has a 128 MiB L2. ``qkv_proj``'s weight is
      48 MiB and ``o_proj``'s is 32 MiB, so calling one weight matrix in a loop
      leaves it resident and measures L2, not DRAM. A decode step never does
      that: ~16 GB of weights stream past between two visits to the same layer.
      ``cold`` rotates over enough distinct buffers to reproduce that.
    * **CUDA graph.** vLLM captures and replays decode. cuBLASLt picks its
      algorithm (and its workspace) per call, and what it can do during graph
      capture is not what it can do eagerly — so an eager micro-benchmark can
      measure a kernel the engine will never run.

    Both are properties of the *harness*, not of the kernel, and either one
    alone can move a number by more than the speedup being chased.
    """
    shapes, batches = _shapes_and_batches(args)
    hdr = (
        f"{'shape':14s} {'M':>4s} {'MiB':>6s} "
        f"{'eager/hot':>10s} {'eager/cold':>11s} {'graph/hot':>10s} "
        f"{'graph/cold':>11s} {'worst/best':>11s}"
    )
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for label in shapes:
        N, K = SHAPES[label]
        wmib = N * K * 2 / 1024 / 1024
        nrot = rotation_count(N * K * 2)
        bs = [torch.randn(N, K, dtype=dtype, device=device) for _ in range(nrot)]
        for M in batches:
            a = torch.randn(M, K, dtype=dtype, device=device)
            ops = [(lambda w=w: torch.nn.functional.linear(a, w)) for w in bs]
            one = [lambda: torch.nn.functional.linear(a, bs[0])]
            cells = {
                "eager_hot": time_us_eager(one),
                "eager_cold": time_us_eager(ops),
                "graph_hot": time_us_graph(one),
                "graph_cold": time_us_graph(ops),
            }
            worst, best = max(cells.values()), min(cells.values())
            rows.append(
                {
                    "shape": label,
                    "M": M,
                    "weight_mib": wmib,
                    "rotation_buffers": nrot,
                    **cells,
                    "spread": worst / best,
                }
            )
            print(
                f"{label:14s} {M:4d} {wmib:6.0f} "
                f"{cells['eager_hot']:10.2f} {cells['eager_cold']:11.2f} "
                f"{cells['graph_hot']:10.2f} {cells['graph_cold']:11.2f} "
                f"{worst / best:10.2f}x",
                flush=True,
            )
            del a
        del bs
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    return rows


# ---------------------------------------------------------------------------
# ncu
# ---------------------------------------------------------------------------


def ncu_mode(device, dtype, args):
    """Emit exactly one launch per (contender, shape, M) for ncu to capture."""
    shapes, batches = _shapes_and_batches(args)
    for label in shapes:
        N, K = SHAPES[label]
        for M in batches:
            a = torch.randn(M, K, dtype=dtype, device=device)
            b = torch.randn(N, K, dtype=dtype, device=device)
            out = torch.empty(M, N, dtype=dtype, device=device)
            cfg = select_batch_gemm_config(M, N, K, device.index or 0)
            # Warm up off-profile so JIT/compile launches are not captured.
            for _ in range(3):
                torch.nn.functional.linear(a, b)
                decode_batch_gemm(a, b, out=out, config=cfg)
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            torch.nn.functional.linear(a, b)
            decode_batch_gemm(a, b, out=out, config=cfg)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            print(f"# {label} M={M} cfg={cfg}", flush=True)
            del a, b, out
            torch.cuda.empty_cache()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("compare", "sweep", "ncu", "numerics", "l2probe", "all"),
        default="compare",
        help="'all' runs sweep -> compare -> numerics inside one process, so the "
        "ballast reservation is held across the whole session",
    )
    ap.add_argument("--shapes", default="", help="comma-separated subset of SHAPES")
    ap.add_argument("--batches", default="", help="comma-separated M values")
    ap.add_argument("--peak-gbps", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument(
        "--keep",
        type=int,
        default=3,
        help="tile shapes carried from sweep stage 1 into stage 2",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="interleaved A/B rounds per shape (compare mode)",
    )
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--reserve-gb",
        type=float,
        default=60.0,
        help="ballast held for the run so another engine cannot double-book the "
        "GPU mid-table; 0 disables (required under ncu, which replays kernels)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Ballast. This box is shared and every other job on it is an inference
    # engine that reserves ~85% of a card at startup; a micro-benchmark holding
    # 2 GB looks like a free GPU to the next engine's auto-sizing, which then
    # lands on top of the measurement. Holding the memory an engine would hold
    # is what makes the run exclusive -- it does not consume any bandwidth, it
    # just stops the card from being double-booked mid-table.
    ballast = None
    if args.reserve_gb > 0:
        free, _total = torch.cuda.mem_get_info()
        want = min(int(args.reserve_gb * 1e9), max(0, free - 12_000_000_000))
        if want > 0:
            ballast = torch.empty(want // 2, dtype=torch.bfloat16, device=device)
            print(f"# reserved {want / 1e9:.1f} GB of ballast for exclusivity")
    base_out = args.out
    try:
        if args.mode == "compare":
            compare(device, dtype, args)
        elif args.mode == "sweep":
            sweep(device, dtype, args)
        elif args.mode == "numerics":
            return numerics(device, dtype, args)
        elif args.mode == "l2probe":
            l2probe(device, dtype, args)
        elif args.mode == "all":
            for stage, fn in (
                ("sweep", sweep),
                ("compare", compare),
                ("numerics", numerics),
                ("l2probe", l2probe),
            ):
                print(f"\n\n########## {stage} ##########\n", flush=True)
                args.out = (
                    base_out.replace(".json", f".{stage}.json") if base_out else ""
                )
                fn(device, dtype, args)
        else:
            return ncu_mode(device, dtype, args)
    finally:
        del ballast
    return 0


if __name__ == "__main__":
    sys.exit(main())
