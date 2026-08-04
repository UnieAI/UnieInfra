#!/usr/bin/env python3
"""E45 staged sweep: mid-M decode GEMM vs cuBLASLt. Scratch, not a deliverable.

Stage 1 ranks tile shapes with fixed pipeline params; stage 2 refines the
survivors over warps / stages / split-K / persistence. Reports achieved DRAM
bandwidth against a measured copy peak and the analytic wave-quantization
utilisation, so the mechanism is visible in the table and not just the time.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore import (  # noqa: E402
    SHAPES,
    _bytes_moved,
    measure_peak_bandwidth,
    nt_gemm,
    rotation_count,
    time_us_graph,
)

TILES = [
    (32, 128, 128),
    (32, 256, 64),
    (64, 64, 128),
    (64, 128, 64),
    (64, 128, 128),
    (64, 256, 64),
    (128, 32, 128),
    (128, 64, 64),
    (128, 64, 128),
    (128, 128, 64),
    (128, 128, 128),
    (128, 256, 64),
    (256, 64, 64),
    (256, 128, 64),
]


def _smem_bytes(bm, bn, bk, stages):
    return (bm * bk + bn * bk) * 2 * stages


def utilisation(M, N, K, cfg, sms=188):
    """Analytic wave-quantisation utilisation: tiles / (resident * rounds)."""
    bm, bn, bk, _gm, sk, warps, stages, persist, _sb = cfg
    tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn) * sk
    # Resident CTAs per SM: shared-memory limited on a 100 KB/SM part.
    per_sm = max(1, min(4, 100_000 // max(_smem_bytes(bm, bn, bk, stages), 1)))
    resident = sms * per_sm
    rounds = -(-tiles // resident)
    return tiles / (resident * rounds), tiles, per_sm


def run(device, dtype, shapes, batches, args):
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    peak = measure_peak_bandwidth(device)
    print(f"# SMs={sms} measured peak DRAM = {peak / 1e12:.3f} TB/s", flush=True)
    if peak < 1.2e12:
        print(
            "# WARNING: peak far below spec -- GPU is contended, results invalid",
            flush=True,
        )

    results = []
    for label in shapes:
        N, K = SHAPES[label]
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
            bw_c = 100.0 * nbytes / (t_cublas * 1e-6) / peak
            tf_c = 2 * M * N * K / (t_cublas * 1e-6) / 1e12
            print(
                f"\n== {label} N={N} K={K} M={M}: cuBLAS {t_cublas:.2f} us "
                f"BW {bw_c:.1f}%  {tf_c:.1f} TFLOP/s",
                flush=True,
            )

            def bench(cfg):
                bm, bn, bk, _gm, sk, warps, stages, _p, _sb = cfg
                if bm > max(32, 1 << (M - 1).bit_length()):
                    return None
                if K % (bk * sk) != 0 or _smem_bytes(bm, bn, bk, stages) > 210_000:
                    return None
                try:
                    nt_gemm(a, bs[0], cfg, out=out, workspace=ws, sms=sms)
                    torch.cuda.synchronize()
                except Exception:
                    return None
                rel = (
                    (out.float() - ref.float()).norm()
                    / ref.float().norm().clamp_min(1e-9)
                ).item()
                if rel > 5e-2:
                    return None
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
                    return None
                return t, rel

            # -- stage 1: tile shape, fixed pipeline --------------------------
            stage1 = []
            for bm, bn, bk in TILES:
                warps = 8 if bm * bn >= 16384 else 4
                cfg = (bm, bn, bk, 8, 1, warps, 3, 0, True)
                r = bench(cfg)
                if r:
                    stage1.append((r[0], cfg, r[1]))
            stage1.sort(key=lambda x: x[0])
            if not stage1:
                print("   (no stage-1 survivor)", flush=True)
                continue
            print("   -- stage 1 (tile shape) --", flush=True)
            for t, cfg, rel in stage1[:5]:
                u, tiles, per_sm = utilisation(M, N, K, cfg, sms)
                print(
                    f"   {t:8.2f} us {t_cublas / t:6.3f}x  tile="
                    f"{cfg[0]}x{cfg[1]}x{cfg[2]}  tiles={tiles} "
                    f"cta/sm={per_sm} util={u * 100:.0f}%",
                    flush=True,
                )

            # -- stage 2: pipeline params on the survivors --------------------
            best = []
            for _t, cfg, _rel in stage1[: args.keep]:
                bm, bn, bk = cfg[0], cfg[1], cfg[2]
                for warps in (4, 8):
                    for stages in (2, 3, 4, 5):
                        for sk in (1, 2, 4):
                            for persist in (0, 1, 2):
                                c = (bm, bn, bk, 8, sk, warps, stages, persist, True)
                                r = bench(c)
                                if r:
                                    best.append((r[0], c, r[1]))
            best.extend(stage1)
            best.sort(key=lambda x: x[0])
            print("   -- stage 2 (best overall) --", flush=True)
            seen = set()
            top = []
            for t, cfg, rel in best:
                if cfg in seen:
                    continue
                seen.add(cfg)
                top.append((t, cfg, rel))
                if len(top) >= args.top:
                    break
            for t, cfg, rel in top:
                u, tiles, per_sm = utilisation(M, N, K, cfg, sms)
                bw = 100.0 * nbytes / (t * 1e-6) / peak
                print(
                    f"   {t:8.2f} us BW {bw:5.1f}% {t_cublas / t:6.3f}x  "
                    f"cfg={cfg} tiles={tiles} cta/sm={per_sm} "
                    f"util={u * 100:.0f}% rel={rel:.1e}",
                    flush=True,
                )

            bt, bc, br = top[0]
            results.append(
                {
                    "shape": label,
                    "N": N,
                    "K": K,
                    "M": M,
                    "cublas_us": t_cublas,
                    "cublas_bw_pct": bw_c,
                    "cublas_tflops": tf_c,
                    "best_us": bt,
                    "best_cfg": list(bc),
                    "best_rel": br,
                    "best_bw_pct": 100.0 * nbytes / (bt * 1e-6) / peak,
                    "speedup": t_cublas / bt,
                    "peak_bytes_per_s": peak,
                    "top": [{"us": t, "cfg": list(c)} for t, c, _ in top],
                }
            )
            del a, out, ws, ref
            torch.cuda.empty_cache()
        del bs
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="")
    ap.add_argument("--batches", default="32,64,128,256")
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    shapes = [s for s in SHAPES if not args.shapes or s in args.shapes.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    res = run(torch.device("cuda"), torch.bfloat16, shapes, batches, args)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    print("\n# summary")
    for r in res:
        print(
            f"{r['shape']:14s} M={r['M']:3d}  cuBLAS {r['cublas_us']:8.2f} us "
            f"({r['cublas_bw_pct']:5.1f}% BW)  triton {r['best_us']:8.2f} us "
            f"({r['best_bw_pct']:5.1f}% BW)  {r['speedup']:.3f}x  {r['best_cfg']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
