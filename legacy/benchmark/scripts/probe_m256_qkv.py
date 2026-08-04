#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Probe the E44 M=256 QKV case: is it reachable by *selection* at all?

E44 attributed ~8% of a 45.7 ms C=128/C=256 decode step to one thing: at
``M = 256`` cuBLASLt switches the QKV projection (``N=6144, K=4096``) to a
``wmma_..._32x32_32x1`` tile and moves the same 50.3 MB in **147.9 us instead of
43.0 us** — about 19% of the achievable rate.

Autotuning can only fix that if there is **more than one candidate to choose
between**. This script answers that empirically, by timing every dense-bf16 path
that already exists on this box for the same shape:

* ``F.linear`` — what the registry's ``torch_linear`` calls, i.e. cuBLASLt.
* ``F.linear`` with ``preferred_blas_library("cublas")`` — the *other* BLAS
  PyTorch can dispatch to, and the one lever that could pick a different tile
  without anybody writing a kernel.
* ``torch.mm`` against an explicitly column-major weight — a different cuBLAS
  entry point with a different heuristic input.
* ``unieinfra_decode_gemv_triton`` (forced) — the first-party decode GEMV,
  outside its gate, for reference.

All timings use L2-defeating weight rotation (GB202's L2 is 128 MiB; a 50 MB
weight would otherwise be resident and the number meaningless) and report
achieved DRAM bandwidth so the result can be read against E44's 43.0 us target.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "unieinfra-kernel"))

# The shape E44 measured, plus the batch sweep that brackets the regression.
_N, _K = 6144, 4096
_BATCHES = (1, 2, 8, 32, 64, 128, 256, 512)

_L2_ROTATION_BYTES = 640 * 1024 * 1024


def _rotations(nbytes: int) -> int:
    return max(2, min(16, -(-_L2_ROTATION_BYTES // max(nbytes, 1))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ.setdefault("UNIEINFRA_DECODE_GEMV_FORCE", "1")

    import torch

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2

    import torch.nn.functional as F

    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} ({props.multi_processor_count} SMs)")

    def time_us(fn, n_variants, warmup=10, iters=50, reps=5):
        for i in range(warmup):
            fn(i % n_variants)
        torch.cuda.synchronize()
        samples = []
        for rep in range(reps):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for i in range(iters):
                fn((rep + i) % n_variants)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1e3 / iters)
        return statistics.median(samples)

    def blas(which):
        """Set PyTorch's preferred BLAS, returning False when unsupported."""
        try:
            torch.backends.cuda.preferred_blas_library(which)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  (preferred_blas_library({which!r}) unavailable: {exc})")
            return False

    try:
        from unieinfra_kernel.ops.gemm.decode_gemv_triton import decode_gemv
    except Exception:  # noqa: BLE001
        decode_gemv = None

    rows = []
    for M in _BATCHES:
        nbytes = _N * _K * 2
        nrot = _rotations(nbytes)
        weights = [
            torch.randn(_N, _K, dtype=torch.bfloat16, device="cuda")
            for _ in range(nrot)
        ]
        a = torch.randn(M, _K, dtype=torch.bfloat16, device="cuda")
        # Column-major weight view: same bytes, different cuBLAS heuristic input.
        weights_t = [w.t().contiguous().t() for w in weights]
        moved = (_N * _K + M * _K + M * _N) * 2

        entry = {"M": M, "N": _N, "K": _K, "bytes": moved}

        if blas("cublaslt"):
            entry["F.linear/cublaslt"] = time_us(
                lambda i: F.linear(a, weights[i]), nrot
            )
        if blas("cublas"):
            entry["F.linear/cublas"] = time_us(lambda i: F.linear(a, weights[i]), nrot)
        blas("cublaslt")
        entry["torch.mm/colmajor-B"] = time_us(
            lambda i: torch.mm(a, weights_t[i].t()), nrot
        )
        if decode_gemv is not None and M <= 8:
            try:
                entry["unieinfra_decode_gemv_triton"] = time_us(
                    lambda i: decode_gemv(a, weights[i]), nrot
                )
            except Exception as exc:  # noqa: BLE001
                entry["unieinfra_decode_gemv_triton"] = None
                entry["gemv_error"] = repr(exc)

        timings = {k: v for k, v in entry.items() if isinstance(v, float)}
        best_name = min(timings, key=timings.get)
        entry["best"] = best_name
        entry["incumbent"] = "F.linear/cublaslt"
        incumbent = timings.get("F.linear/cublaslt")
        if incumbent:
            entry["speedup_vs_incumbent"] = incumbent / timings[best_name]
        rows.append(entry)

        rendering = "  ".join(
            f"{k.split('/')[-1]}={v:.1f}us({moved / v / 1e3:.0f}GB/s)"
            for k, v in sorted(timings.items(), key=lambda kv: kv[1])
        )
        gain = entry.get("speedup_vs_incumbent", 1.0)
        flag = "  <-- incumbent is NOT the best" if gain > 1.02 else ""
        print(f"M={M:<4} {rendering}{flag}")
        del weights, weights_t, a
        torch.cuda.empty_cache()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
