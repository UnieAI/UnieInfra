#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Validate measurement-based kernel selection against the answers we already know.

Autotuning is only worth having if it agrees with hardware. We have hardware
answers already, from E33/E34 on sm_120 with Qwen3-8B's decode projections:

* at ``M <= 2`` on ``qkv_proj`` / ``o_proj`` / ``down_proj`` the first-party
  Triton GEMV beats cuBLAS (1.04x-1.16x), so autotune must select
  ``unieinfra_decode_gemv_triton``;
* by ``M = 8`` it is 0.62x-0.66x, so autotune must **not**;
* ``gate_up_proj`` (``N = 24576``) fails the ``n_le_16384`` trait gate, so the
  GEMV is never even a candidate there.

Two arms, because the dense GEMV's *registered wrapper* internally delegates to
``F.linear`` for any shape its profitability rule declines:

``--arm gated`` (default)
    Production configuration. At ``M = 8`` our wrapper is behaviourally
    ``F.linear``, so the two candidates measure within noise and the margin
    rule keeps the heuristic winner.

``--arm forced``
    ``UNIEINFRA_DECODE_GEMV_FORCE=1``, which bypasses the profitability rule so
    the Triton kernel really runs at every ``M``. This is the arm that tests
    *measurement*: the kernel is genuinely slower at ``M = 8`` and autotune has
    to reject it on the timings alone.

Usage::

    python3 benchmark/scripts/autotune_validate.py --device 3 --out results.json
    python3 benchmark/scripts/autotune_validate.py --device 3 --arm forced
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "unieinfra-kernel"))


# E33's measured table, expressed as the decision autotune must reproduce.
# (label, N, K) -> the M values where our kernel should win.
_SHAPES = (
    ("qkv_proj", 6144, 4096),
    ("o_proj", 4096, 4096),
    ("down_proj", 4096, 12288),
    ("gate_up_proj", 24576, 4096),
)
_BATCHES = (1, 2, 4, 8)
_OURS = "unieinfra_decode_gemv_triton"


def expected_winner(label: str, M: int) -> str | None:
    """The kernel E33/E34 says should win, or ``None`` where we make no claim.

    Args:
        label: Projection name.
        M: Token count in the step.

    Returns:
        A kernel name, or ``None`` for ``M = 4``, where the two timing
        methodologies in E33 disagreed and the profitability rule deliberately
        leaves the shape to cuBLAS without claiming a measurement.
    """
    if label == "gate_up_proj":
        return "torch_linear"  # excluded by n_le_16384; never a candidate
    if M <= 2:
        return _OURS
    if M >= 8:
        return "torch_linear"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=None, help="CUDA device index")
    parser.add_argument("--arm", choices=("gated", "forced"), default="gated")
    parser.add_argument("--out", default=None, help="write JSON results here")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="autotune cache directory (default: a fresh temp dir, so the run "
        "measures rather than replaying an earlier one)",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="do not clear the cache first; use to time a warm second startup",
    )
    args = parser.parse_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if args.arm == "forced":
        os.environ["UNIEINFRA_DECODE_GEMV_FORCE"] = "1"

    cache_dir = args.cache_dir or str(
        _REPO / "benchmark" / "results" / "autotune_cache"
    )
    os.environ["UNIEINFRA_KERNEL_AUTOTUNE_CACHE"] = cache_dir

    import torch

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2

    import unieinfra_kernel  # noqa: F401  (registers the kernels)
    from unieinfra_kernel.boundary import (
        current_platform,
        dense_tensor_format,
        format_signature,
    )
    from unieinfra_kernel.dispatch import shape_traits
    from unieinfra_kernel.selection import selection as sel
    from unieinfra_kernel.selection.autotune import AutotuneParams, autotune_cache

    platform = current_platform()
    print(f"device: {platform.device_name} ({platform.arch}, {platform.sm_count} SMs)")
    print(f"arm:    {args.arm}   cache: {cache_dir}")

    if not args.reuse_cache:
        autotune_cache(platform).clear()

    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16), b=dense_tensor_format(torch.bfloat16)
    )

    # --- Arm A: heuristic (the incumbent decision table) ---------------------
    sel.set_selection_policy(sel.SelectionPolicy())
    heuristic: dict[str, str] = {}
    for label, N, K in _SHAPES:
        for M in _BATCHES:
            key = f"{label}/M{M}"
            try:
                heuristic[key] = sel.select_kernel(
                    "gemm", "mm", signature, traits=shape_traits(M, N, K)
                ).name
            except Exception as exc:  # noqa: BLE001
                heuristic[key] = f"ERROR:{exc}"

    # --- Arm B: autotune (measured) -----------------------------------------
    sel.clear_autotune_log()
    sel.enable_autotuning([("gemm", "mm")], params=AutotuneParams())
    ops = [
        ("gemm", "mm", signature, shape_traits(M, N, K), {"M": M, "N": N, "K": K})
        for label, N, K in _SHAPES
        for M in _BATCHES
    ]
    cold_seconds = sel.warmup_selection(ops)

    measured: dict[str, str] = {}
    for label, N, K in _SHAPES:
        for M in _BATCHES:
            key = f"{label}/M{M}"
            try:
                measured[key] = sel.select_kernel(
                    "gemm",
                    "mm",
                    signature,
                    traits=shape_traits(M, N, K),
                    shape={"M": M, "N": N, "K": K},
                ).name
            except Exception as exc:  # noqa: BLE001
                measured[key] = f"ERROR:{exc}"

    decisions = sel.autotune_log()
    times = {
        f"{d['shape']['N']}x{d['shape']['K']}/M{d['shape']['M']}": d.get("times_us", {})
        for d in decisions
    }

    # --- Warm restart: same decisions, no measurement ------------------------
    from unieinfra_kernel.boundary import KernelRegistry
    from unieinfra_kernel.selection import autotune as autotune_mod

    KernelRegistry.get().clear_cache()
    autotune_mod._cache_by_path.clear()
    sel.clear_autotune_log()
    warm_start = time.perf_counter()
    warm_seconds = sel.warmup_selection(ops)
    del warm_start
    warm = {}
    for label, N, K in _SHAPES:
        for M in _BATCHES:
            warm[f"{label}/M{M}"] = sel.select_kernel(
                "gemm",
                "mm",
                signature,
                traits=shape_traits(M, N, K),
                shape={"M": M, "N": N, "K": K},
            ).name
    warm_sources = {d.get("source") for d in sel.autotune_log()}

    # --- Report --------------------------------------------------------------
    failures: list[str] = []
    rows = []
    for label, N, K in _SHAPES:
        for M in _BATCHES:
            key = f"{label}/M{M}"
            want = expected_winner(label, M)
            got = measured[key]
            ok = want is None or got == want
            if not ok:
                failures.append(f"{key}: expected {want}, autotune chose {got}")
            rows.append(
                {
                    "shape": key,
                    "N": N,
                    "K": K,
                    "M": M,
                    "heuristic": heuristic[key],
                    "autotune": got,
                    "warm": warm[key],
                    "expected": want,
                    "ok": ok,
                    "times_us": times.get(f"{N}x{K}/M{M}", {}),
                }
            )

    width = max(len(r["shape"]) for r in rows)
    print()
    print(f"{'shape'.ljust(width)}  {'heuristic':<32} {'autotune':<32} verdict")
    for r in rows:
        verdict = "ok" if r["ok"] else "MISMATCH"
        if r["heuristic"] != r["autotune"]:
            verdict += "  <-- measurement disagreed with heuristic"
        print(
            f"{r['shape'].ljust(width)}  {r['heuristic']:<32} {r['autotune']:<32} {verdict}"
        )
        if r["times_us"]:
            rendering = "  ".join(
                f"{n}={v:.1f}us" if v is not None else f"{n}=FAILED"
                for n, v in sorted(r["times_us"].items())
            )
            print(f"{' ' * width}    {rendering}")

    disagreements = [r for r in rows if r["heuristic"] != r["autotune"]]
    print()
    print(
        f"cold tune: {cold_seconds:.2f}s for {len(ops)} shapes "
        f"({cold_seconds / max(len(ops), 1) * 1000:.0f} ms/shape)"
    )
    print(
        f"warm tune: {warm_seconds:.2f}s (disk cache; sources={sorted(warm_sources)})"
    )
    print(
        f"warm decisions identical to cold: "
        f"{all(warm[r['shape']] == r['autotune'] for r in rows)}"
    )
    print(f"disagreements with heuristic: {len(disagreements)}")
    print(f"validation failures: {len(failures)}")
    for failure in failures:
        print(f"  ! {failure}")

    payload = {
        "device": platform.device_name,
        "arch": platform.arch,
        "sm_count": platform.sm_count,
        "arm": args.arm,
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "warm_sources": sorted(s for s in warm_sources if s),
        "rows": rows,
        "failures": failures,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.out}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
