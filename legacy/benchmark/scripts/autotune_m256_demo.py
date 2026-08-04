#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Can measured selection capture E44's M=256 QKV loss?

E44 attributed ~8% of a 45.7 ms C=128/C=256 decode step to one thing: at
``M = 256`` cuBLASLt moves the QKV projection's 50.3 MB in **147.9 us instead of
43.0 us**. The question this script answers is not "is that slow" but "is it
reachable by **selection**", which needs two separate facts:

1. **How many candidates does the registry actually have at that shape?**
   Selection can only fix what it can choose between. Part 1 prints the
   candidate set at ``M = 1 / 8 / 256`` for ``(N=6144, K=4096)``.

2. **If a second real candidate existed, would measurement pick it?** Part 2
   registers one — ``F.linear`` against a **K-major** weight, which is the same
   arithmetic through a different cuBLAS heuristic input, with the relayout done
   once by a ``weight_preprocessor`` rather than per call — deliberately at a
   *lower* priority band so the heuristic prefers the incumbent. Then it runs
   both strategies over a batch sweep and reports the microseconds each one
   chose.

The demo kernel is registered **only inside this process**. It is a measurement
instrument, not a proposal: changing a served model's weight layout has memory
and loader consequences this script does not address.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "unieinfra-kernel"))

_N, _K = 6144, 4096  # Qwen3-8B qkv_proj -- the shape E44 measured
_BATCHES = (1, 8, 64, 128, 256, 512)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="relative speedup required to overturn the heuristic",
    )
    args = parser.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["UNIEINFRA_KERNEL_AUTOTUNE_CACHE"] = str(
        _REPO / "benchmark" / "results" / "autotune" / "m256_cache"
    )

    import torch

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2

    import unieinfra_kernel  # noqa: F401  (registers the real kernels)
    from unieinfra_kernel.boundary import (
        KernelRegistry,
        Priority,
        current_platform,
        dense_tensor_format,
        format_signature,
        register_kernel,
    )
    from unieinfra_kernel.dispatch import shape_traits
    from unieinfra_kernel.selection import selection as sel
    from unieinfra_kernel.selection.autotune import AutotuneParams, autotune_cache

    platform = current_platform()
    registry = KernelRegistry.get()
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16), b=dense_tensor_format(torch.bfloat16)
    )
    print(f"device: {platform.device_name} ({platform.arch}, {platform.sm_count} SMs)")
    print(f"shape:  qkv_proj N={_N} K={_K}\n")

    # ---------------------------------------------------------------- part 1
    print("PART 1 -- what can selection even choose between?")
    baseline_candidates = {}
    for M in (1, 8, 256):
        specs = registry.get_for_operator(
            "gemm", "mm", platform=platform, format_signature=signature
        )
        traits = shape_traits(M, _N, _K)
        surviving = [s.name for s in specs if sel.spec_matches_traits(s, traits)]
        baseline_candidates[M] = surviving
        print(f"  M={M:<4} {len(surviving)} candidate(s): {', '.join(surviving)}")
    print(
        "\n  => Autotuning measures only when there are >= 2 candidates. Where the\n"
        "     list above has one entry, measurement has nothing to compare and the\n"
        "     incumbent stands no matter how slow it is.\n"
    )

    # ---------------------------------------------------------------- part 2
    # A second REAL candidate: same arithmetic, K-major weight, relayout done
    # once by the registry's weight_preprocessor rather than on every call.
    def to_k_major(weight: torch.Tensor) -> torch.Tensor:
        """Return a [N, K] weight whose memory is K-major (stride (1, N))."""
        return weight.t().contiguous().t()

    @register_kernel(
        "gemm",
        "mm",
        name="demo_torch_linear_kmajor",
        solution="reference",
        signatures=frozenset({signature}),
        traits={},
        # PORTABLE, i.e. *below* the incumbent torch_linear (PORTABLE + 3), so
        # the heuristic will never pick it and any win must come from measuring.
        priority=Priority.PORTABLE,
        origin="contrib",
        weight_preprocessor=to_k_major,
    )
    def demo_torch_linear_kmajor(
        A,
        B,
        A_scales,
        B_scales,
        out_dtype,
        *,
        alpha=None,
        block_size=None,
        bias=None,
        out=None,
    ):
        """``A @ B.T`` where ``B`` is K-major. See module docstring."""
        del A_scales, B_scales, block_size, alpha
        result = torch.nn.functional.linear(A, B, bias)
        if result.dtype != out_dtype:
            result = result.to(out_dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    # Make the demo kernel's inputs actually K-major, which is what the
    # weight_preprocessor would have arranged at load time.
    from unieinfra_kernel.contracts import gemm as gemm_contract
    from unieinfra_kernel.selection.inputs import set_input_generator

    base_generator = gemm_contract.generate_gemm_inputs

    def generator_with_layout(**kwargs):
        inputs = base_generator(**kwargs)
        inputs["_kmajor_B"] = to_k_major(inputs["B"])
        return inputs

    # The registry times `kernel(**inputs)` with ONE input dict per shape, so
    # the layout variant travels as an extra key and each candidate takes the
    # weight it wants. Every other kernel must have that key removed, or it
    # would (correctly) raise on an unexpected argument and be dropped -- which
    # is the fail-safe working, but it would also destroy the comparison.
    def _bind(impl, use_kmajor):
        def wrapper(**kwargs):
            kwargs = dict(kwargs)
            kmajor = kwargs.pop("_kmajor_B", None)
            if use_kmajor and kmajor is not None:
                kwargs["B"] = kmajor
            return impl(**kwargs)

        return wrapper

    set_input_generator("gemm", "mm", generator_with_layout)
    for spec in registry.list_kernels(family="gemm", mode="mm"):
        registry._impls[spec.name] = _bind(
            registry.get_impl(spec.name),
            use_kmajor=(spec.name == "demo_torch_linear_kmajor"),
        )
    registry.clear_cache()

    print("PART 2 -- with a second real candidate, does measurement find it?")
    autotune_cache(platform).clear()

    rows = []
    for M in _BATCHES:
        traits = shape_traits(M, _N, _K)
        shape = {"M": M, "N": _N, "K": _K}

        sel.set_selection_policy(sel.SelectionPolicy())
        heuristic = sel.select_kernel("gemm", "mm", signature, traits=traits).name

        sel.clear_autotune_log()
        sel.enable_autotuning(
            [("gemm", "mm")], params=AutotuneParams(margin=args.margin, use_cache=False)
        )
        measured = sel.select_kernel(
            "gemm", "mm", signature, traits=traits, shape=shape
        ).name
        log = sel.autotune_log()
        times = log[-1]["times_us"] if log else {}

        incumbent_us = times.get(heuristic)
        chosen_us = times.get(measured)
        gain = (
            (incumbent_us / chosen_us - 1.0) * 100.0
            if incumbent_us and chosen_us
            else 0.0
        )
        rows.append(
            {
                "M": M,
                "heuristic": heuristic,
                "autotune": measured,
                "times_us": times,
                "gain_pct": gain,
            }
        )
        rendering = "  ".join(
            f"{n}={v:.1f}us" if v is not None else f"{n}=FAILED"
            for n, v in sorted(times.items(), key=lambda kv: (kv[1] is None, kv[1]))
        )
        note = ""
        ok_times = {n: v for n, v in times.items() if v is not None}
        if measured != heuristic:
            note = f"  <-- measurement flipped it, {gain:+.1f}%"
        elif len(ok_times) > 1 and incumbent_us:
            runner_up = min(v for n, v in ok_times.items() if n != heuristic)
            edge = (incumbent_us / runner_up - 1.0) * 100.0
            note = (
                f"  (incumbent held; runner-up {edge:+.1f}%"
                f"{', inside margin' if 0 < edge <= args.margin * 100 else ''})"
            )
        print(f"  M={M:<4} heuristic={heuristic:<28} autotune={measured:<28}{note}")
        print(f"        {rendering}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "device": platform.device_name,
                    "shape": {"N": _N, "K": _K},
                    "margin": args.margin,
                    "baseline_candidates": baseline_candidates,
                    "rows": rows,
                },
                indent=1,
            )
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
