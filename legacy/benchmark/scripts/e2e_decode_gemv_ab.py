#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""End-to-end arm of the decode GEMV A/B: batch-1 tok/s (E33).

**One arm per process, on purpose.** Decode runs under CUDA graph, and the graph
is captured during ``Engine(...)`` construction — whichever GEMM is selected at
capture time is baked into the replay. Toggling the kernel after startup changes
nothing (measured: +0.04%, i.e. the same graph twice). So the arm is chosen by
``UNIEINFRA_DECODE_GEMV`` *before* the engine is built, and the driver
alternates whole processes so drift on a shared box cancels::

    for r in 1 2; do
      for arm in 0 1; do
        UNIEINFRA_DECODE_GEMV=$arm \\
        PYTHONPATH=/workspace/roy/UnieInfra/unieinfra-kernel \\
          python3 benchmark/scripts/e2e_decode_gemv_ab.py
      done
    done
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

MODEL = os.environ.get("UNIEINFRA_SMOKE_MODEL", "/workspace/models/Qwen/Qwen3-8B")
PROMPT = "Write a detailed explanation of how a transformer language model works."
MAX_NEW = int(os.environ.get("UNIEINFRA_MAX_NEW", "512"))
ROUNDS = int(os.environ.get("UNIEINFRA_ROUNDS", "3"))

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


def _report_selection() -> None:
    """Print which kernel ``gemm.mm`` picks for the decode shapes.

    Must import ``unieinfra_kernel`` explicitly: the engine imports it inside its
    spawned worker, so without this the parent process reports the incumbent no
    matter which arm is running. Note the arm is *not* visible here by design —
    ``UNIEINFRA_DECODE_GEMV=0`` leaves the kernel registered and selected and
    makes its runtime guard delegate to ``F.linear``, so both arms share one
    dispatch path and differ only in the kernel body.
    """
    import torch
    import unieinfra_kernel  # noqa: F401  (discovery)
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    sig = format_signature(
        a=dense_tensor_format(torch.bfloat16), b=dense_tensor_format(torch.bfloat16)
    )
    for label, (N, K) in (("qkv_proj", (6144, 4096)), ("o_proj", (4096, 4096))):
        traits = {
            "m_le_8": True,
            "n_le_16384": N <= 16384,
            "k_align_16": K % 16 == 0,
            "n_align_16": True,
            "k_align_32": True,
            "n_align_64": True,
            "n_align_128": True,
            "k_align_64": True,
            "k_align_128": True,
            "n_min_128": True,
            "k_min_128": True,
        }
        print(
            f"[select] {label} M=1 -> {select_kernel('gemm', 'mm', sig, traits=traits).name}",
            flush=True,
        )


def _decode_rate(eng) -> float:
    t0 = time.time()
    out = eng.generate(
        prompt=PROMPT,
        sampling_params={"temperature": 0.0, "max_new_tokens": MAX_NEW},
    )
    dt = time.time() - t0
    ntok = len(out.get("output_ids", [])) if isinstance(out, dict) else MAX_NEW
    return ntok / dt


def main() -> int:
    arm = "on" if os.environ.get("UNIEINFRA_DECODE_GEMV", "1") != "0" else "off"
    print(f"[e2e] arm: decode_gemv={arm}", flush=True)

    from tokenspeed.runtime.entrypoints.engine import Engine

    eng = Engine(
        model=MODEL,
        dtype="bfloat16",
        world_size=1,
        ep_size=1,
        trust_remote_code=True,
        enforce_eager=False,  # CUDA graph: the production decode path
        disable_custom_all_reduce=True,
        attention_backend=os.environ.get("UNIEINFRA_ATTN_BACKEND") or "triton",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        log_level="warning",
    )
    _report_selection()

    _decode_rate(eng)  # warm up: JIT, allocator, clocks
    rates = [_decode_rate(eng) for _ in range(ROUNDS)]
    median = statistics.median(rates)
    print(
        f"[e2e] RESULT decode_gemv={arm} median={median:.2f} tok/s "
        f"all={[round(x, 2) for x in rates]}",
        flush=True,
    )
    if os.environ.get("UNIEINFRA_E2E_JSON"):
        with open(os.environ["UNIEINFRA_E2E_JSON"], "a") as fh:
            fh.write(json.dumps({"arm": arm, "median": median, "all": rates}) + "\n")
    eng.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
