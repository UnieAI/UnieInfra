#!/usr/bin/env python3
"""Minimal decode-only profiling harness for the ncu bandwidth-saturation audit.

Env vars:
  UNIEINFRA_SMOKE_MODEL   model path (default Qwen3-8B)
  UNIEINFRA_BS            batch size: 1 or 128 (default 1)
  UNIEINFRA_EAGER         1 = eager (no cuda graph, per-kernel visible to ncu), 0 = graph
  UNIEINFRA_WARMUP_NEW    max_new_tokens for the warmup call (default 1 -> prefill only)
  UNIEINFRA_MEASURE_NEW   max_new_tokens for the measured call (default 4 -> 1 prefill + 3 decode)
  UNIEINFRA_ENABLE_NVTX   1 = pass enable_nvtx=True to Engine (default 1)

This process is meant to be launched directly under `ncu --target-processes all`.
It intentionally does the warmup call FIRST (inside this same process) so Triton
autotune/JIT selection has already happened once matches are found in the on-disk
cache from a prior plain (non-ncu) run of this same script -- see the caller.
"""

import os
import time

MODEL = os.environ.get("UNIEINFRA_SMOKE_MODEL", "/workspace/models/Qwen/Qwen3-8B")
BS = int(os.environ.get("UNIEINFRA_BS", "1"))
EAGER = os.environ.get("UNIEINFRA_EAGER", "1") == "1"
WARMUP_NEW = int(os.environ.get("UNIEINFRA_WARMUP_NEW", "1"))
MEASURE_NEW = int(os.environ.get("UNIEINFRA_MEASURE_NEW", "4"))
ENABLE_NVTX = os.environ.get("UNIEINFRA_ENABLE_NVTX", "1") == "1"

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


def main():
    from tokenspeed.runtime.entrypoints.engine import Engine

    t0 = time.time()
    eng = Engine(
        model=MODEL,
        dtype="bfloat16",
        world_size=1,
        ep_size=1,
        trust_remote_code=True,
        enforce_eager=EAGER,
        disable_custom_all_reduce=True,
        attention_backend="triton",
        gpu_memory_utilization=float(os.environ.get("UNIEINFRA_GPU_UTIL", "0.85")),
        max_model_len=4096,
        max_total_tokens=(
            int(os.environ["UNIEINFRA_MAX_TOTAL_TOKENS"])
            if os.environ.get("UNIEINFRA_MAX_TOTAL_TOKENS")
            else None
        ),
        max_num_seqs=max(BS, 256) if BS > 1 else 32,
        enable_nvtx=ENABLE_NVTX,
        log_level="error",
    )
    print(
        f"[harness] engine up in {time.time()-t0:.1f}s bs={BS} eager={EAGER}",
        flush=True,
    )

    if BS == 1:
        prompts = "The capital of France is"
    else:
        prompts = [
            f"Question {i}: Explain in one paragraph what makes a good API."
            for i in range(BS)
        ]

    # Warmup: triggers Triton JIT/autotune (cached on disk from prior plain run)
    # and, in graph mode, CUDA-graph capture for this exact batch bucket.
    t1 = time.time()
    eng.generate(
        prompt=prompts,
        sampling_params={"temperature": 0.0, "max_new_tokens": WARMUP_NEW},
    )
    print(f"[harness] warmup done in {time.time()-t1:.2f}s", flush=True)

    # Measured region: this is what ncu should be pointed at (via NVTX range
    # instance filtering: skip the first matching instance == prefill, keep
    # the rest == decode steps).
    t2 = time.time()
    out = eng.generate(
        prompt=prompts,
        sampling_params={"temperature": 0.0, "max_new_tokens": MEASURE_NEW},
    )
    dt = time.time() - t2
    if isinstance(out, list):
        ntok = sum(
            len(o.get("output_ids", [])) if isinstance(o, dict) else MEASURE_NEW
            for o in out
        )
    else:
        ntok = len(out.get("output_ids", [])) if isinstance(out, dict) else MEASURE_NEW
    print(
        f"[harness] measured: {ntok} tok / {dt:.4f}s = {ntok/dt:.1f} tok/s", flush=True
    )

    eng.shutdown()
    print("[harness] DONE", flush=True)


if __name__ == "__main__":
    main()
