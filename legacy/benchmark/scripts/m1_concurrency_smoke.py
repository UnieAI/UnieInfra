#!/usr/bin/env python3
"""High-concurrency smoke test: aggregate throughput under a batch of requests.

Single-request tok/s is the wrong metric for a throughput engine. This sends N
prompts at once so the C++ FSM scheduler can continuously batch them, and reports
aggregate output tok/s — the number that matters for RPS-under-SLO.

    UNIEINFRA_SMOKE_MODEL=/workspace/models/Qwen/Qwen3-8B \
    UNIEINFRA_N=64 UNIEINFRA_MAX_NEW=128 \
    python3 scripts/m1_concurrency_smoke.py
"""

import os
import time

MODEL = os.environ.get("UNIEINFRA_SMOKE_MODEL", "/workspace/models/Qwen/Qwen3-8B")
N = int(os.environ.get("UNIEINFRA_N", "64"))
MAX_NEW = int(os.environ.get("UNIEINFRA_MAX_NEW", "128"))

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


def main():
    from tokenspeed.runtime.entrypoints.engine import Engine

    eng = Engine(
        model=MODEL,
        dtype="bfloat16",
        world_size=1,
        ep_size=1,
        trust_remote_code=True,
        enforce_eager=False,  # CUDA graph on
        disable_custom_all_reduce=True,
        moe_backend=os.environ.get("UNIEINFRA_MOE_BACKEND") or "auto",
        attention_backend=os.environ.get("UNIEINFRA_ATTN_BACKEND") or None,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        max_num_seqs=max(N, 256),
        log_level="error",
    )

    prompts = [
        f"Question {i}: Explain in one paragraph what makes a good API."
        for i in range(N)
    ]

    # warmup one request so timing excludes first-call JIT/autotune
    eng.generate(
        prompt=prompts[0], sampling_params={"temperature": 0.0, "max_new_tokens": 8}
    )

    t0 = time.time()
    outs = eng.generate(
        prompt=prompts,
        sampling_params={"temperature": 0.0, "max_new_tokens": MAX_NEW},
    )
    dt = time.time() - t0

    tot = 0
    for o in outs:
        tot += len(o.get("output_ids", [])) if isinstance(o, dict) else 0
    print(f"[concurrency] N={N} concurrent reqs, {MAX_NEW} tok each", flush=True)
    print(f"[concurrency] total output tok: {tot}  wall: {dt:.2f}s", flush=True)
    print(f"[concurrency] AGGREGATE THROUGHPUT: {tot/dt:.0f} tok/s", flush=True)
    print(f"[concurrency] per-request avg: {tot/N/dt:.1f} tok/s", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
