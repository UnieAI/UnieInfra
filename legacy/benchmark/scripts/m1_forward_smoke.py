#!/usr/bin/env python3
"""M1 bring-up smoke test: run a single forward+decode on sm_120.

Verifies that the vendored UnieInfra engine (forked from TokenSpeed) loads a
model and generates coherent tokens on NVIDIA sm_120 (RTX PRO 6000), exercising
the kernel-selection winners audited in M0b (GDN attention, MoE, sampling).

Run inside the Blackwell dev container with the sm_120 env (see
docs/design/tokenspeed-sm120-support.md section 9):

    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export LD_LIBRARY_PATH=<cu13 libs>:<pip nvidia/*/lib>:$LD_LIBRARY_PATH
    python3 scripts/m1_forward_smoke.py
"""

import os
import time

MODEL = os.environ.get(
    "UNIEINFRA_SMOKE_MODEL", "/workspace/models/Qwen/Qwen3.6-35B-A3B"
)
PROMPT = "The capital of France is"

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


def main():
    # NOTE: TokenSpeed's Engine uses `spawn` multiprocessing, so all engine
    # construction MUST live under `if __name__ == "__main__"` — otherwise each
    # spawned child re-imports this module as __main__ and re-spawns.
    from tokenspeed.runtime.entrypoints.engine import Engine

    t0 = time.time()
    eng = Engine(
        model=MODEL,
        dtype="bfloat16",
        world_size=1,
        ep_size=1,
        trust_remote_code=True,
        enforce_eager=os.environ.get("UNIEINFRA_EAGER", "1")
        == "1",  # CUDA graph off by default (bring-up); set 0 for perf
        disable_custom_all_reduce=True,  # single-GPU: skip multi-GPU comm prewarm
        moe_backend=os.environ.get("UNIEINFRA_MOE_BACKEND") or "auto",
        attention_backend=os.environ.get("UNIEINFRA_ATTN_BACKEND") or None,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        log_level="info",
    )
    print(f"[smoke] engine up in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    max_new = int(os.environ.get("UNIEINFRA_MAX_NEW", "32"))
    out = eng.generate(
        prompt=PROMPT,
        sampling_params={"temperature": 0.0, "max_new_tokens": max_new},
    )
    dt = time.time() - t1
    ntok = len(out.get("output_ids", [])) if isinstance(out, dict) else max_new
    print("[smoke] === OUTPUT ===", flush=True)
    print(out, flush=True)
    print(
        f"[smoke] generate(): {ntok} tok / {dt:.3f}s = {ntok/dt:.1f} tok/s "
        f"(eager={eng.server_args.enforce_eager})",
        flush=True,
    )

    eng.shutdown()
    print("[smoke] DONE — forward+decode succeeded on sm_120", flush=True)


if __name__ == "__main__":
    main()
