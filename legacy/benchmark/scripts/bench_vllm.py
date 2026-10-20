#!/usr/bin/env python3
"""vLLM comparison benchmark — mirrors scripts/m1_concurrency_smoke.py.

Same model, same decode length, same single + N-concurrent protocol, so the
numbers are directly comparable to the UnieInfra (tokenspeed-fork) engine.
"""

import os
import time

MODEL = os.environ.get("UNIEINFRA_SMOKE_MODEL", "/workspace/models/Qwen/Qwen3-8B")
N = int(os.environ.get("UNIEINFRA_N", "128"))
MAX_NEW = int(os.environ.get("UNIEINFRA_MAX_NEW", "128"))


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=False,
        max_num_seqs=max(N, 256),
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)

    # single-request
    p0 = ["The capital of France is"]
    llm.generate(p0, SamplingParams(temperature=0.0, max_tokens=8))  # warmup
    t = time.time()
    o1 = llm.generate(p0, sp)
    dt1 = time.time() - t
    n1 = len(o1[0].outputs[0].token_ids)
    print(f"[vllm] single: {n1} tok / {dt1:.3f}s = {n1/dt1:.1f} tok/s", flush=True)

    # N concurrent
    prompts = [
        f"Question {i}: Explain in one paragraph what makes a good API."
        for i in range(N)
    ]
    t = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t
    tot = sum(len(o.outputs[0].token_ids) for o in outs)
    print(
        f"[vllm] N={N} concurrent: total {tot} tok / {dt:.2f}s "
        f"= {tot/dt:.0f} tok/s aggregate, {tot/N/dt:.1f} tok/s per-req",
        flush=True,
    )


if __name__ == "__main__":
    main()
