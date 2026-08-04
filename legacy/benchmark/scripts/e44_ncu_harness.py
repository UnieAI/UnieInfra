#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""E44 step 2: minimal vLLM decode harness shaped for `ncu` kernel replay.

Runs the real engine, warms it (prefix cache + CUDA graphs + inductor JIT),
then brackets a SHORT decode-only region with cudaProfilerStart/Stop so
`ncu --profile-from-start off` only ever sees steady-state decode launches --
never prefill, never warmup, never compile.

The second `generate` replays the SAME prompts, so with prefix caching on its
prefill is a cache hit and every profiled step is a full-width decode step at
the requested concurrency and context.

Keep `--gpu-mem-util` as low as the KV pool allows: ncu's default kernel replay
saves and restores resident device memory on every pass (E26).
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def build_prompts(n: int, ilen: int, vocab: int, seed: int = 0):
    import random

    rng = random.Random(seed)
    lo, hi = 1000, min(vocab, 100000) - 1
    return [[rng.randint(lo, hi) for _ in range(ilen)] for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/Qwen/Qwen3-8B")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--ilen", type=int, default=1024)
    ap.add_argument("--olen", type=int, default=3, help="decode steps to profile")
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--max-model-len", type=int, default=1152)
    ap.add_argument("--eager", action="store_true")
    # E48 additions.  Defaults reproduce E44 exactly, so every number already
    # recorded stays comparable; passing these switches the KV cache dtype and
    # the attention backend so `dram__bytes_read.sum` on the attention kernel
    # can be compared bf16-vs-fp8.  That comparison is the only way to tell a
    # real bandwidth win from a cache that is merely stored small and then
    # upconverted on read -- the two are indistinguishable from throughput.
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--attention-backend", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    extra = {}
    if args.attention_backend:
        extra["attention_config"] = {"backend": args.attention_backend}

    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.concurrency,
        enforce_eager=args.eager,
        enable_prefix_caching=True,
        disable_log_stats=True,
        **extra,
    )
    print(f"[ncu-harness] engine up in {time.time()-t0:.1f}s", flush=True)

    vocab = llm.llm_engine.vllm_config.model_config.get_vocab_size()
    toks = build_prompts(args.concurrency, args.ilen, vocab, seed=5)
    sp_warm = SamplingParams(
        temperature=0.0, max_tokens=4, min_tokens=4, ignore_eos=True
    )
    # Warmup: captures CUDA graphs and JITs every inductor/triton kernel that
    # the decode path uses, so nothing compiles inside the profiled region.
    llm.generate(toks, sp_warm, use_tqdm=False)
    print("[ncu-harness] warmup done", flush=True)

    # Step-granular drive: prefix caching cannot be relied on to suppress the
    # prefill (it did not, first attempt), so instead run the engine one step
    # at a time and only enable the profiler once EVERY request is past prefill
    # and the batch is a full-width steady-state decode.
    import torch

    engine = llm.llm_engine
    room = args.max_model_len - args.ilen - 8
    sp = SamplingParams(
        temperature=0.0, max_tokens=room, min_tokens=room, ignore_eos=True
    )
    for i, t in enumerate(toks):
        engine.add_request(f"r{i}", {"prompt_token_ids": t}, sp)

    ntok: dict[str, int] = {}
    steps = 0
    profiling = False
    prof_steps = 0
    while engine.has_unfinished_requests():
        outs = engine.step()
        steps += 1
        for o in outs:
            try:
                ntok[o.request_id] = len(o.outputs[0].token_ids)
            except Exception:
                pass
        ready = len(ntok) == args.concurrency and min(ntok.values()) >= 2
        if not profiling and ready:
            torch.cuda.synchronize()
            print(
                f"[ncu-harness] steady state after {steps} steps "
                f"(min tokens={min(ntok.values())}); profiler ON",
                flush=True,
            )
            torch.cuda.cudart().cudaProfilerStart()
            profiling = True
            continue
        if profiling:
            prof_steps += 1
            if prof_steps >= args.olen:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                print(
                    f"[ncu-harness] profiled {prof_steps} decode steps at "
                    f"C={args.concurrency}, ctx~{args.ilen + steps}",
                    flush=True,
                )
                break
    for i in range(args.concurrency):
        try:
            engine.abort_request(f"r{i}")
        except Exception:
            pass
    print("[ncu-harness] profiled region done", flush=True)


if __name__ == "__main__":
    main()
