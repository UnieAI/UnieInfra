#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""E44 step 1: coarse attribution of a steady-state vLLM decode step at high C.

Answers, BEFORE any per-kernel hardware counter is touched:
  * total wall time of one steady-state decode step at concurrency C,
  * GPU-busy fraction of that step (union of CUDA kernel intervals / wall),
  * number of kernel launches per step,
  * time share per kernel category (weight GEMM / attention / norm+act /
    sampling / lm_head / elementwise glue / copies / other).

Method: drive the offline `LLM` engine with C fixed-length token prompts and
`ignore_eos`, so every step after prefill is a full-width decode step. Capture
with `torch.profiler` (CUDA activity only), dump the RAW kernel event list
(name, ts, dur) to JSON, and do all segmentation/aggregation offline in
`e44_analyze.py`. One GPU run, all analysis re-runnable without a GPU.

Also emits a profiler-free timing pair (olen=1 vs olen=1+N) so the step wall
time is measured WITHOUT profiler overhead, which is the number to trust.

Env / flags mirror what the ncu pass will use so the two are comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")


def build_prompts(n: int, ilen: int, vocab: int, seed: int = 0):
    """Distinct random token prompts, so prefix caching cannot collapse them."""
    import random

    rng = random.Random(seed)
    # Keep away from special ids; Qwen3 specials live at the top of the vocab.
    lo, hi = 1000, min(vocab, 100000) - 1
    return [[rng.randint(lo, hi) for _ in range(ilen)] for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/Qwen/Qwen3-8B")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--ilen", type=int, default=1024)
    ap.add_argument("--olen", type=int, default=34)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--trace-out", default=None, help="raw kernel events json")
    ap.add_argument(
        "--eager", action="store_true", help="enforce_eager (no CUDA graphs)"
    )
    ap.add_argument("--no-profile", action="store_true", help="timing only")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.concurrency,
        enforce_eager=args.eager,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    setup_s = time.time() - t0
    print(f"[e41] engine up in {setup_s:.1f}s", flush=True)

    vocab = llm.llm_engine.vllm_config.model_config.get_vocab_size()
    toks = build_prompts(args.concurrency, args.ilen, vocab)

    def run(olen: int, prompts, profile: bool = False, tag: str = ""):
        sp = SamplingParams(
            temperature=0.0, max_tokens=olen, min_tokens=olen, ignore_eos=True
        )
        if profile:
            import torch

            act = [torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=act, record_shapes=False) as prof:
                t = time.time()
                out = llm.generate(prompts, sp, use_tqdm=False)
                dt = time.time() - t
            return dt, out, prof
        t = time.time()
        out = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.time() - t
        return dt, out, None

    # ---- warmup (also forces CUDA-graph capture for this batch shape) -------
    warm = build_prompts(args.concurrency, args.ilen, vocab, seed=99)
    run(4, warm)
    print("[e41] warmup done", flush=True)

    # ---- profiler-free step wall time: olen=1 (prefill only) vs olen=1+N ----
    res = {
        "model": args.model,
        "concurrency": args.concurrency,
        "ilen": args.ilen,
        "olen": args.olen,
        "eager": args.eager,
        "gpu_mem_util": args.gpu_mem_util,
        "setup_s": setup_s,
    }
    timings = []
    for rep in range(3):
        p1 = build_prompts(args.concurrency, args.ilen, vocab, seed=1000 + rep)
        p2 = build_prompts(args.concurrency, args.ilen, vocab, seed=2000 + rep)
        dt_pre, _, _ = run(1, p1)
        dt_all, _, _ = run(args.olen, p2)
        n_dec = args.olen - 1
        step_ms = (dt_all - dt_pre) / n_dec * 1e3
        timings.append(
            {
                "rep": rep,
                "prefill_s": dt_pre,
                "total_s": dt_all,
                "decode_steps": n_dec,
                "step_ms": step_ms,
                "decode_tok_s": args.concurrency * n_dec / (dt_all - dt_pre),
            }
        )
        print(f"[e41] rep{rep} step={step_ms:.3f}ms", flush=True)
    res["timings"] = timings
    res["step_ms_median"] = sorted(t["step_ms"] for t in timings)[len(timings) // 2]

    # ---- profiled run: raw kernel events -----------------------------------
    if not args.no_profile:
        p3 = build_prompts(args.concurrency, args.ilen, vocab, seed=7)
        dt, _, prof = run(args.olen, p3, profile=True)
        res["profiled_total_s"] = dt
        events = []
        for e in prof.events():
            # CUDA kernel events carry a device_type of CUDA and a nonzero dur.
            try:
                dev = str(e.device_type)
            except Exception:
                dev = ""
            if "CUDA" not in dev:
                continue
            nm = e.name
            events.append(
                {
                    "name": nm,
                    "ts": (
                        getattr(e, "time_range", None).start
                        if getattr(e, "time_range", None)
                        else None
                    ),
                    "dur": e.self_device_time_total or e.device_time_total,
                }
            )
        # The kineto event list above loses timestamps in some builds; also
        # export the chrome trace, which always has them.
        tr = args.trace_out or (args.out + ".trace.json")
        prof.export_chrome_trace(tr)
        res["trace"] = tr
        res["n_events_pyapi"] = len(events)
        print(f"[e41] trace -> {tr}", flush=True)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[e41] wrote {args.out}", flush=True)
    print(json.dumps({k: v for k, v in res.items() if k != "timings"}, indent=1))


if __name__ == "__main__":
    main()
