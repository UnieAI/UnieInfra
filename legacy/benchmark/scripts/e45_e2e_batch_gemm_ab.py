#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""End-to-end arm of the decode-batch GEMM A/B: C=128 decode throughput (E45).

**One arm per process, on purpose.** Decode runs under CUDA graph and the graph
is captured during engine construction, so whichever GEMM is selected at capture
time is baked into the replay; toggling the kernel afterwards replays the same
graph and measures nothing (E33/E37). The arm is therefore chosen by
``UNIEINFRA_DECODE_BATCH_GEMM`` *before* the engine is built.

Both arms run with ``UNIEINFRA_GEMM=1``, i.e. through the identical dispatch
seam. The ``off`` arm leaves the kernel registered and selected and makes its
runtime guard delegate to ``F.linear`` — so the two arms differ in the kernel
*body* and in nothing else. That is a stricter comparison than "our stack vs
stock" and it is the one that isolates this kernel.

Steady-state decode only: every request is padded to the same length with
``ignore_eos`` and ``min_tokens == max_tokens``, prefill is timed separately and
subtracted, and the reported figure is decode tokens per second across the
whole batch.

Driver: ``run_e45_e2e_ab.sh``, which alternates A,B,A,B whole processes with
``VLLM_DISABLE_COMPILE_CACHE=1`` on every arm — vLLM's compile cache is keyed on
``VllmConfig``, not on env vars, so without it the second arm silently replays
the first arm's compiled graph (this bit E34).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


def build_prompts(n: int, ilen: int, vocab: int, seed: int = 0):
    """Synthetic prompt token ids: identical length, so the batch stays square."""
    import random

    rng = random.Random(seed)
    lo, hi = 1000, min(vocab, 100000) - 1
    return [[rng.randint(lo, hi) for _ in range(ilen)] for _ in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/Qwen/Qwen3-8B")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--ilen", type=int, default=1024)
    ap.add_argument("--olen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    arm = "off" if os.environ.get("UNIEINFRA_DECODE_BATCH_GEMM", "1") == "0" else "on"
    print(f"[e45-e2e] arm: decode_batch_gemm={arm} label={args.label}", flush=True)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.concurrency,
        enforce_eager=False,  # CUDA graph: the production decode path
        trust_remote_code=True,
        disable_log_stats=True,
        # OFF on purpose. The decode rate below is a difference of two timed
        # runs, and with prefix caching on, the second run over the same prompts
        # skips prefill entirely -- which made the "prefill + 1 token" reference
        # slower than the full generation and produced a negative rate.
        enable_prefix_caching=False,
    )
    print(f"[e45-e2e] engine up in {time.time() - t0:.1f}s", flush=True)

    _report_selection(args.concurrency)

    vocab = llm.llm_engine.vllm_config.model_config.get_vocab_size()

    def run(olen: int, seed: int) -> float:
        toks = build_prompts(args.concurrency, args.ilen, vocab, seed=seed)
        sp = SamplingParams(
            temperature=0.0, max_tokens=olen, min_tokens=olen, ignore_eos=True
        )
        t = time.perf_counter()
        llm.generate([{"prompt_token_ids": x} for x in toks], sp, use_tqdm=False)
        return time.perf_counter() - t

    # Warm up: CUDA-graph capture, inductor JIT, allocator, clocks.
    run(8, seed=99)

    samples = []
    totals = []
    seed = 1000
    for rep in range(args.reps):
        # Two figures, because they fail in different ways. ``total`` is the
        # aggregate output throughput of the whole batch -- what a user sees,
        # prefill included, and robust because it is one timed run. ``decode``
        # subtracts a prefill-only run to isolate the phase this kernel touches;
        # it is the sharper number but it is a difference of two measurements,
        # so it carries both their noise. Every run gets a fresh seed so no run
        # can reuse another's KV blocks.
        seed += 1
        t_short = run(1, seed=seed)
        seed += 1
        t_long = run(args.olen, seed=seed)
        total = args.concurrency * args.olen / t_long
        decode_tokens = args.concurrency * (args.olen - 1)
        rate = decode_tokens / max(t_long - t_short, 1e-6)
        samples.append(rate)
        totals.append(total)
        print(
            f"[e45-e2e] rep {rep}: prefill+1={t_short:.3f}s full={t_long:.3f}s "
            f"total={total:.1f} tok/s decode={rate:.1f} tok/s",
            flush=True,
        )

    median = statistics.median(samples)
    median_total = statistics.median(totals)
    print(
        f"[e45-e2e] RESULT arm={arm} label={args.label} "
        f"median={median:.2f} tok/s total={median_total:.2f} tok/s "
        f"all={[round(x, 2) for x in samples]}",
        flush=True,
    )
    if args.json:
        with open(args.json, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "arm": arm,
                        "label": args.label,
                        "concurrency": args.concurrency,
                        "ilen": args.ilen,
                        "olen": args.olen,
                        "median_tok_s": median,
                        "median_total_tok_s": median_total,
                        "samples": samples,
                        "totals": totals,
                    }
                )
                + "\n"
            )
    return 0


def _report_selection(concurrency: int) -> None:
    """Print which kernel ``gemm.mm`` picks for this step's decode shapes.

    Evidence, not inference: the arm is chosen by an env var the engine reads in
    a worker process, so without printing the resolved kernel here a null result
    is indistinguishable from "the flag never reached the worker".
    """
    try:
        import torch
        import unieinfra_kernel  # noqa: F401  (discovery)

        # Through the boundary, not through ``tokenspeed_kernel`` directly: the
        # engine container has no ``tokenspeed_kernel``, so importing it here
        # made the probe fail and left the arm unverified.
        from unieinfra_kernel.boundary import (
            dense_tensor_format,
            format_signature,
            select_kernel,
        )
        from unieinfra_kernel.dispatch import shape_traits
        from unieinfra_kernel.ops.gemm.decode_batch_gemm_triton import (
            select_batch_gemm_config,
        )

        sig = format_signature(
            a=dense_tensor_format(torch.bfloat16),
            b=dense_tensor_format(torch.bfloat16),
        )
        for label, (N, K) in (
            ("gate_up_proj", (24576, 4096)),
            ("down_proj", (4096, 12288)),
            ("qkv_proj", (6144, 4096)),
            ("o_proj", (4096, 4096)),
        ):
            name = select_kernel(
                "gemm", "mm", sig, traits=shape_traits(concurrency, N, K)
            ).name
            cfg = select_batch_gemm_config(concurrency, N, K)
            print(f"[select] {label} M={concurrency} -> {name}  {cfg}", flush=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never fail a run
        print(f"[select] unavailable: {exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
