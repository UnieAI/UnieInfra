#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""One arm of an ours-vs-vLLM *quantized* comparison, in its own process.

Why offline (`LLM`) rather than `vllm bench serve`: the question here is the
GEMM, and the arms differ only inside `LinearMethodBase.apply`. An in-process
generate() loop removes the HTTP client, the tokenizer service and the arrival
process from the measurement, so a 2% GEMM effect is not buried under harness
noise. Prompts are built from **token ids**, not text, so every request has an
exactly equal input length and no two requests share a prefix (prefix caching
would otherwise collapse a C=128 batch into one sequence).

Each arm must run in its own process: vLLM captures decode CUDA graphs at engine
construction, so flipping a kernel afterwards replays the same graph and reports
a false ~0% (E33).

Environment:
    BENCH_MODEL           checkpoint path (required)
    BENCH_QUANT           --quantization value; empty for bf16
    BENCH_TAG             arm name, used in the log lines and the JSON
    BENCH_OUT             output JSON path
    BENCH_CONC            comma-separated concurrency levels (default 1,32,128)
    BENCH_OLEN/BENCH_ILEN output / input tokens per request
    BENCH_REPS            repeats per level; the median is reported
    BENCH_LINEAR_BACKEND  vLLM --linear-backend (sm_120 NVFP4 needs `cutlass`)
    BENCH_ATTN            vLLM --attention-backend (sm_120 fp8-KV needs TRITON_ATTN)
"""

from __future__ import annotations

import json
import os
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

model = os.environ["BENCH_MODEL"]
quant = os.environ.get("BENCH_QUANT") or None
tag = os.environ["BENCH_TAG"]
out = os.environ.get("BENCH_OUT", f"/tmp/bench_{tag}.json")
concs = [int(c) for c in os.environ.get("BENCH_CONC", "1,32,128").split(",")]
olen = int(os.environ.get("BENCH_OLEN", "256"))
ilen = int(os.environ.get("BENCH_ILEN", "512"))
reps = int(os.environ.get("BENCH_REPS", "3"))

kw = {}
if os.environ.get("BENCH_LINEAR_BACKEND"):
    kw["linear_backend"] = os.environ["BENCH_LINEAR_BACKEND"]
if os.environ.get("BENCH_ATTN"):
    kw["attention_backend"] = os.environ["BENCH_ATTN"]

# `BENCH_GPU_UTIL` exists only so a shared box can be measured at all: vLLM
# refuses to start when free memory is below `util * total`, and it derives the
# KV pool from `util * total` (not from what happens to be free), so lowering it
# keeps every arm's pool identical while surviving a co-resident process.
llm = LLM(
    model=model,
    quantization=quant,
    dtype="bfloat16",
    gpu_memory_utilization=float(os.environ.get("BENCH_GPU_UTIL", "0.80")),
    max_model_len=2048,
    enforce_eager=False,
    trust_remote_code=True,
    disable_log_stats=True,
    seed=0,
    **kw,
)

VOCAB_LO, VOCAB_HI = 1000, 60000


def make_prompt(i: int) -> TokensPrompt:
    """A deterministic, unique prompt of exactly ``ilen`` tokens."""
    ids = [
        VOCAB_LO + ((i * 9973 + j * 7919) % (VOCAB_HI - VOCAB_LO)) for j in range(ilen)
    ]
    return TokensPrompt(prompt_token_ids=ids)


sp = SamplingParams(
    temperature=0.0, max_tokens=olen, min_tokens=olen, ignore_eos=True, seed=0
)
res = {
    "tag": tag,
    "model": model,
    "quant": quant,
    "engine_kwargs": kw,
    "olen": olen,
    "ilen": ilen,
    "reps": reps,
    "runs": {},
}

llm.generate(
    [make_prompt(-1)],
    SamplingParams(temperature=0.0, max_tokens=32, min_tokens=32, ignore_eos=True),
    use_tqdm=False,
)

for c in concs:
    samples = []
    for r in range(reps):
        prompts = [make_prompt(c * 1000 + r * 100 + i) for i in range(c)]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        samples.append(
            {"elapsed_s": dt, "out_tokens": ntok, "out_tok_per_s": ntok / dt}
        )
        print(f"[{tag}] C={c} rep={r} {ntok / dt:.2f} tok/s", flush=True)
    tps = sorted(s["out_tok_per_s"] for s in samples)
    res["runs"][str(c)] = {
        "samples": samples,
        "median_out_tok_per_s": tps[len(tps) // 2],
        "min": tps[0],
        "max": tps[-1],
    }
    print(
        f"[{tag}] C={c} MEDIAN {tps[len(tps) // 2]:.2f} tok/s "
        f"[{tps[0]:.2f}-{tps[-1]:.2f}]",
        flush=True,
    )

sanity = llm.generate(
    [
        TokensPrompt(
            prompt_token_ids=llm.get_tokenizer().encode("The capital of France is")
        )
    ],
    SamplingParams(temperature=0.0, max_tokens=24),
    use_tqdm=False,
)
res["sanity"] = sanity[0].outputs[0].text
print(f"[{tag}] sanity: {res['sanity']!r}", flush=True)
with open(out, "w") as handle:
    json.dump(res, handle, indent=1)
print(f"[{tag}] wrote {out}", flush=True)
