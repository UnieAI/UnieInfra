#!/usr/bin/env python3
"""Minimal decode workload for ncu to attach to.

Kept separate from prof_mtp.py because ncu serialises and replays every kernel
it profiles, several times, to collect its metric sets. Anything not needed to
reach the MoE kernels is pure cost: this loads the model, does one short
generate to get past JIT and warmup, and then runs a handful of decode steps for
ncu to sample.

enforce_eager=True: vLLM captures decode into CUDA graphs, and attributing
kernels inside a captured graph is unreliable. Eager makes each launch visible
to the profiler at the price of a different schedule -- so what comes out
describes per-kernel efficiency, not end-to-end speed.

No speculative config: speculation changes the batch shape and would confound
the reading.

No `quantization="unieinfra"` either, unlike the torch-profiler runs -- the
profiling container resolves vLLM from the editable tree but not
`unieinfra_kernel`, so that method is unregistered here.

*** THESE TWO CONCESSIONS INVALIDATED THE RUN THEY WERE MADE FOR. ***

An earlier version of this docstring argued they were harmless, on the grounds
that the kernels under the lens are vLLM/TRT-LLM's own MoE path rather than
UnieInfra's dispatch. That reasoning was wrong. The dispatch decides WHICH MoE
path executes:

  without it   MoeFCGemm + expandInputRows + finalizeMoeRouting
  with it      fused_moe::run_global

So this run profiled a path that deployment does not take. It ranked
expandInputRows and finalizeMoeRouting as the dominant kernels; in the
deployment configuration they hold 1.0% and 2.2% of GPU time, while
fused_moe::run_global -- which was in the -k regex and never appeared in a
sampled launch, because that path was not taken -- holds 20.8%.

The per-kernel efficiency numbers this produces are still true about those
kernels. They are simply about the wrong kernels.

Before reusing this file: make `unieinfra_kernel` importable in the profiling
container and get ncu working against CUDA graphs, so the configuration
profiled is the configuration being optimised. A note in a docstring did not
stop the bad reading; only measuring the right thing will.
"""

import os

MODEL = os.environ.get("PROF_MODEL", "/workspace/models/Qwen/Qwen3.6-35B-A3B")

from vllm import LLM, SamplingParams  # noqa: E402


def main():
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    prompt = "The history of computing is long and detailed. " * 8
    # Warm-up outside the sampled window; --launch-skip is set past this.
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=4,
                                          min_tokens=4, ignore_eos=True))
    print("[ncu] warmup done", flush=True)
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=16,
                                                min_tokens=16, ignore_eos=True))
    print(f"[ncu] {len(out[0].outputs[0].token_ids)} tokens", flush=True)


if __name__ == "__main__":
    main()
