# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Offline batch throughput, both engines, two prompt shapes.

Output tokens/sec over a batch submitted at once. Two shapes because they
stress different things: `unique` prompts force every request to prefill from
scratch, `shared` prompts give every request the same long prefix, which is
what RAG and agent traffic looks like and what a prefix cache is for.

Scored on the whole run rather than a converged tail: a rate counter can be
stable while a draining pipeline still climbs.
"""

import os
import time

BACKEND = os.environ["BACKEND"]
MODEL = os.environ.get("AB_MODEL", "/workspace/models/Qwen/Qwen3-8B")
N_REQ = int(os.environ.get("AB_N", "64"))
PROMPT_WORDS = int(os.environ.get("AB_PROMPT_WORDS", "700"))
OUT_TOKENS = int(os.environ.get("AB_OUT", "128"))

SHARED = " ".join(f"ctx{j}" for j in range(PROMPT_WORDS))


def prompt_sets():
    yield "unique", [
        " ".join(f"u{i}w{j}" for j in range(PROMPT_WORDS)) for i in range(N_REQ)
    ]
    yield "shared", [
        SHARED + " " + " ".join(f"q{i}t{j}" for j in range(32)) for i in range(N_REQ)
    ]


def main():
    if BACKEND == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=MODEL,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=float(os.environ.get("AB_GPU_UTIL", "0.88")),
        )
        sp = SamplingParams(temperature=0.0, max_tokens=OUT_TOKENS, ignore_eos=True)
        tok = llm.get_tokenizer()

        def run(prompts):
            outs = llm.generate(prompts, sp, use_tqdm=False)
            return sum(len(o.outputs[0].token_ids) for o in outs)

    else:
        from tokenspeed.runtime.entrypoints.engine import Engine

        kw = {}
        if os.environ.get("AB_BLOCK_SIZE"):
            kw["block_size"] = int(os.environ["AB_BLOCK_SIZE"])
        if os.environ.get("AB_NO_KVSTORE"):
            kw["disable_kvstore"] = True
        engine = Engine(model=MODEL, dtype="bfloat16", max_model_len=8192, **kw)
        tok = engine.tokenizer_manager.tokenizer

        def run(prompts):
            outs = engine.generate(
                prompts,
                {"temperature": 0.0, "max_new_tokens": OUT_TOKENS, "ignore_eos": True},
            )
            return sum(
                o["meta_info"].get("completion_tokens", OUT_TOKENS) for o in outs
            )

    print(f"backend={BACKEND} n={N_REQ} out={OUT_TOKENS}")
    only = os.environ.get("AB_ONLY")
    for name, prompts in prompt_sets():
        if only and name != only:
            continue
        run(prompts[:4])  # warm up, and populate the prefix for `shared`
        n_in = sum(len(tok.encode(p)) for p in prompts)
        t0 = time.perf_counter()
        n_out = run(prompts)
        dt = time.perf_counter() - t0
        print(
            f"  {name:7s} in={n_in:7d} out={n_out:6d}  {dt:7.2f} s  "
            f"output {n_out / dt:8.1f} tok/s  total {(n_in + n_out) / dt:8.1f} tok/s",
            flush=True,
        )

    if BACKEND != "vllm":
        engine.shutdown()


if __name__ == "__main__":
    main()
