# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""TTFT across the workload shapes that actually occur, both engines.

A unique random prompt per request -- the shape the throughput harness uses, to
keep prefix reuse from contaminating the numbers -- is precisely the one shape
in which a completely dead prefix cache costs nothing. RAG and agent traffic is
the opposite: a long shared prefix (system prompt + retrieved documents +
conversation so far) with a short varying tail, re-sent every turn.

Shapes:
  cold    unique prompt every time            -- no reuse possible
  repeat  the same prompt again               -- 100% prefix hit
  rag     shared prefix + unique 64-word tail -- partial hit, the RAG shape
  turn    conversation grown by 128 each time -- the agent/multi-turn shape

Everything runs inside main(): the engine spawns subprocesses that re-import
this module, so an engine built at module scope recurses.
"""

import os
import statistics
import time

BACKEND = os.environ["BACKEND"]
MODEL = os.environ.get("AB_MODEL", "/workspace/models/Qwen/Qwen3-8B")
PREFIX_WORDS = int(os.environ.get("AB_PREFIX", "1000"))
REPEATS = int(os.environ.get("AB_REPEATS", "6"))

SHARED = " ".join(f"ctx{j}" for j in range(PREFIX_WORDS))


def shapes():
    yield "cold", [
        f"unique{i} " + " ".join(f"u{i}w{j}" for j in range(PREFIX_WORDS))
        for i in range(REPEATS)
    ]
    yield "repeat", [SHARED for _ in range(REPEATS)]
    yield "rag", [
        SHARED + " " + " ".join(f"q{i}t{j}" for j in range(64)) for i in range(REPEATS)
    ]
    turns, convo = [], SHARED
    for i in range(REPEATS):
        convo = convo + " " + " ".join(f"turn{i}w{j}" for j in range(128))
        turns.append(convo)
    yield "turn", turns


def main():
    if BACKEND == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=MODEL,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=float(os.environ.get("AB_GPU_UTIL", "0.88")),
        )
        sp = SamplingParams(temperature=0.0, max_tokens=1)
        tok = llm.get_tokenizer()

        def ttft(prompt):
            t0 = time.perf_counter()
            llm.generate([prompt], sp, use_tqdm=False)
            return (time.perf_counter() - t0) * 1e3

    else:
        from tokenspeed.runtime.entrypoints.engine import Engine

        engine = Engine(model=MODEL, dtype="bfloat16", max_model_len=8192)
        tok = engine.tokenizer_manager.tokenizer

        def ttft(prompt):
            t0 = time.perf_counter()
            engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": 1})
            return (time.perf_counter() - t0) * 1e3

    print(f"backend={BACKEND} prefix~{PREFIX_WORDS} words repeats={REPEATS}")
    for name, prompts in shapes():
        # One untimed pass so the cache is populated exactly as the shape
        # intends: repeat/rag/turn are about the *second* time through.
        ttft(prompts[0])
        times = [ttft(p) for p in prompts]
        tail = times[1:] if name != "cold" else times
        print(
            f"  {name:7s} tokens={len(tok.encode(prompts[-1])):6d}  "
            f"median {statistics.median(tail):8.2f} ms  "
            f"min {min(tail):8.2f}  all {[round(t) for t in times]}",
            flush=True,
        )

    if BACKEND != "vllm":
        engine.shutdown()


if __name__ == "__main__":
    main()
