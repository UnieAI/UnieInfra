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

"""Per-user latency as concurrency rises -- the curve between the two goals.

"Each user as fast as possible" and "serve as many users as possible" are the
same measurement taken at two ends of one axis, and the interesting part is in
between: how far concurrency can go before a user notices. A single-request
latency number and an offline batch-throughput number both miss it, because
neither reports what a user waits for while others are being served.

So this reports, per concurrency level:

  in tok / out tok   total prompt and completion tokens actually processed
  in/req  / out/req  average length of each, so a row is reproducible
  RPS                completed requests per second
  TTFT p50 / p99     what a user waits for the first token, median and tail
  ITL p50 / p99      what a user waits between streamed tokens
  TPS                output tokens per second, aggregate

The knee in TTFT p99 is the answer to "how many users can this serve"; TPS is
the answer to "how much can this machine do".

Requests are issued concurrently and each is timed from its own submission, not
from the start of the batch: a request that queues for 400 ms has a 400 ms TTFT
whatever the batch total says.

    python3 unieinfra/bench/e2e_concurrency.py                 # this engine
    BACKEND=vllm python3 unieinfra/bench/e2e_concurrency.py    # the reference

Shapes follow e2e_shapes.py: `unique` gives every request its own prompt,
`shared` gives them a common prefix with a short tail, which is what RAG and
agent traffic looks like and what a prefix cache is for.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MODEL = "/workspace/models/Qwen/Qwen3-8B"


def percentile(samples: list[float], q: float) -> float:
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def build_prompts(
    shape: str, n: int, prefix_words: int, round_id: int = 0
) -> list[str]:
    """Prompts for one round.

    `shared` deliberately gives every request the same long prefix and a short
    distinct tail, in every round: that is what RAG and agent traffic is, and
    the prefix cache is meant to serve it. Its TTFT is a cache-hit TTFT by
    construction, which is the point rather than an accident.

    `unique` has to vary per round as well as per request. Building the list
    once and replaying it for every repeat turns rounds 2..N into cache hits,
    and with the first round discarded as warm-up that leaves a "cold"
    benchmark measuring nothing but the cache -- 2840 tokens read 18.5 ms where
    a genuine cold prefill of that length costs about 130 ms.
    """
    if shape == "shared":
        shared = " ".join(f"ctx{j}" for j in range(prefix_words))
        return [
            shared + " " + " ".join(f"q{i}t{j}" for j in range(32)) for i in range(n)
        ]
    return [
        " ".join(f"r{round_id}u{i}w{j}" for j in range(prefix_words)) for i in range(n)
    ]


class _Result:
    __slots__ = ("ttft_ms", "total_ms", "n_out", "n_in")

    def __init__(self, ttft_ms: float, total_ms: float, n_out: int, n_in: int = 0):
        self.ttft_ms = ttft_ms
        self.total_ms = total_ms
        self.n_out = n_out
        self.n_in = n_in

    @property
    def itl_ms(self) -> float:
        """Inter-token latency: what a user sees between streamed tokens."""
        return (
            (self.total_ms - self.ttft_ms) / (self.n_out - 1)
            if self.n_out > 1
            else float("nan")
        )


def _run_tokenspeed(engine, prompts, out_tokens) -> list[_Result]:
    """One thread per request against the sync streaming iterator.

    Not `async_generate` from a locally created event loop: the engine bridges
    its queues onto its own background loop with run_coroutine_threadsafe, and
    driving the coroutine from a foreign loop adds about a second of polling
    per request -- measured as a fixed ~1000 ms that does not scale with output
    length, while the marginal cost per token stayed at the true 11.4 ms.
    `generate(stream=True)` is the sync entry point and goes through that
    bridge the way it was built to be used.
    """
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": out_tokens,
        "ignore_eos": True,
    }

    def one(prompt: str) -> _Result:
        start = time.perf_counter()
        first = None
        n_out = 0
        n_in = 0
        for chunk in engine.generate(prompt, sampling, stream=True):
            if first is None:
                first = time.perf_counter()
            n_out = chunk["meta_info"].get("completion_tokens", n_out)
            n_in = chunk["meta_info"].get("prompt_tokens", n_in)
        end = time.perf_counter()
        return _Result(
            (first - start) * 1e3 if first else float("nan"),
            (end - start) * 1e3,
            max(n_out, 1),
            n_in,
        )

    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        return list(pool.map(one, prompts))


async def _run_vllm(engine, prompts, out_tokens) -> list[_Result]:
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True)

    async def one(idx: int, prompt: str) -> _Result:
        start = time.perf_counter()
        first = None
        n_out = 0
        n_in = 0
        async for out in engine.generate(prompt, sampling, f"req{idx}"):
            n_in = len(out.prompt_token_ids or ()) or n_in
            if first is None and out.outputs and out.outputs[0].token_ids:
                first = time.perf_counter()
            if out.outputs:
                n_out = len(out.outputs[0].token_ids)
        end = time.perf_counter()
        return _Result(
            (first - start) * 1e3 if first else float("nan"),
            (end - start) * 1e3,
            max(n_out, 1),
            n_in,
        )

    return list(await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", DEFAULT_MODEL))
    ap.add_argument(
        "--concurrency",
        default="1,4,16,64,128",
        help="comma-separated levels to sweep",
    )
    ap.add_argument("--out-tokens", type=int, default=64)
    ap.add_argument(
        "--repeats",
        type=int,
        default=5,
        help=(
            "rounds per concurrency level, pooled into the percentiles. "
            "Without this a level of 1 has a single sample and its p50 is "
            "whatever that one request happened to cost -- which read 33 ms "
            "against a true median of 18.5 the first time this was run."
        ),
    )
    ap.add_argument("--prefix-words", type=int, default=700)
    ap.add_argument("--shape", default="shared", choices=("shared", "unique"))
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    backend = os.environ.get("BACKEND", "tokenspeed")
    levels = [int(x) for x in args.concurrency.split(",")]

    if backend == "vllm":
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=args.model,
                dtype="bfloat16",
                max_model_len=args.max_model_len,
                gpu_memory_utilization=float(os.environ.get("BENCH_GPU_UTIL", "0.88")),
            )
        )
        runner = _run_vllm
    else:
        from tokenspeed.runtime.entrypoints.engine import Engine

        engine = Engine(
            model=args.model,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
        )
        runner = _run_tokenspeed

    print(
        f"backend={backend} shape={args.shape} out={args.out_tokens} "
        f"prefix~{args.prefix_words} words"
    )
    print(
        f"  {'conc':>5} {'in tok':>9} {'out tok':>8} {'in/req':>7} "
        f"{'out/req':>8} {'RPS':>7} {'TTFT p50':>9} {'TTFT p99':>9} "
        f"{'ITL p50':>8} {'ITL p99':>8} {'TPS':>8}"
    )

    loop = asyncio.new_event_loop() if backend == "vllm" else None

    def run(prompts, out_tokens):
        if loop is None:
            return runner(engine, prompts, out_tokens)
        return loop.run_until_complete(runner(engine, prompts, out_tokens))

    try:
        for conc in levels:
            prompts = build_prompts(args.shape, conc, args.prefix_words, 0)
            # Warm with a full round at this level, not one short request.
            # A 1-request 4-token warm-up leaves the first real round carrying
            # thread-pool and first-touch cost, and with a handful of repeats
            # that single round *is* the p99: it read 33.5 ms at concurrency 1
            # where 30 sequential requests measure 18.4-18.9, a 1.02x spread.
            # The engine has no jitter there; the harness did.
            run(prompts, args.out_tokens)

            results: list[_Result] = []
            produced = 0
            wall = 0.0
            for round_id in range(1, args.repeats + 1):
                # Fresh prompts per round for `unique`; `shared` returns
                # the same set every time on purpose.
                prompts = build_prompts(args.shape, conc, args.prefix_words, round_id)
                t0 = time.perf_counter()
                round_results = run(prompts, args.out_tokens)
                wall += time.perf_counter() - t0
                results += round_results
                produced += sum(r.n_out for r in round_results)

            ttfts = [r.ttft_ms for r in results]
            itls = [r.itl_ms for r in results if r.n_out > 1]
            consumed = sum(r.n_in for r in results)
            n_req = len(results)
            print(
                f"  {conc:5d} {consumed:9d} {produced:8d} "
                f"{consumed / n_req:7.0f} {produced / n_req:8.0f} "
                f"{n_req / wall:7.1f} "
                f"{percentile(ttfts, 0.5):9.2f} "
                f"{percentile(ttfts, 0.99):9.2f} "
                f"{percentile(itls, 0.5) if itls else float('nan'):8.2f} "
                f"{percentile(itls, 0.99) if itls else float('nan'):8.2f} "
                f"{produced / wall:8.1f}",
                flush=True,
            )
    finally:
        if loop is not None:
            loop.close()
        if backend != "vllm":
            engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
