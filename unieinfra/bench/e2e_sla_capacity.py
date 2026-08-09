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

"""How much traffic fits under a latency SLA -- open-loop, like a real service.

`e2e_concurrency.py` is closed-loop: N requests fire together, all complete, the
next round starts. That answers "how fast for N simultaneous users" and it is
the wrong model for capacity planning, because it can never build a queue --
load is capped by the client, not by arrivals.

A service does not work that way. Requests arrive on their own schedule, and
the question a capacity plan asks is the other one: **at what arrival rate does
p99 TTFT cross my SLA?** Below that point the queue drains as fast as it fills;
above it the queue grows without bound and latency runs away. The knee is sharp,
and a closed-loop benchmark cannot find it.

So this drives Poisson arrivals at a fixed RPS, holds it for a fixed duration,
and reports what the users saw. Sweep the rate until p99 TTFT crosses the SLA;
the last rate that holds is the answer.

    python3 unieinfra/bench/e2e_sla_capacity.py --rates 10,20,40,60 --sla-ms 1000
    BACKEND=vllm python3 unieinfra/bench/e2e_sla_capacity.py --rates 10,20,40,60

Arrivals are exponential inter-arrival times at the requested mean, not a fixed
cadence: evenly spaced arrivals understate queueing badly, and real traffic is
bursty. The generator holds its schedule regardless of whether the server keeps
up, so a rate the engine cannot sustain shows as a growing backlog rather than
as the client quietly slowing down.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import threading
import time

DEFAULT_MODEL = "/workspace/models/Qwen/Qwen3-8B"


def percentile(samples: list[float], q: float) -> float:
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


class Sample:
    __slots__ = ("ttft_ms", "total_ms", "n_in", "n_out")

    def __init__(self, ttft_ms, total_ms, n_in, n_out):
        self.ttft_ms = ttft_ms
        self.total_ms = total_ms
        self.n_in = n_in
        self.n_out = n_out

    @property
    def itl_ms(self) -> float:
        return (
            (self.total_ms - self.ttft_ms) / (self.n_out - 1)
            if self.n_out > 1
            else float("nan")
        )


def make_prompt(idx: int, prefix_words: int, shared: str, shape: str) -> str:
    if shape == "shared":
        return shared + " " + " ".join(f"q{idx}t{j}" for j in range(32))
    return " ".join(f"u{idx}w{j}" for j in range(prefix_words))


def run_tokenspeed(engine, args, shared) -> tuple[list[Sample], float, int]:
    """Poisson arrivals from a generator thread; one worker thread per request.

    Threads rather than the async API for the same reason as the concurrency
    sweep: the sync streaming iterator is the entry point this engine bridges
    onto its own loop, and driving its coroutines from a foreign loop adds a
    second of polling per request.
    """
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": args.out_tokens,
        "ignore_eos": True,
    }
    samples: list[Sample] = []
    lock = threading.Lock()
    inflight_peak = 0
    inflight = 0
    threads: list[threading.Thread] = []

    def one(idx: int):
        nonlocal inflight, inflight_peak
        with lock:
            inflight += 1
            inflight_peak = max(inflight_peak, inflight)
        prompt = make_prompt(idx, args.prefix_words, shared, args.shape)
        start = time.perf_counter()
        first = None
        n_in = n_out = 0
        for chunk in engine.generate(prompt, sampling, stream=True):
            if first is None:
                first = time.perf_counter()
            meta = chunk["meta_info"]
            n_out = meta.get("completion_tokens", n_out)
            n_in = meta.get("prompt_tokens", n_in)
        end = time.perf_counter()
        with lock:
            inflight -= 1
            samples.append(
                Sample(
                    (first - start) * 1e3 if first else float("nan"),
                    (end - start) * 1e3,
                    n_in,
                    max(n_out, 1),
                )
            )

    rng = random.Random(1234)
    t_start = time.perf_counter()
    deadline = t_start + args.duration
    idx = 0
    next_at = t_start
    while time.perf_counter() < deadline:
        next_at += rng.expovariate(args.rate)
        delay = next_at - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        t = threading.Thread(target=one, args=(idx,), daemon=True)
        t.start()
        threads.append(t)
        idx += 1
    issued = idx
    for t in threads:
        t.join(timeout=args.drain)
    return samples, time.perf_counter() - t_start, issued, inflight_peak


async def run_vllm(engine, args, shared):
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0, max_tokens=args.out_tokens, ignore_eos=True
    )
    samples: list[Sample] = []
    inflight = 0
    inflight_peak = 0

    async def one(idx: int):
        nonlocal inflight, inflight_peak
        inflight += 1
        inflight_peak = max(inflight_peak, inflight)
        prompt = make_prompt(idx, args.prefix_words, shared, args.shape)
        start = time.perf_counter()
        first = None
        n_in = n_out = 0
        async for out in engine.generate(prompt, sampling, f"r{idx}"):
            n_in = len(out.prompt_token_ids or ()) or n_in
            if out.outputs and out.outputs[0].token_ids:
                if first is None:
                    first = time.perf_counter()
                n_out = len(out.outputs[0].token_ids)
        end = time.perf_counter()
        inflight -= 1
        samples.append(
            Sample(
                (first - start) * 1e3 if first else float("nan"),
                (end - start) * 1e3,
                n_in,
                max(n_out, 1),
            )
        )

    rng = random.Random(1234)
    t_start = time.perf_counter()
    deadline = t_start + args.duration
    tasks = []
    idx = 0
    next_at = t_start
    while time.perf_counter() < deadline:
        next_at += rng.expovariate(args.rate)
        delay = next_at - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(idx)))
        idx += 1
    await asyncio.wait(tasks, timeout=args.drain)
    return samples, time.perf_counter() - t_start, idx, inflight_peak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", DEFAULT_MODEL))
    ap.add_argument("--rates", default="5,10,20,40,60,80", help="arrivals/sec to sweep")
    ap.add_argument("--sla-ms", type=float, default=1000.0)
    ap.add_argument("--duration", type=float, default=30.0, help="seconds per rate")
    ap.add_argument("--drain", type=float, default=120.0, help="seconds to finish")
    ap.add_argument("--out-tokens", type=int, default=64)
    ap.add_argument("--prefix-words", type=int, default=700)
    ap.add_argument("--shape", default="shared", choices=("shared", "unique"))
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    backend = os.environ.get("BACKEND", "tokenspeed")
    shared = " ".join(f"ctx{j}" for j in range(args.prefix_words))

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
        loop = asyncio.new_event_loop()
    else:
        from tokenspeed.runtime.entrypoints.engine import Engine

        engine = Engine(
            model=args.model, dtype="bfloat16", max_model_len=args.max_model_len
        )
        loop = None

    print(
        f"backend={backend} shape={args.shape} out={args.out_tokens} "
        f"prefix~{args.prefix_words} words  SLA=TTFT p99 < {args.sla_ms:.0f} ms"
    )
    print(
        f"  {'rate':>6} {'done':>6} {'in tok':>9} {'out tok':>8} {'in/req':>7} "
        f"{'out/req':>8} {'RPS':>7} {'peak':>5} {'TTFT p50':>9} {'TTFT p99':>9} "
        f"{'ITL p50':>8} {'ITL p99':>8} {'TPS':>8}  SLA"
    )

    try:
        for rate in [float(x) for x in args.rates.split(",")]:
            args.rate = rate
            if loop is None:
                samples, wall, issued, peak = run_tokenspeed(engine, args, shared)
            else:
                samples, wall, issued, peak = loop.run_until_complete(
                    run_vllm(engine, args, shared)
                )
            if not samples:
                print(f"  {rate:6.1f}  no completions in {wall:.0f}s")
                continue
            ttfts = [s.ttft_ms for s in samples]
            itls = [s.itl_ms for s in samples if s.n_out > 1]
            n_in = sum(s.n_in for s in samples)
            n_out = sum(s.n_out for s in samples)
            p99 = percentile(ttfts, 0.99)
            ok = "OK" if p99 < args.sla_ms else "FAIL"
            print(
                f"  {rate:6.1f} {len(samples):6d} {n_in:9d} {n_out:8d} "
                f"{n_in / len(samples):7.0f} {n_out / len(samples):8.0f} "
                f"{len(samples) / wall:7.1f} {peak:5d} "
                f"{percentile(ttfts, 0.5):9.1f} {p99:9.1f} "
                f"{percentile(itls, 0.5) if itls else float('nan'):8.2f} "
                f"{percentile(itls, 0.99) if itls else float('nan'):8.2f} "
                f"{n_out / wall:8.1f}  {ok}",
                flush=True,
            )
            if issued > len(samples):
                print(
                    f"         note: {issued - len(samples)} of {issued} did not "
                    f"finish within the drain window -- the queue was growing"
                )
    finally:
        if loop is not None:
            loop.close()
        elif backend != "vllm":
            engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
