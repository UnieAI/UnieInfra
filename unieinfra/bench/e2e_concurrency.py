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
    if shape in ("shared", "rag"):
        shared = " ".join(f"ctx{j}" for j in range(prefix_words))
        if shape == "rag":
            # A new question against the same document every round. `shared`
            # replays the identical prompt and therefore measures an exact
            # repeat; this is the shape RAG and agent traffic actually has, and
            # the one that needs a recurrent snapshot above the divergence
            # point rather than at the end of the previous request.
            return [
                shared + " " + " ".join(f"r{round_id}q{i}t{j}" for j in range(32))
                for i in range(n)
            ]
        return [
            shared + " " + " ".join(f"q{i}t{j}" for j in range(32)) for i in range(n)
        ]
    # A short distinct salt in front of a common body. A radix prefix cache keys
    # on the prefix, so differing at the first token defeats it entirely, while
    # the body keeps the token count equal to the `shared` shape. Built the
    # other way -- a wholly different vocabulary per request -- the prompts
    # tokenised to 5490 tokens against shared's 2840 and the shapes stopped
    # being comparable.
    body = " ".join(f"ctx{j}" for j in range(prefix_words))
    return [f"salt{round_id}n{i} " + body for i in range(n)]


def describe_engine(backend: str, engine, args) -> dict:
    """The resolved configuration of whichever engine is running.

    Resolved, not requested. Both engines silently override some arguments, and
    a comparison is only a comparison if the two columns can be read against
    each other -- so this reports the same key names for both.
    """
    cfg = {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "out_tokens": args.out_tokens,
        "prefix_words": args.prefix_words,
        "repeats": args.repeats,
        "sampling": "temperature=0.0, ignore_eos=True",
    }
    if backend == "vllm":
        vc = getattr(engine, "vllm_config", None)
        cache = getattr(vc, "cache_config", None)
        sched = getattr(vc, "scheduler_config", None)
        cfg["prefix_caching"] = getattr(cache, "enable_prefix_caching", "?")
        cfg["kv_cache_dtype"] = getattr(cache, "cache_dtype", "?")
        cfg["block_size"] = getattr(cache, "block_size", "?")
        cfg["max_num_seqs"] = getattr(sched, "max_num_seqs", "?")
        cfg["chunked_prefill"] = getattr(sched, "enable_chunked_prefill", "?")
    else:
        sa = getattr(engine, "server_args", None)
        for key, attr in (
            ("prefix_caching", "enable_prefix_caching"),
            ("kv_cache_dtype", "kv_cache_dtype"),
            ("block_size", "block_size"),
            ("max_num_seqs", "max_num_seqs"),
            ("chunked_prefill", "chunked_prefill_size"),
            ("mixed_batch", "enable_mixed_batch"),
            ("mamba_ckpt_slots", "int8_mamba_ckpt_size"),
        ):
            cfg[key] = getattr(sa, attr, "?") if sa is not None else "?"
    return cfg


def check_output_text(texts: list[str]) -> str:
    """One line on whether the generated text is usable, for the report.

    A latency number says nothing about whether the engine produced language.
    These runs set ignore_eos so every request emits exactly out_tokens, which
    is right for timing and is also the condition under which a broken decode
    path degenerates quietly -- into one repeated token, or into replacement
    characters from a detokenizer handed the wrong ids.
    """
    if not texts:
        return "no text captured"
    sample = next((t for t in texts if t), "")
    if not sample:
        return "EMPTY -- every completion was blank"
    bad = sample.count("\ufffd")
    words = sample.split()
    uniq = len(set(words))
    verdict = "ok"
    if bad:
        verdict = f"MOJIBAKE -- {bad} replacement chars"
    elif uniq <= 1 and len(words) > 4:
        verdict = "DEGENERATE -- one token repeated"
    preview = sample[:80].replace("\n", " ")
    return f"{verdict} | uniq_words={uniq}/{len(words)} | {preview!r}"


class _Result:
    __slots__ = ("ttft_ms", "total_ms", "n_out", "n_in", "text")

    def __init__(
        self,
        ttft_ms: float,
        total_ms: float,
        n_out: int,
        n_in: int = 0,
        text: str = "",
    ):
        self.ttft_ms = ttft_ms
        self.total_ms = total_ms
        self.n_out = n_out
        self.n_in = n_in
        # Kept so the run can be checked for mojibake and degenerate repetition;
        # a timing number alone cannot tell a working engine from a broken one.
        self.text = text

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
        text = ""
        for chunk in engine.generate(prompt, sampling, stream=True):
            if first is None:
                first = time.perf_counter()
            n_out = chunk["meta_info"].get("completion_tokens", n_out)
            n_in = chunk["meta_info"].get("prompt_tokens", n_in)
            text = chunk.get("text", text)
        end = time.perf_counter()
        return _Result(
            (first - start) * 1e3 if first else float("nan"),
            (end - start) * 1e3,
            max(n_out, 1),
            n_in,
            text,
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
        text = ""
        async for out in engine.generate(prompt, sampling, f"req{idx}"):
            n_in = len(out.prompt_token_ids or ()) or n_in
            if first is None and out.outputs and out.outputs[0].token_ids:
                first = time.perf_counter()
            if out.outputs:
                n_out = len(out.outputs[0].token_ids)
                text = out.outputs[0].text or text
        end = time.perf_counter()
        return _Result(
            (first - start) * 1e3 if first else float("nan"),
            (end - start) * 1e3,
            max(n_out, 1),
            n_in,
            text,
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
    # The engine-side cap on simultaneously running sequences. Left at its
    # default, a sweep past it measures the queue rather than the engine: the
    # client may open 512 connections while only max_num_seqs of them are ever
    # resident, so both engines have to be told the same number for a row above
    # the default to mean anything.
    ap.add_argument("--max-num-seqs", type=int, default=None)
    # Both engines default to `auto`, i.e. the activation dtype. That is a fair
    # comparison but it leaves fp8 KV switched off, and fp8 KV is measured at
    # +19.0% throughput and 1.92x cache capacity here -- capacity being exactly
    # what saturates first on the unique shape. A row taken with this set is a
    # different configuration from one taken without, not a better one, so the
    # two must not be read side by side.
    ap.add_argument("--kv-cache-dtype", default=None)
    # Mixed prefill+decode in one step. Off by default in the engine, and the
    # hybrid models are exactly where it has never run.
    ap.add_argument("--enable-mixed-batch", action="store_true")
    ap.add_argument("--shape", default="shared", choices=("shared", "unique", "rag"))
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    backend = os.environ.get("BACKEND", "tokenspeed")
    levels = [int(x) for x in args.concurrency.split(",")]

    if backend == "vllm":
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        # Prefix caching must be asked for explicitly. Qwen3.5/3.6 hybrids
        # declare IsHybrid without MambaPrefixCaching, so vLLM resolves
        # enable_prefix_caching=False for them -- with no warning. Leaving it at
        # the default measures the reference engine with its cache switched off,
        # which is not a comparison. Dense Qwen3 already resolves True, so
        # passing it is a no-op there.
        engine_args = AsyncEngineArgs(
            model=args.model,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=float(os.environ.get("BENCH_GPU_UTIL", "0.88")),
            enable_prefix_caching=True,
            **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
            **({"kv_cache_dtype": args.kv_cache_dtype} if args.kv_cache_dtype else {}),
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        # The *resolved* value, not the requested one: a model that does not
        # support the feature has it turned back off here, silently, and asking
        # engine_args would report the request rather than what is running.
        cfg = getattr(engine, "vllm_config", None)
        resolved = getattr(
            getattr(cfg, "cache_config", None), "enable_prefix_caching", "?"
        )
        print(f"  vllm enable_prefix_caching={resolved} (requested True)")
        runner = _run_vllm
    else:
        from tokenspeed.runtime.entrypoints.engine import Engine

        engine = Engine(
            model=args.model,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
            **({"kv_cache_dtype": args.kv_cache_dtype} if args.kv_cache_dtype else {}),
            **({"enable_mixed_batch": True} if args.enable_mixed_batch else {}),
        )
        runner = _run_tokenspeed

    # Every knob that differs between the two engines, printed from the engine
    # rather than from the argv it was asked for: a model that does not support
    # a feature has it turned back off silently (vLLM does exactly that with
    # prefix caching on hybrids), so a report quoting the request describes a
    # run that did not happen.
    print(f"backend={backend} shape={args.shape}")
    for key, val in sorted(describe_engine(backend, engine, args).items()):
        print(f"  cfg {key} = {val}")
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
            # Printed per level rather than once: a decode path can be fine at
            # low concurrency and degenerate under batching, which is exactly
            # the regime these rows are about.
            print(
                f"        text: {check_output_text([r.text for r in results])}",
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
