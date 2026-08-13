# Qwen3.5-27B against vLLM 0.26

Measured 2026-08-13 on one RTX PRO 6000 Blackwell (sm_120, HBM peak 1792 GB/s),
both engines driven by the same client over HTTP, a cold server per arm.

![concurrency ladder](assets/qwen35-27b-ladder.svg)

## The finding in one line

**The advantage is prefix reuse.** Where prompts share a prefix we lead by up to
49% on throughput and 4.3x on time-to-first-token; where they do not, the two
engines are within 3% of each other up to concurrency 64, and we lose 13% at 128.

That is a narrower claim than "faster", and it is the one the measurements
support. It also says where the remaining work is.

## How it was measured

| | |
|---|---|
| client | `vllm bench serve --backend openai-chat --endpoint /v1/chat/completions` — vLLM's own harness, **the same binary against both engines**, over HTTP |
| request shape | `--sharegpt-output-len 64 --num-prompts N --max-concurrency C --ignore-eos --seed 1`, `N = max(32, 2C)` |
| ours | `tokenspeed.cli serve --max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-seqs 452 --enable-mixed-batch` |
| vLLM | `vllm serve --max-model-len 8192 --gpu-memory-utilization 0.80 --max-num-seqs 452 --enable-prefix-caching` |
| model | Qwen3.5-27B, bf16 (64 layers: 48 gated-delta-net, 16 full attention) |

`--enable-prefix-caching` is passed to vLLM explicitly and its resolved value
checked in the running config: on Qwen3.5 hybrids it otherwise resolves to
`False` and says nothing, and a comparison run at vLLM's defaults measures vLLM
with its cache switched off.

In-process benchmarking was not used. This repository's own
`unieinfra/bench/e2e_concurrency.py` reported us 1.5–2.6x ahead on these shapes;
the same models under `vllm bench serve` reported the opposite. The in-process
numbers were honest about their conditions, and those conditions were not
serving.

## Two workload shapes

**Shared prefix (`rag`)** — 16 documents, several questions each. 90.8% of every
prompt's characters are shared with an earlier prompt. Prompts are 606–1015
tokens. For the 64 prompts actually sampled, an oracle with an infinite cache
could reuse 84.1% of all prompt tokens if requests arrived strictly in sequence,
74.4% given the prefill batching this engine actually performs, and 37.6% if all
32 concurrent requests were blind to one another.

**Unique prompts (`chat`)** — the same client and parameters against a corpus
with 0.0% shared prefix by construction. This is the shape where a prefix cache
is worth nothing, so whatever margin survives here is the engine.

## Shared prefix

| C | tok/s ours / vLLM | | median TTFT ours / vLLM | | mean ITL ours / vLLM |
|---:|---|---|---|---|---|
| 1 | 26.7 / 25.9 | +2.9% | 104 / 136 ms | 1.31x | 35.0 / 36.5 ms |
| 16 | 262.5 / 235.3 | **+11.6%** | 737 / 1475 ms | **2.00x** | 43.2 / 45.6 ms |
| 32 | 377.7 / 361.8 | +4.4% | 1525 / 2035 ms | 1.33x | 51.2 / 53.9 ms |
| 64 | 558.8 / 498.9 | **+12.0%** | 779 / 2602 ms | **3.34x** | 70.5 / 79.8 ms |
| 128 | 724.0 / 484.2 | **+49.5%** | 1122 / 4844 ms | **4.32x** | 108.0 / 148.4 ms |

C=32 and C=128 are the mean of three runs. Repeat spread is 7.3% for us and 0.7%
for vLLM, so **the +2.9% at C=1 and +4.4% at C=32 are inside the noise and are
not claims**; C=16, 64 and 128 are outside it.

The lead is reuse, measured on both sides rather than inferred:

| C | prefix reuse ours | vLLM |
|---:|---|---|
| 1 | 72.1% | 19.2% |
| 16 | 41.7% | 19.2% |
| 32 | 38.8% | 35.7% |
| 64 | 50.3% | 43.7% |
| 128 | **62.8%** | 40.2% |

(Ours is aggregated from the engine's own `#cached-token` / `#new-token` per
prefill batch; vLLM's is `vllm:prefix_cache_hits_total / queries_total`. Two
instruments for one comparison — our Prometheus exporter binds a random port
unless pinned, and the pin does not reach this entry point. That is a defect in
the measurement, listed below.)

## Unique prompts

| C | tok/s ours / vLLM | | mean ITL ours / vLLM |
|---:|---|---|---|
| 1 | 25.9 / 25.4 | +2.0% | 35.0 / 36.5 ms |
| 16 | 213.4 / 211.1 | +1.1% | 46.3 / 46.3 ms |
| 32 | 297.1 / 291.2 | +2.0% | 59.6 / 60.8 ms |
| 64 | 360.8 / 350.5 | +2.9% | 95.6 / 106.8 ms |
| 128 | **314.6 / 362.1** | **−13.1%** | 263.6 / 182.9 ms |

Parity to concurrency 64, then a collapse at 128 — inter-token latency 44% worse
than vLLM's. Without a cache to draw on, the engine degrades under saturation
where vLLM does not. This is the clearest single weakness the ladder exposes, and
it is not yet attributed.

## Against the roofline

Comparing only against another engine cannot say when a kernel is finished. A
decode step must read the model's weights once, so `weights / HBM peak` is a
floor that does not move:

| | roofline | measured | achieved |
|---|---|---|---|
| Qwen3.5-27B decode step, C=1 | 31.0 ms (55.6 GB) | 35.0 ms | **88%** |
| Qwen3.5-27B decode step, C=32 | 31.0 ms | 51.0 ms | 63% |
| prefix reuse, shared prefix at C=32 | 74.4% | 38.8% | **52%** |

At C=1 decode is close to the physical limit; the C=32 figure is lower because
prefill interleaves with decode in a continuous batch, and both engines pay it.
Reuse is the number with room in it — half of what this workload's own batching
structure allows.

## Known defects, and what they cost this report

**A cached answer is not bit-identical to a cold one, and that is expected.**
A prefix hit and a cold prefill differ in the first generated token's logprob by
0.04-0.07 at some prompt lengths, enough to flip the argmax and return different
text. This was initially diagnosed as a precision defect in the recurrent
snapshot: the state pool is bf16 while a cold prefill carries its state in an
fp32 accumulator, and round-tripping through bf16 moves the state by 2.3e-03 in
an isolated kernel probe. That diagnosis is **wrong**, and the control that
retires it is a dense model. Qwen3-0.6B has no recurrent layer to snapshot at
all, and shows a perturbation of the same relative magnitude. Storing the
snapshot in fp32 -- or running the entire recurrent path in fp32 -- changes the
end-to-end numbers not at all, to six decimal places. What remains is what any
prefix cache does: attention over restored KV, and a different token count per
forward changing the reduction order inside the GEMMs. vLLM's cache carries the
same property. The isolated probe measured a real mechanism that is not the
cause of the observed effect.

**The unique-prompt collapse at C=128 is over-admission.** Demonstrated on
gemma-4-31B, where the same symptom is louder: the KV pool holds about 30
sequences and the engine accepts `--max-num-seqs 452`, so it admits far more work
than can be resident and every request stalls together instead of a few being
preempted. Lowering the limit to what the pool actually holds, with no code
change, moved that model from 219.6 to 340.4 tok/s (+55%) and its P99 ITL from
1905 to 155 ms — and took its median ITL below vLLM's. vLLM refuses to start in
this situation and names the number to use; we accept it and stall at runtime.
The same mechanism is the likely explanation here and the fix is a startup guard,
not a kernel.

**One dataset per shape, and one output length.** Every request emits exactly 64
tokens under `--ignore-eos`, which weights the comparison toward
prefill-dominated behaviour. Long-output and multi-turn shapes are unmeasured.

**Our reuse counter is not reachable.** `--enable-metrics` is on by default and
`tokenspeed:prefix_cache_hits / prompt_tokens` is the same ratio vLLM publishes,
but the smg Prometheus exporter binds a random free port and
`TOKENSPEED_SMG_PROMETHEUS_PORT` does not reach the `tokenspeed.cli serve` entry
point, so the numbers above come from a log aggregation on one side and a counter
on the other.

## Reproducing

The harness is `.cq.sh` in the workspace root (not in this repository): it starts
a cold server per arm, derives its port from the run tag so parallel runs cannot
answer each other's requests, records branch, HEAD and the md5 of the installed
scheduler extension before each arm, and kills only the engine tree it started.

One thing it cannot do for you: the scheduler extension is chosen per model from
the checkpoint's `architectures` (see `flat_kvcache_select.py`). Qwen loads the
radix build. Running these arms against the flat build instead drops throughput
to 361.8 tok/s at C=64 and takes prefix reuse to exactly zero — the state group's
page grain lifts the global block size to 400, prompts of ~830 tokens end
mid-page, and no snapshot is ever registered.
