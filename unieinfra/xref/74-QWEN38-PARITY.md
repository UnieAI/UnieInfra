# Qwen3.8-27B: three-arm parity, TP=1

Qwen/Qwen3.8-27B, one H200 per arm, `TP=1`. 256 requests per cell, 128 generated
tokens, greedy, `ignore_eos`, `/v1/completions` with the corpus text verbatim,
fresh boot per concurrency point (warmup corpus shares no prefix with the
measured one). All three arms: `--max-num-seqs 80`, 32768 context, chunked
prefill 8192. No speculation, no hierarchical cache in any arm.

Driver: `bench/agent-slo/agentx_sweep.sh` with `PROFILE=qwen38`,
`ab_real/agentx-sweep/run_qwen38_20261019.sh`, table
`ab_real/agentx-sweep/parity_table.py <dir> q38`. Every cell: 256/256 requests
ok, 128.0 tok/req in all three arms, so the comparability gate passes and these
are throughput ratios, not artifacts of different output lengths.

## Throughput (tok/s total)

| conc | ours | sglang | vLLM | ours/sglang | ours/vLLM |
|-----:|-----:|-------:|-----:|------------:|----------:|
| 1 | 64.3 | 64.5 | 60.9 | 1.00x | 1.06x |
| 4 | 223.4 | 221.5 | 203.6 | 1.01x | 1.10x |
| 8 | 380.1 | 369.5 | 335.6 | 1.03x | 1.13x |
| 16 | 583.6 | 568.5 | 498.2 | 1.03x | 1.17x |
| 32 | 683.7 | 759.7 | 680.4 | 0.90x | 1.00x |
| 64 | 591.3 | 805.1 | 822.7 | 0.73x | 0.72x |

## TTFT (ms, p50 / p95)

| conc | ours | sglang | vLLM |
|-----:|------|--------|------|
| 1 | 70 / 208 | 98 / 227 | 170 / 268 |
| 16 | 754 / 2268 | 912 / 2808 | 1105 / 2770 |
| 32 | 2238 / 9081 | 2076 / 5506 | 1472 / 5916 |
| 64 | 7421 / 18301 | 6655 / 11276 | 2511 / 8737 |

## Decode step time (tpot ms, p50 / p90)

| conc | ours | sglang | vLLM |
|-----:|------|--------|------|
| 1 | 15 / 15 | 15 / 15 | 15 / 15 |
| 32 | 24 / 36 | 23 / 30 | 31 / 38 |
| 64 | 31 / 80 | 27 / 36 | 52 / 64 |

## What the table says

Up to concurrency 16 this arm is already ahead: 1.00-1.03x sglang, 1.06-1.17x
vLLM, with the lowest TTFT of the three. Kernel time is not the gap -- at c1
all three engines are within 1% (64.3 / 64.5 / 60.9 tok/s, tpot 15 ms).

The gap is entirely above c32, and it has a shape: ours is the only arm whose
throughput *falls* from c32 to c64 (683.7 -> 591.3) while both references keep
climbing (759.7 -> 805.1, 680.4 -> 822.7). The same transition triples our TTFT
p95 (9081 -> 18301 ms) and doubles decode step p90 (36 -> 80 ms) while sglang's
stays flat (30 -> 36).

A throughput peak followed by a collapse, with decode step p90 rising and TTFT
p95 exploding, is one class: the decode step is no longer the unit of
scheduling. Either prefill work is interleaving into the decode step (the p90
tail, ~44 ms above the p50, is roughly half a prefill chunk), or admission is
retracting and re-prefilling requests it already admitted, which pays the TTFT
twice and loses the KV it had. Both hypotheses are distinguishable from the
engine's own diagnostics and from a profiler trace of the c64 cell; a flag sweep
is not the next step.

## Next measurement

1. `ms/step` and kernels/step at c64 for ours vs sglang from the built-in
   profiler (`/start_profile`, `/stop_profile`), per the worked method in
   `engine-decode-parity` section 1.
2. The scheduler's own counters at c64: retractions per second, prefill tokens
   admitted per decode step, queue depth over time. If retractions are nonzero,
   the defect is admission granularity, not kernel time.
3. Only then a code change, verified at c64 against this table, with c1-c16
   re-measured to prove nothing regressed below the knee.

## The arm definition was wrong before any of this was measured

`ServerArgs.enable_mixed_batch` defaults to `False`, and the sweep never asked
for it, so **every number in the tables above is the no-mix configuration**.
Both reference engines pack prefill and decode into one step by default
(sglang's `mixed_overlap`, vLLM's chunked-prefill scheduling), so the comparison
was not "our scheduler against theirs" -- it was "our scheduler with
prefill/decode overlap switched off against theirs with it on".

The reason the arm never asked: the MHA leaf, which serves this checkpoint's
full-attention layers, asserted a MIXED round away
(`assert not forward_mode.is_mixed()`, `paged/mha.py`). A disable flag is a bug
report (AGENTS.md): this one is now fixed by serving the packed batch the way
the state groups and MLA already do -- extend kernel over the leading rows,
decode kernel over the trailing rows, `num_extends` carried on the decode
metadata. Until the c64 point is re-measured with `--enable-mixed-batch`, the
c32/c64 rows above are a lower bound on this arm, not a parity result.

## What the c64 cell's own counters say

The metrics time series from the no-mix c64 cell (`--enable-metrics`,
`ab_real/agentx-sweep/met-q38-c64/metrics.csv`, 20 Hz) already rules out the
retraction hypothesis and fixes the step accounting:

| quantity | measured |
|----------|----------|
| scheduler iterations over the measured phase | 588 (10.5 iter/s, 95.4 ms/step) |
| tokens processed per iteration | 852 (of an 8192 prefill budget) |
| `num_requests_running` | pinned at 64 (max_num_seqs 80, so slots are free) |
| `num_requests_waiting` | p50 8, p90 38, max 59 |
| `kv_cache_usage_perc` | p50 0.43, max 0.60 (KV is not the constraint) |
| `iteration_tokens_total` across the whole cell | 1,224,803 |

No retraction: the token counter matches the workload's own arithmetic
(prompts + generated + warmup), so nothing was re-prefilled -- retraction would
have inflated it. Admission is not refusing on KV (0.43 of the cache) or on
sequence slots (64 of 80).

What is left is per-step efficiency: 95.4 ms of wall time for 852 tokens, of
which ~64 are decode rows. At c1 the same model decodes a token every 15 ms, so
a c64 step costs 6x per token what a c1 step does -- and the workload is ~94%
prefill, so this step is prefill plus decode executed one after the other, with
the device idling between them. That is exactly what mixed batching removes, and
it is why the fix above comes before any other c64 work.
