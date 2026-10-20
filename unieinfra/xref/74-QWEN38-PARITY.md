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
prefill, so this step is prefill plus decode executed one after the other. That
is what mixed batching removes, and it is why the fix above comes before any
other c64 work.

## The profile of that cell: the device was not idling

The counters above say what the step cost but not what the device did with it, so
the same cell was profiled (`trace-q38-ours-c64/20261020-030130-TP0.trace.json.gz`,
425 MB compressed, 17.4M complete events). `trace_stats.py --list-hosts` picks the
step function by covered time: the forward is wrapped by
`tokenspeed/runtime/execution/device.py(567): _forward`, which is 1141 instances
in the window (the auto-detected candidate list cannot name it because the profiler
records the file's line number in the frame name).

| quantity | measured over the traced window |
|----------|---------------------------------|
| steps | 1141 |
| ms/step | mean 105.51, p50 4.54, p90 492.35, p99 822.27 |
| kernels/step | mean 361.5, p50 121, p90 1181, p99 2302 |
| us/kernel | 289.4 |
| device busy | 104.56 ms/step, **99% of step wall** |
| host/other | 0.96 ms/step |

**The device is saturated.** Per-step wall times cannot be compared across steps
under asynchronous submission -- a decode step returns in 4.5 ms while the GPU is
still chewing the previous prefill step's kernels; the wait lands in
`_commit_forward_results` (138 ms/call mean, of which 116 ms is
`stream.synchronize`). The aggregate is trustworthy where the histogram is not:
of the window's step wall, 99% is covered by running kernels. There is no idle gap
between prefill and decode for mixed batching to reclaim.

Where the device time goes:

| kernel group | device time | share |
|--------------|-------------|-------|
| GEMM (cuBLASLt `nvjet_sm90_*`) | 95.0 s | 80% |
| FlashAttention (16 full-attention layers) | 5.7 s | 4.8% |
| GDN delta rule | 3.0 s | 2.5% |
| `act_and_mul`, `causal_conv1d`, `rmsnorm`, `masked_fill`, split/gating | 7.3 s | 6% |

One GEMM alone -- `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT`, 34,066
launches at 2101 us -- is 71.6 s, 60% of all device time.

That fixes the arithmetic of the c64 gap, and it is not what this document
previously assumed. The corpus is 256 prompts of 8,321 tokens mean (2.13M prompt
tokens), but the requests continue each other's sessions, so most of every prompt
is an L3 prefix hit: the cell's own counters say only 852 tokens per iteration
were actually scheduled, 501k over the measured phase, of which 32.8k are the
generated tokens. The work is therefore ~94% prefill of ~470k tokens, and the
three engines run the same corpus through comparable radix/prefix caches, so they
deliver comparable token counts over the measured phase:

| engine | measured phase | delivered |
|--------|----------------|-----------|
| ours (no mix) | 56.35 s | 526 TFLOP/s (53% of H200 bf16 peak) |
| sglang | 40.7 s | 727 TFLOP/s |
| vLLM | 39.8 s | 744 TFLOP/s |

At 55.6 GB of bf16 weights the checkpoint is 27.8B dense parameters, so a token
costs 5.6e10 FLOPs and 533k tokens are 3.0e16. **The c64 gap is 1.38x of prefill
GEMM efficiency, not lost overlap.** Folding the decode rounds into the prefill
rounds saves at most the few percent of device time those rounds occupy, which is
why the mixed arm is still worth measuring -- and why it cannot be the whole
591 -> 805.

Read the kernel shares as indicative, not as a budget: `PREFILL_ISOLATION.md`
section 7.1 re-measured a comparable profiler table and found it had inflated one
op's per-step time by 16x. The wall-clock arithmetic above is what to trust; the
table only says the step is GEMM-dominated and that the small unfused kernels are
6% of it -- unlike the DeepSeek-V4 analysis in that document, where they were 20%,
so for this checkpoint fusion is not the lever it was there.

The same arithmetic per round, using the counters: ~470k prefill tokens arrive in
~62 chunks of ~7.6k tokens, so a prefill round costs ~870 ms and a decode round
~4.5 ms (526 of the 588 iterations, ~4% of the phase). A 7.6k-token chunk is
4.2e14 FLOPs, i.e. 486 TFLOP/s for that round; the 2.1 ms GEMM above works out to
~645 TFLOP/s on the shape the layer should have issued ([7.6k, 5120] x [5120,
17408]). Both sit inside the 49-53% overall, which leaves two levers, in order:
the ~20% of device time that is not GEMM, and the GEMM tactics themselves.

Padding is not one of them: the breakable prefill graph's ladder spaces rungs at
the largest power of two below `size/8`, capped at 512 tokens
(`prefill_graph.py:113-190`), so a 7.6k-token chunk pads by at most 2%.

The next profile must be taken with `record_shapes` (the profiler accepts it:
`request_handler.py:989`) so the GEMM can be named by its M and N instead of by
its tile shape, and checked against the shape the layer should have issued -- a
second row for the same weights at a smaller M is a chunking defect, and a
correct M with a poor tactic is a tuning defect. The two call for opposite fixes,
which is why the shape has to be read before either is touched.
