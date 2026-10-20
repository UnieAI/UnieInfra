# Qwen3.8-27B: three-arm parity, TP=1

Qwen/Qwen3.8-27B, one H200 per arm, `TP=1`. 256 requests per cell, 128 generated
tokens, greedy, `ignore_eos`, `/v1/completions` with the corpus text verbatim,
fresh boot per concurrency point (warmup corpus shares no prefix with the
measured one). All three arms: `--max-num-seqs 80`, 32768 context, chunked
prefill 8192. No speculation, no hierarchical cache in any arm.

Driver: `bench/agent-slo/agentx_sweep.sh` with `PROFILE=qwen38`, table
`ab_real/agentx-sweep/parity_table.py <dir> q38` with `OURS_TAG=` set to our
arm's sweep. Every cell: 256/256 requests ok, 128.0 tok/req in all three arms,
so the comparability gate passes and these are throughput ratios, not artifacts
of different output lengths.

Our arm is the **mixed** configuration (`--enable-mixed-batch`), re-measured
after the padded-row convolution fix (tag `q38f`,
`ab_real/agentx-sweep/run_q38_fixsweep.sh`). The reference cells are unchanged
from the `q38` sweep. The control (`Q38_MIXED=0`) was re-measured at c16 and c64
on the fixed tree and landed inside the noise floor of its pre-fix numbers
(578.3 vs 583.6, 574.0 vs 591.3), as it must: the fix only writes rows that a
padded slot owns, and a no-mix round has none.

## Throughput (tok/s total)

| conc | ours | sglang | vLLM | ours/sglang | ours/vLLM |
|-----:|-----:|-------:|-----:|------------:|----------:|
| 1 | 64.3 | 64.5 | 60.9 | 1.00x | 1.06x |
| 4 | 221.8 | 221.5 | 203.6 | 1.00x | 1.09x |
| 8 | 380.4 | 369.5 | 335.6 | 1.03x | 1.13x |
| 16 | 589.0 | 568.5 | 498.2 | 1.04x | 1.18x |
| 32 | 807.3 | 759.7 | 680.4 | 1.06x | 1.19x |
| 64 | 998.1 | 805.1 | 822.7 | 1.24x | 1.21x |

Requests per minute, the same cells: ours 30.2 / 104.0 / 178.3 / 276.1 / 378.4 /
467.9 against sglang's 30.2 / 103.8 / 173.2 / 266.5 / 356.1 / 377.4 and vLLM's
28.5 / 95.5 / 157.3 / 233.5 / 318.9 / 385.6.

## TTFT (ms, p50 / p95)

| conc | ours | sglang | vLLM |
|-----:|------|--------|------|
| 1 | 70 / 208 | 98 / 227 | 170 / 268 |
| 8 | 336 / 1052 | 499 / 2354 | 670 / 1849 |
| 16 | 635 / 1775 | 912 / 2808 | 1105 / 2770 |
| 32 | 633 / 5069 | 2076 / 5506 | 1472 / 5916 |
| 64 | 1231 / 7902 | 6655 / 11276 | 2511 / 8737 |

p99 at c32/c64: ours 6292 / 7933, sglang 5508 / 12698, vLLM 6089 / 8925.

## Decode step time (tpot ms, p50 / p90)

| conc | ours | sglang | vLLM |
|-----:|------|--------|------|
| 1 | 15 / 15 | 15 / 15 | 15 / 15 |
| 16 | 20 / 25 | 19 / 22 | 22 / 26 |
| 32 | 30 / 35 | 23 / 30 | 31 / 38 |
| 64 | 45 / 56 | 27 / 36 | 52 / 64 |

## What the table says

This arm is at parity or ahead at every concurrency point. At c1 all three
engines are within 1% (64.3 / 64.5 / 60.9 tok/s, tpot 15 ms) because the cell is
one request wide and the kernel is the whole story. From c8 up the lead grows
with concurrency, to 1.24x sglang and 1.21x vLLM at c64, and the TTFT column
leads everywhere: at c64 the p50 is 5.4x lower than sglang's and 2.0x lower than
vLLM's.

**The c32/c64 collapse the previous version of this table showed was the no-mix
schedule, not the kernels.** The no-mix control on the same tree still falls from
683.7 at c32 to 574-591 at c64; the mixed arm rises through those two points to
998.1. Nothing was tuned: the difference is that a step may now carry prefill and
decode rows together.

What is left is per-token latency above c16: tpot p50 45 ms at c64 against
sglang's 27. Aggregate throughput is higher because the steps deliver more work,
but each decode token waits behind whatever prefill the same step carries, so a
stream that is already admitted pays sglang's step time plus a share of the
chunk. That is the next lever, and it is a scheduling-granularity question (how
much prefill one step may carry beside decode), not a kernel one.

## What remains

1. **Per-token latency above c16** (ours 45 ms p50 at c64 against sglang's 27).
   Aggregate throughput already leads, so the next change must hold c64 total
   tps and TTFT while cutting tpot: it is about how much prefill one step may
   carry beside its decode rows, and about whether a single request's chunk has
   to complete in one step.
2. **The prefill GEMM tactics**, which run at 53% of peak while the no-mix arm
   was recomputing (the only arm whose work is a fixed denominator). It no longer
   explains the c64 gap -- that was redundant prefill, not slow GEMM, and the
   cross-engine TFLOP/s row that claimed otherwise is retracted below -- but it is
   the reason c1 is at parity rather than ahead, and it is what a `record_shapes`
   profile has to name: the shape, then the tactic.
3. **Report cache reuse from the engine, not from its log.** The 91.2%-against-
   77.7% split above was read out of admission log lines by hand; there is no
   metrics series for prefix hits, no count of evicted prefixes and no eviction log
   line. Every arm of every comparison in this directory is therefore blind to the
   one quantity that decided this cell, which is the same class of defect as an
   admission planner that prints its numbers only behind `.plan_trace.on`.

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
metadata.

There is no `--disable-mixed-batch` in this tree (`--enable-mixed-batch` is the
only switch), so the two arms differ by that flag alone and the control is
`Q38_MIXED=0` on the sweep profile. Passing the nonexistent flag makes `ts serve`
exit 2 at argument parsing, which the sweep reports only as a boot that never
completes -- one cell was lost to it before the knob existed.

## The mixed arm also corrupted its output, one unwritten row at a time

Teaching the MHA leaf to serve a packed batch made the arm run, and it then
disagreed with the no-mix arm on 33-37% of greedy replies. Text agreement
saturates on the first flipped token, so the interesting numbers are the ones
that do not: per-token `output_token_logprobs` on eight prompts show **0 id
flips** and |dlogprob| 0.025-0.115 between the arms, against a same-server
self-comparison floor of 0.709-0.742 with one flip in eight. The arms differ by
less than the engine differs from itself. What made the texts diverge was a
poisoned GDN state, not the arithmetic -- and it did not show up in any
throughput number.

The defect was one row of one tensor. A mixed batch carries padded request
slots whose state page is `pad_slot_id` (`-1`). Such a slot owns no cache line,
and `_causal_conv1d_fwd_kernel` correctly skipped its convolution -- but it
skipped the *store* with it, returning before writing its `o` rows. Those rows
are still read: the mixed metadata builds `query_start_loc` from
`query_lens`, whose trailing entry is one token per padded slot, so a batch of
`[968, 3456, 1, 1, ...]` has the pad rows sitting *before* the last boundary
where `scrub_padding` never looks. The caller then scanned the allocation's
leftovers out of `torch.empty_like`, which in bf16 are frequently NaN. One NaN
row corrupted the final partial chunk of the scan, and the conv window
persisted from that chunk (`conv_states[idx]` rewritten from itself) carried the
NaN into the next forward -- self-sustaining poisoning of the page, well beyond
the field's own lifetime.

Three diagnostics settled it, in this order:

1. The program count matched `sum(ceil(len/8))` exactly (567 for lengths
   `[968, 3456, 1 x 14]`), so every row *did* have a program: the mapping was
   never the problem.
2. Dumping the extend metadata showed the mixed rounds carrying trailing
   1-token rows inside `query_start_loc`.
3. Replacing the output buffer with `zeros_like` made nine consecutive runs of
   `probe_short.py 48 24` come back clean (`short=0/48`, no NaN scanner hits),
   and the kernel fix alone reproduced that six times over.

The fix is three lines in `_causal_conv1d_fwd_kernel`: a padded slot stores
zeros across its rows before returning. The rows carry no information the
caller may use, but they must be *defined*, because the scan reads them.
`test/ops/attention/test_causal_conv1d_pad_rows.py` guards it by poisoning the
allocator and asserting every padded row reads as zeros, that the slot's cache
line is untouched, and that the real rows still match a float32 FIR reference.

The lesson generalizes past convolution: **an early return in a fused kernel
owes its output rows to the batch contract.** "This slot has nothing to do" is
about the work, not about the tensor -- and a section of the batch that no
consumer indexes by name is exactly where the debt is forgotten.

## The two c64 arms do different amounts of work

Both arms ran the same cell at c64: same corpus, same 256 requests, same 128
generated tokens each, so both produced exactly 32,768 output tokens.
`iteration_tokens_total` is nevertheless **2.33x apart**
(`ab_real/agentx-sweep/run_q38_preempt.sh`, one `/metrics` read after each cell's
load):

| quantity at c64 | mixed | no-mix |
|-----------------|------:|-------:|
| scheduler iterations | 539 | 615 |
| `iteration_tokens_total` | 217,852 | 506,918 |
| tokens scheduled per iteration | 404 | 824 |
| scheduled tokens per second | 6,642 | 9,005 |
| delivered tok/s total | 998.1 | 574-591 |
| `tokenspeed:num_preemptions_total` | never exported | never exported |
| prefix-cache hit rate | 91.2% | 77.7% |
| `kv_cache_usage_perc` peak | 0.42 | 0.57 |

Three readings of that table, in the order they matter:

1. **Not retraction.** `num_preemptions_total` was never exported by either arm.
   A labelled counter has no child series until its first `.inc()`, so its
   absence is the engine reporting zero retractions, and retraction was the
   standing explanation for the token differential. It is dead.
2. **Not a slower step either.** The no-mix arm schedules *more* tokens per
   second of its own device time (9,005 against 6,642). It is not starving; it
   is being given 2.3x the work for the same output.
3. **It is prefix reuse.** Subtracting the 32,768 generated tokens leaves 185k
   prompt tokens scheduled by the mixed arm and 474k by the no-mix arm. The
   engine prints the split per admission (`Mix batch ... #new-token: 123,
   #cached-token: 11904`): 91.2% of prompt tokens were cache hits in the mixed
   arm against 77.7% in the no-mix arm, measured over the same corpus and the
   same cache size. The 289k-token difference is prefill that the mixed arm
   reused and the no-mix arm recomputed, and its KV high-water (0.57 against
   0.42) is what refusing that reuse looks like.

The mechanism is the schedule, and it is worth stating plainly because it
reverses the intuition: a no-mix step spends its whole 8192-token budget on
new admissions, so it admits several long prompts at once and holds their pages
while they are half-prefilled. Those pages are the room the shared prefixes
were using, so a later request in the same session finds its prefix gone and
pays for it again. A mixed step spends part of the budget on decode rows and
admits fewer new requests per unit time, which keeps the live prefix set small
enough to stay resident. Nothing in the engine reports this: there is no
eviction log and no hit-rate series, only the admission line the sampler above
scraped out of the log. **A cache that has no room evicts silently and the
cost shows up as extra prefill with no counter pointing at it.**

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

That last sentence did not survive the re-measurement: this profile was taken on
**the no-mix arm**, and the mixed arm then delivered 1.69x its c64 tokens per
second (591.3 -> 998.1) with the same prompts, the same 128 tokens of output per
request and a 6x lower TTFT p50. A device that is 99% busy at 53% of peak is busy,
not efficient; the quantity that separates the two arms is what that device time
was spent on, and no busy-fraction argument can decide it. The 99% only rules out
host-side starvation, which is what it was measured for.

Where the device time goes:

| kernel group | device time | share |
|--------------|-------------|-------|
| GEMM (cuBLASLt `nvjet_sm90_*`) | 95.0 s | 80% |
| FlashAttention (16 full-attention layers) | 5.7 s | 4.8% |
| GDN delta rule | 3.0 s | 2.5% |
| `act_and_mul`, `causal_conv1d`, `rmsnorm`, `masked_fill`, split/gating | 7.3 s | 6% |

One GEMM alone -- `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT`, 34,066
launches at 2101 us -- is 71.6 s, 60% of all device time.

That table is what the no-mix cell spent, and the FLOP-per-second row derived from
it has to go, because it rests on an assumption the counters have since killed.
The corpus is 256 prompts of 8,321 tokens mean (2.13M prompt tokens), but the
requests continue each other's sessions, so most of every prompt is a prefix hit:
the no-mix cell scheduled 507k tokens over its phase (824 per iteration), of which
32.8k are the generated ones. Dividing 5.6e10 FLOPs per token into that gave
526 TFLOP/s -- and then the same engine, same corpus, same cache size, scheduled
**217,852** tokens when the schedule allowed mixing. Sched token counts are a
property of the schedule, not of the workload, so "the three engines run the same
corpus, therefore the same work" is false and no cross-engine TFLOP/s number can
be built on it:

| arm | phase | scheduled tokens | FLOP/s, all scheduled tokens |
|-----|-------|------------------|------------------------------|
| ours (no mix) | 56.3 s | 506,918 | 526 (53% of H200 bf16 peak) |
| ours (mixed) | 32.8 s | 217,852 | 317 (32% of peak) |
| sglang / vLLM | 40.7 / 39.8 s | not measured | not derivable |

The mixed arm posts the *lower* FLOP/s of the two arms that can be measured and
wins the cell, because it computes 2.33x fewer tokens: the 591 -> 998 was not the
same work done faster, it
was 289k tokens of work not done. A ratio of FLOPs to wall time measures a kernel
only when the numerator is fixed, which here it is not -- the same trap as reading
a busy device as an efficient one. What the row still legitimately says is that
the no-mix arm's device ran at 53% of peak while it was recomputing, so there is
headroom in the GEMM tactics; it says nothing about the c64 gap.

Read the kernel shares as indicative, not as a budget: `PREFILL_ISOLATION.md`
section 7.1 re-measured a comparable profiler table and found it had inflated one
op's per-step time by 16x. The wall-clock arithmetic above is what to trust; the
table only says the step is GEMM-dominated and that the small unfused kernels are
6% of it -- unlike the DeepSeek-V4 analysis in that document, where they were 20%,
so for this checkpoint fusion is not the lever it was there.

The same arithmetic per round, using the counters: ~470k prefill tokens arrive in
~62 chunks of ~7.6k tokens, so a prefill round costs ~870 ms and a decode round
~4.5 ms. A 7.6k-token chunk is 4.2e14 FLOPs, i.e. 486 TFLOP/s for that round; the
2.1 ms GEMM above works out to ~645 TFLOP/s on the shape the layer should have
issued ([7.6k, 5120] x [5120, 17408]). Both sit inside the 49-53% the no-mix arm
held, which leaves two levers there, in order: the ~20% of device time that is not
GEMM, and the GEMM tactics themselves. Neither is on the critical path for the
parity claim any more -- they are what to reach for once tpot at c64 has been
fixed, since a faster GEMM shortens every decode step too.

Padding is not one of them: the breakable prefill graph's ladder spaces rungs at
the largest power of two below `size/8`, capped at 512 tokens
(`prefill_graph.py:113-190`), so a 7.6k-token chunk pads by at most 2%.

The next profile must be taken with `record_shapes` (the profiler accepts it:
`request_handler.py:989`) so the GEMM can be named by its M and N instead of by
its tile shape, and checked against the shape the layer should have issued -- a
second row for the same weights at a smaller M is a chunking defect, and a
correct M with a poor tactic is a tuning defect. The two call for opposite fixes,
which is why the shape has to be read before either is touched.
