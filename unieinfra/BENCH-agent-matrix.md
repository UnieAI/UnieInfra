# Agent-workload benchmark matrix

Two engines, three scenarios, two models that fit one card, measured
2026-08-14. Everything here is one RTX PRO 6000 (sm_120, 96 GB) unless a row
says otherwise.

## Read this first

**Goodput is the metric, and it has a peak.** A request that misses its latency
budget is worth nothing to the caller, so the number that matters is completed
requests inside the budget, not tokens per second. Every scenario here turns
over: throughput keeps climbing after goodput has started falling, and past the
knee both collapse together.

**Goodput cannot be compared across scenarios.** vLLM does 124.2 qualifying
requests/min on RAG and 18.6 on agent traffic with the same hardware, because a
1.2k-token prompt under a 5 s budget is a different question from a 127k-token
prompt under 10 s. Compare engines within a row, never numbers between blocks.

**Reuse figures come from the engines' own counters** -- ours from
`#cached-token / (#new + #cached)`, vLLM's from its `Prefix cache hit rate` --
never from what the corpus was designed to contain. That distinction cost three
wrong conclusions in one day: a corpus advertising 98% delivered 13.3% once its
sessions were strided, a long-output slice reported 0% that was an artefact of
the ISL cap, and a 35B ladder ran at 46-81% on a corpus built before the
session-contiguity fix.

## What was measured

| | |
|---|---|
| Hardware | 1x RTX PRO 6000, sm_120, 96 GB (2x for the 122B attempt) |
| Client | `vllm bench serve` over HTTP, OpenAI chat endpoint |
| Warmup | 2x concurrency requests, discarded, per the InferenceX rule |
| Requests | 64 measured per arm |

### Models

| name | architecture | precision | notes |
|---|---|---|---|
| Qwen3.5-27B | hybrid GDN, 48 GDN + 16 full-attention | bf16 | dense |
| Qwen3.6-35B-A3B | hybrid GDN, 40 layers | bf16 | MoE 256 experts, 8 active |
| Qwen3.5-122B-A10B | hybrid GDN, 36 GDN + 12 full | FP8 block [128,128] | **did not run** -- see below |

### Scenarios

| name | prompt | output | sharing | SLO |
|---|---|---|---|---|
| `agent-127k` | median 127,040 | median 806 | real, 93-99% | TTFT <= 10 s, TPOT <= 200 ms |
| `agent-26k` | median 26,240 | median 290 | real, 36-74% | TTFT <= 10 s, TPOT <= 200 ms |
| `RAG` | ~1,200 | 64 | 16 documents x ~64 questions | TTFT <= 5 s, TPOT <= 100 ms |
| `general` | 1,024 | 1,024 | InferenceX random | TTFT <= 2 s, TPOT <= 100 ms |

The agent corpora replay SemiAnalysis's `cc-traces-weka-062126` Claude Code
proxy traces. Prefix sharing is not a chosen fraction: each trace request
carries `hash_ids`, the KV-block hashes of its prompt, and the generator maps
each id to a deterministic chunk, so two requests share exactly the blocks the
real traffic shared. Whole sessions are kept contiguous, because a request can
only hit blocks some earlier request in the same file already wrote.

**Why two agent corpora.** The real distribution has median 142,016 input
tokens. On 27B that fits. On 35B-A3B it does not: 67 GB of bf16 weights leave a
458,752-token pool, and the engine clamps maximum input to 53,246, so every
request of the full corpus is rejected. `agent-26k` is the lower tail of the
same traces, and it carries less sharing for a structural reason -- an ISL cap
preferentially keeps session openers, which are exactly the requests that have
no ancestor. The 35B rows therefore sit on a genuinely harder cache workload
than the 27B rows and are not comparable to them.

## Results

### Qwen3.5-27B, real 127k-token agent traces

Both engines, same corpus, same 64 requests, TTFT <= 10 s and TPOT <= 200 ms. Our goodput peaks at C=16 (26.4/min against 18.6, **1.42x**) and both engines collapse by C=64. Note that raw tok/s keeps climbing through C=32 while goodput is already falling, and at C=64 vLLM posts the *higher* throughput while qualifying for nothing.

| C | engine | tok/s | qualifying/min | pass rate | TTFT p50 | TTFT p99 | TPOT p50 | tok/s per user | reuse |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **UnieInfra** | 21.9 | **5.4** | 100% | 269 ms | 4320 ms | 42.8 ms | 23.4 | 96.5% |
| 1 | vLLM | 21.5 | **4.8** | 100% | 480 ms | 3582 ms | 43.7 ms | 22.9 | 96.5% |
| 4 | **UnieInfra** | 66.5 | **15.6** | 100% | 870 ms | 14264 ms | 47.4 ms | 21.1 | 96.9% |
| 4 | vLLM | 55.5 | **13.2** | 100% | 896 ms | 9209 ms | 62.5 ms | 16.0 | 84.8% |
| 16 | **UnieInfra** | 132.2 | **26.4** | 85% | 1222 ms | 49238 ms | 72.8 ms | 13.7 | 97.9% |
| 16 | vLLM | 98.6 | **18.6** | 79% | 1533 ms | 24881 ms | 128.1 ms | 7.8 | 88.7% |
| 32 | **UnieInfra** | 161.1 | **24.6** | 65% | 1791 ms | 82848 ms | 111.1 ms | 9.0 | 98.5% |
| 32 | vLLM | 115.2 | **11.4** | 42% | 2347 ms | 42583 ms | 205.4 ms | 4.9 | 91.4% |
| 64 | **UnieInfra** | 93.9 | **1.2** | 5% | 80872 ms | 152349 ms | 87.1 ms | 11.5 | 98.7% |
| 64 | vLLM | 125.2 | **0.0** | 0% | 39321 ms | 49524 ms | 356.6 ms | 2.8 | 94.1% |

### Qwen3.6-35B-A3B, real agent traces capped at 26k tokens

A different corpus from the block above and not comparable to it -- the full distribution does not fit this model on one card. Two things stand out. The MoE decodes far faster than the dense model (TPOT 6.3 ms against 42.8, so 158 tokens per second per user at C=1, the same order as the 167 observed in the real traces) but its TTFT degrades much harder, because decode touches 3B parameters while prefill still touches all 35B. And vLLM's prefix cache collapses to ~1% from C=16 upward where ours climbs to 83.5%: this model leaves only 375 mamba blocks, and under that pressure vLLM evicts blocks it still needs. vLLM wins C=4 outright.

| C | engine | tok/s | qualifying/min | pass rate | TTFT p50 | TTFT p99 | TPOT p50 | tok/s per user | reuse |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **UnieInfra** | 115.9 | **27.0** | 100% | 519 ms | 1382 ms | 6.3 ms | 158.5 | 35.7% |
| 1 | vLLM | 84.0 | **19.8** | 100% | 106 ms | 319 ms | 10.8 ms | 93.0 | 66.5% |
| 4 | **UnieInfra** | 159.6 | **35.4** | 95% | 1769 ms | 10416 ms | 8.7 ms | 114.8 | 46.6% |
| 4 | vLLM | 191.5 | **45.0** | 100% | 1092 ms | 2473 ms | 16.1 ms | 62.2 | 38.3% |
| 16 | **UnieInfra** | 170.2 | **18.6** | 47% | 15089 ms | 43835 ms | 10.6 ms | 94.5 | 63.6% |
| 16 | vLLM | 68.5 | **0.6** | 3% | 36658 ms | 44809 ms | 62.3 ms | 16.1 | 1.3% |
| 32 | **UnieInfra** | 149.8 | **12.0** | 34% | 39127 ms | 85201 ms | 13.4 ms | 74.9 | 74.3% |
| 32 | vLLM | 78.3 | **0.0** | 0% | 65509 ms | 87604 ms | 73.7 ms | 13.6 | 1.0% |
| 64 | **UnieInfra** | 135.4 | **8.4** | 26% | 51557 ms | 117501 ms | 15.1 ms | 66.2 | 83.5% |
| 64 | vLLM | 80.0 | **0.0** | 0% | 164306 ms | 200357 ms | 64.5 ms | 15.5 | 0.9% |

### RAG: 16 documents, ~64 questions each

~1.2k-token prompts, 64-token answers, stricter budget (TTFT <= 5 s, TPOT <= 100 ms). Everything qualifies on both engines, so this scenario separates them least -- the honest result is a tie on 27B.

| model | engine | tok/s | qualifying/min | pass rate | TTFT p50 | TPOT p50 | tok/s per user | reuse |
|---|---|---|---|---|---|---|---|---|
| 27B | **UnieInfra** | 538.0 | **126.0** | 100% | 1241 ms | 53.5 ms | 18.7 | 69.0% |
| 27B | vLLM | 529.4 | **124.2** | 100% | 1919 ms | 50.9 ms | 19.7 | 54.9% |
| 35B-A3B | **UnieInfra** | 1099.7 | **258.0** | 100% | 400 ms | 27.3 ms | 36.6 | 70.2% |

### General: InferenceX random 1024-in / 1024-out

Not a zero-reuse control: both engines achieve 37-45%, because a chat template's common prefix fills the first of eight 128-token pages. On 27B the two engines produce *identical* throughput -- 585.68 tok/s each, since the same device does the same total work in the same time -- and differ only in who finishes when. That difference is 7.5x in qualifying requests.

| model | engine | tok/s | qualifying/min | pass rate | TTFT p50 | TPOT p50 | tok/s per user | reuse |
|---|---|---|---|---|---|---|---|---|
| 27B | **UnieInfra** | 585.7 | **9.0** | 26% | 2985 ms | 51.7 ms | 19.3 | 44.5% |
| 27B | vLLM | 585.7 | **1.2** | 4% | 4081 ms | 50.7 ms | 19.7 | 37.1% |
| 35B-A3B | **UnieInfra** | 1117.6 | **65.4** | 100% | 566 ms | 28.0 ms | 35.7 | 44.5% |


## What to optimise next, in the order the data ranks it

### 1. The TTFT tail, everywhere

Our median first-token latency beats vLLM's in almost every row, and our p99 is
worse in almost every row. On the 127k agent corpus at C=16 we are 1222 ms at
the median against vLLM's 1533, and 49,238 ms at p99 against its 24,881. The
same inversion holds at C=32 and C=64.

That is a fairness problem, not a speed problem: most requests are served
faster than vLLM serves them while a minority waits far longer. Because goodput
counts requests, not tokens, the tail is what caps it -- at C=32 we still hold
65% pass rate against vLLM's 42%, but both are losing to their own tails rather
than to per-token cost.

The mechanism is identified and the fix is not a parameter. Preemption is
effectively unreachable: a retract victim must owe no outstanding forward
result, and at overlap depth 1 nearly every resident request owes one, so the
scheduler finds no eligible victim. Relaxing that filter crashes the FSM
(`event=ExtendResultEvent; state=Retracted`) because a result issued a step
earlier lands after the retraction. Making it work needs a two-phase retract:
mark the victim, stop scheduling it, and retract once its result lands. A first
attempt at that ran but cost 45% of throughput, because retraction forces a
re-prefill -- so the deferred design has to be paired with a way to keep the
victim's computed prefix.

### 2. Low-concurrency prefill on the MoE model

On 35B-A3B at C=1 vLLM reaches first token in 106 ms against our 519, and at
C=4 in 1092 against our 1769. We win from C=16 upward, but the low-concurrency
end is where a single-user agent actually lives.

**The reuse figures say the same thing more sharply.** At C=1 vLLM achieves
66.5% cache hits against our 35.7% -- nearly twice -- and at C=4 the two nearly
swap (38.3% against 46.6%). Our hit rate then climbs monotonically with
concurrency (63.6 / 74.3 / 83.5%) while theirs collapses. That shape suggests
our cache is finding hits mainly through *co-resident* requests rather than
through history: with one request in flight there is nobody to share with, and
we do worse than an engine whose lookup reaches further back. Worth confirming
against the 27B rows, where at C=1 both sit at 96.5% and the effect disappears
-- which points at the 26k corpus's weaker session chains rather than at the
cache itself, and makes this a measurement to repeat before it is a defect to
fix.

### 3. Throughput on the general shape

At C=128 on random 1024/1024 traffic we reach 848.44 tok/s against vLLM's
1035.84 -- 81.9%, up from 20.3% before today's admission work, but still short.
At a full batch the two engines are identical (1036 against 1036 tok/s), so the
remaining gap is average batch size: 68.9 concurrent decoders against 99.3.

### 4. Prefix-cache eviction under pressure -- theirs, and the lesson for ours

vLLM's cache collapses to ~1% from C=16 on the 35B model, where 375 mamba
blocks are all it has, and its goodput goes to zero with it. Ours climbs to
83.5% on the same workload. This is the one place where our block-level cache
clearly outperforms, and it is worth understanding *why* before assuming it
generalises: on the 27B model, where the pool is larger, vLLM holds 91-94% and
the gap is small.

### 5. Capacity, which bounds what can be measured at all

64 GB of KV per token on 27B means one median real agent request costs 8.67 GB,
a quarter of the pool. 35B-A3B leaves 458,752 tokens after weights and clamps
max input to 53,246, so it cannot serve the real distribution on one card at
all. Neither is a scheduling problem; both are reasons to want FP8 KV,
quantised weights, or more cards -- and the 122B FP8 attempt is blocked on MoE
kernel coverage rather than on memory.

## What did not run, and why

### 35B-A3B cannot serve the real agent distribution on one card

Both engines fail identically, which is what makes it a capacity result rather
than a defect in either: 67 GB of weights against 96 GB leaves 458,752 tokens
of KV, the engine clamps max input to 53,246, and the corpus median is 127k.
Running it needs quantisation or more cards.

### 122B-A10B-FP8: the hardware was never the problem

The model fits and divides cleanly -- 118 GB FP8 over two cards is 59 GB each,
32 attention heads and 256 experts both halve -- and weights load, with the
cache planner reporting 451 parents. It failed in the first forward, during
autotune, inside the MoE.

**vLLM serves this model on the same two cards.** Measured 2026-08-14, TP2 on
the RAG corpus at C=8: 32/32 completed, 0 failed, 402.4 tok/s, TTFT p50 466 ms,
TPOT p50 18.1 ms, 94.3 qualifying requests/min, and the generated text is
ordinary language. So "sm_120 has no FP8 block-scale MoE kernel" was a
statement about this tree, not about this box, and the two were being
conflated. The full five-engine reachability table is in
`unieinfra/xref/46-fp8-moe-reachability.md`.

Three defects, two fixed:

1. **Fixed.** `trtllm_fp8.py` raised `unexpected hidden_states_scale shape
   (196608,); expected (24, 8192)`. TRT-LLM's `per_token_group_quant_8bit`
   returns the `[K/128, M]` group-major scale flattened; the element order is
   already right, so the fix restores the 2D view. It only reshapes when the
   element count matches exactly -- a wrong layout would run and emit plausible
   nonsense rather than failing.
2. **Open, and not what this report first said.** The initial reading was that
   the TRT-LLM-Gen batched GEMM lacked a kernel at the shape autotune probes
   first (`numBatches: 256, GemmMNK: 1 x 1024 x 3072`), i.e. small-M coverage.
   That was wrong on both counts, and `unieinfra/bench/moe_probe.py` shows why:
   it calls the FP8 MoE solutions directly at the model's shape -- 256 experts,
   hidden 3072, per-partition intermediate 512 -- on one card with no weights
   to load, so each backend answers in seconds.

   Only three registered solutions declare `weight_dtype: fp8` at all, not the
   seven assumed above. `gluon`, `marlin`, `triton`, `cuda` and `native` carry
   no FP8 MoE path, which is the same reason `triton` was rejected. The three
   that do:

   | solution | M=1 | M=8 | M=128 | why |
   |---|---|---|---|---|
   | `flashinfer_cutlass` | fail | fail | fail | `NotImplementedError: FP8 block scaling not yet implemented for Blackwell` |
   | `flashinfer_trtllm` | fail | fail | fail | GEMM runner error; the kernel it selects ends `_sm100f` |
   | `deep_gemm` | fail | fail | fail | `DeepEP MoE plan is missing deepep_group` -- an all-to-all EP path |

   So it is not small-M coverage: TRT-LLM-Gen fails at M=8 and M=128 too, and
   the kernel name it reports is compiled for sm_100 (datacenter Blackwell),
   not the sm_120 in this box. Every FP8 block-scale MoE path in the tree was
   either sm_100-only or DeepEP-only.

   One consequence for defect 1: at M=1 the guard now reports `shape (96,);
   expected (24, 1)`, and 96 = 24 x 4, so TRT-LLM pads M to a multiple of four
   in the scale layout. Widening the guard to accept that padding is not worth
   doing while the GEMM beneath it fails at every M.

3. **Fixed: DeepGEMM was reachable all along, behind DeepEP.** The reference
   engine picks DeepGEMM for this model on this card -- `Using DEEPGEMM Fp8 MoE
   backend out of potential backends: [...]` -- while our only DeepGEMM path,
   `deepep_fp8.py`, declares `supports_all_to_all_ep` and raises on a plan with
   no process group. A single-box TP deployment has no such group and never
   will, so the kernel was unreachable by construction rather than missing.

   The compute between DeepEP's `dispatch_b` and `combine_a` is collective-free:
   it consumes FP8 tokens, block scales, routing ids and weights, and per-expert
   row counts. `moe/deep_gemm/contiguous_fp8.py` produces those five locally --
   quantize, histogram, align to 128 -- and reuses the same permute/GEMM/
   unpermute core and the same weight preprocessor, so the two paths cannot
   drift apart numerically. It now runs at the model's shape at M=1, 8 and 128.

   A detail worth keeping: the row counts must pad only the experts that were
   actually routed to. Padding all of them still runs, and turns one decode
   token into 256 x 128 rows instead of 8 x 128 -- visible only as a throughput
   mystery. `test_deep_gemm_contiguous.py` pins it.

   Two environment defects sat underneath, both of which reported themselves as
   source bugs. Two CCCL copies were visible at once, so one translation unit
   took `<cuda/std/utility>` from nvcc's copy and `<cuda/std/version>` from an
   older one on `CPATH` (`missing binary operator before token "("`); and the
   `cu13` wheel ships nvcc 13.2 against CUDART 13.3 headers, which CCCL refuses
   outright. Both are fixed in `sm120_env.sh`, with the reasoning inline. They
   are global changes, so "does not affect the existing Qwen models" has to be
   measured rather than argued.

The four configurations, derived from the resolution logic in `server_args`
(`attn_tp x attn_cp x attn_dp = world`, `moe_ep x moe_tp = world`):

| | attention | dense | MoE | flags |
|---|---|---|---|---|
| A | TP2 | TP2 | ep1 x tp2 | `--tensor-parallel-size 2` |
| B | DP2 | TP1 | ep2 x tp1 | `--data-parallel-size 2 --ep-size 2 --dist-init-addr ...` |
| C | TP2 | TP2 | ep2 x tp1 | `--tensor-parallel-size 2 --ep-size 2` |
| D | DP2 | TP1 | ep1 x tp2 | `--data-parallel-size 2 --moe-tp-size 2` |

## Method notes worth keeping

- **Check `Failed requests` before reading any other number.** Three runs today
  finished with exit code 0 and an empty result: a client pointed at the wrong
  endpoint "completed" 256 warmups in 0.2 s, a port collision returned 404s
  behind a passing readiness check, and a harness syntax error produced a clean
  exit having run nothing.
- **Stopping a harness does not stop the engine.** A `setsid nohup` server
  survives the script and holds 92 GB; the next arm then OOMs, and the message
  names "GPU 0" because `CUDA_VISIBLE_DEVICES` renumbers. Kill by PID, from
  inside the container, where the process is root.
- **Never rebuild the scheduler extension mid-ladder.** Every new engine loads
  it from dist-packages, so a rebuild between points silently mixes two builds
  into one curve.
- **Empty metric fields collapse under `IFS=$'\t'`.** Tab is whitespace-class,
  so repeats merge and every later column shifts left; that turned a goodput of
  0.02 into a throughput of 0.02 in an early table. The results here are parsed
  as label-value pairs instead.
