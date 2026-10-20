<!-- English canonical. docs/design/* are English-only in this repo (no Chinese mirror). -->
# Design: Porting UnieInfra's Admission Policy onto vLLM

Status: **Implemented and measured (E36)** · Date: 2026-07-25 · Owner: royshih

Companion to `docs/design/vllm-base-assessment.md` (the base decision) and
`docs/design/vllm-kernel-plugin.md` (E34, the kernel half of the same re-base).
Code: `unieinfra-scheduler/`. Measurement: `docs/experiments-log.md` E36.

## What this was supposed to be, and what it turned out to be

The brief was to port `derive_adaptive_max_batch` — the old TokenSpeed base's
KV-derived admission cap, and the one measured, repeatable advantage that base
had (E19, E25, E35) — onto vLLM as an *adaptive* policy rather than a copy.

The mechanism, as E35 recorded it, was: size `max_num_seqs` from KV-pool token
capacity (160 -> 480 on this box), extend the CUDA-graph capture ladder to
cover it, and trade queueing latency for batch size.

**On vLLM 0.25.1 that mechanism has no room to operate, and the honest finding
is a smaller delta than E35 suggested.** Before-picture, measured on the
benchmark box (1x RTX PRO 6000, 96 GB, Qwen3-8B bf16, `--max-model-len 4096`,
`gpu-memory-utilization 0.85`):

| Quantity | old TokenSpeed base | stock vLLM 0.25.1 | source |
|---|---|---|---|
| `max_num_seqs` | 160 static -> **480** adaptive | **1024** static | `vllm/engine/arg_utils.py:2432-2435` (the `>= 70 GiB` device tier), `:2671-2673` |
| top captured decode batch | 160 -> **480** | **512** | `vllm/config/vllm.py:1725` |
| per-step token budget | n/a | 8192 | `vllm/engine/arg_utils.py:2428-2431` |
| over-admission guard | reactive (retract) | `scheduler_reserve_full_isl=True` + `watermark` | `vllm/config/scheduler.py:140-152` |
| prefill/decode overlap | none (E31: `Mix batch` count exactly 0) | chunked prefill, on by default | `vllm/config/scheduler.py:252` |
| schedule/forward overlap | none | async scheduling, on by default | `vllm/config/vllm.py:1040` |

vLLM's cap is **more than twice** what the old policy derived, and it is above
every concurrency this project benchmarks. At C <= 256 the cap never binds.
There is no admission wall on vLLM to lift, so the +16.9% at C=128 that E35
attributed to this mechanism cannot be recovered by re-deriving the cap — and
E36 confirms it empirically, from the other direction: freezing prefill
admission for 8 of every 9 steps moves output throughput at C=128 by 0.3%.

Two further corrections to the E35 narrative, both from reading the installed
tree:

* vLLM's 1024 is **not** KV-derived. It is a device-memory tier
  (`>= 70 GiB and not A100 -> 1024`). On this pool the *actual* full-context
  concurrency is 113.75x (`kv_cache_utils.py:2147` logs it), so vLLM's cap sits
  ~9x above what the KV pool can hold at full context. It is over-admissive by
  construction and relies on `scheduler_reserve_full_isl` plus preemption to
  stay correct. A KV-derived cap on vLLM is therefore a *safety* change in the
  **downward** direction — the opposite sign from the old base's, and the
  correct one.
* vLLM already owns the *mechanism* for deferring prefill compute:
  `defer_prefills` (`vllm/v1/core/sched/scheduler.py:436-438, 467, 801`) skips
  in-flight prefill chunks and stops admitting new ones, leaving a pure decode
  step. But it is wired only to a fixed DP-balancing cadence that cannot fire at
  `dp_size == 1`: `EngineCore._should_throttle_prefills` returns `False`
  unconditionally (`vllm/v1/engine/core.py:474-477`) and the real implementation
  is on `DPEngineCoreProc` (`:1916-1923`). What is missing upstream is not the
  mechanism but a *policy* to drive it.

## What we built

Two independent decisions, deliberately separated because they answer different
questions and have different evidence behind them.

### 1. The cap — a minimum over every binding resource

`unieinfra_scheduler.policy.derive_admission_cap`.

The old implementation's defect was structural, not a missing case: it estimated
a physical quantity the allocator already knew exactly (`context_len //
reserve_divisor`), so it was blind three separate times — to the recurrent/Mamba
state pool (E28/E31: over-admission, retraction, C=64 hang, C=128 device-side
assert), to sliding-window budgets (`docs/design/sliding-window-kv.md`, never
fixed), and to speculative decoding's `1 + num_draft_tokens` per state chunk.

The fix is to ask the allocator. Each KV cache group reports its own per-request
cost through the `KVCacheSpec` that vLLM itself sizes the pool with
(`vllm/v1/kv_cache_interface.py`): `SlidingWindowSpec` reports a window-bounded
cost, `MambaSpec` reports a fixed state-chunk cost including
`num_speculative_blocks`, `FullAttentionSpec` reports `cdiv(max_model_len,
block_size)`. Blocks come from one shared pool
(`vllm/v1/core/kv_cache_coordinator.py:91-97`), so the joint bound is
`num_blocks / sum(per-group cost)` — the sum in the denominator *is* the minimum
over resources. Speculative lookahead adds `cdiv(num_lookahead_tokens,
block_size)` to context-proportional groups
(`vllm/v1/core/kv_cache_manager.py:393-396`).

The one distinction the old policy lacked, and the direct cause of its Mamba
bug, is `grows_with_context`. The steady-state occupancy discount is applied
**only** to context-proportional groups. A fixed-size-per-request state pool is
charged in full, because a request that has decoded one token holds exactly as
much recurrent state as one that has decoded a thousand.

On this box: 29,121 blocks, one `FullAttentionSpec` group at 256 blocks per
request at full context, divisor 4 -> **cap 457** (from vLLM's 1024). It does
not bind at C <= 256 and is not expected to change any number in the sweep; it
is a bound, and it is reported as one.

### 2. The gate — where the SLO trade is actually made

`unieinfra_scheduler.policy.PrefillGate`.

Admitting a prefill into a step makes that step slower for every request already
decoding in it. That is the TTFT/TPOT trade in one sentence, and it is the only
lever a scheduler has on this workload. The gate is a hold-then-burst controller
driving vLLM's own `defer_prefills`, with four objectives — `interactive` (off:
admission timing identical to stock), `balanced`, `throughput`, and `static`
(fixed cadence, no load or SLO signals; the ablation).

Every release condition is simultaneously a policy choice and a safety property:
`queue-empty`, `preempted-pending` (deferring a resume would turn a preemption
into a stall), `kv-pressure`, `underloaded` (a throttled step with a small batch
wastes the GPU and TTFT is all the user perceives), `ttft-budget` (the SLO term,
bounding queueing delay directly rather than through a step count), `burst-full`
(holding past a full prefill step buys no extra batching), and `hold-expired`
(a hard bound on added TTFT).

## Why `scheduler_cls`, and what it could not reach

`--scheduler-cls` resolves an arbitrary qualname
(`vllm/config/scheduler.py:180-201`) and `EngineCore` constructs it *after* the
KV cache config is finalised (`vllm/v1/engine/core.py:133, 150`), which is
exactly what makes a KV-derived cap possible from a plugin at all. We subclass
`AsyncScheduler`, not `Scheduler`, because vLLM enables async scheduling by
default here (`vllm/config/vllm.py:1040`) and `get_scheduler_cls` warns that a
`Scheduler` subclass silently loses it (`vllm/config/scheduler.py:192-198`) —
subclassing the wrong one would have turned every A/B in this work into an
async-vs-sync comparison wearing a scheduling policy's clothes.

**What the seam cannot reach: the CUDA-graph capture ladder.** It is sized in
`VllmConfig.__post_init__` (`vllm/config/vllm.py:1330`, sizes at `:1725`) and
captured inside `EngineCore._initialize_kv_caches`
(`vllm/v1/engine/core.py:320`, via `model_executor.initialize_from_config`) —
both strictly before the scheduler exists (`:150`). So a scheduler can only
*lower* the cap; raising it would admit batches the ladder cannot replay.
Extending the ladder from measured KV capacity, the second half of what the old
base did, would need a hook between memory profiling and warm-up, and no plugin
group runs there (`vllm/plugins/__init__.py:16-30`). On this hardware it is moot
— the ladder already saturates at 512 for any `max_num_seqs >= 256` — but it is
a real gap for a box where the KV pool sustains more than 512 requests.

Also worth recording: `SchedulerConfig.get_scheduler_cls` states plainly that
this interface is **not public** and compatibility is not maintained
(`vllm/config/scheduler.py:192-198`). The subclass is pinned to a vLLM version
and needs a compatibility test on every bump.

## Result, and what it means for the roadmap

Measured (E36, partial — 3 of 12 planned arms; r2/r3 and the `static` sweep were
cancelled to release the box). Out tok/s | TTFT mean ms | TPOT mean ms:

| C | stock | throughput (gate on) | interactive (gate off) |
|---|---|---|---|
| 1 | 82.5 / 60.2 / 11.9 | 82.6 / 57.4 / 11.9 | 82.6 / 59.1 / 11.9 |
| 16 | 981.1 / 430.2 / 14.7 | 981.8 / 418.6 / 14.7 | 980.0 / 462.2 / 14.6 |
| 32 | 1497.7 / 713.3 / 18.6 | 1496.2 / 728.5 / 18.6 | 1495.7 / 704.9 / 18.7 |
| 64 | 2103.4 / 916.5 / 26.9 | 2097.8 / 1339.1 / 25.3 | 2100.5 / 957.9 / 26.8 |
| 128 | 2601.6 / 1248.0 / 44.4 | 2604.5 / 1554.8 / 43.2 | 2602.2 / 1254.3 / 44.4 |
| 256 | 2719.7 / 2488.2 / 84.3 | 2740.9 / 2694.9 / 82.8 | 2719.9 / 2463.1 / 84.4 |

Three readings, in order of confidence:

1. **Throughput is flat under every policy.** At C=128 this is backed by six
   independent server launches, not one: 2601.6 / 2602.2 with the gate off
   against 2604.5 / 2605.8 / 2609.8 / 2609.8 with it on — the last two being
   diagnostics that froze prefill admission for 2 and 8 steps out of every 9.
   Nothing an admission policy can do moves this workload by more than 0.3%.
2. **The objective is a real TTFT/TPOT dial.** Same six launches: mean TTFT
   1248/1254 ms with the gate off against 1506-1837 ms with it on (+20% to
   +47%), mean TPOT 44.4 ms against 43.2-41.9 ms (-2.7% to -5.6%). That is a
   usable knob for a batch-scoring SLO. It is *not* a usable knob for the
   agent-oriented, TTFT-dominated workload this engine is positioned for, which
   is why the gate ships **off**.
3. **The `interactive` arm is a clean control.** Our class loaded, gate off, cap
   derived (457) but inert: C=128 2602.2 vs stock's 2601.6 (0.02%), TPOT 44.4
   both. The subclass costs nothing, and the delta in the `throughput` arm is
   the gate and nothing else.

One cell is suggestive and unclaimed: `throughput` at C=256 is +0.8% (2740.9 vs
2719.7). E35's stock N=3 spread at C=256 was 0.14%, so it may be outside noise —
but the treatment is N=1 and it needs repeats before anyone says so.

### Where the E35 gap actually lives

Arithmetic, from measured quantities. At C=128 stock vLLM's *median* ITL is
28.1 ms — a pure decode step — against a 44.4 ms mean, and the prefill token
rate is pinned by the workload's 1024:256 ratio, so

```
out tok/s = 128 / (t_decode_step + 512 * cost_per_prefill_token)
```

Solving with the measured 2602 tok/s gives `t_decode_step ~= 28 ms`. The old
base's 3040 tok/s at TPOT 35.1 ms requires **~21 ms**. The gap is ~25% of
per-step decode cost, which is a kernel and runtime property — and that is
exactly consistent with the gate, which can only move *when* prefill happens,
being unable to touch it. **The next experiment on this thread should profile
the batch-128 decode step, not the scheduler.**

### Roadmap consequences

`docs/design/scheduling-improvements.md` items **1-3** (decouple `max_num_seqs`
from the graph cap, pad the decode batch to the nearest captured bucket, make
the over-cap path cheap) are **already satisfied by vLLM** and should be struck
from UnieInfra's work rather than carried forward — they were written against
the TokenSpeed base's 160-request wall, which no longer exists for us.

What survives, and what this package now holds the seam for: item **4**
(reserved decode headroom with anti-thrash adaptation — the gate's
`preempted-pending` and `kv-pressure` releases are the first half of it), the
SLO-aware part of item **5**, and the agent-workload scheduling this engine
actually exists for (session affinity, prefix-aware admission). None of that is
measured by ILEN 1024 / OLEN 256 random prompts with the prefix cache flushed
between levels, which is every benchmark E1-E36 has ever run.

### Not measured

The multi-signal cap is unit-tested (20 tests, no GPU) but has **never been
exercised on a hybrid/GDN or sliding-window model on hardware**, which is
precisely the case it was written for. Until that runs, the E28/E31 regression
is prevented in theory only.
