<!-- SPDX-License-Identifier: MIT -->
# `unieinfra-scheduler` — UnieInfra's admission policy for vLLM

An out-of-tree vLLM scheduler carrying UnieInfra's **SLO-aware, KV-derived
admission policy**. It loads through vLLM's own `--scheduler-cls` seam. No file
under `third_party/vllm/` or in the installed vLLM tree is modified, and with
the flag absent nothing in this package is imported.

```bash
pip install -e unieinfra-scheduler
UNIEINFRA_SCHED_OBJECTIVE=throughput \
  vllm serve <model> \
    --scheduler-cls unieinfra_scheduler.vllm_scheduler.AdaptiveScheduler
```

## Why this exists, and what it is *not*

It is the scheduling half of the vLLM re-base, following the kernel plugin
(E34). The brief was to port the old TokenSpeed base's `derive_adaptive_max_batch`
— the one measured, repeatable advantage that base had (E19, E25, E35).

**The premise did not survive contact with vLLM 0.25.1, and that is the most
important thing in this README.** On the benchmark box (1x RTX PRO 6000, 96 GB)
stock vLLM already runs `max_num_seqs = 1024`
(`vllm/engine/arg_utils.py:2432-2435`, the `>= 70 GiB` device tier), more than
twice the 480 the old policy derived, and its CUDA-graph ladder is already
captured to 512 (`vllm/config/vllm.py:1725`). At every concurrency we benchmark
(C <= 256) **the cap never binds**, so there is no admission wall to lift. See
`docs/experiments-log.md` E36 for the measurement that settles it.

So this package does two things, and claims only what it measured:

1. **Derives the cap correctly** (`policy.derive_admission_cap`). On vLLM this
   is a *safety* property, not a throughput lever: it can only lower
   `max_num_seqs`, never raise it, because the graph ladder is finalised before
   any scheduler exists.
2. **Trades TTFT against TPOT on an SLO objective** (`policy.PrefillGate`).
   Measured: at C=128 the objective moves mean TTFT by ~50% and mean TPOT by
   ~6%, at no throughput cost in either direction. It does **not** recover the
   +16.9% throughput E35 attributed to the old base.

## The cap: a minimum over every binding resource

The old implementation estimated live KV per request as
`context_len // reserve_divisor` and had to be patched three separate times
because that estimate is blind by construction:

| Blind spot | Symptom | Where it was patched |
|---|---|---|
| Recurrent/Mamba state pool | over-admission, retraction, C=64 hang, C=128 device-side assert on a GDN model | E28 / E31, commit `3ad34a3` |
| Sliding-window budgets | ~8x over-charge for window-bounded layers -> severe under-admission | `docs/design/sliding-window-kv.md` (never fixed) |
| Speculative decoding | each state chunk costs `1 + num_draft_tokens` more | inherited from pool sizing only; the cap itself stayed blind |

The fix is structural, not another special case: **ask the allocator what one
request costs.** Each KV cache group reports its own per-request cost through
its `KVCacheSpec.max_memory_usage_bytes` / `page_size_bytes`
(`vllm/v1/kv_cache_interface.py`) — the same objects vLLM sizes the pool with.
A `SlidingWindowSpec` therefore reports a window-bounded cost, a `MambaSpec`
reports its fixed state-chunk cost *including* `num_speculative_blocks`, and
neither needs a branch here. The blocks come from one shared pool
(`vllm/v1/core/kv_cache_coordinator.py:91-97`), so the joint bound is
`num_blocks / sum(per-group cost)` — the sum in the denominator *is* the
minimum over resources.

One distinction the old policy lacked and that its Mamba bug came from:
`grows_with_context`. A steady-state occupancy discount (`reserve_divisor`) is
applied **only** to context-proportional groups. A fixed-size-per-request state
pool is charged in full, because a request that has decoded one token holds
exactly as much recurrent state as one that has decoded a thousand.

## The gate: where the SLO trade is made

Admitting a prefill into a step makes that step slower for every request already
decoding in it. That is the whole TTFT/TPOT trade, and vLLM already owns the
mechanism to defer it — `defer_prefills`
(`vllm/v1/core/sched/scheduler.py:436-438, 467, 801`) skips in-flight prefill
chunks and stops admitting new ones, leaving a pure decode step. But it is wired
only to a fixed DP-balancing cadence that cannot fire at `dp_size == 1`: the
base engine's `_should_throttle_prefills` returns `False` unconditionally
(`vllm/v1/engine/core.py:474-477`) and the real implementation lives on
`DPEngineCoreProc` (`:1916-1923`).

`AdaptiveScheduler.schedule` supplies that argument itself, from a hold-then-burst
controller whose every release condition is also a safety property:

| Release | Signal | Why |
|---|---|---|
| `queue-empty` | `len(waiting) == 0` | nothing to batch |
| `preempted-pending` | a waiting request is `PREEMPTED` | deferring a resume turns preemption into a stall |
| `kv-pressure` | `kv_usage >= 0.9` | when the pool is nearly full, drain, don't batch |
| `underloaded` | `len(running) < min_decode_batch` | a throttled step with a small batch wastes the GPU, and TTFT is all the user perceives |
| `ttft-budget` | oldest wait >= budget | the SLO term: bounds queueing delay directly |
| `burst-full` | queued prompt tokens >= one step's budget | holding longer buys no extra batching |
| `hold-expired` | `max_hold_steps` reached | hard bound on added TTFT |

Objectives: `interactive` (**the default** — gate off, admission timing
identical to stock), `balanced`, `throughput`, and `static` (fixed cadence,
**no** load or SLO signals — the ablation). The gate ships **off**: E36 measured
it as a TTFT/TPOT dial, not a throughput win, so it has to be asked for.

## Configuration

All environment variables, so no vLLM CLI surface changes:

| Variable | Default | Meaning |
|---|---|---|
| `UNIEINFRA_SCHED_OBJECTIVE` | `interactive` (gate **off**) | `interactive` \| `balanced` \| `throughput` \| `static` |
| `UNIEINFRA_SCHED_RESERVE_DIVISOR` | `4` | steady-state occupancy divisor for context-proportional groups |
| `UNIEINFRA_SCHED_HOLD_STEPS` | per objective | override `max_hold_steps` |
| `UNIEINFRA_SCHED_TTFT_MS` | per objective | override the TTFT budget |
| `UNIEINFRA_SCHED_CAP` | `on` | `off` keeps vLLM's `max_num_seqs` |
| `UNIEINFRA_SCHED_LOG_EVERY` | `0` | log the gate histogram every N steps |

## Layout

* `unieinfra_scheduler/policy.py` — the policy. Stdlib only; no vLLM, no torch,
  no GPU, so it is unit-tested directly (`tests/test_policy.py`).
* `unieinfra_scheduler/vllm_scheduler.py` — the only module that imports vLLM.

It is a **sibling package to `unieinfra-kernel`, not part of it**, on purpose:
`AGENTS.md` makes `unieinfra-kernel` / `tokenspeed-kernel` the kernel boundary,
and a scheduling policy is not a kernel. Mixing them would put a runtime policy
behind an import that pulls in Triton.

## Known limits of the `scheduler_cls` seam

* **The CUDA-graph capture ladder is out of reach.** It is finalised in
  `VllmConfig.__post_init__` (`vllm/config/vllm.py:1330`, sizes at `:1725`),
  and capture itself runs inside `EngineCore._initialize_kv_caches`
  (`vllm/v1/engine/core.py:320`) — both strictly *before* the scheduler is
  constructed (`:150`). A scheduler can therefore only lower the cap. Extending
  the ladder from KV capacity, which is what the old base did, would need a
  second seam that runs between memory profiling and warm-up; no plugin hook
  exists there.
* `SchedulerConfig.get_scheduler_cls` warns that this interface is **not
  public** (`vllm/config/scheduler.py:192-198`), so the subclass is pinned to a
  vLLM version. It also disables async scheduling for anyone subclassing
  `Scheduler`; we subclass `AsyncScheduler` for exactly that reason.
