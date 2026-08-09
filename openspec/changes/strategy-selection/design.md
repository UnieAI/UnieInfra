## Context

Three layers currently answer "which implementation runs?", and they are at very different
maturity levels.

| Layer | Where | State |
|---|---|---|
| Kernel | `tokenspeed-kernel/python/tokenspeed_kernel/{registry,selection,platform}.py` | Registration, capability bounds, priority bands, objectives, heuristic + autotune strategies, YAML/env/context overrides, cached selection. **Satisfies most of the spec already.** |
| Runtime backend | `python/tokenspeed/runtime/layers/attention/registry.py`, `python/tokenspeed/runtime/sampling/registry.py` | Name → class map, arch set, chosen by a server arg with an `auto` default. No objective, no measured selection, no exclusion reporting. |
| Scheduler | `tokenspeed-scheduler/csrc/scheduler/` | **Nothing.** Decisions are hard-coded or booleans on `SchedulerConfig`. |

The kernel layer is the reference implementation of the contract. The work is to state
the contract, close its two real gaps, and extend it downward to the scheduler.

## Goals / Non-Goals

**Goals**

- One contract that all three layers answer to.
- The scheduler gains a policy seam that is cheap enough that porting an alternative
  policy is additive.
- Selection failures become legible: a strategy that cannot run says so, and says why.

**Non-Goals**

- Porting any specific implementation. Each item in `unieinfra/xref/70-PORTS.md` is its
  own change.
- Unifying the three layers behind one API. They have different lifetimes (kernel
  selection is per-shape and cached; scheduler policy is per-process) and forcing one
  abstraction would cost more than it returns.
- Changing any current default.

## Decisions

### 1. The kernel layer is the reference; do not rebuild it

`register_kernel` + `CapabilityRequirement(min_arch_version, max_arch_version)` +
`SelectionStrategy.{HEURISTIC,AUTOTUNE}` + `select_kernel(solution=, override=)` already
express the contract. Ported kernels register as an additional `solution` under the
existing naming convention, which names the **implementation lineage** rather than the
source project (`triton`, `gluon`, `fa3`, `flashinfer_cutlass`, `deep_gemm`, `cuda`, …).

Two gaps to close:

- **Exclusion reporting.** Selection filters candidates before ranking; the reason a
  candidate disappeared is not currently retrievable. Add a report that, for one
  `(family, mode, signature, platform)`, returns every registered candidate with its
  outcome: eligible-and-ranked, excluded-by-capability, excluded-by-signature-or-trait.
- **Loud override failure.** An override or `solution=` request that survives no candidate
  must raise, naming the request and the rule it failed, rather than falling through to
  ranking.

### 2. Hardware eligibility must carry an upper bound, and reviewers must ask for it

`CapabilityRequirement` already has `max_arch_version`; `ops/attention/msa.py` uses
`(10,0)`–`(10,3)` to scope a kernel to SM100. The requirement in the spec exists because
both failure directions are real and were found in reference engines:

- an open lower bound admitting hardware the kernel cannot run on, and
- a closed capability table excluding hardware the feature would work on.

Convention: a kernel that uses architecture-specific instructions declares the range it
was written and tested for. A kernel that is portable declares only a lower bound.
Reviewers reject a registration that uses architecture-specific instructions without an
upper bound.

### 3. Scheduler policies: a name → factory registry, resolved once at construction

The scheduler runs once per rank and must stay deterministic across ranks
(`csrc/scheduler/operations/forward.cpp` tie-breaks on `Id()` for exactly this reason,
because a rank-varying schedule deadlocks the collective). That rules out measured or
adaptive selection inside the scheduler.

Design: a small registry per decision point, resolved **once at scheduler construction**
from a name carried on `SchedulerConfig`, producing a policy object the hot path calls
through. No per-step lookup, no dynamic switching.

First decision points, in the order they are worth doing:

| Decision point | Today | Policy interface |
|---|---|---|
| Preemption victim | hard-coded "largest `TokenSize()`" (`csrc/scheduler/operations/forward.cpp:1277`) | choose a victim from the retractable set |
| Mixed prefill/decode order | `config_.enable_mixed_prefill_decode` boolean (`csrc/scheduler/operations/forward.cpp:1145`) | assign a priority to a request given its state |
| Chunk splitting | implicit in `schedulePrefillFirstChunk` / `schedulePrefill` | choose this step's chunk size given the remaining budget |

Preemption is the first one to convert: it is a pure function of the candidate set, has no
interaction with the FSM, and there are two documented alternatives to port (vLLM's
"latest-arrived" and SGLang's `retraction_key`).

Alternative considered and rejected: exposing the policy as a Python callback. It would
put a Python call on the scheduler's hot path and break the rank-determinism argument,
because the Python side is not guaranteed to be bit-identical across ranks.

### 4. Autotune results persist as a file keyed by an environment fingerprint

`AutotuneParams` exists but results live only in the in-process cache. Persist to a file
whose key covers: device name and architecture, the set of registered candidate names for
that operation, and the operation's shape signature. On load, a key mismatch is reported
and the file is not applied — never silently, because a stale pin that still parses is
worse than no pin. The file is what a release pins and what a benchmark run publishes.

### 5. Runtime backend registries align, but are not merged

`register_backend` gains the same exclusion-reporting surface so that
"why is this attention backend not available" has one answer shape across layers. It does
not gain objectives or autotune in this change — no ported item needs them yet.

## Risks / Trade-offs

- **The scheduler seam adds indirection to a hot path.** Mitigated by resolving once at
  construction and calling through a plain virtual, and by converting one decision point
  first so the cost is measurable before the others follow.
- **Loud override failure changes existing behaviour** for anyone relying on a bad
  override silently degrading. This is intended; it is the failure mode that cost us time.
  Ship it with the exclusion report so the error message can say what to use instead.
- **Requiring upper bounds is a review-time convention**, not something the type system
  enforces. Accepted: encoding "this kernel uses instruction X" mechanically is a much
  larger project.

## Migration Plan

1. Kernel layer gaps (report + loud failure) — no behaviour change to selection outcomes.
2. Autotune persistence — opt-in, off by default.
3. Scheduler preemption policy seam, with the current behaviour registered as the default
   policy under its own name. Verified by showing scheduling decisions are unchanged.
4. Remaining scheduler decision points, one per change, same pattern.

Each step is independently shippable and none changes a default.

## Open Questions

- Where should the autotune pin file live so that a container rebuild does not lose it,
  and so that a benchmark run can publish one alongside its results?
- Should the mixed prefill/decode order stay a boolean once it is a policy, given that
  today's `false` branch is not "prefill first" but "this step does prefill only"?
