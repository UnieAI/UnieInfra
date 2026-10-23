## Why

We are about to port a dozen implementations from vLLM / SGLang / TRT-LLM / TurboMind
1:1 (see `unieinfer/xref/70-PORTS.md`). The rule for that work is **every strategy is
kept** — a ported implementation is registered alongside the existing one and the engine
picks the best per hardware and objective, rather than one replacing the other.

The kernel layer already supports this (`tokenspeed_kernel` has registration, capability
gating, objectives, heuristic and autotune selection). Two things block applying the rule
uniformly:

1. **The C++ scheduler has no selection layer at all.** Preemption victim choice, the
   mixed prefill/decode order, and chunk splitting are hard-coded or plain booleans on
   `SchedulerConfig`. Porting an alternative policy there today means editing the one
   implementation in place — exactly what the rule forbids.
2. **Selection is not observable or safely overridable end to end.** When a strategy is
   unreachable on the current hardware there is no way to ask why, and an override that
   cannot be satisfied degrades silently. This has cost us real time: a fused GDN prefill
   kernel in SGLang is admitted by a `major >= 10` check but needs SM100 tensor-core
   instructions that this hardware does not have, so the gate lets it through into a
   kernel that cannot run. The mirror-image failure exists in vLLM, where a closed
   `family(100)` table silently excludes this hardware from features it could use.

Now, because the port backlog is about to multiply the number of coexisting
implementations, and every one of them lands on this contract.

## What Changes

- Establish the **strategy-selection contract** that all ported implementations follow:
  register alongside, never replace; selection is deterministic, observable, and
  overridable; an unsatisfiable override fails loudly.
- Require hardware eligibility to be expressed with **both bounds**, so a strategy can be
  scoped to the architectures it was actually written for. This is the direct fix for the
  failure class described above.
- Extend the contract to the **C++ scheduler**: preemption, mixed-batch ordering, and
  chunk splitting become named, selectable policies rather than hard-coded behaviour.
- Make **autotune results persistable and pinnable**, so a measured choice can be frozen
  for a release and reproduced.
- Require that **adding a strategy does not change the existing default** unless that is
  the explicit intent of the change.

Non-goals: porting any specific implementation (that is `70-PORTS.md`'s backlog, one
change each), and optimizing any kernel. This change only establishes the contract the
ports land on.

## Capabilities

### New Capabilities
- `strategy-selection`: how multiple interchangeable implementations of the same
  operation coexist, how one is chosen, and how an operator observes, overrides, and pins
  that choice — across kernels, runtime backends, and scheduler policies.

### Modified Capabilities

(None — this is the first spec in the project.)

## Impact

- `tokenspeed-kernel/python/tokenspeed_kernel/{registry,selection,platform}.py` — already
  implements most of the contract; gains observability and override-failure requirements.
- `tokenspeed-scheduler/csrc/scheduler/` — gains a policy registration and selection seam;
  `operations/forward.cpp` preemption and mixed-batch decisions move behind it.
- `python/tokenspeed/runtime/utils/server_args.py` — operator-facing selection and pinning
  flags.
- `python/tokenspeed/runtime/layers/attention/registry.py`,
  `python/tokenspeed/runtime/sampling/registry.py` — runtime backend registries align with
  the same contract.
- Every change in the `unieinfer/xref/70-PORTS.md` backlog depends on this one.
