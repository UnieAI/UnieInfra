# Design: Kernel Plugin Contract — agent-produced kernels → UnieInfra → e2e test

Status: **Scaffold implemented** (registry `origin` axis + `unieinfra-kernel`
package + discovery + one worked example, all CPU-tested; promotion tooling and
real sm_120 bodies still TODO) · Date: 2026-07-25 · Owner: royshih

> The realized structure is in [§ Realized scaffold](#realized-scaffold-what-exists-today).
> The sections above remain the design intent; the section below is what is
> actually built now and honestly flags what is not.

## Purpose
UnieInfra's kernels will increasingly be produced by **kernel-design agents**
(e.g. MIT Han Lab's [KDA](https://github.com/mit-han-lab/kernel-design-agents)).
This doc specifies the **seam** so an agent's output drops into UnieInfra and
reaches an end-to-end test **quickly and painlessly** — no per-kernel manual
wiring. It is deliberately designed *before* we integrate any workflow.

The good news: UnieInfra's engine base already has the receiving end — the
capability-based kernel registry (`tokenspeed_kernel`: `register_kernel` /
`KernelSpec` / `select_kernel` with `SelectionStrategy.AUTOTUNE`). This doc turns
that into an explicit contract.

---

## The core idea: an op's **kernel contract** IS the agent's **task contract**
A KDA task is defined by (from `kernel-design-agents/docs/agent-flow.md` and the
contest prompts): *objective · inputs/outputs · correctness requirement ·
constraints · validation command · promotion criteria*.

UnieInfra publishes, **per operator**, exactly that — a frozen **KernelContract**.
An agent targets the contract; whatever it produces is drop-in by construction.

```
KernelContract (published by UnieInfra, one per (family, mode))
├─ signature      : exact tensor args + shapes/dtypes + return   (== KDA "inputs/outputs")
├─ traits axes    : e.g. weight_dtype∈{unquant,fp8,mxfp4}, activation∈{silu,gelu}, ep∈{...}
├─ reference      : numpy/torch reference impl                    (== KDA "correctness")
├─ tolerance      : atol/rtol/matched-ratio                       (== KDA "acceptance")
├─ workloads      : representative shape set for autotune/bench   (== KDA "workload axes")
└─ capability     : min/max arch + required features (e.g. no tcgen05 on sm_120)
```
This maps 1:1 onto the contest's per-kernel spec (e.g.
`moe_fp8_block_scale_ds_routing_topk8_...`: constant/variable axes, I/O dtypes,
`--atol 1 --rtol 0.3 --required-matched-ratio 0.9`). **We adopt that spec format
as UnieInfra's `KernelContract` format** — see `contract` files below.

---

## The integration pipeline (KDA output → promoted kernel)

```
  ┌─ agent targets a UnieInfra KernelContract (the task spec)
  │
  ▼
KDA output = { kernel source (CUDA/CuTe/Triton), evidence: benchmark.csv, candidates.jsonl, profile/ }
  │
  ▼   (1) DROP-IN — scaffold + register  (minutes, mechanical)
ops/<family>/<solution>/<name>.py:
    @register_kernel(family=…, mode=…, solution="<name>", traits={…},
                     capability=CapabilityRequirement(…), signatures={…},
                     priority=Priority.PERFORMANT)
    def impl(<contract signature>): ...   # thin wrapper around the agent's kernel
  │
  ▼   (2) CORRECTNESS GATE — unit         (seconds)
run reference harness: |impl(x) − reference(x)| within tolerance across workloads
    (reuses tokenspeed_kernel/numerics/reference/)
  │  fail → reject-with-reason (recorded), never silently promote
  ▼
  ▼   (3) PERF GATE — micro               (seconds–minutes)
SelectionStrategy.AUTOTUNE benchmarks the new solution vs incumbents on the
contract's workloads; faster + correct ⇒ it becomes the selected kernel
AUTOMATICALLY. No manual routing.
  │
  ▼   (4) E2E GATE — system               (minutes)
scripts/m1_forward_smoke.py     → correctness on the real model (coherent output)
scripts/m1_concurrency_smoke.py → aggregate throughput / RPS@SLO delta
  │  pass → PROMOTED (live in UnieInfra); fail → reject-with-reason
  ▼
promoted: the op now runs the agent's kernel end-to-end.
```

Gates 2–3 already exist in the engine base (numerics reference + autotune). Gate
4 exists as our smoke/concurrency scripts. Gate 1 needs the scaffold below.

---

## What we build now (design-ahead deliverables)
1. **`contracts/<family>/<mode>.contract.(py|md)`** — one `KernelContract` per op
   UnieInfra exposes, in the contest's spec format (signature, traits, reference,
   tolerance, workloads, capability). This is the agent's task input AND the
   registry's source of truth.
2. **`ops/<family>/<solution>/_scaffold.py`** — a copy-paste plugin template: the
   `@register_kernel(...)` wrapper with the contract signature, leaving only the
   kernel body for the agent to fill.
3. **`tools/promote_kernel.py`** — a one-command runner that takes an agent's
   candidate and runs gates 2→3→4, writing a promotion record
   (`candidates.jsonl` + `benchmark.csv`) exactly like KDA's evidence files, so
   the loop is closed and auditable.
4. **`docs/design/kernel-contracts/`** — the published contracts, starting with
   the ones we need on sm_120 (below).

## First contracts to publish (our actual sm_120 gaps)
| contract | why | first filler |
|---|---|---|
| `moe.apply` weight_dtype=**unquant** bf16, silu | blocks Qwen3.6 on sm_120 (the M1 gap) | port vLLM `fused_experts` Triton (Apache-2.0) — then a KDA-tuned version later |
| `attention.mha_decode` sm_120 | the flashinfer decode winner is datacenter | Triton / KDA GDN-decode |
| `attention.gdn_decode/prefill` sm_120 | Qwen3.5/3.6 linear attn | KDA has a GDN track (B200) as a method reference |

---

## Answering "can the contest repo be used?"
- **Its kernels: no** (directly) — they target **B200 / sm_100** with
  `tcgen05`/TMEM/CuTe, which sm_120 lacks; and the repo ships **task
  prompts/contracts + method**, not finished sm_120 kernel binaries.
- **Its format & method: yes** — the contest's per-kernel spec (definition name,
  constant/variable axes, I/O dtypes, tolerance, workload count) is the **exact
  template for our `KernelContract`**, and its GDN/MoE/DSA tracks are method
  references for the kernels we need. We adopt the format; we write sm_120 bodies.

**Net:** the workflow is *not* integrated yet (per request). But the seam above
means when you run KDA later, its output lands as `ops/<family>/<solution>/…`,
auto-passes the 3 gates, and is live in an end-to-end UnieInfra run with one
command — because the op contract and the KDA task contract are the same object.

---

## Realized scaffold (what exists today)

The seam above is now backed by a concrete, CPU-tested scaffold. The key
decision: UnieInfra's own kernels live in a **separate package**,
`unieinfra-kernel`, kept apart from the forked `tokenspeed-kernel` base but
feeding the **same** registry — so upstream syncs stay clean and every kernel is
attributable, while runtime still sees exactly one kernel boundary (per
`AGENTS.md`).

### Five-layer separation

```
① platform   tokenspeed_kernel.platform   PlatformInfo / CapabilityRequirement / ArchVersion
                       │  (a kernel's capability is matched against the live platform)
② contract   unieinfra_kernel.contracts   KernelContract per (family, mode):
                       │                    signature · traits · reference · tolerance ·
                       │                    workloads · capability   (== the KDA task spec)
③ registry   tokenspeed_kernel.registry   ONE KernelRegistry singleton; register_kernel(...)
             tokenspeed_kernel.selection   select_kernel(...) ranks candidates
                       │
④ impl       kernels, tagged by ORIGIN:
             ├─ vendor       tokenspeed-kernel/…          (forked base + upstream)
             ├─ first_party  unieinfra_kernel/ops/…       (hand-written / SOTA-ported)
             ├─ kda          unieinfra_kernel/kda/…       (agent-generated)
             └─ contrib      unieinfra_kernel/contrib/…   (community)
                       │  (all four self-register into ③; no second registry)
⑤ runtime    tokenspeed runtime consumers  call select_kernel — origin-agnostic;
                                            they only know the tokenspeed boundary
```

### The `origin` / `priority` axis

Two independent knobs decide a winner among capability-matched candidates:

| knob | added? | role | effect |
|---|---|---|---|
| `priority` | pre-existing (`Priority` bands `REFERENCE..PLUGIN`, `[0,20)`) | **primary** lever | a higher band wins outright; `PLUGIN` (16–19) is reserved for out-of-tree overrides |
| `origin` | **added, additive** (`vendor`\|`first_party`\|`kda`\|`contrib`) | **lowest-weight tiebreaker** | only decides an *exact* `(oracle, objective, priority)` tie |

Ranking is now lexicographic on `(oracle, objective, priority, origin)`
(`tokenspeed_kernel.selection.ScoreBreakdown.sort_key`). Origin rank:
`first_party(3) > kda(2) > contrib(1) > vendor(0)`.

**Backward-compat invariant (proven):** every in-tree/upstream kernel keeps the
default `origin="vendor"` (rank 0). The origin dimension is therefore *constant*
across the entire existing corpus, so appending it to the sort key can never
reorder two existing kernels — a stable sort leaves them exactly where they
were. It only lets a UnieInfra kernel win a tie it would previously have lost to
registration order. (Verified two ways: a randomized 2000-trial all-vendor sweep
where the origin-aware winner is identical to the origin-blind winner in 100% of
cases, and `tests/kernels/activation/test_discovery.py`.) Use `priority`, not
`origin`, to actually change a winner.

### `unieinfra-kernel` layout

```
unieinfra-kernel/
  pyproject.toml                     # depends on tokenspeed_kernel; entry-point group
  unieinfra_kernel/
    __init__.py                      # import → discovery runs
    _discovery.py                    # walk ops/ kda/ contrib/, import each *.py (skip _*)
    ops/       <family>/<solution>.py    # first_party
    kda/       <family>/<solution>.py     # kda   (worked example: kda/activation/silu_and_mul_sm120.py)
    contrib/   <family>/…                 # contrib
    contracts/ <family>.py                # frozen KernelContract (example: contracts/activation.py)
  tests/kernels/<family>/            # per-kernel correctness+perf tests
```

### Discovery mechanism

Importing `unieinfra_kernel` calls `_discovery.discover()`, which recursively
`pkgutil.walk_packages` the `ops/`, `kda/`, and `contrib/` trees and imports
every module whose basename does **not** start with `_`. Each module's
`@register_kernel(...)` runs as an import side effect and self-registers into the
shared `tokenspeed_kernel` `KernelRegistry` (imported from tokenspeed-kernel — no
second registry is created). Discovery is idempotent (a one-shot guard + the
module cache) and per-module failures are logged and skipped so one broken
experimental kernel never aborts startup. A `unieinfra_kernel.plugins`
entry-point group lets external distributions add modules to the same scan.

**Drop-in result:** dropping a `@register_kernel`-decorated file under
`ops/`/`kda/`/`contrib/` makes it registered and selectable on next import —
zero edits to the registry or selection core.

### Startup wiring

`tokenspeed/runtime/execution/model_runner.py::_load_first_party_kernels()` does
a guarded `import unieinfra_kernel` once per process, before the model is built
and kernels are selected. A missing package or a failing kernel is tolerated
(logged, not fatal), so a normal engine start now also loads first-party/KDA
kernels when the package is present.

### Benchmark ablation: pinning a kernel by name/origin

The autotune path picks the fastest correct candidate automatically, but for a
**trajectory** (KDA output → discovery → pin → benchmark → show the win) the
harness pins a specific kernel and benchmarks it head-to-head:

* By **name** (exact kernel), any of the existing override channels:
  * env: `TOKENSPEED_KERNEL_OVERRIDE_ACTIVATION_SILU_AND_MUL=kda_silu_and_mul_sm120`
  * scoped: `with kernel_override("activation", "silu_and_mul", "kda_silu_and_mul_sm120"): ...`
  * config: `overrides: { activation.silu_and_mul: { name: kda_silu_and_mul_sm120 } }`
* By **origin** for ablation, filter candidates on `KernelSpec.origin` (e.g.
  benchmark the `vendor` incumbent vs the `kda` challenger for the same op) and
  pin each by name. This yields the before/after pair the CHANGELOG trajectory
  needs. (An `origin:`-keyed override shorthand is a natural follow-up; today
  origin is a filter/tiebreaker, and pinning is by name.)

### Still TODO (not scaffolded)

* Real sm_120 kernel bodies (the example ships a portable Torch reference body).
* `tools/promote_kernel.py` — the gate-2→3→4 runner writing `candidates.jsonl` /
  `benchmark.csv` promotion records.
* `ops/<family>/<solution>/_scaffold.py` copy-paste templates.
* Publishing the first real contracts (MoE unquant bf16, attention/GDN on sm_120).
