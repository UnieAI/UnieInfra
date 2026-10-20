# The UnieInfra Extension Spine

Status: **Landing** · Date: 2026-07-25 · Owner: royshih

This document describes how UnieInfra attaches to its vLLM base — the thing that
makes this tree *ours* rather than a patched copy of somebody else's.

## The problem this solves

UnieInfra owns its vLLM source outright. There is no upstream to sync with and no
plugin API to respect, so any file can be edited at will. That freedom is exactly
what went wrong in the first round of integration. Five subsystems were absorbed,
and each grew its own mechanism:

| Subsystem | How it attached | Its switch |
|---|---|---|
| Dense-linear kernels | `unieinfra_gemm.py` + `if AVAILABLE:` in `layers/utils.py` | `UNIEINFRA_GEMM` |
| Attention backends | `unieinfra_selection.py` + a branch in `v1/attention/selector.py` | `UNIEINFRA_ATTENTION_*` |
| Quantization | `unieinfra_quant.py` + edits across three quant modules | (none) |
| Serve-time defaults | `cli/unieinfra_defaults.py` gated on `sys.argv[0]` | `UNIEINFRA_*` per default |
| Draft-logit grammar | `v1/spec_decode/draft_grammar.py` + a branch in the proposer | (none) |
| Scheduler policy | a **separate pip package**, `unieinfra-scheduler` | `scheduler_cls` |

Twenty-three files touched, eighteen of them upstream source. Individually each
choice was defensible; together they produced three costs that compound:

1. **Adding a capability starts with a research question.** Which of the five
   shapes applies? The answer depends on which subsystem you happen to be in.
2. **"What does UnieInfra change, and why did it pick that here?" has no answer**
   short of reading eighteen files — and no answer at all for the parts gated on
   a runtime probe, where the choice depends on the machine.
3. **Two of the six have no off switch**, so isolating a regression to our code
   means editing source.

The owner's brief was *"make vLLM's architecture as extensible as tokenspeed,
while keeping vLLM's functionality."* Point 1 is precisely the gap: tokenspeed is
extensible not because it has more seams, but because it has **one seam repeated**
— `register_kernel` with a capability requirement, `ops/<family>/<solution>` on
disk, one `select_kernel` to choose. Learn it once, apply it everywhere.

## The design

One convention, in `vllm/unieinfra/`.

An **extension point** is a named decision with a written contract:

```python
ADMISSION_CLAMP = extension_point(
    "scheduler.admission_clamp",
    contract="(vllm_config, kv_cache_config) -> int | None. Return the value "
             "max_num_seqs should be lowered to ..., or None to leave it alone.",
)
```

An **implementation** declares what it is and what it needs:

```python
@provide(ADMISSION_CLAMP, origin="first_party", priority=Priority.PERFORMANT,
         requires=[has_state_pool], note="Lets hybrid models start where upstream raises.")
def clamp_to_binding_pools(vllm_config, kv_cache_config) -> int | None: ...
```

An upstream call site consults it in one line, and keeps its own code as the
fallback:

```python
clamp = unieinfra_resolve(ADMISSION_CLAMP)
if clamp is not None:
    ...
```

`resolve()` filters candidates by their requirements, ranks by priority band then
by origin, memoises, and returns the winner — or `None`, which always means *run
the upstream path unchanged*.

### Four properties, each load-bearing

**Resolution happens once, outside traced code.** `resolve()` returns a plain
callable that the caller stores; selection never executes inside a
`torch.compile` region. This is not hypothetical caution — an earlier integration
measured a −1.99% regression that survived even in an arm where the selector was
never called, proving the cost was graph shape rather than dispatch. Keeping
resolution at setup time keeps that class of cost out of the design.

**Every point degrades to upstream.** No viable implementation, or the flag
turned off, yields `None` and the original code runs. This is what allows a call
site to be a one-line addition rather than a rewrite of its logic, and it is what
makes bisecting our contribution possible without editing source.

**Every flag is derived, not declared.** `scheduler.admission_clamp` implies
`UNIEINFRA_SCHEDULER_ADMISSION_CLAMP`. An operator can predict the flag from the
name, and no hand-maintained table can drift out of date.

**Every point is introspectable.** `describe()` prints each point, its contract,
every registered implementation, which requirements failed and why, and what was
chosen. The question that previously required reading eighteen files is now one
call — including for the machine-dependent parts, where reading source could not
have answered it anyway.

### Priority bands and origin

Both are taken from `unieinfra_kernel.selection`, unchanged, so the two
registries rank things the same way:

`REFERENCE(0) < PORTABLE(4) < PERFORMANT(8) < SPECIALIZED(12) < PLUGIN(16)`, with
`PLUGIN` reserved so out-of-tree code always has headroom over anything in-tree.
Ties break on origin — `vendor < contrib < kda < first_party` — which encodes who
we trust to be right on hardware we have not personally tested.

The bands are duplicated in `spine.py` rather than imported, because the kernel
package is optional in a vLLM process and the spine is not.

## First citizen: the admission clamp

The spine landed with a real problem attached to it, not a placeholder.

A hybrid model — Qwen3.5/3.6, or anything mixing attention with a linear or
state-space layer — allocates a second pool alongside KV: one block per
*sequence*, not per token. That pool binds long before KV does. vLLM's default
`max_num_seqs=1024` is sized against KV, so upstream raises:

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (732).
```

Measured, and kept in `benchmark/results/e51_bootfail/`: **stock vLLM cannot serve
Qwen3.6-35B-A3B on an RTX PRO 6000 at default settings.** Not slower — it does not
start. The operator's only recourse is to guess a number the engine has already
computed.

`scheduler.admission_clamp` lowers `max_num_seqs` to the real ceiling and logs
what it did and how to opt out. It never raises the value: an operator who asked
for fewer sequences has made a decision, and the clamp only moves the ceiling
down to what the hardware can honour.

The generalisation matters as much as the fix. The same policy on the previous
engine base clamped against the KV pool alone and was wrong — because the binding
pool was a different one. So `sequence_capacity()` takes the minimum across every
pool that constrains sequence count, and a new pool kind is added there rather
than at a call site.

It is wired at two places, which is worth explaining because one is not obviously
necessary. The model runner applies it before cudagraph resolution, since that is
where upstream refuses. Engine core applies it again after KV-cache init and
before the scheduler is constructed, because the worker's mutation does not cross
a process boundary under multiproc tensor parallelism — under TP=1 the two are the
same process and the second application is a no-op.

## Migration plan

The spine is in place; the six subsystems are not on it yet. Order is by risk,
lowest first, and each step must leave the tree working:

1. **`scheduler.admission_clamp`** — done, and it was new code, so nothing regressed.
2. **`spec_decode.draft_grammar`** — self-contained, already measured (draft
   tokens 3.00 → 1.83 with accepted length unchanged, so it is a free pre-filter).
3. **`serve.defaults`** — currently gated on `sys.argv[0]`, which is a fine
   *policy* but a poor *mechanism*; the gate becomes a requirement predicate.
4. **`linear.unquantized_gemm`** — the measured-sensitive one; the migration must
   not change where resolution happens relative to compilation.
5. **`attention.backend_selection`** — the largest at ~1,000 lines, and the one
   with the most existing structure to preserve.
6. **`quant.method_selection`** — spread across three upstream modules, so it is
   the one where the diff shrinks most.

Success is measurable and should be measured: `git diff` against the vendored base
should read as a list of one-line hooks, and every UnieInfra behaviour should be
listed by `describe()` and switchable by its derived flag.

## What this is not

It is not a plugin API. There is no stable ABI, no out-of-tree contract to honour,
no versioning promise. Extension points may be renamed whenever the design
improves. `PLUGIN` priority headroom exists for local experiments, not for a
third-party ecosystem we have not committed to supporting.

It is also not a claim that indirection is free. A point that will only ever have
one implementation is worse than a direct call, and the honest reason to declare
one anyway is the introspection and the off switch — which is why `describe()`
and the derived flag are requirements of the design rather than conveniences
layered on top. Where neither is worth having, the right answer is to edit the
file directly and not declare a point at all.
