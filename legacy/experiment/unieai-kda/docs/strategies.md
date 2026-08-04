# Built-in strategies

The AutoML analogy holds at the level of structure: a **search space**, a
**strategy** that proposes points in it, an **evaluator** that scores them, and a
**promotion rule**. It breaks down on cost. A hyperparameter trial is seconds
and independent; a kernel trial here is a 90-second model load plus a 4-minute
measurement, and the measurement can be wrong in ways that look like signal.

Two consequences shape every strategy below:

1. **Trials are expensive, so proposals should be informed.** Random search is
   not competitive when a trial costs 6 minutes. Every strategy that can consume
   profiler evidence should.
2. **Trials can lie, so the loop must be able to distrust itself.** Strategies
   report a `confidence` alongside a result, and the bandit weights by
   *admissible* yield -- a candidate whose measurement was rejected by a gate
   counts as cost with no reward, which correctly penalises strategies that
   produce hard-to-measure candidates.

---

## `sweep` — exhaustive over a small discrete space

Enumerate a handful of known alternatives and measure all of them.

Unglamorous, and the highest hit rate we have evidence for. Several real wins on
this hardware were of the form *"this capability-keyed table has no entry for
sm_120"*:

- `SYMM_MEM_ALL_REDUCE_MAX_SIZES` — no `"12.0"`, symm-mem all-reduce off
- `CUSTOM_ALL_REDUCE_MAX_SIZES` — no `"12.0"`, generic 8 MiB used untuned
- `SP_MIN_HIDDEN_SIZE` — only `{90, 100}`, sequence parallelism unreachable
- `ll_bf16`'s cuteDSL path — cannot lower to sm_120, kernel disabled

None of these needed a new kernel. They needed someone to enumerate the options
and measure. **Run this strategy first on any new subsystem**; it is cheap and
it establishes whether the interesting work is selection or authorship.

Use when: the space is < ~20 points, or the space is a policy table.

---

## `refine` — profile, diagnose, change one thing, re-measure

The KDA loop proper. Serial, evidence-driven, low variance.

1. Profile the current candidate (`ncu`, sections per the ncu-report skill).
2. Match the signal to a diagnosis (bandwidth-bound / occupancy-bound /
   latency-bound / imbalanced). **Do not skip to a fix**; the ncu rule engine's
   `Est. Speedup` ranking usually points at the answer.
3. Apply exactly one change.
4. Re-measure through the gates.

The one-change rule is not fastidiousness. Two changes at once produce a delta
you cannot attribute, and attribution is what the memory stores.

Use when: there is a clear hot kernel and a profiler can see it. High hit rate,
but bounded by whatever the current design can be tuned into.

---

## `evolve` — mutate a population over a parameterised space

Maintain a pool of variants; mutate and recombine the survivors. This is where
the literature concentrates (EvoEngineer, Kernel-Smith, AVO, AdaExplore).

Our variant space for a kernel candidate:

- tile shape (BLOCK_M / BLOCK_N / BLOCK_K)
- num_warps, num_stages
- fusion boundary (which adjacent ops are absorbed)
- backend (CuteDSL / Triton / CUTLASS / vendor library)
- layout and swizzle choice
- for MoE specifically: expert-grouping granularity, whether routing is fused
  into the first GEMM, gather/scatter strategy

Mutation must be **gated on correctness before performance**, because most
mutations in this space produce a kernel that is fast and wrong.

Use when: `refine` has plateaued and the space has continuous or combinatorial
structure. Expensive; needs a cheap correctness oracle to be viable.

---

## `port` — re-qualify an existing kernel on a new shape

Take a kernel that was qualified elsewhere and measure it here.

Called out separately because it is UnieInfra's largest unexplored space and it
does not look like optimization work. `unieinfra_gemm.py` states its kernels
were qualified on **batch-1 Qwen3-8B decode**; measurement showed the layer is
exactly inert at 30B (+0.50% end-to-end, inside noise) because the registry
declines those shapes and falls back upstream. The kernels are not bad at 30B.
They have never been asked.

The port loop is: enumerate the (model, shape, batch) cells the registry
declines, measure the fallback against the specialised kernel where one can be
made to apply, and record which cells are worth authoring for.

Use when: a kernel layer measures as neutral. Neutral usually means *absent*,
not *equal*, and the two call for opposite responses.

---

## `bandit` — allocate GPU-hours across the above

A strategy selector, not a strategy. Tracks per-strategy **admissible yield per
GPU-hour**: total promoted improvement divided by wall-clock spent, counting
gate-rejected measurements as pure cost.

That last clause is the important one. A strategy that proposes candidates whose
measurements keep getting rejected -- because they are unstable, or because they
change tokens-per-step and break naive counting -- is genuinely less useful than
its raw hit rate suggests, and the accounting should say so.

Start with round-robin until each strategy has ~5 admissible trials; only then
weight.

---

## Not implemented, and why

**RL over kernel edits** (Kevin, CUDA-L1, AutoTriton, TritonRL). Needs a reward
signal that is cheap and trustworthy. Ours is neither: 6 minutes per trial, and
this session produced three wrong headline results from measurement bugs. RL
would optimise against the bug faster than we could notice it. Revisit when the
gates have a track record.

**Multi-agent teams** (Astra, STARK, cuPilot, CudaForge). Plausible, but the
bottleneck here has not been idea generation. Reconsider once a single-agent
loop is throughput-limited rather than evidence-limited.

**Cross-task skill memory** (KernelBlaster, KernelSkill). The right long-term
shape, and `kda/memory.py` is deliberately structured so candidates carry parent
links and diagnoses rather than only scores. Extraction into reusable skills
should wait until there are enough promoted candidates to generalise from.
