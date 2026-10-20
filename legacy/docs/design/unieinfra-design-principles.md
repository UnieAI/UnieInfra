<!-- English canonical. Chinese mirror: unieinfra-design-principles.zh.md (keep in sync). -->
# UnieInfra Design Principles

Status: **Founding document** · Date: 2026-07-25 · Applies to: the vLLM-based engine
(`unieinfra-vllm`) and the first-party kernel layer (`unieinfra-kernel`).

UnieInfra is an **agent-oriented inference engine** that absorbs valuable components from
any engine **as owned in-tree source**, never as library dependencies. The engine base is a
hard fork of vLLM; there is no upstream sync. See `vllm-base-assessment.md` for how that
decision was reached.

This document states the principles the architecture is held to. Three of them are adopted
from the published design of `tokenspeed-kernel`
([PyTorch blog](https://pytorch.org/blog/lightseek-tokenspeed-kernel/)), whose kernel
subsystem we forked and have been running all along. We adopt the principles **and record
where that implementation fell short of them**, because we measured those gaps first-hand
and must not reproduce them.

---

## P1. Multi-silicon support is foundational

> *"Multi-silicon support must be foundational"* — hardware platforms are understood
> directly by the kernel system, not scattered through conditional logic.

**What this requires.** A kernel declares the *capabilities* it needs. The platform layer
reports the capabilities a device *has*, derived from measured device properties. Selection
is the intersection. No kernel, model file, or runtime path may branch on an architecture
version number.

**How the original fell short — measured, not assumed (E30):**

- `CapabilityRequirement.required_features` had **zero real uses** in the entire kernel
  package; the only occurrence was a docstring example. The whole feature vocabulary
  (`memory:tma`, `compute:cluster`, `tensor_core:f4`) gated *nothing*, and **100% of arch
  gating was raw version comparison**.
- `ArchVersion` had no concept of tcgen05/TMEM, so sm_120 satisfied **every**
  `min_arch=(9,0)` and `(10,0)` guard and was advertised `compute:cluster` identically to
  B200. Four kernels won selection on sm_120 and then hard-failed at runtime, purely
  because a `max_arch` cap present on their siblings had never been copied.
- `platform.max_shared_memory_per_sm` read **0 on every NVIDIA device** — it queried a
  torch property that has never existed — so every tile-sizing decision downstream of it
  was blind (E28). That one bug blocked both bf16 MoE and Gemma 4's head_dim-512 layers.

**Rules we impose.**

1. A `CapabilityRequirement` with a `min_arch_version` and **neither** a `max_arch_version`
   **nor** `required_features` is **invalid** — make it impossible to write, not merely
   discouraged. That single rule would have prevented every mis-gate above.
2. Capabilities come from **measured device properties or an explicit arch→feature table**,
   never from a version comparison. `compute:cluster` is a driver query
   (`CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH`), not an inference from `major >= 9`.
3. Add a **build/artifact capability tier**. `min_arch_version` cannot express "does the
   installed binary contain SASS for this device", which is the real cause of several
   unresolved gates. Probe it.
4. Vendor must be explicit. AMD gfx950 is `ArchVersion(9,5)` and CDNA5 is `(12,5)` — the
   same numbers NVIDIA uses — so any requirement omitting `vendors=` is latently
   cross-vendor.
5. **Correct classification of a part we have never seen is the test.** sm_103 (B300) is
   newer than the sm_100 most gates were written against; if a new part needs a code change
   to be classified correctly, the model is wrong.

**Where these five rules live in code (E45).** All five are enforced, not documented:

| Rule | Enforcement |
| --- | --- |
| 1 | `CapabilityRequirement.__post_init__` raises `ValueError` at construction. The message names both remedies. |
| 2 | `_detect_cuda_cluster_launch()` reads `CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH` via `ctypes`/`libcuda` and feeds `_get_cuda_sm_features(..., cluster_launch=...)`. The `H20` device-name regex it replaces survives only on the path where the driver cannot be queried. |
| 3 | `selection/artifact.py`: `register_build_capability(name, probe, default=...)` plus four probe constructors, resolved through the `build:` feature namespace inside `CapabilityRequirement.missing_features`. `binary_targets_device` runs `cuobjdump` on an object file. |
| 4 | Same `__post_init__`: an arch bound without an explicit `vendors=` is rejected, as is an unknown or empty vendor set. |
| 5 | A parametrized test classifies sm_103 / sm_109 / sm_121 / sm_130 with no code change, and a second test pins the sm_90 / sm_100 / sm_103 feature sets as literals so no future table edit can move a part we cannot test. |

Two guarantees hold this together. **A detection gap must never narrow what already
worked** — an unqueryable capability resolves to *available*, and a probe that raises falls
back to its declared `default` (the E28 rule). And **tightening a gate must never produce a
`NoKernelFound`** — see P2 rule 1: every specialized registration keeps a portable sibling.

---

## P2. Portability and performance coexist

> *"Portability and high performance should coexist"* — new models get a working path on
> every platform, with gradual optimization available on top.

**What this requires.** Every operator has a portable reference implementation that runs
everywhere, and specialized implementations layer above it by capability. A model author
never chooses a kernel; a kernel author never edits a model.

**Rules we impose.**

1. **Never let a specialized kernel be the only path.** Every specialized registration must
   have a portable sibling for the same operator, so tightening a gate converts a hard
   failure into a slower success — never a `NoKernelFound`.
2. **An arch-locked kernel is a porting target, not a wall.** A kernel bound to tcgen05 is
   an algorithm we have not yet expressed for other silicon; record it as a gap with the
   specific blocker.
3. **"Upstream has this kernel" is a starting line, never a ceiling.** Parity of
   availability says nothing about distance from optimal, and per-architecture tuning is
   largely untouched territory: E26 measured a stock cuBLAS `gemvx` at 70.5%/67.1% of peak
   DRAM bandwidth on the projections that matter, and E33's portable Triton replacement
   reached 87.5%/83.1% — matching a hand-written CUDA kernel while staying portable.
4. **Dispatch decisions live in the selection layer, never in model files.** The
   counter-example is upstream vLLM, where the choice to use a skinny-GEMM kernel sits
   inside a DeepSeek model file and the kernel itself is chosen by a hardcoded `(N,K)`
   C++ template-instantiation whitelist — so the kernel exists but cannot be reached for
   other shapes without recompiling.

---

## P3. Fast iteration requires guardrails

> *"Kernel fast iteration requires guardrails"* — minimal dependencies, independent
> benchmarking, performance profiling.

**What this requires.** Kernels are increasingly produced by agents rather than by hand
(see `kernel-plugin-contract.md`). That is only safe if correctness and performance are
verified automatically, and only useful if the best candidate is *selected* automatically.

**How the original fell short — measured (E30):** `AUTOTUNE` never benchmarks anything. It
falls back to the heuristic ranking and additionally hardcodes `SelectionObjective.DEFAULT`,
discarding the caller's objective — so `AUTOTUNE` is strictly *worse* than `HEURISTIC`.
Heuristic ranking trusts priority bands that encode "the datacenter kernel is best", which
is exactly the assumption that fails on an under-served architecture.

**Rules we impose.**

1. **Selection must be able to measure.** Real autotuning — time the candidates, cache per
   `(device_name, arch, family, mode, format_signature, traits, shape-bucket)`, and wrap
   each candidate so one that raises is dropped rather than crashing the engine. That last
   detail alone would have caught every hard-failing mis-gate P1 describes.
   **Implemented and validated (E46)**; see `docs/design/kernel-autotuning.md`. The
   corollary learned while building it: *measurement only helps where there is more than
   one candidate.* At every `M > 8` dense GEMM the registry has exactly one, so E44's
   M=256 loss is not reachable by better selection — it needs another kernel.
2. **Never tune during serving.** Tuning blocks the launch thread, and under TP it
   desynchronizes ranks into a collective deadlock. Tune at startup, then freeze.
3. **Automated selection beats human curation — we measured it.** In E34, letting the
   declarative trait gate choose across all dense Linears gave **+4.2%** while a
   hand-picked `qkv_proj + o_proj` list gave **+1.6%**; the difference was `down_proj`,
   which the human list omitted and the gate found. Hand-maintained kernel lists are a bug
   source, not an optimization.
4. **Every kernel carries a contract**: signature, trait axes, a numerical reference with
   tolerance, and a benchmark. This is simultaneously the task specification for a
   kernel-design agent and the acceptance test for its output.
5. **Provenance is first-class.** Each kernel declares an `origin`
   (`vendor | contrib | kda | first_party`) used as the final selection tiebreak, so
   agent-generated and community kernels compete on equal terms without displacing a
   validated incumbent at equal priority.

---

## P4. Anything derivable is derived — on every tier

The engine detects its hardware and configures itself. A parameter that *can* be derived
from hardware or load **must not** be a manual flag — on workstation and datacenter alike.

**Where this goes wrong, from our own record.** The adaptive batch-cap derivation shipped
with **three successive single-signal blind spots**: it sized from KV pages alone and
ignored (1) the recurrent/mamba state pool, causing over-admission, retraction, a C=64 hang
and a C=128 device-side assert; (2) sliding-window budgets; and (3) speculative decoding,
where each state chunk costs `1 + num_draft_tokens` more.

**Rule:** an adaptive policy must be the **minimum over every binding resource**, and should
ask the cache manager for a per-request cost estimate rather than re-deriving one from
`context_len`. Prefer one honest signal to three guessed ones.

---

## P5. Optimize for agent workloads

UnieInfra is not a general-purpose engine. Agent traffic is multi-turn (history is resent
every turn), shares a system-prompt and tool-definition prefix across thousands of sessions,
branches (tool-call exploration is tree-shaped, not linear), emits heavily structured output,
and is interactive — so **TTFT matters as much as aggregate throughput**.

Consequences that follow, and that general-purpose engines do not prioritize:

- **Prefix reuse is the lifeline, not a feature.** For agent traffic, skipped prefill *is*
  throughput. Eviction order matters as much as match granularity: a flat free-block LRU
  with no share-count awareness evicts a hot system prompt the moment its last referrer
  finishes, which is precisely backwards.
- **Structured output is the hot path**, so grammar masking belongs inside the captured
  graph, and constraining generation so a malformed tool call is *unrepresentable* beats
  parsing and retrying.
- **Speculation is unusually effective** on structured output, and must be **load-adaptive**:
  spare compute at low concurrency pays for draft+verify, but at saturation the draft work
  competes with real work.
- **Latency and throughput are separate objectives, not one number.** A policy that buys
  +16.9% throughput by adding +44.5% TTFT is the right trade under a batch SLO and the wrong
  one under an interactive SLO. The engine should be told which objective it is serving.

---

## P6. Claims are measured, and measurement is adversarial

Every performance claim carries evidence, and the standard of evidence is set by how often
plausible claims have failed here.

- **Hardware counters over timeline arithmetic.** A "bandwidth is saturated, it's physics"
  conclusion drawn from `bytes ÷ time` was **wrong twice**: ncu showed two projections at
  67–70% of peak, not 90%, and a claimed "25% scheduling bubble" and "sampling costs as much
  as the forward pass" were both host-span artifacts — the GPU was 97–99% busy the whole
  time.
- **Repeat, and report the spread.** A single number on a shared box is not a result. Report
  median with min/max; a delta inside the spread is a tie, and must be called one.
- **Best-vs-best, and the right scenario.** Never handicap a comparison target, never
  hand-tune ourselves per benchmark (make the *default* strong instead), and benchmark the
  workload the product actually serves — every experiment up to E35 used random prompts with
  the prefix cache flushed between levels, i.e. the one scenario where our prefix machinery
  cannot help.
- **Record failures.** `experiments-log.md` keeps every experiment, pass or fail, with the
  root cause. Several of the most useful entries are refutations of our own earlier claims.

---

## What this means structurally

| Layer | Responsibility | Who edits it |
|---|---|---|
| Platform / capability | Detect silicon, report measured capabilities | Rarely; adding a vendor or feature |
| Contract | Per-operator signature, traits, reference, tolerance, benchmark | When an operator is added |
| Selection / registry | Match capability + format + traits; autotune; tiebreak by origin | Rarely; it is the stable seam |
| Kernels (`ops/`, `kda/`, `contrib/`) | Implementations, portable and specialized | Kernel authors and agents |
| Runtime / engine | Ask the registry for an operator; never import a kernel | Engine work |
| Serving / agent surface | Chat, tools, structured output, sessions, routing | Product work |

A new kernel is one file plus one decorator: no core edit, no model edit, no dispatch table.
That property is the point of the whole design — it is what makes agent-generated kernels
practical, and it is the standard the vLLM base is being refactored toward.

---

*Keep the Chinese mirror (`unieinfra-design-principles.zh.md`) in sync. Principles P1–P3 are
adopted from tokenspeed-kernel's published design; P4–P6 are ours, and every "how it fell
short" note above cites an experiment in `experiments-log.md`.*
