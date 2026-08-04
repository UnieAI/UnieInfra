<!-- English canonical. docs/design/* are English-only in this repo (no Chinese mirror). -->
# Project Layout and the Absorption Model

Status: **Implemented (layout) / directional (the rest)** · Date: 2026-07-25 · Owner: royshih

## The thesis in one paragraph

UnieInfra is an **agent-oriented inference engine** that intends to contain
everything valuable from every serious inference engine — and to contain it as
**owned in-tree source, never as a library dependency**. vLLM, SGLang,
tokenspeed, TensorRT-LLM, FlashInfer and JetSpec are *material*, not
dependencies. The final shape has no `pip install vllm` in it. This document
records the layout that follows from that decision, why each directory exists,
and how a new component gets absorbed.

## 1. Layout as of this commit

```
UnieInfra/
├── unieinfra-vllm/        ← THE ENGINE. Hard fork of vLLM @ d02df748, vendored.
├── unieinfra-kernel/      ← Kernel selection registry + our kernels. Base-independent.
├── benchmark/             ← Engine-agnostic harness (results/<engine>/<label>/).
├── docs/                  ← Design docs, guides, experiments log.
├── third_party/           ← Submodules: reference material, not dependencies.
│
├── python/tokenspeed/     ← THE RETIRING ENGINE. Left alone. See §6.
├── tokenspeed-kernel/     ← Its kernel package (112 kLOC op corpus).
├── tokenspeed-scheduler/  ← Its C++ FSM scheduler.
├── tokenspeed-mla/ tokenspeed-kernel-amd/
```

The rule that makes this readable: **a directory at the repository root is
source we own; a directory under `third_party/` is reference material we do
not.** Absorption is exactly the act of moving a component from the second
category to the first, and it is never silent — it comes with a `PROVENANCE.md`
and a `NOTICE` entry.

### 1.1 `unieinfra-vllm/` — the engine base

A hard fork of vLLM at commit `d02df748` (`v0.13.0rc1-6959`, 2026-07-24),
vendored on 2026-07-25. **We do not sync upstream.**

- **Why this commit, not the newer 0.25.1 that was pip-installed on the box?**
  Because `docs/design/vllm-base-assessment.md` cites `file:line` against this
  tree, and E34 had already been forced to re-verify its line numbers when the
  installed wheel diverged from the analysed submodule. Read-code must equal
  run-code. `third_party/vllm` stays pinned at `d02df748` precisely so the fork's
  provenance remains verifiable by `diff -r`.
- **Why not mass-rename `vllm` → `unieinfra` on day one?** 3,837 Python files,
  290 of them model implementations. The diff would be enormous, would break
  every line-number citation we have written, and would buy nothing functional.
  We rename lazily. The distribution name is already `unieinfra-vllm`, the
  console script is already `unieinfra` (with `vllm` kept as an alias), and the
  env prefix for anything we add is already `UNIEINFRA_`. See
  `unieinfra-vllm/README.md` § Naming for what would force the full rename
  earlier — in short, the day we need a *real* `vllm` distribution installed
  beside ours in one environment.
- The complete delta from upstream is in `unieinfra-vllm/PROVENANCE.md`.

### 1.2 `unieinfra-kernel/` — the moat, kept standalone

The capability-based kernel selection layer (`registry` / `selection` /
`platform` / `signature`, ~2,600 LOC) plus our first-party kernels. It imports
**only torch and the stdlib** — no engine, no vendor library. That is the
property that made it portable in the first place, and it is worth protecting,
so it stays its own package rather than being dissolved into `unieinfra-vllm/`.

`unieinfra_kernel.boundary` still routes registration into `tokenspeed_kernel`'s
registry when that package is importable, so the retiring engine keeps working
untouched.

**The engine calls it directly**, not through a plugin API — see §3.

### 1.3 `third_party/` — reference, not dependency

Submodules are how we *read* an engine before deciding whether to absorb any of
it. Nothing under `third_party/` is redistributed and nothing in the engine
imports from it.

## 2. The absorption model

vLLM is the first absorption, not the only one. The intended shape:

| Source | License | What we want from it | Status |
|---|---|---|---|
| vLLM | Apache-2.0 | The whole base: 290 models, the OpenAI server, 44 tool parsers, grammar-enforced tool choice, 4 structured-output backends, `torch.compile` integration, KV-offload connectors | **Absorbed** (`unieinfra-vllm/`) |
| SGLang | Apache-2.0 | Radix prefix reuse with longest-prefix-match admission | Reference (`third_party/sglang`) |
| tokenspeed | MIT | Leaf-first LRU eviction with root-path ref-count pinning; RAII KV ownership; the placement→comm compiler | In-repo, retiring — §6 |
| TensorRT-LLM / FlashInfer | Apache-2.0 | Kernels, ported per-op through `unieinfra-kernel` | Reference + per-kernel ports |
| JetSpec | **MIT** | Tree-shaped speculative drafting | Reference (`third_party/JetSpec`) |
| FastVideo | Apache-2.0 | Diffusion/video serving patterns | Reference (`third_party/FastVideo`) |

### 2.1 The selection criterion

Not "is it good" but **"does it make the agent workload better."** Agent
workloads are multi-turn loops resending history, thousands of sessions behind
one system prompt, branching tool-call exploration, heavy structured output,
short turns where TTFT dominates. A component earns absorption if it moves
cache-hit rate, TTFT under reuse, tool-call correctness, or structured-output
throughput. `docs/design/vllm-base-assessment.md` §7 is blunt that our entire
benchmark history (E1–E35) measured the *opposite* regime — random prompts with
the prefix cache flushed before every level — so an agent-shaped benchmark is a
prerequisite for judging most of this table.

### 2.2 The rules for absorbing anything

1. **Vendor the source at a named commit.** Not a tag, not a branch — a commit.
2. **Write a `PROVENANCE.md`** next to it: upstream, license, commit, date, and
   a complete delta table (removed / renamed / modified). Apache-2.0 §4(b)
   *requires* prominent notice of changed files; the delta table is that notice
   and it is also what keeps the fork auditable by `diff -r`.
3. **Add a per-component entry to `NOTICE`.** One blanket "we use open source"
   paragraph is not attribution. Licenses differ per component — vLLM and
   FastVideo are Apache-2.0, JetSpec is MIT — and the obligations differ with
   them.
4. **Preserve the upstream README** (`UPSTREAM_README.md`) and every SPDX header.
5. **Remove upstream's agent/CI configuration** (`.claude/`, `.github/`,
   `.buildkite/`, `AGENTS.md`, `CLAUDE.md`). Left in place, upstream's
   contributor instructions silently govern agents working in our tree — an
   agent reading vLLM's `CLAUDE.md` will try to follow vLLM's PR process.
6. **State the no-sync policy in the tree itself**, not just in a design doc.

## 3. How the pieces connect

```
   unieinfra-vllm/vllm/model_executor/layers/linear.py
      UnquantizedLinearMethod.apply()
        └─ dispatch_unquantized_gemm()          ← utils.py, UnieInfra arm added
             └─ unieinfra_gemm.unieinfra_unquantized_gemm()   ← new file
                  └─ torch.ops.unieinfra.linear             (opaque to Dynamo)
                       └─ unieinfra_kernel.dispatch.linear()
                            └─ select_kernel("gemm", "mm", sig, traits)
                                 └─ unieinfra_decode_gemv_triton  (M≤2, N≤16384)
                                 └─ …or falls through to F.linear
```

**This replaced a plugin.** E34 reached the same GEMMs through vLLM's supported
extension points — `vllm.general_plugins` → `register_quantization_config` → a
`LinearMethodBase` — and measured +4.2% inside *stock, unmodified* vLLM. That
was the correct shape while the base decision was open, because it was
reversible and it proved portability cheaply. It is the wrong shape now:

- it forces `--quantization unieinfra` on a bf16 checkpoint, which is a lie
  about the weights and a permanent user-facing wart;
- it can only reach the seams upstream chose to expose. The base assessment
  counted **7 of 19** hot-path dispatch decisions as unreachable without
  patching source — including the bf16 skinny-GEMM shape whitelist, which is a
  C++ *template instantiation* list and therefore unreachable by any plugin in
  any language.

Owning the source removes both. The direct wiring is ~6 lines in `utils.py` plus
one new module, and it keeps the two properties that made the plugin cheap: the
`torch.library` custom op stays opaque to Dynamo (no graph break), and the
dispatch arm is chosen by a module-level constant so `torch.compile` folds the
branch instead of tracing it.

`unieinfra-kernel` stays a separate, standalone package. The engine depends on
it; it does not depend on the engine.

## 4. Coexisting with stock vLLM

Both distributions own the import name `vllm`, so they cannot share a
`site-packages`. A/B comparison therefore runs in **separate environments**, one
process at a time:

| Environment | Contains |
|---|---|
| the dev container's `unieinfra-vllm-venv` | this fork |
| `/home/ubuntu/vllm-bench-venv` (host) | stock `vllm` from PyPI |

The `benchmark/` harness is already engine-agnostic and always drives the *same*
stock `vllm bench serve` client, which is what keeps the comparison honest (see
`docs/design/benchmark-methodology`, and E35 for the discipline in practice).

Two traps that have already cost us time:

- vLLM's compile/AOT cache is keyed on `VllmConfig`, **not** on environment
  variables. Two runs differing only in `UNIEINFRA_GEMM` share
  `~/.cache/vllm/torch_compile_cache/<hash>/` and the second silently replays
  the first's graph. Any A/B that changes graph structure needs
  `VLLM_DISABLE_COMPILE_CACHE=1` (E34 trap (c)).
- `VLLM_USE_FLASHINFER_SAMPLER=0` was required by stock vLLM 0.25.1 on sm_120
  (E18). Whether it is still required at this pin is recorded in E36.

## 5. Identity migration (deferred, not forgotten)

| Thing | Now | Target | Trigger to act |
|---|---|---|---|
| Distribution | `unieinfra-vllm` | `unieinfra` | Cosmetic; do it with the import rename. |
| Import package | `vllm` | `unieinfra` | **Blocking** the moment a real `vllm` must be installed beside ours. |
| Console script | `unieinfra` + `vllm` | `unieinfra` | Drop the `vllm` alias one release after the import rename. |
| Env vars | `VLLM_*` + `UNIEINFRA_*` | `UNIEINFRA_*` | Needs a dual-read shim; `VLLM_*` names are a user-facing contract, exactly as `TOKENSPEED_*` is. |

`docs/design/repo-reorg-and-rename.md` already measured this exact class of
migration for `tokenspeed` → `unieinfra` (1,825 import statements, 105 env-var
names, 42 path names) and its runbook — mechanical rewrite, compat shim, one
deprecation cycle on env names — transfers directly. The vLLM rename is larger
(3,837 files) and should be done as one mechanical commit with **no** semantic
change riding along, so it is reviewable by construction.

## 6. Retiring `python/tokenspeed/`

Not now, and not by deletion. It is the engine that currently wins C=32–128
against both vLLM and SGLang (E35: +16.9% at C=128, reproduced N=3), and that
result is a bar the new base has to clear, not a legacy to discard.

Proposed sequence:

1. **Reach parity first.** `unieinfra-vllm` must reproduce E35's sweep on the
   same box, same client, same protocol. Until it does, tokenspeed is the
   engine and vLLM is the challenger.
2. **Port the differentiators, measured one at a time.** The base assessment
   ranks them: hierarchy-aware eviction (leaf-first LRU + root-path ref-count
   pinning) is the one durable advantage for agent workloads, and it is a
   *Python rewrite behind `scheduler_cls`*, not a C++ port — the C++ tree is
   fused to the FSM and cannot be lifted. Adaptive KV-derived admission (E25)
   is 13 lines of policy. Mixed prefill+decode (E31) is blocked on attention
   metadata in both trees.
3. **Freeze, then retire.** Once each differentiator is ported *and measured on
   the new base*, `python/tokenspeed/` goes read-only (no new features), then
   moves out of the default install, then out of the tree. `tokenspeed-kernel/`
   outlives it either way: `unieinfra_kernel.boundary` already routes through
   it, and its 112 kLOC op corpus is a source of kernels for the new base.
4. **Do not delete anything until the agent-shaped benchmark exists.** Every
   comparison we own is from the zero-reuse regime, which is the one regime in
   which tokenspeed's prefix machinery cannot help. Retiring it on that evidence
   would be retiring it on the wrong measurement.
