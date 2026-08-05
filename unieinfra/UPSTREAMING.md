# Re-expressing the fork

`legacy/divergence/*.diff` holds what UnieAI actually changed in the old fork:
3381 lines / 33 files in `tokenspeed-kernel`, 3003 / 21 in the python runtime,
and nothing at all in `tokenspeed-scheduler`. This is the plan for getting each
of them out of a fork and into a place that survives `git rebase`.

The headline from the survey: **most of it is not UnieAI-specific.** It is
generic capability-gating work, bug fixes, and one missing call. Almost none of
it needed a fork; it needed pull requests.

Status is from a survey of every hunk, not a line-by-line read of all 6384. Each
item still needs its diff re-read against current upstream before it is sent —
upstream has moved four months since the fork point.

## A. Upstream PRs — generic, benefits every user

### A1. Call `discover_plugins()` from the runtime (highest value, smallest change)

Upstream ships an out-of-tree plugin system: entry-point group
`tokenspeed_kernel.plugins`, `discover_plugins()`, a CLI, a README. Its docstring
says "Loading is explicit: the host application must call `discover_plugins()`".

**Nothing in `python/tokenspeed/` ever calls it.** The documented plugin story
does not work end to end, and the fork worked around it with a private
`_load_first_party_kernels()` in `runtime/execution/model_runner.py`.

With this one call upstream, `unieinfra/kernel-sm120/` loads by being installed
and UnieAI needs no fork of the runtime at all. This is the single change that
most determines whether the rest of this document stays short.

### A2. Capability properties instead of version comparisons

`platform.py` (+309), plus `required_features=` conversions across
`ops/attention/flashinfer/`, `ops/attention/flash_attn/`, `ops/gemm/flashinfer.py`,
`ops/quantization/flashinfer.py`, `ops/moe/flashinfer/cutlass_unquant.py`.

Upstream **already has** `sm_features`, `required_features` and
`min_arch_version` — the mechanism exists. What the fork adds is the convenience
layer that makes it usable, and the migration of kernels that still gate on
version numbers:

- `has_tcgen05`, `has_cluster_launch`, `has_fp4_tensor_cores` — ask the detected
  capability set instead of comparing versions.
- `KNOWN_VENDORS` — an arch bound without a vendor is latently cross-vendor:
  AMD gfx950 is `ArchVersion(9, 5)`, the same numbers as Hopper-minor.

The bugs this fixes are not sm_120-only:

- `is_blackwell` tests `major == 10` exactly, so it excludes sm_120 — which does
  have FP4 tensor cores.
- `is_blackwell_plus` is open-ended, so sm_120 satisfies it despite having no
  TMEM.
- The **H20** SKU reports sm_90 and cannot launch clusters. Upstream currently
  handles this with a device-name regex in `ops/sampling/cute_dsl.py`; the fork
  replaces it with `compute:cluster` from the driver.

An H20 user hits the third one today. That makes this a fix, not a preference.

### A3. Build-feature probing — `artifact.py` (+296, new), `ops/build_capabilities.py` (+150, new)

`required_features` answers "does the *hardware* have it". These answer "does the
*installed binary* have it" — a FlashInfer wheel built without FP8 block-scale
support is not distinguishable from one built with it by any hardware query.
Used by `ops/gemm/flashinfer.py`. Ships with `test_arch_gating.py` (+225) and
`test_capability_model.py` (+412).

### A4. Small fixes, independently sendable

| file | change |
|---|---|
| `_triton.py` | `import tokenspeed_triton.profiler` is unconditional, but `profiling.py` already handles `proton is None`. Builds without the submodule fail at import. |
| `ops/attention/triton/mha_prefill.py` | missing early-return bounds check; `_has_large_shared_memory` instead of assuming 48 KB |
| `ops/moe/flashinfer/cutlass_unquant.py` | `topk_weights.to(torch.float32)` |
| `runtime/layers/dense/fp8.py` | missing `.contiguous()` before `per_token_group_quant_fp8` |
| `runtime/models/qwen3.py` | skip `.k_scale`/`.v_scale` checkpoint entries absent from `params_dict` |
| `cli/serve_smg.py` | `TOKENSPEED_SMG_PROMETHEUS_PORT` to pin the port; default unchanged |

### A5. `origin` selection tiebreaker — `registry.py` (+17), `selection.py` (+62)

Adds `origin` as the lowest-weight tiebreaker after `(oracle, objective,
priority)` so an out-of-tree kernel can win an exact tie. Worth sending, but
**not needed by UnieAI**: the `PLUGIN` priority band (16-19) already outranks
`PERFORMANT` (8-11), so a plugin that should win can simply say so. Send it as a
nicety or drop it — do not let it block anything.

### A6. Two-granularity KV planning — `runtime/configs/flat_memory_plan.py`

The `r = P/N` work: attention page size `P` and state-snapshot stride `N` are
separate granularities, and coupling the placeholder-block ratio to them is
worth +10.3% at C=32/ILEN=4096 (log E45). This is generic for any hybrid
attention/linear-attention model, not a UnieAI concern. Needs its measurement
re-stated honestly in the PR — the first number reported for it, +26.2%, was an
artifact of short benchmark arms.

## B. Genuinely UnieAI — belongs in `unieinfra/`

| what | where it was | where it goes |
|---|---|---|
| sm_120 FA2 kernel (`ops/attention/flashinfer/fa2.py`, +503) | fork | `unieinfra/kernel-sm120/` — partly there already as `flash_attn2.py` |
| `unieinfra_decode_gemv_triton` in the gemm preference list | `ops/gemm/__init__.py` | plugin registration at `PLUGIN` band |
| adaptive-batch serving defaults (`enable_adaptive_batch`, reserve divisor, max) | `runtime/utils/server_args.py` | UnieAI serve wrapper, not upstream's arg parser |
| TurboMind w4a16 for sm_120 | `legacy/patches/` | `unieinfra/kernel-sm120/` as a `gemm/turbomind` op, per `AGENTS.md` |

`runtime/layers/attention/registry.py` (gemma4 architectures, capacity model,
state group page sizes) needs splitting: the gemma4 architecture list is model
support and belongs upstream; the capacity model is A6.

## C. Check before doing anything

Upstream added `TraitConstraint` / `TraitRange` to `registry.py` after the fork
point — a kernel can now declare `head_dim` as "multiple of 16 in [16, 256]"
instead of enumerating a frozenset. That is exactly the shape-constraint problem
the fork was working around, and it means at least one backlog item is already
solved. Re-read each diff against current upstream before writing anything.

## Order

1. **A1** — one call. Everything in B is easier once plugins actually load.
2. **A4** — six independent small fixes, no design discussion needed.
3. **A2 + A3** — the capability model, as one series. The largest and the one
   with the clearest bug story (H20).
4. **B** — move UnieAI's own kernels into `unieinfra/`.
5. **A6**, then **A5** if it still seems worth it.

---

## Status

Branches are cut from `local-tokenspeed/main` and carry no UnieInfra content, so
each is a self-contained PR to `lightseekorg/tokenspeed`. None is pushed yet.

| branch | item | state |
|---|---|---|
| `pr/runtime-discover-plugins` | A1 | ready — 3 tests, verified failing without the fix |
| `pr/kernel-small-fixes` | A4 (2 of 6) | ready — proton optional; MHA prefill early-return |

**A1** — `ModelRunner.__init__` calls `discover_plugins()` before `load_model()`.
Tests cover the call, the ordering, and that a second worker still calls it.
Confirmed the tests fail on unmodified upstream.

**A4.1** — `_triton.py` proton import made optional. Verified both ways: with the
submodule present `proton_available()` is still True; with the import forced to
fail, `_triton` imports and `proton` is None.

**A4.2** — Triton MHA prefill skips out-of-range query blocks. 2.04x on a
1x2048-prefill + 64-decode batch and 2.27x at 4096+128, with three controls
(all-decode, uniform prefill, ragged prefill) unchanged to within 0.1% and
output bitwise identical on all five shapes. `test/ops/test_attention.py` has
the same pass/fail set before and after — 12 pre-existing failures on this card
in the flashinfer and cuda paths, none in triton.

### Not yet sent, and why

`runtime/layers/dense/fp8.py` is the one A4 item deliberately left out. It
switches per-token activation quantization from `per_token_quant_fp8` to the
group-quant helper because "the TRT-LLM per-token op returns malformed scales on
sm_120". That is a real bug report, but the fix changes the code path for every
architecture and the fork carries no measurement of what it costs on datacenter
parts. It needs its own PR with numbers, or a narrower gate.

`runtime/models/qwen3.py` (skip absent `.k_scale`/`.v_scale`) and
`cli/serve_smg.py` (`TOKENSPEED_SMG_PROMETHEUS_PORT`) are both small and
uncontroversial; they simply have not been done yet.

---

## Where the sm_120 performance work stands (2026-08-05)

Baseline, same harness both arms, 600 s fixed duration, 200 s tail, one session:
**TokenSpeed 208.0 vs vLLM 248.1** at C=32 / ILEN=4096 on Qwen3.5-27B (log E48).

Everything tried since has hit the same shape of problem — the hardware is
capable, a declaration excludes it:

| | | |
|---|---|---|
| XQA decode | page_size constraint never declared | fixed |
| TRT-LLM FMHA | version comparison with no upper bound | fixed |
| Triton prefill tile | shared-memory budget read from a field that was 0 on every device | fixed |
| fp8 KV | Triton decode declares only fp16/bf16, **and has no fp8 dequant at all** | open, #22 |

The first three are why a server starts at all. The fourth is not a declaration
fix: `mha_decode.py` accepts `k_scale`/`v_scale` and ignores them.

### The three that would close the gap, in the order they should be done

1. **Capacity (#12)** — the largest and best understood. The attention page P is
   inflated 64 -> 832 to cover a GDN state row, so 12.8% of every request's KV
   is rounding waste and only 149 pages exist. That is the 17.7% window
   oscillation. The machinery to fix it is already ported; the exact call site
   and the hazard are recorded on the task.
2. **fp8 KV (#22)** — implement the dequant in the Triton decode kernel. No
   checkpoint needed and it does not touch model quality, which is why it should
   come before w4a16.
3. **w4a16 (#21)** — note TokenSpeed already ships a w4a16 path
   (`compressed_tensors_wNa16`, `get_min_capability()` = 90, so sm_120 passes),
   and `marlin.so` exports only `gptq_marlin_repack`. Check whether that path
   actually works before assuming the ported TurboMind kernel is needed.
   Quantization is never automatic — TurboMind repacks the layout at load time
   but the int4 weights have to be produced first, and E49 measured +7.2%
   perplexity for the uncalibrated scheme.

None of these is a wiring job. Each is a day-scale change with a real
verification step, and each can fail silently if the verification is skipped.
