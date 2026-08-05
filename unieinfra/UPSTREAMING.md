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

The work itself lives on `main`. The `pr/*` branches below are *views* of it:
each is cut from `local-tokenspeed/main` and carries only the cherry-picked
commits for its item, so each is a self-contained PR to
`lightseekorg/tokenspeed` with no UnieInfra content in it. None is pushed yet.

Re-cut them after any rebase — they are derived, not authoritative. An earlier
pair of branches was lost in the re-fork (`1dd493da`) while this table still
claimed they existed; if that happens again, `git cherry-pick -x` the commits
listed here back onto `local-tokenspeed/main`.

| branch | item | commits | state |
|---|---|---|---|
| `pr/runtime-discover-plugins` | A1 | `d97efd1c` | ready — 3 tests, verified failing without the fix |
| `pr/kernel-small-fixes` | A4 (3 of 6) | `57312157` `b93a8a8e` `4119f79c` | ready — proton optional; MHA prefill early-return; cutlass MoE scales |
| `pr/runtime-small-fixes` | A4 (2 of 6) | `3eb83e9d` `b12febcc` | ready — Qwen3 KV scales; smg Prometheus port |

Split by package because upstream reviews by area: `pr/kernel-small-fixes`
touches only `tokenspeed-kernel/`, `pr/runtime-small-fixes` only
`python/tokenspeed/`. Each branch's tests were re-run at its own tip, not just
on `main` — 3, 4 and 66 passing respectively.

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

**A4.3** — `cutlass_unquant.py` promotes its self-computed routing weights to
float32; the cutlass fused-MoE kernel rejects a half/bfloat16
`token_final_scales`. The engine's precomputed-topk path was already float32,
which is why this never fired in serving. Verified failing before the fix on
both fp16 and bf16 activations, with the precomputed-topk cases passing in both
states as controls.

**A4.4** — `qwen3.py` skips `.k_scale`/`.v_scale` entries with no matching
parameter, as `qwen3_moe.py` already does. Verified failing on all six scale
spellings; an ordinary-weight case and a non-scale missing name pass in both
states, so the skip provably swallows nothing else.

Noted while testing, not fixed: `"v_proj" in "qkv_proj"` is true, so a
checkpoint key already spelled `qkv_proj.k_scale` is mangled to `qkqkv_proj` by
the stacked-params rewrite before any skip sees it. Same for `qkv_proj.bias`.
Pre-existing in the mapping loop, and separate from A4.

**A4.5** — `serve_smg.py` honours `TOKENSPEED_SMG_PROMETHEUS_PORT`. Unset or
empty leaves the randomised default untouched; an explicit `--prometheus-port`
still wins; an unparsable value raises rather than silently handing back a
random port the operator would then fail to scrape.

### Not yet sent, and why

`runtime/layers/dense/fp8.py` is the one A4 item deliberately left out. It
switches per-token activation quantization from `per_token_quant_fp8` to the
group-quant helper because "the TRT-LLM per-token op returns malformed scales on
sm_120". That is a real bug report, but the fix changes the code path for every
architecture and the fork carries no measurement of what it costs on datacenter
parts. It needs its own PR with numbers, or a narrower gate.

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
| fp8 KV | Triton decode and extend both declare only fp16/bf16 | fixed, #22 |

All four are why a server starts at all. The fourth was **also** a declaration
fix, contrary to what this table said before: it claimed `mha_decode.py` "has no
fp8 dequant at all", and that was wrong. The dequant is there --
`qk = tl.dot(q, k.to(q.dtype))` -- and running the kernel on an fp8 cache,
bypassing selection, scored cosine 0.9996 against a bf16 reference. Nothing had
ever handed it an fp8 cache because `format_signatures` puts one dtype on every
role, so the registration could only ever match a cache that matched the query.
The `k_scale`/`v_scale` arguments it ignores belong to the mxfp8 block-scaled
path, not to per-tensor fp8, which carries no scales at all.

### fp8 KV (#22) — done, 1.92x KV capacity

`996b05f4` adds the mixed-role signatures (a high-precision query over an fp8
cache) to Triton decode **and** extend; `b8a392f8` stops the runtime casting the
query down to the cache dtype on that path.

Measured on Qwen3.5-0.8B, `--attention-backend triton`, one arm per process —
running both engines in a single process gives a *wrong* answer, because the
first pool is still resident when the second is sized (37.7 GB free vs 14.2 GB,
which reported fp8 as 0.68x):

| arm | KV tokens |
|---|---|
| bf16 | 1,271,808 |
| fp8_e4m3 | **2,438,208** (1.92x) |

Under 2x because the non-KV allocation is fixed. Quality is unchanged: 4/5 on
checkable short answers, the same score and the same single miss as bf16 (a
truncated thinking preamble, not a wrong answer). Scored rather than compared
token-for-token, since fp8 KV is lossy and identical tokens are not what success
looks like.

Before the change the same command does not start at all —
`NoKernelFoundError: No kernel found for attention.mha_decode_with_kvcache
(k_cache=dense[storage=torch.float8_e4m3fn], ...) with solution 'triton'`.

Still unmeasured: throughput against the 208.0 baseline at C=32 / ILEN=4096 on
Qwen3.5-27B. Capacity is not throughput, and the baseline is what the gap is
stated in.

### The two left, in the order they should be done

1. **Capacity (#12)** — the largest and best understood. The attention page P is
   inflated 64 -> 832 to cover a GDN state row, so 12.8% of every request's KV
   is rounding waste and only 149 pages exist. That is the 17.7% window
   oscillation. The machinery to fix it is already ported; the exact call site
   and the hazard are recorded on the task.
2. **w4a16 (#21)** — note TokenSpeed already ships a w4a16 path
   (`compressed_tensors_wNa16`, `get_min_capability()` = 90, so sm_120 passes),
   and `marlin.so` exports only `gptq_marlin_repack`. Check whether that path
   actually works before assuming the ported TurboMind kernel is needed.
   Quantization is never automatic — TurboMind repacks the layout at load time
   but the int4 weights have to be produced first, and E49 measured +7.2%
   perplexity for the uncalibrated scheme.

Neither is a wiring job. Each is a day-scale change with a real verification
step, and each can fail silently if the verification is skipped. #22 is the
cautionary case in both directions: budgeted as day-scale kernel work, it turned
out to be two declarations — and the first capacity measurement of it was off by
3x in the wrong direction because two engines shared a process.
