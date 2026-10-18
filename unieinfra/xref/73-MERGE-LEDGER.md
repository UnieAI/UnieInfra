# 73 — Merge ledger: what our pre-merge work actually lost

The 2026-10-18 merge (`ac2f43a8 "Merge upstream/main into unieai"`) resolved
all 72 conflicted files to upstream and deleted the 26 files upstream had
removed. That is the right default when upstream reimplemented the same thing,
but "we took theirs" is not evidence that nothing of ours mattered. This page is
that audit: for every file our work had touched (`a9d63af8` versus its parent
`1890d2a5`), does the behaviour survive in the merged tree, and where.

Verdict vocabulary, one per file:

* **kept** — the merge left our lines in place (no conflict on that file).
* **superseded** — upstream's own code now does the same thing, at a different
  path or under a different name. Nothing to port.
* **obsolete** — the mechanism our patch worked around no longer exists.
* **instrumentation** — the dropped lines were measurement scaffolding, not
  behaviour.
* **port** — behaviour the merged tree genuinely does not have.

## 1. The audit

| file | our delta | verdict | evidence |
| --- | --- | --- | --- |
| `runtime/engine/event_loop.py` | +102 | obsolete + instrumentation | the config-level "can this arch capture prefill" heuristic our patch excepted DeepSeek-V4 from is gone; capturability is now declared per backend (`_admits_capacity_prefill()`, e.g. `layers/attention/backends/state/gdn.py:106`). The rest was loop-timing hooks. |
| `runtime/engine/scheduler_utils.py` | +2 | instrumentation | — |
| `runtime/engine/generation_output_processor.py` | +27 | instrumentation | `_diag_post` / `_t_post*` timers. |
| `runtime/execution/model_executor.py` | +25 | instrumentation | `PROF_GPU_ELAPSED`, `gpu_events` — the CUDA-event bracket. |
| `runtime/execution/prefill_graph.py` | +50 | instrumentation | `_r_t0/_r_pending/_r_gstart/_r_gend` replay timers. |
| `runtime/execution/breakable_cuda_graph.py` | +103 | instrumentation | `_pfc_band/_pfc_state/_pfc_rank/_pfc_prof` band profiler. |
| `runtime/execution/types.py` | +9 | instrumentation | `gpu_elapsed_ms` field feeding the above. |
| `runtime/execution/cuda_graph_wrapper.py` | +62 | superseded | file deleted upstream; the ladder moved to `runtime/execution/forward_step.py:111`. Its two landed fixes are re-ported there (§2). |
| `runtime/layers/attention/kv_cache/base.py` | +59 | obsolete | our memoised `_group_field_segment_specs` optimised a per-(block, field) resolve in `_block_byte_segments`, and that method no longer exists — upstream rewrote page geometry. Re-derive against the new code if the trace shows metadata cost, do not port the memo blindly. |
| `runtime/layers/attention/kv_cache/recipes/plan.py` | +49 | obsolete | same rewrite; `_index_first` has no counterpart. |
| `kernels/ops/kvcache/triton.py` | +12 | obsolete | upstream's kernel writes the same locs. |
| `runtime/utils/env.py` | +9 | instrumentation | `TOKENSPEED_GPU_ELAPSED`. |
| `runtime/utils/server_args.py` | +16 | port (§2) | `--prefill-defer-window` wiring. |
| `runtime/models/qwen3_5.py` | +45 | n/a to the parity target | Qwen3.5-only; re-check only if that model is served again. |
| `runtime/models/deepseek_v4*.py`, `layers/attention/backends/{deepseek_v4,mha,hybrid_linear_attn}.py`, `layers/attention/deepseek_v4_ops.py`, `layers/deepseek_v4_mhc.py`, `models/deepseek_v4_dspark_ops/attention.py`, `kernels/ops/attention/triton/deepseek_v4.py`, `kernels/ops/attention/flashinfer/gated_delta_rule.py`, `kernels/ops/moe/marlin/mxfp4.py`, `kernels/thirdparty/deep_gemm/warmup.py`, `kernels/ops/moe/flashinfer/__init__.py` | +1 700 total | superseded | upstream split the V4.1 stack into `models/deepseek_v41*.py` and moved the fused mHC path into `thirdparty/cuda/mhc.py::mhc_big_fuse` + `ops/residual/`. The three ops we still lack (`fused_compress_quant_cache`, `indexer_k_store`, `fused_wo_a`) are catalogued in `72-PARITY-STRATEGY.md` §3. |
| `kernels/ops/transform/weak_ref.py` | +65 | **kept, re-ported** | upstream's version drops the writable build directory (§2). |
| `scheduler/csrc/scheduler/request.h`, `types.h`, `operations/forward.cpp` | +110 | port (§2) | `prefill_defer_rounds_`, `SchedulerConfig::prefill_defer_window`, and the deferral in the schedule step. |
| `scheduler/csrc/cache/coordinator/kv_cache_admission.cpp` | +12 | superseded | upstream's `ReservePrefillDemands` in `operations/forward.cpp` reserves per cache group from each request's own demand, which is what our patch was reaching for. |
| `scheduler/bindings/python_module.cpp` | +2 | superseded | binding for the flags above. |
| `bench/agent-slo/{agentx_metrics,drive}.py` | +88 | kept | the 4-line difference is `black` reformatting. All required metrics (`ttft_p50/90/95/99`, `tpot_*`, `tps_per_req`, `rpm`, `dur`) are present. |
| `docs/serving/*.md` | +1 555 | kept | — |

## 2. What is actually re-ported, and why

1. **`weak_ref.py` writable build directory** (restored verbatim from
   `a9d63af8`). Without it, a read-only `$HOME` makes the fused-mHC op fail to
   build, the identity fallback takes over silently, and graph-pool blocks stay
   pinned — which shows up only as a single-rung prefill ladder. Correctness
   reaches the hot path here, so it comes back.
2. **`forward_step.py::get_batch_sizes_to_capture` warning**. The ladder moved
   to `forward_step.py` and upstream's version truncates silently again; the
   warning text and the four-argument form are restored exactly, and
   `test/runtime/execution/test_cudagraph_capture_sizes.py` now imports
   `forward_step` instead of the deleted `cuda_graph_wrapper`. The measurement
   behind it is unchanged: capped at 40 with `--max-num-seqs 80`, an AgentX c64
   cell ran 124.5 s at 262 tok/s versus 59.8 s at 547 tok/s uncapped.
3. **`--admission-reserve` removed from the two remaining scripts**
   (`fused_indexer_ab.sh`, `verify_fused_indexer.sh`) and annotated in
   `25-admission.md` / `71-SERVING-CONFIGS.md`: the flag no longer exists, so
   those scripts could not boot.

## 3. The one behaviour left on the table: `prefill_defer_window`

Our C++ scheduler could defer a lone first-chunk prefill for up to
`prefill_defer_window` rounds so prefills accumulate into one batch instead of
running one at a time, and so a single prefill does not take the step away from
every decode behind it. It is a scheduling algorithm, not a knob: it changes
*when* requests are grouped, never *what* the forward computes.

Its recorded justification was that a DeepSeek-V4 cache-hit prefill costs
~140 ms of fixed eager-attention host time regardless of how few new tokens it
carries (`30-prefill.md`). Two of that argument's premises are now unverified:
the merged tree has the prefill graph and the DSA attention through a different
path, and the deferral only pays off when a step is either prefill or decode.

Kill criterion, checked at G0 before any port: in the c16 trace, if prefill
steps and decode steps do not interleave badly (prefill share of wall time <
10 % and no decode step is starved behind a prefill), this stays unported.

## 4. Measurement without scaffolding

All four dropped instrumentation sets were ad-hoc timers edited into the hot
path. G0 does not re-add them: ms/step, kernels/step and device-busy come from
an external `torch.profiler` / `nsys` wrapper around one bounded cell, which
also cannot drift from the code it measures. If a number must be readable from
inside the engine later, it gets a `docs/design/`-sanctioned diagnostic and a
test — not a temporary print.

## 5. The environment was two weeks behind, and four tests target code that is gone

Collected the whole suite (`pytest test/ --collect-only`) after the merge. Three
failure classes came out, and only one of them was ours:

1. **The container had the wrong scheduler.** `ts-env:gpu` shipped
   `tokenspeed-scheduler` **0.1.6**; the tree is **0.1.39** and
   `python/pyproject.toml` requires `>=0.1.39`. Twenty test modules failed to
   import `CacheGroupConfig` / `AttnKind`, which 0.1.6 does not export -- so the
   mismatch had been invisible until something called those names. Worse than
   the test noise: the image's C++ scheduler predates the merge by two weeks, so
   *any* engine run in it (including every number quoted in
   `72-PARITY-STRATEGY.md` §1) executed the old scheduler semantics while
   running the new Python. Rebuilt the wheel from `tokenspeed-scheduler/` (pure
   C++, no CUDA, no GPU needed) and committed a new image `ts-env:gpu-sched39`
   with 0.1.39 installed; the import errors are gone. **Every pre-2026-10-18
   measurement is scheduler-0.1.6 provenance and must be re-baselined before it
   is compared against anything.** G0 runs in the new image.
2. **301 modules need a GPU** to import at all
   (`tokenspeed_kernel` refuses to load without one). Not a defect; collection
   on a CPU-only host can only ever see the CUDA-free subset.
3. **Four tests targeted code the merge removed**, all of them ours (absent from
   `upstream/main`), and all of them referencing names that `grep` finds nowhere
   in the merged tree: `build_image_spans_from_context`, `deepseek_v4_select_experts`
   / `routing_score`, `layers/deepseek_v4_mhc`, and
   `paged_cache_spec.resolve_layer_types_and_window`. Removed:
   `test/runtime/models/test_deepseek_v4_image_span.py`,
   `test/runtime/test_deepseek_v4_moe_routing_scores.py`,
   `test/runtime/test_deepseek_v4_mhc_pre_norm.py`,
   `test/runtime/test_resolve_layer_types_and_window.py`.

The fourth of those was already dead before the merge: the function it loads
from `paged_cache_spec.py` does not exist in that file in any snapshot
(`HEAD`, the WIP snapshot, or the pre-merge safety branch), so it never ran.

The mHC removal is the one with teeth. That test was the only guard on the
property *fused mHC pre == unfused three-launch path plus a norm*, and upstream
does not test its own replacement (`mhc_big_fuse`, reached through
`ops/residual/deep_gemm.py::deep_gemm_mhc_pre`) anywhere under `test/`.
Deleting our version therefore trades a stale guard for none, which is only
acceptable because G0 replaces it at the level that matters: temperature-0
token-for-token agreement against the pre-merge engine on the same prompts
(§4's gate, and the argument in AGENTS.md that the corrupt arm can look faster).
If G0's agreement check is not clean, the missing unit guard is the first thing
to rebuild against the new API.
