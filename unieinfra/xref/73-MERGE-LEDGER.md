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

## 6. What the merge dropped that the tree still needs back

The per-file audit in §1 asked "did upstream keep our change, or replace it?".
It did not ask the complementary question: **does the merged tree still
import?** Two gaps came out of booting it, and both are ours to fix, not
upstream's:

1. **`GeluTanhAndMul`.** The merge took upstream's `runtime/layers/activation.py`
   wholesale, and upstream's copy has no GeGLU (`grep -r GeluTanhAndMul` over
   `upstream/main`: zero hits), while `runtime/models/gemma.py:75` still imports
   it. So the merged tree fails at import time for every model, not just gemma.
   Restored: the class now dispatches to
   `tokenspeed_kernel.ops.activation.flashinfer.gelu_tanh_and_mul` when that
   kernel is registered and the tensor is CUDA on a non-AMD platform, and falls
   back to `F.gelu(gate, approximate="tanh") * up` otherwise — the same
   capability-gated shape as the neighbouring `SiluAndMul`. This is a
   vendor-neutral runtime dependency met through `tokenspeed-kernel`, as the
   dependency boundary requires.
2. **The whole V4 (non-`.1`) model stack.** `models/deepseek_v4.py`,
   `deepseek_v4_dspark.py`, `deepseek_v4_dspark_ops/`, `deepseek_v4_mtp.py` and
   `deepseek_v4_vision.py` exist only in the WIP snapshot (`1890d2a5`); upstream
   replaced the family with `deepseek_v41*` and `models/registry.py` now
   registers `DeepseekV41ForCausalLM` only. Restoring a second, parallel model
   implementation would violate "one execution path", and the checkpoint we
   actually compare against is V4.1 (§7). **Decision: abandon the V4 stack**;
   keeping it would mean maintaining two decoder stacks whose only consumer is a
   checkpoint nobody serves any more.

## 7. The benchmark model was not the reference model

This is the expensive one, and it invalidates every cross-engine comparison in
`72-PARITY-STRATEGY.md` §1.

The reference cell on GPUs 0-3 (`deepseek-DeepSeek-V4.1-Flash-sglang-1`) serves

```
--model-path /models/deepseek-ai/DeepSeek-V4.1-Flash-1017
--served-model-name DeepSeek-V4-Flash-Vision-Exp
```

The served name is an alias. Every cell we drove pointed `--model-path` at
`/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, which is a *different
checkpoint*: its `config.json` declares
`architectures=["DeepseekV4ForCausalLM"]`, `model_type: deepseek_v4`, 43 layers,
hidden 4096. The reference serves `architectures=["DeepseekV41ForCausalLM"]`,
`model_type: deepseek_v41`. Same alias, different weights, different
architecture. So "ours versus vLLM versus sglang" was measured on two different
models, and the reference also runs TP=4/EP=4 against our TP=2.

What that costs and what it fixes:

* **Re-baseline on V4.1-Flash-1017, at TP=4** to match the reference's
  parallelism. `bench/agent-slo/agentx_sweep.sh` now defaults `MODEL` to the
  V4.1 path and takes `TP` (serving `--tp $TP --ep-size $TP`, the same
  expert-sharding layout the reference uses).
* **TP=2 cannot hold this checkpoint at all**, and that is a memory fact, not a
  tuning opinion. Measured on GPUs 6,7: with `--tp 2` the weight load alone
  reaches 135.74 GiB per rank on a 140 GiB H200 and the first MoE weight
  allocation OOMs (`Tried to allocate 2.11 GiB ... 1.38 GiB is free`, inside
  `create_mxfp4_weight_pair`). Adding `--ep-size 2` did not change the
  footprint. The two Engram n-gram tables add to that: their row-sharded
  per-rank part is 45.8 GiB in FP8, which OOMs earlier still
  (`RowShardedEngramEmbedding.__init__`), so they run from host memory via
  `--engram-host-table` (UVA gathers, 1.1 TiB of host RAM free) with
  `--engram-host-table-dir` outside the container's 32 GiB `/dev/shm`.
* Boot cost is real and must be budgeted per cell: the flashinfer `fused_moe_90`
  JIT expands several hundred CUTLASS grouped-GEMM translation units for
  `sm90a` on first use (~49 cores for ~25 minutes), then lands in
  `FLASHINFER_WORKSPACE_BASE` and is reused.

## 8. Two startup failures that only TP=4 exposed

The merged tree never served a TP=4 request before this, so both of these were
invisible until it did. Neither is a tuning question.

### 8.1 The MoE loader held a second copy of every expert weight

`DeepseekV41ForCausalLM.load_weights` builds `params = dict(
self.named_parameters())` (1481 entries) and passes it to
`build_moe_checkpoint_loader(params_dict=params, ...)`. The post-load hooks
(`flashinfer_cutlass_mxfp4_w4a16_moe_weights`) then *replace* each MoE parameter
with its kernel layout, i.e. `w.w13_weight = Parameter(interleaved, ...)`. The
loader kept referencing the caller's dict, so every replaced loader-layout copy
stayed alive beside its replacement.

Measured with per-layer accounting at TP=4: `enter` 73.27 GiB → next `enter`
74.95 GiB, i.e. **+1.68 GiB per MoE layer**, reaching `moe#36 live=133.80 GiB`
and an OOM on a 140 GiB H200. `gc.get_referrers` named the culprit
(`dict(keys=['model.layers.0.ffn.experts.w13_weight'], len=1481)`), and
`gc.collect()` freed nothing (`gc_objs=0`), so it was a live reference, not a
cycle.

The fix is `MoECheckpointLoader.release_params()`, called immediately before
`self.post_load_weights()`: the map is the *caller's* dict object, so clearing
it drops the loader's reference for good. `test/runtime/layers/test_moe_loader_ep.py
::test_release_params_removes_the_loaders_hold_on_weights` pins the behaviour
(the map is empty, and a later `load()` of a released name raises
`MoECheckpointLoadError` instead of silently writing nothing).

### 8.2 The prebuilt CUDA extensions were from before the merge

`scheduler exiting ... TypeError: Mismatched number of arguments when calling
trtllm_mnnvl_allreduce_fusion(...)` is not a Python bug: the Python wrapper in
`thirdparty/cuda/trtllm.py` and the merged C++ sources
(`csrc/trtllm_mnnvl_allreduce_fusion.cu`, `csrc/include/flashinfer/comm/*.cuh`)
carry the newer argument list, while
`thirdparty/cuda/objs/trtllm_comm/trtllm_comm.so` was still the 11 Sep build
shipped in the image. **Any merge that touches `csrc/` invalidates the `.so`
artifacts under `objs/`; rebuild them before benchmarking.**

Two traps when rebuilding:

* `FLASHINFER_CUDA_ARCH_LIST` must be set explicitly. `DEFAULT_CUDA_ARCHS` in
  `tokenspeed-kernel/python/setup.py` is `("100a", "103a")`, i.e. Blackwell
  only — an unqualified rebuild on this box produces a `.so` with no `sm_90a`
  code, which is exactly the hardware under test. Use
  `FLASHINFER_CUDA_ARCH_LIST="9.0a 10.0a 10.3a"`.
* The build identity (gencode flags included) and per-source mtimes decide
  whether a group is skipped, so the arch change is enough to force a rebuild;
  `objs/` is owned by uid 1001 and root inside the container is what can write
  it.

### 8.3 The installed flashinfer rejected a keyword the tree passes

`TypeError: AutoTuner.search_cache() got an unexpected keyword argument
'require_profiling_policy'`. The merged tree's `_reuse_autotune_cache()` asks
flashinfer to *refuse* a cache hit that was profiled under a different policy —
which is the safe direction, because a stale tactic then costs time instead of
correctness. The image ships flashinfer 0.7.1rc2, whose signature is
`search_cache(self, custom_op, runners, input_shapes, tuning_config,
inputs=None)`; the keyword arrived later.

The tree now asks the installed function what it accepts
(`inspect.signature`) and forwards the policy only when it is accepted. A caller
that demands the policy against an older flashinfer gets an explicit
`TypeError` rather than a silently relaxed contract — the whole point of the
keyword is that cache reuse must not be weaker than the caller asked for.
`test/test_tuning_cache.py::test_older_flashinfer_search_cache_is_called_without_the_policy`
pins both halves against a fake `AutoTuner` with the old signature.

This is the general shape of a third-party skew after a merge: probe the
installed API, forward what it takes, and fail loudly on what it does not —
never swallow the request.

### 8.4 The gRPC frontend expects `ServerArgs.canonical_served_model_name`

The engine booted fine and then every `GetModelInfo` RPC failed:
`AttributeError: 'ServerArgs' object has no attribute
'canonical_served_model_name'`, raised by the image's
`smg_grpc_servicer/tokenspeed/servicer.py` at `served_model_name=(...)`.
The frontend and the metrics/exporter code in the image are built against a
`ServerArgs` where `served_model_name` is `str | list[str] | None` (vLLM-style
multi-name) and two properties expose the normalised forms. The merged tree
still had the single-string field, and `# noqa`-free `--served-model-name` with
`type=str`.

The merge is only half a port if one side of an interface moves: the tree now
carries the newer shape — normalisation in `__post_init__` (dedupe, drop empty,
fall back to the model path), `served_model_names`, `canonical_served_model_name`,
and `--served-model-name ... ` with `nargs="+"`. Every call site that needs one
id — metrics labels, ZMQ `instance_id`, KV-transfer args, engine state,
pooling/ASR entry points — takes `canonical_served_model_name`, which is what the
frontend reads. `test/runtime/test_cli_config_compat.py::test_full_server_command`
asserts the normalised list and the canonical name.

Lesson for the next merge: a component shipped *inside* the image (the smg
frontend here) is a consumer of our Python API. Diff its expectations against
`runtime/utils/server_args.py` before a long benchmark run, not after.

### 8.5 The serving image carried a dependency set the merged tree no longer accepts

The TP=4 cell failed 30 seconds into startup:

```
ImportError: cannot import name 'trtllm_fp8_per_channel_scale_moe' from
'flashinfer.fused_moe.core' ... Did you mean: 'trtllm_fp8_per_tensor_scale_moe'?
```

`tokenspeed-kernel/python/requirements/cuda.txt` pins
`flashinfer-python==0.7.2` and `torch==2.14.0`; the image the earlier cells ran
in (`ts-env:gpu-sched39`, built from the `cu130-torch-2.13.0-flashinfer-0.6.18`
runner) had torch 2.13.0 and flashinfer 0.6.17. `per_channel_scale_moe` landed
in flashinfer 0.7.x, and it is not the only one: the tree also imports
`flashinfer.fused_moe.da_moe`, `flashinfer.comm.mnnvl_cutedsl*`,
`flashinfer.fused_moe.utils.get_hybrid_num_tokens_buckets` and `TopKTieBreak`,
none of which exist in 0.6.17. This is a version skew, not a missing optional:
the whole MoE backend moved.

The fix keeps the mount-based benchmark honest — only the dependency layer
changes, the measured tree is still `/workspace` — by building
`ts-env:fi072` from `lightseekorg/tokenspeed-runner:latest` (torch 2.14.0,
`flashinfer-jit-cache-*` 0.7.2 per arch, so H200's `sm90a` cubins are present)
and installing the repo's own `requirements/cuda.txt` into the default user's
site-packages. The in-tree CUDA objects under
`tokenspeed-kernel/.../thirdparty/cuda/objs/*.so` are compiled against torch's
ABI, so they are rebuilt inside that image before the cell is measured; 8.2
already established that a stale prebuilt extension fails this way.

Lesson: pin-check the environment against `requirements/cuda.txt` at the merge
boundary. A dependency upgrade merged as one commit (`Upgrade FlashInfer to
0.7.2`) is invisible in a diff of the runtime code, and every later startup
failure looks like a modelling problem instead of an image problem.
