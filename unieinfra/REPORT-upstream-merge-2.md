# Upstream merge 2: tokenspeed c42a6b12 into `dsv41`

Merging 1078 upstream commits (merge base e7f65f58, upstream head c42a6b12,
2026-10-18) into the fork's 291, to pick up upstream's DeepSeek-V4.1 support
and optimize it on H200 instead of porting V4.1 from vLLM 0.31.

## What the merge costs — read before merging `dsv41` back into `unieai`

| lost | why | follow-up |
|---|---|---|
| **DeepSeek-V4** (V4-Flash, -0731, -Vision-Exp) | upstream deleted it on purpose, no fallback (0c1bc3ed, #2310) | serve V4 from the pre-merge tree, or port V4 onto V4.1's backend |
| **Gemma-4** (per-layer head_dim, plane overlay, triton forcing) | upstream has no Gemma; upstream's `PagedAttention` changed signature | real model port + per-layer head_dim in the cache plan |
| whisper.py / bert.py / gemma.py | upstream `PagedAttention(q,k,v,positions,ctx)`, keyword `rotary_emb=`/`qk_norm=` | port; they fail `test_paged_attention_prologue` |
| MIXED batching on MHA/MSA | upstream backends assert against MIXED | startup guard now refuses it (walks composite backends); re-port 5382c99e |
| `--admission-reserve`, `--context-chunk-policy/-unit`, `--kvstore-backing-path` | backing C++ removed upstream | **40 scripts in /workspace/roy pass `--admission-reserve`, 25 pass `--context-chunk-policy`** |
| NVMe host KV tier, int8 mamba checkpoint pool, multi-snapshot recurrent state | built on replaced L2 executor / state design | decide whether to re-port |
| deterministic DSv4 top-k, DSv4 image-span bidirectional SWA | code rewritten upstream | re-port if needed (top-k determinism matters for RL at T=0) |
| sm_120 FA2 paged extend, bf16 split-K decode MoE, DeepGEMM MoE w/o DeepEP, sm_120 flashinfer GDN | kernels removed or relocated | plugin candidates under `unieinfra/` |

Defaults that changed to upstream's: `enable_mixed_batch` True→False,
`kvstore_io_backend` direct→kernel (the fork's L2 race fix 26470983 is assumed
superseded by upstream's `HostCacheExecutor` — **re-run the RAG C=128
high-reuse shape on both backends before trusting it**).

## Kept, re-implemented on upstream's structure

- scheduler: `max_cold_prefills` (facd44a3; gemma RAG goodput 0.53→0.94),
  per-group bound-parent diagnostic (b5402f2d), new
  `tests/cpp/test_unieinfra_admission.cpp`.
- paged MHA passes `softmax_scale=layer.scaling` (220e4a61) + GPU test;
  fp16 KV for fp16 models; admission startup log on `capacity_report()`;
  symmetric-memory all-gather fallback; `GeluTanhAndMul`; embedding/pooling
  serving end to end; mixed-batch startup guard (now recursive over hybrid
  wrappers and cache-group routers, and MHA/MSA leaves declare
  `supports_mixed_batch = False`).
- kernels: bf16/fp16 query over fp8 KV in Triton MHA (996b05f4); prefill
  early-exit (b93a8a8e) and smem-budget tiles; SM-count flash-decoding split
  (3.54x on 512-wide heads); fa3 head_dim cap; trtllm extend/decode capability
  declarations; fp32 routing weights in cutlass unquantized MoE; RSAG
  multicast error; sm120 plugin trait names fixed (`sinks`, `logit_cap`,
  `value_head_dim` — the old names were silently ignored).

## Dropped because upstream already does it

AlignPrefillChunk (ba69ea89 → upstream takes min(budget, unscheduled)),
page-grid prefill end (1a93b0de → partial-page tails), SWA admission
(39c215ce → reserve capped at window−1), whole-prompt/generation reserve and
two-phase retract (→ `AdmissionHeadroom`), checkpoint stride/slot allocator
(→ two-slot state groups), generation budget (04a18f00), arena bytes
(b5402f2d python half), several kernel items listed in the appendix.

## Verification at merge time

- scheduler: C++ 584/584, python 245/245.
- runtime (merged scheduler on PYTHONPATH, image nctu6/tokenspeed:7ce0d706):
  engine area 287 passed; attention area 141 passed / 19 failed, all 19 from
  the image's old flashinfer/flash_mla or from the gemma/bert/v4 models above;
  kernels 228+4+23 passed. `models.registry` and `models.deepseek_v41` import.
- `pre-commit run --all-files` clean.

Per-area ledgers (every fork commit, decision, evidence) follow.


---

<!-- merge-notes-scheduler -->
## tokenspeed-scheduler merge notes (upstream/main into dsv41)

Approach: I took upstream's version of the whole scheduler, then went through each fork commit since e7f65f58
one at a time. Two small fork features were re-implemented on upstream's structure. Everything else is
already covered by upstream, no longer applies to upstream's design, or is a debug hack.

## Per-file resolution

| file | status | resolution |
|---|---|---|
| CMakeLists.txt | UU | upstream + one test source added (`tests/cpp/test_unieinfra_admission.cpp`) |
| bindings/python_module.cpp | UU | upstream + `SchedulerConfig.max_cold_prefills` (rw) + `Scheduler.cache_group_bound_parents(group_id)` |
| csrc/cache/core/cache_config.h | UU | upstream |
| csrc/cache/core/cache_types.h | UU | upstream |
| csrc/scheduler/operations/cache.cpp | UU | upstream |
| csrc/scheduler/operations/forward.cpp | UU | upstream + max_cold_prefills in `scheduleLocalPrefillWork` (+11 lines) |
| csrc/scheduler/request.cpp / request.h | UU | upstream |
| csrc/scheduler/scheduler.cpp / scheduler.h | UU | upstream + `CacheGroupBoundParents` |
| tests/cpp/integration_test_helper.h | UU | upstream |
| tests/cpp/test_outside_event_handler.cpp | UU | upstream |
| csrc/scheduler/types.h (auto-merged) | M | upstream + `max_cold_prefills` field |
| csrc/scheduler/types.cpp | A (upstream) | + `max_cold_prefills >= 0` check in Validate |
| csrc/cache/coordinator/cache_coordinator.{h,cpp} | A (upstream) | + `GroupBoundLcmBlocks(group_index)` |
| csrc/scheduler/operations/forward.h (auto-merged) | M | reset to upstream (dropped the fork's `mamba_checkpoint_dst_indices`, c286b5f8) |
| csrc/utils.h (auto-merged) | M | reset to upstream (dropped the FatalCheck write to the hardcoded `/workspace/roy/.fatal_check.txt`) |
| python/tokenspeed_scheduler/__init__.py (auto-merged) | M | reset to upstream (dropped the optional `ContextChunkPolicy` export, 0cb87104) |
| tests/cpp/test_unieinfra_admission.cpp | NEW (untracked) | **needs `git add`**: 4 tests for the two re-implemented features |

### Files to `git rm`
UD (changed in the fork, deleted upstream). Decision: delete, because upstream replaced them:
- tokenspeed-scheduler/csrc/cache/coordinator/kv_cache_admission.cpp  (now cache_admission.cpp)
- tokenspeed-scheduler/csrc/cache/coordinator/kv_cache_coordinator.cpp (now cache_coordinator.cpp)
- tokenspeed-scheduler/csrc/cache/manager/kv_cache_manager.h (now cache/cache_group.h + allocator/group_allocator.h)
- tokenspeed-scheduler/csrc/cache/manager/swa_manager.h (now cache/cache_group.h)
- tokenspeed-scheduler/tests/cpp/test_kv_cache_scenarios.cpp (now test_cache_scenarios.cpp)

Fork-only files whose features were dropped. They are still on disk, are no longer in CMake, and also need `git rm`:
- tokenspeed-scheduler/csrc/resource/allocator/checkpoint_slot_allocator.cpp
- tokenspeed-scheduler/csrc/resource/allocator/checkpoint_slot_allocator.h
- tokenspeed-scheduler/tests/cpp/test_checkpoint_slot_allocator.cpp
- tokenspeed-scheduler/tests/cpp/test_forward_checkpoint_channel.cpp
- tokenspeed-scheduler/tests/cpp/test_swa_checkpoint_stride.cpp
- tokenspeed-scheduler/tests/cpp/test_context_chunk_policy.cpp

## Per fork commit (e7f65f58..HEAD touching tokenspeed-scheduler)

Commits before 7471e09c (e1852925 ... 9b2d1d0f) were already dropped or absorbed by the previous merge, which
took upstream's scheduler (REPORT-upstream-merge.md:14-17). Nothing of theirs is left to carry.

| commit | subject | decision | evidence |
|---|---|---|---|
| e00414d4 | drop snapshot-depth test | dropped (no-op) | the test does not exist upstream |
| ba69ea89 | re-aligning chunk keeps going past page boundary | dropped: fixed upstream | upstream no longer page-aligns chunks: `PrefillChunkTokens` takes `plain = min(budget, unscheduled)` (csrc/scheduler/operations/prefill_chunk.cpp:49) and cuts only at a junction or the final replay window. `AlignPrefillChunk` is gone. |
| 1a93b0de | end the last prefill round on the page grid | dropped: fixed upstream | upstream publishes token-level tails at prompt end (cache/prefix/tail_table.h, `recordPublishedTail` cache_admission.cpp:823, tests/cpp/test_token_level_hits.cpp). An unaligned prompt end is now matchable, so leaving a ragged tail for a later round no longer buys any reuse. The state group is two slots (cache_config.h:37-41) and is checkpointed at the prompt end. |
| 3165a40d | trace admission refusals (TOKENSPEED_LOG_ADMISSION) | dropped, follow-up | diagnostic in the deleted kv_cache_admission.cpp. Follow-up: if needed, add an env-gated spdlog line where `CacheCoordinator::Admit` refuses (cache_admission.cpp), printing parents needed vs empty parents per group. |
| b40ea1bc | comment: why eviction stays recency-first | dropped | comment only, in a deleted file |
| cd56c305 | charge whole prompt at admission; widen preempt trigger | dropped: covered upstream | upstream prepays the rest of the prompt plus escalating decode room at first-chunk admission: `PrefillReserve::prompt_headroom_tokens` (operations/group_demands.h) and `Request::AdmissionHeadroom` (request.h:68). Retraction was redesigned (`maybeRetractForCapacity` / `retractVictim`, scheduler.h:373/403). AGENTS.md also records that `preempt_on_blocked_request` measured -36%/-61%. |
| b3356502 | admission reserve policy (chunk/prompt/generation-adaptive/full, new_token_ratio) | dropped: covered upstream | same as above. Upstream's headroom is a single built-in policy (prompt + safe-step decode window, escalating per retraction, request.h:51-71). The `reserve_*_at_admission` / `new_token_ratio_*` knobs no longer exist. |
| 7a0d5539 | two-phase retract (pending_retraction_) | dropped: covered upstream | upstream's retraction defers victims that still owe a forward result through its image/epoch machinery (scheduler.h:347-456). |
| ddefef3a | charge a sparse (checkpoint-stride) group its real pages | dropped: no longer applies | upstream State groups are two slots (input checkpoint + live block), not one page per block (cache_config.h:37-41). There are no per-page state snapshots to thin, so `checkpoint_stride_blocks` has nothing to act on. |
| d18e9020 | don't charge admission for unneeded cache | dropped: no longer applies | same as ddefef3a (it edited kv_cache_manager.h stride accounting) |
| 83e44c7b | expose checkpoint stride as a server arg (scheduler part) | dropped: no longer applies | same as ddefef3a |
| db1240e6 | checkpoint slot allocator | dropped: covered upstream | upstream has its own snapshot pool: `SchedulerConfig::snapshot_allocator` (types.h:45), SnapshotStore/RestoreBatch |
| c286b5f8 | checkpoint destination channel on ForwardBatch (`mamba_track_pool_indices`) | dropped: covered upstream | depends on db1240e6. Upstream's state slots travel in the block tables. |
| 6270a663 / e6996f5e | cap mid-prefill enders / revert | net zero | - |
| facd44a3 | cap cold-cache admissions per step (`max_cold_prefills`) | **re-implemented** | measured win (gemma-4-31B RAG c=64 goodput 0.53 to 0.94), default 0 means off. On upstream it is 11 lines in `Scheduler::scheduleLocalPrefillWork` (forward.cpp): counts new-prompt admissions with `extend_prefix_len == 0`, and once the cap is reached the new-prompt tier stops for this plan. Resident chunks are untouched. Also added the config field and Validate check, plus the binding `SchedulerConfig.max_cold_prefills`. |
| 0cb87104 | equal-progress context chunking policy | dropped, follow-up | experimental, default fcfs, about 200 lines written against the old schedulePrefill and page alignment. Follow-up: port TRT-LLM kEQUAL_PROGRESS onto upstream's `scheduleLocalPrefillWork` + `PrefillChunkTokens`. Offer each Prefilling request one unit per pass until `build.token_budget` is spent, keeping the replay-window rule (`ChunkKeepingFinalWindow`). Re-measure corr(TTFT, input_len) on agent-26k first. The commit's hardcoded `/workspace/roy/.admit_trace.txt` match trace is dropped for good. |
| 39c215ce | bound a sliding group's admission charge at window + one step (AGENTS.md "Worked example") | dropped: fixed upstream | upstream caps a sliding-window group's reserve at the window lookback: `groupReserveTokens` gives `min(prompt_headroom, unrecycled)` with `unrecycled <= lookback = window-1` (operations/group_demands.cpp:54-63). Slid-out pages are reclaimed during admission (`ReclaimExpired`, cache_admission.cpp ~817). The whole-prompt charge that the worked example describes can no longer happen. |
| b5402f2d | per-group bound-parent diagnostic | **re-implemented** | upstream `GroupAvailablePages` still has the same flaw: it is empty parents x packing + free slots (cache_coordinator.cpp:958-964). Added `CacheCoordinator::GroupBoundLcmBlocks`, `Scheduler::CacheGroupBoundParents`, and the binding `cache_group_bound_parents(group_id)`, renamed from `paged_cache_group_bound_parents` to match upstream naming. |
| (diagnostics) | `.plan_trace.on` / `.admit_trace.txt` / `.fatal_check.txt`, hardcoded `/workspace/roy` paths in forward.cpp and utils.h | dropped | debug hacks with absolute paths. Follow-up if wanted: an env-var-selected sink. |

### Impact on the Python side (other agents)
Fork Python calls that no longer exist in the merged extension:
- `PagedCacheGroupConfig(..., checkpoint_stride_blocks=...)`: now `CacheGroupConfig` (block_granularity, attn_kind, ...). Drop the stride and `--state-checkpoint-stride-blocks`.
- `cfg.reserve_full_prefill_at_admission`, `cfg.reserve_generation_at_admission`, `cfg.new_token_ratio_init` (`--admission-reserve`): removed. Drop the flag.
- `cfg.context_chunk_policy`, `cfg.context_chunk_unit`, `ContextChunkPolicy`: removed. Drop `--context-chunk-policy` and `--context-chunk-unit`.
- `ForwardBatch.mamba_track_pool_indices`: removed.
- `scheduler.paged_cache_group_bound_parents(g)` is now `scheduler.cache_group_bound_parents(g)`, and `paged_cache_group_total/available_pages` are now `cache_group_total/available_pages` (upstream rename).
- `cfg.max_cold_prefills` is **kept** under the same name, so `--max-cold-prefills` wiring can stay.
- `cfg.block_size` is now `cfg.prefix_granularity` and `paged_cache_groups` is now `cache_groups` (upstream rename).

## Build / test

Image `ts-env:gpu` has g++ 13 and ninja, but no cmake, nanobind or spdlog. I installed the build tools to a side dir
(`/data3/dsrl-cache/ui-cache` is mounted as `/home/runner/.cache`):

```
UI=<runner script: docker run of the runtime image with this tree on PYTHONPATH>
GPUS=4 $UI bash -lc 'pip install -q --target /home/runner/.cache/ts-sched-dsv41/tools "cmake>=3.22" "nanobind>=2.13.0" "tokenspeed-spdlog==1.15.1" scikit-build-core'
## C++ tests + extension
GPUS=4 $UI bash -lc 'T=/home/runner/.cache/ts-sched-dsv41/tools; export PATH=$T/bin:$PATH PYTHONPATH=$T; \
  cmake -S tokenspeed-scheduler -B /home/runner/.cache/ts-sched-dsv41/cbuild -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=ON \
    -DCMAKE_PREFIX_PATH="$T/tokenspeed_spdlog;$T/nanobind/cmake" && \
  cmake --build /home/runner/.cache/ts-sched-dsv41/cbuild -j 32 && /home/runner/.cache/ts-sched-dsv41/cbuild/tokenspeed_scheduler_tests'
## installable package (python ext) into a target dir, out-of-tree build dir
GPUS=4 $UI bash -lc 'T=/home/runner/.cache/ts-sched-dsv41/tools; export PATH=$T/bin:$PATH PYTHONPATH=$T CMAKE_PREFIX_PATH="$T/tokenspeed_spdlog;$T/nanobind/cmake"; \
  pip install -q --no-deps --no-build-isolation --target /home/runner/.cache/ts-sched-dsv41/pkg \
    -C build-dir=/home/runner/.cache/ts-sched-dsv41/wbuild ./tokenspeed-scheduler'
## python tests
GPUS=4 $UI bash -lc 'export PYTHONPATH=/home/runner/.cache/ts-sched-dsv41/pkg:$PYTHONPATH; cd tokenspeed-scheduler && python3 -m pytest -q -p no:cacheprovider python/tests'
```

Results: the C++ build is clean (about 50 s). The **C++ suite passes 584/584**, including the 4 new UnieInfra tests.
The **Python suite passes 245/245**. Host paths: /data3/dsrl-cache/ui-cache/ts-sched-dsv41/{cbuild,pkg,tools}.
No conflict markers remain under tokenspeed-scheduler/.

---

<!-- merge-notes-engine -->
## Merge notes — engine/execution/cli/server_args scope

Merge of upstream/main into dsv41 (base e7f65f58). Rule: upstream structure wins; fork kept only where still needed.

## .gitignore
- conflict: fork `tokenspeed-scheduler/build*/` (cd56c305) vs upstream `AGENTS.override.md`, `outputs/` -> union.

## AGENTS.md
- c277b35c (perf-work section + MIXED-step KV write worked example): KEPT, placed after upstream's new AMD-section bullets.

## distributed/comm_backend/trtllm_allreduce.py
- 22755d48 (TOKENSPEED_DISABLE_TRTLLM_AR escape hatch + logger): KEPT. Conflict was only the comment above MAX_ONESHOT_BYTES; took upstream's comment, kept fork `logger`.

## execution/types.py
- d21cfed9 (embedding: `output_embeddings` on ModelExecutionResult): KEPT, appended after upstream's new logprob fields, before `_synced`.

## execution/input_buffer.py
- f04e36e5 (reranker packed segment bit-30 clamp): KEPT on top of upstream's new `fill_ngram_inputs(...)` call; upstream's trailing MM-pad comment was moved upstream, so fork's reference comment reworded.

## execution/forward_step.py (upstream renamed cuda_graph_wrapper.py -> forward_step.py)
- d21cfed9 (decode graph disabled for pooling models): KEPT, expressed on upstream's new `self.disable = enforce_eager or not decode_graph_supported or prefill_only` as an extra `or getattr(config, "is_embedding", False)`.

## engine/input_processor.py
- d21cfed9 (cross-type request refusals for embedding servers): KEPT; upstream's new `_validate_data_parallel_rank(obj)` call appended.

## entrypoints/engine.py
- ec6a0761 (`_check_peer_access`, fail multi-GPU launches early on non-peer PCIe pairs): KEPT alongside upstream's new `_force_env`; call site `_check_peer_access(server_args)` still in `_set_envs_and_config`.
- d21cfed9 (`Engine.encode`): auto-merged, kept.

## execution/runtime_states.py
- f04e36e5 (`packed_token_type_ids` ctor arg): KEPT alongside upstream's ngram/request-history fields.
- f162d993 / 5ef6e0f7 (copy_mamba_states / snapshot_mamba_checkpoints / zero_mamba_states with int8 `ckpt_pool`): DROPPED. Upstream removed `mamba_pool` from RuntimeStates entirely; recurrent state is now a cache group handled by `layers/attention/backends/state/mamba.py` (+ upstream 7978c48b single-forward checkpoint materialisation). Nothing in the merged tree constructs a `ckpt_pool` any more. FOLLOW-UP: `cache/mamba_checkpoint_pool.py` (fork int8 checkpoint pool) is now orphaned; re-port onto the state backend only if the int8 checkpoint dtype is still wanted.

## engine/request_handler.py
- d21cfed9 (`handle_embedding_request`): KEPT next to upstream's internal-op / L3 weight-version block. Re-expressed for upstream signatures: `make_spec(..., max_cached_prefix_tokens=UNBOUNDED_CACHED_PREFIX_TOKENS)`, and RequestState built directly (upstream `from_recv_req` now reads logprob fields that TokenizedEmbeddingReqInput lacks and requires `computes_prompt_logprobs`). max_new_tokens=0 = "no declared budget" in upstream's scheduler (request.h AdmissionHeadroom) - valid.
- 04a18f00 (hand scheduler the settled max_new_tokens): OBSOLETE - upstream `_apply_generation_budget` (request_handler.py ~L926-939) does exactly this.

## engine/scheduler_utils.py  -> upstream + max_cold_prefills
- b3356502 admission-reserve policy (reserve_full_prefill_at_admission / reserve_generation_at_admission / new_token_ratio_init): DROPPED - fields gone from the C++ binding (scheduler = upstream). Upstream admission reserves min(remaining max_new, safe_steps*(1+retractions)) with retraction escalation (request.h) - not identical to fork "generation-full", so the flag is not kept as an alias.
- 0cb87104 context chunk policy (ContextChunkPolicy, equal-progress): DROPPED - not in binding.
- facd44a3 max_cold_prefills: KEPT (scheduler agent kept `cfg.max_cold_prefills`): `make_config(max_cold_prefills=)` + validation; event_loop passes `server_args.max_cold_prefills`.
- 83e44c7b checkpoint_stride_blocks on PagedCacheGroupConfig: DROPPED here (class is upstream CacheGroupConfig; group configs come from `pool.arena.runtime_contract`). `--state-checkpoint-stride-blocks` server arg is still read by kv_cache/recipes/qwen35.py (other agent's file).
- 39c215ce SWA group.max_scheduled_tokens admission bound: DROPPED (field gone; upstream CacheGroupConfig has sliding_window_tokens). FOLLOW-UP: verify upstream's SWA admission charge is window+step, not whole span (the gemma-4 defect in AGENTS.md worked example 1) - C++ side.
- b5402f2d (_kv_pool_bytes measures arena allocation, not overlapping views): OBSOLETE - upstream `_kv_pool_bytes` uses `CacheArena.allocated_device_bytes`.

## engine/event_loop.py  -> upstream + max_cold_prefills wiring
- Upstream moved all device construction into `execution/device.py::build_device_side`; create_attn_components / pool_to_paged_cache_groups / aligned_max_scheduled_tokens / L2CacheExecutor call sites are gone.
- a8c0c872 `validate_mixed_batch_backend` (mixed-batch startup guard, REPORT defect #1): KEPT, moved to `execution/model_executor.py` and called in `ModelExecutor.__init__` right after upstream's `validate_scheduler_config` (that is where upstream validates backend/pool compatibility). New `ModelExecutorConfig.enable_mixed_batch`. test/runtime/test_mixed_batch_backend_guard.py import path updated.
  !! The guard reads `attn_backend.supports_mixed_batch` (default True). Upstream's `backends/paged/mha.py:186` and `paged/msa.py:173` still `assert not forward_mode.is_mixed()` without declaring `supports_mixed_batch = False`, so `--enable-mixed-batch` on an MHA/MSA model reproduces defect #1 (healthy server, every request 500) unless the attention agent declares it (or ports MIXED, 5382c99e/c277b35c).
- 07565572/97300c4b/af8deafd/fc897601/da1e8263 `resolve_fi_gdn_prefill_graph` + `model_may_use_fi_gdn`: DROPPED. The gate keyed on TOKENSPEED_ENABLE_FI_GDN_SM120, which controlled the fork's sm_120 flashinfer GDN kernel; that kernel (ops/attention/flashinfer/gated_delta_rule.py) is deleted and upstream's gdn/flashinfer.py registers only sm90..sm103. Keeping it would have disabled the prefill graph for every GDN model on the default path. test/runtime/test_fi_gdn_prefill_graph_gate.py removed from the worktree (tests only the dropped function). FOLLOW-UP if the sm_120 FI GDN kernel is re-ported: re-add the capture guard with it.
- 50bac760 L2 backing_path wiring: DROPPED (see l2/executor.py).
- 1e83b725/35641d19 merges, a2aa5385 defaults: subsumed by the above.

## cache/l2/executor.py -> upstream wholesale
- Upstream rewrote it as HostCacheExecutor (L2 + retraction snapshot tier, per-submission prerequisite streams, staging lanes, pinned-until-ACK publications).
- 26470983 (order write_stream vs load_stream; TOKENSPEED_L2_SERIALIZE_STREAMS): treated as OBSOLETE by redesign: every copy now runs on its own stream ordered after the caller-named prerequisite stream, stream-ordered stores fence the plan's fence stream, and ordinary publications keep their source pages held by the scheduler until the ACK. FOLLOW-UP (must measure, not argue): rerun the RAG C=128 high-reuse shape (the one that lost 146/256 with 2 illegal-memory faults) on the merged tree, both `--kvstore-io-backend kernel` and `direct`.
- 50bac760 / 765e0bd6 / e211bbaa / fc791936 (NVMe file-backed Host tier, O_DIRECT, staging slots, 2GB syscall loop): DROPPED - built on the old L2CacheExecutor; re-porting onto HostCacheExecutor + transfer lanes is a feature port, not a merge resolution. `--kvstore-backing-path` removed. `cache/l2/storage.py` (auto-merged, not my file) still carries the unused `backing_path` option.
- b5402f2d hunk: obsolete (upstream sizing).

## utils/server_args.py
Kept (feature survives): embedding flags (--is-embedding, --pooling-type, --pooling-normalize, --embedding-batch-tokenize, --embedding-skip-kv-write, --pooling-prefill-graph), --cross-attn-path (whisper), --state-checkpoint-stride-blocks (read by qwen35 recipe), TOKENSPEED_DISABLE_ALLREDUCE_FUSION escape hatch (a8882b72), --max-cold-prefills (re-added after scheduler agent kept the C++ field), --disable-mixed-batch (alias store_false into upstream's enable_mixed_batch).
Dropped: --admission-reserve, --context-chunk-policy, --context-chunk-unit, --kvstore-backing-path, kvstore_mem_layout.
Defaults changed to upstream: enable_mixed_batch True -> False (a2aa5385); kvstore_io_backend "direct" -> "kernel" (fork measurement was on the old executor); block_size -> upstream prefix_granularity (--block-size remains a deprecated alias).

## execution/model_executor.py
- d21cfed9 embedding path: KEPT, re-expressed on upstream's structure: config fields is_embedding / pooling_prefill_graph; DP-sampling setup skipped for pooling; `_forward_step` returns PoolerForwardResult early via upstream `_run_target_forward(ctx)`; execute_forward_op skips sampling prep, unpacks either PoolerForwardResult or upstream's 4-tuple, skips `_update_runtime_state`, D2Hs embeddings, supplies an empty ForwardLogprobs for the result.
- f04e36e5 `_packs_token_type_ids` + config field: KEPT.
- 2a2f2fbf whisper: prefill graph disabled for encoder-decoder models (kept as model intent next to upstream's "user intent only" disable). The autotune-forward req_pool_indices/seq_lens kwargs are dropped: upstream's model_runner.forward(ctx, ids, positions, **kwargs) no longer takes them. FOLLOW-UP (models agent): confirm models/whisper.py reads per-request state from ForwardContext under upstream's signature.
- 05d0a15c (prefix cache hit rate in "Prefill batch" line), c812f2f8 (arena-saturation warning), b5402f2d maybe_log_paged_cache_group_pages: DROPPED from this file - upstream moved batch logging to engine/batch_log.py (BatchLogger). FOLLOW-UP: port hit-rate + saturation warning into BatchLogger._log_extend/_log_decode (not in my file list).
- a8c0c872 guard: see event_loop.

## execution/prefill_graph.py
- a653c7d9 (gather embeddings through model's own seam `get_input_embeddings`, gemma sqrt(hidden)/qwen scale_emb): KEPT; also applied to upstream's new narrowing path (line ~829) which called `self._embed_tokens` directly.
- e3f8da8a / 848fb6aa / 2b0ef40b (pooling models capture; named disable reasons; supports_prefill_graph declaration): KEPT, upstream's new conditions (graph_supported, requires_request_token_history, narrowing under DP) folded into the named-reasons list; upstream's `not model_runner.is_generation` replaced by "non-generation model without a pooler"; pooling tail `text_model.pool(ctx, hidden)` after upstream's replay.
- da1e8263: n/a (GDN gate lived in event_loop, dropped).
- capture(): upstream (ForwardStepRunner always owns `.stream`, so fork's getattr guard is unnecessary).

## cli/serve_smg.py, test/cli/test_serve_smg_unit.py
- b12febcc (TOKENSPEED_SMG_PROMETHEUS_PORT pin): KEPT on top of upstream's `_free_port_avoiding_ephemeral_range`; unset -> upstream's ephemeral-avoiding port. Tests: union.

## Other test edits (fork-only tests of my files)
- test/runtime/test_server_args_mixed_batch.py: default-on assertion -> default-off (upstream default).
- test/runtime/test_mixed_batch_backend_guard.py: import from execution.model_executor.
- test/runtime/test_fi_gdn_prefill_graph_gate.py: deleted (worktree only).

## Found outside my scope (needs an owner)
- engine/generation_output_processor.py:813-814 (auto-merged fork d21cfed9): `with nvtx_range(...): results.sync()` - `nvtx_range` undefined there, and upstream's ModelExecutionResult.sync() now RAISES on a second call (PendingExecution.result() already synced). Fix: delete those two lines. Without it every embedding request crashes the commit path.
- attention backends: declare `supports_mixed_batch = False` on upstream paged/mha.py and paged/msa.py (or port MIXED), else the guard is toothless.

## Verification
- py_compile: all 15 .py files in scope OK; pyflakes (in container): clean on all scope files; no conflict markers.
- Import of model_executor/event_loop: blocked by environment, not by these files: tokenspeed_kernel/ops/moe/flashinfer.py:1235 needs `trtllm_fp8_per_channel_scale_moe`, which the image's flashinfer lacks (upstream kernel needs newer flashinfer).
- pytest (GPU 4, PYTHONPATH incl. /home/runner/.cache/ts-sched-dsv41/pkg): test/cli/test_serve_smg_unit.py + test_server_args_mixed_batch.py 126 passed; test_event_loop_scheduler_config 4, test_runtime_states 13, test_request_handler_flush_cache 33, test_request_handler_profile 19, test_l2_cache_hooks 14, test_prefill_graph_input_embeds 8, test_embedding_pooling 16, test_event_loop_in_flight 8 passed. test_prefill_graph (1 fail) and test_mixed_batch_backend_guard (collection) fail only on that same flashinfer import.

---

<!-- merge-notes-attention -->
## Merge notes: attention / kv-cache / models / model_config scope

Worktree /workspace/roy/UnieInfra-dsv41, merge of upstream/main (a24104a7) into dsv41.
Base e7f65f58. Nothing staged; all resolutions are working-tree writes.

## Upstream structure that decides most of this

* ed6f83dc (#1299, "unify eager/graph decode and spec/non-spec sampling") deleted
  `layers/attention/backends/{mha,msa,hybrid_linear_attn,deepseek_v4}.py` and
  `layers/attention/linear/mamba_state_scatter_triton.py`. Replacements:
  `backends/paged/{mha,msa,trtllm,...}.py`, `backends/hybrid/linear.py`,
  `backends/state/{checkpoint,gdn,mamba,...}.py`, `backends/paged/group_tables.py`.
* 0c1bc3ed (#2310) deleted the DeepSeek V4 attention state, V4 backend, and
  `models/deepseek_v4{,_dspark,_next}.py` "Do not consolidate them or keep a
  replacement/fallback", keeping only V4.1 (`models/deepseek_v41*.py`).
* KV is written by the attention prologue (`PagedAttention.prologue` /
  `gqa_prologue`, docs/design/attention-prologue.md), not by the backend.
* Recurrent state is a two-slot table with "natural checkpoints" only
  (docs/design/cache-concepts.md "Recurrent state: two slots, natural checkpoints":
  "No forward computes an interior checkpoint").
* b5adb75b removed MiniMax M2 support upstream.

## DeepSeek-V4 status

**Not served.** Upstream intentionally dropped V4 (0c1bc3ed). `DeepseekV4ForCausalLM`
has no EntryClass anywhere in upstream; only stale names remain in
`model_config._DEEPSEEK_V4_ARCHITECTURES`, `hf_transformers_utils` DSpark renaming
and `cli/serve_smg.py` parser selection. /models/deepseek-ai/DeepSeek-V4-Flash,
-0731 and -Vision-Exp (all `DeepseekV4ForCausalLM`) will fail at model resolution.
Accepted, not resurrected. V4.1 (`DeepseekV41ForCausalLM`, incl. upstream
`deepseek_v41_vision.py`) is the supported path.

## Per file

### UD files: accept upstream deletion (leave on disk; caller `git rm`)
| file | fork commits | decision |
|---|---|---|
| backends/deepseek_v4.py | 481dacee image-span bidirectional attn | dropped with V4 (0c1bc3ed). Upstream V4.1 has no image-span bidirectional path (grep `bidirectional|image_span` over deepseek_v41* is empty) -> follow-up only if V4.1 vision needs it |
| attention/deepseek_v4/metadata.py | 481dacee | same |
| models/deepseek_v4.py | f36960ca, 481dacee, a8882b72 | same; f36960ca (deterministic top-k order) is kernel-side, check with kernel agent |
| models/deepseek_v4_dspark.py | 481dacee | same |
| backends/mha.py | 14 commits (see below) | replaced by backends/paged/mha.py |
| backends/msa.py | 1d759143 (MIXED -> NotImplementedError + supports_mixed_batch=False) | upstream paged/msa.py:173 asserts not mixed; startup guard is follow-up (see mixed batch) |
| backends/hybrid_linear_attn.py | a8c0c872 (mixed guard delegate), af8deafd/b01a4e34 (GDN graph-corruption diagnostics) | replaced by backends/hybrid/linear.py + state/gdn.py; diagnostics were for the fork's GDN prefill-graph investigation, not re-ported |
| linear/mamba_state_scatter_triton.py | f162d993 (checkpoint-pool dtype, three bugs on --enable-int8-mamba-checkpoint path) | obsolete: upstream removed the per-page mamba checkpoint pool for the two-slot state design (state/checkpoint.py) |
| test/runtime/layers/test_gdn_flashinfer_fastpath.py | bdbb3571 (sm_120 fused chunk prefill) | upstream deleted in 4965b57d (GDN torch op removed); kernel-side, sm_120 only |

### backends/mha.py fork commits vs upstream backends/paged/mha.py
* 220e4a61 softmax scale never passed -> **(b) re-implemented**: paged/mha.py now
  passes `softmax_scale=layer.scaling` to mha_prefill, mha_extend_with_kvcache and
  mha_decode_with_kvcache (upstream passed none; every upstream MHA model declares
  head_dim**-0.5 so this is a no-op for them; inkling/qsa already pass layer.scaling).
  Test: test/runtime/layers/test_paged_mha_softmax_scale.py (real Triton prefill vs
  SDPA reference at scale 1.0 and 1/sqrt(d)).
* 974e73e8 gemma non-uniform head_dim -> force Triton: (c) with the Gemma port
  (upstream has no per-layer head_dim at all).
* 5382c99e / c277b35c / 64e632d8 / a8c0c872 mixed prefill+decode on MHA
  (`_forward_mixed`, decode-tail KV locs, supports_mixed_batch): **(c) dropped**.
  Upstream paged/mha.py:186 `assert not forward_mode.is_mixed()`. The MIXED-step
  KV-write defect from AGENTS.md cannot occur (no mixed path); but
  `--enable-mixed-batch` on an MHA/hybrid model asserts at the first mixed step.
  Re-port = mixed metadata in paged/mha.py init_forward_metadata + decode views +
  `padded_write_locations` covering the decode tail, then a capability flag the
  event loop reads (router/hybrid must delegate it).
* 49d1d455 / 5887cec2 flashinfer paged-extend plan install: (c). Upstream has no
  `install_paged_extend_plan`; the fork kernel (if the kernel agent keeps it)
  self-plans with a host sync per layer. sm_120-motivated.
* 14edfe6e / 5887cec2 do not cast q to fp8 on Triton/flashinfer: (c). Upstream
  triton MHA kernels register fp16/bf16 only (no fp8 KV at all); behaviour depends
  on kernel-agent outcome. H200 fp8 KV goes to fa3/trtllm, which want fp8 q.
* 3796b145 embedding skip-KV-write: (c) dropped. Upstream writes KV in the
  prologue; fork commit itself noted it was unreachable on bf16+triton.
  `server_args.embedding_skip_kv_write` is now dead (engine agent's file).
* 4eeb86da gemma prefill tiles / diagnostics: kernel-side / dropped.

### Conflicted files
| file | resolution |
|---|---|
| layers/activation.py | upstream + keep fork `GeluTanhAndMul` (used by fork gemma.py): import `gelu_tanh_and_mul` from flashinfer (None on AMD -> native), `pdl_enabled` from tokenspeed_kernel.platform |
| attention/backends/base.py | upstream. Fork `supports_mixed_batch` (64e632d8) dropped, see mixed batch |
| attention/configs/mha.py | upstream + `_resolve_mha_kv_cache_dtype`: `auto` on a fp16 model -> fp16 cache (974e73e8 Whisper fix; restricted to fp16 since no fp32 paged kernel). Gemma per-layer geometry (09f124ac, 40f2478a) dropped -> (c) |
| kv_cache/base.py | upstream. b5402f2d/48373af8 diagnostics (KV arena allocated log, `maybe_log_paged_cache_group_pages`, `bind_paged_cache_scheduler`) dropped: arena moved to kv_cache/arena.py, scheduler APIs renamed. **model_executor.py:2157/2232 still call `maybe_log_paged_cache_group_pages` (engine agent)** |
| kv_cache/factory.py, kv_cache/mha.py, recipes/ordinary.py, recipes/setup.py | upstream; fork changes were Gemma-4 per-layer head_dim (40f2478a, 9f575f63) -> (c) |
| recipes/plan.py | upstream; 21c4f936/873b731a geometry-family overlay only matters for multi-geometry (Gemma 4) -> (c) |
| recipes/qwen35.py, recipes/spec.py | upstream; 83e44c7b/d18e9020 state checkpoint stride obsolete under two-slot state (state/checkpoint.py); de7c4282 numbered sub-group labels was Gemma's split_groups_to_equal_depth -> (c) |
| attention/registry.py | upstream + re-implemented 9c163582/af1edbe4 "Cache admission" startup log (slots per group, slots=1 groups, ceiling, binding_utilization from upstream `capacity_report()`), f-string, DEBUG under probe like upstream's Cache profile line. Gemma layout dump dropped |
| layers/logits_processor.py | upstream create_state(device=None) wrapped in fork try/except fallback (ec6a0761): upstream `_tp_group_multicast_reachable` only rules out cross-host groups, not single-host PCIe |
| moe/weights/loaders.py | upstream (copy_expert_shard handles e8m0 bytes); fork load_w13 error-context dropped |
| paged_attention.py | upstream; fork `group_id` ctor arg superseded by bind_cache_group, `skip_kv_cache` dropped |
| quantization/modelopt_mixed.py | upstream. Linear W4A16_NVFP4/FP8 obsolete (Nvfp4W4A16Config, fp8_static_config upstream). MoE W4A16_NVFP4/FP8 experts still refused by upstream `_MOE_CONFIG_ATTRS` -> (c), sm_120-only (config min capability 100 anyway) |
| models/base/causal_lm.py | upstream PP structure + fork pooler (d21cfed9): lm_head only on last PP rank and when no pooler; `get_embedding` pools after the PP early return. `_skip_kv_cache_writes` dropped |
| models/qwen3.py | upstream (prologue fuses norm/RoPE/KV write -> edd295cf obsolete) + keep 3eb83e9d skip of ModelOpt k/v scales without a param (same as upstream qwen3_moe ignore_suffixes) |
| configs/model_config.py | upstream + fork pooling config / embedding server-arg resolution / Whisper bits; `embedding_skip_kv_write` stamp dropped; V4 multimodal entries dropped; Whisper kept in multimodal+audio lists; added `server_args is not None` guards around `_clamp_prefill_budget_to_context`, cross-attn stamp and embedding resolution (fork crashed on server_args=None) |
| utils/hf_transformers_utils.py | both lists (V4.1 + Gemma4 composite config), keep `max_target_positions` |

## Fork-only files now broken against upstream (outside conflicted scope)
* models/gemma.py: PagedAttention(group_id=) removed; upstream requires
  keyword `rotary_emb=`, `qk_norm=`, attn call is `attn(q,k,v,positions,ctx)`;
  no per-layer head_dim in cache plan. test_paged_attention_prologue
  `test_models_run_the_prologue_steps_only_through_the_prologue` flags
  gemma.py and bert.py. Needs a real port (c).
* models/whisper.py: same PagedAttention ctor/call API mismatch (TypeError at
  construction). Small port: `rotary_emb=None, qk_norm=None`, call
  `self.attn(q, k, v, positions, ctx)`.
* models/bert.py: flagged by prologue test (own rotary outside prologue).
* models/deepseek_v4_vision.py: only imported by deleted deepseek_v4.py -> git rm.
* runtime/cache/mamba_checkpoint_pool.py imports deleted mamba_state_scatter_triton.
* Tests importing deleted modules: test/runtime/test_mixed_batch_metadata.py,
  test_mixed_batch_backend_guard.py, test_mha_softmax_scale.py,
  test_mha_forwards_softmax_scale.py (AST of deleted backends/mha.py),
  test/runtime/models/test_deepseek_v4_image_span.py, test_gemma4.py,
  test/runtime/cache/test_mamba_checkpoint_snapshot_paths.py.

## Verification
* No conflict markers in scope; all touched files compile (ast+compile).
* Imports OK (ui.sh, merged scheduler on PYTHONPATH): attention.registry,
  backends.paged.mha, configs.mha, configs.model_config, activation,
  logits_processor, paged_attention, modelopt_mixed, hf_transformers_utils,
  models.whisper, models.deepseek_v4_vision.
* Blocked by the image, not the merge: anything importing MoE kernels
  (`flashinfer.fused_moe.core.trtllm_fp8_per_channel_scale_moe` missing) incl.
  causal_lm, qwen3, deepseek_v41; deepseek_v41 also needs
  `flash_mla.fused_norm_rope_attn_rope_cast`. Image's flashinfer/flash_mla are
  older than upstream requires.
* pytest (GPU 4): test_paged_mha_softmax_scale (2), test_embedding_pooling,
  test_hf_transformers_utils, test_group_aware_mha, test_mha_pool_spec_decode_groups,
  test_logits_processor: pass. 141 passed / 19 failed overall; failures are
  env (MoE import, flashinfer TuningConfig `tensor_initializers`) in
  test_cache_setup (16) and test_paged_attention_prologue (2), plus the
  prologue source scan naming gemma.py/bert.py/deepseek_v4.py.

---

<!-- merge-notes-kernel -->
## Merge notes: tokenspeed-kernel/ (+ unieinfra/kernel-sm120)

Merge of upstream/main into dsv41 (base e7f65f58). Rule applied: upstream's
layout wins; a fork change survives only if still needed and expressible on
upstream's structure. Upstream reorganised ops into `ops/<family>/<backend>.py`
with private `_<backend>/` impl dirs (e.g. `ops/attention/triton/mha_decode.py`
-> `ops/attention/mha/_triton/decode.py`, with registration moved to
`ops/attention/mha/triton.py`; `ops/moe/flashinfer/*.py` -> `ops/moe/flashinfer.py`;
`ops/attention/flashinfer/__init__.py` -> `ops/attention/mha/flashinfer.py`;
trait names `support_sinks`/`support_logit_cap` -> `sinks`/`logit_cap`;
`head_v_dim` -> `value_head_dim`).

## Per-file resolution

| path | status | resolution |
|---|---|---|
| `_triton.py` | UU | upstream. Upstream already makes proton optional (`try: import tokenspeed_triton.profiler ... except ImportError`), so 57312157 is in upstream. |
| `benchmark/generators/__init__.py` | UU (false rename from `ops/moe/triton/__init__.py`) | upstream |
| `ops/layernorm/_triton/__init__.py` | UU (false rename from `ops/moe/deep_gemm/__init__.py`) | upstream |
| `ops/attention/__init__.py` | UU | upstream (now a 140-line facade; ops moved into family packages) |
| `ops/tuning.py` | UU | upstream (fork 1c650173 guarded a missing `set_autotune_process_group`; upstream pins flashinfer 0.7.2 and restructured the function around `_autotuner`) |
| `ops/gemm/__init__.py` | UU | upstream (dropped da0619da, see below) |
| `selection.py` | UU | upstream (dropped da0619da env memo and 4465699a `[KERNEL]` log; upstream has `TOKENSPEED_KERNEL_VERBOSE` selection logging, selection.py:460-500) |
| `test/test_selection.py` | clean merge | reset to upstream (its fork hunk only served da0619da) |
| `ops/attention/gdn/_triton/chunk_delta_h.py` | UU | upstream |
| `ops/attention/gdn/_triton/chunk.py`, `chunk_o.py` | clean merge | reset to upstream (h_dtype dropped) |
| `test/ops/test_attention_gdn.py` | clean merge | reset to upstream (h_dtype tests dropped) |
| `ops/attention/gdn/flashinfer.py` | UU | upstream (sm_120 flashinfer GDN raw-kernel path dropped, see follow-ups) |
| `ops/attention/mha/_triton/decode.py` | UU | upstream + machine-sized flash-decoding split count (from 4eeb86da); registration/env knobs dropped |
| `ops/attention/mha/_triton/prefill.py` | UU | upstream + out-of-range query-block early return (b93a8a8e) + measured shared-memory tile gate and sm_120 wide-head tiles (73b27362, 4eeb86da); TS_TRITON_PREFILL_* env knobs dropped |
| `ops/attention/mha/triton.py` | not conflicted | + `_Q_OVER_FP8_KV_SIGNATURES` on triton extend and decode (996b05f4 re-expressed on upstream's registration file) |
| `ops/attention/mha/cuda.py` | not conflicted | + `head_dim` trait `range(8, 257, 8)` on the three fa3 kernels (from fork flash_attn/__init__.py, 974e73e8/ff9a176b) |
| `ops/attention/mha/flashinfer.py` | not conflicted | + `max_arch_version=10.3` on trtllm extend; + `page_size {16,32,64,128}` and `head_dim range(16,257,16)` traits on trtllm decode (48a80ebb, 33d7576b) |
| `ops/communication/triton.py` | UU | upstream + NVLS multicast_ptr==0 guard in `nvidia_create_rsag_state` (from a8882b72) |
| `ops/kvcache/host_transfer.py` | UU | upstream (b5831cf5 debug-only fault attributor dropped) |
| `ops/moe/flashinfer.py` | not conflicted | + cutlass unquant self-routing `topk_weights.to(torch.float32)` (4119f79c re-applied; upstream still had `.to(x.dtype)` at flashinfer.py:188) |
| `numerics/reference/gemm.py` | clean merge | reset to upstream (64e6b8a9 TOKENSPEED_MM_SHAPES diagnostic dropped) |
| `thirdparty/cuda/rope.py` | UU | upstream (936b835f per-call Python micro-opt dropped) |
| `thirdparty/cuda/csrc/dsv4_topk.cu` | clean merge (rename) | reset to upstream (f36960ca prefill half dropped together with its decode half, see follow-ups) |
| `test/ops/test_dsa_topk.py` | UU | upstream |
| `platform.py` | clean merge | kept fork fix 228886f0: `max_shared_memory_per_sm` reads `shared_memory_per_block_optin` (upstream reads a non-existent torch attribute and always gets 0; no upstream non-test consumer, so only the prefill tile gate sees it) |
| `ops/quantization/cuda.py` | clean merge | kept fork a1483ec5 optional `gptq_marlin_gemm` import (error_fn fallback; consumer is the python w4a16 path) |
| `test/ops/attention/test_triton_fp8_kv.py` | fork-added | kept; import -> `ops.attention.mha.triton`, kernel name -> `triton_mha_decode_with_kvcache`, traits renamed |
| `test/ops/attention/test_xqa_head_dim_trait.py` | fork-added | kept unchanged |
| `test/ops/moe/test_cutlass_unquant_scales.py` | fork-added | kept; retargeted at `ops.moe.flashinfer`, `importorskip(exc_type=ImportError)` |

### UD / AU and fork-only files to delete (`git rm`)

All left on disk. None shadows an upstream module: the `ops/moe/flashinfer/`,
`ops/moe/deep_gemm/`, `ops/attention/triton/` dirs have no `__init__.py`, so
upstream's `flashinfer.py`/`deep_gemm.py`/`triton.py` modules win; `load_builtin_kernels`
imports explicitly and never reaches `ops/attention/{flash_attn,flashinfer}`.

- `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/flash_attn/__init__.py` (UD) - fa3 head_dim cap ported to `mha/cuda.py`
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/flashinfer/__init__.py` (UD) - trtllm fixes ported to `mha/flashinfer.py`; FA2 ragged/paged extend dropped (follow-up)
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/deepseek_v4.py` (UD) - image-span SWA dropped (follow-up)
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/gated_delta_rule.py` (UD) - only h_dtype
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/_triton/bf16_decode.py` (AU) - follow-up (plugin)
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/deep_gemm/contiguous_fp8.py` (fork-added) - follow-up (plugin)
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer/cutlass_unquant.py` (UD) - scales fix ported; split-decode hook dropped
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/flashinfer/trtllm_fp8.py` (UD) - superseded (upstream converts canonical scales with `.t()`, flashinfer.py:956-969)
- `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/deepseek_v4_persistent_topk.cuh` (UD) - upstream rewrote it as `dsv4_persistent_topk.cuh`; follow-up
- `tokenspeed-kernel/test/nvidia/thirdparty/test_rope_fused_kv.py` (AU) - upstream has `test/nvidia/ops/test_cuda_rope_fused_kv.py`
- `tokenspeed-kernel/test/ops/moe/test_bf16_split_decode.py`
- `tokenspeed-kernel/test/ops/moe/test_deep_gemm_contiguous.py`
- `tokenspeed-kernel/test/test_deepseek_v4_image_span_swa_gpu.py`
- `tokenspeed-kernel/test/test_extend_split_vo_gate.py`
- `tokenspeed-kernel/test/test_gemma4_extend_head_dim.py`

## Per fork commit (44 commits touching tokenspeed-kernel since e7f65f58)

| commit | decision | evidence |
|---|---|---|
| 57312157 proton optional | in upstream | `_triton.py` `try: import tokenspeed_triton.profiler as proton / except ImportError` |
| b93a8a8e prefill early return | kept (re-applied) | `mha/_triton/prefill.py` `_fwd_kernel`; upstream lacks it |
| 228886f0 real smem budget | kept (clean merge) | `platform.py` |
| 73b27362 prefill tiles by smem | kept (re-applied) | `prefill.py` `_has_large_shared_memory` |
| 48a80ebb trtllm extend/decode on sm_120 | kept (re-applied) | `mha/flashinfer.py` |
| 4119f79c cutlass scales fp32 | kept (re-applied) | `ops/moe/flashinfer.py:188` |
| 996b05f4 triton fp8 KV | kept (re-applied) | `mha/triton.py`; upstream's own fp8 KV is the per-tensor "scaled-fp8" format (mha/__init__.py `_blockscaled_signature_and_scales`), dense fp8 caches still had no triton kernel |
| 4d35899c, bdbb3571, a10d7dc0, 8eb78a11, 07565572 sm_120 flashinfer GDN | dropped | sm_120-only; upstream `gdn/flashinfer.py` gates on hopper/blackwell + CUDA 13. Plugin already ships its own sm_120 fused GDN prefill (`unieinfra_kernel_sm120/gdn.py`). Follow-up below. |
| 936b835f rope wrapper | dropped | micro-opt in upstream file; upstream restructured wrapper |
| da0619da dispatch state | dropped | commit message: "This does not move end-to-end numbers"; upstream rewrote `select_kernel` override resolution (`resolve_kernel_override`) |
| de851d8d, 26d0528d, 29f8eda0, 9c411865 FA2 paged extend / ragged prefill | dropped | lived in deleted `ops/attention/flashinfer/__init__.py`; per UPSTREAMING.md B belongs in `unieinfra/kernel-sm120` (which has `flash_attn2.py`). Follow-up. |
| 33d7576b XQA head_dim | kept (re-applied) | `mha/flashinfer.py` decode traits; `test_xqa_head_dim_trait.py` |
| a90de320 ragged triton bf16 MoE | in upstream | upstream `ops/moe/triton.py` registers `_triton.bf16` |
| a1483ec5 marlin gemm import | kept (clean merge) | `ops/quantization/cuda.py` |
| 2d0fe6cd, 7f89b36e formatting | n/a | |
| 7471e09c, 69e772f2 merges | n/a | |
| 1c650173 autotune symbol | dropped | upstream pins flashinfer 0.7.2 (requirements/cuda.txt) and reworked `set_autotune_process_group` |
| b5831cf5 transfer fault attributor | dropped | debug-only, off by default |
| f162d993 h_dtype | dropped | commit itself retracts the motivating hypothesis ("End to end it is false"); upstream runtime calls `gdn_chunk_prefill(output_h=False)` (state/mamba.py:2165). Follow-up only if the python merge keeps the fp32 mamba checkpoint pool. |
| 4465699a `[KERNEL]` log + bf16 split decode | log dropped (upstream `TOKENSPEED_KERNEL_VERBOSE`); kernel dropped | follow-up (plugin) |
| 5887cec2 fp8 KV on flashinfer paged extend | dropped | the kernel it declared on (fork FA2 paged extend) is gone |
| ec6a0761, 22755d48 DeepGEMM FP8 without DeepEP + split decode hook | dropped | follow-up (plugin); upstream moved deepep_fp8 helpers to `ops/moe/deep_ep.py` |
| db1240e6 checkpoint slot allocator | n/a for kernel | (only scheduler/python files) |
| 7ce0f677 NVFP4 MoE mapping + sm_120 prefill tiles | tiles kept via prefill.py small-smem branch | |
| 4eeb86da gemma wide-head tiles + diagnostics | tiles + KV split kept, env-var diagnostics dropped | |
| 64e6b8a9 GEMM shape histogram | dropped | diagnostic |
| ff9a176b gemma-4 global layers to flashinfer, split VO | fa3 head_dim cap kept; flashinfer split-VO dropped with the FA2 kernel | gemma-4 512-wide layers now fall through to Triton on H200 |
| 974e73e8 five H200 models / gemma head-dim | fa3 head_dim cap kept | `mha/cuda.py` |
| a8882b72 vision without placeholder (rsag guard) | guard kept | `ops/communication/triton.py` |
| 481dacee image-span bidirectional attention | dropped | upstream deleted `combine_{topk,dense}_swa_indices`; follow-up |
| a3873cb6, f36960ca deterministic top-k | dropped | upstream rewrote persistent top-k (`dsv4_persistent_topk.cuh`, 1261 lines, tie set via `handle_tie` value-desc/idx-asc) and renamed `dsv4_topk.cu`; follow-up |
| a5531268 agent-slo bench | n/a for kernel | |

## unieinfra/kernel-sm120 adjustments (upstream API moves)

- `gdn.py`: imports moved to `tokenspeed_kernel.ops.attention.gdn` (`GdnCheckpointLayout`, `GdnChunkPrefillResult`), `...gdn.triton.triton_gdn_chunk_prefill`, `...gdn._triton.*`; trait `head_v_dim` -> `value_head_dim` (old name was silently ignored -> would match V!=128).
- `__init__.py`, `flash_attn2.py`: traits `support_sinks`/`support_logit_cap` -> `sinks`/`logit_cap` (old names are ignored by `spec_matches_traits`, so the PLUGIN-priority kernels would have won sinks/logit-cap requests they cannot serve); FA2 prefill declares `skip_softmax {False}`.
- `flash_attn2.py`, `flashinfer_mha.py`: extend/decode accept `enable_pdl` (upstream dispatchers now pass it).
- Not re-verified (no sm_120 here): `gemm.py`, `moe.py`, `quant.py` call contracts against upstream's dispatchers.

## Tests run (GPU 4, H200)

Image note: `ts-env:gpu` has flashinfer 0.6.17 and an old flash_mla, so upstream's
`ops/moe/flashinfer.py` (needs `trtllm_fp8_per_channel_scale_moe`) and
`ops/attention/dsv41/flash_mla.py` do not import there. Tests were run in
`nctu6/tokenspeed:7ce0d706-20261007` (flashinfer 0.7.1rc2), where every touched
module imports (script: scratchpad/v41/kimports.py, 0 failures).

Command form: `UI_IMAGE=nctu6/tokenspeed:7ce0d706-20261007 GPUS=4 ui.sh python3 -m pytest -q <tests>`

- test/ops/attention/test_triton_fp8_kv.py, test/ops/attention/test_xqa_head_dim_trait.py,
  test/test_selection.py, test/ops/test_attention_gdn.py,
  test/ops/test_attention_mha_prefill_skip_softmax.py,
  test/nvidia/ops/test_fp8_scaled_kv_attention.py, test/ops/moe/test_cutlass_unquant_scales.py:
  228 passed, 27 skipped, 4 failed (813 s). The 4 failures were all
  test_cutlass_unquant_scales: upstream's apply now builds a real cutlass
  workspace before the stubbed kernel call; the test now stubs
  `cutlass_unquant_moe_workspace` too -> rerun 4 passed.
- test/ops/test_attention.py -k triton (covers decode KV-split change, prefill
  early return and tile gate): 23 passed, 28 skipped (dtype variants with no
  triton registration, "solution 'triton' is not registered" from conftest).
- Not runnable here: unieinfra/kernel-sm120 tests (sm_120 only; the plugin's
  register() returns early on sm_90). Plugin modules import cleanly.
- No conflict markers left under tokenspeed-kernel/ or unieinfra/kernel-sm120/.

## Follow-ups

1. Deterministic DSv4 top-k (a3873cb6 + f36960ca): re-port onto upstream's
   rewritten `thirdparty/cuda/csrc/dsv4_persistent_topk.cuh` (the threshold-bin
   ties in `handle_tie` are already value-desc/idx-asc, but the above-threshold
   prefix and the `num_ties <= topk` branch still emit in atomicAdd order:
   lines ~361, 509-514, 703-707) and `dsv4_topk.cu` prefill, together with the
   python-side decode reorder; needs a wheel rebuild (AOT kernel group
   `dsv4_attention` in setup.py) and the determinism test from a3873cb6.
   Check first whether upstream's `--batch-invariant`/rl-bitwise numerics
   mode already gives the required determinism.
2. DSv4 image-span bidirectional SWA (481dacee): upstream removed
   `deepseek_v4_combine_{topk,dense}_swa_indices`; re-implement on whatever
   builds DSv4/DSv4.1 sparse-prefill SWA indices now (ops/attention/dsv4/,
   dsv41/), coordinated with the python merge of the vision path.
3. NVLS guard: upstream `TritonRSAGBackend._get_or_create`
   (python/tokenspeed/runtime/distributed/comm_backend/triton_rsag.py) does not
   catch exceptions, so on a host without multicast the new RuntimeError stops
   startup instead of falling back to NCCL. Python side should catch it and
   use `self._fallback` (python agent's scope).
4. Plugin candidates (UPSTREAMING.md section B): FA2 ragged prefill / paged
   extend with plan-once and split-VO for head_dim 512 (fork
   ops/attention/flashinfer/__init__.py), bf16 decode split-K MoE
   (bf16_decode.py), DeepGEMM FP8 MoE without DeepEP (contiguous_fp8.py; now
   import helpers from ops/moe/deep_ep.py and ops/moe/_deep_ep/permute.py),
   sm_120 flashinfer GDN raw kernel. All are recoverable from HEAD (dsv41).
5. h_dtype for gdn_chunk_prefill: only if the python merge keeps the fp32 mamba
   checkpoint pool (f162d993); otherwise nothing.
6. Plugin gemm/moe/quant registrations: re-verify call signatures/traits
   against upstream dispatchers on an sm_120 box.
7. Runtime image: `ts-env:gpu` (flashinfer 0.6.17, old flash_mla) cannot import
   upstream `ops/moe/flashinfer.py` or `ops/attention/dsv41/flash_mla.py`;
   upstream pins flashinfer 0.7.2. Rebuild the image before serving tests.
8. Upstream PR candidates from this merge: triton fp8-KV signatures, prefill
   early return, smem tile gate + platform smem fix, fa3 head_dim cap, trtllm
   extend max arch, XQA page_size/head_dim traits, cutlass scales fp32,
   NVLS guard.
