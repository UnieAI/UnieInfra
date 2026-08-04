<!-- English canonical. Chinese mirror TODO: sliding-window-kv.zh.md -->
# Design: Sliding-Window KV Exploitation — UnieInfra vs vLLM vs SGLang

Status: **Study (read-only code audit, no benchmark run)** · Date: 2026-07-25

> **Method note.** This is a pure code study. Nothing here was measured on
> device — a decisive benchmark was occupying the GPUs. Every structural claim
> carries a `file:line`. Every number is arithmetic from
> `/workspace/models/google/gemma-4-31B-it/config.json`, labelled as such.
> Claims that are *inferred* rather than *verified by execution* are marked.

---

## Verdict (read this first)

**This is a competitive GAP for us, and a larger one than it first looks.**

The tokenspeed base does contain a genuine, well-engineered sliding-window
(SWA) KV implementation — arguably the equal of vLLM's hybrid allocator in
design. It is **not reachable in any configuration we currently ship or
measure**, for two independent reasons that compound:

1. **The whole SWA machinery is compiled out of the scheduler we run.** All of
   it — the per-group page budgets, the `SwaManager` block-slide, the
   SWA-aware admission credit — is gated behind
   `scheduler_ext_flat_kvcache()`, which reads the `TOKENSPEED_FLAT_KVCACHE`
   build flag (`python/tokenspeed/runtime/configs/paged_cache_spec.py:42-53`).
   This checkout's compiled extension is built with that flag **OFF**
   (`tokenspeed-scheduler/build/CMakeCache.txt`: `TOKENSPEED_FLAT_KVCACHE:BOOL=OFF`;
   corroborated by `docs/experiments-log.md:36` and
   `docs/design/repo-reorg-and-rename.md:232-233`). On the radix build,
   `publish_paged_cache_groups` returns `None`
   (`paged_cache_spec.py:497,530-531`), the KV pool publishes zero cache groups
   (`runtime/layers/attention/kv_cache/mha.py:173-186`), the scheduler config
   receives an empty `paged_cache_groups` list
   (`runtime/engine/scheduler_utils.py:140-145`), and the C++ side therefore
   never constructs the paged-cache adjunct at all
   (`tokenspeed-scheduler/csrc/scheduler/scheduler.cpp:112-116`). **Every
   benchmark number we have ever published was produced by an engine that
   allocates full-length KV for sliding layers.**

2. **Gemma 4 would not reach it even on a flat build.** Gemma 4 ships a wrapped
   multimodal config, and we deliberately keep the wrapper rather than
   unwrapping to the text config
   (`python/tokenspeed/runtime/utils/hf_transformers_utils.py:302-307`). But the
   KV-pool plumbing reads `layer_types` and `sliding_window` off that *wrapper*
   (`runtime/layers/attention/configs/mha.py:85-99`), where neither attribute
   exists — transformers' `PretrainedConfig` has no attribute delegation to
   sub-configs (verified against transformers 5.14.1: a Gemma-3-shaped wrapped
   config returns `MISSING` for both, while `config.text_config` has them). So
   the pool is told the model is 60 layers of uniform full attention. The
   `gemma.py` model file does everything right — it sets
   `sliding_window_size` and `group_id=layer_type` per `PagedAttention`
   (`runtime/models/gemma.py:300-309`) — and the guard that exists to catch
   exactly this mismatch short-circuits when the pool published ≤1 group
   (`runtime/layers/paged_attention.py:108-110`). **The mis-plumbing is
   silent: no crash, no warning, no log line.**

Both vLLM and SGLang exploit sliding windows for Gemma 4 *by name, today*.
vLLM reads `config.layer_types[layer_idx]` in its own `gemma4.py`
(`third_party/vllm/vllm/model_executor/models/gemma4.py:439-443`) and emits a
`SlidingWindowSpec` whose pool sizing is window-bounded
(`third_party/vllm/vllm/v1/kv_cache_interface.py:567-598`). SGLang lists
`Gemma4ForConditionalGeneration` in its hybrid-SWA architecture allowlist
(`third_party/sglang/python/sglang/srt/configs/model_config.py:1915,1927-1929`)
and gives SWA layers a physically separate, smaller pool
(`third_party/sglang/python/sglang/srt/mem_cache/swa_memory_pool.py:19-74`).

The mitigating facts: this is **not a correctness bug** (the kernel-level
window is applied per layer regardless — `runtime/layers/attention/backends/mha.py:799`),
and the design work is largely **already done and in-tree**. The gap is
integration and build configuration, not architecture. But "we have the code
and never run it" is exactly the failure mode that makes a base look better on
paper than it serves in practice, and it is the second instance of that pattern
this month (E24 found the same about the flat FA2 path).

---

## 1. What sliding-window exploitation means, and what it is worth

An SWA layer's attention at position *t* reads only keys in
`[t - W + 1, t]`. KV for tokens below that bound is provably unreachable and
can be freed. A serving engine "exploits" the window when its **memory
manager** — not just its kernel — knows this, so per-sequence KV for those
layers **plateaus at W** instead of growing with sequence length.

This is a memory-manager problem, not a kernel problem. Every engine in this
comparison, including ours, already gets the *compute* right by passing
`window_left` to the attention kernel. The differentiator is whether the
allocator, the admission gate, and the prefix cache all agree that those blocks
are dead.

### Gemma-4-31B: the arithmetic

From `/workspace/models/google/gemma-4-31B-it/config.json` (`text_config`):
60 layers, `layer_types` in a 5:1 sliding:full pattern (50 sliding, 10 full),
`sliding_window: 1024`, `max_position_embeddings: 262144`.

| | sliding layer (×50) | full layer (×10) |
|---|---|---|
| kv heads | `num_key_value_heads: 16` | `num_global_key_value_heads: 4` |
| head dim | `head_dim: 256` | `global_head_dim: 512` |
| K+V per token (bf16) | 2·16·256·2 B = **16 KiB** | 2·4·512·2 B = **8 KiB** |

Note the shape: the *sliding* layers are the expensive ones (16 KiB/token vs
8 KiB), and there are five times as many of them. Gemma 4 is architected on the
assumption that the serving engine bounds them. **91% of this model's
per-token KV bytes live in layers that a window-aware engine caps at 1024
tokens.**

Per-sequence KV, TP=1, bf16, ignoring page rounding:

| context | **not** window-bounded | window-bounded | ratio |
|---|---|---|---|
| 8,192 | 880 KiB/tok → **6.88 GiB** | 0.78 GiB (sliding, constant) + 0.63 GiB (full) = **1.41 GiB** | **4.9×** |
| 32,768 | **27.5 GiB** | 0.78 + 2.50 = **3.28 GiB** | **8.4×** |
| 131,072 | **110 GiB** | 0.78 + 10.0 = **10.8 GiB** | **10.2×** |

The bounded sliding term is constant at `50 · 16 KiB · 1024 = 0.78 GiB` for any
context length; only the 10 full layers scale.

**What this means for concurrency** (arithmetic, *not measured*): on our box,
4× RTX PRO 6000 96 GB at TP=4, Gemma-4-31B bf16 weights ≈ 62 GB → ~15.5 GB/rank,
leaving roughly 70 GiB/rank for KV. TP shards KV heads by 4, so per-rank
per-sequence KV is one quarter of the table above. At 32k context that is
~6.9 GiB/seq unbounded versus ~0.82 GiB/seq bounded — i.e. on the order of
**~10 concurrent sequences versus ~85**. This is the number that matters for
the strategic decision; it is an order-of-magnitude concurrency difference at
long context, not a percentage.

**One extra loss on top, specific to us.** Because the pool is told the model is
uniform, it sizes every layer at the sliding geometry
(`model_config.head_dim = 256`, `num_key_value_heads = 16` — read from
`hf_text_config` at `runtime/configs/model_config.py:463-466,494-495`), so the
10 full layers get a 16 KiB slot to hold 8 KiB of KV. Real per-token cost is
960 KiB rather than the model's true 880 KiB — a further ~9% waste. *(Inferred
from the config plumbing; whether the backend's per-layer view handles the
head_dim-512 reinterpretation correctly in that slot is a separate open
question, flagged in `docs/support-matrix.md` as "E27's latent per-layer-head_dim
KV-pool question is open".)*

---

## 2. Three-way comparison

| | **UnieInfra** (radix build = what we ship/measure) | **UnieInfra** (flat build, `TOKENSPEED_FLAT_KVCACHE=ON`) | **vLLM** | **SGLang** |
|---|---|---|---|---|
| **(a) Allocates less for sliding layers** | ❌ No. Zero cache groups published; one uniform full-history page space for all layers. `paged_cache_spec.py:530-531`, `mha.py:184-186` | ✅ Yes — via two mechanisms: window-bounded per-group page budget (`paged_cache_spec.py:316-338`) and hybrid slab aliasing so sliding+full layers share physical storage (`mha.py:285-297`) | ✅ Yes. `SlidingWindowSpec.max_memory_usage_bytes` = `(cdiv(min(W-1+in_flight, max_len), block)+1) · page_size` vs full's `cdiv(max_len, block) · page_size`. `kv_cache_interface.py:567-598` vs `:258-263` | ✅ Yes. Physically separate `swa_kv_pool` / `full_kv_pool` with different sizes, `swa_memory_pool.py:60-74`; ratio from `--swa-full-tokens-ratio` (default 0.8), `server_args.py:846-857` |
| **(b) Frees out-of-window blocks during decode** | ❌ No. `PagedCacheGroupTable::ReleaseSkipped` exists (`csrc/resource/allocator/paged_cache_group.cpp:386-420`) but its only caller is the paged-cache adjunct (`hybrid_prefix_cache.cpp:1131-1134`), which is never constructed | ✅ Yes, two paths: `SwaManager::ReclaimExpired` nulls fully-slid-out blocks (`csrc/cache/swa_manager.h:75-88`) and `ReleaseSkipped` drops front pages via RAII | ✅ Yes. `remove_skipped_blocks` → `_remove_blocks_in_range` → `block_pool.free_blocks` + null-block substitution (`single_type_kv_cache_manager.py:598-635`, `:571-596`), called from `allocate_slots` every step (`kv_cache_manager.py:491-504`) | ✅ Yes. `free_swa_out_of_window_slots` (`mem_cache/common.py:47-95`) → `free_swa` frees SWA slots only (`allocator/swa.py:347-359`), driven every `SGLANG_SWA_EVICTION_INTERVAL=128` decode steps (`schedule_batch.py:3090-3155`) |
| **(c) Accounted for in admission / batching** | ❌ No, and in **three** places. C++ decode admission counts one global page space (`csrc/scheduler/operations/forward.cpp:455-458`); `ActiveKvPages` dedups radix device pages with no per-group notion (`scheduler.cpp:265-279`); `derive_adaptive_max_batch` sizes the cap from `context_len / reserve_divisor` with no window discount (`runtime/engine/event_loop.py:113-168`, called `:329-346`) | ✅ Yes. `FlatSlideCredit` credits reclaimable SWA blocks into the admission decision (`forward.cpp:112-121`), consumed at `forward.cpp:186,203` by `flatAdmitPrefillChunk`/`flatAdmitDecode` (`scheduler.h:143-146`) | ✅ Yes, *implicitly but soundly*. The scheduler does no SWA math itself; `get_num_blocks_to_allocate` subtracts `num_skipped_blocks` (`single_type_kv_cache_manager.py:193-209`) and the full-sequence gate clamps at `max_admission_blocks_per_request` (`:169-182`), the *same function* used for startup sizing | ✅ Yes, **explicitly**. A second independent budget `rem_swa_tokens` (`schedule_policy.py:589-595`) with per-request SWA charge `max(alloc - W, 0) + (W + page_size)` (`:622-649`); `evict_from_tree_cache` drives the two pools separately (`mem_cache/common.py:110-121`) |
| **(d) Prefix reuse handled correctly** | ⚠️ Correct but unexploited. Since nothing is ever freed, a cached prefix page is always physically live, so full-history matching is *safe*; the window is applied by the kernel mask. No wasted-memory-free reuse, but no wrong answers | ✅ Yes. `SwaManager::Probe` scans right-to-left for a run long enough to resume the window (`swa_manager.h:47-72`, `pagesNeededToResume()` at `:109`); the coordinator re-matches window groups until the bound converges (`kv_cache_coordinator.cpp:64-139`) | ✅ Yes. `find_longest_cache_hit` scans right-to-left for `ceil((W-1)/block)` contiguous blocks, returns a **null-padded** list (`single_type_kv_cache_manager.py:870-967`); hybrid groups reconcile by fixed-point loop (`kv_cache_coordinator.py:685-817`); unreachable blocks are kept out of the hash map (`reachable_block_mask`, `:969-1029`); cascade attention disabled for SWA (`:1059-1066`) | ✅ Yes. Tombstones: a match is accepted only if ≥ W live tokens follow the last tombstone (`swa_radix_cache.py:888-941`); partial recovery / node-split on insert (`:1114-1256`); `PureSWARadixCache` for all-SWA models needs no tombstones (`pure_swa_radix_cache.py:24-30`) |

### Notes on the comparison

- **The two references are not equivalent in approach.** vLLM keeps one block
  pool and bounds the *per-request block count*; SGLang splits the *pool
  itself* into two sized sub-pools. vLLM's design is self-sizing (the same
  function bounds startup memory and runtime admission —
  `kv_cache_interface.py:570-579` states this explicitly as an invariant);
  SGLang's needs a tuned ratio per architecture (0.8 default, but 0.1 for
  DeepSeek-V4 and Inkling). UnieInfra's flat design is closer to vLLM's.
- **Both references have a documented "off" switch, and both warn loudly when
  they take it.** vLLM's `--disable-hybrid-kv-cache-manager` logs *"we do not
  enable any optimizations for saving KV cache memory (e.g., dropping the KV
  cache outside the sliding window)"* (`kv_cache_utils.py:1547-1568`); SGLang's
  `--disable-hybrid-swa-memory` (`server_args.py:859-863`) is force-set for
  Gemma-2/3/3n with a warning (`arg_groups/overrides.py:551-564`) — notably
  **not** for Gemma 4. Our equivalent state is the silent default.
- **SGLang publishes no measured benefit** for this feature anywhere in-tree —
  no benchmark, no changelog number. If we close the gap, the quantified
  comparison is unclaimed territory and worth owning.

---

## 3. What our flat path actually does (so the gap is sized honestly)

This section exists because the gap is *not* "we need to design SWA support".
Most of it is written. Concretely, on a `TOKENSPEED_FLAT_KVCACHE=ON` build:

- **Layer → group mapping.** `layer_group_ids` / `group_specs_from_layer_types`
  collapse layers into one group per distinct `(retention, window)`, vLLM-style
  (`paged_cache_spec.py:426-495`).
- **Window-bounded page budget.** For a sliding group,
  `compute_paged_cache_group_page_counts` charges
  `max_live_requests · ceil(min(W-1, max_context_len) / page) + scheduled_pages
  + …` instead of `ceil(max_total_tokens / page)`
  (`paged_cache_spec.py:316-338` vs `:307-315`). The window bound is real.
- **Bounded CUDA-graph block-table width.**
  `compute_max_logical_pages_for_capture` returns
  `ceil(min(W, ctx) + horizon / page) + 1` for sliding groups
  (`paged_cache_spec.py:556-612`, sliding branch at `:598-608`) — notably *tighter* than vLLM, which leaves
  the worker block-table row at full `max_model_len`
  (`kv_cache_interface.py:220-223`).
- **Physical storage sharing.** The hybrid slab layout makes the i-th layer of
  every group alias one slab tensor (`mha.py:285-297`), with the KV memory
  profile charged at the slab count rather than the layer count
  (`registry.py:65-73`). For GPT-OSS's 24:24 split that is a clean 2×.
- **Block-level slide + null holes.** `SwaManager::ReclaimExpired` and
  `BlocksReclaimableAt` (`csrc/cache/swa_manager.h:75-104`) are a faithful
  analogue of vLLM's `remove_skipped_blocks` / null-block scheme.
- **Reference integration exists.** Inkling (55 sliding + 11 full) is fully
  wired: its config exposes `paged_cache_layer_types` *and* `sliding_window` as
  forwarding properties on the multimodal wrapper
  (`runtime/configs/inkling_config.py:365-366,546-551`) — precisely what Gemma 4
  is missing — and splits sliding layers round-robin into equal-count
  sub-groups `sliding_attention_<k>` so every slab is fully bound
  (`inkling_config.py:375-379`, consumed at `runtime/models/inkling.py:637-638`).

So the honest framing is: **the base's differentiator claim is architecturally
defensible and integration-wise unproven.** One model family (Inkling) uses the
full machinery. Everything else either does not reach it or is not built with it.

---

## 4. Closing the gap: concrete work, sized

Three independent items. Only the first is a real engineering question.

**(1) Make the flat build the default — or make SWA work on radix. [LARGE, and
the actual decision]**

This is the load-bearing item and it is not configuration. Flipping
`TOKENSPEED_FLAT_KVCACHE=ON` locally is one CMake flag (the Docker images
already do it: `docker/Dockerfile.nvidia:35`, `docker/Dockerfile.amd:38`), but
the flat path is a **different FSM** — `#if TOKENSPEED_FLAT_KVCACHE` appears
across `csrc/fsm/forward_events.h`, `csrc/fsm/forward_states.h`,
`csrc/scheduler/scheduler.{h,cpp}`, `csrc/scheduler/outside_event_handler.cpp`
— and `docs/experiments-log.md:36` (E24) records that it is **"code-ready,
e2e-unvalidated"**: never exercised end-to-end because doing so needs a
flat-built scheduler *and* a hybrid model at the same time, which nobody has
run. Startup guards will reject several combinations outright
(`paged_cache_spec.py:169-233`: non-flat-capable backends, DFLASH speculative
decode, state-only pools). Realistic scope: validate flat + Gemma 4 or flat +
GPT-OSS end-to-end, fix what that surfaces, then decide whether flat becomes
the default build. **This is a multi-day de-risking exercise, not a flag flip,
and its outcome is a genuine input to the keep-vs-rebase decision** — if the
flat path proves cheap to validate, the base's claim holds; if it turns into a
month, that is the strongest argument yet for re-basing.

**(2) Unblock Gemma 4's config plumbing. [SMALL — hours]**

Two options, both mechanical:
- Mirror Inkling: add `paged_cache_layer_types` (or `layer_types`) and
  `sliding_window` forwarding properties to the Gemma 4 wrapper, as
  `inkling_config.py:365-378,546-551` does. Requires a tokenspeed-side config
  class for Gemma 4, which does not exist today.
- Or change `MHAAttnConfig.from_server_args` (`configs/mha.py:85-99`) to fall
  back to `hf_config.get_text_config()` / `model_config.hf_text_config` when the
  wrapper lacks the attributes. Smaller, and fixes every future wrapped
  multimodal config at once. Same for
  `model_runner.sliding_window_size` (`runtime/execution/model_runner.py:96-97`).

Either way, **add a loud startup warning** when a model declares
`PagedAttention(group_id=...)` on more than one distinct group but the pool
published ≤1 group — today `validate_paged_cache_group_ids` returns silently in
exactly that case (`runtime/layers/paged_attention.py:108-110`). That guard
inversion is what let this go unnoticed; it should be a `logger.warning` at
minimum, and it is worth adding **independently of everything else**, because it
is the cheapest possible insurance against the same class of silent regression.

**(3) Consider sliding sub-groups for Gemma 4's 5:1 split. [SMALL, follow-up]**

With plain `sliding_attention` / `full_attention` labels, `hybrid_slab_group_size`
returns `max(50, 10) = 50` slabs for 60 layers — only 1.2× storage sharing,
because slabs 10–49 are single-layer (`paged_cache_spec.py:100-167`). Splitting
the 50 sliding layers into 5 sub-groups of 10 (`sliding_attention_0..4`), exactly
as Inkling does, would give 10 fully-bound slabs — **6×**. The sub-group label
vocabulary already exists and is already handled by `_retention_for_label`
(`paged_cache_spec.py:78-98`). This is configuration on top of existing
machinery, not an allocator change.

**Also needed regardless: teach `derive_adaptive_max_batch` about windows.**
It currently estimates live KV per request as `context_len / reserve_divisor`
(default 4) with no per-layer notion (`runtime/engine/event_loop.py:113-168`).
For Gemma 4 at 32k that estimate is ~8× too pessimistic for 50 of 60 layers.
This is the *third* single-signal blind spot in that function — after the
original KV-only derivation and the mamba/GDN state-pool clamp added in E31
(`3ad34a3`). The pattern is worth naming: **`derive_adaptive_max_batch` keeps
being wrong because it estimates a physical quantity the allocator already
knows exactly.** The durable fix is to have it ask the pool for a per-request
byte estimate rather than re-deriving one from `context_len`.

---

## 5. Which other model families this affects

| Family | Declares a window? | Reaches our SWA path? |
|---|---|---|
| **GPT-OSS** | Yes — `layer_types` sliding/full, `config.sliding_window` (`runtime/models/gpt_oss.py:204-213,440-443`) | **On a flat build, yes.** Flat config is not wrapped, so `layer_types` is read correctly. This is the model the machinery was built for. On radix: no. |
| **Inkling** | Yes — `local_layer_ids` → sliding sub-groups (`runtime/configs/inkling_config.py:365-380`) | **On a flat build, yes** — the fully-wired reference integration, including per-layer kv-head heterogeneity (`registry.py:429-439`) and extra sliding conv groups (`registry.py:441-508`). |
| **Gemma 4** (`31B-it` dense, `26B-A4B` MoE) | Yes — 5:1, W=1024 | **No, on either build** (§ Verdict item 2). MoE variant is not yet ported (`runtime/models/gemma.py:376-387`, MoE note) and will inherit the same problem. |
| **DeepSeek-V4** | Yes — `swa_window` default 128 (`runtime/models/deepseek_v4.py:3387`); its cache spec has a `sliding_window` retention branch (`runtime/layers/attention/kv_cache/deepseek_v4.py:315`) | Own KV pool (`DeepseekV4TokenToKVPool`) with its own paged-group path; **not audited here** — worth a follow-up, since SGLang tunes `swa_full_tokens_ratio=0.1` specifically for V4 (`third_party/sglang/.../arg_groups/overrides.py:810-812`). |
| **DFLASH** (drafter) | Yes — per-layer window (`runtime/models/dflash.py:127-134,514-548`) | Explicitly rejected on the flat path (`paged_cache_spec.py:208-213`), so no. |
| **Qwen2 / Qwen3 / Qwen3.5 / Qwen3.6** | Config field exists but `use_sliding_window` defaults **False** → `sliding_window = None` (`runtime/configs/qwen3_config.py:191-205`, `qwen2_config.py:73-86`) | N/A — no window in practice. This is why the gap has been invisible: our entire validated model set is window-free. |
| **Llama / Mistral / Mixtral** | `runtime/models/llama.py` has **no** sliding-window plumbing at all — no `layer_types`, no `sliding_window_size` on its `PagedAttention` | No. Mixtral is windowless in practice, but a Mistral-family checkpoint that declares a window would have it silently ignored *including at the kernel level*, since `model_runner.sliding_window_size` would be the only signal and `llama.py` never passes it per layer. **This is a potential correctness issue, not just a memory one** — worth a separate check before claiming Mistral support (`docs/support-matrix.md` currently lists "Mixtral" as code-supported). |

The pattern worth internalising: **sliding windows are the direction the
frontier is moving** (Gemma 3/4, GPT-OSS, Llama 4, Mistral, DeepSeek-V4,
Inkling, MiMo-V2, Step3p5 — SGLang's allowlist alone names 18 architecture
strings plus a MiMo-V2 expansion,
`third_party/sglang/.../configs/model_config.py:1910-1943`). Our validated set
(Qwen3 dense, Qwen3.5/3.6 GDN) happens to contain none of them, which is
precisely why this went unnoticed. It will not stay unnoticed as the model
coverage broadens.

---

## 6. Open questions this study could not settle

- **Does the flat build actually run?** Never validated end-to-end
  (`docs/experiments-log.md:36`). Until someone boots a flat-built scheduler on
  GPT-OSS or Gemma 4, every "✅" in the flat column of §2 is *code-verified,
  execution-unverified*.
- **Does Gemma 4's per-layer head_dim (256 vs 512) fit the pool's uniform
  slot?** The pool allocates one `(head_num, head_dim)` geometry
  (`mha.py:266-271`). Inkling solves the analogous per-layer *head-count*
  problem via `layer_kv_head_counts` + `group_page_sizes`
  (`registry.py:429-439`), but Gemma 4 varies head *dim*. Flagged as open in
  `docs/support-matrix.md` since E27; still open.
- **DeepSeek-V4's own SWA path** was out of scope and is unaudited.
- **No measurement.** The concurrency figures in §1 are arithmetic. The honest
  next step, once GPUs free up, is: run Gemma-4-31B on vLLM at 8k/32k, sweep
  concurrency to the OOM wall, and record the number the window-aware allocator
  actually buys. That single measurement is more decision-relevant than any
  further code reading.

---

## References

**UnieInfra / tokenspeed**
`python/tokenspeed/runtime/configs/paged_cache_spec.py` ·
`python/tokenspeed/runtime/layers/attention/kv_cache/mha.py` ·
`python/tokenspeed/runtime/layers/attention/configs/mha.py` ·
`python/tokenspeed/runtime/layers/attention/registry.py` ·
`python/tokenspeed/runtime/layers/paged_attention.py` ·
`python/tokenspeed/runtime/engine/{event_loop,scheduler_utils}.py` ·
`python/tokenspeed/runtime/models/{gemma,gpt_oss,inkling}.py` ·
`python/tokenspeed/runtime/utils/hf_transformers_utils.py` ·
`tokenspeed-scheduler/csrc/cache/{swa_manager.h,kv_cache_coordinator.cpp,forward_cache_ops.h}` ·
`tokenspeed-scheduler/csrc/resource/allocator/paged_cache_group.cpp` ·
`tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp` ·
`tokenspeed-scheduler/csrc/scheduler/{scheduler.cpp,scheduler.h,operations/forward.cpp}`

**vLLM** (`third_party/vllm`, git `d02df748bf9e`)
`vllm/v1/kv_cache_interface.py` · `vllm/v1/core/{kv_cache_manager,kv_cache_coordinator,kv_cache_utils,block_pool,single_type_kv_cache_manager}.py` ·
`vllm/v1/core/sched/scheduler.py` · `vllm/model_executor/models/gemma4.py` · `vllm/config/vllm.py`

**SGLang** (`third_party/sglang`, git `a690e5e0b3`)
`python/sglang/srt/mem_cache/{swa_memory_pool,swa_radix_cache,pure_swa_radix_cache,common,registry}.py` ·
`python/sglang/srt/mem_cache/allocator/swa.py` · `python/sglang/srt/model_executor/pool_configurator.py` ·
`python/sglang/srt/managers/{schedule_policy,schedule_batch,scheduler}.py` ·
`python/sglang/srt/configs/model_config.py` · `python/sglang/srt/arg_groups/overrides.py`

**Related in-repo docs**
`docs/design/big-model-serving.md:819` (flat scheduler compiled out) ·
`docs/design/repo-reorg-and-rename.md:232-233` · `docs/experiments-log.md:36` (E24) ·
`docs/support-matrix.md` (Gemma 4 row)
