<!-- English canonical. docs/design/* are English-only in this repo (no Chinese mirror). -->
# Datacenter Absorption: What of TokenSpeed's Large-Scale Tier Is Worth Owning in `unieinfra-vllm`

Status: **Study / read-only recon complete, no code written** · Date: 2026-07-25 · Owner: royshih

## Verdict up front

**The datacenter tier is the *weakest* place to spend porting budget on the TokenSpeed
base, not the strongest.** Going in, the working assumption was the opposite: upstream
built that layer for sm_90a/sm_100a, ships production recipes for DeepSeek-V4 on B200,
and advertises beating TRT-LLM. That claim is about **kernels**. The *machinery* around
the kernels — expert-parallel load balancing, prefill/decode disaggregation, hierarchical
KV, DP-attention, parallelism configuration — is, component by component, **behind vLLM
in every case but one small config knob and two techniques**.

Concretely, on the six components requested:

| # | Component | Verdict |
|---|---|---|
| 1 | Parallelism config: placement→comm compiler | **NOT WORTH IT** (1 consumer, 0 tests) |
| 1b | Parallelism config: per-layer-family `Mapping` | **ABSORB (small, architecture)** — one real gap: `dense_tp` independent of `attn_tp` |
| 2 | EPLB | **vLLM ALREADY BETTER** — same DeepSeek algorithm, but online, async-by-default, 4 transports, 9 test files, CI on H100/B200/MI300. Ours is unwired. |
| 3 | P/D disaggregation | **vLLM ALREADY BETTER** on framework (14 connectors vs 1). **ABSORB one technique**: GPU-step-counter-driven layer-wise KV push. |
| 4 | Hierarchical KV / offload | **vLLM ALREADY BETTER** (8,661 LOC, LRU+ARC, 3 L3 tiers, off by default). Absorb the *operational lesson*, not code. |
| 5 | DP-attention & other parallelism we have | **vLLM ALREADY BETTER** — nothing here vLLM lacks. |
| 6 | What vLLM has that we lack | PP, LoRA, DBO, elastic EP, CP (DCP+PCP), 16 KV connectors, encoder disagg, 337 architectures |

The useful half of this document is the second column. **Six of the eight rows say "do not
port."** That is the finding.

---

## Hardware honesty — stated once, applies to every claim below

**We have no datacenter hardware.** The box is 4× RTX PRO 6000 (sm_120), PCIe only, **no
NVLink anywhere** (`nvidia-smi topo -m` recorded in `docs/design/big-model-serving.md:980-990`:
GPU0↔1 and GPU2↔3 are `PIX`, everything else `NODE`). No InfiniBand, no RNIC.

**Nothing in this document was measured.** There is no benchmark here and no comparison of
throughput between the two engines. Every statement is either *verified by reading code at
a cited `file:line`*, or *inferred* and labelled as such. Where a design difference would
only show up on hardware we do not have, the verdict is **BLOCKED**, not a guess.

### Labels used throughout

- **[V]** — verified by reading the cited `file:line` in this session. It means the code
  exists and says what I claim. It does **not** mean it runs or that it is fast.
- **[V-prior]** — established with `file:line` by `docs/design/big-model-serving.md` and
  spot-re-verified here (some line numbers there have since shifted; the paths below are
  the current ones).
- **[I]** — inference from code, arithmetic, or a documented gate. Explicitly flagged.

### Path conventions

- TokenSpeed paths are relative to the repo root: `python/tokenspeed/…`,
  `tokenspeed-kernel/…`, `tokenspeed-scheduler/…`.
- vLLM paths are relative to `third_party/vllm/` at pin `d02df748bf9e` (2026-07-24).
  `unieinfra-vllm/` is a byte-identical vendored copy of that pin except for three files
  (`unieinfra-vllm/PROVENANCE.md:60-72`), so **every vLLM `file:line` below applies
  unchanged inside `unieinfra-vllm/`**. [V]
- One correction to carry forward: this pin is **substantially newer than v0.27**. It has
  a pluggable EPLB policy registry, four EPLB transports, async EPLB on by default,
  elastic expert parallelism, a V2 model runner, and 337 registered architectures. Claims
  about "vLLM as of v0.26/v0.27" in earlier docs should not be reused. [V]

---

## 1. The ranked ABSORB list

Ranked by (value at scale) × (can we validate it now). Everything not on this list is
covered in §3 with a "do not port" verdict and the reason.

| Rank | Item | What it is | Port cost | Validate now? |
|---|---|---|---|---|
| **A1** | **Fail-fast topology negotiation for P/D** | Make heterogeneous prefill/decode layouts a *startup* error with a named reason, the way `UnsupportedPDLayoutError` does (`python/tokenspeed/runtime/pd/transfer_plan.py:26-30`, raised at `:47-63`, `:78-99`, `:333-352`, `:354-389`) | **~2 days** | **Yes** — pure config-time logic, no fabric |
| **A2** | **GPU-step-counter layer-wise KV push** | Transfer thread polls a device-resident step counter mirrored to pinned host memory (`python/tokenspeed/runtime/pd/utils.py:433-477`), so KV for layer *i* ships while layer *i+1* still computes, with **no per-layer host hook and no CUDA sync** (`python/tokenspeed/runtime/pd/mooncake/prefill.py:582-664`, wait at `:601`) | **~1–2 weeks** as an option inside a first-party vLLM connector | **Partially** — the mechanism is testable over a loopback/TCP connector; the *win* is only measurable with a real P/D pair |
| **A3** | **`dense_tp` independent of `attn_tp`** | TokenSpeed can run attention at TP=8 while the dense MLP is replicated (TP=1) — `--dense-tp-size` (`python/tokenspeed/runtime/utils/server_args.py:294-298`), used in the upstream V4-Pro EP recipe (`docs/recipes/models.md:277-279`). vLLM has the per-layer mechanism (`disable_tp` on `LinearBase`, `vllm/model_executor/layers/linear.py:257-259`) but **only as a model-hardcoded choice** (e.g. `vllm/model_executor/models/deepseek_v2.py:683`, `:920`), not a config knob | **~1–2 days** to expose as config; the mechanism already exists | **Yes** at TP2 on GPUs {0,1} |
| **A4** | **The `Mapping` idea (architecture, not code)** | Per-layer-family parallel decomposition as *independent (size, stride) factorizations over one flat world* (`python/tokenspeed/runtime/distributed/mapping.py:51-64`, sub-mappings at `:94-130`, `:133-195`, `:198-266`, `:269-312`) | **Not a port.** Keep as a design reference for when vLLM's `ep_size = dp×pcp×tp` identity becomes binding | n/a |

**That is the whole list.** Four items, three of them small, and the largest (A2) pays off
only on hardware we do not have. Everything else that looked like a datacenter asset turned
out to be either behind vLLM or dead code (§3, §5).

**Deliberate omission:** the C++ FSM scheduler with compile-time KV ownership and `Role`
as a first-class FSM domain (`tokenspeed-scheduler/csrc/scheduler/request.cpp:34`, 22
`Role` branch sites) *is* the one place TokenSpeed is structurally ahead, and P/D-awareness
inside the scheduler rather than beside it is genuinely a datacenter property. It is left
off this list because it is a multi-month project tracked in
`docs/design/unieinfra-architecture.md`, not a datacenter-tier absorption, and because its
value is scheduling, not scale.

---

## 2. Method: what "absorb" has to beat

UnieInfra's absorption model is: take valuable components from any engine as **owned
in-tree source**, never as a library dependency. The bar for a datacenter component is
therefore not "does TokenSpeed have it" but **"is TokenSpeed's version better than the one
already sitting in `unieinfra-vllm/`, by enough to pay the porting cost and the permanent
maintenance cost of a hand-ported divergence."**

For most of this tier the answer is no, and the reason is structural: the components below
were *ported into* TokenSpeed from the same upstream sources vLLM ported from (DeepSeek's
EPLB, Mooncake's transfer engine), but vLLM then wired them, tested them, and put them in
CI, and TokenSpeed did not.

---

## 3. Component-by-component

### 3.1 Parallelism configuration — the placement→comm compiler and per-layer-family `Mapping`

This was flagged as the most distinctive-sounding item. It splits into two very different
answers.

#### 3.1a The placement→comm compiler — **NOT WORTH IT**

**What it does.** `PlacementType{REPLICATE, SHARD, PARTIAL} × ParallelGroup{ATTN_TP,
DENSE_TP, MOE_TP_EP}` (`python/tokenspeed/runtime/models/base/placement.py:36-49`),
attached to each sub-module as a `ModuleSpec` with input/output placements
(`python/tokenspeed/runtime/models/base/module_spec.py:49-57`). `compile_decoder_layer`
(`python/tokenspeed/runtime/models/base/compiler.py:178-240`) walks the layer's module list,
tracks `(hidden, residual)` placement in a small abstract state
(`compiler.py:73-77`), and **inserts the minimal comm ops to reconcile adjacent
placements** — `AllGatherOp`/`AllReduceOp`/`ReduceScatterOp`/`ResidualSliceOp`/
`DeferredReduceOp`/`FusedReduceNormOp` (`compiler.py:318-338`, `:386-428`, `:431-472`). [V]

It is real, it is well-written, and no reference engine derives collectives this way.

**Why it is not worth porting, in three findings.**

1. **It has exactly one consumer.** Grep for `CompiledDecoderLayer` /
   `CompiledMoEDecoderLayer` across `python/` and `test/`: the only model subclassing it is
   `python/tokenspeed/runtime/models/gpt_oss.py:446`. Every other model — DeepSeek-V3,
   DeepSeek-V4, GLM-5, Qwen3.5-MoE, MiniMax-M2/M3, Inkling, LongCat-Flash, Llama, Qwen2/3,
   the EAGLE3 drafter — runs the hand-written `CommManager` path. The base class's own
   docstring says so: *"`BaseDecoderLayer` uses CommManager for communication (the default
   path). `CompiledDecoderLayer` uses the compiler-driven path"*
   (`python/tokenspeed/runtime/models/base/decoder_layer.py:21-25`), and
   `CompiledDecoderLayer` is marked *"(opt-in)"* at `:238`. [V]
2. **It has zero tests.** No file under `test/` references `compile_decoder_layer`,
   `ModuleSpec`, or the placement types. [V]
3. **It does not derive the key decision — it copies it.** The all-reduce-vs-reduce-scatter
   choice is not inferred from placements; the compiler hardcodes a comparison against
   `ATTN_TP` with a comment explaining that it must match the hand-written path:
   *"AR/RSAG decision must compare against ATTN_TP … This matches
   `CommManager.use_all_reduce(is_moe)`"* (`compiler.py:404-408`, again at `:498-500`). So
   the "compiler" is a second expression of the same heuristic, not a replacement for it. [V]

An abstraction serving one model, with no tests, that re-derives the heuristic it was meant
to subsume, is **elegance without payoff as shipped**. The *idea* — declarative placement
with derived collectives — is worth remembering; the 557 lines are not worth carrying.

**What vLLM does instead.** Collectives are literal calls guarded by per-layer booleans:
`RowParallelLinear.forward` does `if self.reduce_results and self.tp_size > 1:
tensor_model_parallel_all_reduce(...)` (`vllm/model_executor/layers/linear.py:1747-1773`);
`ColumnParallelLinear` gathers on `gather_output`
(`linear.py:590-612`); MoE reduction points are maintained by hand with the invariant
written in prose (`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:701-709`,
reductions at `:416-433`, `:435-465`). [V]

Where vLLM *does* rewrite collectives, it does it as **post-hoc FX pattern matching**, not
placement derivation: `vllm/compilation/passes/fusion/sequence_parallelism.py` rewrites
`all_reduce→rms_norm` into `reduce_scatter→rms_norm→all_gather`
(`:153-183`, `:186-259`, FP8 at `:262`, `:304`, NVFP4 at `:375`), and
`vllm/compilation/passes/fusion/collective_fusion.py` fuses GEMM with an adjacent collective
(`GEMMReduceScatterPattern:341`, `AllGatherGEMMPattern:374`, and seven more). [V]
That is a peephole rewriter over concrete collectives — **less principled than ours, and
load-bearing where ours is not.** Honest summary: TokenSpeed's design is better; TokenSpeed's
implementation is unexercised; vLLM's is uglier and shipping.

#### 3.1b Per-layer-family `Mapping` — **ABSORB (small, architecture)**

**What TokenSpeed has.** Four independent sub-mappings over one flat world, each a
`(size, stride)` view (`python/tokenspeed/runtime/distributed/mapping.py:51-64`):
`AttentionLayerMapping(tp, cp, dp)` `:133-195`, `DenseLayerMapping(tp, dp)` `:94-130`,
`MoeLayerMapping(tp, ep, dp)` with a combined `tp_ep_group` `:198-266`, and
`VisionTowerMapping(tp, dp)` whose world is `attn.tp_size` rather than global `:269-312`.
Each factorization independently multiplies to `world_size`
(`_resolve_parallelism_sizes`, `:47`). [V] This *is* exercised — upstream's own recipes use
`--data-parallel-size 4 --enable-expert-parallel` (`docs/recipes/models.md:242-243`) and
`--tensor-parallel-size 8 --enable-expert-parallel --dense-tp-size 1`
(`docs/recipes/models.md:277-279`). [V]

**What vLLM has.** No `Mapping` object, no `expert_parallel_size` field. There is exactly
one hardcoded family split: attention/dense live on `(pp, pcp, tp)` and are replicated
across DP; MoE lives on the **flattened** `dp × pcp × tp` world.
`FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp`
(`vllm/model_executor/layers/fused_moe/config.py:1112-1121`) computes
`flatten_tp_size = dp_size * pcp_size * tp_size`, and `make()` then sets
**`ep_size = tp_size`** (the flattened product) with `tp_size = 1`
(`config.py:1236-1251`). The EP process group is built the same way
(`vllm/distributed/parallel_state.py:1890-1916`). The docstring at `config.py:1153-1202`
enumerates the exact device tables. [V]

**So how much does vLLM actually lack?** Less than it sounds.

- The two patterns that matter for flagship MoE — DP-attention + EP-MoE, and TP-attention +
  EP-MoE — are both expressible in vLLM today. That is the whole DeepSeek deployment shape.
- Both engines **forbid TP and EP inside the same MoE layer**: TokenSpeed raises
  `"Mixed TP and EP is not supported yet."` (`python/tokenspeed/runtime/layers/moe/expert.py:110-111`);
  vLLM sets `tp_size = 1` whenever `use_ep` (`config.py:1238-1244`). Parity, not an edge. [V]
- The genuine gap is **`--dense-tp-size` independent of `--attn-tp-size`**: replicating the
  dense MLP to avoid an all-reduce while attention stays TP-sharded. vLLM has the
  *mechanism* — `LinearBase` accepts `disable_tp` / `tp_rank` / `tp_size` overrides
  (`vllm/model_executor/layers/linear.py:257-259`, resolved `:282-311`) and models use it
  (`deepseek_v2.py:683`, `:920`; `AXK1.py:451`; `openpangu.py:288`) — but only as a
  model-author decision, not a serving flag. [V]

**Verdict:** ABSORB the knob (A3, ~1–2 days), keep the `Mapping` factorization as a design
reference (A4). Do not port `mapping.py` — it would fight vLLM's global `parallel_state`
group registry for no near-term gain.

**Caveat on our own axis count:** TokenSpeed's `cp` axis is dead. `attn_cp_size` threads
into `scatter_index` (`mapping.py:188-195`) but `CP_METADATA.set()`
(`python/tokenspeed/runtime/layers/utils.py:63`) has **zero call sites** — the only readers
are in `python/tokenspeed/runtime/models/glm5_nextn.py:145-174`, so every CP branch is
unreachable. [V] vLLM, by contrast, has two *working* context-parallel axes (§3.5).

---

### 3.2 Expert parallelism and load balancing — **vLLM ALREADY BETTER, by a wide margin**

This was flagged as potentially high value ("expert imbalance is a first-order throughput
problem"). It is high value. It is also the component where the gap is largest.

**The algorithm is literally the same code lineage.** Both are ports of DeepSeek's EPLB:
`balanced_packing`, `replicate_experts`, `rebalance_experts_hierarchical`,
`rebalance_experts`. TokenSpeed:
`python/tokenspeed/runtime/moe/eplb_algorithms/deepseek.py:28`, `:77`, `:111`, `:208`. [V]
vLLM: `vllm/distributed/eplb/policy/default.py:23`, `:76`, `:104`, `:275`. [V]

**So the algorithm is worth nothing as an absorption target.** What differs is everything
around it.

| Axis | TokenSpeed | vLLM |
|---|---|---|
| Total LOC | 1,682 (`python/tokenspeed/runtime/moe/`) [V] | 3,042 (`vllm/distributed/eplb/`) + `vllm/v1/worker/gpu/eplb_utils.py` (169) [V] |
| Wired? | **No.** `compute_initial_expert_location_metadata` (`expert_location.py:436`) has **zero call sites** repo-wide [V]; `ExpertLocationMetadata.update()` zero callers [V-prior]; `_global_expert_distribution_recorder` is initialised to a no-op with no setter (`distribution_recorder.py:238-240`) [V] | **Yes.** `eplb_step()` runs on **every forward**: `vllm/v1/worker/gpu_model_runner.py:3414-3427`, called at `:4744` and (dummy) `:6203`; V2 runner via a `step_eplb_after` decorator on `_dummy_run`/`sample_tokens`/`pool` (`vllm/v1/worker/gpu/eplb_utils.py:32-48`) |
| Static or online | Static only (and unreachable) | **Online.** `EplbState.step()` (`eplb_state.py:526`) increments a counter (`:629`) and fires `rearrange()` (`:721`) at `step_interval` (`:645-656`); first fire is seeded to 3/4 interval (`:441-445`) |
| Async overlap | none | **On by default.** `use_async=True` (`vllm/config/parallel.py:82`); dedicated thread + dedicated CUDA stream (`vllm/distributed/eplb/async_worker.py:24-47`), **one MoE layer per engine step** (`:110-162`), `CpuGpuEvent` cross-thread primitive (`eplb_utils.py:16-61`) |
| Weight movement | n/a — the metadata is never applied | Per-expert-row P2P through a staging buffer: `rearrange_expert_weights_inplace` (`rebalance_execute.py:511-615`), `move_to_buffer` (`:172-347`), `move_from_buffer` (`:350-424`) |
| Transports | n/a | **Four**: NCCL `batch_isend_irecv` (`eplb_communicator.py:98-152`), **gloo with CPU staging** (`:155-238`), NIXL receiver-initiated RDMA READ (`:241-611`), raw pyNCCL (`:614-655`) |
| Isolation | n/a | Its own process group named `"eplb"` with EP's ranks, explicitly *"to isolate EPLB communications from MoE forward pass collectives and prevent deadlocks"* (`vllm/distributed/parallel_state.py:1917-1939`) |
| Load recording | Hooked at `layers/moe/topk.py:598` into a **no-op** recorder [V-prior] | Fused Triton kernel in the router: `tl.atomic_add(out_ptr + physical_id, 1, ...)` (`vllm/model_executor/layers/fused_moe/router/base_router.py:93`), windowed into `expert_load_window[window_size, layers, physical_experts]` (`eplb_state.py:430-438`, roll at `:610-623`), gated by `should_record_tensor` so it costs nothing outside the measurement window (`:660-687`) |
| Model coverage | routing-time remap exists (`topk.py:108-186`) but `ExpertLocationDispatchInfo.init_new` has zero call sites [V-prior] | `MixtureOfExperts` protocol (`vllm/model_executor/models/interfaces.py:854-931`), **36 model files** implement it |
| Tests | **none** | **9 dedicated test files**, incl. a golden vector taken from `github.com/deepseek-ai/eplb` asserting bit-exact `phy2log`/`logcnt` (`tests/distributed/test_eplb_algo.py:61-72`) |
| CI | none | `.buildkite/test_areas/expert_parallelism.yaml:5-40`, `e2e_integration.yaml` on **4×H100 and 2×B200**, `test-amd.yaml:1331+` on MI300 |
| Extra refinement | none | `preserve_intragpu_slots` (`policy/default.py:192-272`, applied `:326-329`) reorders slots so experts that stay on a GPU keep their slot — **avoids needless copies**. We have no equivalent. |

**Two additional TokenSpeed defects worth recording.** (a) A latent `AttributeError`:
`distribution_recorder.py:263` reads `server_args.enable_deepep_moe`, which is not a field
of `ServerArgs` — `stat_approx` mode would crash. [V] (b) The EP weight loader assumes a
contiguous per-rank expert range and says so: *"Expert ownership is assumed to be a
contiguous per-rank range here. EPLB-aware remapping would need a different planning step"*
(`python/tokenspeed/runtime/layers/moe/loader.py:69-70`). [V-prior] So even the "just wire
it up" estimate in `big-model-serving.md:1207-1214` understates the work.

**Does vLLM's EPLB need NVLink or Blackwell? No.** [V] There is no interconnect check and
no arch gate in `vllm/distributed/eplb/` — the only NVLink mentions are two docstring
comments explaining *why* the hierarchy packs groups per node (`policy/default.py:118`,
`:294`); `num_nodes` comes from `get_node_count()` (`eplb_state.py:802`), i.e. process
placement, not link detection. The hard gates are: CUDA-or-ROCm
(`vllm/config/parallel.py:474-479`), `enable_expert_parallel`
(`:480-481`), `tp×pcp×dp > 1` (`:484-495`). Async EPLB refuses the NCCL transports and
falls back to **gloo with CPU staging** when NIXL is absent (`:104-110`, auto-select
`:945-960`).

**And it works with the PCIe-friendly EP path.** vLLM's default all-to-all backend is
`"allgather_reducescatter"` (`vllm/config/parallel.py:186`) — the *same* all-gather +
masked-local-experts + reduce scheme `big-model-serving.md:208-232` identified as
TokenSpeed's PCIe-viable EP win. All four scheduled EPLB accuracy jobs run with
`BACK="allgather_reducescatter"` (e.g.
`.buildkite/scripts/scheduled_integration_test/qwen30b_a3b_fp8_dp4_async_eplb.sh:20`, `:44`). [V]

**Consequence: vLLM's EPLB is exercisable on our PCIe box today** at EP=2 or EP=4, with the
gloo communicator and the default all-gather EP scheme. That is a *validate-now* item — and
it belongs to vLLM, not to us. **Absorption value from TokenSpeed's EPLB: zero.**

---

### 3.3 Prefill/decode disaggregation — **vLLM ALREADY BETTER on framework; ABSORB one technique**

**TokenSpeed's design is genuinely good.** `Role` is a first-class C++ FSM domain, not a
Python bolt-on: P/D requests start in `fsm::Bootstrapping` rather than `fsm::Submitted`
(`tokenspeed-scheduler/csrc/scheduler/request.cpp:34`), with 22 `Role` branch sites in the
scheduler [V-prior]. Layer-wise streaming is on by default
(`--disaggregation-layerwise-interval` default **1**,
`python/tokenspeed/runtime/utils/server_args.py:317`) [V]. `transfer_plan.py` is a
declarative layout negotiator with a versioned wire protocol
(`TRANSFER_PLAN_PROTOCOL_VERSION = 1`, `python/tokenspeed/runtime/pd/transfer_plan.py:113`)
that emits an `"identity"` plan when prefill TP == decode TP (`:229-246`) and a
`"fragmented"` plan otherwise (`:248-330`). [V]

**But the framework comparison is not close.**

| Axis | TokenSpeed | vLLM |
|---|---|---|
| Abstraction | Mooncake-specific classes; no ABC third parties implement | `KVConnectorBase_V1` ABC, 708 lines, with a documented contract (`vllm/distributed/kv_transfer/kv_connector/v1/base.py:171`, module docstring `:1-40`) |
| Transports | **One** (mooncake), plus a `mooncake_async` variant with **zero references in tests or CI** — it exists only as a CLI choice (`utils/server_args.py:1929`) and an enum (`pd/utils.py:134`) [V] | **14 registered connectors** (`kv_connector/factory.py:152-242`): NIXL (5,194 LOC), Mooncake (2,179), MooncakeStore (3,493), MoRIIO (3,773), LMCache ×2 (354 + 1,226), HF3FS (2,445), FlexKV, offloading, simple CPU offload, DecodeBench, Example, **plus `MultiConnector` for stacking** (667) |
| Data plane on our box | **Hardcoded `"rdma"`** — `initialize(hostname, "P2PHANDSHAKE", "rdma", device_name)` (`python/tokenspeed/runtime/pd/base/mooncake_engine.py:74-79`) [V]. Upstream's own single-node CI works around it with `MC_INTRANODE_NVLINK=1` [V-prior] | **Runnable without an RNIC.** vLLM's Mooncake connector makes the protocol a config field (`mooncake/mooncake_connector.py:928-938`, default `"rdma"`, overridable); NIXL defaults to UCX backends (`nixl/base_worker.py:261-263`) and supports `kv_buffer_device="cpu"` registering DRAM (`:378`, `nixl/utils.py:79-90`); `ExampleConnector` uses a shared filesystem (`example_connector.py:103-107`); `LMCacheMPConnector` is TCP-only (`lmcache_mp_connector.py:486`) |
| Heterogeneous P-TP/D-TP | `plan_for_decode_rank` intersects logical head intervals and emits byte offsets (`transfer_plan.py:221-330`). **MLA bypasses the planner entirely** — it takes `_legacy_mla_route_plan` (`pd/mooncake/receiver.py:350`) [V-prior] | `compute_tp_mapping` (`nixl/tp_mapping.py:65`) covers D-TP≥P-TP (`:79-84`), P-TP>D-TP with GQA dedup via `np.unique` (`:85-95`), a separate SSM/Mamba source-rank path (`:97-106`), byte offsets (`:125-135`), **and differing page/block sizes** via `block_size_ratio` (`nixl/base_worker.py:1574-1583`) — which our planner does not handle |
| Layer-wise streaming | **Yes, and it is the better mechanism** — see below | The ABC has the hooks (`base.py:311` `wait_for_layer_load`, `:325` `save_kv_layer`, seam at `vllm/model_executor/layers/attention/kv_transfer_utils.py:51`, `:57`) and LMCache / HF3FS / FlexKV / Example use them — but **NIXL does not**: both are explicit no-ops (`nixl/connector.py:277-289`) |
| Router | none in-tree (we spawn sglang's `smg` as a subprocess) | none in-tree either — only demo proxies under `examples/disaggregated/`, and the docs say so: *"vLLM relies on third-party connectors for production-level disaggregated prefilling"* (`docs/features/disagg_prefill.md:92`). **Parity.** |
| Encoder disagg (EPD) | Real (`python/tokenspeed/runtime/epd/`), integration-tested | **Also real** — `mm_encoder_only` (`vllm/config/multimodal.py:141-146`) plus a whole `vllm/distributed/ec_transfer/` connector subsystem (1,618 LOC). **Not a differentiator.** [V] |

**The one thing worth absorbing (A2).** TokenSpeed's layer-wise push is not "call a hook per
layer" — it is **a transfer thread polling a GPU-produced step counter**. The counter is a
device `int64` mirrored to pinned host memory and read via `ctypes.c_int64.from_address`
(`python/tokenspeed/runtime/pd/utils.py:433-477`), so `_wait_until_cache_step`
(`pd/mooncake/prefill.py:601`, implementation `:439-446`) blocks the transfer thread
**without a CUDA sync and without touching the forward path at all**. The loop ships layer
groups of `interval` layers, interleaving mamba/GDN state per group
(`prefill.py:582-664`, mamba at `:645-656`). It is driven from `prepare_prefill` *before*
the forward (`pd/prefill_executor.py:195-221`). [V]

vLLM's equivalent is a decorator on the attention layer
(`vllm/model_executor/layers/attention/kv_transfer_utils.py:15`) — which works, but forces
`requires_piecewise_for_cudagraph` (`base.py:609`) because layer-wise ops cannot be captured
in a full CUDA graph. **The step-counter design avoids that trade entirely.** [I — I read
both mechanisms; I did not measure either.] That is a genuine technique worth carrying into
a first-party connector, and it is the only structural idea in our P/D stack that vLLM's
framework does not already express.

**Port cost for A2:** ~1–2 weeks, as an optional push path inside a first-party
`KVConnectorBase_V1` implementation. **Validation: BLOCKED on a second machine or an RNIC**
for the throughput claim; the mechanism itself can be smoke-tested over TCP.

---

### 3.4 Hierarchical KV / offload — **vLLM ALREADY BETTER**

| Axis | TokenSpeed | vLLM |
|---|---|---|
| Scale | `HostKVCache` (`python/tokenspeed/runtime/cache/kv_cache_host.py:128`) + `HostExecutor` + `KVStoreStorage`; **one** L3 backend registered (mooncake, `cache/storage/backend_factory.py:176`) [V] | `vllm/v1/kv_offload/` — **8,661 LOC across 45 files** [V] |
| CPU-tier eviction | Radix-tree strategies incl. LRU/LFU/FIFO/MRU/FILO/Priority (`cache/evict_policy.py:43-71`) | `lru` (default) and **ARC** with T1/T2 real + B1/B2 ghost lists and an adaptive target (`cpu/policies/arc.py:12-171`; registry `cpu/manager.py:31-34`, default at `:50`) |
| L3 tiers | 1 (mooncake) | **3 in-tree**: filesystem (`tiering/fs/`), S3-compatible object store via the NIXL OBJ plugin (`tiering/obj/`), and peer-to-peer remote CPU (`tiering/p2p/`, ~2,500 LOC) — registered at `tiering/factory.py:55-77` |
| Transfer mechanism | pinned host memory + copy kernels [V-prior] | Per-transfer CUDA stream **pool** with event-ordered chaining (`cpu/gpu_worker.py:232-236`, `:362-400`), driver-level **`cuMemcpyBatchAsync`** on CUDA ≥ 12.8 (`csrc/libtorch_stable/cache_kernels.cu:112-135`), Triton fallback for small aligned H2D pages (`gpu_worker.py:35-58`) |
| Scheduling integration | Host-tier prefix match is part of the C++ admission decision — `AdmissionMatch{device, host}` [V-prior] | Connector is consulted at schedule time too: `get_num_new_matched_tokens` at `vllm/v1/core/sched/scheduler.py:742-745`, with `None` meaning "async lookup in flight, ask again" (`base.py:454`). **Functionally equivalent.** |
| Default | **ON.** `enable_kvstore` is declared `False` (`utils/server_args.py:206`) and then set `True` unless `--disable-kvstore` (`:687-695`), at `kvstore_ratio = 2.0` (`:207`) — a host pool **twice the device KV pool in tokens** [V] | **OFF.** `kv_offloading_size` defaults `None` (`vllm/config/cache.py:180`), and the activation path returns immediately when it is `None` (`vllm/config/vllm.py:868-869`) |
| Sizing policy | fixed ratio × device pool; raises only if the host cannot satisfy it (`kv_cache_host.py:170-177`) | **Explicit absolute bytes, mandatory** — `CPUOffloadingSpec` raises if `cpu_bytes_to_use` is absent (`cpu/spec.py:78-82`); CLI `--kv-offloading-size <GiB>` converts (`vllm/config/vllm.py:883-886`) |

**The operational lesson — the part actually worth carrying.** TokenSpeed defaulted a
hierarchical KV tier **on**, at 2.0× the device pool, and it OOM-killed E17 with a 139 GB
startup allocation. Upstream's own CI sets `KVSTORE_RATIO=0.5` and every big-model recipe
in `docs/recipes/models.md` passes `--disable-kvstore` — i.e. **upstream does not run its
own default.** [V-prior] We then inherited that default and ran it in every benchmark since
E4 without once measuring its hit rate.

vLLM already does the right thing on both counts: off by default, and sized in absolute
bytes with a hard error rather than a silent multiplier. The absorption here is a **rule,
not code**: *an offload tier is opt-in, and its size is derived from measured host RAM, never
from a ratio of device memory.* That rule also belongs in the self-configuring-engine work
(memory: `self-configuring-engine.md`).

**Verdict: NOT WORTH IT to port. `vllm/v1/kv_offload/` is broader on every axis we care
about, and the one place we were arguably ahead (host-tier match inside admission) turns out
to be parity.**

---

### 3.5 DP-attention and other large-model parallelism — **vLLM ALREADY BETTER**

**DP-attention.** TokenSpeed: each DP rank runs its own scheduler and KV pool; coordination
is one CPU `all_gather_into_tensor` of `[num_tokens, batch_size, forward_mode]` per step
(`python/tokenspeed/runtime/engine/event_loop.py:1632-1660`), producing `need_idle_forward`
(`:1641`) so empty ranks run a dummy forward and collectives still match. [V]

vLLM: `coordinate_batch_across_dp` (`vllm/v1/worker/dp_utils.py:164-225`) all-reduces a
`[4, dp_size]` int32 of `(orig_tokens, padded_tokens, should_ubatch, cudagraph_mode)`
(`:35-54`), min-reduces the cudagraph mode so all ranks agree (`:92-98`), and pads only when
cudagraphs or microbatching require it (`:145-152`) — otherwise ragged sizes are kept and
fed to ragged `all_gatherv`/`reduce_scatterv` in the a2a manager
(`vllm/distributed/device_communicators/all2all.py:59-67`). Idle ranks call
`execute_dummy_batch` (`vllm/v1/engine/core.py:2042-2060`). A dedicated `DPCoordinator`
process publishes per-engine queue lengths for front-end load balancing
(`vllm/v1/engine/coordinator.py:23-57`). [V]

**Same idea; vLLM's is strictly richer** (ragged sizes instead of forced padding, an NCCL/gloo
switch to avoid a GPU sync under async scheduling at `vllm/config/parallel.py:226-232`, and a
coordinator process we do not have).

**Everything else in this bucket:** we have nothing vLLM lacks.

- **Context parallelism.** Ours is dead (§3.1b). vLLM has **two working axes**: DCP
  (`decode_context_parallel_size`, `vllm/config/parallel.py:340`) wired into 8 attention
  backends with LSE-correcting combines (`vllm/v1/attention/ops/common.py:111`, `:213`,
  `:238`) and a dedicated `DCPGroupColumnParallelLinear` (`linear.py:619-657`); and PCP
  (`prefill_context_parallel_size`, `:124`, implementation
  `vllm/model_executor/layers/attention/pcp.py:11-92`). [V]
- **DP request routing.** Ours has three policies
  (`ROUND_ROBIN`/`SHORTEST_QUEUE`/`MINIMUM_CACHE_USAGE`,
  `python/tokenspeed/runtime/engine/data_parallel_controller.py:56-64`) [V-prior]; vLLM's
  coordinator publishes queue depth to the front end and adds request-wave tracking. Rough
  parity, slight edge to vLLM for the external-LB story.
- **Vision-tower sub-mapping.** Ours gives the encoder its own `(tp, dp)` over the attention
  TP world (`mapping.py:269-312`); vLLM has `mm_encoder_tp_mode="data"` — batch-level DP
  across TP ranks with full weights per rank (`vllm/config/multimodal.py:146-158`) — which
  is the same idea, exposed as a serving flag. [V]

---

### 3.6 What vLLM has that we lack, and that matters at scale

Stated plainly, because a correct ledger is the point.

| Capability | Evidence | Why it matters at scale |
|---|---|---|
| **Pipeline parallelism** | `PPMissingLayer` (`vllm/model_executor/models/utils.py:785`), `make_layers` (`:798-830`), `get_pp_indices` with remainder pushed to middle stages (`vllm/distributed/utils.py:127-172`); **160 models declare `SupportsPP`** | The one parallelism dimension that tolerates slow links — i.e. exactly what a PCIe or cross-node deployment needs. We have **zero** (`big-model-serving.md:336-341`) |
| PP schedule | Not 1F1B (inference has no backward): a depth-`pp_size` queue of whole scheduler steps. `max_concurrent_batches = pp_size` (`vllm/config/vllm.py:516-527`), `step_with_batch_queue` (`vllm/v1/engine/core.py:617-729`), non-blocking `isend/irecv_tensor_dict` with an all-gather optimisation (`vllm/v1/worker/gpu_worker.py:1048-1091`) | |
| PP × async scheduling | Sampled tokens broadcast from the last PP rank on a **sibling NCCL communicator on a side stream** (`vllm/v1/worker/gpu/pp_utils.py:52-88`), with scheduler-side cadence `next_decode_eligible_step = current_step + pp_size` (`vllm/v1/core/sched/async_scheduler.py:45-49`) | |
| **LoRA** | `vllm/lora/` — 42 files, 14,204 LOC; **114 models** declare `SupportsLoRA`; batched multi-adapter kernels group tokens by adapter id in one shrink/expand pair (`vllm/lora/ops/triton_ops/lora_kernel_metadata.py:14-35`); composes with TP (`layers/row_parallel_linear.py:74`), **PP** (`model_manager.py:401-403` skips `PPMissingLayer`), MoE (`punica_gpu.py:419-633`), FP8, multimodal; runtime load/unload HTTP API | We have **zero** LoRA. For a commercial serving product this is a category, not a feature |
| **DBO / dual-batch overlap** | `vllm/v1/worker/ubatching.py` (241), `gpu_ubatch_wrapper.py` (537), `ubatch_utils.py` (265): two host threads, one compute + one comm stream, cooperative yields at the MoE all-to-all boundary (`fused_moe/modular_kernel.py:1201`, `:1221`, `:1241-1246`), SM partitioning via `VLLM_DBO_COMM_SMS=20` (`vllm/envs.py:262`), CUDA-graph capture of the interleaved schedule (`gpu_ubatch_wrapper.py:212-234`) | Comm/compute overlap is the classic large-MoE throughput lever. Our tree has **zero** two-batch overlap. **But see §5 — it is not exercisable on our box** |
| **Elastic EP** | `vllm/distributed/elastic_ep/` (1,402 LOC), stateless NCCL groups (`parallel_state.py:1878-1889`), weight transfer + standby groups (`elastic_execute.py:187`, `:216`, `:350`), HTTP `/scale_elastic_ep` (`vllm/entrypoints/serve/elastic_ep/api_router.py:33`) | Scale EP up/down at runtime without a restart. Nothing comparable exists in our tree |
| Speculative decoding breadth | 12 proposer families incl. ngram (CPU + GPU), suffix decoding, medusa, EAGLE/EAGLE3, DFlash, DSpark, **21 MTP variants**; 49 registry entries (`vllm/model_executor/models/registry.py:599-655`). **Parallel drafting** — multiple tokens per position (`vllm/config/speculative.py:165-169`) | Ours is 3 drafters, **chain-only** (`utils/server_args.py:615-620`). Note vLLM has **no tree speculation either** — that is parity, not a gap |
| Structured output | 4 backends; grammar compiled on a **dedicated thread pool** returning a `Future` (`vllm/v1/structured_output/__init__.py:75-77`, `:166-171`), with the request parked in `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` so one slow grammar never blocks a step | We have xgrammar only |
| Hybrid KV cache manager | `HybridKVCacheCoordinator` (`vllm/v1/core/kv_cache_coordinator.py:521`), 13 cache specs, 7 per-type managers, groups with **different block sizes** (`:560-570`), cascade attention (`vllm/v1/attention/backends/flashinfer.py:1292-1325`) | Sliding-window + full-attention + Mamba sharing one pool is table stakes for modern hybrid models |
| Async scheduling | On by default (`vllm/config/vllm.py:1112`), `AsyncScheduler` (`vllm/v1/core/sched/async_scheduler.py:12`) | |
| Priority scheduling | `SchedulerPolicy` is `fcfs` or `priority` (`vllm/config/scheduler.py:22`, `:99-104`), victim selection by `(priority, arrival_time)` (`vllm/v1/core/sched/scheduler.py:578-583`) | We have **no `schedule_policy` at all** and `RequestSpec` carries no arrival timestamp [V-prior] |
| Multi-node launch | Ray (`vllm/v1/executor/ray_executor.py:64`), mp across nodes (`vllm/config/parallel.py:274-277`), external launcher | |
| Model coverage | 290 files / 190,937 LOC in `vllm/model_executor/models/`; **337 registered architectures** | vs ~30 |

**One place vLLM is behind and we should say so:** preemption is recompute-only — blocks are
freed and `num_computed_tokens = 0` (`vllm/v1/core/sched/scheduler.py:1203-1225`), and
`swap_space` has been **removed** from the tree entirely [V]. TokenSpeed retracts by writing
KV back to the host tier and restoring by loadback, not recompute
(`tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp:499-621`) [V-prior]. On a
KV-pressured big model that is the better policy — but our retraction path has never fired
(E25 observed zero retractions at C=256), so this is an *untested* advantage, not a
demonstrated one.

---

## 4. Architecture worth copying vs code worth porting

| Item | Architecture | Code | Why |
|---|---|---|---|
| Placement→comm compiler | **Yes** — declarative placement with derived collectives is the right shape, and vLLM's FX peephole passes are not it | **No** | 1 consumer, 0 tests, re-derives the heuristic it replaces (`compiler.py:404-408`) |
| Per-layer-family `Mapping` | **Yes** — independent (size, stride) factorizations over one flat world | **One knob only** (`dense_tp`) | vLLM's `ep_size = dp×pcp×tp` identity (`fused_moe/config.py:1236`) is not yet binding for the shapes we will run |
| EPLB algorithm | **No** — vLLM already has the identical DeepSeek algorithm plus `preserve_intragpu_slots` | **No** | Absorption value zero |
| Layer-wise KV streaming | **Yes** — the GPU-step-counter design (`pd/utils.py:433-477`) avoids the piecewise-cudagraph tax vLLM's per-layer hook forces (`base.py:609`) | **Yes**, as an option inside a first-party connector | The only structural P/D idea vLLM's framework does not express |
| `transfer_plan.py` layout negotiation | **The fail-fast discipline, yes** (`UnsupportedPDLayoutError`) | **No** | vLLM's `tp_mapping.py` covers the same cases in 142 lines *and* handles differing block sizes, which ours does not |
| KV offload sizing | **The rule, yes**: opt-in, absolute bytes, derived from measured host RAM | **No** | vLLM already implements the rule (`cpu/spec.py:78-82`, `vllm/config/vllm.py:868-869`) |
| C++ FSM scheduler + `Role` | **Yes** (tracked elsewhere) | Out of scope here | Multi-month; it is a scheduling asset, not a datacenter one |

---

## 5. Validate now vs wait for hardware

### 5.1 Exercisable on 4× RTX PRO 6000, PCIe, no NVLink — today

Ranked by (value at scale) × (confidence it will run). **All four of these test vLLM's own
machinery**, which is the point: the fastest way to know whether we need TokenSpeed's
datacenter tier is to find out where vLLM's actually breaks.

1. **Expert parallelism at EP=2 on {0,1} and EP=4**, with vLLM's default
   `all2all_backend="allgather_reducescatter"` (`vllm/config/parallel.py:186`). No DeepEP, no
   NVLink, no RDMA — all-gather + local experts + reduce over plain collectives. **Zero new
   code.** This is the headline "we own EP and never used it" item from
   `big-model-serving.md:1079-1085`, and vLLM can run it unmodified.
2. **EPLB, online, on our box.** Set `--enable-eplb` with EP≥2; the communicator auto-selects
   **gloo with CPU staging** when NIXL is absent (`vllm/config/parallel.py:945-960`,
   `eplb_communicator.py:155-238`), which needs no fabric at all. Watch
   `log_balancedness` (`vllm/config/parallel.py:73`). This measures the thing that actually
   matters — whether expert imbalance is real on a small MoE — at zero porting cost. **[I]
   on whether the rebalance is net-positive at EP=2; that is exactly what the run answers.**
3. **Pipeline parallelism at PP=2 (and PP=4).** PP is *the* dimension that tolerates PCIe,
   and it is our largest verified absence. Two questions worth answering before datacenter
   hardware: does the depth-`pp_size` batch queue (`vllm/v1/engine/core.py:617-729`) keep
   the pipeline full at our concurrencies, and does PP+async's sibling-NCCL broadcast
   (`gpu/pp_utils.py:52-88`) behave over `NODE`-class PCIe links.
4. **CPU KV offload with an explicitly sized tier** (`--kv-offloading-size`), LRU vs **ARC**
   (`cpu/manager.py:31-34`). This finally answers the question E4–E33 never asked — what a
   host tier is worth — and it does so with vLLM's implementation rather than the one that
   OOM-killed E17.
5. **P/D over TCP.** Unlike our engine, vLLM's P/D is *testable without an RNIC*: NIXL with
   UCX backends and `kv_buffer_device="cpu"` (`nixl/base_worker.py:261-263`, `:378`), or
   `ExampleConnector` over a shared filesystem, or `LMCacheMPConnector` over TCP. **[I]** —
   vLLM passes the backend string straight through to NIXL without validating it, so whether
   UCX-over-TCP works end to end is a property of UCX, not of vLLM. Cheap to find out, and a
   negative result is still worth knowing.
6. **A3 — expose `disable_tp` for dense MLP as a serving flag** (~1–2 days), then measure at
   TP2 whether replicated-dense-plus-one-fewer-all-reduce beats sharded dense on PCIe. This
   is the one config gap where we are genuinely ahead, and PCIe is precisely the regime where
   removing an all-reduce should matter most. **[I]**

### 5.2 Genuinely blocked until datacenter hardware arrives

| Item | Blocker | Evidence |
|---|---|---|
| **DBO / dual-batch overlap** | Requires `data_parallel_size > 1` **and** `all2all_backend ∈ {deepep_low_latency, deepep_high_throughput, nixl_ep}` — a hard assert (`vllm/config/vllm.py:1536-1549`). DeepEP needs NVLink or RDMA/IBGDA | `vllm/v1/worker/dp_utils.py:189-191`, `vllm/config/vllm.py:1536-1549` |
| **DeepEP all-to-all** (either engine) | Normal mode sizes NVLink buffers; low-latency is RDMA/IBGDA-only | `tokenspeed-kernel/.../ops/communication/deep_ep.py:128-155` [V-prior] |
| **MegaMoE** | `torch.cuda.get_device_capability(device)[0] != 10 → raise` — **exactly** SM100, excluding sm_90 and sm_120 — plus NVLink symmetric memory | `python/tokenspeed/runtime/models/deepseek_v4.py:2319-2320` [V] |
| **A2 — layer-wise KV push, throughput claim** | Needs a real P/D pair with a fast link to show a win; the mechanism can be smoke-tested, the benefit cannot | `pd/mooncake/prefill.py:582-664` |
| **EPLB over NIXL RDMA / async at scale** | The gloo path validates correctness; the async NIXL path (`eplb_communicator.py:241-611`) needs an RNIC | `vllm/config/parallel.py:945-960` |
| **Elastic EP** | Needs multi-node DP to be meaningful; forbidden with PP>1 and unsupported on the V2 runner | `vllm/config/parallel.py:818-822`, `vllm/config/vllm.py:2190-2191` |
| **Cross-node anything** | One box | — |

---

## 6. Traps — things that look complete and are not

EPLB's zero call sites was the given example. It is not the only one. Every entry below is a
component that reads as finished and is not, and every one is a reason to *verify wiring
before costing a port*.

| Trap | Evidence | Class |
|---|---|---|
| **EPLB** — 1,682 LOC, faithful DeepSeek port, no call sites | `compute_initial_expert_location_metadata` (`python/tokenspeed/runtime/moe/expert_location.py:436`) has zero call sites repo-wide [V]; recorder is a hard-coded no-op (`distribution_recorder.py:238-240`) [V]; no tests | Unwired |
| **Context parallelism** — `Mapping` axis, zigzag helpers, `scatter_index` threading | `CP_METADATA.set()` (`python/tokenspeed/runtime/layers/utils.py:63`) has **zero callers**; only readers are `models/glm5_nextn.py:145-174`. No CLI flag — env `ENABLE_CP` only | Dead code |
| **The placement→comm compiler** — 557 lines, four comm-op types, cross-layer placement threading | One consumer (`models/gpt_oss.py:446`), zero tests, and the AR/RSAG decision is hardcoded to match `CommManager` (`compiler.py:404-408`) | Looks general, serves one model |
| **`CustomAllReduceBackend` and `PyNcclCommunicator`** | `initialize_comm_backend(use_pynccl, use_custom_allreduce)` (`comm_backend/registry.py:31-45`) is **never called with arguments** — the only call site is `get_global_backend()` with defaults (`registry.py:48-53`) | Dead code |
| **`All2AllBackend.FLASHINFER_NVLINK_ONE_SIDED`** | Declared (`python/tokenspeed/runtime/layers/moe/utils.py:49`) but `_validate_a2a_backend` raises `NotImplementedError` for anything but `none`/`deepep` (`tokenspeed-kernel/.../ops/moe/__init__.py:48-51`, called `:150`) | Declared, unimplemented |
| **DeepEP + `moe_backend=auto`** | Resolver sets `solution="flashinfer_cutedsl_deepep"` while the kernel registers as `"flashinfer_cutedsl"` — an exact-string mismatch yielding `NoKernelFoundError`. Works only because the model gate forces the user to pass `--moe-backend flashinfer_cutedsl` explicitly | Silent misconfiguration [V-prior] |
| **`mooncake_async`** — a full 24 KB alternative P/D transport | A valid CLI choice (`utils/server_args.py:1929`) and an enum member (`pd/utils.py:134`), with **zero references in tests or CI** | Unvalidated |
| **MegaMoE** | `capability[0] != 10 → raise` (`models/deepseek_v4.py:2319-2320`) — **exactly** SM100. Not "Blackwell+", not "Hopper+". This will not generalise to sm_120, sm_103, or anything future | Arch-gated in a way that cannot generalise |
| **MSA cannot be P/D-disaggregated** | `get_contiguous_buf_infos` raises: *"MiniMax sparse cache transfer requires index-key side-cache support."* (`python/tokenspeed/runtime/layers/attention/kv_cache/msa.py:101-104`) | Silent feature hole — MSA models simply cannot be disaggregated |
| **MLA bypasses the transfer planner** | `transfer_plan.py` is the nicest piece of the P/D stack, and the most important model family (MLA: DeepSeek, Kimi, GLM-5) takes `_legacy_mla_route_plan` instead (`pd/mooncake/receiver.py:350`) | The showcase component does not cover the showcase models [V-prior] |
| **`kvstore` default ON at ratio 2.0** | `enable_kvstore: bool = False` (`utils/server_args.py:206`) then flipped `True` in `_handle_kvstore` (`:687-695`); `kvstore_ratio: float = 2.0` (`:207`). Upstream's own recipes all pass `--disable-kvstore` | A default upstream does not run |
| **`distribution_recorder.py:263`** | Reads `server_args.enable_deepep_moe`, which is not a field of `ServerArgs` — `stat_approx` mode would `AttributeError` on first use | Latent crash in unreached code |

**The pattern.** Nine of twelve are in the datacenter tier. That is not a coincidence: this is
the layer upstream built for hardware they had and we do not, wired for models they ran and we
have not, and the parts that were lifted rather than exercised are exactly the parts that look
finished. **Treat "the file exists and is well-written" as evidence of nothing until you have
found the call site.**

---

## 7. What would change these conclusions

- **A benchmark.** There is none here. If vLLM's EPLB turns out to cost more than imbalance
  costs at EP=2–4 on PCIe, item 2 of §5.1 flips from "validate now" to "not our problem yet".
- **Datacenter hardware.** Three verdicts above (DBO, DeepEP, elastic EP) are BLOCKED rather
  than settled. A single 8×H200 or B200 node would settle all three and would also make A2's
  value measurable.
- **A second machine or an RNIC.** vLLM's P/D is *claimed* runnable over UCX/TCP
  (`nixl/base_worker.py:261-263`) but I did not run it. If UCX-over-TCP does not work in
  practice, §5.1 item 5 moves to §5.2 and A2's port cost stops being justifiable at all.
- **Upstream drift.** This pin is 2026-07-24 and vLLM moves ~34 commits/day. Every "vLLM
  already better" verdict gets stronger with time, and every "we are ahead" verdict gets
  weaker. `unieinfra-vllm/` does not sync upstream by design
  (`unieinfra-vllm/PROVENANCE.md:22-25`), so these comparisons should be re-run, not assumed,
  before any port is scheduled.

*Companion documents: `docs/design/big-model-serving.md` (the TokenSpeed-side inventory this
builds on), `docs/design/vllm-base-assessment.md` (the base decision), and
`docs/design/project-layout-and-absorption.md` (the absorption model). Update this document as
items move between ABSORB, BLOCKED, and settled.*
