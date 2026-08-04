<!-- English canonical. Chinese mirror TODO: big-model-serving.zh.md -->
# Big-Model Serving: What We Inherited and Never Exercised

Status: **Study / recon complete, implementation not started** · Date: 2026-07-25 · Owner: royshih

## Why this document

Every experiment we have run (E1–E27) is **Qwen3-8B dense, one sm_120 card**. That is
the small-model corner. The engine we hard-forked was not built for that corner — it
was built for **flagship MoE and MLA models on datacenter NVIDIA**, and its README
advertises Qwen3.5-397B-A17B and Kimi K2.5 on B200. Nearly all of that machinery is
sitting in our tree, unexercised.

This document inventories that machinery with `file:line` evidence, splits it into
*datacenter-gated* vs *portable*, compares it against the reference engines, and
ranks what we can validate **now** on 4× RTX PRO 6000.

**Method and honesty rules.** This is a **read-only code and document study**. No GPU
was used and no source was modified. Claims are labelled:

- **[validated]** — we ran it on our hardware; cites `docs/experiments-log.md`.
- **[code]** — I read the code; the path exists and I cite it. It does *not* mean it works.
- **[inferred]** — my reasoning from code, arithmetic, or upstream docs. Explicitly flagged.

Where I could not determine something, I say so rather than guessing.

---

## 1. Inventory — the big-model machinery already in the fork

Scale of what was inherited: `python/tokenspeed/` ≈ 133k lines, `tokenspeed-kernel/`
≈ 179k, `tokenspeed-scheduler/` ≈ 33k (C++), `tokenspeed-mla/` ≈ 18k,
`tokenspeed-kernel-amd/` ≈ 45k. The model registry carries **30 model files**
(`python/tokenspeed/runtime/models/`) and it is a *flagship* set, not a toy set:
`deepseek_v3.py`, `deepseek_v4.py`, `kimi_k25.py`, `glm5.py`, `minimax_m2.py`,
`minimax_m3.py`, `longcat_flash.py`, `gpt_oss.py`, `inkling.py`, `qwen3_moe.py`,
`qwen3_5_moe.py`, plus five NextN/MTP draft variants.

The upstream's own production launch configs are still in the tree and are the
clearest statement of intended scale — `docs/recipes/models.md` carries working
recipes for DeepSeek V4-Flash on 4× B200 (DP+EP+`mega_moe`),
DeepSeek V4-Pro on 8× B200 (TP8), GLM-5.2-FP8 TP8+EP, Kimi K2.5 NVFP4 TP4+EP with
`trtllm_mla`, MiniMax M3 with MSA + EAGLE3, and Inkling NVFP4 with MTP.

### 1.1 Attention for large models

#### MLA (Multi-head Latent Attention — DeepSeek V3/V4, Kimi K2.5, GLM-5, MiniCPM3)

**The idea as implemented.** MLA compresses K and V into a single low-rank latent
`c_kv` of width `kv_lora_rank` plus a small shared RoPE part `qk_rope_head_dim`.
The KV cache stores *only* that latent — one buffer, `num_kv_heads == 1`, and the
"value" buffer is a **view** of the first `kv_lora_rank` columns, not separate storage
(`python/tokenspeed/runtime/layers/attention/kv_cache/mla.py:100-110`, view at `:283-289`).

The decode path uses **weight absorption**, done at load time rather than in the kernel.
`kv_b_proj` is split into per-head `w_kc`/`w_vc`
(`python/tokenspeed/runtime/models/deepseek_v3.py:128-140`, invoked from `:1734` and `:2166`).
At decode, `q_nopeᵀ · w_kc` folds the K-up-projection into the query
(`deepseek_v3.py:805-810`), so attention scores are taken directly against the
*compressed* latent and per-head K is never materialised; the output is un-absorbed by
`attn_output ᵀ · w_vc` (`deepseek_v3.py:930-937`). Attention itself runs MQA-style —
`self.attn_mqa` is constructed with `num_kv_heads=1, v_head_dim=kv_lora_rank`
(`deepseek_v3.py:604-611`).

Prefill takes the opposite route: it **decompresses** (`kv = self.kv_b_proj(kv_a)[0]`,
`deepseek_v3.py:972`), assembles a full-width `[k_nope | k_pe]` (`:1026-1029`) and runs
plain varlen FMHA — but still writes only the compressed `kv_a` to cache (`:1031-1037`).
The switch is explicit and driven by live batch composition so it survives CUDA-graph
replay: `if num_prefill_tokens > 0: forward_normal_chunked(...)` at `deepseek_v3.py:731`
vs `if num_decode_tokens > 0: forward_absorb(...)` at `:748` (composition derived at `:696-706`).
Chunked-prefix replay re-reads compressed KV per chunk, re-expands, attends non-causally
and merges with `attn_merge_state(..., inplace=True)` (`deepseek_v3.py:1041-1130`, merge at `:1129`).

**Why it matters at scale — the KV arithmetic.** From the cost functions the engine
actually uses to size the pool:

| Scheme | `cache_cell_size()` formula | file:line | DeepSeek-V3 shape, bf16 |
|---|---|---|---|
| MLA plain | `(kv_lora_rank + qk_rope_head_dim) * elem(kv_dtype)` | `configs/mla.py:92-94` | **1152 B**/token/layer (576 B at fp8) |
| MLA `per_token_head` | `kv_lora*elem(kv) + rope*elem(model) + 4` | `configs/mla.py:85-90` | **644 B** |
| DSA | MLA cell **+** `dsa_index_k_row_bytes(index_head_dim)` | `configs/dsa.py:78-82` | +132 B (idx dim 128) |
| GQA/MHA | `max(num_kv_heads // attn_tp_size, 1) * head_dim * 2 * elem` | `configs/mha.py:153-163` | 4096 B for 8 KV heads × 128 |

**An honest caveat the two formulas make plain:** the GQA cell **divides by
`attn_tp_size`**; the MLA cell **does not** — MLA's latent has no head axis to shard.
So MLA's ~3.6× KV advantage holds at TP=1 and *inverts* at high TP (at TP=8, GQA is
512 B vs MLA's 1152 B). MLA's real win is low-TP and fp8-store. [code + arithmetic]

**Backends and their gates.**

| Runtime backend | Kernel it calls | Gate | file:line |
|---|---|---|---|
| `mla` | `tokenspeed_kernel.mla_prefill` / `mla_decode_with_kvcache` → **Triton only on NVIDIA** | none — portable | `backends/mla.py:475`; kernels `ops/attention/triton/mla_prefill.py:274-288`, `triton/mla_decode.py:253-267` |
| `trtllm_mla` | flashinfer TRT-LLM-gen MLA | `ops/attention/flashinfer/__init__.py:65` `if platform.is_blackwell:` rebinds the MLA symbols | `backends/trtllm_mla.py:548` |
| `flashmla` | DeepSeek FlashMLA | import gated `is_nvidia and is_hopper_plus`, `ops/attention/flash_mla/__init__.py:39` | `backends/flashmla.py:836` |
| `tokenspeed_mla` | first-party CuTe DSL (below) | **no Python-level arch guard at all** | `backends/tokenspeed_mla.py:562` |
| `deepseek_v4` | DSv4 sparse + `deep_gemm` MXFP4 indexer | model-side `is_hopper or is_blackwell`, `models/deepseek_v4.py:244, :274` | `backends/deepseek_v4.py:2098` |

**`tokenspeed-mla/` — the first-party kernel package.** 17.6k lines of CuTe DSL
(`nvidia-cutlass-dsl`), declared scope *"Blackwell SM100 and SM103"*
(`tokenspeed-mla/pyproject.toml:8`). It is genuinely sophisticated: `fold_sq_factor`
folds query tokens into the head axis when `num_heads < 128` to fill the BMM1 M-dimension
(`README.md:66-84`), a two-kernel split-KV decode + reduction, 2-CTA UTCMMA to cut shared
memory, split KV-loading warps for latency hiding, and an **AOT binary prefill backend
built with NVIDIA-internal knobs** (`fmha_binary.py`, `TOKENSPEED_MLA_PREFILL_BACKEND=binary`).

Two findings worth flagging:

1. **It is not in the kernel registry.** `ops/attention/tokenspeed_mla/__init__.py` is a
   46-line try/except re-export with **no `@register_kernel`** (`:25-38`). It is consumed
   directly by the runtime backend (`backends/tokenspeed_mla.py:37-42`) and by the model
   (`models/deepseek_v3.py:35, :1105`). `select_kernel("attention", "mla_decode_with_kvcache", …)`
   can therefore never choose it — it bypasses the capability layer entirely. That also
   means **it has no arch guard**: `--attention-backend tokenspeed_mla` on sm_120
   constructs fine, allocates its workspace (`backends/tokenspeed_mla.py:76-91`), and
   then fails inside CuteDSL lowering. A foot-gun, not a graceful fallback.
2. **`objs/` is empty** (`.gitkeep` only) — no AOT `.so` ships in this checkout, so
   `has_binary_prefill()` is False everywhere here.

**Consequence:** on NVIDIA, the only *registered* dense-MLA kernel is **Triton**
(`ops/attention/triton/mla_prefill.py:274`, `triton/mla_decode.py:253`), with
`vendors={nvidia, amd}` and **no arch floor**. FlashMLA registers only the sparse/DSA
entry points (`ops/attention/flash_mla/__init__.py:185`, `:274`), not dense MLA.

#### DSA (DeepSeek Sparse Attention — GLM-5, DeepSeek V3.2/V4)

**The idea.** A cheap "lightning indexer" scores every KV position with a small-head
dot-product, takes a per-row top-k, and the expensive MLA attention then runs **only
over those `index_topk` positions**. This turns decode attention from O(context) to
O(topk) — the difference between viable and non-viable at 128k+ context.

The indexer is small and completely ordinary maths: `wq_b` projects the q-LoRA to
`index_n_heads × index_head_dim`, `wk` is a **single-head** projection of hidden,
`weights_proj` produces per-head weights, plus a LayerNorm and RoPE on the first
`rope_head_dim` columns (`python/tokenspeed/runtime/models/glm5.py:169-269`; fused
`wk_weights_proj` MergedColumnParallelLinear at `:218-224`). Per-head normalisation is
**folded into the scale** (`self.weights_softmax_scale = softmax_scale * index_n_heads**-0.5`,
`glm5.py:190`) so no separate kernel is needed. Layers marked shared can reuse the
previous layer's top-k (`_glm_dsa_skip_indexer_topk`, `glm5.py:104-124`).

The index-K cache is a byte-packed side buffer, `uint8`, `index_head_dim + (index_head_dim/128)*4`
bytes per row (`kv_cache/dsa.py:37-62`, row-bytes at `configs/dsa.py:36-46`). Its physical
layout is **block-split, not per-token interleaved, specifically to match DeepGEMM's ABI** —
the code says so: *"DeepGEMM's `fp8_paged_mqa_logits` expects each page … as
`[page_size * head_dim FP8 values]` followed by `[page_size * num_groups FP32 scales]`
(block-split), NOT a per-token `[fp8 | scale]` interleave"* (`kv_cache/dsa.py:113-119`),
implemented with two `as_strided` aliasing views (`:111-140`).

**Backends.** The *attend* has Triton (`ops/attention/triton/dsa.py:449-474`, `:509`),
FlashMLA (`ops/attention/flash_mla/__init__.py:185`, `:274`, min arch 9.0) and
flashinfer-TRT-LLM (`ops/attention/flashinfer/__init__.py:289`, `:378`, min arch 10.0).
The *indexer top-k* has DeepGEMM (`ops/attention/deep_gemm/__init__.py:106, :155, :312`)
**and a full Triton implementation** (`ops/attention/triton/dsa_topk.py:1098, :1121, :1184`)
with no arch floor. See §2 for the mis-gate that currently makes DeepGEMM win on sm_120.

#### MSA (MiniMax sparse attention — MiniMax M2/M3)

Two-stage **block-sparse**: an indexer scores KV *blocks* of `index_block_size`, keeps
`index_topk_blocks` plus always-on `index_init_blocks` (attention sink) and
`index_local_blocks` (sliding window); the attend runs only over selected blocks.
Config at `configs/msa.py:39-61`; backend `MSAHybridAttnBackend` (946 lines,
`backends/msa.py:942`); per-layer index buffers keyed by layer id because MiniMax
interleaves dense and sparse layers (`kv_cache/msa.py:59-71`, layer typing at
`configs/msa.py:46-49`).

A complete Triton implementation exists for both decode and extend
(`ops/attention/triton/minimax_indexer.py`, `triton/minimax_sparse_attention.py`,
registered at `:656-666` and `:728-737`, `vendors={nvidia, amd}`, no arch floor).
The CuTe SM100 attend is registered *correctly bounded* — see §2.

Note: `kv_cache/msa.py:100-104` raises on `get_contiguous_buf_infos` — **MSA cannot be
P/D-disaggregated**.

#### NSA — not present

A full-tree grep for `NSA|native_sparse|nsa_` finds exactly two hits, both prose
pointers to SGLang's NSA as a *porting reference* (`docs/support-matrix.md:48` and its
Chinese mirror). There is no NSA kernel, op, config, backend or test in the fork.

### 1.2 MoE and expert parallelism

**The layer.** `MoELayer` (`python/tokenspeed/runtime/layers/moe/expert.py`) is a thin
façade with no kernel logic: it takes `(tp_rank, tp_size)` and `(ep_rank, ep_size)`
(`:100-108`), computes `num_local_experts = num_experts // ep_size` (`:113`), freezes an
immutable `MoELayerSpec` (`:116-129`), calls `tokenspeed_kernel.moe_plan(...)` at
**construction time** to pick the concrete kernel (`:187-201`), allocates weights shaped
by the selected solution (`:203-210`), and its `forward()` just dispatches to
`tokenspeed_kernel.moe_apply` (`:246-280`).

**A hard constraint worth knowing up front:**
```
expert.py:110-111    if tp_size > 1 and ep_size > 1: raise ValueError("Mixed TP and EP is not supported yet.")
```
mirrored in the server-args validator (`server_args.py:548-549`). **It is TP-only or
EP-only inside a MoE layer, never both.**

**Two orthogonal sharding axes.**
- **EP shards dim 0** (the expert dimension) of the fused expert tensors, as a contiguous
  per-rank range: `num_local_experts = num_experts // ep_size; start = num_local_experts * ep_rank`
  (`layers/moe/loader.py:62-104`, again at `:240-262`, `:398-411`). The code flags its own
  limitation: *"Expert ownership is assumed to be a contiguous per-rank range here.
  EPLB-aware remapping would need a different planning step"* (`loader.py:69-70`).
- **TP shards the intermediate dimension**: `ispp = intermediate_size // tp_size`;
  `w13_weight` is `[num_local_experts, 2*ispp, hidden]`, `w2_weight` is
  `[num_local_experts, hidden, ispp]` (`layers/moe/weights/unquant.py:32-56`, narrowing at
  `weights/loaders.py:41-113`).

**Where the communication is — and the surprise.** In the **default** EP mode
(`all2all_backend="none"`, `server_args.py:194`) **there is no all-to-all at all.** The
scheme is "EP-as-masked-TP":

```
SparseMoeBlock.forward                          models/qwen3_5_moe.py:259-269
 └─ _forward_tp                                  qwen3_5_moe.py:272-330
     ├─ gate(hidden) -> router_logits
     ├─ comm_manager.pre_mlp_comm  ── token_all_gather over mapping.moe.tp_ep_group
     │                                            distributed/comm_manager.py:216-229
     ├─ TopK.forward                              layers/moe/topk.py:383-412
     ├─ MoELayer.forward -> moe_apply             layers/moe/expert.py:246-280
     └─ comm_manager.post_mlp_fused ── all_reduce OR token_reduce_scatter
                                                  comm_manager.py:255-269
```

Every rank sees *all* tokens (all-gathered), runs only its local expert slice, and the
partials are summed by an all-reduce/reduce-scatter. The "drop tokens that aren't mine"
step lives **inside the kernel** — `_local_topk_for_ep` subtracts `ep_rank * num_local_experts`
and masks out-of-range ids to `-1` with weight 0 (`ops/moe/triton/mxfp4.py:258-276`, used
by `ops/moe/triton/fp8.py:297-301`).

**This is strategically important for us:** compute is sharded, communication volume is
not. It costs one all-gather + one all-reduce of full hidden states per MoE layer, but
it needs **no NVLink, no RDMA, no DeepEP** — plain NCCL. It is the PCIe-viable EP path.

**The real all-to-all (DeepEP)** is only reachable when `all2all_backend=deepep` **and**
`moe_backend=flashinfer_cutedsl` (`models/qwen3_5_moe.py:185-189`), routing through
`DeepEPDispatcher` (`ops/moe/flashinfer/cutedsl_deepep_nvfp4.py:135-229`,
`ops/communication/deep_ep.py:534-578`). Normal mode sizes both NVLink and RDMA buffers
(`deep_ep.py:128-155`); low-latency mode is RDMA/IBGDA-only (`:143-154`). Mode resolution
is normal-for-prefill / low-latency-for-decode (`deep_ep.py:90-96`).

Two dead ends found: `All2AllBackend.FLASHINFER_NVLINK_ONE_SIDED` is declared
(`layers/moe/utils.py:49`) but `_validate_a2a_backend` raises `NotImplementedError` for
anything but `none`/`deepep` (`ops/moe/__init__.py:48-51`); and `all2all_backend=deepep`
with `moe_backend=auto` sets `solution="flashinfer_cutedsl_deepep"` (`ops/moe/__init__.py:152-153`)
while the kernel registers as `solution="flashinfer_cutedsl"`
(`ops/moe/flashinfer/cutedsl_deepep_nvfp4.py:111`) — an exact-string mismatch that yields
`NoKernelFoundError`. The documented path works only because the model gate forces the
user to pass `--moe-backend flashinfer_cutedsl` explicitly.

**Routing / top-k** (`layers/moe/topk.py:446-600`) is a seven-branch ladder: Inkling
shared-expert sink (`:473`), DeepSeek group-limited grouped-top-k without bias
(`grouped_topk_gpu`, `:226-278` — per-group max as group score, `topk_group` groups kept,
then global top-k), grouped-top-k **with correction bias** via a Triton kernel (`:500-522`),
`torch_native` with the aux-loss-free DeepSeek-V3 formulation (selection uses `scores+bias`,
returned weight is the **unbiased** `scores`, `:206-213`), non-grouped biased via a CUDA
`routing_flash` kernel (`:546-567`), plain softmax top-k (`:568`), and a custom hook (`:584`).
Sigmoid/`RoutingMethodType` variants (`layers/moe/utils.py:33-42`) are executed *inside*
the kernel-routing backends, not by `select_experts`.

**EPLB (expert-parallel load balancing) — present, complete, and entirely unwired.**
`python/tokenspeed/runtime/moe/` is a faithful port of DeepSeek/SGLang EPLB:
- `eplb_algorithms/deepseek.py` (261 lines): `balanced_packing` greedy LPT bin packing
  (`:28-74`), `replicate_experts` iteratively replicating the max `load/replica_count`
  expert (`:77-108`), and `rebalance_experts_hierarchical` — pack expert *groups* onto
  nodes, replicate within a node, then pack onto GPUs within a node (`:111-205`), with
  the node dimension existing precisely because *"the intra-node network (e.g. NVLink)
  is faster"* (`:123`).
- `expert_location.py` (467 lines): the logical→physical map,
  `num_physical_experts = num_logical + ep_num_redundant_experts` (`:195-198`) — redundant
  replicas of hot experts are the actual balancing lever — plus a per-layer masked
  in-place `update()` (`:250-282`) which is the hook a *dynamic* rebalancer would use.
- `distribution_recorder.py` (868 lines): hooked at `layers/moe/topk.py:598`, modes
  `stat|stat_approx|per_pass|per_token`, circular buffer, balancedness metrics.
- Routing-time application: `topk_ids_logical_to_physical` (`topk.py:108-186`) with five
  dispatch algorithms (`topk.py:51-57`), fusible into the routing kernel (`:503-524`).

**But:** `compute_initial_expert_location_metadata` (`expert_location.py:436`) has
**zero call sites**; `ExpertDistributionRecorder.init_new` (`distribution_recorder.py:53`)
has zero call sites; `_global_expert_distribution_recorder` is initialised to a no-op with
no setter (`distribution_recorder.py:238-240`); `ExpertLocationDispatchInfo.init_new`
(`topk.py:66-93`) has zero call sites and every `select_experts` caller passes `None`;
`--enable-eplb` (`server_args.py:189, :1330-1333`) feeds only `global_server_args_dict`
and a validator. `ExpertLocationMetadata.update()` has zero callers. There are no tests.
A latent bug: `distribution_recorder.py:263` reads `server_args.enable_deepep_moe`, which
is not a field of `ServerArgs` — `stat_approx` mode would `AttributeError`.

**Verdict on EPLB: a complete, well-factored static rebalancing subsystem that was lifted
into the tree and never connected to the model runner.** Reactivating it is plumbing work,
not algorithm work — plus fixing the contiguous-range assumption at `loader.py:69-70`.
(`ep_num_redundant_experts` *is* live — it inflates `num_experts` at construction,
`models/deepseek_v3.py:301-302`, `qwen3_5_moe.py:214-215` — but without the location
metadata the extra slots get an identity map, so the replicas never function as replicas.)

**Models with MoE**: `deepseek_v3.py:300`, `deepseek_v4.py:2521`, `gpt_oss.py:340`,
`qwen3_5_moe.py:212`, `minimax_m2.py:144`, `minimax_m3.py:206`, `inkling.py:992`,
`longcat_flash.py:234`; `qwen3_moe.py:71` and `qwen3_omni.py:104-149` reuse the
Qwen3.5 block; `glm5.py:1230` inherits DeepSeek-V3's. **`gemma.py` is dense-only** —
`gemma-4-26B-A4B` is explicitly left as a follow-up (`gemma.py:27, :380-384`).

### 1.3 Parallelism

**One flat world, four independent sub-mappings.** `python/tokenspeed/runtime/distributed/mapping.py`
gives each layer *family* its own decomposition, all as `(size, stride)` views over the
same global rank ids (`_make_parallelism_rank`, `:51-53`):

| Sub-mapping | Dims (inner→outer) | file:line |
|---|---|---|
| `AttentionLayerMapping` | `tp`, `cp`, `dp` | `:133-195` |
| `DenseLayerMapping` | `tp`, `dp` | `:94-130` |
| `MoeLayerMapping` | `tp`, `ep`, `dp` (+ combined `tp_ep_group`, `:240-250`) | `:198-266` |
| `VisionTowerMapping` | `tp`, `dp` — world = `attn.tp_size`, not global | `:269-312` |

This is finer-grained than any reference engine (§3). CLI surface:
`--tensor-parallel-size/--tp` is a pure alias for `--attn-tp-size` (`server_args.py:1979-1987`);
`--dense-tp-size`, `--moe-tp-size`, `--ep-size`, `--data-parallel-size` are independent
(`server_args.py:294-298, :165, :170`). `--enable-expert-parallel` auto-sets
`ep_size = world_size` (`server_args.py:501-503`); `dense_tp_size` defaults to the **full**
world (`:494-497`).

**DP-attention** is real: each DP rank runs its own scheduler and its own KV pool, and
the comm pattern flips from all-reduce to reduce-scatter/all-gather — see §1.4.
Coordination is one CPU `all_gather_into_tensor` of `[num_tokens, batch_size, forward_mode]`
per step (`engine/event_loop.py:1493-1548`), producing `need_idle_forward` for empty ranks
(`:1528`), which then run a dummy/IDLE forward so collectives still match
(`execution/model_executor.py:1364-1424`). DP request routing has three policies —
`ROUND_ROBIN`, `SHORTEST_QUEUE` (default), `MINIMUM_CACHE_USAGE`
(`engine/data_parallel_controller.py:56-64`, default at `server_args.py:166`).

**Context parallelism is scaffolding, not a feature.** `attn_cp_size` exists in `Mapping`
and threads into `scatter_index` (`mapping.py:188-195`), and zigzag helpers exist
(`layers/utils.py:78-111`, called from `models/deepseek_v3.py:1433-1478`). But it is only
settable via env `ENABLE_CP` (`server_args.py:465-468`, no CLI flag),
**`CP_METADATA.set()` (`layers/utils.py:63`) has zero callers** so every CP branch is dead,
and no attention backend references `cp_metadata` — there is no ring/Ulysses kernel.

**Pipeline parallelism does not exist.** Exhaustive grep for
`pipeline_parallel|pp_size|pp_rank|1f1b|micro_batch` across `python/`, `tokenspeed-kernel/`,
`tokenspeed-scheduler/`, `docs/`, `test/` returns **zero** hits (all matches were under
`third_party/`). `Mapping` has no `pp` axis; `NcclBackend.send`/`recv` exist
(`comm_backend/nccl.py:169, :185`) but are not on the `CommBackend` ABC and have no
callers — leftover scaffolding. Multi-node here is TP/EP/DP over a flat world only.

**What changes at TP>1 in the forward.** Column/row-parallel linear layers shard as
expected (`layers/linear.py:309, :483, :731, :1084`; QKV replicates KV heads when
`tp_size >= total_num_kv_heads`, `:783-787`). Notably, all mainline decoders pass
`reduce_results=False` to `o_proj` and `down_proj` (`models/llama.py:94,176`;
`qwen2.py:84,159`; `gemma.py:184,289`; `qwen3_5.py:322,569,727,751`) — **the reduction is
hoisted out of the linear layer into `CommManager` so it can be fused with the RMSNorm**.

The AR-vs-RSAG switch is `use_all_reduce(is_moe)` (`distributed/comm_manager.py:128-131`):
uniform TP (`attn_tp == dense_tp == moe_tp_ep`, i.e. plain `--tp N`) gives **exactly two
all-reduces per layer** (`:174`, `:247`/`:263`); non-uniform (DP-attention) switches to
`token_reduce_scatter` after attention (`:186-190`) and `token_all_gather` before MoE
(`:226-230`).

Fused all-reduce + residual + RMSNorm exists behind `--enable-allreduce-fusion`
(default off, `server_args.py:317`), capped by `--comm-fusion-max-num-tokens`
(default 2048, `:316`), auto-enabled on Hopper+/AMD single-node with attn-TP and no
attn-DP (`server_args.py:625-633`).

### 1.4 Communication

**Interconnect detection** (`tokenspeed-kernel/python/tokenspeed_kernel/platform.py`):
`InterconnectInfo` with topology ∈ `{single_gpu, pcie, nvlink_pairs, nvlink_full, nvswitch}`
(`:81-88`); `_detect_cuda_interconnect` (`:508-522`) falls back to `"pcie"`;
`_detect_cuda_nvlink_topology` (`:609-659`) uses `pynvml.nvmlDeviceGetP2PStatus(..., NVML_P2P_CAPS_INDEX_NVLINK)`
over all pairs (`:639-641`). AMD gets xGMI detection from `/sys/class/kfd` (`:525-560`).
Runtime features `comms:nvlink`, `comms:nvlink_full`, `comms:symmetric_memory` are derived
(`:404-415`).

**Critical finding: interconnect detection is essentially decorative.** The only
production consumer of `current_platform().interconnect` in the entire tree is
`device_communicators/custom_all_reduce.py:127-130`. The `comms:*` runtime features are
computed and **never queried** — the kernel selector does not gate on them. Neither the
comm-backend chooser (`comm_backend/auto.py:93-100`, which picks by registration state
and tensor size only) nor the all2all-backend resolver (`layers/moe/utils.py:143-161`)
consults topology.

**Backend inventory** (`python/tokenspeed/runtime/distributed/comm_backend/`):

| Backend | file:line | Requirement |
|---|---|---|
| `NcclBackend` | `nccl.py:34` | none — PCIe-safe |
| `TrtllmAllReduceBackend` | `trtllm_allreduce.py:44` | CUDA IPC workspace (P2P); ≤2 MiB SUM; **degrades gracefully** |
| `TritonAllReduceBackend` | `triton_allreduce.py:37` | **AMD only** — `all_reduce_can_run` requires `platform.is_amd` (`ops/communication/triton.py:1680`) |
| `TritonRSAGBackend` | `triton_rsag.py:45` | **NVLink + NVSwitch multicast** on NVIDIA |
| `CustomAllReduceBackend` | `custom_allreduce.py:33` | NVLink-full above 2 GPUs — **dead code** |
| `PyNcclCommunicator` | `device_communicators/pynccl.py` | **dead code** |

`CustomAllreduce` and PyNccl are both unreachable: `initialize_comm_backend(use_pynccl, use_custom_allreduce)`
(`comm_backend/registry.py:31-45`) is **never called with arguments** anywhere in `python/` —
the only call site is `get_global_backend()` with defaults (`registry.py:48-53`).
`CustomAllReduceBackend.capture()` has no callers, so `_resources` is never populated and
`has_custom_ar()` never returns True. The famous "world_size > 2 and not full_nvlink"
check (`custom_all_reduce.py:134-140`) therefore **never runs in the current engine**.

**What `--disable-custom-all-reduce` actually gates** is the *other* thing with the same
name: the TRT-LLM Lamport one-shot all-reduce. `execution/distributed_initializer.py:179`
gates `trtllm_ar.configure_group(...)` on it, which creates a CUDA IPC workspace across
peers (`comm_backend/trtllm_allreduce.py:86-95`) and then routes every SUM ≤ 2 MiB through
it. Its `configure_group` is wrapped in a bare `except: return False`
(`trtllm_allreduce.py:109-111`) with a `"unavailable (NCCL fallback)"` log
(`distributed_initializer.py:202-206`), so it *does* degrade. On our single-GPU runs the
flag is simply skipping a pointless prewarm (`len(group) > 1` is false anyway).

**The genuine PCIe hazard.** `ops/communication/triton.py` implements reduce-scatter /
all-gather with NVLink **multicast** instructions — `multimem_ld_reduce_128` (`:108-141`,
used at `:949`) and `multimem_st_128` (used at `:989`), passing `symm_mem_hdl.multicast_ptr`
(`:1057`, `:1081`). `nvidia_rsag_reduce_scatter` (`:1093-1122`) and `nvidia_rsag_all_gather`
(`:1125-1155`) call these with **no capability check and no NCCL fallback**. Reachable from:

1. **DP-attention / split-TP** — every `token_all_gather`/`token_reduce_scatter` in
   `CommManager` (`comm_manager.py:143, 161, 179, 186, 213, 226, 249, 265, 279`).
2. **`logits_processor` at any TP>1** — `_init_all_gather_state` (`layers/logits_processor.py:288-307`)
   calls `create_state(...)` with **no try/except**, guarded only by `is_nvidia`,
   `tp_size > 1`, and `vocab_padded % (tp_size*8) == 0`; the gather uses
   `symm_mem_hdl.multicast_ptr` (`triton.py:1837`). **This is on the plain-TP hot path,
   not just DP-attention.** [code — I could not run it, but this is the most likely
   first failure at TP=2 on a PCIe box.]
3. `DpSamplingComm` — `_onesided_available` (`dp_sampling_comm.py:91-110`) checks only
   NVIDIA + torch version, **never the interconnect**, and auto-selects `onesided` (`:135`).
   It does have a documented NCCL fallback and an env override
   (`TOKENSPEED_DP_SAMPLING_BACKEND`, `:79-88`).

Also, the Triton fused all-reduce+residual+RMSNorm is **artificially AMD-gated**:
`allreduce_residual_rmsnorm` is wrapped in `if platform.is_amd:` (`ops/communication/triton.py:1531`)
and `allreduce_residual_rmsnorm_can_run` requires `platform.is_amd` (`:1495`). NVIDIA falls
through to the TRT-LLM path. There is a portable Triton implementation of this fusion in
the tree that NVIDIA is not allowed to use. (It needs PyTorch symmetric memory, so it is
not free on PCIe — but the vendor gate, not the fabric, is what excludes NVIDIA today.)

### 1.5 P/D disaggregation and EPD

**Roles.** Python `DisaggregationMode ∈ {NULL, PREFILL, DECODE, ENCODE}`
(`runtime/pd/utils.py:53-57`); C++ `enum class Role { kP, kD, kFused }`
(`tokenspeed-scheduler/csrc/utils.h:149-153`), bridged at
`runtime/engine/scheduler_utils.py:116-120` from `--disaggregation-mode`
(`server_args.py:306`). **P/D is a first-class C++ FSM domain**, not a Python bolt-on:
the initial state for P/D is `fsm::Bootstrapping` rather than `fsm::Submitted`
(`csrc/scheduler/request.cpp:34`), and there are 22 `Role` branch sites in the scheduler
(e.g. `kP` reserves 0 decode-input tokens and never schedules decodes,
`csrc/scheduler/operations/forward.cpp:993, :1018, :1026`; `kD` uses `PrefillSize()` as
its prefill budget, `:1005`; the flat L2 host sink is `kFused`-only, `csrc/scheduler/types.h:105`).

**Transport: Mooncake Transfer Engine, one-sided RDMA WRITE, prefill pushes to decode.**
`initialize(hostname, "P2PHANDSHAKE", "rdma", device_name)` — **the protocol string is
hardcoded to `"rdma"`** (`runtime/pd/base/mooncake_engine.py:74-79`), data path
`batch_transfer_sync_write` (`:118`). Buffers (KV and mamba state) are registered once as
RDMA MRs at manager construction (`pd/mooncake/conn.py:81-89`) — effectively GPUDirect RDMA.

**Layer-wise streaming is real and ON by default.** `--disaggregation-layerwise-interval`
defaults to **1** (`server_args.py:310`). `send_kvcache_layerwise`
(`pd/mooncake/prefill.py:582-664`) iterates layer groups and **blocks on a GPU-produced
step counter** before shipping each group (`_wait_until_cache_step`, `:603` + `:439-446`);
the counter is a device int64 mirrored to pinned host memory and read via
`ctypes.c_int64.from_address` (`pd/utils.py:433-477`) so the transfer thread polls without
a CUDA sync. It is driven from `prepare_prefill` **before** the forward
(`pd/prefill_executor.py:195-221`, called at `event_loop.py:968`), so KV streams out while
later layers are still computing. Mamba/GDN state is interleaved per layer group (`prefill.py:648-656`).

**`transfer_plan.py` (467 lines) is a declarative layout negotiator** — the nicest piece of
design in the P/D stack. `BufferLayout` describes a buffer's logical axis
(`kv_head`/`state_channel`/`replicated`), logical size, `item_stride_bytes`, and
`tp_replica_group_size` (GQA: several prefill TP ranks hold the same KV head)
(`transfer_plan.py:63`). `plan_for_decode_rank` (`:221`) emits either an `"identity"` plan
when prefill TP == decode TP (`:229-246`) or a `"fragmented"` plan that intersects logical
head intervals and emits per-fragment byte offsets (`:248-330`) — i.e. **heterogeneous
prefill/decode TP is supported**. Plans are versioned and shipped over ZMQ
(`:110-171`, `TRANSFER_PLAN_PROTOCOL_VERSION = 1`). Caveat: **MLA does not use the planner** —
it takes `_legacy_mla_route_plan` (`pd/mooncake/receiver.py:350`, `_calc` at `:398`).

**Handshake:** an aiohttp bootstrap server on `--disaggregation-bootstrap-port` (default
8998) publishing `{role, world_size, dp_size, rank_ip, rank_port, engine_rank, kv_item_lens, …}`
(`pd/base/bootstrap.py:42-195`, prefill registration `pd/mooncake/prefill.py:1059-1105`);
a per-request `bootstrap_room` integer that keys the whole status FSM and selects the DP
group (`target_dp_group = bootstrap_room % prefill_dp_size`, `receiver.py:465`); a ZMQ
control plane for destination-pointer registration and per-request pre-alloc
(`entities.py:118-215`); and a monotone `TransferPoll` FSM
(`Failed < Bootstrapping < Bootstrapped < WaitingForInput < Transferring < Success`,
`pd/base/status.py:22-28`) MIN-all-reduced across the gloo TP group (`pd/utils.py:88-106`).
The **first sampled token travels over ZMQ/TCP, not RDMA** (`prefill.py:666-692`, `entities.py:85-86`).

**EPD (encode-prefill-decode)** splits the **vision tower** onto its own LM-free GPU —
*"The encode worker is LM-free: it runs only the vision tower and ships image embeddings"*
(`test/ci_system/serve_qwen35_122b_nvfp4_epd_1e1p2d.sh:8`). Encode mode disables prefix
caching, the KV pool, and kvstore entirely (`server_args.py:655-664, :681-686`). Patch-budget
batcher with cross-TP-deterministic ordering (`epd/encode_scheduler.py:56-113`, ordering at
`:92-94` — mismatched batches would deadlock NCCL). Pixels move by POSIX SHM
(`epd/encode_worker.py:100-112`); embeddings move by the same Mooncake RDMA path
(`epd/mooncake/encode.py:392-414`), staged in a registered ring with CUDA-event fencing
(`epd/entities.py:115-119`). Prefill side is a poll-driven admission state machine that
must never block the event loop (`epd/prefill_admission.py`, 49 KB), with a lifetime-stable
pre-registered MR pool and **deliberately deferred deregistration** to dodge
`local access violation work queue error` (`:78-84`). `runtime/cache/embedding_cache.py`
adds a GPU→host tiered LRU keyed on `MultimodalDataItem.hash` (`:40`, `:140`, demotion `:205`).

**Both are integration-tested** — `test/runtime/distributed/test_qwen35_pd_1p1d.py` +
`test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh` (Qwen3.5-397B NVFP4, fp8 KV, MTP,
4 GPUs, quality assertions), and `test_qwen35_epd_1e1p2d.py` + its serve script. The
density of hard-won correctness comments (`decode_executor.py:220-230` on a CUDA illegal
access; `epd/mooncake/encode.py:406-411` on the GIL; `prefill.py:368-375` on a hybrid-model
layer-index bug) is strong evidence this has genuinely been run in production.

**Dormant here:** `mooncake_async` (`pd/mooncake/async_conn.py`, a valid CLI choice with a
full 24 KB implementation and **zero references in tests or CI**), and the L3 storage tier.

### 1.6 Memory and capacity — how HBM becomes concurrency

**Four tiers**, coordinated by `MemoryExecutor` (`runtime/cache/executor/memory_executor.py:114`):

| Tier | Implementation | Default |
|---|---|---|
| L1 GPU | `MHA/MLA/DSA/MSATokenToKVPool` (`runtime/layers/attention/kv_cache/`) | always |
| L2 host RAM | `HostKVCache` (`runtime/cache/kv_cache_host.py:128`) + `HostExecutor` | **ON** |
| L2 mamba/GDN state | `mamba_cache_host.py` ("[mamba_l2]") | OFF (`server_args.py:116`) |
| L3 remote/disk | `StorageExecutor` over `KVStoreStorage` (`cache/kvstore_storage.py:50`) | OFF |

**The single most surprising default in the tree: host KV offload is ON.**
`enable_kvstore` is declared `False` (`server_args.py:199`) but `_handle_kvstore` sets it
`True` unless `--disable-kvstore` (`server_args.py:687-688`), at `--kvstore-ratio 2.0`
(`:200`) — i.e. the host pool is **twice the entire device KV pool in tokens**
(`cache/kv_cache_host.py:154`). This is exactly the 139 GB startup allocation that
OOM-killed E17 (log line at `kv_cache_host.py:179-187`). The upstream's own CI sets
`KVSTORE_RATIO=0.5` (`test/ci_system/serve_qwen35_397b_nvfp4_pd_1p1d.sh:17`), and every
big-model recipe in `docs/recipes/models.md` passes `--disable-kvstore`.
**We have been running a hierarchical KV cache in every benchmark since E4 and have never
measured its hit rate or its benefit.** [code + E17]

**The hierarchy is one radix tree with two resource types**, not a separate HiRadixTree:
`TreeNode` carries both a `DeviceResource` and a `HostResource`
(`tokenspeed-scheduler/csrc/resource/radix_tree/tree_node.h:157-158`), selected by
`ResourceType::{Device,Host}` (`csrc/resource/types.h:39-42`), with per-type `NodeRef`
locks that walk to the root (`tree_node.h:166-200`). L3 presence is a per-node
`storage_persisted_` flag (`:154`). Demotion is `Scheduler::newWriteBackOperation`
(`csrc/scheduler/scheduler.cpp:331-346`); promotion is `HostExecutor.enqueue_loadback`
(`cache/executor/host_executor.py:200`); L3 prefetch fires only above
`prefetch_threshold`, **hard-coded to 4** in Python (`event_loop.py:457`).
The L2 tier also backs retraction — the warning at `event_loop.py:343-345` says it
*"will still be used when retraction happens"* even when disabled.

**The capacity formula — how HBM becomes tokens becomes concurrency.**

```
# Step 1 — available bytes                    runtime/layers/attention/utils.py:120-142
cache_memory = available_gpu_memory − total_gpu_memory × (1 − gpu_memory_utilization)

# Step 2 — bytes per token                    configs/{mha,mla,dsa,msa}.py cache_cell_size()
cell_size  = attn_config.cache_cell_size() × num_attention_layers      # utils.py:168
cell_size += draft_attn_config.cache_cell_size() × draft_num_layers    # utils.py:173-175

# Step 3 — pages                              utils.py:180-181
max_num_token = cache_memory // cell_size
max_num_pages = ceil(max_num_token / page_size)

# Step 3b — hybrid GDN split                  utils.py:185-223
kv_memory  = total / (1 + mamba_ratio)                                 # mamba_full_memory_ratio, default 0.9
mamba_pool_total_chunks = mamba_memory // memory_per_mamba_chunk

# Step 4 — running-batch cap (E25 default)    engine/event_loop.py:113-140
reserve_tokens = context_len // adaptive_batch_reserve_divisor         # divisor default 4
kv_capacity    = max_total_num_tokens // reserve_tokens
max_num_seqs   = round_up_32(max(current, min(kv_capacity, adaptive_batch_max)))   # ceiling 1024
```

Two details that matter for big models: `num_attention_layers` counts **only the
full-attention layers** for hybrid models (`registry.py:866-873, :904-911`), and there is a
`_kv_profile_layer_divisor` that charges sliding-window layers less (`registry.py:919-923`).
Per-request GDN recurrent state is `mamba_cache_per_req`
(`runtime/configs/qwen3_5_text_base_config.py:353-364`), with shapes from
`mamba2_cache_params` (`:312-351`) and the SSM dtype from `TOKENSPEED_MAMBA_SSM_DTYPE`
(default **`"float32"`**, `runtime/utils/env.py:291`; `bfloat16` is also accepted, `:341-344`).

**KV quantization.** `--kv-cache-dtype ∈ {auto, fp8, fp8_e4m3, mxfp8}`
(`server_args.py:855-862`), `--kv-cache-quant-method ∈ {none, per_token_head}` (`:864-869`);
`auto` resolves to **bf16**, not the model dtype (`configs/base.py:32-43`). fp8 pools store
as `uint8` because `index_put` is not implemented for fp8 (`kv_cache/base.py:58-62`).
Support by backend: `trtllm`, `mha`, `mla`, `trtllm_mla`, `flashmla`, `dsa`, `msa` all take
fp8_e4m3; `tokenspeed_mla` **requires** it (`backends/tokenspeed_mla.py:166-171`); `mha`
alone adds mxfp8 (needs `--block-size 128`, `configs/mha.py:186-189`). DSA fp8 KV is
hardware-gated to `is_blackwell_plus or is_cdna4_plus` (`configs/dsa.py:62-68`).
`e5m2` is **not selectable from the CLI**.

### 1.7 Quantization at scale

Nine modules in `python/tokenspeed/runtime/layers/quantization/`: `fp8.py`,
`w8a8_fp8.py`, `nvfp4.py`, `mxfp4.py`, `modelopt_mixed.py`, `compressed_tensors/`.
MoE weight kinds are fp8 / nvfp4 / mxfp4 / **mxint4** / unquant
(`python/tokenspeed/runtime/layers/moe/weights/`).

**Two distinct fp8 paths, and this is the crux of E23:**

- **Block-wise fp8 (DeepSeek-style, `weight_block_size = [128, 128]`)**, detected from the
  checkpoint (`layers/quantization/fp8.py:47-70, :95-101`), dispatches with
  `quant="mxfp8", block_size=[128,128]` (`layers/dense/fp8.py:369-392`) to one of three
  kernels: `deep_gemm_mm_fp8_blockscale` (datacenter, `ops/gemm/deep_gemm.py:57-73`),
  `flashinfer_mm_fp8_blockscale` (`ops/gemm/flashinfer.py:91-107`, min 10.0 **no max**),
  or **`triton_mm_fp8_blockscale` — `vendors={amd, nvidia}` with NO arch bound at all**
  (`ops/gemm/triton.py:782-795`). There is even an **explicit sm_120 special case**: at
  `arch_version == ArchVersion(12, 0)` the activation quantizer is forced to Triton
  (`ops/gemm/__init__.py:198-211`). **So the DeepSeek group-128 scheme is portable and
  already sm_120-aware — only the `deep_gemm` *kernel* is datacenter.** [code]
- **Per-channel dynamic fp8** (what `--quantization fp8` does to a bf16 checkpoint):
  `per_token_group_quant_fp8(weight, weight.shape[-1])` → one scale per output row, on a
  **portable Triton** kernel, then `triton_mm_fp8_scaled` (`layers/dense/fp8.py:305-320`).
  This is the path E23 fixed and validated: batch-1 77→110 t/s (+42%), C=128 peak
  2731→3519 (+29%). The code comments now record the reasoning
  (*"the TRT-LLM 1x128 kernel only supports group_size=128"*, `dense/fp8.py:307-310`).
  **[validated — E23]**

MoE kernel matrix (all registrations from `tokenspeed-kernel/.../ops/moe/`):

| Solution | Quant | Vendor | min arch | max arch |
|---|---|---|---|---|
| `flashinfer/cutlass_unquant.py:50-77` | **bf16/fp16 unquant** | nvidia | **8.9** | — |
| `flashinfer/cutlass_fp8.py:55-83` | fp8 block (128,128) | nvidia | 9.0 | — |
| `flashinfer/cutlass_nvfp4.py:134-161` | nvfp4 | nvidia | 10.0 | — |
| `flashinfer/cutedsl_deepep_nvfp4.py:107-134` | nvfp4 + DeepEP | nvidia | 10.0 | — |
| `flashinfer/trtllm_fp8.py:73-101` | fp8 block | nvidia | 10.0 | — |
| `flashinfer/trtllm_nvfp4.py:361, :413` | nvfp4 | nvidia | 10.0 | **10.3** |
| `flashinfer/trtllm_unquant.py:207, :258` | bf16 | nvidia | 10.0 | **10.3** |
| `flashinfer/trtllm_mxfp4.py:342-370` | mxfp4 (bias) | nvidia | 10.0 | **10.3** |
| `flashinfer/trtllm_mxint4.py:206-235` | **mxint4** | nvidia | 10.0 | **10.3** |
| `triton/fp8.py:197-222` | fp8 block **(1,32)** = mxfp8 | **nvidia** | — | — |
| `triton/fp8.py:223-274` | fp8 block (128,128), ±EP | **amd** | — | — |
| `triton/mxfp4.py:357-406` | mxfp4, ±EP | **amd** | — | — |
| `triton/mxfp4.py:407-430` | mxfp4 | **none — any vendor, any arch** | — | — |
| `gluon/bf16.py:38-68` | **bf16 unquant** | amd | 9.5 | 9.5 (gfx950) |
| `gluon/mxfp4.py` (4 regs) | mxfp4 | amd | 9.5 / 12.5 | same |

**The E6/E14 gap in one line: there is no Triton unquantized/bf16 MoE kernel anywhere.**
`grep unquant ops/moe/triton/` returns nothing. bf16 MoE on NVIDIA has exactly one
implementation — flashinfer cutlass — and the portable fallback that would make it safe
does not exist. A bf16 fused-expert kernel *does* exist in-tree, but it is
`gluon/bf16.py`, gated `vendors={"amd"}, arch [9.5, 9.5]` (gfx950 only).

**On the E14 one-line fix.** *State note: the fix landed in the working tree (uncommitted)
while this study was being written — `cutlass_unquant.py:93-97` now casts to
`torch.float32` with an explanatory comment. What follows is the verification, and the
two siblings that are still unfixed.*

The bug was that `ops/moe/flashinfer/cutlass_unquant.py:96` cast
`topk_weights` to `x.dtype` before passing it as `token_final_scales`. Verified against
the TRT-LLM C++ this binds to: `CHECK_INPUT(token_final_scales.value(), at::ScalarType::Float)`
(`third_party/TensorRT-LLM/cpp/tensorrt_llm/thop/moeOp.cpp:415`), consumed as
`float const*` (`moeOp.cpp:660`, `moe_kernels.h:534`, `moe_kernels.cu:280`). **The fix is
confirmed correct.** Two refinements: (a) the bad cast is in the `topk_weights is None`
fallback branch, so it bites the *in-kernel-routing* path — the full-model path supplies
precomputed fp32 top-k (`layers/moe/expert.py:269-277`, and every `select_experts` branch
returns fp32: `topk.py:278, :556, :218`); (b) **the identical latent cast still exists in
`cutlass_fp8.py:102` and `cutlass_nvfp4.py:180`** (both verified present at the time of
writing, both feeding `token_final_scales=` at `:108` and `:192`) — fix those two as well.

**nvfp4 / mxfp4 — the finding that matters most here.** sm_120 *has* FP4 tensor cores, and
`platform.py:398-399` correctly grants it `tensor_core:f4`. The nvfp4 **GEMM** is fine:
`cublaslt_mm_nvfp4` (`ops/gemm/trtllm.py:78-90`) registers `min_arch (10,0)`, **no max**,
at the highest nvfp4 priority, with no `is_blackwell` import guard. What breaks is the
**activation quantizer**: `layers/dense/nvfp4.py:25` imports `fp4_quantize` from
`ops/quantization/flashinfer.py`, where the real symbol is bound inside
`if platform.is_nvidia and platform.is_blackwell:` (`:68-70`) — and `is_blackwell` means
`major == 10` **exactly** (`platform.py:144-145`). On sm_120 it stays `error_fn`, so
`Nvfp4LinearMethod.apply` (`dense/nvfp4.py:195-197`) raises
`"Kernel implementation not found"` on the first forward. The same `is_blackwell` import
gate hides `mm_fp4` (`ops/gemm/flashinfer.py:287`). **This is a one-word gating bug
(`is_blackwell` → `is_blackwell_plus`), not a hardware wall** — though whether the
flashinfer binaries are built for `sm_120a` is a separate question the repo never checks.
[code — untested]

**Dense mxfp4 has no NVIDIA kernel at all**, on sm_100 either: the only `mm` kernel is
`triton_mm_mxfp4`, `vendors={"amd"}` (`ops/gemm/triton.py:896-906`), and its quantizer
`triton_quantize_mxfp4` is likewise AMD-only (`ops/quantization/triton.py:517-522`).
This matches E3's finding that `quantization/mxfp4` had *no* candidate on sm_120.
Also hard-locked: `nvfp4_gemm_swiglu_nvfp4_quant` raises on `major != 10`
(`ops/gemm/cute_dsl.py:165-169`); it is called from `models/qwen3_5_moe.py:147-153` and
`models/deepseek_v3.py:204-208`, but **only under `_use_nvfp4_gemm_swiglu_nvfp4_quant`**,
i.e. nvfp4 checkpoints — so it is *not* on our bf16 Qwen3.6 path. [code — verified]

**Two latent bugs worth fixing while we are here.** (1) `--quantization w8a8_fp8`
**reproduces both E23 defects, unfixed**: `layers/dense/w8a8_fp8.py:66` does
`weight_scale.t().contiguous()` yielding `[1, N]` (E23 part b), and `:132-138` calls
`mm(...)` **without `quant="fp8"`** so the signature matches no kernel (E23 part c); it
also uses the TRT-LLM `per_token_quant_fp8` that E23 found returns malformed scales on
sm_120. (2) `layers/quantization/compressed_tensors/compressed_tensors.py:363-383`
references `CompressedTensorsW8A8Fp8`/`CompressedTensorsW8A16Fp8`, which **are not
defined anywhere** — `schemes/__init__.py:21-33` exports only `CompressedTensorsWNA16` —
silenced by `# ruff: noqa: F821` at `:59`. Any compressed-tensors FP8 checkpoint routed
through that config will `NameError` at layer construction.

DeepGEMM is an external pip dependency (`tokenspeed-deepgemm==2.5.0.post20260629`,
`tokenspeed-kernel/python/requirements/cuda-thirdparty.txt:2`), reached through a
3-file shim (`thirdparty/deep_gemm/__init__.py:27-90`) and **absent from
`rocm-thirdparty.txt`**. Its MoE use is DeepSeek-V4 MegaMoE only, which self-gates:
```
models/deepseek_v4.py:2320-2322   if torch.cuda.get_device_capability(device)[0] != 10:
                                      raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
```
Note `!= 10` — **exactly** SM100, excluding both sm_90 and sm_120. It further requires
`moe.ep_size > 1`, forbids `moe.tp_size != 1` and forbids redundant experts
(`deepseek_v4.py:2463-2481`), and uses an NVSHMEM-style symmetric buffer (`:2368-2392`).

### 1.8 Speculative decoding / MTP

Three drafters behind a clean `BaseDrafter` ABC: `eagle.py` (565 lines), `mtp.py` (1096),
`dflash.py` (1036) + `_dflash_fused_kv.py`, in
`python/tokenspeed/runtime/execution/drafter/`. Dispatch:
```
model_executor.py:93   DRAFTER_MAPPING = {"EAGLE3": Eagle, "MTP": Eagle, "DFLASH": DFlash}
model_executor.py:96-101  # "MTP" covers two algorithms: Eagle-like MTP (DeepSeek) -> Eagle;
                          #  vanilla multi-layer MTP (Inkling) -> Mtp
```
CLI choices are `{EAGLE3, MTP, DFLASH}` (`server_args.py:1576`).

**MTP needs no separate draft checkpoint**: with `--speculative-algorithm MTP` and no
`--speculative-draft-model-path`, the engine uses the base model
(`server_args.py:582-589`). The NextN model classes are registered for our on-disk
models — `Qwen3_5ForConditionalGenerationNextN` and `Qwen3_5MoeForConditionalGenerationNextN`
(`runtime/models/qwen3_5_nextn.py:152, :403, EntryClass :416`; also listed in
`layers/attention/registry.py:119-124`) — plus `deepseek_nextn.py`, `glm5_nextn.py`,
`inkling_nextn.py`, `deepseek_v4_mtp.py`.

**A hard limitation, asserted at startup:**
```
server_args.py:615-620   speculative_eagle_topk > 1 (tree spec) is not currently supported.
                         Only chain spec (topk=1) is wired end-to-end.
```
So this is **chain speculation, not tree speculation** — no `build_tree_kernel`, no
tree-attention verify.

**Verification is exact rejection sampling, not relaxed acceptance.** Greedy uses a
`verify_chain_greedy` CUDA kernel with a pure-torch fallback
(`sampling/backends/greedy.py:86-118`, fallback `:47`); stochastic uses
`chain_speculative_sampling_target_only` (`backends/flashinfer.py:520-533`). Both
acceptance thresholds are **1.0** — `SPECULATIVE_ACCEPT_THRESHOLD_SINGLE` and
`..._ACC` at `sampling/backends/base.py:40-41` — i.e. lossless, with no "typical
acceptance" relaxation. A portable Triton verifier exists
(`ops/sampling/triton/verify.py:34-130`). There is a strictly-positive coin floor with the
reason spelled out (`sampling/utils.py:11-24`).

**Rejected draft KV is never evicted — the rollback is logical.** All `bs × spec_num_tokens`
candidate rows get real KV slots; commit is simply `valid_cache_lengths += accept_lengths`
(`execution/runtime_states.py:66-69`, driven from `model_executor.py:1049-1058`), and the
next step overwrites the uncounted tail. Attention metadata is explicitly kept on the
accepted prefix so stale draft KV is never read (`drafter/eagle.py:355-364`, comment at
`:356-357`; again at `:412-418`).

**The CUDA graph captures the whole speculative step, not just the target forward.**
`CudaGraphWrapper._forward_func` is bound to `ModelExecutor._forward_step`
(`execution/cuda_graph_wrapper.py:307`), and `_forward_step` (`model_executor.py:944-1021`)
runs target forward → `get_candidates` → `verify` → multi-step draft → next-input write,
all inside `torch.cuda.graph` (`cuda_graph_wrapper.py:515-516`). Capture forces
`CaptureHiddenMode.FULL` whenever a drafter exists (`:410-413`).

Scheduler interaction: `decode_input_tokens = speculative_num_draft_tokens`
(`event_loop.py:219-223`) multiplies every KV reservation and every `input_length`; the
mamba chunk sizing multiplies `per_layer_mamba_chunk_memory` by `(1 + speculative_num_draft_tokens)`
(`registry.py:895-901`) — i.e. **MTP over hybrid-GDN models is anticipated in the sizing
code**. The upstream recipe notes MTP forces the non-overlap scheduler
(`docs/recipes/models.md`, DeepSeek V4 MTP section). Also relevant: the Triton MLA decode
kernel declares `traits={"q_len": frozenset({1})}` (`ops/attention/triton/mla_decode.py:261-264`)
— **it cannot verify multiple draft tokens**, so MTP over MLA on the portable path is blocked.

**Maturity, honestly:** the drafters are wired to seven models and have a dozen structural
tests (`test/runtime/test_drafter_accept_indexing.py`, `test_inkling_mtp_e2e.py`,
`test_deepseek_v4_mtp_prefix_cache.py`, `test_mamba_mtp_state_reset.py`, …) plus one eval
config. But **there is no on-device speculative-decode throughput experiment anywhere in
`docs/experiments-log.md`** — E22 and E26 both *named* MTP as the top remaining lever and
neither of us has ever run it.

### 1.9 Scheduling at scale

The C++ FSM scheduler (`tokenspeed-scheduler/`, ~33k lines). Request states:
`Submitted → Prefetching → Prefilling ⟳ → PrefillDone → Decoding → Draining → WritingBack → Finished`,
with a preemption branch `Decoding|PrefillDone → Retracting → Retracted → Decoding`
(`csrc/fsm/forward_states.h:66-474`). **KV ownership *is* the state object**: every state
is an RAII bundle holding a `DeviceNodeRef` (pinning the shared radix prefix) plus a
`LocalKVAllocator` owning the request-exclusive tail (`forward_states.h:98`, `:170`), so
pages are freed exactly when a state dies, and copy constructors are deleted (`:84-85`) —
double-free of KV pages is a **compile error**.

All batch-formation policy is one function, `Scheduler::newForwardOperation`
(`csrc/scheduler/operations/forward.cpp:947`).

| Mechanism | Default | Evidence |
|---|---|---|
| Chunked prefill | **ON**, 8192 | `server_args.py:107, :413-414` → `max_scheduled_tokens` (`event_loop.py:451`); budget shared across prefills at `forward.cpp:965-975`, chunk at `:294`/`:402` |
| **Mixed prefill+decode** | **OFF** | `server_args.py:109`; when off, `if (!enable_mixed_prefill_decode && pushed_prefill) break;` at `forward.cpp:1020, :1027` — **any step that schedules a prefill chunk runs zero decodes** |
| Overlap scheduling | **ON** | `server_args.py:269`; loop at `event_loop.py:1770-1915` — step N's forward is launched (`:1883`) before step N−1's results are committed (`:1893`); reservation widened to `(depth+1) × decode_input_tokens` (`forward.cpp:68-79`) |
| Adaptive KV-derived batch | **ON** (E25) | `server_args.py:98`; `derive_adaptive_max_batch` at `event_loop.py:113-140`, applied `:260-289` incl. raising the graph cap |
| Retraction | reactive | fires only when `ops.empty() && !candidates.empty()` (`forward.cpp:1046`); victim = **longest** `TokenSize()` (`:1055-1057`); one victim per step; KV written back to host and restored via loadback, **not recomputed** (`forward.cpp:499-585, :587-621`) |
| Prefill CUDA graph | **ON** | `server_args.py:275`; token-bucketed breakable graph, `execution/prefill_graph.py:76-142` — *"the `torch.compile`-free alternative to piecewise CUDA graphs"* (`breakable_cuda_graph.py:30`) |
| Prefix caching | **ON** | `server_args.py:261` |

**Admission math** (`forward.cpp:271-396`):
```
unscheduled       = PrefillSize − max(device_matched, host_matched) × block_size    :291
tokens_this_round = min(token_budget, unscheduled)                                   :294
num_tokens        = loadback_tokens + tokens_this_round + decode_input_tokens        :302
pages_needed      = ceil(num_tokens / block_size)                                    :303
admit iff kv_prefix_cache_.EnsureCapacityByEvict<Device>(pages_needed)               :308
```
**The only decode headroom reservation is `+ decode_input_tokens`** (1, or
`speculative_num_draft_tokens`). There is no `new_token_ratio`, no per-running-seq
reserve, and no reservation decay after a retraction — exactly gap #4 in
`docs/design/scheduling-improvements.md:44-52`. `SchedulerConfig::num_pages_reserved_for_retracted_or_running`
(`csrc/scheduler/types.h:124`) is declared and never read.

**There is no `schedule_policy` at all.** Grep for `schedule_policy|lpm|longest_prefix|dfs_weight|fcfs`
over `python/` and `csrc/` returns nothing; `RequestSpec` carries no arrival timestamp
(`csrc/scheduler/request_spec.h:30-35`). Ordering is state-priority then **lexicographic
request id** (`forward.cpp:948-962`), chosen for TP determinism (*"a rank-varying op
deadlocks NCCL"*, `:958`). There is no priority scheduling and no SLO awareness — which is
precisely what `unieinfra-architecture.md:57,70` scopes as our work.

**TP synchronisation is by construction, not by contract.** Every attn-TP rank runs its
**own mirrored C++ scheduler**; no batch is ever broadcast (`event_loop.py:329-332`).
Sync rests on identical inputs (`request_handler.py:180-186`), identical config, and the
deterministic id tie-break. Async cache-op completions are TP-intersected before being
applied (`scheduler_utils.py:275-296`). There is **no per-step assert that the ranks agreed**.

**Flat KV-cache scheduler is compiled out.** `scheduler_ext_flat_kvcache()` reads a
build-time attribute (`runtime/configs/paged_cache_spec.py:43-53`), and E24 confirmed it is
OFF in our build. That means the flat reserve ledger, `flatFreeBudget`, wedge detection,
two-round starvation escalation, and OOM terminalisation (`forward.cpp:167-268`) — roughly
40% of the admission logic — **does not run**.

---

## 2. Datacenter-gated vs portable

**The systemic defect first, because it explains most of the rest.**
`ArchVersion` has **no notion of tcgen05 / TMEM / cluster-capable Blackwell versus consumer
Blackwell**. `_get_cuda_sm_features` (`tokenspeed-kernel/.../platform.py:382-401`) grants
`memory:tma` + `compute:cluster` at `>= (9,0)` and `tensor_core:f4` at `>= (10,0)`, so
sm_120 (12.0) is advertised **identically to B200** — including `compute:cluster`, which it
does not have. `is_blackwell_plus` is `>= (10,0)` (`platform.py:175-177`) with tuple
comparison (`:64-65`), so **every `min_arch_version=ArchVersion(10,0)` or `(9,0)` guard in
the tree is satisfied on sm_120.** Only two sites in the entire attention corpus use
`max_arch_version` (`ops/attention/msa.py:97`, `ops/attention/gluon/__init__.py:89`).
This was confirmed on-device in E1 and scoped as Steps A/B in
`docs/design/tokenspeed-sm120-support.md:73, :85, :116-120` — **it is still not implemented.**

Consequently there are three failure classes, and only the second is a real wall:

- **(i) Correctly gated → falls back cleanly.** MSA CuTe attend (`min=(10,0)`,
  `max=(10,3)`, plus a module-level `is_blackwell` meaning `major == 10` exactly,
  `ops/attention/msa.py:64, :90-99`) — the reference pattern. All five `trtllm_*` MoE
  kernels (`max=(10,3)`). The whole flashinfer-Blackwell attention block
  (`ops/attention/flashinfer/__init__.py:65`).
- **(ii) Genuinely arch-locked kernel → must be rewritten, not recompiled.**
- **(iii) Mis-gated → wins selection on sm_120, then crashes at runtime.** This is a bug
  class, not a hardware limit.

### The split, mechanism by mechanism

| Mechanism | Algorithm portable? | Kernel status on sm_120 | Verdict |
|---|---|---|---|
| **MLA** | ✅ Already portable — absorption is two `torch.bmm` (`deepseek_v3.py:805, :930`), cache is a plain tensor (`kv_cache/mla.py:100`), and **Triton prefill+decode register with no arch floor** (`triton/mla_prefill.py:274`, `triton/mla_decode.py:253`) | `tokenspeed-mla/` is **irrecoverably locked**: unconditional `tcgen05.CtaGroup.TWO` (`mla_decode_fp16.py:500`), hand-emitted `nvvm.tcgen05_mma_ws` (`mla_decode_fp8.py:109-131`), TMEM allocation (`:1424`, `mla_decode_fp16.py:1179`), and a softmax row-max with **only** sm_100/sm_103 branches (`mla_decode_fp16.py:2703, :2736`). Prefill raises `NotImplementedError` by construction for any other arch (`mla_prefill.py:91-101`) | **Portable path exists.** Work: guard the `tokenspeed_mla` backend (it has none); lift `q_len:{1}` on Triton MLA decode for MTP; validate numerics |
| **DSA** | ✅ Fully — the indexer is linear projections + LayerNorm + RoPE + weighted dot + top-k (`glm5.py:169-269`), and **Triton implements all three top-k ops plus attend, no arch floor** (`triton/dsa_topk.py:1098, :1121, :1184`; `triton/dsa.py:449, :509`) | **Mis-gated (class iii).** `deep_gemm` DSA registers `min_arch=(9,0)` with **no max** (`ops/attention/deep_gemm/__init__.py:112, :161, :318`) → sm_120 matches, outranks Triton on priority (`PERFORMANT` > `PORTABLE`), gets selected, and fails inside DeepGEMM. **Measured in E3/M0b**: the 3 datacenter winners on sm_120 were exactly `dsa_decode_topk`, `dsa_plan`, `dsa_prefill_topk` | **Smallest, highest-value fix in the tree.** One `max_arch_version` per site |
| **MSA** | ✅ Full Triton path for decode *and* extend (`triton/minimax_sparse_attention.py:656, :728`) | Correctly gated → clean fallback. Upstream itself declined to register the CuTe decode (`msa.py:9-12`) | **~Nothing to do for correctness**; perf only |
| **MoE — EP scheme** | ✅ The default all-gather + masked-local-experts + all-reduce scheme is **pure NCCL** (`comm_manager.py:216-269`, `triton/mxfp4.py:258-276`) | Works on PCIe | **The portable-technique win.** This is how we do EP without NVLink |
| **MoE — bf16 unquant kernel** | ⚠️ Portable in principle; **no Triton implementation exists** | Only `flashinfer/cutlass_unquant.py` (min 8.9, no max) matches sm_120 — the E14 path (fixed in the working tree; `cutlass_fp8.py:102` and `cutlass_nvfp4.py:180` still carry the same cast) | Finish the sibling fixes; **write the Triton fallback** so bf16 MoE stops having a single implementation |
| **MoE — trtllm family** | — | **Hard-blocked**: all five capped at `max=(10,3)`. Removes the entire kernel-routing family; mxfp4 and mxint4 lose their only NVIDIA implementation | Datacenter-only |
| **MoE — MegaMoE** | — | **Hard-blocked**: `capability[0] != 10 → raise` (`deepseek_v4.py:2320-2322`), plus NVLink symmetric memory and EP>1 | Datacenter-only |
| **DeepEP all-to-all** | — | Arch check *passes* on sm_120 (`min=(10,0)`), but the **fabric** is the blocker: normal mode sizes NVLink buffers (`deep_ep.py:129-142`), low-latency is RDMA/IBGDA-only (`:143-154`) | **Genuinely blocked — no NVLink, no IB.** Use the default EP scheme instead |
| **EPLB** | ✅ 100% portable — `eplb_algorithms/deepseek.py` is CPU PyTorch, `expert_location.py` is tensor indexing, the recorder is counting + `reduce` | No arch dependency anywhere | **Portable, and unwired.** Blocked on plumbing, not hardware |
| **TP (uniform, `--tp N`)** | ✅ | All-reduce-based, PCIe-safe — **except** the `logits_processor` symm-mem all-gather (`logits_processor.py:288-307`), which is unguarded on the plain-TP path | Portable **after** guarding one call site |
| **DP-attention** | ✅ in principle | **Blocked on PCIe**: every `token_all_gather`/`token_reduce_scatter` routes to `TritonRSAGBackend` → `multimem.ld_reduce`/`multimem.st` with no fallback (`ops/communication/triton.py:1093-1155`) | Needs an NCCL fallback path added |
| **Custom all-reduce** | — | TRT-LLM Lamport needs CUDA IPC/P2P; degrades gracefully. `CustomAllreduce` allows PCIe at **world_size == 2** (`custom_all_reduce.py:134-140`, `_SUPPORTED_WORLD_SIZES=[2,4,6,8]`) but the whole backend is dead code | TP2 is the interesting case — see §4 |
| **PP** | — | Does not exist | Not a gate; an absence |
| **P/D disagg** | ⚠️ Control plane is HTTP/ZMQ/gloo — portable. **Data plane hardcodes `"rdma"`** (`pd/base/mooncake_engine.py:74-79`) | Single-node CI works via `MC_INTRANODE_NVLINK=1` (`serve_qwen35_397b_nvfp4_pd_1p1d.sh:29-30`) — we have no NVLink | **Blocked without an RNIC or a Mooncake PCIe/TCP transport.** The *design* (layer-wise streaming, `transfer_plan.py`) is fully portable |
| **KV offload L2 / hierarchical cache** | ✅ Pinned host memory + copy kernels (`kv_cache_host.py:194, :347-353`) | Pure PCIe H2D/D2H | **Fully portable, ON by default, never measured** |
| **fp8 per-channel weights** | ✅ | Portable Triton (`dense/fp8.py:305-320`) | **[validated — E23]** |
| **fp8 block-128 (DeepSeek scheme)** | ✅ **Already portable** — `triton_mm_fp8_blockscale` has **no arch bound** (`ops/gemm/triton.py:782-795`) and there is an explicit sm_120 quantizer special-case (`ops/gemm/__init__.py:198-211`) | Only the `deep_gemm` kernel is datacenter — and it is **mis-gated** (`min=(9,0)`, no max, `ops/gemm/deep_gemm.py:63`) so it wins selection on sm_120 and then fails at JIT | **Scheme portable; bound the deep_gemm kernel.** Hard sub-case: DeepSeek-V4 `is_bmm` o_proj **raises at load** with no fp32 fallback (`dense/fp8.py:271-281`) |
| **nvfp4** | ✅ sm_120 has FP4 tensor cores and is tagged `tensor_core:f4` (`platform.py:398-399`); `cublaslt_mm_nvfp4` passes its gate (`ops/gemm/trtllm.py:78-90`) | **Blocked by a one-word import gate**, not the GEMM: `fp4_quantize` is bound inside `if ... platform.is_blackwell:` (`ops/quantization/flashinfer.py:68-70`) where `is_blackwell` is `major == 10` exactly → stays `error_fn` → first forward raises. Same for `mm_fp4` (`ops/gemm/flashinfer.py:287`) | **Fixable one-word bug** (`is_blackwell` → `is_blackwell_plus`), then test whether the flashinfer binaries cover `sm_120a` |
| **mxfp4 (dense)** | — | **No NVIDIA kernel exists at all**, sm_100 included — `triton_mm_mxfp4` is `vendors={"amd"}` (`ops/gemm/triton.py:901`), quantizer likewise (`ops/quantization/triton.py:517-522`). Matches E3's "no candidate" | Not a portability gap — an *absence* on all NVIDIA |
| **mxfp8 1×32 cute-dsl** | ✅ fallback exists | Genuinely locked (`max=(10,3)`, `ops/gemm/flashinfer.py:201`) **and** import-gated; but `triton_mm_fp8_blockscale` accepts the `(1,32)` uint8 signature | Clean fallback |
| **MTP / EAGLE / DFlash** | ✅ **No `ArchVersion` gate anywhere in `drafter/*.py` or the spec-sampling kernels**; both verifiers have Triton/pure-torch fallbacks (`ops/sampling/cuda.py:41-48`, `backends/greedy.py:47`) | Full-step graph capture needs only `runtime:cuda_graph` | **Fully portable.** What blocks spec decode on sm_120 is the *target model* (MLA/DSA/nvfp4), not the drafter. Feature ceiling: chain-only (`server_args.py:615`), no tree, no n-gram, MLA `q_len:{1}` |
| **Scheduler (all of it)** | ✅ C++/Python, no device code | — | **Fully portable** |

**Bottom line on the split:** almost nothing in the big-model stack is *algorithmically*
datacenter-bound. The genuine hardware walls are exactly three — **DeepEP (needs NVLink or
RDMA)**, **MegaMoE (needs sm_100 exactly + NVLink symmetric memory)**, and **P/D's
hardcoded RDMA data plane**. Everything else is either already portable, a mis-gate to fix,
or an arch-locked kernel with a working portable sibling.

---

## 3. Versus the reference engines

Reference submodules are all at 2026-07-24/25 HEAD: `sglang a690e5e0b3`,
`vllm d02df748b` (`v0.23.1rc0-1451`), `TensorRT-LLM bfb0fca859`, `lmdeploy 9140d372`
("DeepSeek V4 support"), `tilert a8368a6`.

**A provenance correction that changes the strategic picture.** The prompt's premise —
and our own architecture doc — assumed tokenspeed is SGLang-derived and that
`tokenspeed-mla` is a first-party edge. Both need refining:

1. `ACKNOWLEDGEMENTS.md:10-16` lists TensorRT-LLM **first**, then
   `meituan-longcat/SGLang-FluentLLM`, then vLLM, then lightllm, then upstream sglang. The
   code supports "SGLang-family rewrite" rather than "SGLang fork": the package is
   `tokenspeed.runtime.*` not `sglang.srt.*`, and the whole scheduler is C++.
2. **`tokenspeed-mla` has already been upstreamed into both SGLang and vLLM.**
   `third_party/sglang/python/sglang/srt/layers/attention/tokenspeed_mla_backend.py:1`
   carries `Copyright (c) 2026 LightSeek Foundation`; vLLM has
   `third_party/vllm/vllm/v1/attention/backends/mla/tokenspeed_mla.py` (decode-only,
   FP8-KV-only per its docstring at `:3`). SGLang even vendored the fused Triton
   pack/quantize kernel. **The inherited *head start* on this specific kernel is gone —
   but that is a statement about parity of availability, not about the kernel's ceiling.**
   Upstreaming resets the starting line; it says nothing about how far the kernel is from
   optimal, and nothing at all about non-datacenter hardware: these kernels are tuned for
   sm_100/H200 (tcgen05, that shmem/occupancy budget), and **nobody has tuned them for
   sm_120 or AMD**. Our own measurements show mature, widely-used paths routinely leave
   large margins on the table — E26 measured QKV-proj at 70.5% and O-proj at 67.1% of peak
   DRAM bandwidth on a generic cuBLAS `gemvx`; E23 found fp8 blocked by three dispatch bugs;
   E27 found no head_dim-512 attention kernel exists for sm_120 at all. Treat "upstream has
   this kernel" as a starting line, never as a ceiling. What remains exclusive is the AOT
   binary prefill path (`fmha_binary.py`) and the deeper prefill/chunked-prefix/draft-extend
   integration (our backend is 562 lines vs vLLM's 316 decode-only).

### Per-axis

| Axis | Who is stronger | Evidence |
|---|---|---|
| **MLA breadth** | **sglang / vLLM dominate.** They have tokenspeed's kernel *plus* everything else — sglang has 33 attention backends incl. `flashmla`, `cutlass_mla`, `nsa_backend`, `dsa_backend`, `aiter`, `wave`, `xpu`; vLLM has 19 MLA backends incl. `flashinfer_mla_sparse_sm120.py` (**note the name — vLLM already has an sm_120 sparse-MLA backend**) | `third_party/sglang/python/sglang/srt/layers/attention/`, `third_party/vllm/vllm/v1/attention/backends/mla/` |
| **MLA integration depth** | **tokenspeed.** Ours does prefill + decode + chunked-prefix + draft-extend and ships the fused KV-pack kernel end-to-end | `backends/tokenspeed_mla.py` (562 L) |
| **MoE EP / EPLB** | **sglang > vLLM ≫ tokenspeed.** Both references execute **online** rebalancing with real weight movement (sglang `srt/eplb/eplb_manager.py:92 rebalance()`, `expert_location_updater.py`, `elastic_ep/`; vLLM `distributed/eplb/rebalance_execute.py`, `async_worker.py`). Ours is startup-only *and unwired*. sglang has **9 token-dispatcher backends** (deepep, mooncake, moriep, nixl, flashinfer, ascend_tp, standard) vs our DeepEP wired into **one** model | §1.2 |
| **P/D disagg** | **vLLM strongest on abstraction** — a stable `KVConnectorBase_V1` ABC third parties implement, ~8 transports (NIXL 13 files, mooncake, moriio, LMCache, hf3fs, flexkv, offloading) plus `multi_connector.py` for stacking. sglang has NIXL/mooncake/mori/ascend/fake. **We have one transport and zero NIXL** | `third_party/vllm/vllm/distributed/kv_transfer/kv_connector/v1/` |
| **P/D design** | **tokenspeed on layer-wise streaming + layout negotiation.** Our `transfer_plan.py` declarative `ParallelLayout`/`BufferKind` negotiation with `UnsupportedPDLayoutError` is cleaner than sglang's ad-hoc TP mapping, and P/D being a C++ FSM domain is unique | §1.5 |
| **KV offload / hierarchical** | **sglang decisively.** `mem_cache/` has `hiradix_cache`, `hi_mamba_radix_cache`, `hisparse_memory_pool`, `unified_radix_cache` + per-type components, SWA-aware trees, and **11 storage backends**. We have **one** (mooncake) | `third_party/sglang/python/sglang/srt/mem_cache/storage/` |
| **KV hierarchy design** | **tokenspeed.** Host-tier prefix matching happens *inside* the C++ scheduler's admission decision — `AdmissionMatch{device, host}` (`csrc/cache/kv_cache_coordinator.h:60-68`) — rather than bolted on | §1.6 |
| **Speculative** | **sglang > vLLM ≫ tokenspeed.** sglang has 36 files incl. ngram + `cpp_ngram/`, suffix decoding, `adaptive_spec_params.py`, `ragged_verify.py`; vLLM has 18 incl. medusa, ngram-GPU, `suffix_decoding.py`. We have 3 drafters, **chain-only**. **DFlash is not our edge** — vLLM added it 2026-03-30 (#36847) and sglang 2026-04-07 (#22077), both before tokenspeed's public launch | §1.8 |
| **Quantization breadth** | **sglang ~5× wider** (50 modules: AWQ, GPTQ, Marlin, bitsandbytes, GGUF, auto_round, quark, `nvfp4_online.py`, `moe_runner/deep_gemm.py`). On the FP4/FP8 MoE formats that matter on Blackwell we are competitive and add **mxint4**, which I did not find in sglang's MoE weight path. Our gaps: **no DeepGEMM MoE**, no AWQ/GPTQ/Marlin | §1.7 |
| **Model coverage** | sglang 210, vLLM 290, TRT-LLM 71, lmdeploy 56, **us ~30** | `runtime/models/` |

### What is genuinely ours (no counterpart anywhere)

1. **The C++ FSM scheduler with type-enforced KV ownership.** Request state is a
   `std::variant` of 13 RAII state classes (`csrc/fsm/states.h:31-32`), each *owning* its
   pages via move-only `OwnedPages` with deleted copy constructors
   (`csrc/fsm/forward_states.h:84-85`). sglang's scheduler is Python with only the radix
   tree in C++; vLLM's is Python; TRT-LLM's C++ `batch_manager/` is procedural with no
   typed FSM. **This is the strongest structural differentiator in the repo.**
2. **The placement-annotation communication compiler.**
   `PlacementType{REPLICATE, SHARD, PARTIAL} × ParallelGroup{ATTN_TP, DENSE_TP, MOE_TP_EP}`
   (`runtime/models/base/placement.py:20-24`), with `compiler.py:20-25` walking each
   layer's `ModuleSpec` chain and **deriving** the minimal comm ops to reconcile adjacent
   placements. sglang's `layers/communicator.py` (1457 L) *enumerates* modes with
   hand-written decision trees. Ours is the more principled design.
3. **Per-layer-family `Mapping`** — independent `attn(tp,cp,dp)` / `dense(tp,dp)` /
   `moe(tp,ep,dp)` / `vision` (`mapping.py`). Finer-grained than any reference.
4. **The capability-based kernel registry** with `CapabilityRequirement` + `Priority` bands
   + `plugins/`, and our own `unieinfra-kernel/` first-party plugin package registering
   into the same singleton registry.
5. **Flat KV-cache groups** pushed all the way into the attention backend and CUDA-graph
   buffers (`backends/flat_groups.py:20-33`) — build-gated off today.
6. **AOT binary MLA prefill** with NVIDIA-internal knobs — the one MLA capability the
   upstreaming did not transfer.

### Claimed-but-not-unique (be honest)

**EPD encode disagg** — sglang has `srt/disaggregation/{encode_server,encode_receiver,encode_grpc_server}.py`.
**DFlash** — in vLLM and sglang before tokenspeed launched. **deepseek_v4 / MHC** — sglang
has `deepseek_v4_backend.py`, vLLM has its own CuTeDSL compressor, lmdeploy landed V4 on
2026-07-24. **MSA** — sglang has `minimax_sparse_backend.py`, vLLM has `models/minimax_m3/`.
**tokenspeed-mla** — upstreamed to both.

### Verified absences

Pipeline parallelism (0 hits). LoRA (0 hits — only `kv_lora_rank`). NIXL (0 hits).
Two-batch overlap. DeepGEMM MoE grouped GEMM. Working context parallelism. Grammar
backends beyond xgrammar. AWQ/GPTQ/Marlin/GGUF/bitsandbytes. A standalone router — we
spawn sglang's `smg` as a subprocess (`cli/serve_smg.py:21`).

---

## 4. What we can validate NOW on 4× RTX PRO 6000

### Our actual hardware constraint

```
$ nvidia-smi topo -m
        GPU0  GPU1  GPU2  GPU3
GPU0     X    PIX   NODE  NODE
GPU1    PIX    X    NODE  NODE
GPU2    NODE  NODE   X    PIX
GPU3    NODE  NODE  PIX    X
```

**No NVLink anywhere.** GPU0↔1 and GPU2↔3 are `PIX` (one PCIe bridge — the best pairs);
everything else is `NODE` (across PCIe host bridges). This confirms E1's
`interconnect=pcie` and sharpens it: **TP2 on {0,1} or {2,3} is the good topology; TP4
crosses host bridges.** Concretely:
- `CustomAllreduce` allows PCIe **only at world_size == 2** (`custom_all_reduce.py:134-140`)
  — though that backend is dead code today (§1.4).
- DeepEP needs NVLink or IB → **unavailable at any TP**.
- Mooncake P/D hardcodes `"rdma"`; the upstream single-node CI works around this with
  `MC_INTRANODE_NVLINK=1` → **unavailable to us**.

### Models on disk and their capacity arithmetic

Computed from the engine's own cost functions (`configs/mha.py:153-163`,
`qwen3_5_text_base_config.py:353-364`). **These are [inferred] estimates from the formulas,
not measurements** — the real numbers come from the startup log.

| Model | Arch | Weights (bf16) | KV/token | GDN state/req | Notes |
|---|---|---|---|---|---|
| **Qwen3-32B** | dense GQA, 64 L, 8 KV heads × 128 | 62 GiB | **256 KiB** (4096 B × 64 L) | — | KV is 16× Qwen3-8B's per-token cost relative to free HBM |
| **Qwen3.5-27B** | hybrid GDN, 64 L (**16 full** / 48 linear), 4 KV × 256 | 52 GiB | **64 KiB** (4096 B × 16 L) | **146.8 MiB** (fp32 SSM) | `full_attention_interval=4`; ships MTP weights |
| **Qwen3.6-35B-A3B** | GDN **+ MoE** (256 experts, top-8), 40 L (**10 full** / 30 linear), 2 KV × 256 | 67 GiB | **20 KiB** | **61.4 MiB** | ships MTP weights; needs the E14 MoE fix |
| **gemma-4-26B-A4B** | MoE (128 experts), sliding hd=256 / global hd=**512** | 49 GiB | — | — | **E27-blocked** on the hd-512 sm_120 kernel; also `gemma.py` is dense-only |

Two things jump out. **First, hybrid GDN models are KV-cheap and state-expensive** — 27B
holds 4× less KV per token than 32B despite being the same size class, but pays ~147 MiB
of recurrent state per *request*, which becomes the concurrency ceiling instead. Second,
`TOKENSPEED_MAMBA_SSM_DTYPE` defaults to **`float32`** (`runtime/utils/env.py:291`) and
`bfloat16` is accepted (`qwen3_5_text_base_config.py:341-344`) — **halving the SSM dtype
roughly halves the state and doubles GDN concurrency**, at unknown accuracy cost. That is
a one-env-var experiment nobody has run.

### Ranked by (value × feasibility)

**Tier 1 — do these first. High value, low risk, zero-to-a-handful of lines of code.**

1. **Qwen3-32B dense, TP1, bf16 → then `--quantization fp8`.**
   Pure reuse of the validated E15/E23 path on a 4× bigger model. The interesting question
   is not tok/s, it is **whether the E25 adaptive-batch derivation behaves when KV capacity
   is the binding constraint instead of the graph cap** — at 256 KiB/token the derived cap
   (`event_loop.py:113-140`) will land far below 160 and the floor-only clamp
   (`:138`) will silently pin it at the static 160. That is a *predicted* interaction we
   should measure. fp8 weights roughly halve the weight footprint and should more than
   triple KV capacity. **Zero new code.**

2. **Host KV offload (L2 kvstore) — measure it, finally.**
   It has been **ON in every benchmark since E4** (`server_args.py:687-688`) at ratio 2.0,
   and we have never measured its hit rate or cost. Run `--kvstore-ratio 0.5` (upstream CI
   value) vs `--disable-kvstore` vs default on Qwen3-32B with a repeated-prefix workload
   and `--enable-cache-report`. This also retires the E17 host-OOM footgun. **Zero new code.**

3. **MTP on Qwen3.5-27B and Qwen3.6-35B-A3B.**
   Both checkpoints **ship MTP weights** — I verified 15 and 19 `mtp.*` tensors respectively
   in their `model.safetensors.index.json` — and both NextN classes are registered
   (`qwen3_5_nextn.py:416`, `layers/attention/registry.py:119-124`). `--speculative-algorithm MTP`
   with no draft path uses the base checkpoint (`server_args.py:582-589`). E22/E26 both
   named MTP as the top remaining lever and we have never tried it. Constraints to expect:
   chain-only (`server_args.py:615-620`), and MTP forces the non-overlap scheduler.
   **Zero new code** for 27B; 35B-A3B needs item 4 first.

4. **Qwen3.6-35B-A3B MoE — run it end to end.**
   The E14 cast fix has landed in the working tree (`cutlass_unquant.py:93-97`) but
   **no MoE model has actually generated on sm_120 yet** — E14 only validated the op. The
   kernel's gate is `min_arch 8.9`, no max, so it matches sm_120 (`cutlass_unquant.py:56-58`).
   This unlocks **our only MoE model**, and at 20 KiB/token KV it should reach high
   concurrency at TP1 despite the 67 GiB weights. Also fix the two sibling casts
   (`cutlass_fp8.py:102`, `cutlass_nvfp4.py:180`) while here.

5. **nvfp4 on sm_120 — the one-word gate fix (§1.7).**
   `is_blackwell` → `is_blackwell_plus` at `ops/quantization/flashinfer.py:68` and
   `ops/gemm/flashinfer.py:287`, then try `--quantization nvfp4` on Qwen3-32B. sm_120 has
   FP4 tensor cores and the nvfp4 GEMM (`cublaslt_mm_nvfp4`) already passes its gate.
   Weight footprint is what gates big-model concurrency, and nvfp4 would roughly halve it
   again on top of E23's fp8. High expected value, ~2 lines, and a cheap negative result if
   the flashinfer binaries lack `sm_120a`. **Also worth a probe: block-128 fp8** — we now
   know `triton_mm_fp8_blockscale` is arch-unbounded with an explicit sm_120 path, so
   DeepSeek-style fp8 checkpoints may already load.

**Tier 2 — real value, real risk, small amounts of new code.**

6. **TP2 on {0,1} — Qwen3-32B.**
   Value is large: TP2 shards KV heads 8→4 *and* halves weights, which by the formulas
   should give roughly **6× the KV token capacity** of TP1 bf16 — capacity is what gates
   big-model concurrency. **But expect a specific failure**: `logits_processor._init_all_gather_state`
   (`layers/logits_processor.py:288-307`) creates PyTorch symmetric-memory state with no
   try/except and no interconnect check whenever `tp_size > 1` and
   `vocab_padded % (tp_size*8) == 0` — true for Qwen3's 151936 vocab at TP2 — and the
   gather uses `multicast_ptr`, which requires NVLink multicast. **This is our predicted
   first TP>1 blocker on PCIe, and the fix is a guard + NCCL fallback at one call site.**
   Also confirm whether the TRT-LLM Lamport IPC path comes up or logs its graceful
   `"unavailable (NCCL fallback)"` (`distributed_initializer.py:202-206`).

7. **MoE expert parallelism at EP2 — Qwen3.6-35B-A3B on {0,1}.**
   This is the headline "we own EP and never used it" item, and it is **PCIe-viable**
   because the default `all2all_backend=none` scheme is all-gather + masked local experts
   + all-reduce over NCCL (§1.2) — no DeepEP. `--enable-expert-parallel` auto-sets
   `ep_size=world_size` (`server_args.py:501-503`); remember TP and EP are mutually
   exclusive inside a MoE layer (`expert.py:110-111`), so this is EP2 with `moe_tp_size=1`.
   Depends on items 4 and 6.

8. **`TOKENSPEED_MAMBA_SSM_DTYPE=bfloat16` on Qwen3.5-27B.**
   One env var; roughly halves the 147 MiB/request recurrent state and should roughly
   double GDN concurrency. Needs an accuracy check, not just a throughput number.

9. **Mixed prefill+decode batching (`--enable-mixed-batch`).**
   Currently OFF (`server_args.py:109`), and with it off **any step that schedules a
   prefill chunk runs zero decodes** (`forward.cpp:1020, :1027`). On a 32B model with 8192-token
   chunks this is the dominant TPOT-jitter source. The machinery is complete on both sides.
   **This is the single highest-leverage unflipped switch in the scheduler.**

**Tier 3 — needs porting work first.**

10. **DSA / MLA on sm_120.** Both have complete portable Triton paths; what stands in the
   way is the mis-gate (§2). Implementing Steps A/B of `tokenspeed-sm120-support.md`
   (add a `compute:tcgen05` feature; bound `deep_gemm`, `cublaslt_mm_nvfp4` and the
   unbounded `min_arch=(10,0)` registrations) would make the portable path *reachable*.
   But we have **no MLA or DSA model on disk** — validating this means downloading a
   DeepSeek/GLM-5 checkpoint, and flagship sizes need TP across ≥4 cards.

11. **Gemma-4-26B-A4B.** Double-blocked: E27's hd-512 sm_120 attention kernel gap, and
    `gemma.py` is dense-only (`gemma.py:27, :380-384`). Needs the kernel plus a MoE block.

### Genuinely blocked until datacenter hardware arrives

| Item | Why | Evidence |
|---|---|---|
| **DeepEP all-to-all** | Needs NVLink (normal mode NVL buffers) or RDMA/IBGDA (low-latency) | `ops/communication/deep_ep.py:128-155` |
| **DeepGEMM MegaMoE** | `capability[0] != 10 → raise` — exactly SM100 — plus NVLink symmetric memory | `models/deepseek_v4.py:2320-2322, :2368-2392` |
| **`tokenspeed-mla` CuTe kernels** | tcgen05 + TMEM + 2-CTA MMA; a rewrite, not a recompile | `mla_decode_fp16.py:500, :1179`; `mla_decode_fp8.py:109-131` |
| **MSA CuTe attend** | `[10.0, 10.3]` + `is_blackwell` — but Triton covers it | `ops/attention/msa.py:64, :90-99` |
| **trtllm MoE family (incl. mxfp4/mxint4)** | all capped `max=(10,3)` | §1.7 table |
| **P/D disaggregation** | Mooncake data plane hardcodes `"rdma"`; upstream single-node CI needs `MC_INTRANODE_NVLINK=1` | `pd/base/mooncake_engine.py:74-79`; `serve_qwen35_397b_nvfp4_pd_1p1d.sh:29-30` |
| **True multi-node scaling** | No PP; TP/EP across nodes only, and our fastest comm paths need NVLink | §1.3, §1.4 |

---

## 5. Recommended big-model roadmap

Framed against the throughput physics we have actually evidenced: **at high batch the
decode path is compute-bound** (E26: SM 84–91%, DRAM 10–15% at batch-128; Gate+Up = 41% of
layer time), **at batch-1 it is latency-hiding-bound, not bandwidth-bound** (E26 corrected
E22: QKV-proj 70.5%, O-proj 67.1% of DRAM ceiling), and **for big models HBM capacity gates
concurrency before either of those bites**.

### (a) Unlock now on sm_120

Ordered by (value ÷ effort). Every item names existing code.

1. **Finish the E14 MoE cast fix.** The main site
   (`ops/moe/flashinfer/cutlass_unquant.py:93-97`) was fixed in the working tree while this
   study was written; the **identical latent cast remains at `cutlass_fp8.py:102` and
   `cutlass_nvfp4.py:180`**. Verified correct against
   `third_party/TensorRT-LLM/cpp/tensorrt_llm/thop/moeOp.cpp:415`
   (`CHECK_INPUT(token_final_scales, at::ScalarType::Float)`). Then **run Qwen3.6-35B-A3B
   end-to-end** — the fix is unvalidated until a MoE model actually generates.

2. **Implement Steps A/B of `tokenspeed-sm120-support.md` — the capability model.**
   Add a `compute:tcgen05` (+`memory:tmem`) feature granted only for `arch == (10,x)`;
   drop `compute:cluster` for sm_120; keep `tensor_core:f4`/`memory:tma` (E1 established
   both are real on sm_120). Then bound the unguarded registrations:
   `ops/attention/deep_gemm/__init__.py:112, :161, :318`; `ops/gemm/deep_gemm.py:63`;
   `ops/gemm/trtllm.py:84`; `ops/moe/flashinfer/cutlass_nvfp4.py:140`. **This is the
   single highest-leverage change in the tree** — it converts a class of runtime crashes
   into clean fallbacks and makes every portable path *reachable*. E3/M0b already measured
   the exact three DSA winners this fixes.

3. **Unblock nvfp4 on sm_120 — a one-word change.** `is_blackwell` → `is_blackwell_plus`
   at `ops/quantization/flashinfer.py:68` and `ops/gemm/flashinfer.py:287`. The nvfp4
   *GEMM* already passes its gate (`cublaslt_mm_nvfp4`, `ops/gemm/trtllm.py:78-90`); only
   the activation quantizer is hidden behind `major == 10`. sm_120 has real FP4 tensor
   cores. If it works, this is the **next big weight-footprint lever after E23's fp8** —
   and weight footprint is exactly what gates big-model concurrency (§4). If the
   flashinfer binaries turn out not to cover `sm_120a`, we learn that cheaply and the
   answer becomes a first-party `unieinfra-kernel` nvfp4 quantizer.

4. **Fix the two latent quantization bugs found in this study.**
   (a) `--quantization w8a8_fp8` still reproduces both E23 defects
   (`layers/dense/w8a8_fp8.py:66` and `:132-138`) — same fix as E23, ~2 lines.
   (b) `compressed_tensors.py:363-383` references undefined FP8 scheme classes behind a
   `noqa: F821`; any compressed-tensors FP8 checkpoint `NameError`s at layer construction.

5. **Guard the `tokenspeed_mla` backend.** `backends/tokenspeed_mla.py` has no arch check
   at all; add `is_blackwell` (major == 10) so `--attention-backend tokenspeed_mla` on
   sm_120 refuses cleanly instead of failing inside CuteDSL.

6. **Measure host KV offload.** It has been on since E4 and never measured (§4 item 2).
   Then decide the default: keep it, lower `kvstore_ratio` to 0.5 (upstream CI's value),
   or follow the recipes and default it off for big models. Self-configuring-engine
   principle: this should be **derived from host RAM**, not a fixed 2.0×.

7. **Fix the DP-unaware adaptive-batch derivation.** `derive_adaptive_max_batch` is fed
   this rank's `max_total_num_tokens` (`event_loop.py:262-282`) but its result is assigned
   to the **global** `max_num_seqs`, which is then divided again at
   `per_rank_max_batch = max_num_seqs // dp_size` (`event_loop.py:293`). At DP=4 that is a
   4× under-provision. Fix before any DP experiment. (TP is unaffected.)

8. **Flip `--enable-mixed-batch` after measuring it.** Highest-leverage unflipped scheduler
   switch (§4 item 9). Our standing rule is defaults, not hand-tuning — so measure at
   C=1/32/128/256 with long prompts, then make it the default if it wins.

9. **Guard `logits_processor`'s symmetric-memory all-gather** (`logits_processor.py:288-307`)
   with an interconnect check + NCCL fallback. **This is the gate on all our TP>1 work.**
   Same treatment for `dp_sampling_comm.py:91-110`, which checks NVIDIA + torch version but
   never the fabric.

10. **Write the portable bf16 unquant Triton MoE kernel** into `unieinfra-kernel/`
   (`unieinfra_kernel/ops/moe/`). We have a reference implementation in-tree —
   `ops/moe/gluon/bf16.py`, written for gfx950 — plus vLLM's `fused_experts`. This closes
   E6 permanently, removes our single-implementation risk on bf16 MoE, and doubles as the
   AMD path. This is the flagship first-party kernel target the plugin package was built for.

11. **Cheap arch probes with high information value:** does DeepSeek-style **block-128 fp8
   actually run end-to-end on sm_120** now that we know `triton_mm_fp8_blockscale` is
   unbounded and there is an sm_120 quantizer special-case (`ops/gemm/__init__.py:198-211`)?
   Does `TOKENSPEED_MAMBA_SSM_DTYPE=bfloat16` hold accuracy? Does a P2P handshake succeed
   between GPU0 and GPU1? All three are one-line experiments with real capacity or
   throughput consequences.

### (b) Design/port now, validate when datacenter hardware arrives

12. **Wire up EPLB.** The algorithm, metadata, recorder and routing-time remap are all
    present and portable (§1.2); what is missing is ~4 call sites:
    `compute_initial_expert_location_metadata` at model-runner init, a real recorder
    instead of the no-op (`distribution_recorder.py:238-240`), `ExpertLocationDispatchInfo`
    threaded into `TopK.forward`, and the contiguous-range assumption at `loader.py:69-70`
    relaxed. Also fix `distribution_recorder.py:263` (`enable_deepep_moe` is not a
    `ServerArgs` field). **Static EPLB is testable at EP2 on our box**; online rebalancing —
    which is where sglang and vLLM are ahead — is the design work to do now.

13. **An sm_120 MLA path, end to end.** The Triton kernels exist; the work is (a) the
    capability fixes in item 2, (b) lifting `q_len: {1}` on `triton/mla_decode.py:261-264`
    so MTP can verify over MLA, (c) numerics validation. Needs a DeepSeek-class checkpoint
    downloaded. Do the design now; MLA's KV advantage is exactly what makes 600B-class
    models servable.

14. **An sm_120 DSA top-k.** Purely a gating fix plus validation — Triton already
    implements `dsa_plan`/`dsa_decode_topk`/`dsa_prefill_topk` with no arch floor. Sparse
    attention is what makes 128k+ context affordable, so this matters more as context grows
    than as parameters grow. References named in `docs/support-matrix.md:48`.

15. **A P/D transport that is not RDMA-only.** The *design* — layer-wise streaming gated on
    a GPU step counter, and `transfer_plan.py`'s declarative layout negotiation — is
    genuinely good and fully portable. What is not portable is
    `mooncake_engine.py:74-79`'s hardcoded `"rdma"`. Adding a TCP/PCIe transport (or
    adopting vLLM's `KVConnectorBase_V1` shape, which is the reference-class abstraction)
    would make P/D testable on our box and close the transport-breadth gap in one move.

16. **Decode headroom reservation** — roadmap item 4 of `scheduling-improvements.md`, still
    unimplemented. `csrc/scheduler/types.h:124` has the dead field waiting for it. Matters
    much more for big models, where one decode step across a deep batch consumes many pages
    and the current `+decode_input_tokens` reserve permits admit→retract oscillation.

17. **Exercise retraction deliberately.** It fires only when a step schedules literally
    nothing (`forward.cpp:1046`), retracts one victim per step, and E25 observed **zero
    retractions** at C=256. On a 32B model where KV pressure is real, the whole
    `Retracting`/`Retracted`/loadback path — plus abort-on-host-exhaustion (`forward.cpp:667-673`)
    — is dark code. Write a test that forces it.

18. **Pipeline parallelism.** The one parallelism dimension we have zero of, and every
    reference has. Not needed at 4 GPUs; needed the moment we go multi-node with a
    600B-class model, because our fastest comm paths assume NVLink and PP is the dimension
    that tolerates slow links. Design it into `Mapping` now while the abstraction is young.

### (c) Genuinely blocked

DeepEP, MegaMoE, the `tokenspeed-mla` CuTe kernels, the trtllm MoE family (mxfp4/mxint4),
the MSA CuTe attend, and P/D over RDMA — see the §4 table. Note the honest framing: of
these, only **DeepEP**, **MegaMoE** and **RDMA P/D** are blocked by *hardware*. The rest
are arch-locked *kernels* with working portable siblings — porting targets, not walls.

### A strategic note on where our edge actually is

The reference comparison (§3) says something uncomfortable and useful: **the inherited
head start on MLA specifically has been given away** — `tokenspeed-mla` shipped in SGLang
and vLLM in May 2026 — and on breadth (models, quantization schemes, KV backends,
spec-decode variants, EPLB maturity) we are behind both by large factors and will not close
that by out-porting them.

That is *not* the same as "kernels are a closed front." Parity of availability says nothing
about a kernel's distance from its ceiling, and per-architecture tuning is a largely
untouched surface: every one of these kernels is tuned for datacenter Blackwell/Hopper, and
sm_120 (no tcgen05, different shmem/occupancy budget) and AMD have had no such attention.
Our own ncu evidence (E26: QKV/O-proj at 67–70% of peak bandwidth on a stock cuBLAS
`gemvx`), the fp8 dispatch bugs (E23), and the missing head_dim-512 kernel (E27) all say the
kernel surface is wide open — which is exactly the bet behind the `unieinfra-kernel` package
and agent-designed kernels: out-iterate hand tuning, per architecture.

The two moats are complementary and both are worth funding. What is uniquely ours is
**architectural**: the C++ FSM scheduler with compile-time KV ownership, the placement→comm
compiler, the per-layer-family `Mapping`, and the capability registry — and this one is
empirically load-bearing, since E18's +17% at C=128 was attributed to scheduling rather than
attention, and E25's adaptive-batch removed the C=256 admission cliff outright. Those are the assets that support the product thesis of *beating* vLLM/SGLang
rather than matching them — because they are where an SLO-aware, self-configuring scheduler
can be built that the references' Python schedulers structurally cannot match. The
big-model work above should be sequenced to *feed* that: every item that lights up a large
model is also a workload that stresses the scheduler in ways Qwen3-8B never will.

---

## Appendix — what would change these conclusions

- **A P2P capability test on our box.** I did not run one (no GPU use). Whether
  `torch.cuda.can_device_access_peer` succeeds between GPU0 and GPU1 decides whether the
  TRT-LLM Lamport all-reduce and CUDA IPC paths are available at TP2, and therefore how
  much of §4 item 6 is real. This is the single cheapest experiment that would sharpen this
  document.
- **The actual startup log for Qwen3-32B.** Every capacity number in §4 is arithmetic from
  the engine's own formulas; the engine prints the real `max_total_num_tokens`. One launch
  replaces four estimates with measurements.
- **Whether flashinfer's sm_120 coverage has moved.** E11 found the engine's flashinfer
  attention is TRT-LLM-gen FMHA hardcoded to sm_100/103; vLLM meanwhile ships
  `flashinfer_mla_sparse_sm120.py`. If flashinfer has since added sm_120 kernels, several
  "datacenter-gated" rows above become portable.

*Every ✅/validated claim here traces to `docs/experiments-log.md`. Everything else is
labelled [code] or [inferred]. Update this document as items move between the three
categories, and keep the Chinese mirror in sync once written.*
