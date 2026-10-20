# Engine feature matrix — TokenSpeed vs. the field

Surveyed 2026-08-02 by reading source. Nothing here was benchmarked; every claim
is a code claim, and where the code did not settle the question the cell says
**unverified** rather than guessing.

Scope is the **engine**: schedulers, memory management, serving features. Kernel
selection and kernel performance are deliberately out of scope (other agents
cover them); the kernel registry appears only where it is an engine-level
extensibility surface.

## Trees read, and what they actually are

| Engine | Path | Commit / version |
|---|---|---|
| **TokenSpeed** | `/workspace/roy/tokenspeed` | `3f88dcc2` |
| vLLM | `/workspace/roy/UnieInfra/third_party/vllm` | `d02df748b`, 2026-07-24 — **v0.26.1-dev** (128 commits before `v0.26.1rc0`). Not pristine: `vllm/v1/attention/backends/mla/tokenspeed_mla.py` is a local addition. |
| SGLang | `/workspace/roy/UnieInfra/third_party/sglang` | `a690e5e0b3` — 0.5.16-dev |
| TRT-LLM | `/workspace/roy/UnieInfra/third_party/TensorRT-LLM` | `bfb0fca859` — `1.3.0rc23` (PyTorch backend) |
| TurboMind | `/workspace/roy/UnieInfra/third_party/lmdeploy` | `9140d372` — `0.14.0` |
| tile-RT | `/workspace/roy/UnieInfra/third_party/tilert` | `a8368a6` — v0.1.5 |
| Modular MAX | `/workspace/roy/UnieInfra/third_party/modular` | `efa70fbe98` — MAX `26.5.0.dev2026072406` |

Two of these are not comparable engines and are marked "n/a" below rather than
"absent":

- **tile-RT is not a serving engine.** 26k lines of Python, two model families
  (`tilert/models/deepseek_v3_2/`, `tilert/models/glm_5/`), no scheduler, no
  paged KV manager, no continuous batching. Its serving story is
  `tilert/pd_vllm/` — a decode server that takes prefill from *vLLM* over a
  wire protocol (`pd_router.py`, `prefill_connector.py`, `transport.py`). It is
  a decode-latency runtime that borrows someone else's engine. Judging it on
  scheduling rows is a category error.
- **Modular MAX is a full serving stack**, and is compared normally.

Two TokenSpeed facts govern how to read every row:

1. **The flat KV cache is a compile-time option, default OFF.**
   `tokenspeed-scheduler/CMakeLists.txt:12` —
   `option(TOKENSPEED_FLAT_KVCACHE "..." OFF)`. The build publishes
   `tokenspeed_scheduler.FLAT_KVCACHE`. There are two live KV architectures in
   this tree (flat `KvCacheCoordinator` vs. the radix `LocalKVAllocator`), and
   several rows below differ between them. **Check which one your build has
   before trusting a capacity number.**
2. **The SMG gateway is not in this tree.** `python/pyproject.toml:73` pins
   `tokenspeed-smg==1.8.0.post20260728` as a binary wheel. Tool-call parsing,
   routing policy, retries and circuit breaking live there. Anything attributed
   to SMG below is **unverified** — I could not read it.

Legend: **Y** present · **P** partial · **—** absent · **n/a** not applicable.

---

## A. Scheduling

| Feature | TokenSpeed | vLLM | SGLang | TRT-LLM | TurboMind | MAX | tile-RT |
|---|---|---|---|---|---|---|---|
| Continuous batching | Y | Y | Y | Y | Y | Y | — |
| Batching **policy** choice | **— (FCFS only)** | Y (`fcfs`/`priority`) | **Y (7 policies)** | Y (`GUARANTEED_NO_EVICT`/`MAX_UTILIZATION`/`STATIC_BATCH`, + `PRIORITY`) | P | P | n/a |
| Cache-aware batching (LPM) | — | — | **Y** | — | — | — | n/a |
| Chunked prefill | Y, always on | Y, default on | Y | Y, default **off** | Y | Y | n/a |
| Mixed prefill+decode in one step | **P — off by default** | Y (inherent) | Y | Y | Y | Y | n/a |
| Priority scheduling | **—** | Y | Y | Y | — | unverified | n/a |
| Preemption | **P — deadlock-breaker only** | P — recompute only | **Y — two retraction policies** | Y | unverified | unverified | n/a |
| Admission control knobs | P | Y | **Y** | Y | P | Y | n/a |
| CPU/GPU overlap scheduler | Y (depth 0 or 1) | Y (`--async-scheduling`) | Y (default on) | Y | Y | unverified | n/a |
| Two-batch / dual-batch overlap | — | Y (`--enable-dbo`) | Y (TBO + SBO) | unverified | — | — | n/a |
| P/D disaggregation | Y | Y | Y | Y | Y | Y | P (borrows vLLM prefill) |
| P/D transports | **1 (mooncake, ±async)** | **16 connectors** | **5** (mooncake, mori, nixl, ascend, fake) | native + nixl | own `conn/` | connectors + `dkv` | custom wire |
| Pluggable scheduler | — | Y (`--scheduler-cls`) | P | — | — | Y (4 scheduler classes) | n/a |

**TokenSpeed detail.** The scheduler is a C++ FSM
(`tokenspeed-scheduler/csrc/`, ~14.8k lines) with a Python execution plane;
request lifecycle and KV ownership are encoded as typed states
(`csrc/fsm/{forward,cache,pd}_states.h`) so KV reuse is checked at compile time.
That is a real architectural asset. But `SchedulerConfig`
(`csrc/scheduler/types.h:77-136`) has **no priority field and no policy field**:
candidate ordering is arrival order, full stop.

Preemption exists but is not an admission mechanism. In the flat path,
`Scheduler::resolveFlatStarvation` (`csrc/scheduler/operations/forward.cpp:328`)
only retracts after **two consecutive wedged rounds**, and picks the largest
token holder — it is a deadlock breaker, and its own comment says so ("to unwedge
the pool"). In the radix path `scheduleRetract`
(`csrc/scheduler/operations/forward.cpp:779`) writes the victim's pages back to
the host allocator, which is closer to real swap-preemption than vLLM has
(vLLM V1 is recompute-only, `vllm/v1/core/sched/scheduler.py:1203`) — but it is
still triggered by resource exhaustion, not by a policy.

`--enable-mixed-batch` defaults **False** (`server_args.py:92`). Chunked prefill
is always on with `--chunked-prefill-size` defaulting to 8192
(`server_args.py:403`). So by default TokenSpeed alternates prefill-only and
decode-only iterations rather than piggybacking decodes onto prefill chunks the
way vLLM and SGLang do. That is a deliberate design choice with a real TTFT/TPOT
consequence, and it is worth measuring before it is assumed optimal.

---

## B. KV cache

| Feature | TokenSpeed | vLLM | SGLang | TRT-LLM | TurboMind | MAX | tile-RT |
|---|---|---|---|---|---|---|---|
| Paged layout | Y (block 64 default) | Y | Y | Y | Y | Y | — |
| Prefix caching | Y, default on | Y, default on | Y (radix, C++ impl too) | Y (`enable_block_reuse`) | Y (opt-in) | Y | — |
| Partial-block / partial reuse | Y | Y | Y | Y (`enable_partial_reuse`) | unverified | unverified | n/a |
| **Prefix caching for hybrid/recurrent** | **Y — state-resumable** | **Y — `--mamba-cache-mode align`** | **Y** (`MambaRadixCache`) | unverified | — | unverified | n/a |
| Sliding-window / hybrid allocator | Y | Y (7 manager types) | Y (SWA pool + SWA radix) | Y | Y | unverified | n/a |
| Eviction policies | **P — LRU live; 5 others are dead code** (see below) | LRU | LRU | priority-aware | LRU | LRU | n/a |
| KV offload → CPU (L2) | Y (`--kvstore-*`) | Y (2 paths) | Y (HiCache) | Y (`host_cache_size`) | Y (`num_cpu_blocks`) | Y (`host_kvcache_swap_space_gb`) | n/a |
| KV offload → disk/remote (L3) | **P — mooncake only** | Y (fs / object / p2p tiers) | **Y — 9 backends + dynamic** | P | — | Y (`disk_offload_dir`) | n/a |
| **Recurrent-state offload to host** | **Y — unique** | — | Y (`HiMambaRadixCache`) | — | — | — | n/a |
| KV quantization | fp8_e4m3, **mxfp8** | fp8_e4m3, fp8_e5m2 | fp8 e4m3/e5m2, **mxfp8, nvfp4** | fp8/int8/int4 | int4/int8 (`quant_policy`) | unverified | n/a |
| KV-event publishing (for routers) | Y (ZMQ) | Y (ZMQ) | Y | Y | — | unverified | n/a |
| Cache-aware routing | **unverified (external SMG wheel)** | — in-tree | **Y** (`sgl-model-gateway/src/policies/cache_aware.rs`) | Y (`serve/router.py`) | — | unverified (`serve/router/`) | — (`pd_router.py` is a free-node `Pool.acquire()`, not cache-aware) |
| Encoder / multimodal cache | Y | Y | Y | Y | Y | unverified | n/a |

### The hybrid-KV capacity trade, stated precisely

This is the most consequential row in the whole document for Qwen3.5/3.6, so it
gets its own treatment.

vLLM's `align` mode caches recurrent state only at block boundaries and only for
the last token of a scheduler step (`vllm/config/cache.py:139-150`), and sizes
the mamba page at `page_size_bytes * (2 + num_speculative_blocks)`
(`vllm/v1/kv_cache_interface.py:715`) — a small constant number of state copies,
independent of sequence length.

TokenSpeed's **flat** plan does something stronger and more expensive. In
`python/tokenspeed/runtime/configs/flat_memory_plan.py`, state components are
`ComponentSpec`s with `const_bytes > 0` packed into the *same page-id space* as
KV rows (`plan_tensors`, L230), and `solve_page_geometry` (L127) inflates the
page size until the widest linear KV row covers the widest constant state row.
The module docstring is candid that this is "the vLLM hybrid layout" and "vLLM
align" — for *page geometry*. What differs is **density**: every page carries a
state snapshot for every `linear_attention` layer, not just boundary pages. That
is what buys state-resumable prefix caching (any cached prefix boundary can
restore recurrent state without a re-scan), and it is what the measurement
already on record costs: **74% of a 208 MB page by bytes, 135,616 KV tokens vs.
vLLM's 478,162 on the same card and model.** A 3.5x capacity difference.

The knob that would amortise this **exists in name and is dead code**:

```
python/tokenspeed/runtime/utils/server_args.py:96   mamba_track_interval: int = 256
python/tokenspeed/runtime/utils/server_args.py:1022 --mamba-track-interval
python/tokenspeed/runtime/utils/env.py:66           "mamba_track_interval": ...
```

`grep -rn track_interval` over the entire repository returns exactly those three
lines. Nothing consumes it — not the Python runtime, not the C++ scheduler, not
the tests. The help text ("The interval to track the mamba state during decode")
describes precisely the fix, and it is not wired to anything. This is the same
failure mode already recorded for `sm_120` allowlists and for prefix caching on
hybrids: a feature that reads as present and is unreachable.

The **radix** (non-flat) path is different again — it sizes a separate
`MambaChunkAllocator` pool (`csrc/resource/allocator/mamba_chunk_allocator.h`)
against `--mamba-full-memory-ratio` (default 0.9) with chunk granularity
`max(FLA_CHUNK_SIZE, block_size)`
(`server_args.py:312`), decoupling state count from page count. Which of the two
paths a given build uses is the `TOKENSPEED_FLAT_KVCACHE` question above.

---

## C. Speculative decoding

| Algorithm | TokenSpeed | vLLM | SGLang | TRT-LLM | TurboMind | MAX |
|---|---|---|---|---|---|---|
| EAGLE / EAGLE3 | Y (EAGLE3) | Y | Y (+ multi-layer) | Y | Y | Y |
| MTP / NextN | Y | Y | Y (+ frozen-KV MTP) | Y | Y (deepseek, qwen3_5) | Y |
| DFlash | Y | Y | Y | Y | — | Y |
| DSpark | — | Y | Y | Y (`dspark.py`) | — | — |
| **n-gram** | **—** | Y (CPU + GPU) | Y (+ C++, external corpus) | Y | — | — |
| **Suffix decoding** | **—** | Y | — | Y (`suffix_automaton.py`) | — | — |
| Medusa | — | Y | — | unverified | — | — |
| Standalone draft model | **—** | Y | Y | Y (draft-target) | — | Y |
| PARD / lookahead | — | — | — | Y (`pard.py`) | — | — |
| Tree drafting (topk > 1) | **— (`--speculative-eagle-topk` `choices=[1]`)** | Y | Y | Y (+ dynamic tree) | unverified | unverified |
| **Adaptive / dynamic spec length** | **—** | Y (`num_speculative_tokens_per_batch_size`) | Y (`--speculative-adaptive`) | Y (`speculation_gate.py`, `auto_heuristic.py`) | — | unverified |
| Spec decode + structured output | unverified | Y (explicit, bitmask sized for draft tokens) | Y | unverified | — | unverified |
| Acceptance metrics | Y (3 counters) | Y (+ per-position acceptance vector) | Y | Y | Y | unverified |

**TokenSpeed detail.** `--speculative-algorithm choices=["EAGLE3","MTP","DFLASH"]`
(`server_args.py:1561`). All three require draft weights. The drafter interface
is small and clean — `BaseDrafter` in
`python/tokenspeed/runtime/execution/drafter/base.py` is 83 lines with three
abstract methods (`get_candidates`, `run`, `draft`) — so *adding* an algorithm is
cheap. What is missing is not the plumbing.

Two absences matter more than the algorithm count:

- **No weight-free drafter.** n-gram and suffix decoding need no checkpoint and
  no extra VRAM. On agentic workloads (long shared context, heavy verbatim
  copying from tool output) they are frequently the best drafter available, and
  they are the *only* drafter available for a checkpoint that ships no MTP/EAGLE
  head — which per `model-coverage.md` is exactly Gemma 4's situation.
- **No adaptive speculation length.** Every other engine in this table gates or
  scales speculation by batch size or measured acceptance. TokenSpeed's
  `--speculative-num-steps` is static. This is the mechanism that stops
  speculation from becoming a throughput *loss* at high concurrency — and the
  concurrency-dependence of TokenSpeed's own throughput is already on record.

---

## D. Quantization and weight loading

| Format | TokenSpeed | vLLM | SGLang | TRT-LLM | TurboMind | MAX |
|---|---|---|---|---|---|---|
| fp8 (per-tensor / block) | Y (`fp8`, `w8a8_fp8`) | Y | Y (+ blockwise) | Y | Y | Y |
| mxfp8 | Y | Y | Y | Y | — | unverified |
| nvfp4 | Y | Y (`modelopt_fp4`) | Y (+ `nvfp4_online`, `petit_nvfp4`) | Y | — | unverified |
| mxfp4 | Y | Y | Y | Y | — | unverified |
| modelopt mixed | Y | Y | Y | Y | — | — |
| compressed-tensors | Y (wNa16 scheme only) | Y (full) | Y | Y | — | — |
| **AWQ** | **—** | Y | Y (+ marlin) | Y | Y (`lite/quantization/awq.py`) | unverified |
| **GPTQ** | **—** | Y | Y (+ marlin) | Y | Y | unverified |
| **int8 W8A8 / smoothquant** | **—** | Y | Y (+ blockwise int8) | Y | Y | unverified |
| bitsandbytes | — | Y | Y | — | — | — |
| GGUF | — | **— (removed)** | Y | — | — | Y |
| torchao | — | Y | Y (separate flag) | — | — | — |
| Online / dynamic quantization | — | Y (6 shorthands) | Y | Y | Y (`lmdeploy lite`) | unverified |
| **Registered methods (count)** | **7** | **~29** | **~25** | many | ~6 | unverified |

**Weight loading.** TokenSpeed `--load-format` accepts
`auto, pt, safetensors, npcache, dummy, extensible` (`server_args.py:803`). vLLM
has 8 loader classes covering runai-streamer, sharded-state, tensorizer,
bitsandbytes, modelexpress; SGLang has 16 `--load-format` values including
`remote`, `remote_instance`, `layered`, `flash_rl`, `fastsafetensors`,
`sharded_state`, `gguf`. TokenSpeed's `extensible` format is a plugin hook
(`runtime/models/extensible.py`) rather than a transport.

For sm_120 specifically this row is less alarming than the counts suggest:
fp8/nvfp4/mxfp4 are the formats that matter on Blackwell, and TokenSpeed has all
three. The exposure is **checkpoint availability** — a large share of community
Qwen and Gemma quantizations ship as AWQ or GPTQ, and TokenSpeed cannot load
them at all.

---

## E. Parallelism

| Axis | TokenSpeed | vLLM | SGLang | TRT-LLM | TurboMind | MAX |
|---|---|---|---|---|---|---|
| TP | Y | Y | Y | Y | Y | Y |
| **Split TP (attn / dense / MoE separately)** | **Y — `--attn-tp-size` `--dense-tp-size` `--moe-tp-size`** | P (`moe` only via EP) | P (`moe_dense_tp_size`) | P | Y (`attn_tp_size`/`mlp_tp_size`/`moe_tp_size`) | — |
| **PP** | **— (no flag, no code)** | Y | Y | Y | Y | unverified |
| EP | Y | Y | Y | Y | Y | unverified |
| EPLB (expert rebalance) | Y (`--enable-eplb`, algorithms + recorder) | Y | Y (+ elastic EP) | Y (wide-EP) | unverified | — |
| DP | Y (3 LB methods) | Y (+ coordinator, hybrid LB) | Y (4 LB methods) | Y | Y | Y (`data_parallel_degree`) |
| Attention-DP | Y | Y | Y | Y | Y | unverified |
| Context / sequence parallel | **P — scaffolding, `max_num_seqs` must be 1** | Y (decode CP + prefill CP) | Y (attn CP + decode CP + DSA prefill CP) | Y (`cp_config`, helix) | — | unverified |

**TokenSpeed detail.** Split TP is a genuine differentiator: the three-way
`attn / dense / moe` split plus `--mm-encoder-tp-mode data` lets a Kimi-K3-class
deployment run a wide-QKV vision tower at TP1/DP8 inside a TP8 text model
(`docs/serving/parallelism.md`). Only TurboMind matches this granularity.

Pipeline parallelism is **entirely absent** — `grep -rn "pipeline_parallel\|pp_size"`
over `python/` returns nothing. For a single RTX PRO 6000 this costs nothing, so
it does not appear in the shortlist; for multi-node it is a hard ceiling.

Context parallelism exists as `Mapping.cp_size`
(`runtime/distributed/mapping.py:140-170`) and is wired into `comm_manager.py`
and `models/base/comm_ops.py`, but `server_args.py:710` rejects
`has_attn_cp and max_num_seqs > 1`. It is a single-request long-context path,
not a serving axis.

---

## F. Structured output, tool calling, reasoning

| Feature | TokenSpeed | vLLM | SGLang | TRT-LLM | TurboMind | MAX |
|---|---|---|---|---|---|---|
| Grammar backends | **1 (xgrammar), default `none`** | **4** (xgrammar, guidance, outlines, lm-format-enforcer) | **3** (xgrammar, outlines, llguidance) | 2 (xgrammar, llguidance) | 1 | xgrammar (`max/_xgrammar`) |
| Constraint kinds | json, grammar, structural_tag | 6 (json, regex, choice, grammar, json_object, structural_tag) | Y + jump-forward | Y | Y | unverified |
| **Grammar inside CUDA graph** | **Y — `capturable_grammar.py`, unique** | — | — | — | — | — |
| Reasoning-aware structural tags | Y (`reasoning_structural_tag.py`) | Y (`enable_in_reasoning`) | Y (`reasoner_grammar_backend.py`) | unverified | unverified | — |
| Tool-call parsers | **in SMG wheel — unverified** | **~40** | **~30** | **18** | **12** | unverified |
| Reasoning parsers | Y (`--reasoning-parser`, count unverified) | **24** | **24** | Y | Y | unverified |

**TokenSpeed detail.** `--grammar-backend` defaults to `"none"`
(`server_args.py:222`) — structured output is off unless asked for. The one
thing TokenSpeed has that nobody else does is
`runtime/grammar/capturable_grammar.py` (569 lines): grammar bitmask application
that survives CUDA-graph capture, instead of forcing an eager fallback for
constrained requests. On a workload that is mostly tool calls this is a real
throughput asset and is worth writing up as a differentiator.

---

## G. Multimodal

| Feature | TokenSpeed | vLLM | SGLang | TRT-LLM | TurboMind | MAX |
|---|---|---|---|---|---|---|
| Vision | Y | Y | Y | Y | Y | Y |
| Audio | Y (`qwen3_audio`, `qwen3_asr`, `qwen3_omni`) | Y | Y (7 families) | Y | P | unverified |
| Video | Y (mrope handles video) | Y (+ EVS pruning) | Y (+ EVS) | Y | Y | unverified |
| MM processor count | ~6 families | many | **51** | many | many | unverified |
| MM hashing / mm prefix caching | Y (`multimodal/hash.py`) | Y | Y | unverified | unverified | unverified |
| Encoder CUDA graph | Y (`encoder_cudagraph.py`) | unverified | Y | unverified | — | unverified |
| **Encoder disaggregation (EPD)** | **Y** — `--disaggregation-mode encode` | **Y** — EC-transfer, 1P1D only | **Y** — encode server + adaptive dispatch | — | — | — |
| Encoder TP mode | Y (`weights` / `data`) | Y (same two) | Y (`--mm-enable-dp-encoder`) | — | — | — |
| Embedding transport | Y (shm + RDMA + mooncake) | Y (shm, GPU IPC) | Y | — | — | — |

EPD is a genuine three-way TokenSpeed strength — a full `runtime/epd/` package
with its own scheduler (`encode_scheduler.py`), rank-synced admission
(`prefill_admission.py`, 1092 lines), and mooncake transport. But vLLM and
SGLang both have it now, so it is parity, not differentiation.

---

## H. Observability

| Feature | TokenSpeed | vLLM | SGLang | TRT-LLM | TurboMind | MAX |
|---|---|---|---|---|---|---|
| Prometheus metrics | Y (17 families) | **Y (~40 families)** | Y | Y | Y (+ Grafana dashboards shipped) | Y (`serve/telemetry/`) |
| KV block lifetime / reuse-gap histograms | — | Y (`v1/core/kv_cache_metrics.py`) | — | unverified | — | — |
| **OpenTelemetry tracing** | **—** | Y (`--otlp-traces-endpoint`) | Y (`--enable-trace`) | Y (`llmapi/tracing.py`) | unverified | unverified |
| Torch profiler hooks | Y (`start_profile`/`stop_profile`) | Y (+ per-stage config) | Y (+ per-stage) | Y | Y (`profiler.py`) | Y (`max/profiler/`) |
| NVTX ranges | Y (`--enable-nvtx`) | Y (`--enable-layerwise-nvtx-tracing`) | unverified | Y | — | unverified |
| Expert distribution recorder | Y (4 modes) | P | Y (4 modes) | Y (`expert_statistic.py`) | — | — |
| Per-request time stats | Y (`--enable-log-request-stats`, incl. preemption time) | Y | Y | Y | Y | unverified |
| NaN detection | **Y (`--enable-nan-detection`, + metric)** | — | unverified | — | — | — |

TokenSpeed's metric names are deliberately vLLM-shaped
(`runtime/metrics/collector.py` even annotates where it diverges: "vLLM has no
direct equivalent"), so dashboards port. The two gaps are distributed tracing
(absent entirely — `grep opentelemetry` returns nothing) and KV-block lifetime
histograms.

---

## I. LoRA / adapters

| Feature | TokenSpeed | vLLM | SGLang | TRT-LLM | TurboMind | MAX |
|---|---|---|---|---|---|---|
| Multi-LoRA serving | **— (nothing)** | Y | Y | Y (`_torch/peft/lora`) | Y (`pytorch/adapter/`) | Y (`pipelines/lora/`) |
| Dynamic load/unload API | — | Y (2 endpoints) | Y (3 endpoints) | unverified | unverified | unverified |
| CUDA graph + LoRA | — | Y | Y (dual graph variants) | unverified | unverified | unverified |
| LoRA + MoE | — | Y | Y | unverified | — | unverified |
| LoRA + quantized base | — | Y | unverified | Y | Y | unverified |

`find /workspace/roy/tokenspeed/python -iname "*lora*"` returns **nothing**. The
only occurrences of the string are (a) a `lora_request` field in `bench.py` used
to drive *other* engines in A/Bs, and (b) inherited comments in
`vocab_parallel_embedding.py` and `rotary_embedding.py`. TokenSpeed is the only
engine in this table with no adapter support at all.

---

## J. RL weight transfer

| Feature | TokenSpeed | vLLM | SGLang | TRT-LLM | TurboMind | MAX |
|---|---|---|---|---|---|---|
| Weight-transfer backends | Y (`nccl`, `ipc`) | Y (`nccl`, `ipc`, `sparse_nccl`, plugin) | Y (distributed / tensor / disk / ipc) | unverified | — | unverified |
| HTTP control plane | **Y — 25 endpoints** | Y (dev/RLHF router) | Y | unverified | — | unverified |
| Pause / resume generation | Y | P | Y | unverified | — | unverified |
| Memory occupation release/resume | Y | Y (sleep mode) | Y | unverified | — | unverified |
| **Weight version stamped on responses** | **Y** (`--weight-version`, `/update_weight_version`) | — | Y | — | — | — |
| Draft-model weight update | unverified | Y | unverified | unverified | — | — |
| checkpoint-engine integration | — | — | Y | — | — | — |

This is TokenSpeed's strongest parity row. `runtime/entrypoints/control_server.py`
exposes `init_weight_transfer_engine`, `start_weight_update`, `update_weights`,
`finish_weight_update`, `update_weights_from_{distributed,tensor,disk}`,
`init/destroy_weights_update_group`, `pause`/`resume`,
`pause_generation`/`continue_generation`,
`release/resume_memory_occupation`, `get/update_weight_version`. Weight-version
stamping on generation responses (commit `500a55ee`) is something vLLM does not
do at all.

---

## Where TokenSpeed is genuinely ahead

Stated plainly, because this half of the goal is easy to lose under a gap list.

1. **Typed-FSM C++ scheduler.** Request lifecycle, KV ownership and overlap
   timing as a compile-checked state machine (`csrc/fsm/`, `csrc/scheduler/`).
   Every other engine here enforces KV-reuse safety by convention and review.
   This is the extensibility bet, and it is real.
2. **State-resumable prefix caching for recurrent models.** vLLM's `align` and
   SGLang's `MambaRadixCache` cache state at coarse boundaries; TokenSpeed can
   resume from any cached page. It costs 3.5x KV capacity today (§B) — but the
   capability is strictly stronger, and the fix is an amortisation knob, not a
   redesign.
3. **Recurrent-state host offload.** `--enable-mamba-l2` with its own host
   allocator, layout and IO backend (`csrc/resource/allocator/mamba_host_allocator.h`,
   `runtime/cache/mamba_cache_host.py`). vLLM has nothing equivalent. SGLang's
   `HiMambaRadixCache` is the only comparable thing in the field.
4. **CUDA-graph-capturable grammar** (`runtime/grammar/capturable_grammar.py`,
   605 lines). Constrained decoding without dropping out of graph replay.
   Unique here.
5. **Three-way split TP** (`attn` / `dense` / `moe`) plus per-modality encoder TP
   mode. Only TurboMind is comparable.
6. **Full-capture prefill CUDA graphs** (`--enable-prefill-graph`,
   `--prefill-graph-capture-sizes`, `runtime/execution/prefill_graph.py`). Not
   unique but not common: SGLang has `prefill_cuda_graph_runner.py`; vLLM's
   default `FULL_AND_PIECEWISE` captures *full* graphs for decode batches only
   and runs prefill/mixed batches **piecewise**
   (`vllm/config/compilation.py:630-632`).
7. **RL control plane breadth** and weight-version stamping (§J).
8. **NaN detection as a first-class serving feature** (`--enable-nan-detection`
   plus a `num_nan_aborted_requests` counter) rather than a debug flag.

### A near-miss worth recording: eviction policies

`runtime/cache/evict_policy.py` defines six strategies — LRU, LFU, FIFO, MRU,
FILO and **Priority** — and `runtime/cache/prefix_cache.py:151-162` wires all six
behind a `CacheInitParams.eviction_policy` string. It would be a real
differentiator if it ran. It does not:

- `runtime/cache/prefix_cache.py` is **imported by nothing** in the live runtime.
  The only reference anywhere is a `TYPE_CHECKING` import of `TreeNode` from
  `evict_policy.py:34`. It is the legacy Python radix tree, superseded by the
  C++ scheduler.
- The live prefix cache is `csrc/resource/kv_prefix_cache/`, and its
  `ResourceManager<RType>::Evict` (`eviction.h:119-137`) walks `lru_leaves_`
  oldest-first. **LRU only.** There is no policy hook and no CLI flag.

Same failure mode as `--mamba-track-interval` (§B) and the dead
`PriorityStrategy` referenced in shortlist item 4: capability that reads as
present in Python and is unreachable from the C++ control plane. Worth a
one-pass audit of the whole `runtime/cache/` Python layer for other modules
stranded by the C++ migration.

---

# Prioritised shortlist

Ranked for **Qwen3.5-27B / Qwen3.6-35B-A3B / Gemma 4 on one RTX PRO 6000
Blackwell (sm_120, 96 GB)**. Multi-GPU and multi-node concerns (PP, wide-EP,
disagg transports) are deliberately demoted — they do not move this deployment.

Cost is engineering time for someone already fluent in this tree, and includes
tests. Treat the LOC figures as order-of-magnitude.

### 1. Amortise recurrent-state snapshots — wire `--mamba-track-interval`

**Why first.** It is the only item here worth a 3.5x change in a first-order
resource. 135,616 vs. 478,162 KV tokens on the same card and model is the
difference between serving ~10 concurrent 12k-token agent sessions and serving
~35. Both Qwen3.5 and Qwen3.6 are hybrids, so it lands on two of the three
target models, and at high concurrency — the operating point where TokenSpeed
already wins — KV capacity *is* throughput.

**Surface.**
- `python/tokenspeed/runtime/configs/flat_memory_plan.py` — `components_from_layers`
  / `plan_tensors` must emit state rows at a page *stride* rather than per page.
- `python/tokenspeed/runtime/configs/hybrid_cache_plan.py` — `_build_layout` /
  `plan_flat_hybrid_cache` slab sizing follows.
- `tokenspeed-scheduler/csrc/resource/radix_tree/mamba_slot.h` and
  `csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.{h,cpp}` — `Match` must
  return the nearest *snapshotted* ancestor and report the re-scan span;
  `AlignMambaCacheSeqlen` already hints at this shape.
- `csrc/scheduler/operations/forward.cpp` — the prefill first-chunk event must
  schedule the re-scan of tokens between the snapshot and the match point.
- Plumb `mamba_track_interval` through `SchedulerConfig`
  (`csrc/scheduler/types.h:77`), which today carries `mamba_cache_chunk_size` but
  not the interval.

**Cost.** ~1200–1800 LOC across Python and C++, 2–3 weeks. The genuinely hard
part is not the allocator, it is the correctness argument: a prefix hit now
implies replaying `k` pages of recurrent scan before the first new token, so
TTFT gets a length-dependent term. Budget a week of that time for a
snapshot-stride sweep (stride 1 = today, stride ∞ = vLLM `align`) measuring the
TTFT/capacity Pareto rather than picking 256 because it is the current default.

**Do this before anything else on hybrids**, and re-run any prior TokenSpeed
throughput comparison afterwards — the "+3.8% at C=128, −7.5% at C=32" result
was measured against a 3.5x KV handicap and may not survive its removal.

### 2. A weight-free drafter (n-gram) plus adaptive speculation length

**Why.** Speculative decoding is worth +27–45% here when it works, and all three
of TokenSpeed's drafters need a checkpoint. Gemma 4 ships no MTP/EAGLE head at
all, so today it gets **zero** speculation on TokenSpeed while vLLM offers it
n-gram, suffix, *and* a `gemma4_mtp` / `gemma4_dspark` drafter. Separately, a
static `--speculative-num-steps` is the known way to turn a decode win into a
throughput loss as batch grows — every other engine in the table gates this and
TokenSpeed does not.

**Surface.**
- `python/tokenspeed/runtime/execution/drafter/ngram.py` — new, implementing the
  3-method `BaseDrafter` (`drafter/base.py`, 83 lines). No model, no weights, no
  extra CUDA graph; a rolling-hash suffix table over the request's own tokens.
- `python/tokenspeed/runtime/utils/server_args.py:1561` — add `NGRAM` to
  `choices`, plus `--speculative-ngram-{min,max}-match-len`.
- `python/tokenspeed/runtime/execution/factory.py` and
  `cuda_graph_wrapper.py:1041` — drafter dispatch (`spec_algo` is already a
  string switch there).
- Adaptive length is a separate, smaller change in `engine/event_loop.py` and
  `runtime/execution/model_executor.py`: read the acceptance counters that
  `metrics/collector.py:221-241` already maintains and clamp draft length by
  effective batch size. Copy the shape of vLLM's
  `num_speculative_tokens_per_batch_size` (a piecewise `[lo, hi, k]` table) —
  it is declarative and testable.

**Cost.** n-gram ~500–700 LOC, ~1 week. Adaptive length ~200–300 LOC, ~3 days.
Highest value per line in this list.

### 3. Port Gemma 4

**Why.** Pure capability: `python/tokenspeed/runtime/models/` has no Gemma of any
generation, and one of the three named target models therefore does not run at
all. vLLM has `gemma4.py`, `gemma4_mm.py`, `gemma4_unified.py`, `gemma4_mtp.py`,
`gemma4_dspark.py`, a `gemma4` reasoning parser and a `gemma4` tool parser — a
complete reference to port from.

**Surface.** Per `model-coverage.md`, two of the four hard config fields are
already served (`final_logit_softcapping` by `layers/logits_processor.py`,
`sliding_window` by `kv_cache/mha.py`'s `sliding_window_tokens`). The two that
are new:
- `attention_k_eq_v: true` — no concept exists in this tree. Do **not** work
  around it by materialising V; done properly it halves KV bytes per token on
  these checkpoints and is the largest capacity win available on Gemma 4. Lands
  in `runtime/layers/attention/kv_cache/mha.py` and `configs/paged_cache_spec.py`.
- `global_head_dim: 512` alongside `head_dim: 256` — per-layer-type KV geometry.
  `flat_memory_plan.py`'s `components_from_layers` already builds per-layer
  `ComponentSpec`s with differing byte counts for the hybrids, so the plan layer
  can express it; the open question is which sm_120 attention backend accepts
  head_dim 512, and that is **unverified** and should be tested first.
- New: `runtime/models/gemma4.py`, `runtime/configs/gemma4_config.py`, plus
  audio/vision wiring (these are omni checkpoints, not text-only).

**Cost.** ~2500–3500 LOC, 3–4 weeks, and it has a hard prerequisite: confirm a
head_dim-512 attention kernel exists on sm_120 before starting. Per
`model-coverage.md` this should be sequenced *after* the Qwen3.5/3.6 bring-up,
which isolates the sm_120 question from the porting question.

### 4. Scheduling policy: priority ordering and real preemption

**Why.** TokenSpeed's scheduler is FCFS with no policy field
(`csrc/scheduler/types.h:77-136`), and its only preemption is a two-round
deadlock breaker. That is fine on a uniform benchmark and poor under the mixed
agentic load this engine is explicitly built for, where a 200k-token background
summarisation currently head-of-line-blocks every interactive request behind it.
vLLM has `--scheduling-policy priority`; SGLang has seven policies plus two
retraction policies plus a preemption threshold.

**Surface.**
- `csrc/scheduler/types.h` — add `priority` to `RequestSpec`
  (`csrc/scheduler/request_spec.h`) and a policy enum to `SchedulerConfig`.
- `csrc/scheduler/operations/forward.cpp` — `newForwardOperation` takes
  `std::vector<Request*> candidates`; the ordering decision belongs here, in one
  place.
- Promote `scheduleRetract` (already writes pages back to the host allocator —
  genuine swap-preemption, which vLLM V1 no longer has) from
  exhaustion-triggered to policy-triggered.
- `python/tokenspeed/runtime/engine/io_struct.py` + `request.py` for the wire
  field; `--scheduling-policy` in `server_args.py`.
- The FSM makes this safer than it sounds: reordering candidates cannot violate
  KV ownership, because ownership is typed.

**Cost.** ~600–900 LOC, mostly C++, ~1.5 weeks. A request priority also gives
the C++ evictor (`csrc/resource/kv_prefix_cache/eviction.h`, LRU-only today) a
sort key it does not currently have, which is the prerequisite for the
priority-aware eviction that the *dead* Python `PriorityStrategy` was written
for. Do not resurrect the Python class — port the idea into the C++ evictor.

### 5. LoRA / multi-adapter serving

**Why last, and why still on the list.** It buys nothing on a fixed-weights
throughput benchmark, so it is correctly below the four items above. But it is
the single largest *capability* hole in the matrix — every other engine here has
it, including TurboMind and MAX — and it is the one gap that disqualifies
TokenSpeed outright from a whole class of deployment rather than making it
slower. On one 96 GB card, serving 8 task-specific Qwen3.5-27B adapters off one
base is a materially better use of the card than 8 processes.

**Surface.** This is a new subsystem, not a patch.
- New `python/tokenspeed/runtime/lora/` — adapter registry, GPU adapter pool
  (bounded by `--max-loras`), per-batch index construction, punica-style
  grouped-GEMM ops behind `tokenspeed-kernel` (the `AGENTS.md` dependency rule
  means the kernels must land in `tokenspeed-kernel/ops/` and be registered, not
  imported directly).
- `runtime/layers/linear.py` (1265 lines) — every `*Parallel*Linear` grows a
  delta path.
- `runtime/execution/cuda_graph_wrapper.py` (1296 lines) — SGLang's answer is
  dual `lora`/`nolora` captured variants; expect to copy that shape.
- `csrc/scheduler/` — batches must be groupable by active adapter set, which is
  a scheduling constraint, not just an execution one.
- `runtime/entrypoints/control_server.py` — `/load_lora_adapter`,
  `/unload_lora_adapter`.

**Cost.** ~3000–4000 LOC, 5–7 weeks, and the CUDA-graph interaction is where the
schedule will actually go.

---

## Deliberately not in the top five

- **Pipeline parallelism** (absent entirely). Zero value on one card. Real
  ceiling the moment a model exceeds 96 GB — revisit when multi-node matters.
- **AWQ / GPTQ loading** (absent). fp8/nvfp4/mxfp4 are the right formats on
  sm_120 and TokenSpeed has them; this is a checkpoint-availability tax, not a
  performance gap. ~1 week each if a specific checkpoint forces it.
- **OpenTelemetry tracing** (absent). ~400 LOC, 3 days, no throughput effect.
- **Cache-aware routing.** Single-replica deployment; and the engine already
  emits the ZMQ KV events a router would consume (`--kv-events-config`). The
  gateway side is unverifiable from this tree anyway.
- **Grammar backend breadth** (1 of 4). xgrammar is the fastest of them and
  TokenSpeed's capturable-grammar path is ahead of everyone. Adding backends is
  parity theatre.
- **Sampling completeness — worth a bug, not a roadmap item.**
  `--sampling-backend` defaults to a mode where "min_p and penalties [are]
  silently ignored" (`server_args.py:1410-1432`); only `flashinfer_full` and
  `triton_full` honour them, and the full variants allocate a
  `[max_req_pool_size, vocab_size]` int32 buffer. Silently dropping a requested
  sampling parameter is a correctness defect regardless of priority. Fix by
  erroring on unsupported params rather than ignoring them: ~50 LOC.

## Unverified — do not cite these as findings

- SMG gateway internals: routing policies (incl. cache-aware), tool-call parser
  inventory, retry/circuit-breaker behaviour. Binary wheel
  `tokenspeed-smg==1.8.0.post20260728`, not vendored.
- TokenSpeed reasoning-parser count (the names in `cli/serve_smg.py` are
  defaults passed *to* SMG, not an inventory).
- Whether TokenSpeed's structured output composes with speculative decoding.
- Modular MAX rows marked unverified: quantization format list, KV quantization,
  TP/PP/EP axes, tracing, spec-decode drafter management.
- TRT-LLM Medusa; TurboMind preemption and overlap scheduling.
- Which sm_120 attention backend, if any, accepts `head_dim=512` (blocks
  shortlist item 3).
- Whether the local TokenSpeed build has `TOKENSPEED_FLAT_KVCACHE` on. Every
  capacity number in §B assumes it does; confirm with
  `python -c "import tokenspeed_scheduler as s; print(s.FLAT_KVCACHE)"`.
