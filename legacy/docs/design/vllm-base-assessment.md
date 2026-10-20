<!-- English canonical. docs/design/* are English-only in this repo (no Chinese mirror). -->
# Can UnieInfra's Architecture Sit on vLLM? — A Base Decision for an Agent-Oriented Engine

Status: **Study / read-only recon complete, no decision taken** · Date: 2026-07-25 · Owner: royshih

## Verdict

**Yes — our kernel architecture is portable to vLLM, and for the kernel layer alone a
*pure plugin* is genuinely viable, not a fork.** The selection core
(`registry.py` / `selection.py` / `platform.py` / `signature.py`, 2,573 LOC) imports
nothing but `torch` and the stdlib — there is not one `sglang` or `tokenspeed`
runtime import in it — so it lifts out of `tokenspeed-kernel` cleanly. And vLLM has
the seam we need: an out-of-tree layer class registered through `PluggableLayer`
(`vllm/model_executor/layers/linear.py:312`, `:417`, `:1611`) or an out-of-tree
`QuantizationConfig` registered through `register_quantization_config`
(`vllm/model_executor/layers/quantization/__init__.py:59`) yields a
`LinearMethodBase` whose `apply()` **is** the GEMM for every dense Linear in every
model (`linear.py:402`, `:597`, `:1763`), with an in-tree test proving the pattern
end to end (`tests/quantization/test_register_quantization_config.py:121-146`).
**But the kernel layer is not what this decision is about.** Two things I set out to
confirm did not survive contact with the code. First, the motivating insight —
"vLLM has an excellent skinny GEMM whose dispatch is a hardcoded (N,K) whitelist, so
a real registry would pick it automatically for Qwen3-8B" — is *half* right, and the
wrong half matters: the whitelist is a **C++ template-instantiation list**
(`csrc/libtorch_stable/bf16_skinny_gemm_entry.cu:29-39`, `:82-107`), so no selection
layer of any kind can route Qwen3-8B to that kernel without adding template
instantiations and recompiling vLLM's C++. Second, the radix-vs-APC premise: our
radix tree matches at **page granularity floored to whole pages**
(`csrc/resource/radix_tree/radix_tree.cpp:48-60`) with a default page size of **64**
(`server_args.py:117`), while vLLM's block-hash APC defaults to **16**
(`vllm/config/cache.py:47`) — so on match granularity vLLM is currently *finer*, not
coarser, and `hybrid_prefix_cache` turns out to mean hybrid *attention models*
(Mamba/GDN/SWA), not GPU+CPU tiering, which vLLM also has and in a broader form
(`vllm/v1/kv_offload/`, with ARC on its CPU tier). The radix tree's real, durable
advantage is narrower but sharp: **eviction structure**. vLLM's is one flat
free-block LRU with *zero* share-count awareness
(`vllm/v1/core/kv_cache_utils.py:184-204`), so a hot system prompt is evicted like
any cold block the moment its last referrer finishes; ours makes the shared trunk
structurally the last thing to go (leaf-first LRU with root-path ref-count pinning,
`csrc/resource/kv_prefix_cache/eviction.h:45-72`, `tree_node.h:186-190`). For an
agent workload — thousands of intermittently-active sessions behind one system
prompt — that is the right property, and it is the one thing here worth keeping a
base for.

**Recommendation: do not re-base yet.** Take the plugin route as a
*cheap, reversible experiment* — port the registry to stand alone and run our
kernels inside stock vLLM behind a plugin — because it costs little and it
de-risks the base decision either way. Do not decide the base until we have run an
**agent-shaped benchmark**, which we have never run: every experiment E1–E33 is
random prompts at ILEN 1024 / OLEN 256 with the prefix cache **explicitly flushed
before every concurrency level** (`benchmark/run_sweep.sh:53-54`), i.e. we have
been measuring, deliberately, the one scenario in which our prefix-cache machinery
cannot help.

---

## Method and honesty rules

This is a **read-only code and document study**. No GPU was used and no source was
modified — a decisive benchmark was running on GPU 3 throughout. Claims are labelled:

- **[validated]** — measured on our hardware; cites `docs/experiments-log.md`.
- **[code]** — I read the code, the path exists, and I cite `file:line`. It does
  *not* mean it works or that it is fast.
- **[inferred]** — my reasoning from code or arithmetic, explicitly flagged.

vLLM paths are relative to `third_party/vllm/` at submodule pin `d02df748bf9e`
(2026-07-24, between v0.26.0 and v0.27). vLLM is a **submodule** (`.gitmodules`),
not vendored, so a re-base means adopting it as a dependency or as a tracked fork —
not absorbing a snapshot. Where I could not determine something, I say so.

---

## 1. What the product is now — and why that changes the question

The framing has moved. UnieInfra is **an agent-oriented inference engine**, not a
general-purpose one. That reorders every axis:

| Agent workload property | What it stresses in an engine |
|---|---|
| Multi-turn loops resending history | Prefix reuse is the lifeline, not a nice-to-have |
| System prompt + tool definitions shared across thousands of sessions | Eviction must protect a hot shared trunk |
| Branching tool-call exploration | Tree-shaped, not linear, reuse |
| Heavy structured / JSON output | Constrained-decoding throughput, grammar compile latency |
| Short turns, many of them | TTFT dominates aggregate throughput; TPOT matters less |
| Session identity | Session/turn-aware scheduling priority and affinity |

`docs/design/unieinfra-architecture.md:5-12` still states the north star as a
general engine ("as easy as vLLM", `max RPS s.t. TTFT_p99 ≤ X ∧ TPOT_p99 ≤ Y",
hardware-universal). That doc should be updated: for an agent engine, **cache-hit
rate is a first-class KPI alongside RPS@SLO**, and hardware universality — the thing
that doc calls "the spine" (`:13`) — becomes a second-order concern relative to
context management.

The consequence for the base decision: the question is no longer "which base is the
better general engine" but **"which base gets us to the best agent inference engine
fastest, and which differentiators survive the choice."** The KDA / kernel-registry
moat survives *either* base. Radix, the C++ FSM scheduler, and the placement→comm
compiler survive *only* the tokenspeed base. So the decision hinges on what those
are worth for agent workloads — which §5 costs out honestly.

---

## 2. vLLM's extension points, enumerated

### 2.1 The plugin system

Five entry-point groups, all in `vllm/plugins/__init__.py:16-30`:

| Group | Line | Loaded where |
|---|---|---|
| `vllm.general_plugins` | `:18` | **Every process** — front end, EngineCore, Worker, model-inspect subprocess |
| `vllm.platform_plugins` | `:23` | Lazily, on first `current_platform` access |
| `vllm.io_processor_plugins` | `:20` | Front end only |
| `vllm.stat_logger_plugins` | `:26` | Front end only, async mode |
| `vllm.endpoint_plugins` | `:30` | API-server front end only; **strict opt-in** via `VLLM_PLUGINS` (`:121`) |

`load_general_plugins()` (`vllm/plugins/__init__.py:77-90`) is process-global-guarded
and **invokes** each entry point. Timing is exactly what we need: [code]

- `vllm/engine/arg_utils.py:765-767` — inside `EngineArgs.__post_init__`, **before**
  `create_engine_config()`, so before `VllmConfig`, before platform config hooks,
  before any model construction.
- `vllm/engine/arg_utils.py:2743` — before CLI arg parsing, with the comment at
  `:2740-2742` explicitly anticipating *"the plugin may add a new kind of
  quantization method to --quantization argument"*.
- `vllm/v1/engine/core.py:110-112` — **first statement** of `EngineCore.__init__`,
  before `self.model_executor = executor_class(vllm_config)` at `:125`.
- `vllm/v1/worker/worker_base.py:245-247` — before `worker_cls` resolution, device
  init, and model load.

This is a 1:1 structural match for our own hook,
`python/tokenspeed/runtime/execution/model_runner.py:49-70` (`_load_first_party_kernels`,
one guarded `import unieinfra_kernel` before `self.load_model()` at `:129-134`).
Porting the wiring is a few lines. [inferred]

Failure handling is lenient — `load_plugins_by_group` swallows exceptions
(`vllm/plugins/__init__.py:71-72`), matching our own tolerant discovery
(`unieinfra-kernel/unieinfra_kernel/_discovery.py`, per-module try/except).

### 2.2 Platform plugins

`resolve_current_platform_cls_qualname()` (`vllm/platforms/__init__.py:211-251`): an
out-of-tree platform, if exactly one activates, **wins unconditionally over any
builtin** (`:235-237`). `Platform` (`vllm/platforms/interface.py:134`) exposes a very
wide override surface — `get_attn_backend_cls` (`:373`), `get_device_communicator_cls`
(`:1040`), `get_punica_wrapper` (`:1012`), `supported_quantization` (`:170`),
`check_and_update_config` (`:576`), `pre_register_and_update` (`:546`),
`import_ir_kernels` (`:255`), `get_default_ir_op_priority` (`:1276`),
`fp8_dtype` (`:1087`), `use_custom_allreduce` (`:1103`), and ~60 more.

**However** — the only surface CI actually proves is narrow. The single in-tree
platform plugin (`tests/plugins/vllm_add_dummy_platform/`) overrides just `_enum`,
`device_name`, `device_type`, `dispatch_key`, `check_and_update_config`, and
`get_attn_backend_cls` (`dummy_platform.py:13-34`) — and its `get_attn_backend_cls`
has already **drifted from the base signature** (instance method, 9 old args vs the
base classmethod's 3 at `interface.py:373-378`). That is a useful signal about how
much of this surface is load-bearing versus aspirational. Real out-of-tree platforms
named in `docs/design/custom_op.md:266-269`: vllm-ascend, vllm-spyre, vllm-gaudi,
vllm-neuron, vllm-metal, plus non-official vllm-metax, vllm-kunlun, vllm-musa.

**Important for us: we do NOT want to be a platform plugin.** Declaring
`PlatformEnum.OOT` would take us off `CudaPlatform` and off every CUDA fast path.
We want to run *on* `CudaPlatform` and replace individual ops.

### 2.3 CustomOp — replace an op's implementation without a custom platform

Two global registries (`vllm/model_executor/custom_op.py:21-22`): `op_registry`
(keyed by op string) and `op_registry_oot` (keyed by **class name**). The swap
happens in `__new__` (`:109-128`) and is **unconditional — no platform check**:

```python
op_cls_to_instantiate = op_registry_oot[op_name]
return super().__new__(op_cls_to_instantiate)
```

So `@SiluAndMul.register_oot` on a subclass overriding `forward_cuda` fires on plain
CUDA. `register_oot` at `:323-352`. Coverage is broad: ~15 activations
(`layers/activation.py:78-632`), RMSNorm family (`layers/layernorm.py:36-171`),
RoPE (`rotary_embedding/base.py:14`), the MoE method objects
(`fused_moe/unquantized_fused_moe_method.py:45`,
`fused_moe/fused_moe_modular_method.py:33`), activation-fp8 quant
(`quantization/input_quant_fp8.py:29`), MLA (`attention/mla_attention.py:1299`),
conv, mamba/GDN.

**Two traps, both real.** [code]

1. `forward_oot` (`custom_op.py:169-172`) is reached *only* via
   `current_platform.is_out_of_tree()` (`:202`). On CUDA it is dead code — an
   out-of-tree class must override `forward_cuda`.
2. **The default disables custom ops entirely under Inductor.** `vllm/config/vllm.py:1256-1263`
   appends `"none"` to `custom_ops` whenever the Inductor backend is on and
   compilation is not `NONE`, so `default_on()` (`custom_op.py:291-304`) is False and
   `dispatch_forward` returns `forward_native` (`:194`). **A third-party
   `forward_cuda` override is then silently never called.** A plugin must pass
   `enforce_enable=True` (`:130`, `:185`) or require `custom_ops=["+silu_and_mul", ...]`.
   Additionally `register_oot` rewrites `cls.name` to the *class* name (`:346`), so
   the `+silu_and_mul` enable key silently becomes `+SiluAndMul`.

`PluggableLayer` (`custom_op.py:32-101`) is the sibling mechanism: whole-class
replacement with no `enabled()` gate. Registered layers include
`replicated_linear` (`linear.py:312`), `column_parallel_linear` (`:417`),
`row_parallel_linear` (`:1611`), `routed_experts`, `gate_linear`,
`vocab_parallel_embedding`, `multi_head_latent_attention`, and the mamba/GDN mixers.

**Gap worth knowing:** the swap key is `cls.__name__` (`custom_op.py:49`), and
`QKVParallelLinear` (`linear.py:1021`) and `MergedColumnParallelLinear` (`:660`) are
*subclasses* of `ColumnParallelLinear` that are **not themselves registered**. So
replacing `column_parallel_linear` does **not** capture `qkv_proj` or `gate_up_proj`
— which are exactly the E33 hot shapes. [code] The quantization-config route (§2.4)
has no such hole because it dispatches per layer *instance*.

### 2.4 Quantization method registry — the route that reaches every GEMM

`register_quantization_config(quantization: str)`
(`vllm/model_executor/layers/quantization/__init__.py:59`, exported at `:186-192`)
appends to the mutable `QUANTIZATION_METHODS` list (`:12-48`) and to
`_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` (`:56`), which is merged **last** in
`get_quantization_config` (`:181`) so custom entries win over in-tree ones. It also
appends to `current_platform.supported_quantization` (`:90-91`) so config validation
passes. Re-registering an existing name is allowed with only a debug log (`:84-86`).

The subclass must implement `get_quant_method(layer, prefix)`
(`base_config.py:179-192`), which is called once per layer at construction:

```python
# vllm/model_executor/layers/linear.py:273-279
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
```

and the returned object's `apply()` is invoked from every Linear forward —
`linear.py:402` (`ReplicatedLinear`), `:597` (`ColumnParallelLinear`, inherited by
`QKVParallelLinear` and `MergedColumnParallelLinear`), `:1763` (`RowParallelLinear`).
Same seam for embeddings (`vocab_parallel_embedding.py:278`), MoE
(`fused_moe/routed_experts.py:199`), attention KV-cache quant
(`attention/attention.py:191`, `:611`), MLA (`mla_attention.py:1079`).

Because `prefix` is the full state-dict name, an out-of-tree config can dispatch
**per layer** — e.g. our decode GEMV only on `qkv_proj` and `o_proj`. The in-tree
test `tests/quantization/test_register_quantization_config.py:31-146` does exactly
this shape of thing, including an end-to-end `vllm_runner(model, quantization="custom_quant")`
on an unquantized model, and the public docs walk it through
(`docs/features/quantization/README.md:76-231`).

**This is the answer to "is GEMM pluggable in vLLM?" — yes, via the quantization
seam, for both quantized and unquantized models.** The wart is cosmetic: the user
must pass `--quantization unieinfra` on a bf16 model. [inferred]

### 2.5 Attention backends, quantized-linear kernels, IR ops, compile passes

- **Attention:** `AttentionBackendEnum` (`vllm/v1/attention/backends/registry.py:34`)
  values are class-path strings; `register_backend` (`:233-285`) can add a backend or
  **override an in-tree one**, and a `CUSTOM = None` slot exists (`:120`). Caveat: the
  auto-selection *priority order* is a literal list per capability in
  `vllm/platforms/cuda.py:83-163`, so a registered OOT backend is reachable by name
  (`--attention-backend`) but **never auto-selected**.
- **Quantized linear kernels:** `register_linear_kernel(kernel_class, platform, kernel_type)`
  (`vllm/model_executor/kernels/linear/__init__.py:987-1029`) appends to one of eight
  priority lists (`:314-486`). It **appends**, i.e. lowest priority, and
  `LinearBackend` is a closed `Literal` of 21 names (`vllm/config/kernel.py:142-163`),
  so an OOT kernel cannot be requested by name.
- **IR ops — the most interesting finding.** vLLM now has a genuine op registry:
  `IrOp.registry` (`vllm/ir/op.py:156`), `register_impl(provider, supported=..., supports_args=...)`
  (`:244-302`), a hot-path `dispatch` (`:327-364`, annotated *"MUST BE FAST"*), a
  user-facing priority list `IrOpPriorityConfig` (`vllm/config/kernel.py:19-119`,
  CLI `--ir-op-priority` at `vllm/engine/arg_utils.py:1503`), and even
  `register_input_generator` / `generate_inputs` / `get_tolerance`
  (`vllm/ir/op.py:452`, `:456`, `:469`) — which is, structurally, **our KernelContract**.
  It covers exactly **two ops** today (`rms_norm`, `fused_add_rms_norm`,
  `vllm/ir/ops/layernorm.py:9`, `:43`) with four providers (`vllm/kernels/{vllm_c,aiter_ops,oink_ops,xpu_ops}.py`).
  `Platform.import_ir_kernels` (`interface.py:255`) is the OOT hook. **Nothing in
  `vllm/ir/ops/` defines a GEMM.**
- **torch.compile passes:** `CompilationConfig.inductor_passes: dict[str, str]`
  (`vllm/config/compilation.py:599-604`) accepts a callable *or* a qualified name
  (`:943-958`), installed under `Platform.pass_key` (`interface.py:177`) and appended
  by `vllm/compilation/backends.py:955-969`. So yes, a plugin can insert a fusion or
  op-replacement pass without forking. Built-in pass *enabling and ordering* remains a
  boolean if/else (`vllm/compilation/passes/pass_manager.py:135-192`, with the
  project's own `# TODO(luka) better pass enabling system.` at `compilation.py:185`).
- **Scheduler:** `SchedulerConfig.scheduler_cls: str | type` (`vllm/config/scheduler.py:117`,
  resolved at `:170-191`, used at `vllm/v1/engine/core.py:140`) is a fully open
  extension point. `SchedulerInterface` (`vllm/v1/core/sched/interface.py:37`) is a
  ~20-method ABC, and — critically — **the `KVCacheManager` is constructed inside the
  Scheduler** (`vllm/v1/core/sched/scheduler.py:271`), not injected. A custom
  `scheduler_cls` can therefore bring its own KV-cache manager and its own prefix-reuse
  structure, as long as it emits a valid `SchedulerOutput` (block ids,
  `vllm/v1/core/sched/output.py:193-223`). This is the single most important extension
  point for the agent story, and it is exercised in CI
  (`tests/plugins_tests/test_scheduler_plugins.py:12`, `:27`). [code]

---

## 3. How hardcoded is vLLM's kernel dispatch?

Roughly **19 distinct decision points** on the hot decode path. Tally:
**7 fully reachable from a supported extension point · 5 partial or workaround-only ·
7 require patching vLLM source.** The gap is concentrated in exactly one place.

| Decision point | file:line | Reachable? |
|---|---|---|
| `UnquantizedLinearMethod.apply` → `dispatch_unquantized_gemm()` | `layers/linear.py:229`; `layers/utils.py:342-348` | ⚠️ Not hookable itself (CUDA arm is a hardwired `F.linear`), but reachable by replacing the *method* (§2.4) or the *layer class* (§2.3) |
| ROCm 6-way skinny/aiter/tgemm chain | `layers/utils.py:149-194` | ❌ patch (env on/off only) |
| ROCm aiter-triton (m,k) whitelist | `layers/utils.py:101-120` | ❌ patch |
| CPU GEMM dispatch (5-way) | `layers/utils.py:228-339` | ❌ patch |
| **bf16 skinny GEMM (N,K) whitelist** | `csrc/libtorch_stable/bf16_skinny_gemm_entry.cu:29-39`, `:82-107` | ❌ **patch + C++ recompile** |
| `dsv3_fused_a_gemm` (N,K) whitelist | `csrc/libtorch_stable/dsv3_fused_a_gemm.cu:763-771` | ❌ patch + recompile |
| Model-file `skinny_max` dicts | `vllm/models/deepseek_v32/nvidia/attention.py:348-350`; `mtp.py:65-77` | ❌ patch |
| MoE router GEMM 6-tier chain + 3 shape sets | `fused_moe/router/gate_linear.py:36-48`, `:168-221` | ⚠️ whole class via `PluggableLayer` |
| Quantized linear kernel choice (8 priority lists) | `kernels/linear/__init__.py:508`, `:689` | ✅ `register_linear_kernel` (`:987`) |
| Attention backend class | `v1/attention/selector.py:101`, `:178`; `platforms/cuda.py:397` | ✅ `register_backend` |
| Attention auto-priority ordering | `platforms/cuda.py:83-163` | ❌ patch to be auto-selected |
| Runtime TRTLLM-vs-FlashInfer decode | `utils/flashinfer.py:499-501` (`num_tokens <= 256`) | ⚠️ boolean force only |
| MoE backend oracle | `fused_moe/oracle/unquantized.py:48-146`, `:192` | ⚠️ ABC, no registry; `MoEBackend` `Literal` closed (`config/kernel.py:122-140`) |
| MoE method class | `fused_moe/unquantized_fused_moe_method.py:45` | ✅ `CustomOp.register_oot` |
| MoE tuned-config JSON by `(E,N,dtype,block,device)` | `fused_moe/fused_moe.py:1036-1048`, `:1387-1389` | ✅ `VLLM_TUNED_CONFIG_FOLDER` (`:1080-1086`) |
| RMSNorm / fused_add_rms_norm | `layers/layernorm.py:74-115` → `ir/op.py:327` | ✅ `IrOp.register_impl` + `--ir-op-priority` |
| Activations native-vs-cuda | `layers/activation.py:117-159`; `custom_op.py:174-207` | ⚠️ swap via `register_oot`; on/off via `custom_ops` |
| torch.compile fusion passes | `config/compilation.py:599`, `:943-958` | ✅ `inductor_passes` |
| Scheduler / KV-cache manager | `config/scheduler.py:117`; `v1/engine/core.py:140` | ✅ `scheduler_cls` |

### 3.1 The skinny-GEMM finding, corrected

E33 recorded that vLLM's `bf16_skinny_gemm` is at the memory roofline (a statistical
tie with our Triton kernel at M=1: qkv 34.46 vs 34.34 µs, 86.2% vs 87.5% of peak
DRAM BW) and beats ours from M≥4, but that its dispatch is a hardcoded (N,K)
whitelist that never fires on Qwen3-8B. [validated] That is confirmed — and the
mechanism is worse than "dispatch":

```cpp
// csrc/libtorch_stable/bf16_skinny_gemm_entry.cu:29-38
static inline bool bf16_skinny_gemm_supported(int n, int k) {
  if (k == 12288 && (n == 768 || n == 1536 || n == 6144)) return true;
  if (k == 2048 && n == 2048) return true;   // q_b_proj (TP8)
  if (k == 6144 && n == 2624) return true;   // fused_qkv_a
  if (k == 7168 && n == 2112) return true;   // DSv3.2 fused_qkv_a
  if (k == 14336 && n == 7168) return true;  // DSv3.2 eh_proj
  if (k == 6144 && n == 512) return true;    // shared-expert gate_up
  return false;
}
```

`N` and `K` are **template parameters** — the launcher at `:82-107` instantiates
`SkinnyLoopUnroller<kNPB, kN, kK, 1, 32>` per pair, with explicit instantiations at
`bf16_skinny_gemm.cu:192-220`. The shape set is baked into the `.so`. Worse, the
*decision* to call it lives inside two model files
(`vllm/models/deepseek_v32/nvidia/attention.py:348-350`, `mtp.py:65-77`), gated on
`current_platform.is_device_capability_family(100)` — Blackwell **datacenter only**,
so it would not fire on our sm_120 parts even for DeepSeek-V3.2.

**Honest conclusion:** this kernel is not evidence that "a registry would pick it
automatically". A registry cannot pick what does not exist for the shape. It *is*
strong evidence for a different, still-valid thesis: **vLLM's kernel selection is
structurally per-model and per-shape-literal, so good kernels stay stranded.**
Our decode GEMV (`unieinfra-kernel/.../decode_gemv_triton.py`) reaches Qwen3-8B
precisely because it is declared with capability + trait gates rather than a shape
literal. That is the moat — and it is portable to either base.

### 3.2 The window on that moat is closing

`vllm/ir/` is the same idea, built by vLLM, three months old, covering two ops. Its
priority mechanism is a **user-supplied ordered list of provider names** — weaker
than our capability + trait + objective + origin scoring, and with no autotuning —
but it has `register_input_generator`, `generate_inputs`, and `get_tolerance`
(`vllm/ir/op.py:452-469`), which is our `KernelContract` by another name. Separately,
`vllm/kernels/helion/` (1,030 LOC, 7 fused ops with shape-keyed pre-tuned config
tables) is a real tuning layer. If we are betting on selection as a moat, we should
assume vLLM extends `IrOp` to GEMM within a few releases. [inferred]

**And we must be honest about our own side: our autotuning is not implemented.**
`_autotune_select` (`tokenspeed-kernel/python/tokenspeed_kernel/selection.py:776-800`)
*falls back to heuristic ranking and never benchmarks*:

```python
scored = _rank_by_objective(candidates, SelectionObjective.DEFAULT, platform, traits)
winner = scored[0][0]
logger.debug("[tokenspeed_kernel:autotune] falling back to heuristic for %s.%s(%s)", ...)
```

The benchmark machinery exists (`tokenspeed_kernel/benchmark/`, 1,219 LOC with a
`BenchmarkRunner`, `ThroughputCalculator`, verification) but is **not wired to
selection**. So today our advantage over vLLM's `IrOp` is the richer *declarative*
gate (capability, traits, objective, origin bands), not autotuning. Closing that wire
is arguably the highest-leverage kernel work available, and it is base-independent.

---

## 4. Radix vs block-hash APC — the agent-critical comparison

This section answers the coordinator's questions (a)–(d) directly. Two of the
premises did not hold.

### 4.1 Is it a real radix tree? Yes — but page-granular

`TreeNode` (`tokenspeed-scheduler/csrc/resource/radix_tree/tree_node.h:53-162`) holds
a whole token span (path compression), and the child-map key is the child's **first
page** of tokens (`radix_tree.cpp:34-36`). The walk is `WalkDownUtilMismatch`
(`radix_tree.cpp:134-195`), and partial matches inside a node do cause a genuine
**split** (`splitChild`, `radix_tree.cpp:64-76`; physical work in
`tree_node.cpp:58-105`, which splits tokens, page hashes, block hashes and the page
arrays of *both* tiers while copying the ref-count).

But match granularity is floored to whole pages: [code]

```cpp
// radix_tree.cpp:48-60
const std::size_t matched_tokens = std::distance(node_tokens.begin(), node_it);
return matched_tokens / page_size;      // integer division: partial page discarded
```

and callers must pass page-aligned input or it throws
(`kv_prefix_cache.cpp:174-177`, `:208-211`).

### 4.2 (a) Is radix materially better than block-hash APC for agent workloads?

**On match granularity: no — vLLM is currently finer.** Our default page size is
**64** (`python/tokenspeed/runtime/utils/server_args.py:117`); vLLM's default block
size is **16** (`vllm/config/cache.py:47`, `:255`). A 100-token shared prefix
recovers 64 tokens with us and 96 with vLLM. This directly refutes the assumed
advantage. (Both are configurable, so this is a *default* difference, not an
architectural one — but defaults are what ships.) [code]

vLLM's structure is a flat chained-hash dict — `BlockPool.cached_block_hash_to_block`
(`vllm/v1/core/block_pool.py:56-59`, `:184`), keys from `hash_block_tokens`
(`vllm/v1/core/kv_cache_utils.py:596-623`), which folds the *parent* block hash into
each child, giving prefix semantics without a tree. Lookup is a linear scan with an
early break, since a miss implies all later blocks miss
(`vllm/v1/core/single_type_kv_cache_manager.py:706-714`) — O(num_blocks) dict gets on
the scheduler hot path, not O(depth) pointer chasing.

Two vLLM warts worth recording, both in our favour: partial in-block matches are
wasted for any single-KV-group (i.e. plain dense) model, and on a **full** prompt hit
vLLM sets `max_cache_hit_length = num_tokens - 1` to keep a token for logits, which
its own comment admits "can trigger recomputation of an entire block, rather than
just the single last token" (`vllm/v1/core/kv_cache_manager.py:249-254`). A
sub-block matching path with copy-on-write **does** exist
(`cache_config.prefix_match_unit`, `vllm/config/cache.py:56-67`;
`single_type_kv_cache_manager.py:716-736`, `:373-401`) but
`resolve_kv_cache_block_sizes` returns early for `len(groups) <= 1`
(`kv_cache_utils.py:649-651`) **before reading it**, so it is unreachable for a dense
transformer today. [code]

**On eviction structure: yes, and this is the real difference.** Our eviction is
hierarchy-aware:

- Candidates are **leaves only**, where "leaf" means "has no child holding pages of
  this tier" (`eviction.h:45-72`). An interior node — precisely what a shared
  system-prompt/tool-definition trunk *is* — is not an eviction candidate at all
  until every descendant is gone, at which point `updateLeaf(parent)` promotes it
  (`eviction.h:159-160`).
- Ordering is LRU over `std::set<tuple<timestamp, seq_id, TreeNode*>>`
  (`tree_resource.h:137-146`), with `seq_id` rather than pointer as tiebreaker
  *specifically* for TP determinism (comment at `:140-144`: pointer ASLR would make
  ranks evict different leaves "and eventually wedging the next NCCL collective").
- **Every active request pins its entire root path**: `NodeRef::Lock()` walks
  `node → parent → ... → root` incrementing ref-counts (`tree_node.h:186-190`). With
  N sessions on one system prompt, the trunk carries ref-count N.
- Every `Match` re-`Touch()`es every node on the matched path (`radix_tree.cpp:184`),
  so a hot shared prefix continuously refreshes its LRU key across all sessions.

vLLM's eviction is a **single flat free-block LRU list with no sharing awareness at
all**. `FreeKVCacheBlockQueue` (`vllm/v1/core/kv_cache_utils.py:184-204`) is a
hand-rolled doubly-linked list ordered by free time; the only secondary signals are a
binary "blocks without a hash go to the front" split
(`vllm/v1/core/block_pool.py:727-740`) and freeing a request's blocks in reverse so
tail blocks die first (`single_type_kv_cache_manager.py:502-503`). In-use blocks are
protected because `ref_cnt > 0` removes them from the free list entirely
(`block_pool.py:702-717`, `:730-736`).

**The consequence is the sharpest structural difference in this document.** In vLLM,
a widely-shared system prompt is protected *only while at least one live request
references it*. The moment the last sharer finishes, its blocks go to the tail of one
global LRU and are then evicted in pure recency order — indistinguishable from a
one-off block belonging to a single junk request. There is no pinning, no
share-count boost, no frequency term. A burst of long unique prompts will walk the
head of the free list and evict a hot 2,000-token system prefix that 500 idle
sessions would have hit. vLLM does compute an all-requests-share-it common prefix
(`get_num_common_prefix_blocks`, `single_type_kv_cache_manager.py:795-803`), but
consumes it for **cascade attention only** (`sched/scheduler.py:1069-1074`) — the
one place `ref_cnt` is used as a sharing signal, and it is explicitly *not* used for
eviction. Our tree, by contrast, makes the shared trunk structurally the last thing
evicted. That is a real and durable advantage for exactly the agent shape:
thousands of intermittently-active sessions behind one system prompt.
[code for both structures; the *magnitude* is [inferred] — neither has been measured]

It also compounds with preemption: vLLM's preemption is recompute-only with a full
reset (`_preempt_request`, `sched/scheduler.py:1203-1225`, sets
`num_computed_tokens = 0` and frees all blocks), so a preempted request recovers work
**solely via the prefix cache** — precisely where the eviction blindness bites
hardest. Our retraction writes KV back to host and restores it rather than
recomputing (`csrc/scheduler/operations/forward.cpp:499-621`, `:1046-1057`).

**On branching: roughly a tie, and neither has a first-class primitive.** Our
`children_map_t` is natively many-child, so two divergent tool-call continuations of
one prompt become two children of one prefix node sharing one physical page set, and
eviction respects that (`eviction.h:45-72`). But `grep -rn "fork\|Fork"` over `csrc/`
returns only `mamba_branching_seqlen` — **there is no `ForkRequest` / `CloneSession`
API**. Branch sharing is emergent, content-addressed reuse. vLLM gets the same
emergent sharing from block hashing. Our genuine extra is **Mamba/SSM branching with
copy-on-write** (`MatchResult::mamba_branching_seqlen` / `mamba_cow_src_index`,
`csrc/resource/types.h:70-73`, consumed at `forward.cpp:295-300`) — which matters
only for hybrid-attention models.

**On the policy that exploits reuse: we are behind.** Our scheduler has **no
prefix-aware admission ordering at all**. `Scheduler::newForwardOperation`
(`forward.cpp:947-962`) sorts by FSM-state priority then lexicographic request id,
with the tie-break chosen for TP determinism. There is no LPM (longest-prefix-match)
policy, no priority, no arrival timestamp on `RequestSpec`
(`csrc/scheduler/request_spec.h:30-35`). SGLang has LPM; we have the tree without the
policy. vLLM has `policy: fcfs | priority` with priority-based preemption
(`vllm/config/scheduler.py:99`, `vllm/v1/core/sched/scheduler.py:578-581`) — also not
prefix-aware, but it at least has a priority axis we lack. **This is a
differentiator we own but have not cashed in.**

### 4.3 (b) What is `hybrid_prefix_cache`? — the premise is wrong

**It is hybrid *attention models*, not GPU+CPU tiering.** `HybridPrefixCache` wraps a
`KVPrefixCache&` and adds two adjuncts — a Mamba/SSM slot adjunct and a
paged-cache-group adjunct (`hybrid_prefix_cache.h:49-102`):

```cpp
// hybrid_prefix_cache.cpp:117-122
MatchResult HybridPrefixCache::Match(const token_vec_t& token_ids, MatchIntent intent) {
    auto match = kv_prefix_cache_.Match(token_ids, intent);
    augmentMatch(match);            // Mamba/SSM state
    augmentMatchPagedCache(match);  // per-attention-family paged groups
    return match;
}
```

The "families" are attention families: `PagedCacheGroupFamily { History, State }`
with `Retention { FullHistory, SlidingWindow }`
(`csrc/resource/allocator/paged_cache_group.h:43-67`), mapped from HF `layer_type`
labels in `python/tokenspeed/runtime/configs/paged_cache_spec.py:58-77`. `augmentMatch`
*truncates* the KV match back to the deepest node that also has a Mamba checkpoint,
because a recurrent layer cannot resume mid-stream (`hybrid_prefix_cache.cpp:187-188`).

**GPU+CPU tiering does exist, one level below**, in `KVPrefixCache`: every `TreeNode`
carries both a `DeviceResource` and a `HostResource` (`tree_node.h:155-156`), and the
single walk tracks the deepest device-resident and deepest host-resident node in one
pass (`radix_tree.cpp:154-163`, `:186-188`) — that *is* a genuine architectural nicety
for session persistence. Demotion is `ReleaseDeviceResourcesPresentOnHost`
(`kv_prefix_cache.cpp:356-386`). L3 is a Python `ThreadPoolExecutor` over a
`StorageBackendFactory` (`runtime/cache/executor/storage_executor.py:186-200`); the
C++ side has only a flag and a queue (`cache_coordinator.h:42-50`).

**vLLM's counterpart is broader, not absent — this premise also does not hold.**
In-tree: `OffloadingConnector`
(`vllm/distributed/kv_transfer/kv_connector/factory.py:206-210`; scheduler side 1,404
LOC) over `vllm/v1/kv_offload/` with a pinned-host `cpu/` tier and a
`tiering/{fs,obj,p2p}` secondary tier, DMA-async via `cudaMemcpyAsync`, metricated,
event-emitting, and unit-tested (`tests/v1/kv_offload/`). Notably its CPU tier
supports **both LRU and ARC** (`vllm/v1/kv_offload/cpu/policies/{lru,arc}.py`) — a
more sophisticated replacement policy than anything on its GPU tier *or* ours. Plus
the connector ecosystem: LMCache, FlexKV, Mooncake, NIXL, hf3fs, MoRIIO,
`MultiConnector` stacking (`factory.py:152-243`). So "hierarchical prefix cache" is
**not** a tokenspeed exclusive, and calling it one would be wrong.

What *is* exclusive is that ours is **fused into the admission decision**:
`AdmissionMatch{device, host}` (`csrc/cache/kv_cache_coordinator.h:60-68`) means the
scheduler knows at admission time how much of the prefix sits on host. vLLM consults
the connector tier *after* the GPU dict lookup, from the scheduler
(`vllm/v1/core/sched/scheduler.py:742-757`), and counts those hits separately
(`:769-772`). Whether one-pass admission-time knowledge is worth anything measurable
is unknown. [inferred]

### 4.4 (c) Cost of bringing radix to a vLLM base

Three options, in rising order of cost:

1. **Live with block-hash APC and reduce `--block-size` to 16.** Cost: ~0. Given
   §4.2, this is *not obviously worse* on match granularity and may be better. Loses
   the hierarchy-aware eviction and the fused device/host admission match.
2. **Custom `scheduler_cls` with a Python radix tree.** vLLM's `scheduler_cls`
   (`vllm/config/scheduler.py:117`) is open, the `KVCacheManager` is a scheduler-owned
   implementation detail (`vllm/v1/core/sched/scheduler.py:271`), and `SchedulerOutput`
   is block-id based (`vllm/v1/core/sched/output.py:193-223`) — a page-granular radix
   tree emits exactly that. **This is the realistic path**, and it is a *plugin*, not a
   fork. Estimated effort: a Python radix tree with leaf-first LRU + root-path pinning
   is on the order of 1–2 kLOC, plus a `SchedulerInterface` implementation that
   reuses vLLM's for everything else. It would also be the natural home for the LPM
   admission policy we do not currently have. [inferred]
3. **Port the C++ tree.** The tree *core* (`radix_tree.{h,cpp}` + `tree_node.{h,cpp}`
   + `node_range.h`, ~600 LOC) does not reference `Request`, `RequestSpec`, or the FSM.
   But it is hard-wired to this repo's `OwnedPages`/`PageAllocator` RAII allocator, to
   the two-tier `ResourceType{Device,Host}` split baked into `TreeNode`'s members and
   into `NodeRef`, to `MambaSlot`, and to `PagedCacheSnapshot`; and `tree_node.h:140`
   declares `friend class HybridPrefixCache`. The surrounding machinery it is fused
   with — `kv_prefix_cache/` (1,055 LOC) and `hybrid_prefix_cache/` (2,131 LOC) — is
   where the FSM entanglement lives: `Prefetching` holds a `HostNodeRef`
   (`csrc/fsm/cache_states.h:61`), `BaseState` a `DeviceNodeRef`
   (`csrc/fsm/forward_states.h:157`), and events construct/re-seat those locks inline
   (`csrc/fsm/cache_events.cpp:35-38`, `forward_events.cpp:158-205`). **Porting the C++
   means porting the FSM, i.e. it is not a port — it is keeping the base.**

**So (c)'s answer is: bringing radix to vLLM is a Python rewrite behind
`scheduler_cls`, not a C++ port.** That is real work but bounded, and it is the same
work whether or not we re-base — because a Python radix scheduler would also be how
we'd get LPM policy on our own base.

### 4.5 (d) What vLLM has for agents that we lack — be fair

| Capability | vLLM | UnieInfra | Verdict |
|---|---|---|---|
| Tool-call parsers | **44 registered / 46 files** (`vllm/tool_parsers/__init__.py:24-210`) | **0** in-repo (lives in the external `smg` gateway, `cli/serve_smg.py`) | **vLLM, decisively.** This is agent surface area, and it is enormous |
| **Grammar-enforced tool choice** | `tools=[...]` compiles into an xgrammar **structural tag**, so a malformed tool call is *unrepresentable* (`vllm/tool_parsers/structural_tag_registry.py:12-26`) | none | **vLLM.** Arguably the single strongest agent-correctness feature in either tree |
| Reasoning parsers | **28** (`vllm/reasoning/__init__.py:22+`) | 1 (`runtime/grammar/reasoning_structural_tag.py`) | vLLM |
| Structured output | 4 backends — xgrammar, guidance, outlines, lm-format-enforcer (`vllm/v1/structured_output/`, 2,481 LOC), `backend: auto` (`config/structured_outputs.py:21`); **async compile** off the hot path (request parks in `skipped_waiting`, `sched/scheduler.py:1956-1966`, promoted at `:2596-2604`) and bitmask fill **overlapped with the forward pass** (`v1/engine/core.py:587-596`) | xgrammar only (`runtime/grammar/`, 1,894 LOC) | vLLM broader and better integrated; ours has `capturable_grammar.py` (CUDA-graph-capturable), a genuine nicety. vLLM's residual risk: serial Python bitmask loop below batch 128 (`structured_output/__init__.py:60-68`, `:272-351`) |
| MCP tool server / Anthropic Messages API | both in-tree (`vllm/entrypoints/mcp/`, `vllm/entrypoints/anthropic/`) | none | vLLM |
| OpenAI server surface | Full (`vllm/entrypoints/openai/`: chat, completions, responses, batch, DP supervisor) | `vllm_compat_http.py` is **251 lines of weight-transfer/pause control endpoints only** — no `/v1/chat/completions` | **vLLM, decisively.** Our own architecture doc calls the frontend "BUILD NEW" (`unieinfra-architecture.md:55`) |
| Request priority / preemption | `policy: fcfs \| priority` (`config/scheduler.py:99`), priority preemption (`sched/scheduler.py:576-609`), `priority` on the chat API (`entrypoints/openai/chat_completion/protocol.py:363-370`) | **none** — FSM-state + lexicographic id (`forward.cpp:947-962`) | vLLM. Caveat: no aging, so priority-0 floods can starve priority-1 |
| Prefix-cache hit metric | `vllm:prefix_cache_{queries,hits}` **plus separate external-tier counters** (`v1/metrics/loggers.py:584-630`) | `tokenspeed:prefix_cache_hits` (`runtime/metrics/collector.py:359-363`) | vLLM slightly ahead (GPU vs offload tier separable) |
| **Prefix-aware / session-affinity routing** | **absent.** DP LB is `waiting * 4 + running`, prefix-blind (`v1/engine/core_client.py:1413-1447`); vLLM ships only the *telemetry* — `BlockStored`/`BlockRemoved` KV events over ZMQ (`distributed/kv_events.py:51-116`, `block_pool.py:373-443`) — and leaves the router to llm-d / production-stack | absent | **Open territory for both — see below** |
| Agent-shaped bench datasets | `prefix_repetition`, `ConversationDataset`, `MTBenchDataset`, **`BFCLDataset`** (Berkeley Function Calling Leaderboard, `vllm/benchmarks/datasets/datasets.py:4536`) | random only | **vLLM — and we can use these today** |
| Models | 290 | 30 | vLLM |
| Cascade attention (shared-prefix decode kernel) | yes, FlashAttention backend, gated ≥256 common prefix ∧ ≥8 reqs (`v1/attention/backends/flash_attn.py:1520-1596`) | none found | vLLM |

This table is the strongest argument *for* vLLM in the whole document, and it has
nothing to do with kernels. **For an agent product, tool parsing + grammar-enforced
tool choice + structured output + a real OpenAI server is table stakes, and we have
essentially none of it in repo.** Building 44 tool parsers is not research; it is
months of unglamorous work that vLLM has already done, and grammar-enforced tool
choice is a correctness property we would have to design from scratch.

**One genuine opening.** Prefix-aware routing is absent from vLLM *by design* — it
publishes `BlockStored` / `BlockRemoved` events over ZMQ specifically so an external
gateway can learn which instance holds which prefix (`block_pool.py:373-443`, added
with that stated rationale), and leaves the router out of tree. For an agent product
where thousands of sessions map to a small set of shared prefixes, **session-affinity
routing is plausibly a larger end-to-end win than any single-node cache-structure
difference**, and nobody has claimed it inside either engine. That is worth flagging
as a candidate differentiator independent of the base decision. [inferred]

---

## 5. The honest loss ledger — what leaving tokenspeed costs

| Asset | Size | vLLM equivalent? | Agent relevance |
|---|---|---|---|
| **C++ FSM scheduler, RAII KV ownership** — 13-state `std::variant` (`csrc/fsm/states.h:32-33`), move-only states owning `DeviceNodeRef` + `LocalKVAllocator` (`forward_states.h:95-111`) so double-free is a compile error | 14.2 kLOC C++ + 15.0 kLOC tests | **Nothing structurally similar.** vLLM's scheduler is Python; `scheduler_cls` is open but the safety property is a C++ type-system property | Medium. The safety is real but invisible to agent users; the *policy* it enables (P/D, admission) is what matters, and our policy is thin (§4.2) |
| **Radix prefix tree** (`csrc/resource/radix_tree/`, 1,066 LOC + `kv_prefix_cache/` 1,055 + `hybrid_prefix_cache/` 2,131) | ~4.4 kLOC | Partial: block-hash APC + KV-offload tiering + connectors. **Missing: hierarchy-aware leaf-first eviction, root-path pinning, single-pass device+host match fused into admission** | **High** — but see §4.2: finer default granularity is vLLM's today, and we lack the LPM policy that would exploit the tree |
| **Placement→comm compiler** (`runtime/models/base/{placement,compiler,comm_ops}.py`) — derives all-reduce vs reduce-scatter+all-gather and fuses reduce+residual+norm from placement annotations | ~1.36 kLOC | Weaker: vLLM hand-writes comm per model | Low for single-GPU agent serving; high later for big MoE |
| **Per-layer-family `Mapping`** (`runtime/distributed/mapping.py`) — independent attn/dense/MoE/vision decompositions | 414 LOC | Weaker/absent | Low now |
| **P/D disaggregation** with heterogeneous-TP layout negotiation (`runtime/pd/transfer_plan.py`, 467 LOC), layer-wise streaming on by default | ~3 kLOC | vLLM has ~8 KV transports (NIXL, Mooncake, LMCache, …) — **broader, differently shaped** | Medium — P/D is a TTFT lever, and TTFT dominates agent workloads |
| **Speculative decoding** — Eagle/MTP/DFlash drafters, full spec step inside one CUDA graph (`execution/cuda_graph_wrapper.py:307`, `:410-413`) | 3.2 kLOC | vLLM has spec decode too, broader | Medium. Chain-only for us (`server_args.py:615-620` rejects `eagle_topk>1`) |
| **Big-model machinery** (MLA, DSA, MSA, MoE EP, DP attention, EPLB, hierarchical KV) — inventoried in `docs/design/big-model-serving.md` | — | Mixed; **`tokenspeed-mla` is already upstreamed into both sglang and vLLM** (`big-model-serving.md:958-965`) | Low for agents now |
| **`tokenspeed-kernel` op corpus** | 112 kLOC | vLLM's is comparable or larger | Neutral |

Counterweight, stated plainly: `docs/design/big-model-serving.md` — **written by us,
about our own tree** — reports that much of this is `[code]`-only. EPLB is "present,
complete, entirely unwired" (zero call sites). MLA is implemented but "never
exercised". nvfp4 is blocked by a one-word gate. `TOKENSPEED_FLAT_KVCACHE` is
compiled **off**, disabling ~40% of the admission logic. And the whole C++ scheduler
arrived in **one squashed fork commit** (`ff37a93`) and has **not been modified
since** — `git log --oneline -- tokenspeed-scheduler` returns exactly 1 commit. The
recent scheduler win (`d84e250`, adaptive batch as default) was **13 lines of
Python** in `server_args.py`, not C++.

So the ledger is: we would lose a large amount of *inherited, high-quality,
largely unexercised* machinery, plus a smaller amount of genuinely differentiating
architecture (radix eviction structure, RAII KV ownership, the comm compiler), and
we would gain an enormous amount of *agent-relevant, boring, already-built* surface
(§4.5).

### What we would keep either way

Measured head-to-head, our base is currently ahead of vLLM at mid concurrency
[validated] (`benchmark/results/TRAJECTORY.md`, `output_throughput`, Qwen3-8B bf16,
sm_120):

| C | unieinfra | vllm | sglang | winner |
|---|---|---|---|---|
| 1 | 77.4 | 82.6 | 82.7 | sglang |
| 16 | 973.0 | 978.6 | 954.9 | vllm |
| 32 | 1551.9 | 1491.6 | 1480.9 | **unieinfra** |
| 64 | 2263.1 | 2090.7 | 2067.0 | **unieinfra** |
| 128 | 3026.9 | 2577.3 | 2553.9 | **unieinfra** |
| 256 | 2737.2 | 2687.3 | 2849.5 | sglang |

Re-basing forfeits the C=32–128 advantage as a starting point and re-earns it, or
not. **But see §7: this table measures a workload that is not our product.**

---

## 6. What transfers, concretely

| Ours | LOC | Coupling to tokenspeed runtime | Transfer verdict |
|---|---|---|---|
| `registry.py` (`KernelSpec`, `KernelRegistry`, `Priority`, `register_kernel`) | 518 | **None.** Imports `logging`, `collections`, `dataclasses`, `enum`, `typing`, plus `tokenspeed_kernel.{platform,signature}`; `torch` only under `TYPE_CHECKING` | **Lift-and-rename.** Zero runtime dependency |
| `selection.py` (`select_kernel`, scoring, overrides, `kernel_override`, `explain_selection`) | 960 | **None.** Stdlib + `platform`/`registry`/`signature`; `yaml` lazily | Lift-and-rename |
| `platform.py` (`PlatformInfo`, `CapabilityRequirement`, `ArchVersion`, detection) | 802 | **None.** `torch`, `torch.distributed`, `pynvml`, stdlib | Lift-and-rename. Overlaps vLLM's `Platform` — would wrap it rather than replace it |
| `signature.py` (`FormatSignature`) | 293 | **None.** stdlib; `torch` under `TYPE_CHECKING` | Lift-and-rename |
| `plugins/` (entry-point group `tokenspeed_kernel.plugins`, `:42`) | 318 | None | Lift-and-rename; maps onto `vllm.general_plugins` |
| **Selection core total** | **2,891** | **zero tokenspeed/sglang imports** | **This is the moat, and it is free-standing** |
| `unieinfra-kernel/` (origin axis, discovery, contracts, KDA tree, decode GEMV) | 1,734 | Only via `tokenspeed_kernel.{registry,platform,signature}` | Transfers with the core |
| `KernelContract` (`contracts/__init__.py:39-73`) | 73 | Only `platform` + `signature` | Transfers. Note vLLM is building the same idea (`vllm/ir/op.py:452-469`) |
| E33 decode GEMV (`ops/gemm/decode_gemv_triton.py`) | 594 | `register_kernel` + contract | **Literal drop-in.** Its signature is `A[M,K]`, `B[N,K]`, `bias[N]` — exactly vLLM's `apply(layer, x, layer.weight, bias)` |
| `tokenspeed_kernel/ops/` (33 `select_kernel` call sites across 8 families) | 112 kLOC | This is the *façade* — `ops.gemm.mm()` etc. wrap `select_kernel` | Kernels transfer individually; the façade would be re-pointed at vLLM's layer seams |
| `benchmark/` harness | — | **Already engine-agnostic**: `results/<engine>/<label>/`, with `bench_vllm.py`, `launch_sglang_host.sh`, and a vLLM/SGLang comparison built into `trajectory.py` | **Transfers unchanged** |

**Crux answered: the registry is not entangled with tokenspeed at all.** `grep` for
`from sglang`, `import sglang`, `from tokenspeed.`, `from tokenspeed import` across
`tokenspeed-kernel/python/` returns **nothing**. The only reason it lives in
`tokenspeed-kernel` is packaging history. Extracting it into a standalone
`unieinfra-kernel-registry` distribution is a rename plus a `pyproject.toml`, and it
would be worth doing **regardless of the base decision** — it makes the moat legible
as its own artifact.

---

## 7. The benchmark problem — we are measuring the wrong thing

`benchmark/run_sweep.sh:22-23` sets `ILEN=1024`, `OLEN=256` and drives
`--dataset-name random` (`:60`). Worse, `:53-54` **explicitly flushes the prefix
cache before every concurrency level**:

```bash
# --- hygiene: flush server cache so this level doesn't ride on a prior level's cached prefixes ---
if [ "$FLUSH" = "1" ]; then curl ... /flush_cache ... || curl ... /reset_prefix_cache ...; fi
```

That was the right call for isolating kernel and scheduler changes. For an
agent-oriented product it is exactly backwards: it nulls out the single biggest agent
lever *and* the exact capability we are deciding whether to keep. **Every number in
§5's head-to-head table is from the zero-reuse regime.** [validated]

### Recommended agent-shaped benchmark

vLLM's own harness already ships most of it, and our harness already drives
`vllm bench serve`, so this is cheap:

1. **Shared-prefix sweep** — `--dataset-name prefix_repetition` with
   `--prefix-repetition-prefix-len {512, 2048, 8192}` (system prompt + tool
   definitions), `--prefix-repetition-num-prefixes {1, 10, 100}` (how many distinct
   agents), suffix 128–512 (`vllm/benchmarks/datasets/datasets.py:1854-1884`).
   **Do not flush between levels** — reuse is the measurement.
2. **Multi-turn sessions** — `ConversationDataset` / `MTBenchDataset`
   (`datasets.py:2240`, `:2216`), replaying full history each turn.
3. **Tool calling** — `BFCLDataset` (`datasets.py:4536`), which is a real
   function-calling corpus with tool schemas in the prompt.
4. **Structured output** — JSON-schema-constrained generation, measuring both
   steady-state throughput and grammar-compile stall on first use.
5. **Bursty arrival** — Poisson/`TimedTrace` (`datasets.py:1437`) rather than fixed
   concurrency, since agents arrive in bursts.

**New primary metrics** (none of which `benchmark/summarize.py:22-23` currently
records): **prefix-cache hit rate** (we expose `tokenspeed:prefix_cache_hits`,
`runtime/metrics/collector.py:359-363`; vLLM exposes the same), **TTFT p99 under
reuse**, **cache hit rate under memory pressure with N sessions** (the eviction-policy
discriminator), and **tokens recomputed per turn**.

**This benchmark is the decision procedure.** If radix's eviction structure does not
show a measurable hit-rate advantage over vLLM APC at `--block-size 16` under many
concurrent sessions, then the single strongest argument for keeping the tokenspeed
base evaporates, and §4.5's table decides the question. If it does show one, we have
quantified our moat for the first time.

---

## 8. The three options, costed

### (a) Pure plugin on stock vLLM — **viable for kernels; viable for scheduling too**

*What it is:* `unieinfra-kernel` + the extracted registry as a pip package advertising
`vllm.general_plugins`, registering (i) an out-of-tree `QuantizationConfig` whose
`get_quant_method` routes dense GEMMs through `select_kernel`, (ii) `register_oot`
subclasses for activations/norms/RoPE/MoE method, (iii) `register_backend` for
attention, (iv) `IrOp.register_impl` for norms, (v) optionally a `scheduler_cls` with
a Python radix tree.

*What breaks:*
- Must handle the `custom_ops=["none"]` Inductor default (§2.3 trap 2) or every
  `forward_cuda` override is silently ignored.
- `register_oot` renames `cls.name`, breaking `+op` enable keys.
- Out-of-tree attention backends are never *auto*-selected (`platforms/cuda.py:83-163`).
- We cannot reach the 7 patch-only decision points (§3) — but 6 of those are ROCm/CPU
  or DeepSeek-specific, so the practical loss is small.
- No control over vLLM's compile-pass ordering.

*Maintenance:* lowest by far, because a plugin depends on **APIs**, not on file
contents — churn only hurts us if a signature changes. The APIs we would consume are
vLLM's most stable: `custom_op.py` saw **3 commits in 180 days**,
`plugins/__init__.py` **2**, `attention/selector.py` **0**.
`quantization/__init__.py` churned **28** times, but that is the in-tree method table
(`get_quantization_config`'s lazy dict at `:115-171`), not the two-line
`register_quantization_config` contract, which is public, documented
(`docs/features/quantization/README.md:76-231`), and CI-tested. Contrast what a fork
would sit on: `fused_moe/layer.py` **65**, `config/compilation.py` **56**,
`platforms/interface.py` **40**, `linear.py` **34**, `layernorm.py` **24**. [code]

*Effort estimate:* extracting the registry ≈ 1–2 days. A working GEMM plugin with
E33's kernel on Qwen3-8B inside stock vLLM ≈ 1 week. [inferred]

### (b) Thin fork of vLLM

*What it is:* (a) plus a small patch set routing selected ops through our registry
directly and adding the missing seams: a `gemm` op in `vllm/ir/ops/`, our capability
scoring inside `IrOp.dispatch`, an OOT slot in the attention priority list, and
whatever the agent scheduler needs beyond `SchedulerInterface`.

*How small:* the minimum useful patch is ~10 lines in
`vllm/model_executor/layers/utils.py:342-348` (make `dispatch_unquantized_gemm`
consult a registry) plus an `IrOp` GEMM definition. That is genuinely small.

*Rebase cost, measured:* vLLM landed **1,016 commits in the last 30 days** (~34/day).
Over 180 days, the files a thin fork would touch churned: `linear.py` **34**,
`utils.py` **20**, `layernorm.py` **24**, `activation.py` **13**,
`fused_moe/layer.py` **65**, `config/compilation.py` **56**,
`platforms/interface.py` **40**, `quantization/__init__.py` **28**,
`custom_op.py` **3**, `attention/selector.py` **0**. A patch confined to
`utils.py` + `ir/` is a low-conflict bet; one that touches `fused_moe/` or
`compilation/` is a monthly tax. [code]

*Note in our favour, and it cuts both ways:* we are **already** a hard fork with **no
upstream sync** (`CHANGELOG.md`: *"There is no upstream sync — the fork is the base we
build on"*), carrying 245 kLOC. A vLLM fork we actually track would be *more*
disciplined than what we do today, not less — but it would also be the first time we
pay a rebase tax at all.

### (c) Keep the tokenspeed base (status quo)

*What we keep:* everything in §5, plus a measured C=32–128 throughput lead, plus the
freedom to change any layer without a rebase.

*What we keep paying:* 245 kLOC of inherited code with no upstream, of which a large
fraction is `[code]`-only and unexercised; a 30-model zoo against vLLM's 290; **no
OpenAI chat server, no tool parsers, no reasoning parsers** — i.e. the entire agent
frontend is still "BUILD NEW" (`unieinfra-architecture.md:55`); one structured-output
backend; no request priority. And every bug in 14 kLOC of C++ scheduler is ours alone.

**Ranking, for an agent-oriented product: (a) → (c) → (b).** Option (a) first because
it is cheap, reversible, and produces the information the real decision needs.
Option (c) as the default until that information exists. Option (b) last — a deep
fork of a project moving at 34 commits/day is the highest-variance choice, and
nothing we have learned yet requires it.

---

## 9. Recommended decision procedure

1. **Extract the registry into a standalone distribution.** Base-independent, 1–2
   days, and it makes the moat a legible artifact. Do this regardless.
2. **Wire `_autotune_select` to `BenchmarkRunner`.** Our claimed differentiator over
   `vllm/ir` is autotuning, and it does not exist yet (`selection.py:776-800`).
   Base-independent.
3. **Build the agent-shaped benchmark (§7) and run it on all three engines.** This is
   the gate. Until it exists, every base argument is speculation.
4. **Spike the plugin (option a): E33's decode GEMV inside stock vLLM on Qwen3-8B.**
   One week. It proves or refutes plugin viability with a number, and if it works we
   get a second engine to sell the kernel work into.
5. **Then decide**, on this test: *does the radix eviction structure produce a
   measurably higher cache-hit rate than vLLM APC at `--block-size 16`, under many
   concurrent sessions sharing a system prompt, at memory pressure?* Design the run
   so the discriminator is exercised: N idle-but-returning sessions on one long
   shared prefix, interleaved with a burst of long unique prompts that forces
   eviction. That is precisely the case where vLLM's flat LRU should discard the
   shared trunk and our leaf-first LRU should not. If we win, keep the base and cash
   it in by adding an LPM admission policy (§4.2 — we do not have one). If we do not,
   the §4.5 agent-surface table decides, and it points at vLLM.

### What else we would need to know

- **Does vLLM's APC hit-rate actually lag ours under session pressure?** Unmeasured on
  both sides. This is the crux and it is one experiment away.
- **What does a Python radix `scheduler_cls` cost in per-step overhead?** vLLM's
  scheduler is Python already, so the comparison is like-for-like, but our tree is C++.
- **How much of the 44-tool-parser gap can be closed by vendoring vLLM's parsers?**
  They are Apache-2.0 and mostly self-contained — but grammar-enforced tool choice
  (`vllm/tool_parsers/structural_tag_registry.py:12-26`) is coupled to their
  structured-output stack and would not vendor as cleanly. [inferred]
- **Is prefix-aware session routing worth more than the cache-structure difference?**
  vLLM deliberately leaves the router out of tree and publishes KV events for it. If
  multi-instance session affinity dominates, the base question matters less than we
  think, and the differentiator is a gateway neither engine has.
- **Is cross-AMD still a hard requirement?** It is an open question in project memory
  and it materially changes the base calculus — vLLM's ROCm support is far ahead of
  ours, but our `tokenspeed-kernel-amd` (45 kLOC pure Triton/Gluon) is real.
- **Does `--block-size 16` cost vLLM anything measurable** in attention-kernel
  efficiency that offsets the finer match granularity?

---

## Appendix — what would change these conclusions

- **If vLLM lands a `gemm` op in `vllm/ir/ops/`** with capability-aware dispatch, the
  kernel-selection moat narrows sharply and option (a) becomes strictly better than
  (c) for the kernel layer.
- **If the agent benchmark shows radix eviction is worth >10% hit rate** at scale, §5
  flips and (c) becomes the clear answer.
- **If `--block-size 16` on our own engine closes the granularity gap for free**, one
  of vLLM's advantages disappears without any base change — worth testing first, it is
  a one-line experiment.
- **If someone ports the tool parsers**, the largest single argument for vLLM in §4.5
  shrinks to the model zoo.
- **If we adopt `PlatformEnum.OOT`** (we should not), most of §2.3's CUDA analysis is
  void — `forward_oot` becomes live and `forward_cuda` becomes dead.
- **If E33's `unieinfra-kernel` install problem is not fixed** (the package is not
  installed in the dev container, so the scaffold silently never loads), none of §6's
  transfer inventory is being exercised on either base today.
