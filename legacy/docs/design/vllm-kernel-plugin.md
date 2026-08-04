<!-- English canonical. docs/design/* are English-only in this repo (no Chinese mirror). -->
# UnieInfra's Kernel Layer Inside Stock vLLM — Step 1 of the Re-base

Status: **Implemented and measured** · Date: 2026-07-25 · Owner: royshih
Evidence: `docs/experiments-log.md` E34 (dense) and E38 (fp8 / NVFP4) ·
Background: `docs/design/vllm-base-assessment.md`

## What this is

The first step of re-basing UnieInfra onto vLLM's architecture: UnieInfra's
capability-based kernel **selection layer** and its first-party kernels now run
inside an unmodified vLLM, and the E33 decode GEMV is genuinely selected and
genuinely faster end to end.

Headline, Qwen3-8B bf16, batch-1 greedy decode, sm_120 (RTX PRO 6000), vLLM
0.25.1, one process per arm, two independent pairs:

| Arm | median tok/s | vs stock |
|---|---|---|
| stock vLLM | 84.84 / 85.19 | — |
| + UnieInfra kernel layer, **all dense Linears** (default) | 88.52 / 88.68 / 88.43 | **+4.34% / +4.10%** |

A third configuration isolates where the win comes from (single pair, same
conditions): restricting the plugin to a hand-picked `qkv_proj,o_proj` list
gives 86.58 (+1.6%) against a 85.18 baseline — i.e. **more than half the win is
`down_proj`, which the hand-picked list omitted and the trait gate found.**

Nothing under `third_party/vllm/` was modified. Nothing about stock vLLM's
behaviour changes when the flag is absent.

## Where the code lives, and why

**Decision: a standalone `unieinfra-kernel` distribution that stock vLLM loads
as a plugin. No edits inside `third_party/vllm`.**

The reasons, in order of weight:

1. **A plugin depends on APIs, not on file contents.** vLLM landed ~1,016
   commits in 30 days. The three APIs we consume — `vllm.general_plugins`,
   `register_quantization_config`, `LinearMethodBase` — are among its most
   stable and are CI-tested in tree
   (`tests/quantization/test_register_quantization_config.py`). A fork of
   `linear.py` (34 commits/180d) or `fused_moe/layer.py` (65) is a monthly tax
   we would start paying today for no benefit.
2. **It works either way.** If the re-base completes, this package *is* the
   kernel layer. If it does not, the package still works on the tokenspeed base
   unchanged (see "One package, two bases"). Nothing is forked, so nothing has
   to be un-forked.
3. **It makes the moat a legible artifact.** The selection core is the thing
   the base assessment identified as portable and differentiating; extracting it
   into its own importable distribution is worth doing regardless of the base
   (assessment §6, §9.1).
4. **It is honest about the seam.** If the plugin route had failed to reach the
   GEMM, that would have been a finding. It did not fail.

## The seam

```
pip install unieinfra-kernel
  -> entry point group `vllm.general_plugins`
     -> unieinfra_vllm_plugin.register()          # top-level module, see below
        -> register_quantization_config("unieinfra")(UnieInfraConfig)

vllm serve <model> --quantization unieinfra
  -> ModelConfig._verify_quantization accepts it (the registrar appended the
     name to current_platform.supported_quantization)
  -> LinearBase.__init__: quant_method = quant_config.get_quant_method(self, prefix)
  -> UnieInfraLinearMethod.apply(layer, x, bias)
     -> torch.ops.unieinfra.linear(x, layer.weight, bias)
        -> select_kernel("gemm", "mm", <dense bf16 signature>, traits={...})
           -> unieinfra_decode_gemv_triton   (M<=8, N<=16384)
           -> torch_linear (F.linear)        (everything else)
```

`--quantization unieinfra` is a **dispatch selector, not a numeric format**.
Weights stay bf16; `create_weights` / `process_weights_after_loading` are
inherited verbatim from `UnquantizedLinearMethod`, so the checkpoint path is
byte-identical to stock.

The quantization seam was chosen over `CustomOp.register_oot` /
`PluggableLayer` because it dispatches per layer **instance** and therefore
reaches `QKVParallelLinear` and `MergedColumnParallelLinear` — see "Traps"
below.

### Why the entry point lives outside the package

vLLM invokes `vllm.general_plugins` in **every** process it starts, including
runs that never ask for our kernels — and this is a shared benchmarking box. If
the entry point pointed at `unieinfra_kernel.integrations.vllm:register`, merely
loading it would execute the package `__init__` and therefore full kernel
discovery (which imports Triton) in every stock-vLLM run in the same
environment.

So the implementation lives in a top-level `unieinfra_vllm_plugin` module that
imports vLLM and nothing else; `unieinfra_kernel` is first imported inside
`UnieInfraConfig.from_config`, i.e. only when a model actually asks for
`--quantization unieinfra`. Verified:

```
load_general_plugins()          -> unieinfra_kernel imported: False
                                   'unieinfra' in QUANTIZATION_METHODS: True
get_quantization_config('unieinfra').from_config({})
                                -> unieinfra_kernel imported: True
```

`unieinfra_kernel.integrations.vllm` re-exports it so the code is still
discoverable at the obvious import path.

### Why a `torch.library` custom op

`apply()` is traced by Dynamo. Any Python-level selection inside it is either a
graph break or gets specialized on a symbolic batch size. Wrapping the façade in
`torch.library.custom_op("unieinfra::linear")` makes Inductor treat it as
opaque, which buys three things at once:

* no graph break, no dynamic-shape guard on `M`;
* the registry's Python work runs at **CUDA-graph capture time only** — vLLM
  captures decode graphs at engine construction, so a replayed step pays zero
  dispatch overhead;
* the fallback branch inside the op is free to be arbitrary Python.

Measured cost of the wrapper itself: with the kernel force-disabled
(`UNIEINFRA_DECODE_GEMV=0`, so the wrapper delegates to `F.linear`) the arm
measured 85.28 tok/s against a stock baseline of 84.88 — i.e. the scaffolding is
free within noise.

## One package, two bases

`unieinfra_kernel.boundary` is the single import site for registry / selection /
platform / signature symbols, and it picks a backend once per process:

| Situation | Backend | Why |
|---|---|---|
| `tokenspeed_kernel` importable (tokenspeed base) | `tokenspeed` | The runtime calls `tokenspeed_kernel.ops.gemm.mm`; registering elsewhere would make our kernels invisible. One boundary, per `AGENTS.md`. |
| Not importable (vLLM base) | `standalone` | `unieinfra_kernel.selection`, the vendored core. |
| `UNIEINFRA_KERNEL_REGISTRY=standalone\|tokenspeed` | forced | Tests, and the eventual default flip. |

The vendored core (`unieinfra_kernel/selection/{registry,selection,platform,
signature}.py`, 2,586 LOC) is the tokenspeed core with imports rewritten and two
substantive edits: `load_builtin_kernels()` now walks *our* `ops/kda/contrib`
trees, and the env-var prefix is `UNIEINFRA_KERNEL_` with the historical
`TOKENSPEED_KERNEL_` spelling still honoured. A test asserts the four modules
contain no `tokenspeed` or `sglang` reference at all.

Because the vendored core starts empty, the standalone build also registers
`torch_linear` — an `F.linear` at `PORTABLE + 3`, the same band tokenspeed's
`torch_mm` occupies. That keeps selection a real decision rather than
"our kernel or an exception", and makes the non-decode path bit-identical to
what stock vLLM's `dispatch_unquantized_gemm` does. It is **not** registered
when the boundary resolved to tokenspeed, because that registry already has
`torch_mm` at the same priority and a second entry could break an existing tie.

## Traps

Three, all verified in the installed vLLM 0.25.1. The first two are the ones the
base assessment predicted; the third we found by measuring.

1. **The Inductor default disables custom ops.** `vllm/config/vllm.py:1182-1189`
   appends `"none"` to `compilation_config.custom_ops` whenever the backend is
   Inductor and compilation is not `NONE`, so a third-party `forward_cuda`
   override is never dispatched. Confirmed live: our engine log shows
   `'custom_ops': ['none']`. **Did not bite** — a `LinearMethodBase` has no
   `enabled()` gate, so the quantization seam is immune. It *would* have bitten
   the `CustomOp.register_oot` route.
2. **`PluggableLayer` keys on `cls.__name__`.** `custom_op.py:49` looks up
   `op_registry_oot[cls.__name__]`, and `linear.py` registers only
   `ReplicatedLinear` (:291), `ColumnParallelLinear` (:396) and
   `RowParallelLinear` (:1536). `QKVParallelLinear` (:942) and
   `MergedColumnParallelLinear` (:580) are **unregistered subclasses**, so
   replacing `column_parallel_linear` misses `qkv_proj` — the exact E33 hot
   shape. **Did not bite** — `get_quant_method` is called per layer instance
   (`linear.py:274-279`), so subclasses are covered by construction.
3. **vLLM's torch.compile/AOT cache is keyed on `VllmConfig`, not on our
   environment.** Two runs that differ only in a `UNIEINFRA_*` env var land in
   the same `~/.cache/vllm/torch_compile_cache/<hash>/` and the second silently
   replays the first's compiled graph. This *did* bite: the first
   "route all dense Linears" arm reused a graph traced when only `qkv_proj` and
   `o_proj` were ours, and reported byte-identical call counters. Any A/B that
   changes the **graph structure** must set `VLLM_DISABLE_COMPILE_CACHE=1`. An
   A/B that only changes behaviour *inside* our opaque custom op (e.g.
   `UNIEINFRA_DECODE_GEMV=0`) is unaffected, because the graph is the same by
   construction.

To that list add the standing E33 lesson, which is why every number above comes
from a separate process: **CUDA graphs are captured at engine construction**, so
flipping a kernel after startup replays the same graph and shows a false ~0%.

## Why "all dense Linears" is the default

`UNIEINFRA_VLLM_LAYERS` defaults to `all`. Confinement is the registry's job,
not a hand-maintained layer list:

* `n_le_16384` excludes `gate_up_proj` (N = 24576) — measured as `torch_linear`.
* `m_le_8` excludes every prefill shape (M = 13 and M = 4096 in the logs).
* The wrapper's runtime profitability rule declines M > 2 and delegates to
  `F.linear`.

The measured difference between `all` and a hand-picked `qkv_proj,o_proj` list
is **+4.2% vs +1.6%**, and the difference is `down_proj` — a layer a human list
had omitted and the trait gate found. That is the self-configuring-engine rule
in miniature: the gate is declarative, so it generalizes to models nobody
enumerated.

## Quantized checkpoints: delegate the format, override only the GEMM

`--quantization unieinfra` also serves fp8 and NVFP4 checkpoints. The rule is
the same one that made the dense path safe — **inherit everything, override one
method** — applied to vLLM's own quantization configs instead of
`UnquantizedLinearMethod`:

```
UnieInfraConfig.from_config(hf_quantization_config)
  -> mode = nvfp4 | fp8 | dense               (checkpoint, or UNIEINFRA_VLLM_QUANT)
  -> delegate = ModelOptNvFp4Config | Fp8Config | None
  -> get_quant_method(layer, prefix):
       method = delegate.get_quant_method(layer, prefix)   # stock vLLM object
       method.__class__ = subclass_of(type(method))        # re-class in place
```

The subclass overrides `apply` (route through `select_kernel`) and
`process_weights_after_loading` (decide whether we claim the layer). Everything
else — `create_weights`, the weight loader, `uses_meta_device`, MoE, the
KV-cache method, ModelOpt's `exclude_modules` — is the delegate's, unmodified.
Two properties fall out:

* **The A/B is exact.** Both arms hold the same weight tensors produced by the
  same loader; only the GEMM differs.
* **A declined shape is stock vLLM.** When no kernel matches, dispatch calls the
  engine GEMM that the plugin injected via
  `dispatch.register_engine_fallback(...)` — the same
  `scaled_fp8_quant`+`cutlass_scaled_mm` / `scaled_fp4_quant`+`cutlass_scaled_fp4_mm`
  pair vLLM's own kernels call. `unieinfra_kernel` never imports vLLM; the
  engine hands its GEMMs in, which keeps the kernel package engine-neutral.

### The activation format is a dispatch decision

`gemm_format_signature` reproduces `tokenspeed_kernel.ops.gemm._gemm_format_signature`
verbatim (dense / `fp8` / `nvfp4` / `mxfp8` / `mxfp4`) and adds one case the
tokenspeed façade never needed: when the activation has *not* been quantized, the
signature keeps a **dense `a` role**. That is what lets a kernel say "give me the
bf16 activation and the quantized weight" — W8A16 / W4A16 — instead of being
forced into the checkpoint's symmetric W8A8 / W4A4 shape.

On a decode step that is the right call, and it is the whole reason the
quantized kernels win: the projection moves `N x K` weight bytes against `M x K`
activation bytes, so at `M = 1` the activation is ~0.03% of the traffic.
Quantizing it saves nothing and costs a launch, while the W4A4/W8A8 tensor-core
GEMM it enables wastes >98% of every MMA on padding. The symmetric signatures
are still built, so a future prefill kernel can claim them; no first-party
kernel fulfils them today, which is why prefill stays on CUTLASS.

### Three things this cost us

1. **`WEIGHT_LOADER_V2_SUPPORTED` is a class-*name* list.**
   `linear.py:473` and `:1618` choose the modern `BasevLLMParameter` loader with
   `self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED`. Naming
   the wrapper subclass anything other than the base class drops every quantized
   layer onto the legacy loader, which then copies a packed `[N, K//2]` uint8
   shard into an unpacked parameter and dies on
   `assert param_data.shape == loaded_weight.shape`. The subclass therefore
   keeps `base_cls.__name__`; our identity lives in a `unieinfra_mode` attribute.
2. **The declarative gate must match the profitability rule exactly.** The dense
   GEMV can declare `m_le_8` and delegate internally to `F.linear`, because that
   is both CUDA-graph safe and what the incumbent would have run. A quantized
   kernel's internal fallback is a dequantize-then-matmul the engine would never
   have run — and CUDA-graph capture walks `M = 1, 2, 4, ... 512`. So the
   quantized kernels declare `m_le_2` (a new, additive trait) and above the cut
   *nothing is selected*, handing the shape to the engine.
3. **The dispatch op is not free where it buys nothing.** It is opaque to
   Inductor, so wrapping a layer suppresses fusion around its GEMM: a control arm
   with the kernel unregistered but the wrapper live measured ~2% below stock at
   `C >= 32`. `dispatch.layer_is_claimable(kind, N, K, dtype)` therefore asks the
   registry, at load time, whether *any* batch size could select one of our
   kernels for that projection; `gate_up_proj` (N = 24576, fails `n_le_16384`)
   never can, and is left entirely unwrapped.

### sm_120 notes

* vLLM's `auto` NVFP4 backend picks FlashInfer, whose JIT rejects `major == 12`
  ("No supported CUDA architectures found for major versions [12]"), so stock
  `--quantization modelopt_fp4` does not start on this part. `--linear-backend
  cutlass` fixes it and both arms of a comparison must use it.
* A ModelOpt NVFP4 checkpoint declares `kv_cache_quant_algo: FP8`, which makes
  vLLM auto-select the FlashInfer attention backend — also un-JIT-able here.
  `--attention-backend TRITON_ATTN` is the sm_120 shim, again for both arms.

## What still needs to move across

| Item | Where it is now | Note |
|---|---|---|
| Arch gating audit (E30) | `tokenspeed-kernel/ops/**` capability declarations, `compute:tcgen05` | The `CapabilityRequirement` machinery came across with the core; the *audited gates* did not, because those kernels did not. |
| Shared-memory tile sizing (E28) | tokenspeed MoE / flashinfer-cutlass path | No MoE kernel is registered standalone yet; needs `FusedMoE` seam work (`fused_moe/routed_experts.py:199` is the analogous `get_quant_method` hook). |
| fp8 / nvfp4 dispatch (E30) | **done** — `unieinfra_kernel.dispatch.gemm_format_signature` | All four scaled branches are ported and fp8/NVFP4 are wired end-to-end (E38). `mxfp8`/`mxfp4` have signatures but no engine seam and no checkpoint to test against. Online activation quantization lives on the engine-fallback path; no first-party W8A8/W4A4 kernel is registered yet, so prefill stays on CUTLASS. |
| Autotuning | **done** — `selection/autotune.py`, `selection/inputs.py`, `selection/tuning.py` | Was ported verbatim, defect included (it never benchmarked, and discarded the caller's objective). Now measures: times each surviving candidate on real shapes, drops any that raises, caches per `(device_name, arch, family, mode, format_signature, traits, shape-bucket)`, and freezes at the end of startup. See `docs/design/kernel-autotuning.md` and E46. |
| The rest of the op corpus | `tokenspeed_kernel/ops/` (112 kLOC) | Attention, MoE, norms, sampling. Each needs its own vLLM seam; `register_backend`, `CustomOp.register_oot` and `IrOp.register_impl` are the candidates. |

## Files

```
unieinfra-kernel/
  pyproject.toml                       # advertises vllm.general_plugins
  unieinfra_kernel/
    boundary.py                        # the single kernel-boundary import site
    dispatch.py                        # dense linear façade + torch custom op
    selection/{registry,selection,platform,signature}.py   # ported core
    ops/gemm/decode_gemv_triton.py         # E33 dense kernel (+ call counters)
    ops/gemm/decode_gemv_fp8_triton.py     # W8A16 fp8 decode GEMV (E38)
    ops/gemm/decode_gemv_nvfp4_triton.py   # W4A16 NVFP4 decode GEMV (E38)
    ops/gemm/torch_linear.py           # standalone incumbent
    integrations/vllm/__init__.py      # docs + re-export of the entry point
  unieinfra_vllm_plugin.py             # the entry point + QuantizationConfig
  tests/test_boundary_and_dispatch.py
benchmark/results/vllm-plugin/         # raw per-arm JSON + engine logs
```
