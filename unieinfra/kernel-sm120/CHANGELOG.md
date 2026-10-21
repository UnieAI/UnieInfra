# Changelog — unieinfra-kernel-sm120

Out-of-tree `tokenspeed-kernel` plugins for NVIDIA sm_12x: sm_120 (RTX PRO 6000
Blackwell Server Edition) and, since 0.8.0, sm_121 (GB10 / DGX Spark). Design provenance for each kernel — what it was
derived from, what it reuses, what it changes — is in
[`docs/kernel-provenance.md`](docs/kernel-provenance.md).

## 0.8.0 — 2026-10-19

Ported from the `unieai` branch to `main` (commits 717e0b98, b7e86154,
07696f3d). The entries below this one describe the plugin as it was on
`unieai`; where they say a family is "not wired", this entry is current.

### Changed

- Registers on every sm_12x device (`arch_version.major == 12`), so GB10
  (12.1) no longer falls out of an exact `(12, 0)` comparison.
- `register()` now wires `flash_attn2`, `gdn` and `quant` after the
  FlashInfer paged MHA decode. **gemm and moe stay unregistered on main**:
  their ops select on traits (activation, EP, bias, routing, strides) the
  kernels never declared, and an undeclared trait matches every value.
- Adapted to main's kernel package: `register_kernel` lost `tags` (#1654);
  GDN helpers moved to `ops/attention/gdn/`; MHA ops pass `enable_pdl`;
  trait names are main's (`sinks`, `logit_cap`, `value_head_dim`,
  `dequantize=False`), with `head_dim` and `skip_softmax` declared.
- Dropped the removed `PlatformInfo.fp8e4m3fn`; the fused GDN prefill accepts
  `h_dtype`.

### Added

- `tests/test_registration.py` asserts every expected kernel registered (a
  failed family is otherwise silent); `tests/test_gdn_prefill.py` holds the
  fused GDN prefill to the Triton result.

### Known gaps on main

- Nothing in main calls the decode kernel's `prepare_paged_mha_decode` hook,
  so CUDA-graph capture with the plugin decode raises; `--attention-backend
  triton` (the GB10 CI and runbook flags) keeps it from being selected anyway.
- The fused GDN prefill is not in main's `_DEVICE_BOUNDS_CHUNK_PREFILL` list,
  so prefill loses graph capture when it is selected.
- FA2 needs an aarch64 `vllm_flash_attn` matching torch 2.14; none is known.

## 0.7.0 — 2026-08-02

FlashAttention-2 for MHA. TokenSpeed had no FlashAttention path on sm_120 at
all — FA4 is pinned to sm_100 and needs `tcgen05`, FA3 ships `sm_90` cubins with
no PTX, and FA2 was simply not installed — while vLLM served the same model on
the same card on FA2. It turns out FA2 runs here only because vLLM's fork builds
it as `8.0+PTX` and the driver JIT-compiles Ampere PTX for sm_120; there is no
native sm_120 FA2 anywhere.

### Added

- `unieinfra_fa2_mha_prefill_sm120` — `Priority.PLUGIN + 1` (17). **3.2-3.3x
  faster** than `triton_mha_prefill`, its only competitor, at b4/s1024 and
  b1/s4096.
- `unieinfra_fa2_mha_extend_sm120` — `Priority.PLUGIN + 1` (17). **3.0-13.7x
  faster than `unieinfra_flashinfer_paged_mha_extend_sm120` as that kernel is
  registered today**, and 2.1-2.3x faster than in-tree Triton. The margin over
  FlashInfer is *not* kernel speed: FlashInfer's extend path calls `plan()`
  (6-8 ms) and rebuilds the CSR page index (3-5 ms) on every call of every
  layer, and its kernel alone is a tie-to-1.6x-faster than FA2. It is also the
  only one of the three that accepts TokenSpeed's padded `q`; the FlashInfer
  extend kernel raises on it.
- `fa2_mha_decode_with_kvcache` — written, verified, and **not registered by
  default**. FA2's varlen paged path enables split-KV only when
  `max_seqlen_q == 1`, so it loses decode to FlashInfer at every shape swept
  and loses 3.6-5.9x at speculative `q_len 4`. Set
  `UNIEINFRA_FA2_REGISTER_DECODE=1` to register it anyway; it is the only decode
  path here that needs no plan and no CUDA-graph prepare hook.

### Requires

- The `vllm_flash_attn` extension, from a vLLM image whose torch build matches
  this one (torch 2.11.0+cu130 → `vllm/vllm-openai:cu130-nightly`). It is not
  installable from PyPI. `register_flash_attn2()` logs and returns if it is
  absent; nothing else is affected.

### Not yet wired

- `unieinfra_kernel_sm120/__init__.py::register()` does not call
  `register_flash_attn2()`, so these kernels are inert until that one line is
  added to the module loop at the end of it. That file has a separate owner.

## 0.6.0 — 2026-08-02

Quantization. On sm_120 this family was not merely slow, it was partly
unreachable: `quantization/mxfp4` had no admissible kernel, `quantization/nvfp4`
had nothing registered at all, and `fp8_with_scale` at `granularity="tensor"`
or `"token"` raised `NoKernelFoundError`. All four holes were gating, not
hardware.

### Added

- `unieinfra_flashinfer_quantize_mxfp4_sm120` — MXFP4 via FlashInfer's
  purpose-built `sm120a` FP4 module (`sf_vec_size=32, sf_use_ue8m0=True`).
  Serves all three layouts the tokenspeed API names — `linear`, `128x4`, `8x4`
  — including its own default of `128x4`, which the in-tree Triton kernel
  cannot produce. UE8M0 scale bytes and E2M1 codes are **100%** identical to an
  independent float32 model of the scheme; reconstruction error is exactly
  0.5x the format's quantization step.
- `unieinfra_flashinfer_quantize_nvfp4_sm120` — NVFP4 (per-16 E4M3 scale
  factors). Fills an operator that had zero registered kernels here, because
  `ops/quantization/flashinfer.py` guards on `platform.is_blackwell`, which is
  `arch_version.major == 10`. Scale-factor bytes 100% identical to a float32
  reference.
- `unieinfra_triton_quantize_fp8_with_scale_sm120` — dynamic FP8 at
  `granularity="tensor"` and `"token"`, neither of which had any admissible
  kernel. E4M3 codes 100% identical to a float32 reference; max absolute error
  0.99-1.00x of the format's own bound (half a ULP at 448), max relative error
  0.058 against the 2^-4 bound.
- `unieinfra_triton_quantize_fp8_sm120` — same numerics as the in-tree FP8
  cast, **bit-identical** at every shape tested, with flat 1-D launch geometry
  instead of `BLOCK_M x next_pow2(N)` tiles. 1.5-1.9x on ordinary shapes and
  **63x** at `[4096, 7168]`, where the in-tree tiling puts 131072 elements in
  one program and spills.

### Verified

Every number above is against an independent float32 model of the *format* —
the E4M3 and E2M1 grids, the UE8M0 shared exponent — built from torch
primitives, not against another kernel. Full tables and the stated precision
bounds are in [`docs/kernel-provenance.md`](docs/kernel-provenance.md).

**Stated precision difference.** The MXFP4 kernel is *not* bit-identical to the
Triton path and is not meant to be. FlashInfer uses `ceil(log2(amax/6))` for
the shared exponent; the in-tree kernel uses OCP `floor(log2(amax)) - 2`. They
disagree on **11.8%** of blocks and **3.2%** of dequantized elements --
exactly the blocks where the in-tree recipe has to clip. The
in-tree recipe wins ~4% on mean absolute error but must clip the block maximum
in **11.8%** of blocks; this one never clips. Both stay within the half-step
bound. Cosine against the float32 input is 0.9933 vs 0.9937 — the gap is the
4-bit format, not the kernel.

### Known limitations

- **Nothing is fused.** Every surveyed engine fuses activation quantization
  into the preceding RMSNorm or SiLU (vLLM `rms_norm_dynamic_per_token_quant`,
  TRT-LLM `fused_add_rms_norm_quant`, SGLang `dsv4_norm_rope.cu`, lmdeploy
  `v4_swiglu_and_quant_fp8`). That is the largest remaining win and it is not
  reachable from a plugin: tokenspeed's `quantize_*` are standalone ops whose
  signatures contain no producer.
- **`scale_encoding="ue8m0"` for `fp8_with_scale` is still unserved.** The
  contract leaves its encoded integer layout backend-specific and there is no
  consumer on this card to validate against, so it is left unclaimed rather
  than guessed.
- No serving-level throughput was measured for any of these.

### Companion changes in the tokenspeed tree

- `tokenspeed-kernel/.../ops/quantization/triton.py` — **`triton_quantize_mxfp4`
  wrote a corrupt UE8M0 scale byte for every block whose shared exponent was
  negative**, i.e. every block with `amax < 4`, which is essentially all real
  activations. It computed `scale_e8m0_unbiased.to(tl.uint8) + 127`; Triton's
  float-to-uint8 conversion *saturates* negatives to 0 on the NVIDIA backend
  rather than wrapping, so the stored byte collapsed to 127 (scale `2^0`). The
  E2M1 nibbles stayed correct because they are computed from the float
  exponent, so the corruption was invisible in the data and lived entirely in
  the scale: the tensor dequantized `2^-e` too large. At `amax ~ 0.023` the
  reconstruction error was **600x the input RMS**. Fixed by biasing before the
  conversion, `(scale_e8m0_unbiased + 127).to(tl.uint8)`, which is
  behaviour-preserving on the wrapping backend and correct on both.
- `tokenspeed-kernel/.../ops/quantization/triton.py` — the same registration was
  gated `vendors=frozenset({"amd"})`, and being the only MXFP4 entry that left
  `quantize_mxfp4` with no admissible kernel on any NVIDIA card. Widened to
  `{"amd", "nvidia"}`, mirroring the `fp8_with_scale` widening in #646. The two
  changes belong together: widening the gate without the scale-byte fix would
  have exposed a silently wrong kernel as the PORTABLE fallback.
- `tokenspeed-kernel/test/ops/test_quantization.py` — new
  `test_quantize_mxfp4_scale_byte_across_magnitudes`, a 10-point sweep from
  1e-3 to 1024. The pre-existing test used a fixed input and only ever ran on
  AMD, so it could not see either problem; the sweep fails at 6 of 10
  magnitudes without the fix.
- `tokenspeed-kernel/test/test_kernel_api_selection.py` — new
  `test_mxfp4_quantizer_is_admissible_on_nvidia`, which also pins the honest
  boundary that the Triton kernel declares `scale_layout {linear}` only.

**Wiring gap.** `unieinfra_kernel_sm120/__init__.py::register()` does not call
`register_quant()`. Until it does, this module is inert in the live engine —
the same class of gap as `register_moe()` in 0.2.x, and as `discover_plugins()`
before that.

## 0.5.0 — 2026-08-02

Audits the whole `gemm` family on sm_120. The headline finding is a **silent
correctness bug in the tree**, not a missing kernel; the two kernels added here
are the two real speed gaps that the audit turned up.

### Added

- `unieinfra_bmm_fp8_scaled_sm120` — per-tensor-scaled FP8 batched GEMM on
  FlashInfer's `bmm_fp8` (cuBLASLt). Registered for `gemm/bmm` at
  `Priority.PLUGIN` (16). **`gemm/bmm` had no non-reference kernel at all**, on
  any architecture: the only three registrations tree-wide are the `torch_*`
  references, and the FP8 one dequantizes both operands to FP32 and runs
  `torch.bmm` in FP32. Replacing it is worth **2.2x-14.2x** (449.8 TFLOP/s at
  batch 16, M 1024, N 5120, K 512, versus 17.5 for the reference).
  Cosine 0.9999986-0.9999987 against the reference at every shape.
- `unieinfra_mm_fp8_blockscale_sm120` — FP8 (128,128) block-scale GEMM that
  picks between the two in-tree backends by M. Registered for `gemm/mm` at
  `Priority.PLUGIN` (16). Worth **1.20x-1.54x** for M <= 256 on Qwen3.5-27B
  weight shapes, 1.00x above; peak 690 TFLOP/s. Numerics are identical to the
  backend it routes to (cosine 0.9999985-0.9999987 vs an FP32 reference).

### Why these exist, and what the audit actually found

The starting suspicion was that `gemm/bmm` resolving to a `"reference"`
solution in the PORTABLE band meant every model using `bmm` ran a naive path.
**That part is a false alarm and should not be repeated.** `torch_bmm` is
`torch.bmm`, i.e. cuBLAS strided-batched; it sustains 296.7 TFLOP/s at batch 16
M1024 N5120 K512 against a 415.7 TFLOP/s `torch.mm` ceiling on this card. A
Triton batched GEMM written to displace it was **rejected**: it loses at every
small shape (a ~20 us Triton launch floor against cuBLAS's ~8 us) and wins only
5-7% at the two largest, which is inside run-to-run spread. `solution` is a
provenance label, not a speed claim, and the band it sits in is not evidence.
The cross-engine survey agrees: **no engine has a hand-written bf16 bmm that
runs on sm_120** — TRT-LLM has the only one and it raises
`"CuteDSL BF16 BMM only supports SM 100 family"`.

The same applies to `gemm/mm` dense bf16, which also resolves to `torch_mm`:
376-413 TFLOP/s, ~99% of the raw `torch.mm` ceiling. Nothing to fix.

What *was* broken was invisible from the inventory.

### Companion changes in the tokenspeed tree

Not part of this package. The first is a correctness fix and matters
independently of anything here.

- **`tokenspeed-kernel/.../ops/gemm/deep_gemm.py` — `deep_gemm_mm_fp8_blockscale`
  returns all-NaN on sm_120 and was winning selection.** DeepGEMM's kernels
  need tcgen05/TMEM, which sm_12x does not have; the package compiles and
  launches anyway and produces NaN at every shape (verified raw via
  `deep_gemm.fp8_gemm_nt`, bypassing TokenSpeed entirely, at six shapes from
  M1 N512 K512 to M4096 N5120 K17408 — `nan_frac` 0.25 to 1.000). It was
  registered with `min_arch_version=(9,0)` and **no upper bound**, so sm_120
  inherited a Hopper-era gate. It outranks `triton_mm_fp8_blockscale` (14 > 11),
  so for any shape with `N % 64 == 0` but `N % 128 != 0` — which excludes
  FlashInfer on its `n_align_128` trait but not DeepGEMM on its `n_align_64` —
  it won, and `tokenspeed_kernel.mm(quant="mxfp8")` returned all-NaN end to end.
  Confirmed at N=5184, K=5120.
  The registration is now **skipped entirely** on sm_12x rather than merely
  capability-bounded, because `runtime/layers/dense/fp8.py` sets
  `override="deep_gemm_mm_fp8_blockscale"` from `_use_deep_gemm_fp8` with no
  arch check, and `_resolve_override` does not consult capability — a
  `max_arch_version` alone would have left that path silently poisoned. Skipping
  registration turns it into a loud `NoKernelFoundError`. A `max_arch_version`
  of `(10,3)` is set as well, to document the constraint. After the fix, N=5184
  selects `triton_mm_fp8_blockscale` and returns cosine 0.999669.
  SGLang reached the same conclusion independently:
  `deep_gemm_wrapper/configurer.py:21-22`, *"DeepGEMM requires TMEM/tcgen05
  (SM100+datacenter), not available on SM120"*. vLLM disagrees, but only
  because it pins DeepGEMM's `nv-dev` branch specifically for sm120 support
  (`cmake/external_projects/deepgemm.cmake:32-33`) — a build TokenSpeed does not
  vendor.

- **`tokenspeed-kernel/.../ops/gemm/__init__.py` — out-of-tree kernels were
  structurally excluded from the mxfp8 online-quantization path.**
  `_online_quantize_mxfp8` dispatches on kernel *name* through a hard-coded
  if-chain and raises `ValueError: No online quantization defined for kernel
  ...` for anything it does not recognise, which is every plugin. The escape
  hatch, `_kernel_handles_online_mxfp8`, keyed on `solution == "reference"` —
  so the only way for a plugin to receive unquantized activations was to
  mislabel itself as a reference kernel, which is exactly the mislabelling that
  triggered this audit. It now also honours a declared
  `self_quantizes_mxfp8` trait. Four lines, backwards compatible: the reference
  kernels still match on `solution`.

### Known limitations

- `register_gemm()` is **not** wired into the package's `register()`; the
  caller must add it. `unieinfra_kernel_sm120/__init__.py` is owned elsewhere.
- The `gemm/mm` split point is one constant (M=256) fitted to three weight
  shapes and is wrong for the widest of them (0.93x at M=1, N=34816). The real
  boundary depends on N as well as M. See `docs/kernel-provenance.md`.
- The `gemm/bmm` kernel delegates back to the reference below its crossover,
  because `gemm/bmm` builds traits from N and K only — never batch or M — so
  the condition cannot be declared. Thresholds are fitted, not derived.
- **No serving-level number.** Everything here is kernel-level, and GPU 3 had
  up to five co-tenant processes throughout, so all timings are min-of-9
  blocks. No throughput claim is made or implied.
- **Not ported: the SM120 CUTLASS FP8 blockwise GEMMs.** vLLM
  (`scaled_mm_blockwise_sm120_fp8.cu`), SGLang
  (`fp8_blockwise_scaled_mm_sm120.cuh`) and TRT-LLM
  (`sm120_fp8_gemm_1d1d.cuh`) each ship a hand-written one, all three with a
  small-M swapAB path. That is the real fix for the `gemm/mm` split above and
  would likely beat both backends it chooses between.

## 0.4.0 — 2026-08-02

Gives sm_120 a GDN chunked-prefill kernel that beats the in-tree Triton
pipeline. This is the dominant prefill cost for Qwen3.5-27B and Qwen3.6-35B-A3B
— 48 of 64 layers are GDN linear attention.

### Added

- `unieinfra_gdn_chunk_prefill_sm120` — fuses the FLA pipeline's `chunk_h` and
  `chunk_o` stages into a single kernel, so the per-chunk state checkpoint
  tensor `h` (201 MB for one 8192-token sequence, 403 MB at 16384) is never
  written to HBM, and `v_new` (another 100 MB) is never allocated. The
  recurrent state is also read and written K-last in place, removing the
  `initial_state.transpose(-2, -1).contiguous()` copy the in-tree wrapper does
  on every call of all 48 GDN layers. Registered for
  `attention/gdn_chunk_prefill` at `Priority.PLUGIN` (16), above the only
  admissible in-tree candidate (Triton, `PORTABLE` 4) — FlashInfer's GDN
  prefill is gated on compute-capability major 10 and never reaches sm_120.

  Verified **bit-identical** to the Triton kernel it displaces: max abs diff
  0.0, mean abs diff 0.0, cosine 1.0000000000 across 16 shapes (T from 1 to
  16384, N from 1 to 8, ragged batches, chunk-boundary lengths, bf16 and fp16,
  zero and non-zero initial state). A negative control confirms the comparison
  can fail.

  Measured 1.14x-1.54x end to end at N>=2. **At N=1 the fusion loses**
  (0.90x-0.98x) because its only parallelism is `(V/BV) x N x Hv` and the chunk
  loop is serial; `FUSED_MIN_SEQS=2` delegates single-sequence batches to the
  in-tree kernel unchanged, so N=1 measures 0.99x-1.00x rather than a
  regression. Requires wiring: `register_gdn()` is exported from
  `unieinfra_kernel_sm120.gdn` and is not called by the package `register()`.

### Notes

- `output_h` is declared `{False}`. The speculative-decode h-tracking path asks
  for the per-chunk checkpoints this kernel exists in order not to write, and
  those calls correctly fall back to Triton.
- Tile configuration branches on the driver-reported
  `shared_memory_per_block_optin`, not on the architecture name. sm_120 offers
  101376 B against sm_100's ~227 KB; every `num_stages>=3` configuration of
  this kernel, and a third of vLLM's `chunk_h` autotune space, is unreachable
  here.
- Sweeping the in-tree `chunk_h` and `chunk_o` tile constants found the shipped
  defaults already within 1-3% of the best reachable config on sm_120. Retuning
  is not the lever; the `h` round-trip is.

## 0.3.0 — 2026-08-02

Adds `moe/apply`. Gives sm_120 a block-scaled FP8 MoE, which it did not have:
both in-tree FP8 candidates raise on this architecture.

### Added

- `unieinfra_triton_fp8_block_moe_apply_sm120` — expert apply for FP8 weights
  with `(128, 128)` block scales, registered at `Priority.PLUGIN` (16).
  Fills a hole rather than displacing anything: `flashinfer_trtllm_fp8_moe_apply`
  (SPECIALIZED, 12) raises `unexpected hidden_states_scale shape (1024,)` and
  `flashinfer_cutlass_fp8_moe_apply` (PERFORMANT, 8) raises
  `NotImplementedError: FP8 block scaling not yet implemented for Blackwell`.
  Verified against an oracle that applies the same fp8 rounding the kernel
  applies: cosine 0.9999200 / 0.9999164 / 0.9999163 at 1 / 64 / 512 tokens.
  Requires `ispp % 128 == 0` and declares `ispp_alignment` accordingly — below
  that the loader's w13 scale grid (`2 * ceil(ispp/128)` rows) does not tile
  the N axis.
- `unieinfra_triton_unquant_moe_apply_sm120` — the same kernel for bf16/fp16
  weights, also at `Priority.PLUGIN` (16). Verified against the kernel it
  displaces, `flashinfer_cutlass_unquant_moe_apply`: cosine 0.9999868–0.9999871
  across 1/7/64/512 tokens at the Qwen3.6-35B-A3B shape and 0.9999871–0.9999875
  at the gemma-4-26B-A4B shape; expert-parallel `ep_size=2` reassembles to
  cosine 0.9999894 against the full fp32 oracle.
  **This one displaces a kernel that works.** The argument for it is
  arch-portability — no FlashInfer JIT, no startup autotune, and tiles from
  tables tuned on this exact device — not a measured speedup. No throughput A/B
  was run. If one says otherwise, drop this registration and keep the FP8 one.

### Not wired yet

`unieinfra_kernel_sm120/__init__.py::register()` does not call `register_moe()`,
so these are inert in the live engine until it does. Same class of gap as
`discover_plugins()` in 0.1.0, one level down.

### Reachable only under `--moe-backend auto`

`MoeBackend` is a closed enum with no plugin member, and `solution=` is a hard
filter, so `solution="unieinfra_triton"` can never be requested by flag. Any
benchmark run with an explicit `--moe-backend` measures a different kernel.
Separately: `--moe-backend triton` for a bf16 MoE on sm_120 raises
`NoKernelFoundError` — there is no in-tree NVIDIA Triton unquantized MoE apply.

### Also found while surveying (not fixed here — in-tree bugs)

- `flashinfer_cutedsl_deepep_nvfp4_moe_apply` is unreachable on **every**
  architecture: `moe_plan` asks for `solution="flashinfer_cutedsl_deepep"` while
  the kernel registers `solution="flashinfer_cutedsl"`, and `solution=` is a
  hard filter.
- `triton_mxfp8_precomputed_moe_apply` returns wrong output for
  `activation="silu"` (which is `MoELayer`'s default): its `(1, 32)` branch
  rewrites `[gate | up]` into row-interleaved order, but the `swiglu_arg is
  None` fallback chunks the gemm1 output in half as if it were still
  concatenated. Cosine **-0.0017** against a correct oracle, **0.9989** against
  one reproducing that permutation.
- `triton_mxfp4_moe_apply` cannot compile on sm_120:
  `ptxas: Feature '.tile::scatter4' not supported on .target 'sm_120a'`.

## 0.2.0 — 2026-08-02

Makes the decode kernel usable in the live engine. 0.1.0 registered it but it
could not be selected: FlashInfer's `plan()` does host-side scheduling, and
TokenSpeed captures decode in CUDA graphs.

### Changed

- `unieinfra_flashinfer_paged_mha_decode_sm120` now splits plan from run. The
  wrapper is constructed with `use_cuda_graph=True` over persistent
  `qo_indptr` / `paged_kv_indptr` / `paged_kv_indices` /
  `paged_kv_last_page_len` buffers, one bucket per (batch, tokens-per-request,
  geometry); `plan()` is hoisted into a new `prepare_paged_mha_decode()` that
  the runtime calls from outside the captured region; only `run()` is captured.
  This is SGLang's and vLLM's protocol, not a new invention — see
  `docs/kernel-provenance.md` for the cross-engine table it came from.
- The CSR page index is now built without a device-to-host sync. The previous
  version compacted page ids with a boolean mask and used the result's length as
  a Python int; page ids are now scattered to their CSR destinations with
  out-of-range columns aimed at a trailing scratch slot.
- Plan reuse within a step. `prepare` bumps an epoch counter and a slot planned
  in the current epoch against the same metadata tensors is reused, so the 16
  full-attention layers of a step share one plan instead of paying for 16. This
  closes the "plan-per-call" cost recorded in 0.1.0. A runtime that never calls
  `prepare` keeps re-planning per call and stays correct.
- The FlashInfer float workspace grows on demand instead of being fixed at
  128 MiB. Split-KV scratch at head_dim 256 is ~148 MB for a small bucket and
  scales with batch; 0.1.0 would have failed with `Buffer overflow when
  allocating memory for batch_prefill_tmp_v`. Displaced buffers are retained
  rather than freed — graphs captured against them baked their address.

### Verified

Against `triton_mha_decode_with_kvcache_cached` through the ops dispatch, at
24 q heads / 4 kv heads / head_dim 256 / page_size 832, bf16:

- eager and hoisted-plan, `max_seqlen_q` 1 and 4, batch 1/8/33:
  cosine 0.99999428-0.99999893
- capture + replay, batch 8 and 33, three replays each with fresh page tables
  and sequence lengths: cosine 0.99999452-0.99999851
- negative control — replay *without* re-planning, across page-count changes:
  cosine 0.246 / 0.427 / 0.465. The re-plan is load-bearing.
- padding rows (`seq_len` 0 and 1): output finite, live rows cosine 0.99999440
- missing hook under capture raises a `RuntimeError` naming the hook

Per-call latency vs Triton, plan hoisted, on a shared GPU: 4.8x at batch 1,
5.4x at batch 8, 1.13x at batch 32, 1.10x at batch 128, 1.53x at batch 32 with
`max_seqlen_q=4`. Serving-level throughput was not measured.

### Known limitations

- One device-to-host sync per step remains inside `plan()`. vLLM and SGLang
  avoid it with a host mirror of the sequence lengths; TokenSpeed's graph-path
  decode metadata has none.
- Each captured bucket holds a private 8 MiB FlashInfer int workspace. It cannot
  be shared: a captured graph reads its schedule from that buffer, and target
  and draft graphs are both planned before either replays.
- **DFLASH block decode is not served.** Its `seq_lens` are written by the
  drafter inside the captured graph, so no outside-the-graph schedule can be
  correct. The kernel reports the unprepared bucket rather than running on the
  wrong lengths.

### Companion changes in the tokenspeed tree

This version is inert without them:

- `tokenspeed-kernel/.../ops/attention/__init__.py` — new `mha_decode_prepare`
  op. It selects the same kernel `mha_decode_with_kvcache` would, looks for a
  `prepare` attribute on it, and forwards; returns `False` when the selected
  kernel needs no host pass. Vendor-neutral by construction — nothing about
  FlashInfer appears in it.
- `tokenspeed-kernel/python/tokenspeed_kernel/__init__.py` — export it.
- `runtime/layers/attention/backends/mha.py` — `_prepare_decode_kernel()`,
  called from `init_forward_metadata` (decode), from
  `init_forward_metadata_capture_cuda_graph` immediately before
  `torch.cuda.graph(...)` records the bucket, and from
  `init_forward_metadata_replay_cuda_graph` immediately before
  `graph.replay()`. One call per cache group, since heterogeneous block sizes
  give each group its own page table and page size.

  These three metadata hooks are the only decode code TokenSpeed runs *outside*
  the captured region, which is why the hook has to live there rather than in
  the kernel. That is the honest answer to "can this be solved inside the
  kernel alone": no. The kernel can detect the problem and refuse, and it does,
  but it cannot fix it — nothing it is called from executes outside the graph.

## 0.1.0 — 2026-08-02

First release. Gives sm_120 a FlashAttention-class MHA path, which the engine
did not have on this architecture at all.

### Added

- `unieinfra_flashinfer_paged_mha_extend_sm120` — MHA extend against a paged KV
  cache, on FlashInfer's general paged prefill wrapper. Registered for
  `attention/mha_extend_with_kvcache` at `Priority.PLUGIN` (16), above the
  in-tree trtllm entry (12) and Triton (4).
  Verified against the Triton kernel it displaces: max abs 6.1e-05, cosine
  0.99999678, and 0.99999583-0.99999690 across padded-query shapes.
- `unieinfra_flashinfer_paged_mha_decode_sm120` — MHA decode through the same
  wrapper, treating decode as a uniform short-query extend. Verified: cosine
  0.99999523 at `max_seqlen_q=1`, 0.99999601 at `max_seqlen_q=4` (speculative
  draft block).
  **Not usable in the live engine yet** — FlashInfer's `plan()` performs
  host-side scheduling that CUDA graph capture rejects
  (`operation not permitted when stream is capturing`), and TokenSpeed captures
  decode. Registered but must be left unselected until the plan step is hoisted
  out of the captured region.

### Why these exist

TokenSpeed had no FlashAttention path for MHA on sm_120 through four separate
dead ends, none of them hardware:

1. `ops/attention/flash_attn/` registers FA4 only, every entry pinned
   `min_arch == max_arch == (10, 0)` — datacenter Blackwell exclusively.
2. The runtime image's `flash_attn` distribution is an empty namespace
   directory: `__file__` is `None`, `dir()` is empty.
3. FlashInfer's trtllm XQA decode rejects page sizes outside
   {16, 32, 64, 128}; Qwen3.5's GDN state row forces the attention block to 832.
4. FlashInfer's AOT `BatchDecodeWithPagedKVCacheWrapper` reports
   `Unsupported group_size: 6` at head_dim 256 — 24 query heads over 4 KV heads.

Meanwhile vLLM 0.26 runs FlashAttention-2 on the same GPU for the same model, at
block size 784. The gap was wiring, not silicon.

### Known limitations

- **Plan-per-call.** FlashInfer's wrappers are built to plan once per forward
  pass and run per layer; both kernels plan on every call, so all 16
  full-attention layers pay for scheduling one of them needed. A data-pointer
  cache key is not safe here because TokenSpeed reuses metadata buffers in place
  across steps, which CUDA graph capture requires it to do. Unmeasured.
- **Narrow traits by design.** Sliding windows, logit caps, attention sinks and
  the block-scaled path are all declared unsupported, so selection falls back to
  Triton for any request needing them. A kernel that claims a feature it does
  not implement wins selection and returns wrong answers.
- **Decode is registered but unsafe under CUDA graphs.** See above.

### Companion changes in the tokenspeed tree

Not part of this package, but this package is inert without them:

- `runtime/utils/common.py` — added `ensure_kernel_plugins_loaded()`.
  `discover_plugins()` was called nowhere in the TokenSpeed runtime, so the
  reserved plugin band 16-19 was permanently empty and any platform whose best
  kernel lived out of tree silently ran the in-tree fallback.
- `runtime/entrypoints/engine.py`, `runtime/engine/event_loop.py` — call it from
  both the launcher and the scheduler subprocess. A registry does not cross a
  fork: discovering only in the launcher produces a plugin that logs its own
  registration and is then "not found" by the worker that runs the model.
- `tokenspeed-kernel/python/setup.py` — `GROUP_REQUIRED_ARCHS`, so the
  `attn_res` kernel group (which needs `tcgen05` tensor-memory instructions that
  sm_120 does not have) is skipped for unsupported targets instead of failing
  the whole wheel. Dropped groups are named on stdout, not skipped silently.
- `tokenspeed-kernel/.../ops/attention/triton/mha_prefill.py` — branch tile size
  on the device's reported shared memory rather than the architecture family.
  sm_120 satisfies `is_hopper_plus` but carries ~100 KB like sm_86/sm_89 rather
  than sm_100's 227 KB, so it took the Hopper tiles and overflowed.
- `tokenspeed-kernel/.../platform.py` — `max_shared_memory_per_sm` read
  `getattr(props, "max_shared_memory_per_block", 0)`, an attribute torch does
  not expose, and so returned **0 on every NVIDIA device**. Now reads
  `shared_memory_per_block_optin` (101376 here, matching exactly the limit
  Triton reports when it refuses to launch). Also added the missing sm_120
  entry to `generation_name`.
