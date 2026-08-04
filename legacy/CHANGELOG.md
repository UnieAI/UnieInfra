# Changelog

All notable changes to UnieInfra are documented in this file, written to be
legible to the community — each entry says *what* changed and *why*, with links
into the design docs where deeper rationale lives.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
UnieInfra will adopt [Semantic Versioning](https://semver.org/) once the first
release is cut; until then all changes accumulate under **[Unreleased]**.

## [Unreleased]

### Added
- **`unieinfra serve` is now UnieInfra's own interface, with its own defaults.**
  `unieinfra` and `vllm` remain the same console script, but they are no longer
  the same product: `unieinfra serve` applies a small set of measured serving
  defaults and prints a reasoned banner at startup, while `vllm serve` stays
  byte-for-byte upstream. The gate is `sys.argv[0]`, which is what lets both
  arms of a comparison run on one binary. The OpenAI-compatible HTTP API is
  unchanged — this is the launcher and the configuration surface, not the wire
  protocol. See `unieinfra-vllm/README.md` §"Defaults", and E49 in
  `docs/experiments-log.md`.
  - **Speculative decoding (MTP) auto-enables when — and only when — the
    checkpoint actually ships MTP weights**, read from the tensor index rather
    than trusted from the config. Qwen3.6-35B-A3B (19 `mtp.*` tensors) and
    Qwen3.5-27B (15) get it; Gemma 4 and Qwen3-8B ship none and silently run
    without it. Depth is load-aware (deep at low batch, shallow at high batch)
    via upstream's `num_speculative_tokens_per_batch_size`. Two costs are
    documented rather than hidden: the KV pool is still sized from the static
    maximum, and a runtime-varying verification length downgrades CUDA graphs
    from `FULL_AND_PIECEWISE` to `PIECEWISE` (`UNIEINFRA_SPEC_SCHEDULE=static`
    opts out).
  - **Host KV offload defaults on, sized from measured available host RAM** —
    `min(10% of available, 32 GiB, available − 64 GiB reserve)`, and off
    entirely below an 8 GiB floor or when available RAM cannot be read. Never a
    ratio of the device pool: that is how the retiring engine OOM-killed a
    server at startup on a 139 GB allocation (E17).
  - **Prefix caching is left on `auto`** so models that cannot support it
    degrade instead of failing. Hybrid/Mamba models — including
    Qwen3.6-35B-A3B — force it off, and the banner says so.
  - **fp8 KV cache ships opt-in, not on.** On sm_120 FlashAttention refuses fp8
    KV (it needs FA3 on SM90 or FA4 on SM100) and vLLM responds by silently
    substituting another backend, one of whose prefill paths dequantizes fp8
    pages back to bf16 — capacity win, bandwidth loss. The accuracy question is
    also still open. Enable with `UNIEINFRA_DEFAULT_KV_FP8=on` or
    `--kv-cache-dtype fp8`.
- **`unieinfra-vllm/` — UnieInfra's engine base, a hard fork of vLLM vendored as
  source.** UnieInfra is an agent-oriented inference engine whose design is to
  **absorb** the valuable parts of other engines as owned in-tree source rather
  than depend on them as libraries. vLLM is the first absorption: the tree at
  upstream commit `d02df748` (`v0.13.0rc1-6959`, 2026-07-24) now lives at
  `unieinfra-vllm/`, and **we do not sync upstream** — no merges, no rebases.
  - *Attribution:* upstream is Apache-2.0 and stays that way. `LICENSE` is
    unmodified, every upstream SPDX header is intact, upstream's README is
    preserved as `UPSTREAM_README.md`, and `unieinfra-vllm/PROVENANCE.md` lists
    every file we removed, renamed or modified — which is both Apache-2.0
    §4(b) compliance and what keeps the fork auditable by `diff -r`. `NOTICE`
    and `ACKNOWLEDGEMENTS.md` record provenance **per absorbed component**,
    because licenses differ per component (vLLM and FastVideo Apache-2.0,
    JetSpec MIT).
  - *Naming:* the distribution is `unieinfra-vllm` and the console script
    `unieinfra` (with `vllm` kept as an alias), but the **import package is
    still `vllm`** — a deliberate, documented deferral. Renaming 3,837 files on
    day one would be an enormous diff with no functional gain. See
    `unieinfra-vllm/README.md` § Naming for what would force it earlier.
  - *Verified on sm_120:* builds from source (398 targets, `TORCH_CUDA_ARCH_LIST=12.0a`),
    serves Qwen3-8B with coherent greedy output, and the import provably
    resolves inside the vendored tree with zero modules leaking to any
    installed `vllm`. E18's `VLLM_USE_FLASHINFER_SAMPLER=0` sm_120 shim is **no
    longer needed** at this pin. Build runbook and a troubleshooting table for
    every error hit are in `docs/guides/build-and-run.md` §10; layout and the
    absorption model in `docs/design/project-layout-and-absorption.md`.
- **UnieInfra's kernel selection layer is now wired directly into the engine's
  dense-GEMM path**, replacing the plugin route. `dispatch_unquantized_gemm()`
  gained a UnieInfra arm calling `torch.ops.unieinfra.linear`; there is no
  `--quantization unieinfra` flag and no `vllm.general_plugins` hop. The plugin
  shape was correct while the base decision was open (it proved portability
  against *stock* vLLM in E34) but it can only reach the seams upstream chose to
  expose — the base assessment counted 7 of 19 hot-path dispatch decisions as
  unreachable without patching source. Owning the source removes that ceiling.
  - *Measured:* Qwen3-8B bf16 on sm_120, one process per arm, rotated, two
    pairs: batch-1 decode **87.04 → 90.07 tok/s (+3.47%)**, with the winning
    kernel *proven* per shape (`fast=648, delegated=0`, the same counter E34
    recorded) and the control arm proven clean (`fast=0`). See E37.
  - *Set `UNIEINFRA_GEMM=0`* to restore upstream's `F.linear` path exactly.
- **`unieinfra_decode_batch_gemm_triton` — a first-party decode GEMM for the
  batch sizes a *served* engine actually decodes at (`16 <= M <= 256`).** E44's
  hardware-counter audit found vLLM's `C = 128` decode step losing about a third
  of the machine on its weight projections to wave quantization rather than to
  any hardware ceiling: the incumbent CUTLASS tile launches 384 CTAs that are
  shared-memory limited to one resident CTA per SM on a 188-SM part, so the grid
  drains in three rounds of 188/188/8 and `384 / (3 x 188) = 68%` of the GPU is
  used — which is exactly the `60.6 / 90.5` its counters reported. This kernel
  recovers it, in portable Triton, on any NVIDIA part from sm_80 and on AMD.
  - *Measured* (Qwen3-8B on sm_120, cold L2, CUDA-graph timing, interleaved A/B
    against the cuBLASLt kernel the engine runs today): `gate_up` **1.38x** at
    `M = 128` and 69.7% → 96.2% of achievable DRAM bandwidth; `down` 1.17x,
    `o_proj` 1.20x, `qkv` 1.05x; 14 of 16 (shape, batch) cells faster, best
    2.12x. See E47.
  - *The intuitive fix was refuted.* "Launch more, smaller CTAs until the rounds
    divide evenly" is slower than a wide tile that leaves **half the SMs idle**:
    the winning configuration drops waves per SM from 2.04 to **0.51** and moves
    the same DRAM bytes in less time. What was missing was memory-level
    parallelism per CTA, not CTAs. The utilisation model is kept as a
    diagnostic — it reproduces the incumbent's measured loss — and deliberately
    is not what config selection maximises.
  - *Gated declaratively and narrowly.* Traits `m_min_16`, `m_le_256`,
    `n_le_32768`, `n_align_64`, `k_align_128`; the trait gate and the runtime
    profitability rule are the **same predicate**, checked by a test over every
    `M` in `1..512` — the range CUDA-graph capture walks — so no shape can be
    declined mid-capture (the E38 lesson). Disjoint from the E33 batch-1 decode
    GEMV (`m_le_2`), which is unchanged. Any shape it declines falls through to
    `F.linear`, i.e. bit-identical to the incumbent.
  - *Set `UNIEINFRA_DECODE_BATCH_GEMM=0`* to disable it without unregistering
    it, which is what the end-to-end A/B arms differ by.
- **Attention backends are now chosen by capability, not by a hardcoded arch
  tree** — Step A of `docs/design/kernel-system-design-principles.md` §5.
  Upstream resolves the backend inside `CudaPlatform._get_backend_priorities()`,
  a function whose body is a tree of `device_capability.major == 10` / `== 12`
  branches. Each of those branches is now a **registration** in the same
  capability registry the GEMM path uses: "prefer FlashInfer on datacenter
  Blackwell" is a spec requiring the `compute:tcgen05` *feature*; "except when
  non-causal" is a trait on that spec; "sm_120 gets Triton MLA" is what remains
  once every other MLA spec is either feature-gated or bounded above.
  `AttentionSelectorConfig` is unchanged — it stays the request description, and
  only the resolution step moved.
  - *Why it matters:* the open-ended `>=` arch gate is a real, crashing bug
    class in this engine — E37 found upstream's `ll_bf16` router GEMM gated on
    `has_device_capability(90)`, which admits sm_120 and killed EngineCore at
    warmup. No registration here may state a bare `min_arch_version`; a test
    enforces it.
  - *Portability is guaranteed, not hoped for:* `TRITON_ATTN` and `TRITON_MLA`
    are registered with no vendor and no arch requirement at all, so tightening
    any other gate degrades to a slower success rather than a `NoKernelFound`.
    `UNIEINFRA_ATTENTION_OBJECTIVE=portability` selects them outright.
  - *Behaviour is unchanged where it already works:* the registry reaches
    upstream's decision on **150/150** request shapes across simulated
    sm_80/86/89/90/100/120, including the sparse-MLA ordering flip on
    `num_heads <= 16`. Where it cannot decide, it declines and upstream's
    selector runs untouched, so the layer can only ever change a decision it is
    able to make. See E43.
  - *Set `UNIEINFRA_ATTENTION=0`* to restore upstream's selector exactly.
- **Two sm_120 fixes in the vendored engine**, both recorded in
  `unieinfra-vllm/PROVENANCE.md`: the `ll_bf16` CuteDSL router GEMM is no longer
  offered on compute capability ≥ 12 (upstream's `has_device_capability(90)`
  gate is open-ended, and CuteDSL cannot lower the kernel for a part with no
  tcgen05 — the same failure shape E30 catalogued in our own tree), and the
  distribution rename's fallout on `importlib.metadata.version("vllm")`, which
  was silently disabling Triton engine-wide.
- **`--quantization unieinfra` now serves fp8 and NVFP4 checkpoints**, not just
  bf16. On a quantized checkpoint the plugin delegates the *entire* checkpoint
  path — parameter creation, the weight loader, MoE, ModelOpt's excluded
  modules, the fp8 KV-cache method — to vLLM's own quantization config for that
  format, and overrides exactly one method, `LinearMethodBase.apply`. Both arms
  of a comparison therefore hold identical weights and only the GEMM differs.
  Two new first-party Triton kernels do the work at decode:
  `unieinfra_decode_gemv_fp8_triton` (W8A16) and
  `unieinfra_decode_gemv_nvfp4_triton` (W4A16).
  - *The design claim:* at decode, a quantized checkpoint should be served
    **weight-only**, not in its own symmetric format. At batch 1 the activation
    is ~0.03% of the bytes a projection moves, so quantizing it saves nothing
    and costs a launch — while the W8A8/W4A4 tensor-core GEMM it enables wastes
    over 98% of every instruction on padding. Keeping the activation in bf16 is
    both faster and more accurate.
  - *Measured* (Qwen3-8B, sm_120, vLLM 0.25.1, one process per arm, 2 rounds x
    3 repeats, median), against vLLM's **own** fp8 / `modelopt_fp4` paths on the
    same checkpoint: batch-1 decode **+7.4% (fp8)** and **+17.8% (NVFP4)**. At
    C >= 32 our kernels are outside their profitability band and we are
    **2.2-2.9% slower**; a control arm with the kernel unregistered isolates
    that as the cost of the Inductor-opaque dispatch op, not of the delegation
    (the fp8 control matches vLLM native to within 0.07%). Reported as a loss
    because it is one. See E38 in `docs/experiments-log.md`.
  - *Accuracy:* against the unquantized bf16 answer, our W8A16 fp8 measures
    rel-err 0.026 against vLLM's W8A8 0.038, and our W4A16 NVFP4 0.095 against
    vLLM's W4A4 0.134 — not quantizing the activation is worth ~30% of the
    error on both formats.
  - *Also:* `unieinfra_kernel.dispatch.gemm_format_signature` now reproduces all
    four scaled branches of the tokenspeed `gemm.mm` signature builder
    (`fp8`/`nvfp4`/`mxfp8`/`mxfp4`) plus a weight-only variant; `mxfp8`/`mxfp4`
    have signatures and unit tests but no engine seam and no checkpoint to test
    against yet. `UNIEINFRA_KERNEL_SELECTION_DUMP` writes the per-shape
    selection record from inside vLLM's worker process, which is the only way to
    prove which kernel ran.
- **`unieinfra-scheduler` — UnieInfra's admission policy as an out-of-tree vLLM
  scheduler**, the scheduling half of the vLLM re-base, loaded through vLLM's own
  `--scheduler-cls` seam with no file under `third_party/vllm/` touched. It
  derives the running-request cap as a **minimum over every binding resource**,
  asking each KV cache group's own `KVCacheSpec` what one request costs instead
  of re-deriving it from `context_len` — which structurally closes all three of
  the old TokenSpeed policy's successive blind spots at once (the recurrent/Mamba
  state pool that caused E28/E31's retractions and device-side assert,
  sliding-window budgets, and speculative decoding's `num_lookahead_tokens`).
  A `grows_with_context` flag ensures a fixed-size state pool is never given the
  steady-state discount, which is what the E28/E31 over-admission actually was.
  Also ships an **SLO-aware prefill gate**, off by default.
  - *The headline is a refutation, and it is the point.* This work set out to
    port `derive_adaptive_max_batch`, the mechanism E35 credited with a +16.9%
    throughput advantage at C=128. **On vLLM 0.25.1 that premise does not hold.**
    Stock vLLM on this hardware already runs `max_num_seqs=1024`
    (`vllm/engine/arg_utils.py:2432-2435`) — 2.1x the cap the old policy derived,
    and above every concurrency we benchmark — with the CUDA-graph ladder already
    captured to 512. **The cap never binds, so there is no admission wall to
    lift.** Measured from the other direction too: forcing prefill deferral, up to
    freezing it for 8 steps in 9, moves C=128 throughput by **+0.3%** across six
    independent server launches. See E36 in `docs/experiments-log.md`.
  - *What the gate is good for, stated honestly:* it is a **TTFT/TPOT dial, not a
    throughput win** — at C=128, mean TTFT 1248 → 1555 ms (+25%) against mean TPOT
    44.4 → 43.2 ms (−2.7%). Since UnieInfra is positioned as an agent-oriented
    engine where interactive TTFT dominates, that trade is backwards for the
    target workload, so the gate **ships off**: the default objective is
    `interactive`, which leaves admission timing identical to stock vLLM.
  - *Where the E35 gap actually lives:* by arithmetic on measured quantities,
    stock vLLM's batch-128 decode step costs ~28 ms against the ~21 ms the old
    base's numbers imply. That is a kernel and runtime property, not a scheduling
    one, and it is where the next experiment should go. Consequently
    `docs/design/scheduling-improvements.md` items 1–3 are **already satisfied by
    vLLM** and are struck from UnieInfra's roadmap. Full rationale in
    `docs/design/vllm-admission-policy.md`.
  - *Seam limit worth recording:* the CUDA-graph capture ladder is sized and
    captured before any scheduler exists (`vllm/config/vllm.py:1330`,
    `vllm/v1/engine/core.py:320` vs `:150`), so a `scheduler_cls` plugin can only
    **lower** the cap, never raise it. Extending the ladder from measured KV
    capacity — the other half of what the old base did — has no plugin hook.
- **UnieInfra's kernel selection layer now runs inside stock vLLM** — the first
  step of re-basing onto vLLM's architecture. `unieinfra-kernel` became a
  self-contained distribution: the capability registry and selection pipeline
  (`registry` / `selection` / `platform` / `signature`, 2,586 LOC) were ported
  into `unieinfra_kernel/selection/`, which imports nothing but `torch` and the
  standard library. A `vllm.general_plugins` entry point registers an
  out-of-tree `unieinfra` quantization config whose `get_quant_method` returns a
  `LinearMethodBase` — so `vllm serve <model> --quantization unieinfra` routes
  every dense Linear GEMM through `select_kernel`, and the E33 decode GEMV is
  selected for `qkv_proj`, `o_proj` and `down_proj` at batch 1.
  - *Measured:* Qwen3-8B bf16 on sm_120, vLLM 0.25.1, one process per arm, two
    independent pairs: batch-1 decode **84.84 → 88.52 and 85.19 → 88.68 tok/s
    (+4.34% / +4.10%)**, coherent greedy output. See E34 in
    `docs/experiments-log.md` and `docs/design/vllm-kernel-plugin.md`.
  - *Why a plugin and not a fork:* no file under `third_party/vllm/` is touched.
    The three APIs consumed are among vLLM's most stable and are CI-tested in
    tree, so the package survives a 34-commits/day upstream without a rebase
    tax, and it works on either base — `unieinfra_kernel.boundary` still
    registers into `tokenspeed_kernel`'s registry when that package is present,
    so the existing engine is unchanged.
  - *Notable:* letting the declarative trait gate (`m_le_8`, `n_le_16384`)
    choose across **all** dense Linears beat a hand-picked `qkv_proj,o_proj`
    list by more than 2× (+4.2% vs +1.6%) — the difference is `down_proj`, which
    the human list had missed. Routing all dense Linears is therefore the
    default.
- **Attention backends declare whether they can serve mixed prefill+decode
  batches**, and `--enable-mixed-batch` is now rejected at startup on a backend
  that cannot. New `AttentionBackend.supports_mixed_batch` (default `True`,
  `False` for the MHA family and MSA) plus `validate_mixed_batch_backend()`,
  called from the engine before the scheduler is constructed. The two backends'
  internal `assert not forward_mode.is_mixed()` checks became
  `NotImplementedError`, because an `assert` disappears under `python -O` and the
  unguarded path would run extend metadata over decode rows — silent wrong
  output rather than a crash.
  - *Why:* `--enable-mixed-batch` was accepted by every configuration but only
    fails on the first iteration that actually mixes a prefill chunk with
    decodes, which can be many minutes into serving. It now fails in the first
    seconds, naming the backend and the flag. See E31 in
    `docs/experiments-log.md` for the measurement behind this, including why
    mixed batching is worth wanting: with it off, a step that admits any prefill
    runs **zero** decodes, which at concurrency 256 is 38.6% of wall clock with
    no decode progress.
- **`EXTRA_ARGS` passthrough in `benchmark/launch/launch_unieinfra.sh`** for
  `tokenspeed serve` flags that have no dedicated harness knob yet (recorded in
  `launch.json` so saved results stay reproducible).
- **Project genesis — hard fork of TokenSpeed.** UnieInfra begins as an
  independent, community-driven fork of
  [TokenSpeed](https://github.com/lightseekorg/tokenspeed) (MIT, LightSeek
  Foundation), vendored into the repository root as first-party code:
  `python/` (engine), `tokenspeed-kernel/`, `tokenspeed-kernel-amd/`,
  `tokenspeed-mla/`, `tokenspeed-scheduler/`. There is **no upstream sync** — the
  fork is the base we build on. Attribution is preserved in `LICENSE` (MIT),
  `NOTICE` (fork lineage + vendored third-party components), and
  `README.tokenspeed.md` (the upstream engine's original README).
  - *Why:* UnieInfra targets a hardware-universal, extreme-performance engine
    (max RPS under SLO) with a vLLM-compatible frontend; TokenSpeed's
    capability-based, pluggable kernel registry is the right foundation,
    especially for agent-driven kernel optimization. See
    `docs/design/unieinfra-architecture.md`.
- **Design docs.**
  - `docs/design/unieinfra-architecture.md` — the ①–⑦ component stack and the
    extension map (where each thing we build lands).
  - `docs/design/tokenspeed-sm120-support.md` — first hardware bring-up plan for
    NVIDIA sm_120 (RTX PRO 6000 / GB202) workstation GPUs.
- **Reference submodules** under `third_party/` (not vendored, not
  redistributed): `vllm`, `modular` (Mojo/MAX), `tilert`.

### Known issues / findings
- **sm_120 is mis-modeled as datacenter Blackwell (B200).** Empirically confirmed
  on 4× RTX PRO 6000 (milestone M0a): `current_platform()` reports
  `is_blackwell_plus = True` and advertises the `compute:cluster` feature, which
  sm_120 does **not** have (no clusters > 1, no `tcgen05`/tensor-memory). As a
  result the capability-based kernel selector can offer `tcgen05` kernels that
  fail at runtime. `tensor_core:f4` and `memory:tma` are valid on sm_120 and are
  kept. Fix planned in `docs/design/tokenspeed-sm120-support.md` (Step A/B):
  drop `compute:cluster` on sm_120 and add a datacenter-only `compute:tcgen05`
  feature to gate the affected kernels. `interconnect` correctly detected as
  `pcie` (no NVLink) → TP + NCCL path.

### Development — sm_120 bring-up progress
- **Kernel package builds and imports on sm_120 (RTX PRO 6000).** The vendored
  kernel package compiles for `TOKENSPEED_CUDA_ARCH=120a` — confirming the 17
  hand-written native kernel groups are Blackwell-generic (not `tcgen05`), so
  only runtime JIT/DSL kernels are arch-locked. Built and imported inside the
  org's `nctu6/unieinfra-autobuild-dev:cuda-12.9.1-blackwell_only` image.
- **Kernel-selection audit on sm_120 (27 operators):** 23 select portable
  (triton/flashinfer/cuda) winners; only 3 select a datacenter `deep_gemm`
  kernel — all DSA (DeepSeek/GLM sparse attention), off the Qwen3.x path; 1
  (`quantization/mxfp4`) has no sm_120 candidate (fp4-only). Details and the
  reproducible build recipe are in `docs/design/tokenspeed-sm120-support.md` §9.
- Known reproduction friction to fold into a pinned dev image / source fix:
  cuBLAS headers ship via pip wheels (need `CPATH`); HEAD pulls cu13 kernel
  wheels needing cu13 runtime libs on a cu12.9 base; flashinfer version-check
  must be bypassed. The clean source fix is to make datacenter-kernel imports
  lazy so sm_120 never loads kernels it won't call.

### Milestone — first end-to-end run on sm_120 (M1)
- **UnieInfra runs end-to-end on NVIDIA sm_120 (RTX PRO 6000) and generates
  correct output.** A dense Qwen3-8B forward+decode produces coherent text
  ("The capital of France is" → " Paris. … Rome … Madrid … Berlin …") — the full
  stack (C++ FSM scheduler + model executor + Triton attention + sampling +
  sm_120-native kernels) works. Reproduce with `scripts/m1_forward_smoke.py`.
- **First real sm_120 kernel gap identified:** bf16 *unquantized* MoE has no
  working sm_120 kernel (the flashinfer cutlass MoE hits a version-skew FFI
  mismatch; the Triton MoE covers only quantized weights). This blocks MoE
  models (Qwen3.6-35B-A3B) and is the first target for portable-kernel work —
  see `docs/design/tokenspeed-sm120-support.md` §9. Engine + scheduler install
  needs `scikit-build-core` + `libssl-dev`; `python/` installs `--no-deps`
  because `tokenspeed-mooncake` (P/D-only) is unpublished.
- Product direction logged: the engine should auto-detect single- vs multi-GPU
  topology (zero-config parallelism), a frontend responsibility.

### Notes
- Internal package/module names (`tokenspeed`, `tokenspeed_kernel`, …) are
  retained for now; a deliberate rename to the UnieInfra namespace is planned as
  a separate, well-documented change once the engine runs on the target
  hardware. Priority order for the project is: **make it run → verify
  correctness → optimize.**
