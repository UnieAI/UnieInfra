# Design: sm_120 (RTX PRO 6000) support for the TokenSpeed engine in UnieInfra

Status: **Draft / recon complete, implementation not started**
Date: 2026-07-24
Owner: royshih
Scope: make the TokenSpeed engine core run on NVIDIA **sm_120** (GB202 / RTX PRO 6000, RTX 50-series silicon), as a prerequisite to benchmarking `tokenspeed` vs `vLLM` and to basing UnieInfra on tokenspeed.

---

## 1. Context & goal

UnieInfra = vLLM-compatible frontend + **tokenspeed engine core** + load-class routing (see the project brief). Upstream tokenspeed only builds for datacenter Blackwell (B200) and Hopper; our dev/target hardware is **4× RTX PRO 6000 (sm_120)**. Before any tokenspeed-vs-vLLM benchmark is meaningful, tokenspeed must actually run on sm_120. sm_120 support is also a capability UnieInfra must **own independently of upstream** (upstream cares only about B200/MI355X).

This doc: what the gap actually is, why tokenspeed's kernel architecture makes it tractable, the concrete work items, and the risks.

Non-goals: porting TileRT (closed B200 binary, cannot run here); the vLLM-frontend adapter (separate design).

---

## 2. Hardware reality — why this is not a build flag

| | sm_100 (B200, upstream target) | sm_120 (our RTX PRO 6000) |
|---|---|---|
| Tensor Memory (TMEM) | yes | **no** |
| `tcgen05.mma` (async large-tile MMA → TMEM) | yes | **no** |
| Clusters > 1 / CTA-pair MMA | yes | **no** (single-CTA only) |
| FP4/FP8 tensor cores | yes | yes (via regular `mma.sync`) |
| MMA path | `tcgen05` + TMEM | `mma.sync` / `wgmma.async` |

`tcgen05` PTX assembled for `--gpu-name=sm_120` → `ptxas: instruction not supported`. **No fallback; a tcgen05 kernel must be rewritten**, not recompiled. tokenspeed builds kernels only for `CUDA_ARCH_LIST="9.0a 10.0a"` (`docker/Dockerfile.nvidia:9`).

Interconnect: RTX PRO 6000 boxes are typically **PCIe, no NVLink** → any NVLink/IB-only comm path (DeepEP) must be avoided.

---

## 3. The good news — tokenspeed has a capability-based pluggable kernel framework

tokenspeed's kernel stack is **two-level pluggable**, and the lower level selects kernels by **hardware capability**, which is almost exactly the mechanism we need.

### 3.1 Level 1 — `tokenspeed_kernel` selection framework
- Each op family lives under `ops/<family>/<solution>` (per `AGENTS.md`). Every family ships **multiple backend solutions**, and — verified — **every family has a Triton (portable) solution**:

  | family | solutions present | Triton? | tcgen05/vendor backends (sm_100-bound risk) |
  |---|---|---|---|
  | attention | `triton/`, `flashinfer/`, `cuda/`, `cute_dsl/`, `tokenspeed_mla/`, `flash_mla/`, `flash_attn/`, `msa.py`, `gluon/`, `deep_gemm/` | ✅ | cute_dsl, tokenspeed_mla, flash_mla, msa |
  | gemm | `triton.py`, `cuda.py`, `cute_dsl.py`, `deep_gemm.py`, `flashinfer.py`, `trtllm.py`, `gluon.py` | ✅ | cute_dsl, deep_gemm, trtllm |
  | moe | `triton/`, `cuda.py`, `flashinfer/`, `gluon/` | ✅ | flashinfer/cutlass_nvfp4, cutlass_fp8, trtllm_mxfp4 |
  | quantization | `triton.py`, `cuda.py`, `flashinfer.py`, `trtllm.py` | ✅ | trtllm |
  | sampling | `triton/`, `cuda.py`, `cute_dsl.py`, `flashinfer.py`, `gluon/` | ✅ | cute_dsl |
  | activation / layernorm / embedding / kvcache / conv / transform / communication | `triton*` + others | ✅ | mostly none (communication: deep_ep) |

- Selection: `select_kernel()` (`tokenspeed_kernel/selection.py`) ranks candidates from `KernelRegistry`/`KernelSpec` (`registry.py`), filtered by **`CapabilityRequirement.satisfied_by(platform)`** (`platform.py:254-280`).
- `CapabilityRequirement` (`platform.py:254`) = `min_arch_version` / `max_arch_version` / **`required_features`** (frozenset of strings) / `vendors`.
- `PlatformInfo` (`platform.py:101`) carries `arch_version`, `sm_features`, `interconnect` (detects `pcie` vs `nvlink_*`, `platform.py:508`).
- `SelectionObjective.PORTABILITY = "Prefer solutions that work across vendors (Triton)"` (`selection.py:90`) — a first-class knob to bias toward Triton.
- Registering a new backend = drop a file under `ops/<family>/` + `@register_kernel(capability=CapabilityRequirement(...))`. A `plugins/` dir exists as an extension point. `AGENTS.md`: *"Use Triton for portable solutions across vendors. Vendor libraries should stay optional."*

### 3.2 Level 2 — runtime attention backend registry
`runtime/layers/attention/registry.py:111` `register_backend(name, archs, cls)`; registry keyed by **attention algorithm** (`AttentionArch.MHA/MLA/DSA/MSA`), **not** GPU arch. Backends: `mha, mla, flashmla, tokenspeed_mla, trtllm, trtllm_mla, dsa, msa, deepseek_v4, hybrid_linear_attn, inkling`. GPU-arch selection happens *inside* each backend via the Level-1 kernel selection.

**Implication:** we do not fight the build system or rewrite CUTLASS. We make the capability layer route sm_120 to portable solutions, and only hand-fill families where the *only* fast path is tcgen05.

---

## 4. Gap analysis — what actually breaks on sm_120 today

### 4.1 Bug: sm_120 is mis-modeled as B200
`_get_cuda_sm_features(arch_version)` (`platform.py:382`) uses `>=` on the major version:
```
if arch_version >= ArchVersion(9, 0):  features |= {"memory:tma", "compute:cluster"}
if arch_version >= ArchVersion(10, 0): features |= {"tensor_core:f4"}
```
sm_120 has major=12 ≥ 10, so it is granted **`compute:cluster` and `tensor_core:f4` exactly like B200** — but sm_120 has **no clusters>1 and no tcgen05**. Also `is_blackwell` is `major == 10` (`platform.py:144`) so sm_120 is **not** `is_blackwell` but **is** `is_blackwell_plus` (`>= 10,0`, `platform.py:176`); `generation_name` returns unknown `"SM12.0"` for (12,0) (`platform.py:236`). NVIDIA sm_120 is simply unmodeled.

**✅ CONFIRMED on-device (M0a, 2026-07-24)** — `current_platform()` on the 4× RTX PRO 6000 box (standalone `platform.py`, no install) returns:
```
arch_version      : 12.0   device: NVIDIA RTX PRO 6000 Blackwell Server Edition  x4
generation_name   : SM12.0            # unmapped
is_blackwell      : False             # major==10 gate
is_blackwell_plus : True              # >=10.0 gate  ← the trap
interconnect      : pcie              # confirms no NVLink → TP + NCCL (Step D)
sm_features       : compute:cluster, memory:async_copy, memory:tma,
                    tensor_core:f16, tensor_core:f4, tensor_core:f8, tensor_core:int8
```
Refinement from the measurement: `tensor_core:f4` and `memory:tma` are **actually valid on sm_120** (FP4 via regular MMA; TMA exists since Hopper) — they should **not** be stripped. The single wrong flag is **`compute:cluster`**, plus the **absence of any feature that distinguishes tcgen05/tmem** (which sm_120 lacks) from regular Blackwell — so tcgen05 kernels have nothing to be gated on. This sharpens Step A below: drop `compute:cluster` for sm_120, keep f4/tma, and add a datacenter-only `compute:tcgen05` (+`memory:tmem`) flag to tag the tcgen05 kernels against.

### 4.1b The real blocker is selection *priority*, not just matching
Selection ranks candidates by `Priority` band (`registry.py`): **`PORTABLE=4 < PERFORMANT=8 < SPECIALIZED=12`**. So even when Triton *also* satisfies the capability filter, a mis-gated `SPECIALIZED` "Blackwell" kernel with unbounded `min_arch=(10,0)` **outranks Triton and wins** on sm_120 — then fails at runtime. This means fixing gating (Step B) is not optional polish; it is what makes the portable path *reachable at all*.

**Precedent to copy:** `ops/gemm/__init__.py:198-211` is the **only** correct sm_120-aware branch in the tree today — it special-cases `arch_version == ArchVersion(12, 0)` to route fp8-blockscale quantize to Triton. The sm_120 fixes should generalize this pattern.

### 4.2 Arch-gating audit — three buckets
- ✅ **Already correctly excludes sm_120** (safe): kernels bounded by `max_arch_version` or gated by `is_blackwell (major==10)`:
  - `ops/quantization/trtllm.py:72` `max_arch_version=ArchVersion(10, 9)` → sm_120 (12.0) excluded.
  - `ops/moe/flashinfer/trtllm_mxfp4.py:350-351` `10.0–10.3` → excluded.
  - `ops/attention/msa.py:64,96-97` gated by `is_blackwell` **and** `max_arch 10.3` → excluded.
  - `ops/attention/flashinfer/__init__.py:65` module-level `if platform.is_blackwell:` → the whole flashinfer-Blackwell attention block **does not register on sm_120** (falls back to triton).
- ⚠️ **Dangerously unbounded** (`min_arch=(10,0)`, no max, not `is_blackwell`-gated) → sm_120 wrongly matches, then crashes:
  - `ops/moe/flashinfer/cutlass_nvfp4.py:140-142` `min_arch_version=ArchVersion(10, 0)` (verify no max) — cutlass tcgen05 MoE GEMM.
  - Any other `min_arch=(10,0)`-only registrations (audit needed — pending full recon).
- 🟢 **Always-available portable fallback**: every family's `triton.py`. On sm_120 the selector should land here for the tcgen05 families.

### 4.3 The real question: does the Triton/portable path actually cover a full forward+decode?
Every family *has* a Triton solution, so functionally sm_120 should run by fallback. Unknowns to validate:
1. **MLA attention** — is there a non-tcgen05 MLA decode path (triton/flashinfer-sm120) that the `mla` backend selects, or is the only fast MLA the CuteDSL/`tokenspeed_mla` tcgen05 one? (Qwen3.6-35B-A3B is GQA/MoE, likely **MHA/GQA not MLA**, so MLA may not even be on our critical path — confirm the model's attention type.)
2. **MoE grouped-GEMM** — does the `moe/triton/` path run on sm_120, or is it Blackwell-gated? This is the hot path for Qwen3.6-35B-A3B.
3. **flashinfer sm_120 build** — flashinfer has been adding sm_120 kernels; which of tokenspeed's flashinfer calls have an sm_120 path vs are `is_blackwell`-gated away.

---

## 5. Design — sm_120 support strategy

Work **with** the capability framework, in four steps:

### Step A — model sm_120 correctly (`platform.py`)
- Give NVIDIA (12,0) a correct feature profile: keep `tensor_core:f4/f8`, `memory:async_copy`, `memory:tma`(sm_120 has TMA); **drop `compute:cluster`**; and introduce an sm_100-only feature flag, e.g. `compute:tcgen05` (+ `memory:tmem`), granted only for `arch == (10,x)` datacenter Blackwell, **not** by `>=`.
- Add `is_sm120` / `generation_name` mapping for (12,0) = "Blackwell (GB202)"; fix any code relying on `is_blackwell` that should mean "datacenter Blackwell (tcgen05)".

### Step B — tag the tcgen05 kernels so selection auto-rejects them on sm_120
- Add `required_features=frozenset({"compute:tcgen05"})` (or a bounded `max_arch_version=ArchVersion(10, 9)`) to every tcgen05/TMEM/cluster kernel that currently has an unbounded `min_arch=(10,0)` (start with `moe/flashinfer/cutlass_nvfp4.py`). After Step A, the capability filter drops them cleanly on sm_120.

### Step C — guarantee portable coverage per op on the critical model path
- For Qwen3.6-35B-A3B (dense-attn MoE), enumerate every op the forward+decode touches and confirm a **capability-satisfying non-tcgen05 candidate** registers and runs on sm_120: attention (GQA), MoE grouped-GEMM, dense GEMM, quant, rmsnorm, activation, sampling, kvcache.
- Where the only candidate is tcgen05 → author/enable a Triton (or flashinfer-sm120) solution for that op. Expected small set (likely MoE GEMM and possibly an MLA path we may not need).
- Build: add `12.0a` to `CUDA_ARCH_LIST` for the **portable** CUDA solutions that compile on sm_120 (non-tcgen05 `cuda.py` kernels); Triton solutions JIT automatically — no arch list needed.

### Step D — comm layer for PCIe / no-NVLink
- `platform.py` already detects `pcie`. For multi-GPU: **avoid DeepEP** (`ops/communication/deep_ep.py`, NVLink/IB). Use NCCL / Triton / flashinfer comm, or run **TP instead of EP** for MoE on 1–4 PCIe cards. For single-GPU benchmarking (Qwen3.6-35B-A3B ≈70GB bf16 fits one 96GB card), comm is moot for the first milestone.

---

## 6. Work breakdown & risk

| # | Item | Effort | Risk | Notes |
|---|---|---|---|---|
| A | sm_120 platform feature profile + `is_sm120` | S | Low | pure Python, `platform.py` |
| B | tag/bound tcgen05 kernels' capability | S | Low | mechanical once A lands; needs full audit of `min_arch=(10,0)`-only sites |
| C1 | validate Triton coverage for Qwen3.6 forward+decode | M | **Med** | the crux; may surface 1–3 missing ops |
| C2 | fill missing portable ops (likely MoE GEMM) | M–L | **Med–High** | writing/enabling a Triton MoE GEMM if none runs |
| D | pick PCIe-safe comm (TP over EP) | S–M | Low | single-GPU milestone avoids it |
| — | build with `12.0a` for portable cuda.py | S | Low–Med | some `cuda.py` may still include sm_100 intrinsics |

**Milestone 1 (functional, single GPU):** Qwen3.6-35B-A3B forward+decode on one sm_120 card via Triton fallback. Unlocks the tokenspeed-vs-vLLM benchmark.
**Milestone 2 (multi-GPU):** TP across 2–4 PCIe cards.
**Milestone 3 (perf):** replace hot Triton fallbacks with tuned sm_120 kernels.

**Biggest risk:** C2 — if a hot op (MoE grouped-GEMM) has *no* sm_120-capable solution and only a tcgen05 one, we must write a Triton/CUTLASS-sm120 kernel, which is real work. Mitigation: measure first (Milestone 1 may run, just slowly), optimize later.

---

## 7. Validation plan
1. Import `tokenspeed_kernel` on an sm_120 card; dump `current_platform()` — confirm the (buggy) feature set, then confirm the fixed set after Step A.
2. For each op family, call `explain_selection()` / `select_kernel()` on sm_120 and record which solution is chosen and whether it's tcgen05.
3. Bring up Qwen3.6-35B-A3B offline (`Engine`/`bench`) single-GPU; capture any ptxas/runtime arch errors and map each to a missing portable op.
4. Once forward+decode is green, run `tokenspeed bench serve` vs `vllm bench serve` with the **same client, same model, same sm_120 box**.

---

## 8. Open questions — RESOLVED (2026-07-24, checked on the box)
- **Qwen3.6-35B-A3B attention type:** it is `Qwen3_5MoeForConditionalGeneration` (`model_type: qwen3_5_moe`) — a **hybrid GDN** model (linear attention + full attention interleaved), natively supported by tokenspeed (`runtime/models/qwen3_5_moe.py`, listed in `_HYBRID_GDN_ARCHITECTURES`, attention `registry.py:119`). The GDN linear-attention kernels live in `ops/attention/triton/linear/` — **Triton, portable, sm_120-OK**. Not MLA → the tcgen05 MLA gap is NOT on our path.
- **Checkpoint:** bf16, no `quantization_config` → no fp4/fp8 tcgen05 quant path forced; ~70GB fits one 96GB card for Milestone 1.
- **Interconnect:** `nvidia-smi topo -m` shows pure PCIe (PIX pairs GPU0-1 / GPU2-3, no NVLink) → Step D confirmed: TP + NCCL, no DeepEP.
- **smg gateway availability:** `tokenspeed-smg 1.8.0` and `smg-grpc-servicer 0.7.0` ARE on PyPI → upstream serving path is installable; benchmarking does NOT block on building our own frontend.

---

## 9. Bring-up log

### M0a — platform mis-detection (2026-07-24) ✅
Confirmed on-device (see §4.1): `current_platform()` reports sm_120 as `is_blackwell_plus=True` and advertises `compute:cluster` (wrong). `interconnect=pcie`.

### M0b — native build + kernel-selection audit (2026-07-24) ✅
**Environment (reproducible):** the org's prebuilt Blackwell dev image
`nctu6/unieinfra-autobuild-dev:cuda-12.9.1-blackwell_only-4ad1404` — nvcc 12.9
(supports `sm_120`), torch 2.11.0+cu129, 4× RTX PRO 6000. Container:
`docker run -d --name unieinfra-dev --gpus all --ipc=host -v /workspace:/workspace -w /workspace/roy/UnieInfra`.

**Build.** Editable install of the vendored kernel package **succeeds for
`sm_120a`** — confirming the hypothesis that the 17 hand-written native kernel
groups are Blackwell-generic (`__CUDA_ARCH__>=900/1000`), **not** tcgen05, so
they compile for sm_120. tcgen05 code is runtime JIT/DSL only.
```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TOKENSPEED_CUDA_ARCH=120a
export CPATH=$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/include | tr '\n' ':')$CPATH        # cuBLAS headers ship via pip wheels, not the toolkit
export LIBRARY_PATH=$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/lib | tr '\n' ':')$LIBRARY_PATH
pip install -e tokenspeed-kernel/python --no-build-isolation
```
Environment gaps hit and worked around (candidates for a pinned dev image / a
lazy-import source fix):
1. Missing `tokenspeed-proton` → `pip install tokenspeed-proton`.
2. flashinfer version-check (`jit-cache 0.6.12` vs `python 0.6.15`) → `FLASHINFER_DISABLE_VERSION_CHECK=1`.
3. cuBLAS headers absent from the toolkit include dir → add pip `nvidia/*/include` to `CPATH`.
4. HEAD pins pull **cu13** kernel wheels (`tokenspeed-deepgemm`, `tokenspeed-trtllm-kernel`) whose `_C` needs `libcudart.so.13`/`libcublas.so.13` on a cu12.9 base → put a complete cu13 lib dir on `LD_LIBRARY_PATH`. **Proper fix (our fork):** make the datacenter-kernel imports in `ops/*/__init__.py` lazy/guarded so sm_120 never eagerly loads tcgen05/cu13 wheels it won't call.

**`import tokenspeed_kernel` succeeds** with those env vars.

**Selection audit (27 operators registered, on sm_120):**
- **23 portable winners** — triton / flashinfer / cuda / reference.
- **3 datacenter (deep_gemm) winners** — all **DSA** (`dsa_decode_topk`, `dsa_plan`, `dsa_prefill_topk`). DSA = DeepSeek/GLM sparse attention, **not on the Qwen3.x path**. These are exactly the mis-gated `deep_gemm` (`min_arch=(9,0)`, no max) kernels Step B must cap.
- **1 no-candidate** — `quantization/mxfp4` (fp4 models only; not bf16 Qwen3.6).
- **Caveat (needs runtime confirmation in M1):** several winners are `flashinfer`/`flashinfer_trtllm` (e.g. `moe/apply`, `gdn_decode_*`, `mha_decode`). A static audit can't tell whether the pinned flashinfer 0.6.15 has working **sm_120** kernels for these — M1 must exercise them.

**Verdict:** for Qwen3.6-35B-A3B (GDN + MoE, bf16) the selected attention path is
portable (triton/flashinfer); no DSA/mxfp4 op is on its path. Proceed to **M1**:
install engine + scheduler, run a forward+decode, and confirm the `flashinfer*`
winners actually execute on sm_120.

### M1 — end-to-end forward+decode on sm_120 (2026-07-24) ✅ (dense) / ⚠️ (MoE gap)
Installed all three packages in the dev container (kernel already built for
sm_120a; `tokenspeed-scheduler` needs `scikit-build-core`+`libssl-dev`;
`python/` engine installed `--no-deps` since `tokenspeed-mooncake` is
unpublished — P/D-only, not needed offline). Ran `scripts/m1_forward_smoke.py`
on a **free** GPU (the box is shared — GPU0 was held by another user's vLLM).

**✅ Dense path proven — DENSE Qwen3-8B generates correct output on sm_120:**
```
prompt : "The capital of France is"
output : " Paris. The capital of Italy is Rome. The capital of Spain is Madrid.
          The capital of Germany is Berlin. ..."   # semantically correct
engine up 46.6s · generate 32 tok / 1.90s (enforce_eager, no CUDA graph)
```
Config that worked: `enforce_eager=True`, `disable_custom_all_reduce=True`,
`attention_backend="triton"`. This validates the whole sm_120 stack end-to-end:
C++ FSM scheduler + model executor + Triton attention + sampling + native
kernels all run on RTX PRO 6000.

**⚠️ First real sm_120 kernel gap — bf16 *unquantized* MoE (blocks Qwen3.6/MoE):**
`moe.apply` for `weight_dtype=unquant, dtype=bf16` has **no working sm_120
kernel**:
- `flashinfer` cutlass MoE (`ops/moe/flashinfer/cutlass_unquant.py:99`) — the M0b
  `moe/apply` winner — fails at an FFI `init` (expects 7 args, HEAD passes 8:
  flashinfer-python 0.6.15 caller vs flashinfer-jit-cache 0.6.12 kernels — the
  version skew the `FLASHINFER_DISABLE_VERSION_CHECK` bypass papered over).
- `moe_backend="triton"` → `NoKernelFoundError`: the Triton MoE solutions cover
  only **quantized** weights (fp8/mxfp4), not unquant bf16.
Fix options (next): (a) align `flashinfer-jit-cache` to 0.6.15 so the cutlass MoE
init matches, and check its sm_120 coverage; (b) add a portable **bf16 unquant
Triton MoE** solution — a clean first target for the agent-driven kernel work.

### Product note (raised 2026-07-24)
The engine should **auto-detect single- vs multi-GPU topology** and configure
parallelism/comm itself (vLLM-style) — logged as a UnieInfra frontend/config
responsibility in `unieinfra-architecture.md`. (Here `world_size=1` was set
explicitly for bring-up.)

---

## Appendix — future kernel-toolchain options (not for now)
- **Modular Mojo / MAX** (`github.com/modular/modular`):
  - *What it is vs Triton:* both target portable high-perf kernels across NVIDIA+AMD. **Triton** = a Python-embedded **tile-level DSL**, narrow/mature, already wired into every tokenspeed op family (the sanctioned portable path per `AGENTS.md`). **Mojo** = a full MLIR-based **systems language** with finer low-level control than Triton (warp specialization, custom scheduling, novel dtypes) + a large hand-optimized **MAX AI kernels** library. Similar *goal*, different *level* — Mojo can reach closer-to-SOTA than generic Triton on hard ops, at the cost of a heavier toolchain.
  - *License:* **Apache-2.0 with LLVM exceptions** (open) — legally integrable.
  - *Integration mechanism:* Mojo supports **PyTorch custom ops**, so a Mojo kernel can be wrapped as a `ops/<family>/mojo.py` solution + `@register_kernel(capability=...)` — fits the tokenspeed registry at the API level. Cost: adds the Mojo/MAX toolchain to build+runtime and Mojo kernels to maintain; we'd want only the kernel language, not the MAX *runtime* (which is a competing engine).
  - *Cross-vendor fit:* Modular explicitly targets NVIDIA + AMD (26.2 added AMD RDNA consumer GPUs; CDNA datacenter too) — aligns with UnieInfra's cross-AMD goal, arguably better than CuTe-DSL (NVIDIA-only).
  - *Caveat:* "SOTA" is per-op/per-model (e.g. FLUX). Must benchmark a Mojo kernel vs Triton on **our** hot ops (Qwen MoE grouped-GEMM, GQA decode) before committing.
  - **Verdict: not needed for sm_120 bring-up (Triton covers functional). Strong candidate for Milestone 3 perf on hot ops where Triton underperforms — and it doubles as the cross-AMD peak-perf path.**
- **TileRT as black-box decode backend**: closed B200 binary — cannot run on sm_120. Off the table for this hardware.
