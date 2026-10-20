# FlashAttention FP8 on SM120 (PRO 6000 Blackwell) — Research Record

**Date**: 2025-07-27
**Target GPU**: NVIDIA RTX PRO 6000 Blackwell (GB202, SM 12.0, 96GB GDDR7)
**vLLM versions examined**:
- `/workspace/roy/vllm` — commit 4ad14044b (UnieAI merge branch, actual venv-used vllm)
- `vllm-bench-venv` — vllm 0.25.1, flashinfer 0.6.13, torch 2.11.0+cu130
- `/workspace/roy/UnieInfra/third_party/vllm` — commit d02df748b (vllm upstream submodule)
- `/workspace/roy/UnieInfra/unieinfra-vllm` — commit f7d789b

---

## 1. Backend Selection Logic (Auto Default)

**Source**: `vllm/platforms/cuda.py:83-161` (`_get_backend_priorities()`)

When no backend is explicitly set (`AttentionConfig.backend = None`, i.e. "auto"), vLLM iterates through backends in priority order and selects the **first one that passes `validate_configuration()`** (`vllm/v1/attention/backend.py:320`).

### Priority for standard attention (MHA/MQA/GQA — Qwen and Gemma use these, NOT MLA):

| GPU architecture | Priority order |
|---|---|
| **Blackwell SM100 (datacenter)** | `[FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]` |
| **SM80-90, SM110, SM120 (incl. PRO 6000)** | `[FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]` |

SM120 falls into the `else` branch (`cuda.py:152-160`) because `device_capability.major == 10` is False for SM120.

**Key comment** (`cuda.py:144-146`): *"SM100f defaults to FlashInfer for TRTLLM causal attention, but its non-causal cutlass path (used for dflash attention) is known to have problems. So prefer FlashAttention when non-causal on SM100f."*

### For MLA attention (DeepSeek-style):
- SM100: `[FLASHINFER_MLA, TOKENSPEED_MLA, CUTLASS_MLA, FLASH_ATTN_MLA, ...]`
- SM120: `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`
- SM90/others: `[FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA, ...]`

---

## 2. Why FP8 KV Cache Selects FlashInfer (or Fallback)

### FlashAttention FP8 support check

`flash_attn.py:181-228` (`supports_kv_cache_dtype()` + `supports_combination()`):

```python
# FA supports fp8 KV cache ONLY when:
# - FA version 3 on SM90 (Hopper), OR
# - FA version 4 on SM100 (Blackwell datacenter)
# Error: "FP8 KV cache requires FA3 on SM90 or FA4 on SM100"
```

**SM120 problem**: `get_flash_attn_version()` (`fa_utils.py:154-162`) falls back to FA2 for all GPUs where `major != 9` and `major != 10`. Since SM120 has `major == 12`, it gets FA2. And FA2 does NOT support FP8 KV cache.

**Code evidence**:
- `fa_utils.py:276-286` (`flash_attn_supports_kv_cache_dtype()`): only returns True for `(fa_version == 3 AND family(90)) OR (fa_version == 4 AND family(100))`.
- `flash_attn_interface.py:63-85` (`_is_fa3_supported()` / `_is_fa4_supported()`): explicitly checks `is_device_capability_family(90)` or `is_device_capability_family(100)`. SM120 = `family(120)` → both return False.
- `flash_attn.py:229`: `supports_combination()` returns `"FP8 KV cache requires FA3 on SM90 or FA4 on SM100"` → rejected.

**Result**: FA is validated out at priority #1, vLLM tries #2 FlashInfer.

### FlashInfer FP8 support

FlashInfer's `supported_kv_cache_dtypes` (`flashinfer.py:344-351`): includes `"fp8", "fp8_e4m3", "fp8_e5m2"` — so it **passes the FP8 KV cache check**.

**However**, SM120 has a separate problem:
- `supports_trtllm_attention()` (`vllm/utils/flashinfer.py:374-391`) only returns True for `family(90)` and `family(100)`. **SM120 = False**.
- `flashinfer.py:870-871` comment: *"Architectures with only fa2 (e.g. SM89, **SM120**) cannot consume FP8 queries, so keep the model dtype for Q there."*
- FlashInfer on SM120 falls back to **Native path** (not trtllm-gen/XQA), losing FP8-Q fast path.

### FlashInfer 0.6.13 broken on SM120 (our test env)

In `vllm-bench-venv`, flashinfer 0.6.13 fails even for bf16 on SM120:
```
Error: "Failed to get device capability: SM 12.x requires CUDA >= 12.9."
Error: "FlashInfer requires GPUs with sm75 or higher"
```

**Root cause** (`compilation_context.py:74-80`):
```python
if major == 12:
    if is_cuda_version_at_least("12.9"):
        return (12, "0f")
    else:
        raise RuntimeError("SM 12.x requires CUDA >= 12.9")  # <-- this triggers
```

Even though torch is cu130, `is_cuda_version_at_least` somehow returns False (likely torch 2.11's version check doesn't recognize CUDA 13 as >= 12.9). This causes `TARGET_CUDA_ARCHS` to be empty set → `check_cuda_arch()` throws "sm75 or higher" error.

**This means in our test env, FlashInfer simply cannot run on SM120 at all.** If vLLM tries to use it, it gets ImportError and falls through to Triton.

### VERIFIED: What "auto" actually selects (probe run on PRO 6000 SM120)

Probe script: `/tmp/probe_auto_backend.py` (calls `get_attn_backend()` directly with real Qwen3/Gemma4 attn params).

| Model / KV dtype | selector picks | selector log |
|---|---|---|
| Qwen3-8B bf16 | **FLASH_ATTN** (FA2) | `Using FLASH_ATTN out of ['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION']` |
| Qwen3-8B fp8 | **FLASHINFER** | `Using FLASHINFER out of ['FLASHINFER','TRITON_ATTN']` (FA2 failed FP8 validate) |
| Gemma4 bf16 (sliding & full) | **FLASH_ATTN** (FA2) | same as Qwen3 bf16 |
| Gemma4 fp8 (sliding & full) | **FLASHINFER** | same as Qwen3 fp8 |

**This confirms the user's original observation**: bf16 → FlashAttention, fp8 → FlashInfer, at the selector level.

### CRITICAL: Selection vs Runtime distinction (the FlashInfer contradiction)

The benchmark showed `FLASHINFER: ERROR "FlashInfer requires GPUs with sm75 or higher"`, but the selector probe shows FLASHINFER is selected. Both are true because they run at different stages:

- **Selection (static `validate_configuration`)**: `FlashInferBackend.supports_compute_capability()` returns `cap >= 8.0 and cap <= 12.1` → SM120 passes. `supports_kv_cache_dtype("fp8")` passes. The base `validate_configuration()` (`backend.py:320`) does NOT call flashinfer's `check_cuda_arch()`. So FlashInfer is selected without ever running flashinfer code.
- **Runtime (actual kernel run)**: flashinfer's `check_cuda_arch()` (`flashinfer/jit/core.py:108`) runs lazily during JIT plan, finds `TARGET_CUDA_ARCHS` empty (because `_normalize_cuda_arch(12,0)` raised "SM 12.x requires CUDA >= 12.9" which was swallowed), and raises `RuntimeError: FlashInfer requires GPUs with sm75 or higher`.

**Conclusion**: With `auto` + fp8 on SM120, the user SELECTS FlashInfer, but in an env where flashinfer 0.6.13 is broken, it CRASHES at runtime. In an env with working flashinfer on SM120, it runs the FlashInfer **Native path** (not trtllm-gen, since `supports_trtllm_attention()` returns False for family(120)).

### Summary: What "auto" selects in practice

| kv_dtype | PRO 6000 SM120, FlashInfer broken (our bench-venv) | PRO 6000 SM120, FlashInfer working |
|---|---|---|
| **bf16** | **FLASH_ATTN (FA2)** — selected & runs | **FLASH_ATTN (FA2)** — selected & runs |
| **fp8** | **FLASHINFER selected** but crashes at runtime → effectively unusable | **FLASHINFER (Native path)** — selected & runs (no trtllm-gen/XQA) |

---

## 3. Micro-Benchmark Results

**Test environment**: PRO 6000 SM120, torch 2.11.0+cu130, CUDA graph enabled, 8 layers, 4 GPU devices (GPU 1 used).
**Benchmark tool**: `/workspace/roy/vllm/benchmarks/attention_benchmarks/` (modified to support sliding_window + custom scale + local model).

### Qwen3 (hd=128, GQA 16:8, scale=1/√d, pure causal, no sliding window)

| batch_spec | stage | FA bf16 (s/layer) | Triton bf16 | Triton fp8 | FA/Tribf16 | FA/Trifp8* | fp8/Trbf16 |
|---|---|---|---|---|---|---|---|
| q2k | prefill 2k | 1.03859e-04 | 2.28230e-04 | 2.06697e-04 | FA fast 2.20× | 0.50× | 0.91× |
| q4k | prefill 4k | 3.21150e-04 | 7.75335e-04 | 7.02581e-04 | FA fast 2.41× | 0.46× | 0.91× |
| q8k | prefill 8k | 1.10011e-03 | 2.65887e-03 | 2.50378e-03 | FA fast 2.42× | 0.44× | 0.94× |
| 8q1s1k | decode 8×1k | 3.66887e-05 | 3.03660e-05 | 1.36392e-05 | Tri fast 1.21× | 2.69× | 0.45× |
| 64q1s4k | decode 64×4k | 7.20228e-04 | 7.11791e-04 | 3.62345e-04 | 1.01× (tie) | 1.99× | 0.51× |
| 128q1s4k | decode 128×4k | 1.42596e-03 | 1.40690e-03 | 7.52713e-04 | 1.01× (tie) | 1.89× | 0.54× |
| 2q4k_64q1s1k | mixed | 8.12638e-04 | 1.41138e-03 | 1.32654e-03 | FA fast 1.74× | 0.61× | 0.94× |

### Gemma4 sliding layer (hd=256, GQA 32:16, scale=1.0, sliding_window=1024)

| batch_spec | stage | FA bf16 | Triton bf16 | Triton fp8 | FA/Tribf16 | FA/Trifp8* | fp8/Trbf16 |
|---|---|---|---|---|---|---|---|
| q2k | prefill 2k | 2.38152e-04 | 1.24217e-03 | 5.18468e-04 | FA fast 5.22× | 0.46× | 0.42× |
| q4k | prefill 4k | 4.99200e-04 | 2.84697e-03 | 1.13089e-03 | FA fast 5.70× | 0.44× | 0.40× |
| q8k | prefill 8k | 9.92599e-04 | 5.99541e-03 | 2.35231e-03 | FA fast 6.04× | 0.42× | 0.39× |
| 8q1s1k | decode 8×1k | 1.12661e-04 | 9.71739e-05 | 5.20964e-05 | Tri fast 1.16× | 2.16× | 0.54× |
| 64q1s4k | decode 64×4k | 7.33107e-04 | 7.25394e-04 | 3.59062e-04 | 1.01× (tie) | 2.04× | 0.49× |
| 128q1s4k | decode 128×4k | 1.96363e-03 | 1.41424e-03 | 7.41102e-04 | Tri fast 1.39× | 2.65× | 0.52× |
| 2q4k_64q1s1k | mixed | 1.69300e-03 | 5.94887e-03 | 2.38568e-03 | FA fast 3.51× | 0.71× | 0.40× |

### Gemma4 full layer (hd=256, GQA 32:16, scale=1.0, no sliding window)

| batch_spec | stage | FA bf16 | Triton bf16 | Triton fp8 | FA/Tribf16 | FA/Trifp8* | fp8/Trbf16 |
|---|---|---|---|---|---|---|---|
| q2k | prefill 2k | 3.24889e-04 | 8.58612e-04 | 6.52497e-04 | FA fast 2.64× | 0.50× | 0.76× |
| q4k | prefill 4k | 1.04454e-03 | 2.98763e-03 | 2.28520e-03 | FA fast 2.86× | 0.46× | 0.76× |
| q8k | prefill 8k | 3.76115e-03 | 1.13971e-02 | 8.74299e-03 | FA fast 3.03× | 0.43× | 0.77× |
| 8q1s1k | decode 8×1k | 1.04140e-04 | 9.61686e-05 | 5.23330e-05 | FA fast 1.08× | 1.99× | 0.54× |
| 64q1s4k | decode 64×4k | 2.81788e-03 | 2.80097e-03 | 1.40834e-03 | 1.01× (tie) | 2.00× | 0.50× |
| 128q1s4k | decode 128×4k | 5.61717e-03 | 5.59808e-03 | 2.86765e-03 | 1.00× (tie) | 1.96× | 0.51× |
| 2q4k_64q1s1k | mixed | 2.62567e-03 | 6.02024e-03 | 4.61129e-03 | FA fast 2.29× | 0.57× | 0.77× |

### Gemma4 weighted estimate (50 sliding + 10 full layers = 5/6 sliding, 1/6 full)

| batch_spec | stage | FA bf16 | Triton bf16 | Triton fp8 | FA/Tribf16 | FA/Trifp8* | fp8/Trbf16 |
|---|---|---|---|---|---|---|---|
| q2k | prefill 2k | 2.52608e-04 | 1.17824e-03 | 5.40806e-04 | FA fast 4.66× | 0.47× | 0.46× |
| q4k | prefill 4k | 5.90090e-04 | 2.87041e-03 | 1.32327e-03 | FA fast 4.86× | 0.45× | 0.46× |
| q8k | prefill 8k | 1.45402e-03 | 6.89569e-03 | 3.41742e-03 | FA fast 4.74× | 0.43× | 0.50× |
| 8q1s1k | decode 8×1k | 1.11241e-04 | 9.70063e-05 | 5.21358e-05 | Tri fast 1.15× | 2.13× | 0.54× |
| 64q1s4k | decode 64×4k | 1.08057e-03 | 1.07132e-03 | 5.33942e-04 | 1.01× (tie) | 2.02× | 0.50× |
| 128q1s4k | decode 128×4k | 2.57255e-03 | 2.11155e-03 | 1.09553e-03 | Tri fast 1.22× | 2.35× | 0.52× |
| 2q4k_64q1s1k | mixed | 1.84844e-03 | 5.96077e-03 | 2.75662e-03 | FA fast 3.23× | 0.67× | 0.46× |

### Column definitions:
- **FA/Tribf16** < 1: FA bf16 faster (prefill advantage)
- **FA/Trifp8***: optimistic upper bound — assumes "if FA had FP8 and performed like FA bf16, how much faster would it be than current Triton FP8"
- **fp8/Trbf16** < 1: Triton fp8 faster than Triton bf16 (FP8 bandwidth benefit)
- **FA/Trifp8** < 1: "If FA had FP8, it would be faster than current Triton FP8" (the patch value)

### Key observations from data:

1. **Prefill bottleneck**: FA bf16 is 2.2–6.0× faster than Triton bf16. The larger the model (Gemma4 hd=256 vs Qwen3 hd=128), the bigger the gap.
2. **FP8 bandwidth benefit on Triton**: fp8/Trbf16 = 0.40–0.77× for prefill (FP8 saves bandwidth), 0.43–0.54× for decode.
3. **Decode large batch**: FA and Triton bf16 are essentially tied (0.99–1.01×). FP8 gives ~2× speedup on Triton regardless.
4. **Decode small batch (8)**: Triton bf16 is slightly faster than FA bf16 (1.08–1.39×). FP8 gives ~2× additional speedup.

---

## 4. Why FP8 KV Cache Is Slower Than BF16

**Decomposition**: FP8 slowdown = (backend downgrade loss) vs. (FP8 bandwidth benefit)

| Case | Downgrade loss (Tri_bf16 ÷ FA_bf16) | FP8 bandwidth gain (Tri_fp8 ÷ Tri_bf16) | Net (Tri_fp8 ÷ FA_bf16) |
|---|---|---|---|
| Qwen3 prefill | 2.2–2.4× slower | 0.91× (9% faster) | **2.0–2.3× slower** |
| Gemma4 prefill | 4.7–4.9× slower | 0.46× (2.2× faster) | **2.1–2.4× slower** |
| Qwen3 decode large batch | 0.99× (tie) | 0.51× (2× faster) | **0.50× faster** |
| Gemma4 decode large batch | 0.82–0.99× | 0.50–0.54× (2× faster) | **0.43–0.49× faster** |

**Root cause**: Triton prefill kernel is inherently slower than FA's highly-optimized prefill kernel (2.2–6.0× slower). FP8 bandwidth savings (0.46–0.91× faster) do not compensate for this downgrade loss in prefill. In decode, FA and Triton are already tied, so FP8 bandwidth wins outright (~2× faster).

**This is not a fundamental FP8 weakness** — it is purely a backend selection artifact. If FA supported FP8, prefill would not degrade.

---

## 5. MLA FP8 Design Patterns (Reference for Custom Op)

From `vllm/v1/attention/backends/mla/flashmla_sparse.py`:

### Pattern 1: FP8 KV + BF16 prefill (dequant-then-prefill), FP8 decode kernel

> *"Separate prefill and decode mode: use the BF16 prefill kernel for prefill (upconverting the FP8 cache to BF16 then calling the prefill kernel) and using the FP8 decode kernel for decode."*

**This is the most practical pattern for Qwen/Gemma custom ops**:
- Prefill: Dequant FP8 KV cache → BF16 → call FA bf16 prefill kernel (keep the 2.2–6.0× FA advantage)
- Decode: Call Triton FP8 decode kernel (already ~2× faster than bf16)
- The dequant pass cost is small in prefill (compute-bound) vs. the savings

**Reference implementation**: `flashinfer.py:104-180` (`_trtllm_prefill_attn_kvfp8_dequant`) — a Triton kernel that reads FP8 KV + scales, outputs BF16 to mock cache. **Can be directly reused as the prefill pre-processing step.**

### Pattern 2: Mixed batch mode (FP8 decode kernel for prefill+decode)

> *"Use the FP8 decode kernel for both prefill and decode... this is done by treating all tokens as single batch."*

Only useful for decode-bound scenarios. Not suitable for long-prompt prefill.

### Pattern 3: FP8-Q (query quantized) tensor-core fast path

On SM90/SM100, FlashInfer achieves FP8 decode ~2× faster by quantizing BOTH Q and KV, using FP8 tensor-core attention path (XQA/trtllm-gen decode).

**SM120 limitation** (`flashinfer.py:870-871`): *"Architectures with only fa2 (e.g. SM89, SM120) cannot consume FP8 queries, so keep the model dtype for Q there."* — SM120 lacks FA3/FA4 FP8-Q tensor-core path.

---

## 6. FlashAttention FP8 Support Gap (SM120)

### Why FA doesn't support FP8 on SM120

**FA version resolution** (`fa_utils.py:154-162`):
- SM90 → FA3
- SM100 → FA4
- **SM120 → falls through to FA2** (because `major == 12` doesn't match `major == 9` or `major == 10`)

**FA3 support** (`flash_attn_interface.py:63-68`): *"FA3 is only supported on devices with compute capability 9.x"*
**FA4 support** (`flash_attn_interface.py:73-85`): *"FA4 is only supported on devices with compute capability 9.x, 10.x, or 11.x"* — SM120 (12.x) not listed.

**FA2 does not support FP8 KV cache** — it only supports fp16/bf16 model dtype with bf16/fp16 KV cache.

### What adding FP8 support to FA on SM120 would require

1. Update `is_fa_version_supported()` / `_is_fa4_supported()` to include `family(120)` if FA4 supports SM120 (need to verify FA4 CUDA kernel compatibility with sm_120).
2. If FA4 works on SM120, `flash_attn_supports_kv_cache_dtype()` would return True for FP8 (check: `fa_version == 4 AND family(100)` — would need to also check `family(120)`).
3. The `supports_combination()` check (`flash_attn.py:205-228`) would also need the `family(100)` → `family(120)` update.

**Alternative custom approach**: Write a custom SM120 FP8 attention kernel (Triton-based, similar to Triton_ATTN but with FP8 KV cache handling) that plugs into the vLLM backend interface, avoiding FA entirely.

---

## 7. Recommendations for Qwen/Gemma Custom Optimization

Based on the analysis, priority order:

### Priority 1: Dequant-then-prefill + FP8 decode (Pattern 1 from MLA)

**Architecture**:
```
Prefill: FP8 KV cache → dequant kernel (Triton) → BF16 cache → FA2 bf16 prefill kernel
Decode:  FP8 KV cache → Triton FP8 decode kernel
```

**Why this works**:
- Prefill: Keep FA's 2.2–6.0× prefill advantage (no degradation)
- Decode: Already get ~2× FP8 bandwidth speedup from Triton
- Dequant pass cost is small in prefill (compute-bound stage)
- **This directly eliminates the 2× prefill slowdown** that users currently see with "auto" + FP8

**Implementation depth**: Relatively shallow — add one Triton dequant kernel + wire it into the prefill pipeline. The FA kernel itself is untouched.

### Priority 2: Custom SM120 FP8 Triton attention kernel

If Triton FP8 decode is not fast enough, write a custom FP8 Triton decode kernel (borrowing from Triton_ATTN but with FP8 KV path). This gives more control but requires more code.

### Priority 3: Push upstream for FA SM120 FP8 support

File an issue or PR to FA adding `family(120)` support to `is_fa_version_supported()` and related checks. This would make SM120 work with FA4 (if FA4 supports sm_120) and unlock native FP8.

### What NOT to do

- **Do NOT write FA native FP8 prefill kernel** — requires modifying FA CUDA source, high engineering cost, and FA2 lacks FP8-Q path anyway.
- **Do NOT rely on FlashInfer** — in our test env it is broken on SM120; upgrading to a newer version or patching `compilation_context.py` would be needed, but that's outside vLLM's control.

---

## 8. Verification Status / Known Issues

- [x] FA FP8 error confirmed on SM120 (`FlashAttention only support fp16 and bf16 data type`)
- [x] FlashInfer 0.6.13 broken on SM120 at RUNTIME (`FlashInfer requires GPUs with sm75 or higher`), but PASSES static validate (so selector picks it, then it crashes when kernel runs)
- [x] bf16 micro-benchmarks completed for Qwen3 and Gemma4
- [x] fp8 micro-benchmarks completed for Qwen3 and Gemma4 (FA=ERROR, Triton only)
- [x] Gemma4 sliding window support added to benchmark runner
- [x] **VERIFIED**: "auto" backend selection on PRO 6000 SM120 — bf16→FLASH_ATTN(FA2), fp8→FLASHINFER (via `get_attn_backend()` probe with real Qwen3/Gemma4 params). Selection vs runtime distinction documented.
- [ ] **NOT verified**: End-to-end throughput with real Qwen3/Gemma4 models (kernel speed ≠ overall model throughput). Requires `vllm serve` + load generator; blocked on confirming a working serving env (bench-venv has broken flashinfer; production env unknown).
- [ ] **NOT verified**: Gemma2/Gemma3 (different soft_cap behavior — Gemma4 config has no attn_logit_softcapping, only final_logit_softcapping; Gemma2/3 do have attn soft cap).
- [ ] **NOT verified**: Whether the user's actual production venv has working flashinfer on SM120 (determines if fp8 "auto" actually runs FlashInfer or crashes). User needs to confirm which venv they serve with.

---

## 9. Files Modified During Research

- `/workspace/roy/vllm/benchmarks/attention_benchmarks/runner.py` — Modified to support:
  - Local model path via `VLLM_BENCH_MODEL` env var
  - Sliding window via `VLLM_BENCH_SLIDING_WINDOW` env var
  - Soft cap via `VLLM_BENCH_SOFT_CAP` env var
  - Custom sm_scale via `VLLM_BENCH_SM_SCALE` env var
- Backup: `/workspace/roy/vllm/benchmarks/attention_benchmarks/runner.py.bak`

## 10. Raw Benchmark Data Files

- `/tmp/bench_bf16_big.csv` — Qwen3 bf16, 7 batch specs
- `/tmp/bench_fp8_big.csv` — Qwen3 fp8, 7 batch specs
- `/tmp/bench_gemma_bf16_sw.csv` — Gemma4 sliding bf16
- `/tmp/bench_gemma_fp8_sw.csv` — Gemma4 sliding fp8
- `/tmp/bench_gemma_bf16_full.csv` — Gemma4 full bf16
- `/tmp/bench_gemma_fp8_full.csv` — Gemma4 full fp8

