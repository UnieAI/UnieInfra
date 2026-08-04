<!-- English canonical. Chinese: support-matrix.zh.md (keep in sync). -->
# UnieInfra Support Matrix

**Honest status.** We distinguish what has been **validated on real hardware** from
what is merely **code-supported** (a code path exists / upstream builds for it, but
*we* have not run it there). Do not read "code-supported" as "it works" — it means
"it should, untested." Every ✅ links to reproducible evidence.

**Legend**
- ✅ **Validated on-device** — we actually ran it on this exact hardware; see `docs/experiments-log.md`.
- 🟡 **Code-supported (unvalidated by us)** — the path exists / upstream targets it, but we have not run it on this hardware.
- 🔨 **In progress** — known gap being worked, with the specific blocker noted.
- ❌ **Not supported / blocked.**

> As of 2026-07-25, our on-device validation covers **Qwen3-8B (dense)**, **Qwen3.6-35B-A3B (GDN + bf16 MoE)** and **Qwen3.5-9B / 27B (GDN dense, incl. MTP speculative decoding)** on 1× RTX PRO 6000 (sm_120). Datacenter NVIDIA and AMD are code-supported by upstream but **not yet validated by us** (no access exercised). We will not label anything ✅ until we run it.

---

## Hardware (GPU)

| GPU | Arch | Status | Notes / evidence |
|---|---|---|---|
| **RTX PRO 6000** | Blackwell sm_120 | ✅ **Validated** | dense Qwen3-8B correct + benchmarked (exp E4, E15). Native kernels build for `120a` (E2). Attention = Triton. |
| RTX 5090 | Blackwell sm_120 | 🟡 Code-supported | same silicon as PRO 6000; not separately run. |
| H100 / H200 | Hopper sm_90 | 🟡 Code-supported | upstream builds `sm_90a`; **not validated by us**. |
| B200 / B300 | Blackwell datacenter sm_100/103 | 🟡 Code-supported | upstream's primary target (tcgen05 kernels); **not validated by us**. |
| A6000 | Ampere sm_86 | 🟡 Code-supported | platform detects it; no fp8/tcgen05 → Triton-path only; **untested**. |
| L40 / RTX 4090 | Ada sm_89 | 🟡 Code-supported | detected; untested. |
| MI300 / MI350 | AMD CDNA3/CDNA4 | 🟡 Code-supported | `tokenspeed-kernel-amd` (Triton/Gluon) exists; **not validated by us**. |
| Ascend / Qualcomm | — | ❌ Not supported | not CUDA/ROCm; a future per-vendor backend program. |

---

## Models — with explicit sm_120 (our validated hardware) status

Grouped by attention type, because that determines what runs on sm_120 today.

| Attention | Model families | Code-supported (any HW) | On **sm_120** (our box) |
|---|---|---|---|
| **GQA / MHA** (dense) | **Qwen3-8B** | ✅ | ✅ **Validated** (correct output + benchmarked) |
| GQA / MHA | Qwen2, Qwen3, Llama, GPT-OSS, Mixtral, Inkling | ✅ | 🟡 same dense path as Qwen3-8B, but **each model not individually run** |
| **GDN** (linear hybrid) + **bf16 MoE** | **Qwen3.6-35B-A3B (qwen3_5_moe)** | ✅ | ✅ **Validated** (E28) — correct output eager *and* CUDA graph; batch-1 149 tok/s. **Caveat: keep concurrency ≤ ~32** — GDN state is ~58 MB/seq so the pool holds only ~114 seqs, but adaptive-batch admits from KV pages alone and over-admits (C=64 hangs, C=128 device-side assert) |
| **GDN** (linear hybrid, dense) | **Qwen3.5-9B**, **Qwen3.5-27B** | ✅ | ✅ **Validated** (E32) — correct output under CUDA graph with `--attention-backend triton`; 9B 82.9 tok/s and 27B 26.5 tok/s at C=1 (27B ≈ 80% of its bf16 memory roofline). **Same concurrency caveat as Qwen3.6, and one step further:** E32 landed the adaptive-batch state-pool clamp that stops the over-admission, but a **second** GDN bug remains (`hybrid_linear_attn.py:425` OOB write on the prefill path) that still caps this family at **C ≤ 16** on our box |
| GDN (linear hybrid) | LongCat-Flash | ✅ | 🟡 same GDN path as Qwen3.5/3.6, but **not individually run** |
| **MLA** | DeepSeek V3/V4, Kimi K2.5, MiniCPM3 | ✅ | 🔨 MLA attn HAS a Triton path (untested on sm_120); DSA-topk gap for V3.2/V4; flagship sizes need TP across ≥4 cards |
| **DSA** (sparse) | GLM-5 | ✅ | 🔨 DSA topk uses `deep_gemm` (datacenter) — needs a Triton/ported sm_120 kernel (refs: SGLang NSA, contest `dsa-topk-indexer`) |
| **MSA** (MiniMax sparse) | MiniMax M2/M3 | ✅ | 🔨 MSA sparse kernels are sm_100 CuTe — needs an sm_120 path |
| Multimodal / audio | Qwen3-Vision/Audio/Omni/ASR | ✅ | status follows the underlying attention type |
| GQA + per-layer head_dim | **Gemma 4** (dense `gemma-4-31B-it`) | 🟡 `gemma.py` added — text decoder, structural tests pass; loads real weights | 🟡 **E27's triton head_dim-512 blocker is fixed by E28** (the shared-mem over-budget 114688 > 101376 B was the generic sm_120 tile bug, not Gemma-specific — `mha_extend_with_kvcache[triton-d512]` now passes at kernel level). **Full model still unrun**; flashinfer JIT nvcc/cu13 header incompat remains (triton is the default backend, so this is not blocking), and E27's latent per-layer-head_dim KV-pool question is open. MoE `gemma-4-26B-A4B` = follow-up |

---

## Speculative decoding — measured on sm_120

| Algorithm | Draft source | Code-supported | On **sm_120** (our box) |
|---|---|---|---|
| **MTP** (chain, topk=1) | the **base checkpoint's own** `mtp.*` weights (no separate draft model; `--speculative-algorithm MTP` alone) | ✅ | ✅ **Validated** (E32) on Qwen3.5-9B: **+131% output throughput at C=1, +62% at C=16**, TPOT −59% / −34%. Acceptance 3.72/4 (C=1) and 3.58/4 (C=16) on the benchmark corpus, **3.37/4 on realistic prose/code**. **Requires** `--attention-backend triton` **and** `--sampling-backend greedy` (the flashinfer sampler allocates `[bs×k, vocab]` fp32 inside the CUDA graph → 23–30 GB graph pools → OOM on a 96 GB card). Costs **5× GDN state per request** on linear-attention models. Greedy output-equivalence vs no-spec: **7/8 prompts token-identical**, 1 near-tie divergence — not yet declared exactly output-preserving |
| EAGLE3 / DFLASH | separate draft checkpoint | ✅ | 🟡 drafters exist (`execution/drafter/`) and share the MTP verify path, but **not run by us** |
| Tree spec (`--speculative-eagle-topk > 1`) | — | ❌ | ❌ rejected at startup — only chain spec (topk=1) is wired end-to-end (`server_args.py:615`) |

---

## Weight precision (dense linear path) — measured on sm_120

Same client, seed, warmup and flush for every row (`benchmark/run_sweep.sh`,
ILEN 1024 / OLEN 256, GPU 2). Qwen3-8B.

| Precision | Checkpoint | Status on **sm_120** | C=1 | C=32 | C=128 |
|---|---|---|---|---|---|
| **bf16** | `Qwen/Qwen3-8B` | ✅ Validated (E15) | 77.4 | 1415 | 2731 |
| **fp8** (online weight quant) | bf16 checkpoint, `--quantization fp8` | ✅ Validated (E23) | 109.7 | 2085 | 3519 |
| **nvfp4** | `nvidia/Qwen3-8B-NVFP4` (ModelOpt) | ✅ **Validated (E30)** | **109.1** | **2442** | **4341** |
| mxfp4 | — | ❌ no NVIDIA mxfp4 GEMM registered (`triton_mm_mxfp4` is `vendors={"amd"}`) | — | — | — |

nvfp4 is the **fastest configuration measured on sm_120 to date** (C=128 +59% over
bf16, +23% over fp8) and cuts dense weights ~16 GB → ~6 GB. Caveats: greedy output
is coherent but **no accuracy benchmark has been run**; the checkpoint's FP8 KV
scales are ignored (KV stays bf16), so the saving is weights-only; and nvfp4
requires a pre-quantized ModelOpt checkpoint — there is **no online nvfp4
quantization path** (unlike fp8).

---

**Honest bottom line on sm_120:** **Qwen3-8B (dense)**, **Qwen3.6-35B-A3B (GDN + bf16 MoE)** and **Qwen3.5-9B / 27B (GDN dense, incl. MTP speculative decoding)** are validated end-to-end today — the GDN family at low concurrency only, with the mamba-pool admission gap noted above. The GQA family is high-confidence-but-unrun; MLA/DSA/MSA have known, specific gaps being worked (not fundamental walls — vLLM/SGLang run these on sm_120 and are our references and the bar to beat).

---

*Update this matrix whenever an item is validated or a gap closes; every ✅ must
point to an experiment in `docs/experiments-log.md`. Keep the Chinese mirror
(`support-matrix.zh.md`) in sync.*
