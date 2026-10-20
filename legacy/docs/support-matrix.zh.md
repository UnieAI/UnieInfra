<!-- 中文版。英文正本：support-matrix.md（兩者需同步）。 -->
# UnieInfra 支援矩陣

**誠實原則。** 我們嚴格區分「**實機驗證過**」和「**程式理論支援**」（有 code path／上游有編，但**我們沒在該硬體上跑過**）。「程式支援」**不等於**「能跑」，只是「理論上應該、未測」。每個 ✅ 都連到可複現的證據。

**圖例**
- ✅ **實機驗證** — 我們真的在這個硬體上跑過；見 `docs/experiments-log.md`。
- 🟡 **程式支援（我們未驗證）** — 有 code path／上游有目標,但我們沒在此硬體跑過。
- 🔨 **攻克中** — 已知缺口，附具體 blocker。
- ❌ **不支援 / 受阻。**

> 截至 2026-07-25,我們**唯一實機驗證過的是 Qwen3-8B(dense)在 1× RTX PRO 6000(sm_120)**。資料中心 NVIDIA 和 AMD 上游有支援,但**我們尚未實機驗證**。沒跑過的一律不標 ✅。

---

## 硬體（GPU）

| GPU | 架構 | 狀態 | 備註 / 證據 |
|---|---|---|---|
| **RTX PRO 6000** | Blackwell sm_120 | ✅ **已驗證** | dense Qwen3-8B 正確 + 已 benchmark(實驗 E4、E15)。原生 kernel 編 `120a`(E2)。attention=Triton。 |
| RTX 5090 | Blackwell sm_120 | 🟡 程式支援 | 與 PRO 6000 同矽晶;未單獨跑。 |
| H100 / H200 | Hopper sm_90 | 🟡 程式支援 | 上游編 `sm_90a`;**我們未驗證**。 |
| B200 / B300 | 資料中心 Blackwell sm_100/103 | 🟡 程式支援 | 上游主要目標(tcgen05 kernel);**我們未驗證**。 |
| A6000 | Ampere sm_86 | 🟡 程式支援 | platform 認得;無 fp8/tcgen05 → 只走 Triton;**未測**。 |
| L40 / RTX 4090 | Ada sm_89 | 🟡 程式支援 | 偵測到;未測。 |
| MI300 / MI350 | AMD CDNA3/CDNA4 | 🟡 程式支援 | 有 `tokenspeed-kernel-amd`(Triton/Gluon);**我們未驗證**。 |
| Ascend / Qualcomm | — | ❌ 不支援 | 非 CUDA/ROCm;未來的逐廠商後端計畫。 |

---

## 模型 — 附明確的 sm_120（我們驗證過的硬體）狀態

依注意力型態分組（這決定了 sm_120 今天能跑什麼）。

| 注意力 | 模型家族 | 程式支援(任一硬體) | 在 **sm_120**(我們的機器) |
|---|---|---|---|
| **GQA / MHA**(dense) | **Qwen3-8B** | ✅ | ✅ **已驗證**(正確輸出 + 已 benchmark) |
| GQA / MHA | Qwen2、Qwen3、Llama、GPT-OSS、Mixtral、Inkling | ✅ | 🟡 與 Qwen3-8B 同 dense 路徑,但**各模型未逐一跑** |
| **GDN**(線性混合)+ **bf16 MoE** | **Qwen3.6-35B-A3B(qwen3_5_moe)** | ✅ | ✅ **已驗證**(E28)— eager 與 CUDA graph 皆輸出正確;batch-1 149 tok/s。**注意:併發請保持 ≤ ~32** — GDN 狀態約 58 MB/序列,狀態池僅容納約 114 條序列,但 adaptive-batch 只依 KV page 計算准入而超額收單(C=64 掛住、C=128 device-side assert) |
| **GDN**(線性混合,dense) | **Qwen3.5-9B**、**Qwen3.5-27B** | ✅ | ✅ **已驗證**(E32)— 搭配 `--attention-backend triton` 在 CUDA graph 下輸出正確;C=1 時 9B 82.9 tok/s、27B 26.5 tok/s(27B 約為其 bf16 記憶體 roofline 的 80%)。**與 Qwen3.6 相同的併發限制,並再往前一步:** E32 已修好 adaptive-batch 的狀態池 clamp,阻止超額收單,但仍有**第二個** GDN 缺陷(`hybrid_linear_attn.py:425` prefill 路徑的越界寫入),使本家族在我們機器上仍卡在 **C ≤ 16** |
| GDN(線性混合) | LongCat-Flash | ✅ | 🟡 與 Qwen3.5/3.6 同一條 GDN 路徑,但**未逐一跑** |
| **MLA** | DeepSeek V3/V4、Kimi K2.5、MiniCPM3 | ✅ | 🔨 MLA attn **有 Triton 路徑**(sm_120 未測);V3.2/V4 有 DSA-topk 缺口;旗艦尺寸需 TP 跨 ≥4 卡 |
| **DSA**(稀疏) | GLM-5 | ✅ | 🔨 DSA topk 走 `deep_gemm`(datacenter)— 需 Triton/移植的 sm_120 kernel(範本:SGLang NSA、contest `dsa-topk-indexer`) |
| **MSA**(MiniMax 稀疏) | MiniMax M2/M3 | ✅ | 🔨 MSA 稀疏 kernel 是 sm_100 CuTe — 需 sm_120 路徑 |
| 多模態 / 語音 | Qwen3-Vision/Audio/Omni/ASR | ✅ | 依底層注意力型態而定 |
| GQA + 逐层 head_dim | **Gemma 4**(稠密 `gemma-4-31B-it`) | 🟡 已新增 `gemma.py` —— 文本解码器，结构性测试通过；可加载真实权重 | 🟡 **E27 的 triton head_dim-512 阻塞已由 E28 修復**(共享內存超限 114688 > 101376 B 其實是 sm_120 的通用 tile 缺陷,並非 Gemma 專屬 —— `mha_extend_with_kvcache[triton-d512]` 現已在 kernel 層通過)。**整模型仍未跑**;flashinfer JIT nvcc/cu13 頭文件不相容依舊存在(但 triton 是預設後端,故不構成阻塞),E27 提出的逐層 head_dim KV pool 潛在問題仍未解。MoE `gemma-4-26B-A4B` 为后续工作 |

---

## 推測解碼（speculative decoding）— sm_120 實測

| 演算法 | Draft 來源 | 程式碼支援 | 在 **sm_120**(我們的機器) |
|---|---|---|---|
| **MTP**(chain,topk=1) | **base checkpoint 自帶的** `mtp.*` 權重(不需另外的 draft model;只要 `--speculative-algorithm MTP`) | ✅ | ✅ **已驗證**(E32),Qwen3.5-9B:**C=1 輸出吞吐 +131%、C=16 +62%**,TPOT 分別 −59% / −34%。接受長度 C=1 為 3.72/4、C=16 為 3.58/4(benchmark 語料),**真實散文/程式碼提示為 3.37/4**。**必須**搭配 `--attention-backend triton` **與** `--sampling-backend greedy`(flashinfer sampler 會在 CUDA graph 內配置 `[bs×k, vocab]` fp32 → graph pool 23–30 GB → 96 GB 卡上 OOM)。在線性注意力模型上會佔用 **5 倍的 GDN 狀態**。與非推測 greedy 的輸出等價性:**8 題中 7 題 token 完全一致**,1 題在近似平手處分歧 —— 尚未宣告完全保序 |
| EAGLE3 / DFLASH | 獨立 draft checkpoint | ✅ | 🟡 drafter 已存在(`execution/drafter/`)且與 MTP 共用 verify 路徑,但**我們未跑過** |
| Tree spec(`--speculative-eagle-topk > 1`) | — | ❌ | ❌ 啟動即拒絕 —— 端到端只接了 chain spec(topk=1)(`server_args.py:615`) |

---

## 權重精度（dense linear 路徑）— sm_120 實測

每一列都用同一套 client、seed、warmup 與 flush（`benchmark/run_sweep.sh`，
ILEN 1024 / OLEN 256，GPU 2）。模型為 Qwen3-8B。

| 精度 | 權重檔 | **sm_120** 狀態 | C=1 | C=32 | C=128 |
|---|---|---|---|---|---|
| **bf16** | `Qwen/Qwen3-8B` | ✅ 已驗證（E15） | 77.4 | 1415 | 2731 |
| **fp8**（線上權重量化） | bf16 權重 + `--quantization fp8` | ✅ 已驗證（E23） | 109.7 | 2085 | 3519 |
| **nvfp4** | `nvidia/Qwen3-8B-NVFP4`（ModelOpt） | ✅ **已驗證（E30）** | **109.1** | **2442** | **4341** |
| mxfp4 | — | ❌ NVIDIA 沒有註冊 mxfp4 GEMM（`triton_mm_mxfp4` 是 `vendors={"amd"}`） | — | — | — |

nvfp4 是目前 sm_120 上**實測最快的組態**（C=128 比 bf16 +59%、比 fp8 +23%），
且把 dense 權重從約 16 GB 壓到約 6 GB。但要誠實標註三點：greedy 輸出連貫，
但**尚未跑任何精度評測**；權重檔內的 FP8 KV scale 目前被略過（KV 仍是 bf16），
所以省的只有權重；以及 nvfp4 **必須用預先量化好的 ModelOpt 權重檔** —— 目前
**沒有線上 nvfp4 量化路徑**（fp8 有）。

---

**sm_120 誠實結論:** 今天真正端到端驗證過的是 **Qwen3-8B(dense)**、**Qwen3.6-35B-A3B(GDN + bf16 MoE)** 與 **Qwen3.5-9B / 27B(GDN dense,含 MTP 推測解碼)** —— GDN 家族目前皆僅限低併發,並帶有上述 GDN 狀態池准入缺口。GQA 家族「高信心但未跑」;MLA/DSA/MSA 有已知的具體缺口正在攻克(**不是物理死牆 —— vLLM/SGLang 在 sm_120 都跑這些,是我們的範本兼要超越的靶子**)。

---

*每當一項通過驗證或補完缺口就更新此矩陣;每個 ✅ 必須指向 `docs/experiments-log.md` 的一個實驗。與英文正本 `support-matrix.md` 保持同步。*
