# 60 — 量化路徑五方橫向對照

版本見 `00-METHOD.md`。所有行號只對該表的 pinned commit 有效。

三條路徑：**fp8 KV cache**、**w4a16 權重量化**、**NVFP4 MoE**。
本機是 sm_120（RTX PRO 6000 Blackwell），所以每一節都問同一個問題：
**在這張卡上，這條路實際走得到嗎。**

---

## 1. fp8 KV cache

| # | 步驟 | UnieInfer | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 1 | 開關 | `runtime/utils/server_args.py:848`（`auto` / `fp8` / `fp8_e4m3` / `mxfp8`） | `--kv-cache-dtype` | `--kv-cache-dtype` | executor KV config | 權重轉換期決定 |
| 2 | 別名正規化 | `runtime/utils/server_args.py:329`（`fp8` → `fp8_e4m3`） | `config/cache.py` | `srt/layers/quantization/kv_cache.py` | C++ | — |
| 3 | scale 管理 | `runtime/layers/quantization/fp8.py` | `model_executor/layers/quantization/kv_cache.py` | `srt/layers/quantization/kv_cache.py` | `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` | `src/turbomind/kernels/attention/quantization.h` |
| 4 | KV 寫入（量化） | attention backend 內 | attention backend 內 | attention backend 內 | kernel 內 | kernel 內 |
| 5 | KV 讀取（dequant） | **Triton attention kernel 內就地 dequant**：`ops/attention/triton/mha_decode.py:35`、`ops/attention/triton/mha_prefill.py:33` | FlashAttention / FlashInfer kernel 內 | 同 | kernel 內 | 同 3 |
| 6 | fp4 KV | — | — | `srt/layers/quantization/fp4_kv_cache_quant_method.py`、`kvfp4_tensor.py` | — | — |

**分歧點：只有 SGLang 有 fp4 KV。** 其他四家最低到 fp8。

**sm_120 上的可達性**：UnieInfer 的 fp8 KV 走 Triton attention，Triton 對 sm_120 沒有
arch 閘，所以可用（既有量測：容量 1.92x、吞吐 +19.0%）。這條路能通是因為
**探針問的是 Triton 而不是 FlashAttention** —— FlashAttention 的 fp8 KV 在 sm_120 上
沒有預編譯 kernel。

---

## 2. w4a16（權重 4-bit、啟動 16-bit）

| # | 步驟 | UnieInfer | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 1 | checkpoint 格式 | `compressed-tensors` wNa16 | AWQ / GPTQ / compressed-tensors | AWQ / GPTQ / compressed-tensors | AWQ / modelopt | AWQ / GPTQ：`turbomind/weight_format.py:200 AWQFormat` |
| 2 | scheme 類別 | `runtime/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py` | `model_executor/layers/quantization/auto_awq.py`、`auto_gptq.py` | `srt/layers/quantization/awq/`、`gptq/` | `_torch/modules/linear.py` | `src/turbomind/kernels/gemm/` |
| 3 | 權重重排 | `ops/quantization/cuda.py` 的 `gptq_marlin_repack` | Marlin repack | `srt/layers/quantization/marlin_utils.py` | modelopt | `src/turbomind/kernels/gemm/convert_v3.cu`、`unpack.cu` |
| 4 | GEMM kernel | **無** | Marlin | Marlin | trtllm-gen | **自有**：`src/turbomind/kernels/gemm/gemm.cu` |
| 5 | MoE 版本 | `.../compressed_tensors/gptq_marlin_moe.py` | `moe_wna16.py` | `srt/layers/moe/` | `_torch/modules/fused_moe/` | `src/turbomind/models/llama/moe_ffn_layer.cc` |

### 2.1 UnieInfer 的 w4a16 是死碼（已重新驗證）

`compressed_tensors_wNa16.py` 引用了 **10 個在整個檔案與其 import 中都不存在的名字**，
以 AST 檢查（載入所有 import 與區域繫結後）得到：

```
MarlinLinearLayerConfig      apply_gptq_marlin_linear     check_marlin_supports_shape
marlin_is_k_full             marlin_make_empty_g_idx      marlin_make_workspace
marlin_permute_scales        marlin_repeat_scales_on_all_ranks
marlin_sort_g_idx            marlin_zero_points
```

唯一有實作的是 `gptq_marlin_repack`（`runtime/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py:32` 從
`tokenspeed_kernel.ops.quantization.cuda` import），也就是**只有權重重排，沒有 GEMM**。
這個檔案在 `create_weights` 就會 `NameError`。

→ **「開啟 w4a16」在這個引擎不是設定問題，是缺一個 Marlin GEMM。** TurboMind 有自己的
u4 GEMM（`src/turbomind/kernels/gemm/`，含 `cast.cu` / `unpack.cu` / `convert_v3.cu`），
是可移植的參考實作 —— 但那是一個 port，不是一個開關。

### 2.2 TurboMind 的 GEMM 沒有 sm_120 config

`src/turbomind/kernels/gemm/arch/` 只有 `config_simt.h`、`config_sm70_s884.h`、
`config_sm75_s16816.h`、`config_sm80_s16816.h`；sm90 的變體在
`gemm_universal_sm90*.h`。**沒有 sm_120 專用 config**。也就是說即使移植過來，
在這張卡上會落到 sm80 的 s16816 MMA 路徑。

這與 GDN 恰好相反：TurboMind **有**手寫 sm_120 GDN kernel（`30-prefill.md` 第 0 節），
但**沒有** sm_120 GEMM config。不能因為一個結論就推另一個。

---

## 3. NVFP4 MoE

| # | 步驟 | UnieInfer | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 1 | 量化設定 | `runtime/layers/quantization/nvfp4.py:33 Nvfp4Config` | `model_executor/layers/quantization/modelopt.py` | `srt/layers/quantization/modelopt_quant.py`、`fp4_utils.py` | `_torch/modules/fused_moe/` | — |
| 2 | MoE kernel（cutlass 路徑） | `ops/moe/flashinfer/cutlass_nvfp4.py:137 flashinfer_cutlass_nvfp4_moe_apply` | flashinfer / cutlass | flashinfer / cutlass | trtllm-gen | — |
| 3 | MoE kernel（cute-dsl + DeepEP） | `ops/moe/flashinfer/cutedsl_deepep_nvfp4.py:110` | — | — | — | — |
| 4 | arch 閘 | `ops/moe/flashinfer/cutlass_nvfp4.py:142 min_arch_version=ArchVersion(10, 0)` | modelopt 內的能力檢查 | 同 | C++ | — |

### 3.1 UnieInfer 的 arch 閘是開區間（與 vLLM 的 GDN 閘相反）

`tokenspeed_kernel` 的能力比對是 `platform.arch_version >= self.min_arch_version`
（`tokenspeed-kernel/python/tokenspeed_kernel/platform.py:269`），`min_arch_version` 是
`ArchVersion(10, 0)`（`ops/moe/flashinfer/cutlass_nvfp4.py:142`）。sm_120 →
`(12,0) >= (10,0)` → **通過**。

對照 `30-prefill.md` 第 0 節：vLLM 對 GDN 用的是封閉的 `family(100)`，sm_120 被排除。
**同一個問題，UnieInfer 的 kernel registry 用的是正確的開區間形式** —— 這是這一輪
少數 UnieInfer 結構優於參照引擎的地方，值得記下來別改壞。

（既有量測：NVFP4 MoE 在 35B-A3B 上 +36.4%，但那是 *fallback* backend 的數字；
sm_12x 的 cutlass kernel 卡在 CUDA 13，是工具鏈問題不是硬體問題。）

---

## 4. 可量測候選（下一輪的輸入）

| 候選 | 量什麼 | 為什麼值得 |
|---|---|---|
| F1 fp8 KV 三家對照 | UnieInfer / vLLM / SGLang 開 fp8 KV 的容量與吞吐 | UnieInfer 已有 +19.0% / 1.92x，缺對照組 |
| F2 SGLang fp4 KV | 能不能在 sm_120 跑起來、容量與品質 | 唯一有 fp4 KV 的引擎；若可行是 2 倍於 fp8 的容量 |
| F3 w4a16 缺口 | 先確認 TurboMind GEMM port 的工作量，再談量測 | 現在無從量起（死碼）；這是實作題不是量測題 |
| F4 NVFP4 MoE backend | fallback vs cutlass（CUDA 13 環境）的差距 | +36.4% 是 fallback 的數字，真正的 kernel 還沒量過 |

---

## 5. 未解格（`?`）

一格：**TRT-LLM 的 w4a16 與 NVFP4 在 sm_120 上實際選到哪個 kernel**。
trtllm-gen 的選擇邏輯在 autotuner 與 C++ 之間，沒有單一判斷點可引用。與
`30-prefill.md` 第 7 節同一個原因，補法也一樣：實跑一次看 log。
