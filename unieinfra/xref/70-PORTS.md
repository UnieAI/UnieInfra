# 70 — 功能效率 code review 與 1:1 移植計畫

版本見 `00-METHOD.md`。證據底稿見 `90-DIVERGENCES.md` 與各階段文件。

## 0. 原則

TokenSpeed 是早期專案。既定策略是：

1. **第一步不自己做。** 把別人的功能程式碼 1:1 整合、跑通，先追平別人的效果。
   先擴大用途，再談優化。
2. **第二步才在同等效果上再優化。**
3. **每一個策略都保留。** 移植進來的實作**不取代**既有實作，而是**多註冊一個
   solution**；哪一個生效交給選擇層決定。

第 3 點決定了這份文件的形狀：每一項移植都要回答「它註冊成哪個 solution」，而不是
「它取代什麼」。排序依據是**「1:1 可移植性 × 缺口大小」**，不是預期效能收益 ——
一個能力缺口（別人有、我們沒有）排在一個效率落差（兩邊都有、我們慢）前面。

每一項的效率評估都是**讀碼**得出的靜態指標（launch 數、同步點、演算法形狀、
記憶體流量），不是量出來的 —— 量測是移植完成後用來確認追平，不是用來決定做不做。

---

## 1. 策略保留與選擇：現有機制

### 1.1 kernel 層 — 已經完整，直接沿用

`tokenspeed_kernel` 的選擇層已經是「保留所有策略 + 選最佳」的模型：

| 元件 | 位置 | 作用 |
|---|---|---|
| 註冊 | `tokenspeed-kernel/python/tokenspeed_kernel/registry.py:375 register_kernel` | 同一個 `(family, mode)` 可註冊任意多個 kernel，各帶 `name` 與 `solution` |
| 硬體閘 | `tokenspeed-kernel/python/tokenspeed_kernel/platform.py:255 CapabilityRequirement` | `min_arch_version` **與** `max_arch_version`，不滿足者在排序前就被濾掉 |
| 優先權帶 | `tokenspeed-kernel/python/tokenspeed_kernel/registry.py:75 Priority` | `PORTABLE 4` / `PERFORMANT` / `SPECIALIZED 12` / `PLUGIN 16` |
| 選擇目標 | `tokenspeed-kernel/python/tokenspeed_kernel/selection.py:88` | `LATENCY` / `THROUGHPUT` / `PORTABILITY` / `DETERMINISM` / `DEBUG` |
| 選擇策略 | `tokenspeed-kernel/python/tokenspeed_kernel/selection.py:95 SelectionStrategy` | `HEURISTIC`（score 排序，即時）／**`AUTOTUNE`（實跑候選挑最快）** |
| 排序 | `tokenspeed-kernel/python/tokenspeed_kernel/selection.py:100 ScoreBreakdown` | 字典序 `(oracle, objective, priority)`；`SelectionOracle` 是 per-family 可插鉤子 |
| 入口 | `tokenspeed-kernel/python/tokenspeed_kernel/selection.py:616 select_kernel` | 有 cache；`solution=` 限定血緣、`override=` 強制指名 |
| 外部覆寫 | `tokenspeed-kernel/python/tokenspeed_kernel/selection.py:261 load_config_overrides` | YAML（`TOKENSPEED_KERNEL_OVERRIDES_FILE`）→ context manager → 環境變數 `TOKENSPEED_KERNEL_OVERRIDE_<FAMILY>_<MODE>`，優先權由低到高 |

**現有 solution 命名慣例是「實作血緣」而非「來源引擎」**：`triton`、`gluon`、`fa3`、
`fa4`、`flashinfer`、`flashinfer_trtllm`、`flashinfer_cutlass`、`deep_gemm`、`cuda`、
`torch`、`flashmla`、`msa`、`trtllm`、`flashkda`。新移植沿用這個慣例。

### 1.2 runtime backend 層 — 也有，但較簡單

`runtime/layers/attention/registry.py:116 register_backend`（name + 適用 arch 集合）、
`runtime/sampling/registry.py`；server args 以 `attention_backend` / `moe_backend` /
`sampling_backend` / `kda_backend` 等選擇，預設 `auto`
（`runtime/utils/server_args.py:206` 起）。**沒有 objective、沒有 autotune、沒有優先權帶。**

### 1.3 C++ scheduler 層 — **沒有這一層**

搶佔受害者選擇、混批順序、chunk 切分現在都是 `SchedulerConfig` 上的布林或常數，
不是可註冊的 policy：

- 混批：`csrc/scheduler/operations/forward.cpp:1145` 讀 `config_.enable_mixed_prefill_decode`
- 搶佔：`csrc/scheduler/operations/forward.cpp:1277` 硬寫「挑 `TokenSize()` 最大者」
- chunk：切法隱含在 `csrc/scheduler/operations/forward.cpp:374` / `:557` 裡

要把「每個策略都保留」延伸到這裡，需要一個對等的註冊/選擇介面。
**這是整份清單裡唯一真正需要設計的部分**，其餘都是機械搬運 ——
所以它是 openspec change proposal 的主題，見 `openspec/`。

### 1.4 移植的落地位置（已存在，不必新建）

`AGENTS.md` 已經規定：

> All direct third-party code should be placed in `thirdparty/` and imported into
> `ops/` then registered via `register_kernel`.

| 層 | 位置 | 現有先例 |
|---|---|---|
| 第三方原始碼原樣放置 | `tokenspeed_kernel/thirdparty/<source>/` | `thirdparty/{trtllm,deep_gemm,msa,cutedsl_kda,cute_dsl,triton,cuda}` |
| CUDA/C++ | `thirdparty/cuda/csrc/`，編進 `objs/<name>/<name>.so`，TVM-FFI 匯出 | `csrc/gptq_marlin_repack.cu` → `objs/marlin/marlin.so` |
| 對外 op 包裝 | `ops/<family>/<solution>.py` | `ops/quantization/cuda.py` |
| runtime 層 | `runtime/layers/...` 直接抄對應檔案 | `runtime/layers/quantization/compressed_tensors/schemes/` 已是 SGLang 的複本 |

**授權**：tokenspeed 是 MIT，vLLM / SGLang / TRT-LLM / LMDeploy **四家全是 Apache-2.0**。
Apache-2.0 併入 MIT 專案沒問題，條件是保留授權與出處標示 —— repo 已有
`ACKNOWLEDGEMENTS.md` 與各套件的 `THIRDPARTYNOTICES`，既有檔案用
`SPDX-License-Identifier: MIT AND Apache-2.0` 雙標。**照既有格式做就是合規的。**

---

## 2. 逐功能 code review

「1:1 度」欄：**A** = 幾乎是複製檔案；**B** = 複製 + 改名接管線；**C** = 要改寫。
「註冊成」欄 = 這一項移植進來之後在選擇層裡叫什麼；`—` 表示該層沒有選擇機制（見 §1.3）。

### 2.1 能力缺口（別人有、我們沒有 — 優先）

| # | 功能 | 我們現況（code review） | 來源 | 1:1 度 | 註冊成 | 規模 |
|---|---|---|---|---|---|---|
| **T1** | **w4a16 GEMM** | `compressed_tensors_wNa16.py` 是 SGLang 同名檔的複本，**只刪掉 `marlin_utils` 的 import 區塊**，並加 `# ruff: noqa: F821` 壓掉隨之而來的 lint。10 個名字全未定義，`create_weights` 必 `NameError`。`thirdparty/cuda/csrc/tokenspeed_marlin.cuh` 檔頭自己寫著「Minimal Marlin utilities **needed by gptq_marlin_repack**」—— 只有重排，沒有 GEMM | Python：SGLang `srt/layers/quantization/marlin_utils.py`（1007 行）<br>CUDA：vLLM `csrc/libtorch_stable/quantization/marlin/`（8936 行，ABI-stable，與已 vendored 的 repack 同血緣） | **A** / **B** | `("quantization","gemm")` solution **`marlin`** | 1k 行 Py + 9k 行 CUDA |
| **T2** | ngram drafter | 沒有。`runtime/execution/drafter/` 只有 `mtp` / `eagle` / `dflash` | vLLM `v1/spec_decode/ngram_proposer.py`（純 Python，最小）；之後可加 `ngram_proposer_gpu.py` | **B** | drafter 名 **`ngram`**（`runtime/execution/drafter/base.py:39` 的介面） | ~300 行 |
| **T3** | 樹狀 draft | 只有線性 draft | TRT-LLM `_torch/speculative/spec_tree_manager.py` + `dynamic_tree_ops.py` | **C** | drafter 選項 **`tree`** | 中 |
| **T4** | fp4 KV cache | 最低到 fp8 | SGLang `srt/layers/quantization/fp4_kv_cache_quant_method.py` + `kvfp4_tensor.py` | **B** | `kv_cache_dtype` 值 **`fp4`** + attention solution | 小 |
| **T5** | chunk 切分 policy | 隱式 FCFS | TRT-LLM `cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp:299`（三種特化） | **C** | **—**（需 §1.3 的介面） | 小 |
| **T6** | 串流 stop 前綴抑制 | 沒有；產生就送 | SGLang `srt/managers/schedule_batch.py:1438 check_match_stop_str_prefix` + `srt/managers/scheduler_components/output_streamer.py:384` | **A** | 行為旗標（無需 solution） | ~30 行 |

### 2.2 效率落差（兩邊都有、我們慢）

| # | 功能 | 我們現況（靜態效率指標） | 來源 | 為什麼它快 | 1:1 度 | 註冊成 |
|---|---|---|---|---|---|---|
| **T7** | **GDN prefill** | **9 個 Triton launch/層**，48 層 = 432 launch/forward，全在 CUDA graph 外：`l2norm(q)`、`l2norm(k)`、`cumsum`、`kkt`、`solve_tril`×2、`recompute_w_u`、`fwd_h`、`fwd_o`（`ops/attention/triton/linear/chunk.py:95` 起）。**無 device sync**（backend 內無 `.item()` / `.cpu()`），這點是好的 | **TurboMind `src/turbomind/kernels/linear_attn/kernel/sm_120/`** —— 唯一在 sm_120 上跑得動的融合版 | cumsum+prepare、kkt+solve_tril、h+o 各融成一個 → **3 launch**；TMA + 一般 `tiled_mma`，不碰 tcgen05 | **C** | `("attention","gdn_chunk_prefill")` solution **`turbomind`**，與現有 `triton` 並存 |
| **T8** | GDN 前置融合 | in_proj 兩次 GEMM 再拆；conv1d 與 SiLU 分開；q/k 兩次 `l2norm_fwd` | TurboMind `src/turbomind/models/llama/GatedDeltaNetLayer.cc:375` / `:393 invokeFusedConv1dSiLU` / `:421 invokeL2NormalizeQK` | 每層省 3 個 launch，少兩趟中間張量往返 HBM | **C** | 併入 T7 的 `turbomind` solution |
| **T9** | spec verify 的 GDN | `runtime/layers/attention/backends/hybrid_linear_attn.py:2176` 判 `q_len_per_req > 1` 就轉呼叫 `forward_extend` → 每個 verify step 付 432 launch，純 decode 只要 96 | SGLang `sglang/kernels/ops/attention/fla/gdn_replayssm_spec_decode.py`、`gdn_replayssm_spec_fold.py`，另有獨立的 verify backend 選項 | verify 的 q_len 通常 ≤ 8，recurrent 展開比走 chunked prefill 便宜 | **B** | 新 mode `("attention","gdn_verify")`，solution **`replayssm`** |
| **T10** | hybrid 前綴匹配 | `HybridPrefixCache::Match`（`csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp:117`）先做純 KV 匹配，再 `augmentMatch`（`:132`）從 KV 終端**往上**找 mamba 快照；找不到就 `last_node = root`（`:163`）**整個 KV 匹配丟掉** | SGLang `srt/mem_cache/mamba_radix_cache.py:1068 _match_prefix_helper`：下降中維護 `best_last_node`，回傳帶快照的最深節點 | 一次走訪得到兩個答案；更深的快照不會消失 | **C** | **—**（需 §1.3 的介面） |
| **T11** | 引擎端重複 detokenize | `enable_inline_detokenizer` 預設 `True`（`runtime/utils/server_args.py:111`），每個 output token 一次 HF decode；但 gRPC 只回 token id（`smg_grpc_servicer/tokenspeed/servicer.py:1460` / `:1483`），smg gateway 自己再 detokenize 一次 | 不需移植 —— 這是**關掉**不是新增 | **A** | 既有旗標，改的是預設 |
| **T12** | draft 專用 CUDA graph | draft 與 target 共用 decode graph，形狀不同需 padding 或 fallback | SGLang `srt/speculative/eagle_draft_cuda_graph_runner.py` 等三支 | draft 的 batch 是 `bs × k`，另捕一組圖避開 padding | **B** | graph runner 選項 |
| **T13** | 搶佔受害者選擇 | 硬寫「挑 `TokenSize()` 最大者」（`csrc/scheduler/operations/forward.cpp:1277`） | vLLM 取 running 尾端（`v1/core/sched/scheduler.py:1145`）；SGLang 用 `retraction_key`（`srt/managers/schedule_batch.py:2839`） | 兩者都以「最晚到達」為主軸，不以「釋放最多」為主軸 | **C** | **—**（需 §1.3 的介面；這一項是該介面最好的第一個使用者） |

### 2.3 不要動（我們比較好或已對齊）

| 項目 | 說明 |
|---|---|
| kernel registry 的 arch 閘 | 同時有 `min_arch_version` 與 `max_arch_version`（`ops/attention/msa.py:96` 用 `(10,0)`–`(10,3)` 把 kernel 圈在 SM100）。SGLang 缺的正是上界，才會把 sm_120 放進 SM100 kernel。**別把它改成表** |
| GDN decode 路徑 | 已是 2 launch/層（`runtime/layers/attention/backends/hybrid_linear_attn.py:2249` + `:2311`），融合程度僅次於 TurboMind |
| prefill 專用 breakable graph | 與 SGLang 同級（vLLM 只有 piecewise，TRT-LLM 只有 decode graph） |
| C++ scheduler 的 TP 決定性 | `csrc/scheduler/operations/forward.cpp:1152` 明寫理由；每 rank 各跑一份 scheduler 的架構必須這樣 |

---

## 3. 移植順序

**能力缺口 > 效率落差**；同級內 **1:1 度 A > B > C**；規模小者先。

### 第一波 — 純複製、無阻塞（先把用途擴大）

| 順序 | 項目 | 為什麼排這裡 |
|---|---|---|
| 1 | **T1 w4a16** | 唯一「已經抄了一半」的項目；補完就從 `NameError` 變成可用的量化路徑。而且它落在 kernel registry 上，不必先解 §1.3 的設計題 |
| 2 | **T6 stop 前綴抑制** | ~30 行，無阻塞，修的是正確性 |
| 3 | **T11 關掉重複 detokenize** | 改預設，不是加碼 |
| 4 | **T2 ngram drafter** | 四家都有而我們沒有，純 Python 版最小 |

### 第二波 — 需要接管線

5. **T9** spec verify kernel（前置：第一波；需驗證 SSM state pool 相容）
6. **T4** fp4 KV（前置：attention kernel 支援 fp4 讀取）
7. **T12** draft 專用 graph（前置：T9）

### 第三波 — 需要 §1.3 的 policy 介面，或大工程

8. **scheduler policy 介面**（openspec 的主題）→ 第一個使用者是 **T13**
9. **T7 + T8** GDN prefill 融合 —— 清單裡唯一無法用 Python 複製解決的效率項，見 §4
10. **T10** hybrid 前綴匹配（結構改寫）
11. **T3** 樹狀 draft、**T5** chunk policy

---

## 4. T7 的特別說明：為什麼不能照抄 SGLang / vLLM

兩個 Python 引擎**看起來**都有現成的融合 GDN prefill kernel，但：

- SGLang `kernels/ops/attention/linear/gdn_blackwell/kernel_o.py:22` 的類別名就叫
  `Sm100ChunkOKernel`，內部用 `_tcgen05.alloc`
  （`kernels/ops/attention/linear/gdn_blackwell/kernel_o.py:278`）、`_tcgen05.mma_f16`
  （`kernels/ops/attention/linear/gdn_blackwell/kernel_o.py:311`）與 tensor memory
  （`kernels/ops/attention/linear/gdn_blackwell/kernel_o.py:194` 起）。
  **tcgen05 / tmem 是 SM100（B100/B200/GB200）的第五代 tensor core 指令，sm_120 沒有。**
  它的 `_is_blackwell()` 用 `major >= 10`，會把 sm_120 放行進一個跑不了的 kernel。
- vLLM 的兩條融合路徑都要求 `is_device_capability_family(100)`，在 sm_120 上不可達 ——
  **對這個 kernel 而言這道閘是對的**。
- TurboMind 為 sm_120 寫了獨立一份：`kernel/` 底下是 `pre_sm90` / `sm_90` / `sm_120`
  三個目錄、**沒有 sm_100**；用 TMA + 一般 `tiled_mma`，
  `src/turbomind/kernels/linear_attn/kernel/sm_120/plan.cc:116` 的註解直接針對 SM120 調過。

所以 T7 的 1:1 來源只有一個，而且是 C++/CUDA。開工前先做成本探勘。

---

## 5. 第一波的具體作法

### T1 — w4a16（最高優先）

**狀態：Python 半邊已完成（2026-08-08），CUDA GEMM 待做。**

已做：
- `runtime/layers/quantization/marlin_utils.py` 新增（SGLang 版的必要子集，
  Apache-2.0 標頭與出處保留）；刻意不移植 `MarlinConfig` / `MarlinLinearMethod`、
  `unified_apply_gptq_marlin_gemm*` 與 `get_tc_piecewise_forward_context` 分支
  （那是 SGLang 的 torch.compile piecewise 整合，本引擎用 breakable graph）、
  以及 AWQ / MoE 變體。
- `runtime/layers/quantization/utils.py` 補上 `pack_cols`（`unpack_cols` 的反向，
  `marlin_zero_points` 需要）。
- `compressed_tensors_wNa16.py` 還原 import 區塊、**刪除 `# ruff: noqa: F821`**。
- `tokenspeed_kernel/ops/quantization/cuda.py` 匯出 `gptq_marlin_gemm`，未建置時
  解析到 `error_fn` —— 失敗點從權重載入期的 `NameError` 變成 apply 期的明確錯誤。
- `test/runtime/layers/test_marlin_utils.py`（30 個測試）。

驗證：`create_weights` 從必然 `NameError` 變成可用；所有移植函式與 vLLM 的
`marlin_utils` 在相同輸入上**逐位元相同**（`get_scale_perms`、`marlin_permute_scales`
×3 形狀、`marlin_zero_points` 4/8-bit、`marlin_sort_g_idx`、`marlin_is_k_full` ×4、
`marlin_repeat_scales_on_all_ranks` ×3、`check_marlin_supports_shape` ×3，以及
`pack_cols` ×6 形狀），`test/runtime/layers` 99 passed / 31 skipped 無回歸。

待做：下方第 3–5 步（CUDA GEMM、註冊、端到端數值等價）。

---

原始步驟：

已驗證：tokenspeed 的 `compressed_tensors_wNa16.py`（339 行）與 SGLang 的（340 行）
**逐行對應**，連 `self.kernel_config = MarlinLinearLayerConfig(` 都在同一個行號 106。
差異只有授權標頭、參數類別改名、以及**被刪掉的 `marlin_utils` import 區塊**。

1. 複製 SGLang `srt/layers/quantization/marlin_utils.py`（1007 行）到
   `runtime/layers/quantization/marlin_utils.py`，只改 namespace 與參數類別名，
   保留 Apache-2.0 標頭與出處。
2. 還原 `compressed_tensors_wNa16.py` 的 import 區塊，**刪掉 `# ruff: noqa: F821`** ——
   那行是這個 bug 被壓住的原因，留著就會再發生一次。
3. GEMM：把 vLLM `csrc/libtorch_stable/quantization/marlin/` 的 `marlin.cu` /
   `marlin_template.h` / `kernel_selector.h` / `dequant.h` / `marlin_mma.h` /
   `marlin.cuh` / `marlin_dtypes.cuh` 併入 `thirdparty/cuda/csrc/`，在
   `flashinfer_marlin_binding.cu` 加 `TVM_FFI_DLL_EXPORT_TYPED_FUNC(marlin_gemm, marlin_gemm)`，
   `ops/quantization/cuda.py` 加包裝。選 `libtorch_stable` 這一份是因為它不依賴 ATen
   內部 API；而且已 vendored 的 `gptq_marlin_repack.cu` 就是同一份血緣。
4. **註冊而非取代**：`register_kernel("quantization", "gemm", name="marlin_gemm_cuda",
   solution="marlin", capability=CapabilityRequirement(min_arch_version=ArchVersion(8, 0)),
   priority=Priority.SPECIALIZED)`。`min 8.0` 是因為 Marlin 需要 `cp.async`，
   `thirdparty/cuda/csrc/tokenspeed_marlin.cuh` 自己就有 `__CUDA_ARCH__ < 800` 的守衛。
5. 驗證：先跑數值等價（對照 SGLang 同 checkpoint 的輸出），**再**談效能。

### T6 — stop 前綴抑制

抄 SGLang `srt/managers/schedule_batch.py:1427 tail_str` / `:1438
check_match_stop_str_prefix`，接到 `runtime/engine/request.py:297 check_finished` 同一個
類別上；串流輸出處比照 `srt/managers/scheduler_components/output_streamer.py:384` 加
`should_output &= not ...`。

### T11 — 關掉重複 detokenize

先確認 `runtime/engine/request.py:344`（scheduler 程序的 stop-string 比對）用的是
scheduler 自己的 detokenize 狀態，與主程序 `OutputProcessor` 那一份無關。確認後在 smg
路徑上把 `enable_inline_detokenizer` 關掉，以 `/v1/completions` 的文字輸出逐 token 比對
確認無回歸。

### T2 — ngram drafter

抄 vLLM `v1/spec_decode/ngram_proposer.py` 的 `propose`，接成
`runtime/execution/drafter/ngram.py`，實作 `runtime/execution/drafter/base.py:39` 的
`draft()` 介面。先用純 Python 版追平，之後再考慮 GPU 版。

---

## 6. 追平判準

**驗收順序固定：先數值等價，再效能不退，最後才談贏。**

1. **數值等價** — 同 checkpoint、同 prompt、greedy、同 seed，輸出文字與來源引擎逐
   token 相同。沿用 `results/_equiv/` 慣例。
2. **效能不退** — 移植前後跑同一個 sweep，確認沒把別的路徑弄慢。**新 solution 註冊
   進來不得改變既有預設的選擇結果**，除非那是這次移植明確要做的事。
3. **追平** — 對照來源引擎的同一項功能，達到同一量級即算追平。**這一步不要求贏。**
   `AGENTS.local.md` 的「+10% 才算過」是第二步（優化）的門檻，不是移植的門檻。
4. 移植完成後才進 `90-DIVERGENCES.md` §2 的量測，用來建立「優化前」的基線，
   並餵給 `AUTOTUNE` 選擇策略。

量測本身的固定檢查見 `90-DIVERGENCES.md` §3。
