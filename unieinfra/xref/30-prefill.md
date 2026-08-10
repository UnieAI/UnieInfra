# 30 — Prefill 五方橫向對照（到 kernel）

版本見 `00-METHOD.md`。所有行號只對該表的 pinned commit 有效。

範圍：從 scheduler 交出的 batch，到 logits。兩條路徑分開列：
**A = dense + full attention（`Qwen3-8B`）**、**B = hybrid GDN linear attention（`Qwen3.5-27B`）**。

---

## 0. 兩個結論先講

**(1) GDN prefill 的 kernel launch 數，五家差一個數量級。**

| engine | 每層 GDN launch 數 | 來源 |
|---|---|---|
| UnieInfra | **9** | Triton/FLA：`ops/attention/triton/linear/chunk.py:95 ChunkGatedDeltaRuleFunction.forward` |
| vLLM（sm_120 實際走的路） | **9** | 同一套 FLA：`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:360 fla_chunk_gated_delta_rule` |
| SGLang（預設） | **9** | Triton |
| SGLang（`--linear-attn-prefill-backend cutedsl`） | 3，**但只在 SM100，本機跑不了**（見 §0.1） | `srt/layers/attention/linear/kernels/gdn_cutedsl.py:33 CuteDSLGDNKernel` |
| **TurboMind** | **3**（快路徑）／**6**（context-parallel 路徑） | 手寫 sm_120 kernel：`src/turbomind/kernels/linear_attn/kernel/sm_120/entry.cu:43 RunDirectChunk32` |

**(2) 在 sm_120 上，vLLM 的融合 GDN 路徑是「結構上不可達」，不是「沒開」。**

`_resolve_gdn_prefill_backend` 的兩個快路徑都要求
`current_platform.is_device_capability_family(100)`
（`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:189`），而該函式的定義是
`(capability.to_int() // 10) == (100 // 10)`
（`platforms/interface.py:466 is_device_capability_family`）。sm_120 → `120 // 10 = 12 ≠ 10`
→ **FlashInfer 與 CuteDSL 兩條路在這張卡上都拿不到，連 opt-in 都不行**。

### 0.1 但 SGLang 的 CuteDSL kernel 在 sm_120 上**跑不了** —— 這裡有兩個相反的 bug

SGLang 對「是不是 Blackwell」的判斷是 `major >= 10`
（`srt/layers/attention/linear/kernels/gdn_cutedsl.py:25 _is_blackwell`），sm_120 通過。
但它的 kernel 是 **SM100 專屬**的：`kernels/ops/attention/linear/gdn_blackwell/kernel_o.py:22 Sm100ChunkOKernel` 用
`_tcgen05.alloc`（`kernels/ops/attention/linear/gdn_blackwell/kernel_o.py:278`）、`_tcgen05.mma_f16`（`kernels/ops/attention/linear/gdn_blackwell/kernel_o.py:311`）與 tensor memory
（`kernels/ops/attention/linear/gdn_blackwell/kernel_o.py:194` 起的 `qk_tmem` / `p_tmem` / `out_tmem`）。**tcgen05 與 tmem 是 SM100
（B100/B200/GB200）的第五代 tensor core 指令，sm_120 這張卡沒有。**

所以：

| | 閘 | kernel 實際需要 | 結論 |
|---|---|---|---|
| vLLM | `family(100)`（封閉，排除 12x） | SM100 | **閘是對的** |
| SGLang | `major >= 10`（開放，放行 12x） | SM100 | **閘是錯的** —— 放行到一個跑不了的 kernel |

兩個引擎在同一個判斷上犯了**相反方向**的錯，而在這個 kernel 上剛好是 vLLM 對。
`>=` 誤放行與封閉表誤排除是同一個問題的兩面，不能因為看到其中一種就假設另一種。

**結果：TurboMind 是五家裡唯一一個有「在 sm_120 上真的跑得動」的融合 GDN prefill
kernel 的引擎。** 它為 sm_120 寫了獨立的一份（`src/turbomind/kernels/linear_attn/kernel/`
底下三個 arch 目錄：`pre_sm90`、`sm_90`、`sm_120`，**沒有 sm_100**），用的是 TMA +
一般 `tiled_mma`，不碰 tcgen05；`src/turbomind/kernels/linear_attn/kernel/sm_120/plan.cc:116` 的註解直接寫「On SM120 the chunk32
CP setup/correction cost does not amortize until…」，是針對這張卡調過的。

---

## 1. 步驟表 — 路徑 A（dense + full attention，`Qwen3-8B`）

| # | 步驟 | UnieInfra | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 1 | forward 入口 | `runtime/execution/model_executor.py:959 _forward_step` | `v1/worker/gpu_model_runner.py:4068 execute_model` | `srt/model_executor/model_runner.py:1452 forward` | `_torch/pyexecutor/model_engine.py:6417 forward` | `src/turbomind/models/llama/unified_decoder.cc:164 UnifiedDecoder::Forward` |
| 2 | batch metadata 準備 | `runtime/execution/forward_batch_info.py` | `v1/worker/gpu_input_batch.py` | `srt/model_executor/forward_batch_info.py` | `_torch/metadata.py` + `_torch/attention_backend/interface.py` | `src/turbomind/engine/scheduler.cc:1313 CommitResults` 產生的 tensor |
| 3 | embedding | `runtime/layers/vocab_parallel_embedding.py` | `model_executor/layers/vocab_parallel_embedding.py` | `srt/layers/vocab_parallel_embedding.py` | `_torch/modules/embedding.py` | `src/turbomind/models/llama/unified_decoder.cc` 內 |
| 4 | input layernorm | `runtime/layers/layernorm.py` | `model_executor/layers/layernorm.py` | `srt/layers/layernorm.py` | `_torch/modules/layer_norm.py` | `src/turbomind/kernels/norm/` |
| 5 | QKV proj | `runtime/layers/linear.py` | `model_executor/layers/linear.py` | `srt/layers/linear.py` | `_torch/modules/linear.py` | `src/turbomind/models/llama/LlamaLinear.cu` |
| 6 | RoPE | `runtime/layers/rotary_embedding.py` | `model_executor/layers/rotary_embedding/` | `srt/layers/rotary_embedding/` | `_torch/modules/rotary_embedding.py` | `src/turbomind/models/llama/llama_rope.h` |
| 7 | attention（prefill） | `runtime/layers/attention/backends/mha.py:1037 forward_extend` | `v1/attention/backends/flash_attn.py` | `srt/layers/attention/triton_backend.py` | `_torch/attention_backend/trtllm.py` / `triton_prefill.py` | `src/turbomind/models/llama/unified_attention_layer.cc`；kernel 在 `src/turbomind/kernels/attention/` |
| 8 | o_proj | 同 5 | 同 5 | 同 5 | 同 5 | 同 5 |
| 9 | post norm + residual | 同 4 | 同 4 | 同 4 | 同 4 | `src/turbomind/models/llama/unified_decoder.cc:104 AllreduceResidualRMSnorm` |
| 10 | MLP / MoE | `runtime/layers/dense/`、`runtime/moe/` | `model_executor/layers/fused_moe/` | `srt/layers/moe/` | `_torch/modules/fused_moe/` | `src/turbomind/models/llama/{LlamaFfnLayer.cc,moe_ffn_layer.cc}` |
| 11 | final norm + logits | `runtime/layers/logits_processor.py` | `model_executor/layers/logits_processor.py` | `srt/layers/logits_processor.py` | `_torch/pyexecutor/handle_logits.py` | `src/turbomind/models/llama/language_model.cc` |

路徑 A 五家的**結構**是一樣的（同樣的 11 步、同樣的順序）。差別只在第 7 步選到哪個
backend，以及第 10 步的 MoE 實作。**這條路不是落差所在**，列出來是為了對照組。

---

## 2. 步驟表 — 路徑 B（GDN linear attention，`Qwen3.5-27B`）

第 4、8–11 步與路徑 A 相同，以下只列不同的部分。

| # | 步驟 | UnieInfra | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 5' | in_proj（qkvz + ba） | `runtime/models/qwen3_5.py:466 _forward_input_proj` | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | `srt/layers/attention/linear/gdn_backend.py` | `_torch/modules/fla/` | `src/turbomind/models/llama/GatedDeltaNetLayer.cc:375`（**一次 GEMM 出全部**） |
| 5''| qkvz/ba 拆分 reshape | `runtime/models/qwen3_5.py:471 fused_qkvzba_split_reshape_cat_contiguous`（融合）或 `:481 fix_query_key_value_ordering`（未融合） | `fla/ops/fused_gdn_prefill_post_conv.py` | `sglang/kernels/ops/attention/triton_gdn_fused_proj.py` | `_torch/modules/fla/` | 併在 conv kernel 裡 |
| 6' | causal conv1d + SiLU | `runtime/layers/attention/backends/hybrid_linear_attn.py:2707 causal_conv1d_fn`（獨立 kernel） | 獨立 kernel | 獨立 kernel | 獨立 kernel | **融合**：`src/turbomind/models/llama/GatedDeltaNetLayer.cc:393 invokeFusedConv1dSiLU` |
| 7a | q/k L2 norm | **兩次獨立呼叫**：`ops/attention/triton/linear/chunk.py:112` 的 `l2norm_fwd(q)` 與 `l2norm_fwd(k)` | 同（FLA） | 同（Triton 路徑） | `_torch/modules/fla/l2norm.py` | **一次**：`src/turbomind/models/llama/GatedDeltaNetLayer.cc:421 invokeL2NormalizeQK` |
| 7b | gate cumsum | `ops/attention/triton/linear/chunk.py:54 chunk_local_cumsum` | 同 | 同 | `_torch/modules/fla/` | 融進 `src/turbomind/kernels/linear_attn/kernel/sm_120/entry.cu:52 LaunchChunk32LocalCumsumAndPrepareDirect` |
| 7c | KKT + solve_tril + w/u | `ops/attention/triton/linear/chunk.py:56 chunk_scaled_dot_kkt_fwd` → `:59 solve_tril` → `:60 recompute_w_u_fwd`（3 個以上 launch） | 同 | CuteDSL 路徑合成 1 個：`sglang/kernels/ops/attention/linear/gdn_blackwell/kernel_kkt_inv_uw.py` | `_torch/modules/fla/` | **1 個**：`src/turbomind/kernels/linear_attn/kernel/sm_120/entry.cu:64 LaunchSm120KktSolve` |
| 7d | chunked state h | `ops/attention/triton/linear/chunk.py:68 chunk_gated_delta_rule_fwd_h` | 同 | `gdn_blackwell/kernel_h.py` | 同 | 融進 7e |
| 7e | 輸出 o | `ops/attention/triton/linear/chunk.py:77 chunk_fwd_o` | 同 | `gdn_blackwell/kernel_o.py` | 同 | **1 個**：`src/turbomind/kernels/linear_attn/kernel/sm_120/entry.cu:73 LaunchSm120FusedGdrFwd`（h + o 融合） |
| 7f | gated RMSNorm | `runtime/models/qwen3_5.py:523 self.norm(core_attn_out, z)` | 獨立 | 獨立 | 獨立 | `src/turbomind/models/llama/GatedDeltaNetLayer.cc:512 invokeRMSNormGated` |

### 2.1 launch 數逐項對帳（UnieInfra，每層）

依 `ops/attention/triton/linear/chunk.py:95 ChunkGatedDeltaRuleFunction.forward` 逐行數：

| # | 呼叫 | 位置 | launch |
|---|---|---|---|
| 1 | `l2norm_fwd(q)` | `ops/attention/triton/linear/chunk.py:112` | 1 |
| 2 | `l2norm_fwd(k)` | `ops/attention/triton/linear/chunk.py:113` | 1 |
| 3 | `chunk_local_cumsum(g)` | `ops/attention/triton/linear/chunk.py:54` | 1 |
| 4 | `chunk_scaled_dot_kkt_fwd` | `ops/attention/triton/linear/chunk.py:56` | 1 |
| 5–6 | `solve_tril` | `ops/attention/triton/linear/chunk.py:59` → `ops/attention/triton/linear/solve_tril.py:392` | 2（`solve_tril_16x16_kernel` + merge） |
| 7 | `recompute_w_u_fwd` | `ops/attention/triton/linear/chunk.py:60` | 1 |
| 8 | `chunk_gated_delta_rule_fwd_h` | `ops/attention/triton/linear/chunk.py:68` | 1 |
| 9 | `chunk_fwd_o` | `ops/attention/triton/linear/chunk.py:77` | 1 |
| | | **合計** | **9** |

48 個 GDN 層 × 9 = **432 個 launch/forward**，全部在 CUDA graph 之外（見第 3 節）。

TurboMind 的同一段是 3 個（`src/turbomind/kernels/linear_attn/kernel/sm_120/entry.cu:52`, `:64`, `:73`）：48 × 3 = 144。

---

## 3. CUDA graph 策略 — 五家都把 linear attention 排除在外，但排法不同

| engine | 機制 | 排除清單所在 |
|---|---|---|
| UnieInfra | **breakable prefill graph**，一個 token bucket 一張圖，attention 當 eager break | `runtime/execution/prefill_graph.py:176 PrefillGraph`；預設開（`runtime/execution/model_executor.py:198 disable_prefill_graph = False`） |
| vLLM | piecewise compile，`splitting_ops` 逐 op 切 | `config/compilation.py:750 "vllm::mamba_mixer2"`、`:753 "vllm::linear_attention"`、`:755 "vllm::qwen_gdn_attention_core"` |
| SGLang | **prefill 專用 breakable graph**（`srt/model_executor/runner/prefill_cuda_graph_runner.py:243 PrefillCudaGraphRunner`，預設開：`srt/server_args.py:1847 disable_prefill_cuda_graph = False`）＋`eager_on_graph` 裝飾器 | `srt/layers/radix_linear_attention.py:160 bcg_unified_linear_attention_with_output` |
| TRT-LLM | piecewise，以 custom op 為邊界 | `_torch/compilation/piecewise_optimizer.py:28 "gdn_custom_op_inplace"` |
| TurboMind | prefill 沒有 graph（純 C++，launch 數本來就少） | — |

**架構上五家一致：沒有人把 linear attention 放進圖裡。**

「prefill 專用 graph」則是 UnieInfra 與 SGLang **兩家都有**，而且兩家都預設開、都用
breakable graph 把 attention 留在圖外（UnieInfra `runtime/execution/breakable_cuda_graph.py`、
SGLang `srt/model_executor/runner_backend_utils/breakable_cuda_graph/`）。vLLM 的
piecewise compile 效果相近但機制不同；TRT-LLM 只有 decode 側的 graph；TurboMind 完全
不用 CUDA graph（見 `40-decode.md`）。

所以 GDN prefill 的成本差**不是 graph 策略造成的，是 launch 數造成的** —— 這是這一章
最重要的區分，因為「把它放進 graph」是個聽起來合理但走不通的方向。

---

## 4. 分歧點

### 4.1 融合的粒度

TurboMind 在四個地方做了其他家沒做的融合：

1. **in_proj 一次 GEMM 出 qkvzba**（`src/turbomind/models/llama/GatedDeltaNetLayer.cc:375`），其他家分兩次再拆；
2. **conv1d + SiLU 融合**（`:393 invokeFusedConv1dSiLU`）；
3. **q 與 k 的 L2 norm 一個 kernel**（`:421 invokeL2NormalizeQK`），UnieInfra 是兩次
   `l2norm_fwd`；
4. **KKT + solve_tril 一個 kernel**、**h + o 一個 kernel**
   （`src/turbomind/kernels/linear_attn/kernel/sm_120/entry.cu:64`, `:73`）。

UnieInfra 已經做了其中一項對應的融合（`runtime/models/qwen3_5.py:471
fused_qkvzba_split_reshape_cat_contiguous`），但只在 `num_v_heads // num_k_heads ∈
{1,2,4}` 時走（`runtime/models/qwen3_5.py:470`），且融的是**拆分**而不是 in_proj 本身。

### 4.2 動態 grid 的處理方式

TurboMind 在 `entry.cu` 依有效 grid 佔用度在兩條路徑間選擇：
`RunDirectChunk32`（3 launch）與 `RunContextParallelChunk32`（6 launch，多做 TMA
descriptor 準備、segment state 修正）。判準寫在 `src/turbomind/kernels/linear_attn/kernel/sm_120/entry.cu:36-38`：把總工作量用最長序列
正規化，grid 太窄就走 context-parallel 把序列切段平行。

其他四家沒有這個切換 —— Triton 版的 grid 直接由 `cu_seqlens` 決定，varlen batch 不平衡
時就是長尾。

### 4.3 mixed batch 裡 prefill 與 decode 的切分順序

這就是 `AGENTS.local.md` 記的那一條，屬於 **kernel 側**（不是 `20-scheduler.md` 的排程
優先權）：

- TurboMind 把 recurrent（decode）與 chunked（prefill）**分派到兩條 stream**
  （`src/turbomind/models/llama/GatedDeltaNetLayer.cc:440 chunk_stream`），兩段可以重疊；
- 其他四家在同一條 stream 上依序做兩段。

TurboMind 另外一項：傳 offset 而不是實體化 slice（`args.q_offsets` 一路傳到 kernel），
省掉一整類對齊 bug 與一次 gather。

### 4.4 conv1d 要不要切

`AGENTS.local.md` 記錄：vLLM 的 GDN 路徑**不切 convolution**，而 mamba2 路徑會切。
本表確認 vLLM 的 GDN 走的是 `qwen_gdn_linear_attn.py`，與 `mamba2` 是不同檔案 ——
survey 時讀錯檔案會得到相反的結論。

---

## 5. sm_120 上實際可達的 backend（本機四張 RTX PRO 6000）

| engine | GDN prefill 預設 | 可 opt-in 到融合版？ | 判準 |
|---|---|---|---|
| UnieInfra | Triton/FLA（9） | 無此選項 | — |
| vLLM | Triton/FLA（9） | **否** | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:189` 要求 family(100)；`platforms/interface.py:466` 排除 12x |
| SGLang | Triton（9） | **否**（閘會放行，kernel 需要 tcgen05 → 跑不了，見 §0.1） | `srt/layers/attention/linear/kernels/gdn_cutedsl.py:25` 用 `major >= 10`，但 kernel 是 `Sm100ChunkOKernel` |
| TRT-LLM | Triton（`_torch/modules/fla/`） | 未查證 | 需實測確認 |
| TurboMind | **手寫 sm_120 kernel（3）** | 預設即是 | `src/turbomind/kernels/linear_attn/kernel/sm_120/plan.cc` |

---

## 6. 可量測候選（下一輪的輸入）

| 候選 | 量什麼 | 為什麼值得 |
|---|---|---|
| C1 GDN launch 數與時間 | nsys 逐 kernel 表，`Qwen3.5-27B` prefill，UnieInfra vs SGLang(triton) vs SGLang(cutedsl) | 9 → 3 是原始碼層的事實；剩下要知道它值多少 ms |
| C2 prefill 主機端地板 | `forward_step` 對 prompt 長度 1/64/256/1024/4096 的曲線 | 找與 token 數無關的地板，對照 432 個 launch 的預期 |
| C3 SGLang cutedsl 在 sm_120 | 開起來確認它**如何**失敗（PTX 編譯期還是執行期） | 讀碼結論是跑不了（tcgen05）；跑一次是為了確認失敗模式，不是期待它成功 |
| C4 prefill graph 覆蓋率 | UnieInfra `PrefillGraph` 實際命中 bucket 的比例與省下的 launch 數 | 預設開但 attention 是 eager break，實際省多少沒量過 |
| C5 路徑 A 對照組 | `Qwen3-8B` prefill 三家逐 kernel | 確認落差確實只在 GDN，不在 dense 路徑 |
| C6 融合項逐項 | in_proj 融合、conv+SiLU 融合、L2 norm 合併各自值多少 | TurboMind 的四項融合可以逐一移植，需要知道優先順序 |

---

## 7. 未解格（`?`）

一格：**TRT-LLM 在 sm_120 上的 GDN prefill 實際選到哪個 kernel**。
`_torch/modules/fla/` 是 Triton 實作，但 TRT-LLM 另有 `_torch/attention_backend/` 的
trtllm-gen 路徑，選擇邏輯散在 `_torch/model_config.py` 與 autotuner，沒有像 vLLM
`_resolve_gdn_prefill_backend` 那樣的單一判斷點可引用。要補這一格需要實跑一次看 log。
