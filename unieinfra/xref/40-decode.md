# 40 — Decode 五方橫向對照（到 kernel）

版本見 `00-METHOD.md`。所有行號只對該表的 pinned commit 有效。

範圍：一個 decode step，從 scheduler 交出的 batch 到 sampled token 回到前端。
與 `30-prefill.md` 一樣分 **A = dense full attention**、**B = GDN linear attention** 兩條路。

---

## 0. 一句話的結論

**Decode 不是落差所在，而且原因在原始碼裡看得出來：GDN 的 decode 路徑只有 2 個
kernel launch/層（prefill 是 9），而且整個 step 進得了 CUDA graph。**

| engine | decode step 是否整段進 graph | 每層 GDN decode launch |
|---|---|---|
| UnieInfra | **是**（整個 forward 一張圖，依 batch size 分桶） | 2 |
| vLLM | piecewise（`splitting_ops` 切開 attention） | 2 |
| SGLang | 是（`DecodeCudaGraphRunner`） | 2 |
| TRT-LLM | 是（`CUDAGraphRunner`） | 2 |
| TurboMind | **完全不用 CUDA graph** | 1（`LaunchSm120Recurrent`） |

TurboMind 是五家裡唯一一個不用 CUDA graph 的 —— `src/turbomind/` 整棵樹 grep
`cudaGraph` 零命中。它靠的是「本來就是 C++、launch 數本來就少」，不是靠重播。
（lmdeploy 另一個 PyTorch 後端才有 graph：`lmdeploy/pytorch/models/utils/cudagraph.py:53
CudaGraphMixin`，且 `:69 support_cuda_graph` 明寫只在 decoding context 回 True。）

---

## 1. 步驟表

| # | 步驟 | UnieInfra | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 1 | decode step 入口 | `runtime/execution/model_executor.py:959 _forward_step`（被 `:554` 的 `CudaGraphWrapper` 包住） | `v1/worker/gpu_model_runner.py:4068 execute_model` | `srt/model_executor/model_runner.py:1452 forward` | `_torch/pyexecutor/model_engine.py:6417 forward` | `src/turbomind/models/llama/unified_decoder.cc:164 UnifiedDecoder::Forward` |
| 2 | graph 命中判定 | `runtime/execution/cuda_graph_wrapper.py:1023 can_run` | `v1/cudagraph_dispatcher.py:235 dispatch` | `srt/model_executor/runner/decode_cuda_graph_runner.py:584 can_run_graph` | `_torch/pyexecutor/cuda_graph_runner.py:433 needs_capture` / `:545 replay` | — |
| 3 | graph 捕捉 | `runtime/execution/cuda_graph_wrapper.py:303 capture` | vLLM compile 期捕捉 | `srt/model_executor/runner/decode_cuda_graph_runner.py:933 capture` | `_torch/pyexecutor/cuda_graph_runner.py:460 capture` | — |
| 4 | 沒命中的 fallback | 走 eager `_forward_step` | eager / piecewise | eager runner（`srt/model_executor/runner/eager_runner.py`） | eager forward | 無此分支 |
| 5 | attention（A：full） | `runtime/layers/attention/backends/mha.py:985 forward_decode` | `v1/attention/backends/flash_attn.py` | `srt/layers/attention/triton_backend.py` | `_torch/attention_backend/trtllm.py` | `src/turbomind/models/llama/unified_attention_layer.cc` |
| 6 | attention（B：GDN） | `runtime/layers/attention/backends/hybrid_linear_attn.py:2161 forward_decode` | `v1/attention/backends/gdn_attn.py` | `srt/layers/attention/linear/gdn_backend.py` | `_torch/modules/fla/` | `src/turbomind/kernels/linear_attn/kernel/sm_120/entry.cu:238 LaunchSm120Recurrent` |
| 7 | conv1d 狀態更新 | `runtime/layers/attention/backends/hybrid_linear_attn.py:2249 causal_conv1d_update` | 獨立 kernel | 獨立 kernel | 獨立 kernel | 融進 recurrent kernel |
| 8 | recurrent 掃描 | `runtime/layers/attention/backends/hybrid_linear_attn.py:2311 gdn_decode_step` → `ops/attention/__init__.py:551 gdn_decode_step`（單一融合 kernel，含 L2 norm） | fused recurrent | fused recurrent | fused recurrent | 同 6 |
| 9 | MLP / MoE | `runtime/moe/` | `model_executor/layers/fused_moe/` | `srt/layers/moe/` | `_torch/modules/fused_moe/` | `src/turbomind/models/llama/moe_ffn_layer.cc` |
| 10 | logits | `runtime/layers/logits_processor.py` | `model_executor/layers/logits_processor.py` | `srt/layers/logits_processor.py` | `_torch/pyexecutor/handle_logits.py` | `src/turbomind/models/llama/language_model.cc` |
| 11 | logits processor / grammar | `runtime/sampling/custom_logit_processor.py`、`runtime/grammar/` | `v1/sample/logits_processor/` | `srt/sampling/custom_logit_processor.py` | `_torch/pyexecutor/guided_decoder.py` | `src/turbomind/generation/logits_processor.cc`、`guided_decoding.cc` |
| 12 | sampling | `runtime/sampling/backends/{flashinfer,triton,greedy}.py` | `v1/sample/sampler.py:20 Sampler` | `srt/layers/sampler.py:70 Sampler` | `_torch/pyexecutor/sampler/sampler.py:187 Sampler` | `src/turbomind/generation/sampling.cc:92 Sampling::Forward` |
| 13 | stop 判定 | Scheduler 程序：`runtime/engine/request.py:297 check_finished` | API server 程序：`v1/engine/output_processor.py:639` | Scheduler 程序：`srt/managers/schedule_batch.py:1467` | executor 側 | **GPU**：`src/turbomind/generation/stop_criteria.cc:58 StopCriteria::Forward` |
| 14 | KV append | 在 attention backend 內 | 在 attention backend 內 | 在 attention backend 內 | `cpp/tensorrt_llm/batch_manager/allocateKvCache.cpp` 配置、寫入在 kernel 內 | `src/turbomind/kernels/attention/` 內 |
| 15 | 輸出回主機 | `runtime/engine/event_loop.py` → ZMQ | `v1/engine/core.py` → ZMQ | ZMQ → DetokenizerManager | callback / IPC | 同程序 callback：`turbomind/turbomind.py:447 _func` |

---

## 2. 分歧點

### 2.1 CUDA graph 的捕捉粒度 — 四種

| engine | 捕捉什麼 | key |
|---|---|---|
| UnieInfra | **整個 decode forward**（含 sampling 前的所有層） | batch size 分桶 |
| vLLM | **piecewise**：以 `splitting_ops` 切成多段，attention 在段之間 | 依 compile 的動態形狀 |
| SGLang | 整個 decode forward，另有 dedup mixin 讓多個形狀共用圖 | `srt/model_executor/runner/shape_key.py` |
| TRT-LLM | 整個 decode forward | `_torch/pyexecutor/cuda_graph_runner.py:247 get_graph_key` |
| TurboMind | 不捕捉 | — |

「整段捕捉」的代價是 **fallback 是全有全無**：形狀不在桶裡就整個 step 走 eager。
vLLM 的 piecewise 沒有這個懸崖，但每段之間有 graph launch 的固定成本。

### 2.2 sampling 在哪一層

四家 Python 引擎的 sampler 都是一個 `nn.Module`／類別，跑在 model runner 之後、
還在同一個 GPU stream 上。TurboMind 的 sampling 是**生成管線裡的一個 stage**
（`src/turbomind/generation/generation.cc` 串起 `logits_processor` → `sampling` →
`stop_criteria`），三者都是 GPU kernel，中間不回主機。

實務後果：**TurboMind 的一個 decode step 完全不需要 host 參與**（除了 launch），
其他四家至少要一次 D2H 把 sampled token 帶回排程端。

### 2.3 GDN decode 的融合程度

UnieInfra 的 decode 路徑已經是融合的：
`runtime/layers/attention/backends/hybrid_linear_attn.py:2311 gdn_decode_step` 是單一
kernel（L2 norm 在 kernel 內），加上 `:2249 causal_conv1d_update` 共 2 個 launch。
這與 prefill 的 9 個形成對比 —— **同一個模型、同一層，prefill 沒有做 decode 已經做到的
融合**。

TurboMind 更進一步把 conv 也融進去，1 個 launch。

### 2.4 多 token decode（spec verify）走哪條路

UnieInfra 在 `runtime/layers/attention/backends/hybrid_linear_attn.py:2176` 判斷
`q_len_per_req > 1` 就**轉呼叫 `forward_extend`** —— 也就是說 speculative verify 的
GDN 層走的是 **prefill 的 9-launch 路徑**，不是 decode 的 2-launch 路徑。這條在
`50-spec-decode.md` 再展開。

---

## 3. 結構差異（同功能、反結構）

| 功能 | 多數作法 | 例外 | 後果 |
|---|---|---|---|
| decode 步的主機參與 | 每步至少一次 D2H 取 sampled token | TurboMind：全程在 GPU，stop 也在 GPU | 主機端 per-step 成本近乎零，代價是 stop 只能比對 token id |
| graph 粒度 | 整段捕捉（UnieInfra / SGLang / TRT-LLM） | vLLM piecewise；TurboMind 無 | 整段捕捉有形狀懸崖，piecewise 沒有但常數成本高 |
| GDN prefill 與 decode 的融合水準 | 兩邊一致 | UnieInfra：decode 融合、prefill 沒有 | 落差集中在 prefill，decode 打平 |

---

## 4. 可量測候選（下一輪的輸入）

| 候選 | 量什麼 | 為什麼值得 |
|---|---|---|
| D1 每 token 主機端 / 裝置端 | batch 1 / 8 / 32 / 128，三家 | 驗證 decode 確實打平（既有數據 36.4 vs 36.9 ms 只有 batch 1） |
| D2 graph 覆蓋率與 fallback | 各家實際命中 graph 的比例、fallback 觸發條件 | 整段捕捉的形狀懸崖只有量了才知道多常踩到 |
| D3 逐 kernel 佔比 | nsys：GEMM vs attention vs sampling vs 其他 | 找出 batch 增大時先飽和的是哪一項 |
| D4 sampling 成本 | 三家 sampler 在 batch 128 的耗時 | flashinfer / triton / greedy 三個 backend 的選擇沒被量過 |
| D5 spec verify 的 GDN 路徑 | `q_len_per_req > 1` 走 prefill 路徑的實際成本 | 若 verify 每步付 9 launch/層，spec decode 的淨效益會被吃掉 |

---

## 5. 未解格（`?`）

無。步驟表所有格子皆為 `file:line`、目錄指標或 `—`。
