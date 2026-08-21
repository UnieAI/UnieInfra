# 五引擎原始碼橫向對照 — 方法與版本

這一系列文件把 **API layer → scheduler → prefill → decode** 四個階段的每一步，
在五個推論引擎中的對應 function / operator / kernel 攤平成同一張表。

目的是把 `AGENTS.local.md` 流程的第 2 步（survey 每個引擎怎麼實作）從「每次為一個
feature 臨時做一次」變成查表。跳過那一步在單一 feature 上曾花掉五個 debug cycle（E54）。

**這一輪只做原始碼對照，不含任何量測。** 量測是下一輪，範圍由 `90-DIVERGENCES.md`
決定。

## 文件

| 檔 | 內容 |
|---|---|
| `10-api.md` | API layer |
| `15-encoder-embedding.md` | encoder-only（embedding）與 encoder-decoder（Whisper）；四方，TurboMind 缺席 |
| `16-diffusion.md` | 影像/影片 diffusion（FLUX / Qwen-Image / MiniMax-H3）；只有 SGLang 與 FastVideo 做了 |
| `20-scheduler.md` | scheduler |
| `30-prefill.md` | prefill（到 kernel） |
| `40-decode.md` | decode（到 kernel） |
| `50-spec-decode.md` | propose / verify / accept |
| `60-quant.md` | fp8 KV / w4a16 / NVFP4 MoE |
| `90-DIVERGENCES.md` | 所有分歧點彙總 + 下一輪量測候選 |
| `check_xref.py` | 驗證所有 `file:line symbol` 引用仍存在 |

## 每份階段文件的固定結構

1. **步驟表** — 一列一步，欄位 = 5 個引擎，格內 `path/file.ext:NNN symbol()`。
   - 對不上填 `—`（此引擎沒有這一步）。**沒有對應本身就是發現。**
   - 沒讀到填 `?`，並在該節寫明原因。完成的標準是 `?` 收斂到 0。
2. **分歧點** — 逐列寫：各家決策差在哪、誰跟誰同一邊、這個選擇的後果。
3. **結構差異** — 同功能被放在不同的層（「同功能、反結構」專節）。
4. **可量測候選** — 這一階段哪些分歧點值得下一輪量，量什麼。

## 紀律

- **讀實作同一個 feature 的模組，不是它的表親。** GDN 讀 GDN 路徑，不能拿 mamba2 頂替；
  hybrid 的 prefix cache 讀 hybrid 那支，不能拿 full-attention 的頂替。這一條是因為
  GDN mixed batching 的 survey 第一次就讀錯了模組，兩個關鍵決策都不一樣。
- 每個引用帶 `file:line`，且該行必須是下表 pinned commit 上的真實行。
- 每格只寫「這個引擎在這一步呼叫什麼」，不寫推測。

## 版本（pinned）

對照表的每一個行號都只對這些 commit 有效。**維持 pin，不要 fetch 到最新**，
否則這一輪讀的和下一輪量的不是同一份程式碼。

| engine | 路徑 | commit | 備註 |
|---|---|---|---|
| UnieInfra (tokenspeed) | `/workspace/roy/UnieInfra` | `4d35899ca734` | 工作樹，upstream = `lightseekorg/tokenspeed` |
| vLLM | `/workspace/roy/vllm` | `4ad14044b254` | UnieAI fork；與 upstream 的合流點 `7cc2e8e74fa0`（2026-07-08），之後 40 個 UnieAI commit |
| SGLang | `.../legacy/third_party/sglang` | `b61a06921ef0` | |
| TensorRT-LLM | `.../legacy/third_party/TensorRT-LLM` | `d8b67db26082` | |
| TurboMind (LMDeploy) | `.../legacy/third_party/lmdeploy` | `a1e567416831` | |

### vLLM fork 與 upstream 的差異檔案

引用落在這幾個檔案時要標注「這是 fork 的行，不是 upstream 的行」：

```
vllm/config/speculative.py          vllm/config/vllm.py
vllm/entrypoints/unieai.py          vllm/entrypoints/cli/serve_optuna.py
vllm/models/deepseek_v4/nvidia/model.py
vllm/v1/attention/backends/turboquant_attn.py
vllm/v1/core/sched/output.py        vllm/v1/worker/workspace.py
vllm/benchmarks/sweep/{cli,serve,serve_optuna}.py
```

其餘 vLLM 檔案與 upstream `7cc2e8e74fa0` 相同。

### sparse checkout 範圍

三個參照 repo 是 `blob:none` partial clone + cone-mode sparse checkout。
本輪為了覆蓋 spec decode 與量化路徑做了擴充；擴充後的完整清單：

**sglang** `b61a06921ef0`
```
python/sglang/kernels            python/sglang/srt/compilation
python/sglang/srt/constrained    python/sglang/srt/entrypoints
python/sglang/srt/function_call  python/sglang/srt/layers
python/sglang/srt/managers       python/sglang/srt/mem_cache
python/sglang/srt/model_executor python/sglang/srt/model_loader
python/sglang/srt/models         python/sglang/srt/sampling
python/sglang/srt/speculative    python/sglang/srt/utils
```

**lmdeploy** `a1e567416831`
```
lmdeploy/pytorch/{backends,engine,kernels,models,nn,paging}
lmdeploy/serve                   lmdeploy/turbomind
src/turbomind/engine
src/turbomind/kernels/{attention,copy,core,gemm,linear_attn,norm}
src/turbomind/models
```
注意 `src/turbomind/layers/` 在此版本不存在（TurboMind 已把 layer 併進
`src/turbomind/models/llama/`）。

**TensorRT-LLM** `d8b67db26082`
```
cpp/tensorrt_llm/{batch_manager,executor,kernels}
tensorrt_llm/_torch/{attention_backend,compilation,custom_ops,distributed,
                     models,modules,pyexecutor,speculative}
tensorrt_llm/serve
```

要再擴充時用 `git -C <repo> sparse-checkout add <dir>`（cone mode），**不要**
`git fetch` 動到 pin。

## 模型路徑

| 路徑 | 代表模型 | 誰有 |
|---|---|---|
| dense + full attention | `Qwen3-8B` | 五方都有 |
| hybrid GDN linear attention | `Qwen3.5-27B` | 逐引擎確認，沒有就是一格發現 |

## 術語對照

各家對同一件事的命名不同，讀表時先看這張：

| 概念 | UnieInfra | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|
| 一次排程的產物 | `ExecutionPlan` | `SchedulerOutput` | `ScheduleBatch` → `ModelWorkerBatch` | `ScheduledRequests` | `ScheduleState`（就地改 `Sequence`） |
| 排程入口 | `Scheduler::NextExecutionPlan` | `Scheduler.schedule` | `Scheduler.get_next_batch_to_run` | `PyExecutor._schedule` | `Scheduler::Schedule` |
| 前綴快取 | `KvPrefixCache` / `HybridPrefixCache` | `KVCacheManager` + `BlockPool` | `RadixCache` 家族（含 `MambaRadixCache`） | `KVCacheManager`（C++） | `PrefixTrie` |
| 每步的 forward 入口 | `ModelExecutor.forward_step` | `GPUModelRunner.execute_model` | `ModelRunner.forward` | `PyTorchModelEngine.forward` | `UnifiedDecoder::Forward` |
| 請求物件 | `GenerateReqInput` → C++ `Request` | `EngineCoreRequest` → `Request` | `TokenizedGenerateReqInput` → `Req` | `LlmRequest` | `Request` / `Sequence` |
| prefill 的別名 | prefill / extend | prefill（無此階段概念，只有 token 預算） | extend | context | prompt |
| decode 的別名 | decode | decode | decode | generation | generation |
| KV 分頁單位 | page | block | page | block | block |

各家對 prefill / decode 的稱呼不同，讀原始碼時特別容易誤判：TRT-LLM 的 `context` 就是
prefill、`generation` 就是 decode；SGLang 的 `extend` 就是 prefill。
