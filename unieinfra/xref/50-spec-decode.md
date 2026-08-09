# 50 — Speculative decoding 五方橫向對照

版本見 `00-METHOD.md`。所有行號只對該表的 pinned commit 有效。

範圍：propose → verify → accept 三步，以及被拒絕之後的狀態回捲。

---

## 0. 誰有什麼

| engine | draft model | MTP / NextN | EAGLE | ngram | 位置 |
|---|---|---|---|---|---|
| UnieInfra | ✓ | ✓ | ✓ | — | `runtime/execution/drafter/` |
| vLLM | ✓ | ✓ | ✓ | ✓（CPU 與 GPU 兩版） | `v1/spec_decode/` |
| SGLang | ✓ | ✓ | ✓（v2、multi-layer） | ✓ | `srt/speculative/` |
| TRT-LLM | ✓ | ✓ | ✓（含 dynamic tree） | ✓ | `_torch/speculative/` |
| **TurboMind** | **—** | **—** | **—** | **—** | **C++ 引擎沒有 spec decode** |

TurboMind 這一格是空的，而且是結構性的：`src/turbomind/` 整棵樹只有一處提到 draft
token（`src/turbomind/engine/engine.cc:455` 的 `inflight_new_tokens /* plus draft tokens */`
註解，用來算 context 長度）。LMDeploy 的 EAGLE 實作只存在於**另一個後端**
（`lmdeploy/pytorch/models/llama_eagle.py`、`llama_eagle3.py`），跟 TurboMind 不共用程式碼。

→ 這一章 TurboMind 欄位一律 `—`，不是沒讀到。

---

## 1. 步驟表

| # | 步驟 | UnieInfra | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 1 | drafter 介面 | `runtime/execution/drafter/base.py:39 BaseDrafter` / `:82 draft` | `v1/spec_decode/eagle.py:10 EagleProposer` | `srt/speculative/base_spec_worker.py:57 EagleDraftWorkerBase` / `:67 draft` | `_torch/speculative/drafter.py:12 Drafter` / `:27 prepare_draft_tokens` | — |
| 2 | MTP propose | `runtime/execution/drafter/mtp.py:304 Mtp` / `:976 draft` | `v1/spec_decode/` 的 draft model 路徑 | `srt/speculative/frozen_kv_mtp_worker_v2.py` | `_torch/speculative/mtp.py:280 MTPWorker` | — |
| 3 | EAGLE propose | `runtime/execution/drafter/eagle.py:73 Eagle` / `:463 draft` | `v1/spec_decode/eagle.py:10 EagleProposer` | `srt/speculative/eagle_worker_v2.py:128 EagleDraftWorker` / `:489 draft` | `_torch/speculative/eagle3.py` | — |
| 4 | ngram propose | — | `v1/spec_decode/ngram_proposer.py:135 propose`（另有 `ngram_proposer_gpu.py`） | `srt/speculative/ngram_worker.py`（另有 `cpp_ngram/`） | `_torch/speculative/ngram.py` | — |
| 5 | draft 階段的 CUDA graph | 共用 `runtime/execution/cuda_graph_wrapper.py:173 CudaGraphWrapper` | 共用 decode graph | **專用**：`srt/speculative/eagle_draft_cuda_graph_runner.py`、`eagle_draft_extend_cuda_graph_runner.py`、`frozen_kv_mtp_cuda_graph_runner.py` | `_torch/pyexecutor/cuda_graph_runner.py` | — |
| 6 | verify（一次多 token 的 target forward） | 走 `forward_extend`（見 2.1） | `v1/spec_decode/` + 主 runner | `srt/speculative/eagle_worker_v2.py:1492 verify` | `_torch/speculative/mtp.py:248 MTPSampler` | — |
| 7 | accept / reject 取樣 | `runtime/sampling/` | `v1/sample/` | `srt/speculative/eagle_utils.py` | `_torch/speculative/spec_sampler_base.py` | — |
| 8 | 樹狀 draft | — | `v1/spec_decode/dynamic/` | `srt/speculative/multi_layer_eagle_utils.py` | `_torch/speculative/dynamic_tree_ops.py`、`spec_tree_manager.py` | — |
| 9 | 拒絕後的 KV 回捲 | scheduler 的 `csrc/scheduler/operations/forward.cpp:632 scheduleDecode`（`verify_width`） | scheduler `v1/core/sched/scheduler.py:396 schedule` 的 `num_output_placeholders` | `srt/speculative/eagle_utils.py` | `_torch/pyexecutor/py_executor.py:5377 _schedule` | — |

---

## 2. 分歧點

### 2.1 verify 的 GDN 層走哪條 kernel 路徑（UnieInfra 的隱藏成本）

`runtime/layers/attention/backends/hybrid_linear_attn.py:2176` 判斷
`q_len_per_req > 1` 就直接轉呼叫 `forward_extend`：

```python
q_len_per_req = kwargs["mixed_qkv"].shape[0] // bs if bs > 0 else 1
if q_len_per_req > 1:
    return self.forward_extend(...)
```

也就是說 **spec verify 的每一個 GDN 層付的是 prefill 的 9 個 Triton launch，不是 decode
的 2 個**（見 `30-prefill.md` 第 2.1 節與 `40-decode.md` 第 2.3 節）。48 層 × 9 = 432 個
launch，每個 verify step 一次。

SGLang 為同一件事做了相反的選擇：它有**專門的 verify kernel 路徑**
（`srt/speculative/eagle_worker_v2.py:1492 verify`，搭配
`sglang/kernels/ops/attention/fla/gdn_replayssm_spec_decode.py` 與
`gdn_replayssm_spec_fold.py`），以及**獨立的 verify backend 選項**
（`srt/layers/attention/linear/utils.py:98` 附近的 `get_linear_attn_verify_backend`）。

→ 這是一個「同功能、反結構」：一邊把 verify 當成短 prefill，一邊為 verify 另寫 kernel。

### 2.2 draft 階段要不要專用的 CUDA graph

SGLang 是唯一為 draft 階段開專用 graph runner 的（三個檔案：
`eagle_draft_cuda_graph_runner.py`、`eagle_draft_extend_cuda_graph_runner.py`、
`frozen_kv_mtp_cuda_graph_runner.py`）。UnieInfra 與 TRT-LLM 共用主 decode graph，
vLLM 走 piecewise。

draft model 的 batch 形狀與 target 不同（draft 是 `bs × k`，target 是 `bs`），共用 graph
表示其中一邊要 padding 或 fallback —— 值得量。

### 2.3 樹狀 draft 的支援程度

- UnieInfra：線性 draft（無 tree ops）
- vLLM：`v1/spec_decode/dynamic/`
- SGLang：`multi_layer_eagle_utils.py`
- TRT-LLM：最完整，`spec_tree_manager.py` + `dynamic_tree_ops.py` + `eagle3_dynamic_tree.py`

樹狀 draft 提高 accept rate，代價是 verify 的 attention mask 變複雜。四家都做了，
UnieInfra 沒有 —— 這是能力缺口，不是實作差異。

### 2.4 ngram

UnieInfra 沒有 ngram drafter。其他四家都有，且 SGLang 與 vLLM 各有一個非 Python 實作
（SGLang `srt/speculative/cpp_ngram/`、vLLM `v1/spec_decode/ngram_proposer_gpu.py`），
表示 Python 版的 ngram 匹配成本大到值得重寫。

---

## 3. 可量測候選（下一輪的輸入）

| 候選 | 量什麼 | 為什麼值得 |
|---|---|---|
| E1 verify 的 GDN 成本 | UnieInfra spec verify step 的逐 kernel 表，對照純 decode step | 2.1 預測每個 verify 多付 432 個 launch；若成立，spec decode 在 hybrid 模型上的淨效益要重算 |
| E2 accept rate | 同一工作負載下 UnieInfra vs SGLang vs vLLM 的 accept rate | 沒有 accept rate 就無法把「每步成本」換算成「每 token 成本」 |
| E3 draft graph | UnieInfra draft 階段的 graph 命中率 | 共用 decode graph 的形狀是否對得上 |
| E4 concurrency 掃描 | spec decode 的淨效益 vs concurrency | 既有結論 +27~45% 是在特定 concurrency 下量的；需要曲線 |

**量測陷阱（重複提醒）**：spec decode 的 token 數一律取 `usage.completion_tokens`，
不要數 SSE chunk。數 chunk 曾把 +27~45% 的結果反轉成「永遠是輸的」。

---

## 4. 未解格（`?`）

無。TurboMind 欄位全為 `—`（結構性缺席，已在第 0 節說明）；UnieInfra 的 ngram 與
tree draft 亦為 `—`（能力缺口）。
