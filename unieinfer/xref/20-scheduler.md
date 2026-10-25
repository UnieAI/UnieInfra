# 20 — Scheduler 五方橫向對照

版本見 `00-METHOD.md`。所有行號只對該表的 pinned commit 有效。

範圍：從請求進入引擎佇列，到一個可以送去 forward 的 batch 物件。**不含** forward
本身（那是 `30-prefill.md` / `40-decode.md`）。

---

## 0. 一句話的結構差異

| engine | scheduler 在哪 | 一步排程的產物 | 結構 |
|---|---|---|---|
| UnieInfer | **C++**（`tokenspeed-scheduler`，nanobind） | `ExecutionPlan` | 單一 pass，FSM 狀態決定優先權 |
| vLLM | Python | `SchedulerOutput` | 兩段：先 running 再 waiting，統一 token 預算 |
| SGLang | Python | `ScheduleBatch` | **prefill 與 decode 互斥**，二選一 |
| TRT-LLM | Python 殼 + **C++ 實作** | `ScheduledRequests` | **兩個獨立 scheduler**：capacity 再 micro-batch |
| TurboMind | **C++** | `ScheduleState`（就地改 `Sequence`） | 兩階段准入（required / optional）+ memory replay |

五家沒有兩家的結構是一樣的。下面每一節說明差異在哪一步。

---

## 1. 步驟表

`—` = 這個引擎沒有這一步。

| # | 步驟 | UnieInfer | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 1 | 每步排程入口 | `csrc/scheduler/scheduler.cpp:405 NextExecutionPlan` | `v1/core/sched/scheduler.py:396 schedule` | `srt/managers/scheduler.py:2961 get_next_batch_to_run` | `_torch/pyexecutor/py_executor.py:5377 _schedule` | `src/turbomind/engine/scheduler.cc:1018 Schedule` |
| 2 | 新請求入列 | `csrc/scheduler/scheduler.cpp:228 SubmitRequests` | `v1/core/sched/scheduler.py:2017 add_request` | `srt/managers/scheduler.py:2665 _add_request_to_queue` | `_torch/pyexecutor/executor_request_queue.py:122 enqueue_request` | `src/turbomind/engine/gateway.cc:31 Gateway::push` |
| 3 | waiting queue 策略 | FSM 狀態優先權 + `Id()` 平手（`csrc/scheduler/operations/forward.cpp:1140`, `:1153`） | `v1/core/sched/request_queue.py:75 FCFSRequestQueue` / `:131 PriorityRequestQueue` | `srt/managers/schedule_policy.py:211 SchedulePolicy`（LPM / DFS-weight / FCFS…） | `_torch/pyexecutor/scheduler/waiting_queue.py:96 FCFSWaitingQueue` / `:161 PriorityWaitingQueue` | `unique_id` 遞增（`src/turbomind/engine/scheduler.cc:1075`） |
| 4 | 容量准入（誰放得下） | 與批次組成同一個迴圈 | 同一個迴圈 | 同一個迴圈 | **獨立**：`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:708 operator()` | **兩階段**：`src/turbomind/engine/scheduler.cc:1105 RunRequiredAdmission` / `:1219 RunOptionalAdmission` |
| 5 | 前綴快取查詢 | `csrc/resource/kv_prefix_cache/kv_prefix_cache.cpp`（KV）＋`csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp:117 Match`（hybrid） | `v1/core/kv_cache_manager.py:202 get_computed_blocks` → `v1/core/block_pool.py:199 get_cached_block` | `srt/mem_cache/radix_cache.py:352 match_prefix`；hybrid 走 `srt/mem_cache/mamba_radix_cache.py:498 match_prefix` | `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp`（radix；`getEstimatedReusableTokens`） | `src/turbomind/engine/scheduler.cc:364 MatchPrompt` → `src/turbomind/engine/prefix_trie.h:15 PrefixTrie` |
| 6 | KV 分頁配置 | `csrc/cache/kv_cache_coordinator.cpp:255 AcquirePrefix` / `:347 Acquire` | `v1/core/kv_cache_manager.py:244 allocate_slots` → `v1/core/block_pool.py:542 get_new_blocks` | `srt/mem_cache/allocator/`＋`req_to_token_pool` | `cpp/tensorrt_llm/batch_manager/allocateKvCache.cpp` | `src/turbomind/engine/scheduler.cc:326 EnsureBlocks`；池在 `src/turbomind/engine/block.h:146 CacheBlockPool` |
| 7 | chunked prefill 切分 | `csrc/scheduler/operations/forward.cpp:374 schedulePrefillFirstChunk` / `:557 schedulePrefill` | 由 `token_budget` 隱式切（`v1/core/sched/scheduler.py:416`） | `srt/managers/schedule_policy.py:1187 add_one_req` | `cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp:299 setCtxRequestsChunkSize`（三種 policy） | `src/turbomind/engine/scheduler.cc:1000 ClampForwardEnd` |
| 8 | **prefill / decode 混批決策** | `csrc/scheduler/operations/forward.cpp:1145`（旗標）＋`:1232`, `:1246`（關閉時 break） | 恆混：running 先、waiting 後（`v1/core/sched/scheduler.py:441`, `:636`） | **互斥**：`srt/managers/scheduler.py:3071`「Run prefill first if possible」，否則才 decode | 同一個迴圈同時收 context 與 generation（`cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp:347`） | 無此概念：`unified_decoder` 同一次 forward 同時吃 prompt 與 generation |
| 9 | 搶佔 / 驅逐 | `csrc/scheduler/operations/forward.cpp:1277`（挑 `TokenSize()` 最大者 retract）＋`:779 scheduleRetract` | `v1/core/sched/scheduler.py:1145 _preempt_request`（從 running 尾端） | `srt/managers/schedule_batch.py:2770 retract_decode` | `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:676` 三種 policy（`kMAX_UTILIZATION` / `kGUARANTEED_NO_EVICT` / `kSTATIC_BATCH`） | `src/turbomind/engine/scheduler.cc:54 EvictingIterator` + `:1283 ReplayMemory` |
| 10 | batch 物件建構 | `csrc/scheduler/operations/forward.cpp:961`, `:1053 applyEventAndGenerateOp` | `SchedulerOutput`（`v1/core/sched/output.py`） | `srt/managers/schedule_batch.py`（`ScheduleBatch` → `ModelWorkerBatch`） | `_torch/pyexecutor/scheduler/scheduler.py:135 ScheduledRequests` | `src/turbomind/engine/scheduler.cc:1313 CommitResults` |
| 11 | 送 worker | `runtime/engine/event_loop.py` 取 `ExecutionPlan` | `v1/engine/core.py` EngineCore step | `srt/managers/tp_worker.py` | `_torch/pyexecutor/py_executor.py:3618 _prepare_and_schedule_batch` | 同程序：`src/turbomind/engine/model_executor.cc:147 Start` |
| 12 | 完成回收 | `csrc/scheduler/scheduler.cpp:421`（`erase_if` on `fsm::Finished`） | `v1/core/sched/scheduler.py:2041 finish_requests` | `srt/managers/schedule_batch.py`（`filter_batch`） | `_torch/pyexecutor/py_executor.py:7165 _pause_requests` 等 | `src/turbomind/engine/scheduler.cc:792 Finalize` / `:751 Release` |
| 13 | 前綴快取插入 | `csrc/cache/kv_cache_coordinator.cpp:359 CacheFullBlocks` | `v1/core/kv_cache_manager.py:596 cache_blocks` → `v1/core/block_pool.py:226 cache_full_blocks` | `srt/mem_cache/mamba_radix_cache.py:525 insert`（hybrid）/ `radix_cache.py` | `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` | `src/turbomind/engine/scheduler.cc:714 MarkProduced` → `src/turbomind/engine/prefix_trie.h:76 Insert` |
| 14 | KV event 發布 | `csrc/scheduler/scheduler.cpp:199 DrainKvEvents` | `v1/core/kv_cache_utils.py`（KV connector） | `srt/managers/scheduler_components/kv_events_publisher.py` | `cpp/tensorrt_llm/batch_manager/kvCacheEventManager.cpp` | `src/turbomind/engine/cache_registry.cc` |

---

## 2. 分歧點

### 2.1 prefill / decode 的順序 — 三種答案，其中一種是「這個問題不存在」

| engine | 一個 step 裡 |
|---|---|
| UnieInfer（`--enable-mixed-batch`） | decode 優先權 0、in-flight prefill 1、新請求 2 → **decode-first 混批** |
| UnieInfer（預設，未開旗標） | prefill 1、新請求 2、decode 3；**而且一旦排進任何 prefill，這一步的 decode 全部跳過** |
| vLLM | 恆混：先掃 `self.running`（其中同時有 decode 與 in-flight prefill chunk），再掃 `self.waiting` |
| SGLang | **互斥**：有 prefill 就只跑 prefill，沒有才跑 decode |
| TRT-LLM | 一次掃過 `activeRequests`，context 與 generation 同時收，只受 `maxNumTokens` / `maxBatchSize` 限制 |
| TurboMind | 沒有這個決策：`unified_decoder` 本來就同時吃 prompt 與 generation 的 token |

UnieInfer 未開旗標時的 `break`（`csrc/scheduler/operations/forward.cpp:1232`, `:1246`）
是五家裡最激進的：它不是「prefill 優先」，是**「這一步只做 prefill」**。SGLang 是同樣
的語意但寫在更高層（`srt/managers/scheduler.py:3071` 的 if/else）。vLLM 與 TRT-LLM 兩家
都沒有這個開關 —— 對它們而言混批是唯一模式。

**注意不要和另一件事搞混。** `AGENTS.local.md` 記的「vLLM 與 TurboMind decode-first、
TRT-LLM 與 SGLang prefill-first」講的是 **GDN linear attention kernel 內部**把一個混合
batch 切成 prefill 段與 decode 段的順序，不是這裡的排程優先權。同一組詞、兩件事，
兩張表要分開讀（kernel 側見 `30-prefill.md`）。

### 2.2 容量准入與批次組成：一個迴圈還是兩個

TRT-LLM 是唯一把這兩件事拆開的：

- `CapacityScheduler`（`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:708`）
  只回答「哪些請求的 KV 放得下」，並在放不下時依 policy 暫停或驅逐；
- `MicroBatchScheduler`（`cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp:347`）
  才從放得下的集合裡組出這一步的 batch，並做 chunk 切分；
- `SimpleScheduler`（`_torch/pyexecutor/scheduler/scheduler.py:422`）把兩者串起來。

好處是 policy 可換（三種 capacity policy 是 executor API 的一級選項）。其他四家把
「放得下嗎」和「要不要排進來」寫在同一個迴圈裡，換 policy 就得改迴圈。

TurboMind 用另一種切法：**required / optional 兩階段**
（`src/turbomind/engine/scheduler.cc:1105` / `:1219`），中間各有一次 `ReplayMemory`
（`:1283`）把記憶體意圖真正提交。第一階段配置「非做不可」的空間（含驅逐），第二階段
才從閒置記憶體撿「有空就做」的。這讓驅逐決策與配置決策可以先模擬再提交。

### 2.3 搶佔的受害者選擇 — 四種規則

| engine | 挑誰 | 依據 |
|---|---|---|
| UnieInfer | **token 數最多的那一個** | `csrc/scheduler/operations/forward.cpp:1277`（`max_element` on `TokenSize()`） |
| vLLM | running 的**最後一個**（最晚排進來的） | `v1/core/sched/scheduler.py:1145 _preempt_request` |
| SGLang | 依 `retraction_key` 排序後挑 | `srt/managers/schedule_batch.py:2839 retraction_key` |
| TRT-LLM | 由 capacity policy 決定，`kGUARANTEED_NO_EVICT` 根本不驅逐 | `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:676` |
| TurboMind | LRU-ish：`EvictingIterator` 依 cache 時間戳走 | `src/turbomind/engine/scheduler.cc:54` |

UnieInfer 挑最長的那一個，是四家裡唯一以「釋放最多記憶體」為目標而非以「最晚到達」
為目標的。這在長短請求混合的工作負載上會系統性地懲罰長請求。

### 2.4 hybrid 模型的前綴快取：折進 descent，還是事後補掛

這是目前已知差最多的一格，兩邊的程式碼結構正好相反。

**SGLang**（`srt/mem_cache/mamba_radix_cache.py:1068 _match_prefix_helper`）：
在 radix 下降的**過程中**維護 `best_last_node`，每經過一個 `mamba_value is not None`
的節點就更新一次，回傳的是**帶著 mamba 快照的最深節點**。KV 匹配與 recurrent-state
匹配是同一次走訪的兩個輸出。

**UnieInfer**（`csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp:117 Match`）：
先做完純 KV 的 `kv_prefix_cache_.Match`，再呼叫 `:132 augmentMatch` 補檢查。
`augmentMatch` 從 KV 終端節點（一個**完全不看 mamba 快照**選出來的節點）出發，用
`:237 FindLastMambaNode` 往**上**走找快照。找不到就

```cpp
match.device.last_node = root;   // hybrid_prefix_cache.cpp:163-164
match.host.last_node   = root;
```

—— **整個 KV 匹配被丟掉**。比 KV 終端更深一層的快照因此永遠看不見。

同一個功能、相反的結構，而且只有把兩邊擺在一起才看得出來。

### 2.5 chunk 切分策略

只有 TRT-LLM 把 chunk 切分寫成可選 policy
（`cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp:299 setCtxRequestsChunkSize`
分派到 `kEQUAL_PROGRESS` / `kFIRST_COME_FIRST_SERVED` / `kFORCE_CHUNK` 三個特化）。
`kEQUAL_PROGRESS` 讓所有 context 請求同步推進，其餘四家都等價於 FCFS：先來的請求先
把 token 預算吃滿。

vLLM 更極端 —— 它**沒有** chunk 的概念，chunk 是 `token_budget` 減到 0 的副產品
（`v1/core/sched/scheduler.py:416`，註解明說「There's no "decoding phase" nor "prefill
phase" in the scheduler」）。

### 2.6 TP 決定性

只有 UnieInfer 在排序裡明寫 TP 決定性的理由：

```cpp
// TP-determinism: tie-break on Id() so every rank schedules the same subset
// (a rank-varying op deadlocks NCCL).
```
（`csrc/scheduler/operations/forward.cpp:1152-1153`）

其他四家靠的是「scheduler 只有一份、結果廣播給所有 rank」的架構前提，所以不需要這條。
UnieInfer 的 C++ scheduler 是**每個 rank 各跑一份**，這是它獨有的約束，也解釋了為什麼
它的優先權函式必須是純函式（不能看 wall clock、不能看 rank-local 狀態）。

---

## 3. 結構差異（同功能、反結構）

| 功能 | 多數作法 | 例外 | 後果 |
|---|---|---|---|
| scheduler 語言 | Python（vLLM、SGLang） | UnieInfer、TurboMind 純 C++；TRT-LLM Python 殼包 C++ | Python 側 profiler 看不到 C++ scheduler 的成本，要用它自己的計時器 |
| scheduler 份數 | 單一份，廣播給 rank | UnieInfer：每 rank 一份，靠純函式保證一致 | 優先權函式不能引入任何 rank-local 資訊 |
| hybrid 前綴匹配 | 折進 descent（SGLang） | UnieInfer：事後 `augmentMatch` | 深一層的快照不可見，整個 KV 匹配被丟棄 |
| 容量與批次 | 同一迴圈 | TRT-LLM：兩個 scheduler | policy 可換 vs 迴圈要改 |
| 記憶體提交 | 邊排邊配 | TurboMind：先收集 intent，`ReplayMemory` 再提交 | 驅逐可以先模擬再定案 |
| 搶佔目標 | 最晚到達 | UnieInfer：token 最多 | 系統性懲罰長請求 |

---

## 4. 可量測候選（下一輪的輸入）

| 候選 | 量什麼 | 為什麼值得 |
|---|---|---|
| B1 每步 scheduler CPU 時間 | 在途請求 1 / 8 / 32 / 128 下，一次排程呼叫的耗時（三家實測） | C++ vs Python 的差距是這一章最直接的假設，但沒人量過 |
| B2 混批旗標 | `--enable-mixed-batch` 開 / 關對 TTFT 與 TPOT 的影響 | 關閉時是「這一步只做 prefill」，不是「prefill 優先」；代價可能遠大於預期 |
| B3 hybrid 前綴命中率 | `Qwen3.5-27B` 重複 prompt 下的實際命中率（UnieInfer vs SGLang） | 2.4 的結構差異預測 UnieInfer 會系統性漏掉；已知曾量到 0% vs 83.3% |
| B4 前綴查詢成本 | 單次 `Match` / `match_prefix` / `get_computed_blocks` 的耗時 vs 前綴長度 | radix 走訪是 per-request 成本，長 prompt 下可能不可忽略 |
| B5 搶佔行為 | 長短混合工作負載下，各家被搶佔的請求長度分布 | UnieInfer 挑最長的，其他家不是；影響尾延遲 |
| B6 chunk policy | TRT-LLM `kEQUAL_PROGRESS` vs FCFS 的 TTFT 分布 | 只有 TRT-LLM 有這個選項；值得知道它買到什麼 |

---

## 5. 未解格（`?`）

無。步驟表所有格子皆為 `file:line` 或說明性描述。

兩處只給了檔案而沒給行號，因為該步驟散在整個檔案而非單一函式：
`cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp`（TRT-LLM 的前綴查詢與插入）、
`srt/mem_cache/allocator/`（SGLang 的分頁配置）。要往下鑽時從這兩個位置開始。
