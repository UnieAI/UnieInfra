# 25 — 飽和 KV pool 下的准入與搶佔：五方橫向對照

版本見 `00-METHOD.md`。四個參照 checkout 的 pin 未動（`4ad14044b254` / `b61a06921ef0` /
`d8b67db26082` / `a1e567416831`，逐一確認過）。**UnieInfra 這一輪的行號對工作樹
`f162d993ea59`**，`00-METHOD.md` 記的 `4d35899ca734` 已前進；引用 UnieInfra 的行號時以本文為準。

範圍：**pool 被填滿之後**的行為 —— 誰還准進來、放不下時誰被踢出去、有沒有保留水位與
遲滯、啟動時擋不擋。不含排程結構本身，也不含快照配置。

### 與既有文件的分工（不重複）

`20-scheduler.md` 已經寫過三格，本文只引用不重寫：

- 步驟表 #4「容量准入（誰放得下）」與 #9「搶佔 / 驅逐」的**入口** `file:line`；
- §2.2「容量准入與批次組成：一個迴圈還是兩個」—— TRT-LLM 拆成
  `CapacityScheduler` + `MicroBatchScheduler`，TurboMind 拆成 required / optional 兩階段；
- §2.3「搶佔的受害者選擇 —— 四種規則」—— 五家挑誰的一句話對照。

`36-cache-tiering.md` §2 / §4（D20、D21）已經寫過遞迴狀態快照的配置池與配不到時的行為，
本文不重複那條路徑。

本文往下鑽的是那三格**沒有問**的四件事：

1. 准入到底 charge 多少「未來」需求（不是「誰放得下」，是「放得下什麼」）；
2. 搶佔在**什麼條件下觸發**（不是挑誰）；
3. 有沒有保留水位、有沒有防止 admit/preempt 來回擺盪的遲滯；
4. 啟動時擋不擋，並且報不報出可用的數字。

---

## 0. 一句話的結論

**五家沒有一家是靠 `max_num_seqs` 擋住過度准入的。擋住它的是「這個請求未來還要多少頁」
被問在哪一層 —— 而 UnieInfra 是五家裡唯一從不問這個問題的：它只問「這一個 chunk 加一個
decode token 放得下嗎」。**

| engine | 准入時 charge 的未來需求 | 搶佔的觸發條件 | 每步幾個受害者 |
|---|---|---|---|
| vLLM | **整條 prompt**（`scheduler_reserve_full_isl`，預設 True） | running loop 內**單一請求**配不到頁 | 迴圈，直到配得到 |
| SGLang | **prompt + `max_new_tokens` × 回饋係數 `new_token_ratio`** | `check_decode_mem()` 為 false | 迴圈，直到剩下的放得下（**至少一個**） |
| TRT-LLM（預設 `GUARANTEED_NO_EVICT`） | **prompt + `max_new_tokens`，charge 到完成** | **不會發生**（paused 集合恆空） | 0 |
| TRT-LLM（`MAX_UTILIZATION`） | 下一步（可選 two-step lookahead） | 單一請求配不到頁 | 迴圈，一次一個 |
| TurboMind | 只有目前 context，但**整批先模擬再提交** | 每個請求自己驅逐比它低優先權的 block | 迴圈，一次一個 block |
| **UnieInfra** | **這一個 chunk + 1 個 decode token** | **整步一個 op 都排不出來，且連續兩輪** | **1（每兩輪）** |

這張表直接對應觀察到的形狀。gemma-4-31B、C=32、pool 釘在 `#pages(active/cached/total):
7257-7395 / 7400`：median ITL 61.2 ms 對 vLLM 57.6（打平），**P99 ITL 1905 ms 對 60.6**
（31x，而 vLLM 的 P99 等於它自己的 median），throughput 219.6 對 456.2 tok/s；整個 run 只
有一次 `flat retract ... to unwedge the pool`。手動把 `--max-num-seqs` 降到 32、不改任何
程式碼，得到 340.4 tok/s（+55%）、P99 ITL 155 ms（12x 改善）、median ITL 56.8 ms（**低於**
vLLM）。同一個現象在樹內已有紀錄：`runtime/layers/attention/registry.py:998`。

三件事**一起**產生這個形狀，缺一不可：

1. **准入只 charge 一個 chunk**（`csrc/scheduler/operations/forward.cpp:211`），所以 452 個
   請求全部被放進來 —— 唯一的准入上限是 request slot 數
   （`csrc/scheduler/operations/forward.cpp:383 AvailableSlots`），而那個數就是
   `max_num_seqs`。
2. **decode 逐請求被擋**（`csrc/scheduler/operations/forward.cpp:247`）。pool 100% 滿時，
   任何 tail page 用完的請求都拿不到新頁，只能等別人結束 —— P99 就是「等到下一個完成」
   的時間。
3. **搶佔永遠不觸發**。觸發條件是 `made_progress == false`，而 `made_progress` 定義為
   `!ops.empty()`（`csrc/scheduler/operations/forward.cpp:1402`）。只要還有**一個**請求
   在自己的 tail page 裡有位子，這一步就排得出 op，`flat_starved_rounds_` 被清零
   （`csrc/scheduler/operations/forward.cpp:331`），retract 不會發生。所以是「大家一起
   卡住」而不是「少數被搶佔、其餘跑滿」。

**而且 —— vLLM 的啟動檢查救不了 gemma-4。** `AGENTS.local.md` 引為典範的那條訊息
（`max_num_seqs (512) exceeds available Mamba cache blocks (452)`）住在
`config/compilation.py:1457`，而它的守衛條件是 `kv_cache_config.has_mamba_layers` **且**
`cudagraph_mode.has_full_cudagraphs()`（`config/compilation.py:1449`）。gemma-4 沒有 mamba
層，這條**根本不會觸發**。vLLM 在這個模型上 P99 等於 median，靠的是 runtime 的
`scheduler_reserve_full_isl`（預設 `True`，`config/scheduler.py:140`），不是啟動守衛。

**要抄的是准入，不是啟動守衛。** 這是本文最重要的一句。

---

## 1. 步驟表

`—` = 這個引擎沒有這一步。**沒有對應本身就是發現。**

| # | 步驟 | UnieInfra | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 1 | 准入入口 | `csrc/scheduler/operations/forward.cpp:380 schedulePrefillFirstChunk` | `v1/core/sched/scheduler.py:640`（waiting 迴圈） | `srt/managers/schedule_policy.py:1187 add_one_req` | `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:218 impl` | `src/turbomind/engine/scheduler.cc:1105 RunRequiredAdmission` |
| 2 | 併發上限比什麼 | request slot 數 = `max_num_seqs`（`csrc/scheduler/operations/forward.cpp:383 AvailableSlots`）＋ batch 上限 `csrc/scheduler/operations/forward.cpp:1325` | `len(running) >= max_num_seqs`（`v1/core/sched/scheduler.py:644`） | `srt/managers/scheduler.py:3098 get_num_allocatable_reqs` = `min(pp_max_micro_batch_size - running, req_to_token_pool.available_size())` | `maxNumRequests`（`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:272`） | **無 batch 上限**；`max_batch_size` 只擋 gateway 取件（`src/turbomind/engine/engine.cc:794`） |
| 3 | 准入 charge 的未來需求 | **chunk + 1 decode token**（`csrc/scheduler/operations/forward.cpp:208`） | **整條 prompt**（`v1/core/kv_cache_manager.py:374 full_num_tokens`） | **prompt + `max_new` + 1 page**（`srt/managers/schedule_policy.py:1215`），running 側再乘 ratio | **到完成**（`cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:3628 getRemainingBlocksToCompletion`） | **無預測**；`max_seq_len` 算了但 scheduler 不讀（`src/turbomind/engine/engine.cc:407`） |
| 4 | 對 running 集合的未來需求 | `flat_reserved_pages_`：每個 running 請求 **1 個 decode token** 的頁（`csrc/scheduler/scheduler.h:342 flatFreeBudget`） | `_inflight_prefill_reserved_blocks`（`v1/core/sched/scheduler.py:908`） | `rem_total_token_offset` 由 `srt/managers/schedule_policy.py:640 _get_running_request_total_token_offset` 播種 | `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:284 enoughAvailableBlocks` 先替所有 in-flight 保留到完成 | — |
| 5 | 未來需求的預測函式 | `csrc/scheduler/request.h:101 RemainingOutputTokens`（**定義了、零呼叫者**） | `v1/core/kv_cache_manager.py:372 full_sequence_must_fit` | `srt/managers/scheduler_components/new_token_ratio_tracker.py:14 NewTokenRatioTracker` | `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:3668 numTotalBlocksPerBeam`（`mPromptLen + mMaxNewTokens`） | — |
| 6 | 保留水位 | **無** | `config/scheduler.py:146 watermark`（**預設 0.0 = 關**），換算 `v1/core/kv_cache_manager.py:163 watermark_blocks` | 無全域水位；每請求 1 page（`srt/managers/schedule_policy.py:849`） | **無**（`cpp/tensorrt_llm/batch_manager/scheduledBlocksManager.h:52` 是裸 `>`） | 無數值水位；`pass.floor` 時間戳帶（`src/turbomind/engine/scheduler.cc:1087`） |
| 7 | 水位套用範圍 | — | 只對 WAITING/PREEMPTED 且已有請求被排（`v1/core/kv_cache_manager.py:365`） | — | — | optional 階段只能碰 `timestamp < floor` 的 inactive block（`src/turbomind/engine/scheduler.cc:1235`） |
| 8 | 遲滯 / 防抖 | **無** | 該步只要發生過搶佔就**整段跳過** waiting 迴圈（`v1/core/sched/scheduler.py:637`） | `new_token_ratio` 回饋環：retract 後拉高（`srt/managers/scheduler.py:3472`）、無 retract 的每一步衰減（`srt/managers/scheduler.py:3500`） | `GUARANTEED_NO_EVICT` 使搖擺結構上不可能（`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:432`） | 兩階段：optional 只吃 phase 1 提交後的剩餘（`src/turbomind/engine/scheduler.cc:1235`） |
| 9 | 搶佔觸發條件 | `csrc/scheduler/operations/forward.cpp:330`：`!made_progress && flatPoolWedged`，且 `:334` 連續 2 輪 | `v1/core/sched/scheduler.py:580`：`allocate_slots` 回 None | `srt/managers/scheduler.py:3437`：`not batch.check_decode_mem()` | `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:561`：`trySchedulingRequestMaxUtilization` 失敗 | `src/turbomind/engine/scheduler.cc:1173`：`Allocate` 失敗 |
| 10 | 「配不到」的判定式 | `csrc/scheduler/operations/forward.cpp:247`（per-request，靜默 defer） | `v1/core/kv_cache_manager.py:420 required_blocks > available_blocks` | `srt/managers/schedule_batch.py:2768 available_size() >= num_tokens` | `cpp/tensorrt_llm/batch_manager/scheduledBlocksManager.h:52 needed > availableBlocks` | `src/turbomind/memory/object.h:99 ScratchAllocator::Allocate` 回 false |
| 11 | 受害者選擇 | token 數最多（`csrc/scheduler/operations/forward.cpp:345`） | running 最後一個（`v1/core/sched/scheduler.py:572`） | **輸出最少者**（`srt/managers/schedule_batch.py:2834 length_key`，`(len(output_ids), -len(origin_input_ids))` 降冪後 `pop()`） | 最後一個「已開始」的（`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:563`） | 時間戳最舊（`src/turbomind/engine/block.cc:52`） |
| 12 | 每步幾個受害者 | 1（且每 2 個 wedged 輪才一次，`csrc/scheduler/operations/forward.cpp:334`） | 迴圈直到配得到（`v1/core/sched/scheduler.py:534`） | 迴圈直到剩下的放得下；**至少一個**（`srt/managers/schedule_batch.py:2778 first_iter`） | 迴圈，每次一個（`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:581`） | 迴圈，每次一個 **block**（不是一個請求，`src/turbomind/engine/scheduler.cc:1172`） |
| 13 | 受害者 KV 去哪 | **全丟＋重算**（`csrc/fsm/forward_events.cpp:1063 applyRetract`：`RebasePrefill` + `FreeRequest`）；radix 路徑則寫回 host（`csrc/scheduler/operations/forward.cpp:882 newRetractOperation`） | **全丟＋重算**；v1 **沒有 swap-to-CPU**（`v1/core/sched/scheduler.py:1154`） | **全丟＋重算**，且**不**插回 radix（`srt/mem_cache/common.py:132 release_kv_cache` 以 `is_insert=False`）；只有 PD-decode 會 offload（`srt/managers/schedule_batch.py:1894`） | pause = 釋放 KV 重算（`cpp/tensorrt_llm/batch_manager/utils/inflightBatchingUtils.cpp:90 terminateRequest`）；V2 才 offload 到 host（`_torch/pyexecutor/kv_cache_manager_v2.py:2454 suspend_request`） | **全丟＋重算**，無 host 層（`src/turbomind/engine/scheduler.cc:1299 Deallocate`）；有 checkpoint 時從快照 D2D 復原（`src/turbomind/engine/scheduler.cc:628`） |
| 14 | 受害者的 prefix 狀態 | 生成 token 併入 prompt 重新排隊為 `Submitted`（`csrc/fsm/forward_events.cpp:1069 RebasePrefill`） | 放回 waiting **佇列頭**，`num_computed_tokens = 0`（`v1/core/sched/scheduler.py:1158`） | `srt/managers/schedule_batch.py:1645 reset_for_retract` 清掉 `prefix_indices` / `last_node` | `LlmRequest::pause`（實作在 `cpp/include/`，不在 checkout 內 → `?`） | `src/turbomind/engine/scheduler.cc:514 PlanResume` 從倖存的 valid block 重算 `resume_len` |
| 15 | 全滿且無人可搶時 | `csrc/scheduler/operations/forward.cpp:363 AbortEvent`（終結 head-of-line deferred 非持有者） | `break`，下一步再試（不失敗請求） | 中止最後一個（`srt/managers/schedule_batch.py:2801 FINISH_ABORT`） | `GUARANTEED_NO_EVICT` 不會到這；V2 有死結中止（`_torch/pyexecutor/scheduler/scheduler_v2.py:439`） | `src/turbomind/engine/engine.cc:575 FailStalledHeadOfLine` → `src/turbomind/engine/engine.cc:605` |
| 16 | 啟動檢查：批次 vs 池 | **無**（`runtime/utils/server_args.py:845 validate` 只檢查 `attn_dp_size` 與 CP） | `config/compilation.py:1457`（**只在 mamba + full cudagraph 時**） | `srt/mem_cache/kv_cache_configurator.py:1719 resolve_max_num_reqs`（**必經**） | **無**（`_torch/pyexecutor/_util.py:943` 註解自承缺這條） | **無**（`src/turbomind/turbomind.cc:299` 只從記憶體比例算池） |
| 17 | 啟動時報出可用值 | — | 是：`lower max_num_seqs to at most {num_blocks}` | 是：`srt/mem_cache/kv_cache_configurator.py:1766`「reduced from the requested %d to %d」 | 只有 attention window 有（`_torch/pyexecutor/resource_manager.py:2217`），且是靜默夾住不是拒絕 | — |
| 18 | 併發上限由誰決定 | **operator**（`runtime/engine/event_loop.py:281 per_rank_max_batch`，零 clamp） | **operator**（`engine/arg_utils.py:2673` 只跟 `max_num_batched_tokens` 比，且僅在未設時） | **引擎推導**：`min(requested, token_capacity // 2, max_mamba_cache_size // ratio)`（`srt/mem_cache/kv_cache_configurator.py:1729`、`:1751`） | **operator**（`_torch/pyexecutor/_util.py:2910 scheduler_capacity`） | **operator**，預設查 GPU 型號表（`utils.py:394 get_max_batch_size`） |

---

## 2. 分歧點

### 2.1 准入 charge 多遠 —— 五個答案，而且它們真的不同意

這是本輪最大的一格，**五家給了五個答案，沒有一個是另一個的嚴格改良**。不要平均掉。

| engine | charge 到哪 | 依據 | 買到什麼 / 付出什麼 |
|---|---|---|---|
| UnieInfra | 這一個 chunk + 1 token | `csrc/scheduler/operations/forward.cpp:208` | 併發最高；尾延遲不設防 |
| vLLM | 整條 prompt | `v1/core/kv_cache_manager.py:374`（`min(request.num_tokens, max_model_len)`） | **不需要任何靜態併發數字**；輸出很長時仍可能後續配不到 → 才有搶佔 |
| SGLang | prompt + `max_new` × `new_token_ratio` | `srt/managers/schedule_policy.py:641` | 可調的保守度；係數猜錯就會 retract（然後係數自動升高） |
| TRT-LLM | prompt + `max_new`，到完成 | `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:3668` | **結構上不可能搶佔**；請求沒給 `max_tokens` 就退化成整個 window，併發被壓死 |
| TurboMind | 不預測 | — | 用「整批先模擬再提交」取代預測（見 §2.3） |

三點必須說清楚：

**(a) vLLM charge 的是 prompt，不是 prompt + 輸出。**
`full_num_tokens = min(request.num_tokens, self.max_model_len)`
（`v1/core/kv_cache_manager.py:374`），`num_tokens` 是 prompt 加**已生成**的 token，不含
`max_tokens`。所以 vLLM 是「保證第一次 prefill 一定放得下整條 prompt」，不是「保證跑到
完成」。它的搶佔路徑因此仍然是活的，而且是常態機制。

**(b) TRT-LLM 的預設就是 charge 到完成。**
`llmapi/llm_args.py:3392` 的 `capacity_scheduler_policy` 預設是
`CapacitySchedulerPolicy.GUARANTEED_NO_EVICT`。它先替**每一個** in-flight 請求保留到完成
（`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:284`），再考慮任何新請求
（`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:397`），第一個放不下就 `break`（head-of-line，不做 backfill，
`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:426`）。它的 paused 集合是硬寫死的空 vector
（`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:432`）—— **搶佔在這個 policy 下不
是「不常發生」，是不存在。**

**(c) SGLang 的係數是一個回饋環，不是一個常數。**
`new_token_ratio` 從 `0.7 × schedule_conservativeness` 起
（`srt/managers/scheduler_components/new_token_ratio_tracker.py:22`），下限是起始值的
0.14 倍（`:27`），每一個沒發生 retract 的 decode step 線性衰減一格
（`:34 decay_step`，600 步走完全程，`srt/environ.py:406`），而 retract 一發生就用實測值
重估並拉高（`srt/managers/scheduler.py:3472` ← `:41
estimate_new_token_ratio_after_retract`）。**這是五家裡唯一一個會自己學習保守度的准入。**
它的 help text 也直說了用途：`--schedule-conservativeness`「Use a larger value if you see
requests being retracted frequently」（`srt/server_args.py:876`）。

### 2.1b 五格在本引擎上的實測（2026-08-14）

上表是各家的原始碼；這一格是**把其中四格移植進來、在同一條件下量到的數字**。
Qwen3.5-27B、單張 RTX PRO 6000、InferenceX 隨機形狀 1024 in / 1024 out、
C=128、256 個請求另加 256 個暖機丟棄、GMU 0.92、`--max-model-len 16384`。
兩邊實際達成的前綴重用 32–42%（不是 0%，chat template 的共同前綴加上 128-token
的頁大小），所以這不是「無快取」對照。

| 准入 charge 到哪 | 來源 | tok/s | 總時長 | P99 TPOT | 平均同時解碼數 |
|---|---|---|---|---|---|
| 這一個 chunk + 1 token | 原本 | 210.69 | 1244 s | — | — |
| 整條 prompt | vLLM | 168.55 | 1555 s | 1365 ms | 15.7 |
| + `max_new × 自適應係數` | SGLang | 360.95 | 726 s | 529 ms | 31.3 |
| **+ `max_new` 全額** | **TRT-LLM** | **848.44** | **309 s** | **94.67 ms** | **68.9** |
| （參考）vLLM 引擎本身 | — | 1035.84 | 253 s | 139.46 ms | 99.3 |

**四格的排序與各家的保守程度一致，而且最保守的贏最多。** 機制與 §2.1 的預測相符：
charge 到完成讓搶佔在結構上不必發生，於是沒有重算成本、沒有遲滯震盪 —— P99 TPOT
從 1365 收到 94.67，比 vLLM 的 139.46 更好。

**代價也如預告**：隨機形狀的 TTFT 中位數變差 39%（9695 → 13516 ms），因為請求要等
完整的輸出預算放得下才准進來。但在 agent 形狀上**反而更快**（2815 → 1808 ms，−36%），
因為那裡輸出中位數 806 對提示 127k 幾乎免費 —— 同一個策略在兩個形狀上的代價差約
150 倍。agent 形狀 C=32 的 goodput 也從 0.39 升到 0.41。

**兩條走不通的路，都是實測走完的：**

1. **搶佔（R3）不是答案。** 觸發條件放寬後，撤回會逼請求重新 prefill，付出多於節省：
   兩階段撤回實測 140.42 tok/s，比什麼都不做的 210.69 還差 45%。
2. **per-request 搶佔在本架構下需要兩階段撤回。** 受害者必須不欠任何 outstanding
   forward result，而 overlap 深度 1 下幾乎人人都欠著。把過濾放寬成「不欠**本步**的」
   會崩：`FSM transition invalid: event=ExtendResultEvent; state=Retracted`。原本那道
   「整個 map 非空就不搶佔」不是保守，它就是該不變量的正確實作。

實作在 `SchedulerConfig::reserve_full_prefill_at_admission` /
`reserve_generation_at_admission` / `new_token_ratio_*`，由 server arg
`--admission-reserve {chunk,prompt,generation-adaptive,generation-full}` 選擇，
預設 `generation-full`。C++ 測試 `AdmissionChunkSuite` 與
`AdmissionGenerationFullSuite` 釘住「同一組請求、同一個池子，chunk 收 3 個、
generation-full 收 1 個」。

### 2.2 搶佔的觸發條件 —— 我們是孤例，而且差在「整步」與「單一請求」

`20-scheduler.md` §2.3 已經寫過**挑誰**。這裡是**何時**，而這一格差得更遠。

| engine | 觸發的粒度 |
|---|---|
| vLLM | **單一請求**：`allocate_slots` 對某個 running 請求回 None，就在原地搶佔（`v1/core/sched/scheduler.py:534` 的 `while True`） |
| SGLang | **整批**：`check_decode_mem()` 說下一個 decode step 放不下（`srt/managers/scheduler.py:3437`） |
| TRT-LLM `MAX_UTILIZATION` | **單一請求**：`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:561` |
| TurboMind | **單一 block**：某個 block 配不到就驅逐一個比它舊的（`src/turbomind/engine/scheduler.cc:1172`） |
| **UnieInfra** | **整步 + 兩輪遲滯**：`!ops.empty()` 就算有進展（`csrc/scheduler/operations/forward.cpp:1402`），而且要**連續兩輪**沒進展（`:334`） |

UnieInfra 的條件比 SGLang 的「整批」還嚴一級：SGLang 問的是「下一個 decode step 放得下
嗎」，我們問的是「這一步有沒有排出任何一個 op」。這兩者在飽和 pool 上的差別是決定性的
—— pool 100% 滿時，仍有大量請求的 tail page 沒用完，它們照常被排進來，`ops` 非空，
`flat_starved_rounds_` 歸零。**搶佔的觸發器被一群還跑得動的請求擋在門外。**

`flatPoolWedged` 的三個前提（`csrc/scheduler/operations/forward.cpp:312`、`:314`、`:318`）
還要求 **`pending_forward_results_` 與所有 cache op ledger 全空**，也就是必須是一個
完全靜止的 step。這是為了讓 retract 的安全性論證成立（`:348-352` 的註解與 assert），
但代價是觸發窗口窄到一個 run 只命中一次。

### 2.3 遲滯 —— 三種機制，我們一種都沒有

| engine | 機制 | 引用 |
|---|---|---|
| vLLM | **這一步發生過搶佔，就整段跳過 waiting 迴圈** | `v1/core/sched/scheduler.py:637 if not preempted_reqs` |
| SGLang | **回饋係數**：retract 拉高保守度，之後 600 步緩慢衰減回去 | `srt/managers/scheduler.py:3472` / `:3500` |
| TRT-LLM | 預設 policy 讓搖擺**結構上不可能** | `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:432` |
| TurboMind | **時間戳帶**：optional 階段只能碰 `timestamp < pass.floor` 的 inactive block | `src/turbomind/engine/scheduler.cc:1087` / `:1235` |
| UnieInfra | 連續兩輪 wedged 才 retract（`csrc/scheduler/operations/forward.cpp:334`）—— 這是**降低誤報**，不是防搖擺；retract 完的下一步，新請求照樣可以立刻被准入 | — |

vLLM 的那一行是四者裡最便宜也最直接的：搶佔與准入在同一個 step 裡互斥。SGLang 的回饋環
是最貴也最聰明的。TurboMind 的做法值得單獨看一眼，因為它把「保留」表達成**排序上的一段
區間**而不是一個數字：`pass.floor` 是這一輪開始時的時間戳，任何本輪被排到的請求持有的
block 都被重新蓋章到 `floor` 以上（`src/turbomind/engine/scheduler.cc:1085`），於是
optional 階段的驅逐游標永遠碰不到它們。**沒有任何一個 byte 被保留，但保護是完全的。**

至於水位：只有 vLLM 有一個真正的 free-block 水位（`config/scheduler.py:146`），而且
**預設是 0.0，也就是關的**，並且只對 WAITING/PREEMPTED 狀態套用
（`v1/core/kv_cache_manager.py:365`）—— 已在跑的 decode 從不付這筆錢。這個「水位只擋新
來的，不擋已在跑的」的不對稱，是四家水位設計裡唯一一個明說出來的。

### 2.4 「先模擬再提交」是 TurboMind 獨有的，而且比水位更強

TurboMind 的 admission 不是比對數字，而是**對一個假的 allocator 做一次真的配置**：
`ScratchAllocator`（`src/turbomind/memory/object.h:86`）持有 committed allocator 的容量
副本，`Allocate` / `Evict` 只動副本並把動作記進 journal
（`src/turbomind/engine/scheduler.cc:1117`）。每個請求成功後才把 journal 前綴標成已提交
（`src/turbomind/engine/scheduler.cc:1193`），失敗就把 journal 截回上一個完整請求
（`src/turbomind/engine/scheduler.cc:1215`）。真正的配置只發生在
`ReplayMemory`（`src/turbomind/engine/scheduler.cc:1283`），而且那裡的
`TM_CHECK(c.allocation.a)`（`:1303`）是**斷言**不是檢查 —— 註解直接寫
`// admission guarantees capacity`。

這解決的問題和水位不同：水位是「留一點餘裕免得算錯」，模擬是「不必算，直接試」。
對我們這種多群組、每群組 page size 不同、還有 SWA 滑動回收的 pool 而言，
`BlocksNeededFor` 這類解析式估算正是最容易算錯的地方（`csrc/cache/kv_cache_coordinator.cpp:296`
要把六個 group 的需求加起來，`csrc/scheduler/operations/forward.cpp:113 FlatSlideCredit`
還要把 SWA 能回收的頁加回去）。

### 2.5 啟動守衛：只有 SGLang 是必經的，而 vLLM 那條救不了 gemma-4

| engine | 有沒有 | 涵蓋範圍 | 報不報出可用值 |
|---|---|---|---|
| vLLM | 有 | **只在 mamba 層 + full cudagraph**（`config/compilation.py:1449`） | 是 |
| SGLang | 有 | **必經**：`resolve_max_num_reqs` 在池定案後一律跑（`srt/mem_cache/kv_cache_configurator.py:1719`） | 是（`:1766`，且 mamba 夾住時另有一則 `:1740`） |
| TRT-LLM | 無 | `_torch/pyexecutor/_util.py:943` 的註解自承「If not able to allocate ... the max batch size should be adjusted」，然後只是把估算工作量抬高 | — |
| TurboMind | 無 | 完全沒有；`max_batch_size` 的預設是查 GPU 型號表（`utils.py:403`） | — |
| UnieInfra | 無 | `runtime/utils/server_args.py:845 validate` 只檢查 `max_num_seqs >= attn_dp_size`（`:847`）與 CP（`:854`） | — |

SGLang 的那三行是本輪**最直接可移植**的東西：

```python
max_num_reqs = min(requested_per_worker, token_capacity // 2)   # kv_cache_configurator.py:1729
mamba_cap    = get_schedule().max_mamba_cache_size // ratio      # kv_cache_configurator.py:1737
max_num_reqs = min(max_num_reqs, mamba_cap)                      # kv_cache_configurator.py:1751
```

它不拒絕啟動，它**夾住並印出前後兩個數字**。這正好對上「operator 沒有辦法知道那個數字是
30」這半個問題。

但要誠實地說出它的極限：**`token_capacity // 2` 換到我們身上擋不住 452。** 我們的
`max_total_num_tokens` 定義是 `usable_pages * layout.block_size`
（`runtime/configs/hybrid_cache_plan.py:541`），也就是「page id 列數 × 每頁 token 數」，
不是「序列 token 數」。gemma-4 的六群組幾何是 5 個 sliding group（page 16）+ 1 個 full
group（page 32），一個序列 token 吃 5/16 + 1/32 = 0.34375 個 page id，所以 7400 個 id 的
真實容量約 21,500 個序列 token，而 `max_total_num_tokens` 會報 118,400 —— **高估 5.5 倍**
（推導見 `runtime/layers/attention/registry.py:976`）。`118400 // 2 = 59200 > 452`，檢查
會通過。**任何寫在這個數字上的守衛都會通過**，這就是為什麼現在的 452 沒有被擋下來的
機制性原因。

### 2.6 兩家把資料算好了卻沒有人讀 —— 而其中一家是我們

這一格是本輪最意外的發現，而且兩家的形狀一模一樣。

**UnieInfra。** `csrc/scheduler/request.h:101 RemainingOutputTokens()` 的 docstring 就是
本文結論本身：

> Used to reserve KV for requests that are already decoding so a new arrival cannot be
> admitted into pages they will need.

它在整棵 `csrc/` 裡**沒有任何呼叫者**。而且資料鏈是接好的：
`RequestSpec::max_new_tokens`（`csrc/scheduler/request_spec.h:45`）已經有 nanobind 綁定
（`bindings/python_module.cpp:279`），`Request` 的建構子也讀它
（`csrc/scheduler/request.cpp:37`）。Python 側 `runtime/engine/scheduler_utils.py:164
make_spec` 的 docstring 甚至把失效模式整段寫完了：

> without it, admission can only look at pages free right now, over-admits, and something
> already running has to be retracted -- losing its whole prefill.

而唯一的呼叫點 `runtime/engine/request_handler.py:271` **沒有傳 `max_new_tokens`**，
所以它一律是 0，`RemainingOutputTokens()` 一律回 0。另有一處 docstring
（`runtime/engine/event_loop.py:1881`）寫著「the C++ scheduler has no notion of
`max_new_tokens` at all」—— 那句話在欄位加進來之後就過期了。

**TurboMind。** 完全一樣的形狀：`Sequence::max_seq_len` 在
`src/turbomind/engine/engine.cc:396` 算成 `min(prompt_len + max_new_tokens, session_len)`
並存進 `src/turbomind/engine/engine.cc:407`，而 `src/turbomind/engine/scheduler.cc` 裡 `grep max_seq_len` 零
命中。TurboMind 是刻意的（它用整批模擬取代預測）；我們不是。

另有一個純粹的死宣告：`csrc/scheduler/scheduler.h:158 reservedPagesForRunning()` 宣告了
但整棵樹沒有定義也沒有呼叫。

---

## 3. 結構差異（同功能、反結構）

| 功能 | 多數作法 | 例外 | 後果 |
|---|---|---|---|
| 准入問的問題 | 「這個請求的未來需求放得下嗎」（vLLM / SGLang / TRT-LLM 三種深度） | **UnieInfra：「這一個 chunk 放得下嗎」**；TurboMind：不問，改成整批模擬 | 452 個請求全被准入到一個容得下 30 個的池 |
| 搶佔的觸發粒度 | 單一請求 / 單一 block（vLLM、TRT-LLM、TurboMind）或整批（SGLang） | **UnieInfra：整步 0 op 且連續兩輪** | 一個 run 只觸發一次；卡住的是全體而非少數 |
| 遲滯 | 都有（跳過准入 / 回饋係數 / 結構性 / 時間戳帶） | **UnieInfra：無** | 目前無害（因為搶佔本來就不發生），做了 §6 R2 之後會變成必要 |
| 容量檢查的形式 | 解析式估算（四家） | TurboMind：對容量副本做真配置，再 replay 提交 | 多群組幾何下，解析式是最容易算錯的地方 |
| 併發上限 | operator 設（vLLM、TRT-LLM、TurboMind、UnieInfra） | **SGLang：從池容量推導並印出降幅** | 只有 SGLang 的 operator 不必猜這個數字 |
| 受害者 KV | 全丟重算（五家的預設路徑都是） | TRT-LLM V2 / SGLang PD-decode：offload 到 host | 我們的 radix 路徑寫回 host，flat 路徑不寫 —— 同一個引擎內部就不一致 |
| 「未來需求」的欄位 | 有預測的三家都真的讀它 | **UnieInfra 與 TurboMind：算好了不讀** | TurboMind 是設計選擇；我們是未完成的工作 |

---

## 4. 可量測候選（下一輪的輸入）

| 候選 | 量什麼 | 為什麼值得 |
|---|---|---|
| A1 准入深度 A/B | §6 R1（整條 prompt）開 / 關，量 P99 ITL、throughput、resident 序列數 | 這是本文預測效果最大的一項，且不需要任何靜態數字；預期把 P99 從 1905 ms 拉到 155 ms 附近（等於手調 `--max-num-seqs 32` 的效果） |
| A2 `max_num_seqs` 掃描 | 8 / 16 / 32 / 64 / 160 / 452，量 throughput 與 P99 | 建立「真實容量是 30」的曲線，並驗證 A1 之後這條曲線變平（自我限制的定義） |
| A3 retract 觸發率 | 各家在同一工作負載下的搶佔次數 / 秒 | 我們是 1 次 / run，vLLM 與 SGLang 有計數器可讀；差幾個數量級本身就是結論 |
| A4 `token_capacity` 單位 | 用 `num_device_pages / BlocksNeededFor(1 token)` 算出的真實序列 token 容量，對照 `max_total_num_tokens` | 5.5x 的高估是推導出來的（§2.5），量一次確認 |
| A5 SGLang 回饋係數 | `new_token_ratio` 在飽和 pool 上的軌跡 | 我們若要做 §6 R5，係數法與硬上限法的差別只有量了才知道 |
| A6 受害者選擇 | 長短混合負載下，`TokenSize()` 最大 vs SGLang 的「輸出最少」的尾延遲分布 | 承 `20-scheduler.md` B5；A1 之後搶佔變罕見，這一項的優先度會下降 |

---

## 5. 未解格（`?`）

步驟表沒有 `?`。三處是**checkout 範圍外**，不是沒讀：

- `LlmRequest::pause` 的實作在 `cpp/include/tensorrt_llm/batch_manager/llmRequest.h`，
  不在 TRT-LLM 的 sparse checkout 內（步驟表 #14）。同理
  `schedulingHasFreeBlocks` 的函式體。要往下鑽時先
  `git -C legacy/third_party/TensorRT-LLM sparse-checkout add cpp/include`，**不要 fetch**。
- TurboMind 的 `cache_chunk_size` 在 `src/turbomind/engine/engine_config.h:25` 宣告、由
  Python 傳入（`turbomind/turbomind.py:250`），但 C++ 側零讀取。是刻意退役還是接線殘留
  無法從原始碼判定 → `?`。
- SGLang 這棵樹與 upstream 分歧很大：`init_new_token_ratio` / `base_min_new_token_ratio`
  / `new_token_ratio_decay` 三個 `ServerArgs` 欄位在 `b61a06921ef0` 上**不存在**，等價物
  是 `srt/environ.py:404-406` 的三個環境變數加 `NewTokenRatioTracker` 的欄位。引用舊名
  的文獻對不上這個 pin。

---

## 6. 最小的改動：吸收四家最好的部分

排序依「淨價值 ÷ 成本」，不是依重要性。

### 有一家明顯較好，但不是同一家在兩個問題上都較好

**准入這一題，vLLM 明顯較好。** 理由不是它保留得最多（TRT-LLM 保留得更多），而是它的
保留是**動態的、per-request 的、不需要任何靜態併發數字**：
`scheduler_reserve_full_isl`（`config/scheduler.py:140`）把 `max_num_seqs` 從「一個必須
猜對的數字」降級成「一個上限」。這正好對上我們的病症 —— 我們的問題不是 452 這個數字錯，
是**沒有任何東西在把它換算成頁**。SGLang 的 `resolve_max_num_reqs` 解決的是同一個問題的
另一半（可觀測性與保底），但 `token_capacity // 2` 是啟發式，換到我們的單位上擋不住
（§2.5）。

**「配不到位子時怎麼辦」這一題，TRT-LLM 明顯較好**：`GUARANTEED_NO_EVICT` 讓搶佔在結構
上不可能發生（`cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp:432`），沒有搶佔就
沒有遲滯問題、沒有受害者選擇問題、沒有重算成本。代價是併發被 `max_tokens` 綁死，而請求
不給 `max_tokens` 時退化成整個 context window。

**真正的分歧、不要平均掉的那一格：未來需求要 charge 到哪裡。** chunk（我們）/ 整條
prompt（vLLM）/ prompt + max_new × 回饋係數（SGLang）/ prompt + max_new（TRT-LLM）/
不 charge 改用整批模擬（TurboMind）。這五個答案是五種不同的併發-尾延遲取捨，沒有一個
是另一個的嚴格改良。**建議先落在 vLLM 那一格**（R1），量到 P99 收斂之後再決定要不要
往 SGLang 或 TRT-LLM 移動。

---

### R1 — 准入時 charge 整條 prompt（抄 vLLM `scheduler_reserve_full_isl`）

- **檔案**：`csrc/scheduler/operations/forward.cpp:199 flatAdmitFirstChunk`
- **改動**：在既有的 chunk 檢查（`:211`）**之前**多一次比較 ——
  `coordinator_.BlocksNeededFor(request->PrefillSize() + decode_reserve_tokens)`
  對 `flatFreeBudget(request->Id())`；放不下就 `return std::nullopt`，維持 `Submitted`
  且不配任何頁。這正是 vLLM `v1/core/kv_cache_manager.py:372-387` 的形狀，連
  「先檢查整條、再走原本的逐塊配置」的順序都一樣。
- **成本**：每個新請求多一次 `BlocksNeededFor`（純算術，per group O(1)，
  `csrc/cache/kv_cache_coordinator.cpp:305`）。沒有新狀態、沒有新事件、沒有改 FSM。
- **效果**：**併發自我限制。** 452 不再有害 —— 第 31 條 prompt 放不下就不會進來，也就
  不會有 452 個請求擠在一個容得下 30 個的池裡輪流卡住。這一項單獨就應該重現手調
  `--max-num-seqs 32` 的 +55%。
- **必須配套的出口**：prompt 大於整個池時會永遠不被准入。既有的
  `flatTerminalizeOom`（`csrc/scheduler/operations/forward.cpp:284`）已經處理這件事
  （`:465` 的 `minimum_completion_pages > capacity`），但目前只在 `enable_flatkv_pd`
  且 role 為 P/D 時開（`:286` 的 `_assert`）。要把它擴到 `Role::kFused`。
- **風險**：SWA 群組的滑動回收（`csrc/scheduler/operations/forward.cpp:113
  FlatSlideCredit`）讓「整條 prompt 需要的頁數」在 sliding group 上不是單調的；
  `BlocksNeededFor(fresh, N)` 對 SWA group 應該已經被 window 夾住，但這一點要在
  `csrc/cache/kv_cache_manager.h:138` 讀清楚再動。

### R2 — 啟動時把 `max_num_seqs` 夾到池撐得住的數字並印出（抄 SGLang `resolve_max_num_reqs`）

- **檔案**：`runtime/engine/event_loop.py:281 per_rank_max_batch`（現在是
  `max_num_seqs // dp_size`，零 clamp），以及 `runtime/utils/server_args.py:845 validate`。
- **改動**：在 `geometry` 已知之後（`runtime/engine/event_loop.py:263`）把
  `per_rank_max_batch` 夾到一個從 `geometry.num_device_pages` 推導的數字，並在夾住時
  `logger.warning` 印出**前後兩個值**，照抄 SGLang
  `srt/mem_cache/kv_cache_configurator.py:1766` 的訊息形狀。
- **前置**：**不能用 `max_total_num_tokens`。** 它是 `usable_pages * block_size`
  （`runtime/configs/hybrid_cache_plan.py:541`），對多群組模型高估 5.5x（§2.5）。要用
  `geometry.num_device_pages / coordinator_.BlocksNeededFor(1)` 這一類真實的
  ids-per-token，或直接把「每個 resident 序列的最小頁數」暴露成一個 scheduler 查詢。
- **成本**：中。主要成本在把單位修對，不在夾住本身。
- **效果**：解決「operator 無從得知那個數字是 30」的一半。**正確性由 R1 提供，這一項提供
  的是可觀測性與保底**；R1 做完之後它不再是必要的，但仍然值得，因為它把一個現在只能靠
  A/B 掃出來的數字變成啟動日誌的一行。

### R3 — 搶佔改成單一請求觸發（抄 vLLM `_preempt_request` 的位置）

- **檔案**：`csrc/scheduler/operations/forward.cpp:329 resolveFlatStarvation`、
  `:1401` 的呼叫點、`:311 flatPoolWedged`。
- **改動**：把觸發條件從「這一步排出 0 個 op」換成「有請求因為配不到頁被 defer」。
  vLLM 的形狀是把搶佔放進 running 迴圈裡、對**這一個**配不到頁的請求做
  （`v1/core/sched/scheduler.py:534` 的 `while True`），而不是等整步空掉。
- **成本**：**四項裡最貴的一項。** `flatPoolWedged` 的
  `nothing_in_flight`（`csrc/scheduler/operations/forward.cpp:318`）與
  `pending_forward_results_.empty()` 是 retract 安全性的前提 —— `:351` 的 assert
  明說「retract victim must not owe a forward result」，而
  `:348-350` 的註解說明放寬它會破壞 rebased prefill。改成單一請求觸發之後這一步必然有
  in-flight op，所以受害者的挑選必須先排除所有本步已排入 `ops` 的請求，安全性論證要重寫。
- **效果**：R1 做完之後，這一項的邊際價值大幅下降 —— 准入正確時，搶佔本來就該是罕見的
  補救路徑。**建議在 A1 量完之後再決定要不要做。**

### R4 — 防抖：retract 發生的那一步跳過新請求准入（抄 vLLM 一行）

- **檔案**：`csrc/scheduler/operations/forward.cpp:1324` 的 candidates 迴圈。
- **改動**：等價於 `v1/core/sched/scheduler.py:637 if not preempted_reqs` —— 本步發生過
  retract 時，`fsm::Submitted` / `fsm::PrefetchDone` 分支直接 `continue`。
- **成本**：接近零（一個旗標）。
- **效果**：**單獨做沒有用**（現在 retract 一個 run 才發生一次），但它是 R3 的前提條件：
  一旦搶佔變成常態機制，沒有這一行就會出現 admit → preempt → admit 的擺盪。與 R3 綁在
  一起做。

### R5 — charge 到完成（抄 TRT-LLM `GUARANTEED_NO_EVICT`）

- **檔案**：`runtime/engine/request_handler.py:271`（Python 一行）＋
  `csrc/scheduler/operations/forward.cpp:199`（C++ 一個比較）。
- **改動**：`make_spec` 目前**沒有傳 `max_new_tokens`**，所以它一律是 0。傳進去之後
  `csrc/scheduler/request.h:101 RemainingOutputTokens()` 就會回真值，准入端把它加進
  R1 的比較即可。整條資料鏈已經接好（§2.6），這是**已寫好但沒接上的功能**，不是新功能。
- **成本**：程式碼上最小的一項。
- **但它的代價是真的**：TRT-LLM 用 `req.mMaxNewTokens`
  （`cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:3668`），請求沒給 `max_tokens` 時
  就退化成整個 attention window，併發會被壓得很低。SGLang 正是因為這個代價才改用回饋
  係數而不是硬上限（§2.1c）。
- **建議**：**先做 R1，量 A1，再決定。** 若 R1 之後 P99 已經收斂到 vLLM 水準，R5 只會
  換來更低的併發；若 P99 仍有長尾（表示問題出在生成長度而非 prompt 長度），再走 R5，
  或走 SGLang 的係數版本。這一格是四家真正意見不合的地方，不要在量測之前選邊。


---

## 7. 為什麼「每步都填滿批次」目前做不到（實測，2026-08-16）

R3（搶佔改成單一請求觸發）在本引擎上**還不能落地**，而擋住它的不是搶佔本身，是撤回
之後的優先級。三個相互耦合的位置：

| # | 位置 | 現況 | 後果 |
|---|---|---|---|
| 1 | `forward.cpp` Prefilling 分支，`admission_failed` 時 `break` | 一個請求配不到頁就終止整步 | 排在它後面、**不跟它搶容量**的常駐 decode 也丟掉槽位 |
| 2 | `retractForCapacity` 觸發條件 `operations.empty()` | 只有整步空掉才撤回 | 修掉 #1 之後步永遠不空，撤回**再也不會發生** |
| 3 | `blanket_forward_guard` | 只要**任何**請求欠著 forward result 就整個 return | 比兩階段撤回路徑（`deferred_victim`）更早擋掉，而那條路正是為此存在 |

三個一起改之後，撤回會**連鎖**：三請求容量案例把三個全撤光、`DecodingSize()` 歸零。
根因是 `priority()`（`forward.cpp:670`）給 `recovery_queue_` 隊首的請求 **2**，而常駐的
Prefilling 與 Decoding 都是 **3** —— 被撤回的請求下一步就搶回自己剛釋出的頁，把下一個
請求擠爆，如此遞推。加上「只為 `capacity_blocker` 撤回」與「同時只允許一個撤回在飛」
兩道節流都擋不住，因為每完成一次撤回就會再觸發一次。

vLLM 沒有這個問題：被搶佔的請求回到 waiting，**不會**插隊到 running 之前
（`v1/core/sched/scheduler.py:442` 的迴圈先跑完 `self.running`）。所以下一步要做的是
優先級順序，不是這三個分支。

**已量到的代價**：`k2trace` 那次跑（35B agent 26k、C=16、64 req）2,337 個有效步裡
95.1% 只用掉 8192 預算中的 <2048，99.9% 停在 `candidates exhausted`。但拒絕時
`free_lcm` 中位數是 3 —— arena 是真的滿了，所以填滿批次要先有容量，
`fp8 KV`（2x 容量）比排程更可能是下一格。

**已落地的部分**：head-of-line 只關閉准入而不終止整步（`176c8344`），以及兩處
「找不到可撤回請求」從 `FatalCheck` 改成跳過 —— 後者解鎖了 35B agent-127k 的五格，
它們先前全部無產出。


---

## 8. 曲線前段輸在哪，以及三次都修錯的方向（實測，2026-08-17）

goodput 只在一個期限上報一個數字，會蓋掉曲線的形狀。35B agent-127k 在報告採用的
10 秒期限上我們大勝（c=64 是 21.2 對 9.8），但**在 5 秒以前每一格都輸**：

| 期限 | c=64 我們 | vLLM | c=128 我們 | vLLM |
|---|---|---|---|---|
| 250 ms | 0.00 | **8.90** | 0.00 | **8.66** |
| 1,000 ms | 3.19 | **9.59** | 0.80 | **9.00** |
| 5,000 ms | 6.37 | **9.59** | 1.60 | **9.00** |
| 10,000 ms | **21.24** | 9.82 | **15.98** | 10.33 |

首字分佈是兩種形狀：vLLM **42/64 個請求在 1 秒內回覆**（p50 236 ms），其餘 22 個拖到
90 秒；我們只有 3 個，p10 已經是 5.5 秒、p50 27 秒。**它有快車道，我們沒有。**

我們這側的機制查到了：那一輪准入拒絕 **78,065 次**，被拒請求的 `unscheduled`
中位數只有 **52 個 token**，而當下有 **77 個空 parent**。device 命中中位數是 0 ——
命中全在主機層，准入要求整段前綴一次搬回 device，那才是被拒的原因。

**三個修法都量過，三個都更差**：

| 改動 | 假設 | 結果（35B，goodput req/s） |
|---|---|---|
| 關掉主機 L2 | 搬回是元兇 | agent-127k c=64 **0.26 → 0.21**，快車道仍是 3/64 |
| 按請求長度推導 checkpoint 間距 | 短 prompt 拿不到快照 | RAG c=64 **4.26 → 2.49**（P99 首字 4.7s → 9.0s） |
| 放不下就用較短前綴放行 | 全有或全無太嚴 | agent-127k c=64 **0.34 → 0.18**（時長 56.5s → 94.6s） |

三者指向同一件事：**在高併發下 arena 是綁住的資源，而重用比准入延遲值錢。**
少一點重用換早一點放行，總成本都是上升的；現在的操作點已經接近這條軸上的最佳。

（第一版的縮短前綴還額外犯了一個效能錯誤：`matchPrefixAtAdmission` 會把整條 prompt
重新雜湊，放進重試迴圈後等於每步每個候選付 1,500 頁的雜湊，排程器直接卡死 —— 這格
正常 44 秒，那一版跑 16 分鐘還沒完。把雜湊提到迴圈外之後才量得到上表的 0.18。）

**所以曲線前段不是准入政策能解的。** 要有快車道，得讓熱前綴留在 device 而不是掉到
主機層，也就是 device 容量本身 —— `fp8 KV` 在 sm_120 上已證實可用（2x 容量），
那是下一格。


---

## 9. 移植 vLLM 的就地搶佔：做了，更差，而且 vLLM 在那一格根本不搶佔（2026-08-17）

§2.2 說我們是孤例 —— 搶佔的觸發是「整步空掉」，而飽和 pool 上那永遠不會發生。
RAG c=64 實測證實了後果：56,141 個 decode 候選裡有 **6,988 個被拒**（12.4%），
**搶佔零次**，每步平均只有 36.7 個請求拿到 decode，而常駐是 53–64。

於是照 vLLM 的形狀移植：`scheduleDecode` 因容量失敗時就地挑一個 victim
（不欠 forward result、且不是本步已排到的 —— 我們的 op 是邊排邊提交 FSM 與 block，
沒有 vLLM `scheduler.py:548` 那種把已排請求撤出批次的回滾能力，所以 victim 只能取
候選順序中還在我們後面的那一段），撤回後重試一次，並加上 vLLM 的
`if not preempted_reqs`（本步搶佔過就不再准入）。

機制有動 —— 日誌可見 `preempt: released request b (7 tokens) so a could decode` ——
**結果更差**：

| RAG c=64 | 就地搶佔 | 現行 |
|---|---|---|
| tok/s | **656.6** | 1089 / 1117 |
| goodput req/s | **2.56** | 4.26 / 4.37 |
| 時長 | **24.95 s** | 15.0 s |
| 搶佔次數 | 52 | 0 |

**然後看 vLLM 在同一格的日誌：`Running: 64 reqs, Waiting: 0 reqs`。它一次都沒搶佔。**
它把 64 個全部同時跑完，9.9 秒結束。所以「移植搶佔」從一開始就是錯的槓桿 ——
vLLM 快不是因為它搶佔得好，是因為**它根本不需要**。

### 真正的差別是每請求的 parent 佔用

| | bytes/block | 每 parent 放得下 | 每請求 |
|---|---|---|---|
| `full_attention` | 2,621,440 | 16 | 4k prompt → 2 parents |
| `linear_attention_0/1/2` | 21,463,040（fp32） | **1**（浪費 49%） | **3 parents**（三個群組各一） |

parent 是 41,943,040 bytes，而一個狀態快照是 20.47 MB —— **差一點點就放得下兩個**，
所以每個 parent 只放一個、將近一半空著。一個 4k 的 RAG 請求因此吃掉約 5 個 parent，
64 個請求要 320 個，而池子只有 224。vLLM 的 mamba 是單一群組、`"align"` 模式兩頁，
所以它裝得下。

`mamba_ssm_dtype` 目前是 `float32`。SGLang 的 `Int8CheckpointStore`
（`srt/mem_cache/mamba_checkpoint_pool.py:186`）把 SSM 狀態存成 int8，scale 是
per-(layer, slot, head, k-channel)、縮減軸取 `d_v`，對齊 `diag(alpha)` 的 per-k-channel 衰減。
換算過來是 **5.12 MB/block、每 parent 7 個**，也就是狀態群組的 parent 佔用降到 1/7。

**這是下一格，而且它跟今天失敗的四個走的是不同的軸** —— 前四個都在既有容量上重新分配
（checkpoint 密度、host 層、前綴長度、搶佔），這一個是把每請求的佔用直接砍掉。
