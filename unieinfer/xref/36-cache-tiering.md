# 36 — 一個被快取的遞迴狀態，如何抵達一個執行中的序列

四家對照。這一題是 `35-hybrid-prefix-reuse.md` 的下一層：那一份問「快照該存在哪個深度」，
這一份問「命中之後，那個狀態怎麼變成請求正在算的狀態」。

> ## ⚠ 更正（同日）：本文件第一版說「我們別名不複製」，那是錯的
>
> 我們**有**複製：`runtime/execution/runtime_states.py:98 copy_mamba_states`，
> 在 forward 之前對 `conv_state` 與 `ssm_state` 各做一次
> `fused_mamba_state_copy(src=cow_src, dst=working)`。
>
> 第一版的依據是「Python 裡三個 `mamba_cow_src_indices` 讀取點全在 `is_target_verify`
> 底下」。那三個是**推測解碼**的讀取點；一般命中的複製在 executor 裡，我沒有搜到就下了結論。
> op 裡也一直都有目的地 —— 就是 `mamba_working_idx`，只是它不叫 `cow_dst`。
>
> **結論沒有改變，理由改變了。** 採用獨立 int8 池仍然正確，但原因不是「別名逼得同池」，
> 而更簡單：快照本來就是從活池配的（`LocalMambaAllocator::AllocateCheckpoints`）。
> 複製語意早就在位置上了，所以要接的只有「來源換成 checkpoint 池、複製換成解量化」。
>
> 教訓和 `port-first-then-optimize` 記的是同一條：**沒有讀到不等於不存在。**

---

## 1. 步驟表

| 步驟 | UnieInfer（我們） | SGLang | vLLM | TRT-LLM | TurboMind |
|---|---|---|---|---|---|
| 快取狀態存在哪 | 活池，同精度 | **獨立 `int8_ckpt_pool`** | KV block pool，同一套 | 雙緩衝，不走前綴樹 | — |
| 命中時如何取用 | **複製**：`copy_mamba_states` 把 cow_src 複製進 working slot | **複製**：`load_to_active` 解量化進請求自己的 slot | **複製**：把 state 複製進本步新配的 block | replay 重算 | — |
| 複製發生在哪 | `runtime/execution/runtime_states.py:98 copy_mamba_states`，forward 之前 | `srt/model_executor/model_runner.py:1581 load_to_active`，**forward stream 上、mamba 層讀池之前** | `v1/core/single_type_kv_cache_manager.py:1163 remove_skipped_blocks` | `_torch/pyexecutor/mamba_cache_manager.py:232 _advance_replay_state` | — |
| 有沒有 dst 索引 | 有，但叫 `mamba_working_idx` 而不是 cow_dst | `mamba_cow_src_indices` + `mamba_cow_dst_indices` | 新 block id 即 dst | 雙緩衝的另一半 | — |
| 沒有壓縮時的退路 | — | `mamba_pool.copy_from(src, dst)`，仍然是複製 | 同上 | — | — |

引用（pinned commits，見 `00-METHOD.md`）：

- SGLang `srt/model_executor/model_runner.py:1581 load_to_active`
- SGLang `srt/mem_cache/mamba_checkpoint_pool.py:253 load_to_active`
- vLLM `v1/core/single_type_kv_cache_manager.py:1163 remove_skipped_blocks`
- TRT-LLM `_torch/pyexecutor/mamba_cache_manager.py:69 MIN_REPLAY_HISTORY_SIZE`

---

## 2. 分歧點

### D20 —— 快照從哪個池子配（**我們是孤例**）

四家在「命中時複製」這件事上其實**一致**。分歧在更前面一步：**被快取的狀態是從哪個池子配的。**

- **SGLang**：獨立 `int8_ckpt_pool`，和活池完全分開
- **vLLM**：和 KV 同一個 block pool，但那個池子有統一的 refcount 與驅逐，且排不進去時
  回傳 `num_gpu_blocks + 1` 讓 scheduler 不排它 —— **不會稍後 throw**
- **我們**：`LocalMambaAllocator::AllocateCheckpoints` 從**活池**配，而 admission
  完全沒把它算進去，所以配不到時在 `SchedulePrefillEvent` 裡直接 throw

這就是 `failed to allocate Mamba checkpoint slot` 的根因 —— conc=64 當機、加鎖後 512 仍然斷。
**不是複製語意的問題，是配置來源與 admission 的問題。**

### 這筆複製到底多貴

值得算清楚，因為「省一次複製」聽起來像是我們贏了：

- 複製是**每次命中一次**，不是每 token、不是每層每步
- 它取代的是重算 ~2560 個 token 的 prefill（`bf156ad4` 量到 437 → 89.7 ms）
- 48 層 × 2 個張量的搬移，對照的是幾百毫秒的 prefill

**在真正的成本尺度上這筆複製看不見** —— 這一段的意義因此不是「該不該複製」（我們本來就複製），
而是「換成解量化之後會不會變貴」：不會，它取代的仍然是幾百毫秒的 prefill。

### D21 —— 複製排在哪條 stream 上

SGLang 把 COW 排在 **forward stream、mamba 層讀池之前**，註解寫明理由是
「so the copies don't race the scheduler copy stream」。這不是效能考量而是正確性考量。

我們的 `copy_mamba_states` 由 `execute_forward_op` 在 forward 之前呼叫，位置相同 ——
這一格我們本來就對，第一版誤以為我們沒有這一步。

### D22 —— TRT-LLM 不走前綴樹

它做的是**用 replay 重算**：雙緩衝 + `MIN_REPLAY_HISTORY_SIZE=16` 的歷史窗口，
在推測解碼的接受/拒絕之後推進狀態。這解的是另一個問題（spec decode 的回捲），
不是跨請求的前綴重用，所以在這一題上它不是候選方案。

註解裡有一條硬體理由值得記：**歷史窗口超過 16 會被 pad 到 32，明顯更慢**，
因為 tensor core 的 tile 只有變大沒有變小。

---

## 3. 最快的做法是哪一個

以「命中一次的端到端成本」排序，而不是以「這一步做了幾次記憶體搬移」排序：

| 做法 | 命中成本 | 容量 | 併發天花板 |
|---|---|---|---|
| 複製 + 快照配在活池（我們現在） | 1 次複製 | 快取吃活池，同精度 | **conc≥64 當機 / 512 斷** |
| 複製 + 同精度獨立池 | 1 次複製 | 1x | 無爭搶 |
| **複製 + int8 獨立池（SGLang）** | 1 次複製 + 解量化 | **~4x** | 無爭搶 |

**採用 SGLang 的做法。** 那 1 次複製在 prefill 尺度上量不出來，換到的是四倍快取容量
和一個消失的失效模式。這也是唯一有現成實作可以 1:1 移植的路徑
（`runtime/cache/mamba_checkpoint_pool.py` 已移植完成並測試）。

**不採用**「保留別名、只把池子放大」：那只是把 conc=512 的斷點往後推，
而且 fp32 的快取要四倍記憶體才能換到同樣的深度。

---

## 4. 要改的東西

更正之後只剩兩件，因為複製語意和它的放置點我們本來就有：

1. ~~op 要帶 cow_dst~~ —— 已經有，叫 `mamba_working_idx`
2. ~~複製排在 forward 之前~~ —— 已經是
3. **`copy_mamba_states` 在有 int8 池時改走 `load_layer_to_active`**（來源換池、複製換解量化）✅ 已接
4. **admission 要把 mamba slot 算進去** —— vLLM 用「回傳一個排不進去的數字」表達，
   我們在 `SchedulePrefillEvent` 裡 throw。這是 512 斷點剩下的那一半，尚未做。
