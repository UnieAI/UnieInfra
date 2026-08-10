# 36 — 一個被快取的遞迴狀態，如何抵達一個執行中的序列

四家對照。這一題是 `35-hybrid-prefix-reuse.md` 的下一層：那一份問「快照該存在哪個深度」，
這一份問「命中之後，那個狀態怎麼變成請求正在算的狀態」。

會單獨開一份，是因為我們在這一步和其他所有人**反著做**，而那個反向正是容量天花板的成因。

---

## 1. 步驟表

| 步驟 | UnieInfra（我們） | SGLang | vLLM | TRT-LLM | TurboMind |
|---|---|---|---|---|---|
| 快取狀態存在哪 | 活池，同精度 | **獨立 `int8_ckpt_pool`** | KV block pool，同一套 | 雙緩衝，不走前綴樹 | — |
| 命中時如何取用 | **別名（alias）**：working index 直接就是樹的 slot | **複製**：`load_to_active` 解量化進請求自己的 slot | **複製**：把 state 複製進本步新配的 block | replay 重算 | — |
| 複製發生在哪 | 不發生 | `srt/model_executor/model_runner.py:1581 load_to_active`，**forward stream 上、mamba 層讀池之前** | `v1/core/single_type_kv_cache_manager.py:1163 remove_skipped_blocks` | `_torch/pyexecutor/mamba_cache_manager.py:232 _advance_replay_state` | — |
| 有沒有 dst 索引 | **只有 `mamba_cow_src_idx`，沒有 dst** | `mamba_cow_src_indices` + `mamba_cow_dst_indices` | 新 block id 即 dst | 雙緩衝的另一半 | — |
| 沒有壓縮時的退路 | — | `mamba_pool.copy_from(src, dst)`，仍然是複製 | 同上 | — | — |

引用（pinned commits，見 `00-METHOD.md`）：

- SGLang `srt/model_executor/model_runner.py:1581 load_to_active`
- SGLang `srt/mem_cache/mamba_checkpoint_pool.py:253 load_to_active`
- vLLM `v1/core/single_type_kv_cache_manager.py:1163 remove_skipped_blocks`
- TRT-LLM `_torch/pyexecutor/mamba_cache_manager.py:69 MIN_REPLAY_HISTORY_SIZE`

---

## 2. 分歧點

### D20 —— 命中時複製 vs 別名（**我們是孤例**）

**SGLang 與 vLLM 都複製**，而且兩家都不是為了壓縮才複製：SGLang 在**沒有** int8 池時
仍然走 `mamba_pool.copy_from(src, dst)`，vLLM 則把 state 複製進本步新配的 block。
複製是它們的預設語意，量化只是讓被複製的來源可以更便宜。

**我們別名**：一次普通的前綴命中不在 Python 消費 `mamba_cow_src_indices`（三個讀取點
全部在 `is_target_verify` 底下），也沒有任何 `cow_dst` —— 請求直接讀樹擁有的那一格。

別名看起來更快，**每次命中省下一次複製**。代價是它把兩件事鎖死在一起：

1. 快取狀態必須和活狀態**在同一個張量、同一個精度**，否則沒得別名
2. 因此快取必然和執行中的序列**搶同一個池子**

而那正是 `failed to allocate Mamba checkpoint slot` 的來源 —— conc=64 當機、
加鎖之後 conc=512 仍然斷。**別名不是一個可以保留的優化，它是天花板本身。**

### 這筆複製到底多貴

值得算清楚，因為「省一次複製」聽起來像是我們贏了：

- 複製是**每次命中一次**，不是每 token、不是每層每步
- 它取代的是重算 ~2560 個 token 的 prefill（`bf156ad4` 量到 437 → 89.7 ms）
- 48 層 × 2 個張量的搬移，對照的是幾百毫秒的 prefill

**在真正的成本尺度上這筆複製看不見。** 我們用它換來的是一個把快取和活請求綁在一起的
資料佈局，然後為了那個佈局付掉一整天：兩次失敗的修法、一個只是在調停不該存在之競爭的鎖。

### D21 —— 複製排在哪條 stream 上

SGLang 把 COW 排在 **forward stream、mamba 層讀池之前**，註解寫明理由是
「so the copies don't race the scheduler copy stream」。這不是效能考量而是正確性考量：
scheduler 的複製流和 forward 流是兩條，把 COW 放在 forward 流上就不需要跨流同步。

我們沒有這個位置可以放，因為我們沒有這一步。**移植複製語意的時候，這個放置點要一起移植**，
否則就是把一個競態換成另一個。

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
| 別名（我們現在） | **0 次複製** | 快取吃活池，同精度 | **conc≥64 當機 / 512 斷** |
| 複製 + 同精度獨立池 | 1 次複製 | 1x | 無爭搶 |
| **複製 + int8 獨立池（SGLang）** | 1 次複製 + 解量化 | **~4x** | 無爭搶 |

**採用 SGLang 的做法。** 那 1 次複製在 prefill 尺度上量不出來，換到的是四倍快取容量
和一個消失的失效模式。這也是唯一有現成實作可以 1:1 移植的路徑
（`runtime/cache/mamba_checkpoint_pool.py` 已移植完成並測試）。

**不採用**「保留別名、只把池子放大」：那只是把 conc=512 的斷點往後推，
而且 fp32 的快取要四倍記憶體才能換到同樣的深度。

---

## 4. 要改的東西

1. C++ 的 op 要帶 **`mamba_cow_dst_idx`**，不能只有 src —— 沒有 dst 就沒有複製的目的地
2. 前綴命中時，working index 不再等於樹的 slot；請求讀自己的 slot
3. 複製排在 **forward stream、mamba 層讀池之前**（SGLang 的位置，理由是避開 scheduler copy stream 的競態）
4. 有 int8 池時走 `load_layer_to_active`，沒有時走一般複製 —— 兩條路都要在，和 SGLang 一樣

第 1、2 點是 `41939351` 停下來的原因：我當時不確定我們的命中是別名還是複製，
而猜錯**不會噴錯，只會從錯的遞迴狀態接續**。現在確定了是別名。
