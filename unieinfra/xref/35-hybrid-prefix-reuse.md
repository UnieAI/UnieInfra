# 35 — Hybrid 前綴重用：四家怎麼做，我們要怎麼做

一句話問題：**共用文件 + 不同問題（RAG／agent 的形狀），在 Qwen3.5-27B 上我們重用 0。**

量到的（`unieinfra/bench/`，`reuse_probe`，2690-token 文件 + ~130-token 尾巴）：

| 情況 | 我們 | vLLM（**快取關閉，見下方更正**）|
|---|---|---|
| 冷（第一次） | 510 ms, cached=0 | 582 ms, cached=0 |
| 完全重複 | **60 ms, cached=2752** | 412 ms, cached=0 |
| 換尾巴（RAG） | 437 ms, cached=0 | 431 ms, cached=0 |

> **更正（2026-08-10）：這一欄的 vLLM 不是 vLLM。**
>
> 這些數字是用 vLLM 的預設值量的，而 `Qwen3_5ForConditionalGeneration` 宣告 `IsHybrid`
> 卻沒宣告 `MambaPrefixCaching`，所以 vLLM 把 `enable_prefix_caching` 解析成 **False，
> 且不發任何警告**。上表量到的是一個**前綴快取被關掉的 vLLM**，那個 cached=0 是設定造成的，
> 不是能力上限。
>
> 明確傳 `enable_prefix_caching=True` 之後，同一台卡、同一個模型、同一支 harness 的
> RAG e2e（Qwen3.5-27B，conc=1/4/16 的 TTFT p50）：
> **447.87 / 1620.73 / 4134.22 ms → 112.76 / 358.40 / 1160.97 ms。**
> vLLM 在 RAG 形狀上**確實會重用**，它只是預設不重用。
>
> 因此「RAG 兩邊都是 0」這句話是錯的，`bf156ad4` commit message 裡的
> 「vLLM reuses nothing in either case」同樣是錯的。本次改動仍然成立
> （我們從 437 ms 降到 89.7 ms），但**它不是「補上一個 vLLM 也沒有的能力」，
> 而是「補上一個 vLLM 打開開關就有的能力」**，而且領先幅度遠比原本記載的小。
> 對照表見 `unieinfra/REPORT-commits.md` §12。
>
> 一般化的教訓已寫進 `AGENTS.local.md` 的量測紀律：**參照引擎的預設值要查，不能假設。**

---

## 為什麼是 0：一個 chunk 只發佈一個快照

Full attention 的 KV 是可切片的：匹配到第 n 頁就用前 n 頁。線性注意力不是 ——
它是遞迴狀態，要從第 n 個 token 接續，就必須有「處理完前 n 個 token 之後」的那個
狀態本身。所以 hybrid 的前綴重用等於 `min(KV 匹配深度, 最深可用快照深度)`。

我們每個 chunk 只發佈一個快照，位置在
`csrc/fsm/forward_events.cpp:130 MambaCheckpointPublishDepth`：

```cpp
const std::int32_t publish_depth = (final_len / page_size) * page_size - page_size;
```

也就是**緊貼請求自己的結尾**。`augmentMatch`
（`csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp:132`）從 KV 匹配終點
**往上**找帶快照的祖先 —— 而一個發在 2768 的快照，對於在 2690 分岔、匹配終點 2688 的
請求來說是**子孫不是祖先**，看不見。找不到就把整個 KV 匹配丟掉，不是截斷，是丟掉。

完全重複之所以能過，是因為它的匹配終點就在自己的快照之後。

### 兩個已經試過並回退的修法

1. **把快照往回退 256 token**（讓它落在分岔點之前）。結果**連完全重複也壞了**：
   `AlignMambaCacheSeqlen(track_len) != track_len` 直接回 -1，等於什麼都沒發佈。
   退讓量不是自由參數，只能是 GPU chunk 網格（`FLA_CHUNK_SIZE = 64`）的倍數。
2. **調 `--mamba-track-interval`**（64 與 256）。RAG 仍是 0，完全重複仍是 2752 ——
   那個旋鈕控制的是 flat 路徑上狀態群組的頁面網格，不是這條發佈路徑。

---

## 四家的策略

| 引擎 | 每序列保留幾個遞迴快照 | 存在哪 | RAG 重用 |
|---|---|---|---|
| **SGLang** | 多個，掛在 radix 節點上 | 獨立的 `MambaCheckpointPool`，SSM 存 **int8** | 有 |
| **vLLM** | `mamba_cache_mode="align"` 的對齊邊界 | KV block 池同一套管理 | **有，但預設關閉**（見開頭更正） |
| **TRT-LLM** | 多個，但用途不同（replay 重算） | 雙緩衝 + `MIN_REPLAY_HISTORY_SIZE=16` | 不走前綴樹 |
| **TurboMind** | 無此機制 | `kernels/linear_attn/` 只有前向 kernel | 無 |
| **我們** | **一個** | working slot 旁的單一 `checkpoint_` | **無** |

### SGLang：把檢查折進 descent，快照放獨立池

`srt/mem_cache/mamba_radix_cache.py:1068 _match_prefix_helper` —— 下降時**沿路記錄**
最深的帶快照節點，回傳的是那個節點：

```python
while len(key) > 0 and child_key in node.children.keys():
    child = node.children[child_key]
    if node.mamba_value is not None:      # 下降前先記
        best_value_len = len(value)
        best_last_node = node
    ...
if node.mamba_value is not None:          # 收尾再記一次
    best_value_len = len(value)
    best_last_node = node
return value, best_last_node, best_value_len
```

**重用長度是 `min(KV, 快照)`，永遠不會因為終點沒快照就歸零。** 這是結構差異：
我們先挑 KV 終點、再事後往上找；它是一邊下降一邊挑「兩者都有」的最深點。

`srt/mem_cache/mamba_checkpoint_pool.py:186 MambaCheckpointPool` 解掉「多份快照很貴」：

- **獨立於 working 池**，容量自己設，不和活躍請求搶
- SSM 狀態存 **int8**（`Int8CheckpointStore`），scale 是 per-(layer, slot, head, k-channel)，
  縮減軸取 `d_v`，對齊 `diag(alpha)` 的 per-k-channel 衰減
- conv 窗口保持原 dtype（本來就小）
- `store_from_active` / `load_to_active` 在兩池之間搬

我們 `mamba_ssm_dtype='float32'`，換 int8 就是 **4x** 容量。

### TRT-LLM：不是前綴樹，是 replay

`_torch/pyexecutor/mamba_cache_manager.py` 走另一條路：保留一個歷史窗口，需要時
**重算**而不是查表。`MIN_REPLAY_HISTORY_SIZE = 16` 是因為 replay kernel 會把 token 維
補到 tensor-core tile，小於 16 不會更快、卻要更頻繁地寫 checkpoint。

值得記的是它的取捨方向和我們相反：它用「重算」換「不存」。對 RAG 那種
「一份文件被幾十個問題共用」的形狀，存一次給多次用顯然划算，所以我們不走這條。

### TurboMind：沒有

`src/turbomind/kernels/linear_attn/` 只有 `delta_rule.cu` 和前向 kernel，沒有任何
前綴／重用機制。sm_120 的 `tma_desc_prepare.cu` 在，值得之後看它的 kernel 寫法，但
在這一題上沒有可參考的結構。

---

## 設計：把 K 個中間狀態搬出來

### 已經確認可用的東西

**kernel 完全不用改。** `gdn_chunk_prefill(output_h=True)` 已經回傳每一段：

```
fi_checkpoints        [total_ckpts, H, V, K] float32，K-last
                      checkpoint k = 處理完 chunk k 之後的狀態
                      每序列數量 = L_i // FLA_CHUNK_SIZE（64）
checkpoint_cu_starts  [N+1] int64，每序列在緩衝區的起點
```

2690-token 的前綴就有 **42 個**中間狀態躺在裡面。而複製那一步是純 gather/scatter
（`runtime/layers/attention/backends/hybrid_linear_attn.py:2763`）：

```python
ssm_states[track_ssm_h_dst]  = fi_h_checkpoints[h_src]
conv_states[track_ssm_h_dst] = mixed_qkv_t[:, track_conv_indices].transpose(0, 1)
```

**索引張量拉長 K 倍就直接能用。**

### 瓶頸在傳遞與儲存

`csrc/resource/allocator/local_mamba_allocator.h:56` 每請求持有恰好一個：

```cpp
std::unique_ptr<MambaSlot> checkpoint_{};
```

一路到 `mamba_checkpoint_dst_idx`（單一 int）→ `FlatForwardOperation` →
bindings（`bindings/python_module.cpp:396`）→ `input_buffer.py` → 索引計算。

### 改動清單

| 檔案 | 改什麼 |
|---|---|
| `local_mamba_allocator.{h,cpp}` | `checkpoint_` → vector；`AllocateCheckpoints(n)` / `DetachCheckpoints()` |
| `scheduler/operations/forward.{h,cpp}` | `mamba_checkpoint_dst_idx` → 攤平的 indices + 每請求 count（CSR 形式） |
| `fsm/forward_events.cpp` | `MambaCheckpointPublishDepth` 回傳深度**列表**；逐一 `SplitAt` + `InsertMamba` |
| `bindings/python_module.cpp` | 暴露攤平形式 |
| `runtime/execution/input_buffer.py` | 帶上攤平的 dst 與 count |
| `hybrid_linear_attn.py` | `_compute_track_ssm_indices` / `_compute_track_conv_indices` 每請求發 K 個 |
| 新增 | checkpoint 池的配額與 LRU 淘汰 |

### K 取多少

不必搬 42 個。快照間隔 G 決定「分岔時最多損失多少重用」：

| G | 2690-token 前綴的快照數 | 最差重用 |
|---|---|---|
| 64（全搬） | 42 | 2688 |
| 512 | 5 | 2048 |
| 1024 | 2 | 2048 |

**建議 G = 512**（`K = ceil(prefix_len / 512)`）：每請求 5 個快照，最差仍重用 2048／2690。
配合 int8 存放（4x），實際 slot 成本約等於現在的 1.25 倍。

### 不變式（兩次失敗換來的）

1. 發佈深度必須滿足 `AlignMambaCacheSeqlen(d - chunk_begin) == d - chunk_begin`。
   GPU 只在自己的 chunk 網格上留中間狀態；不在網格上的深度會被靜靜丟掉（回 -1）。
2. Python 的 `publish_lens` 與 C++ 的 `MambaCheckpointPublishDepth` **必須逐位一致** ——
   C++ 用它把 slot 掛到 radix 節點，Python 用它決定搬哪一個索引。兩邊算不一樣，
   狀態會被歸檔在一個它並不描述的深度上，之後的命中會從錯誤的遞迴狀態接續。
   靜默且錯誤。這條在 `runtime/layers/attention/backends/hybrid_linear_attn.py:1508` 已有註解，改動時最容易破壞。
3. `augmentMatch` 必須改成「回傳 min(KV, 快照) 的截斷」而不是「找不到就丟棄」——
   否則就算發了 K 個，只要最深的那個比 KV 終點深，仍會歸零。

---

## 結果（已實作）

`reuse_probe`，同一組 prompt，改動前後：

| 情況 | 之前 | **之後** |
|---|---|---|
| 冷（第一次） | 510 ms, cached=0 | 543 ms, cached=0 |
| 完全重複 | 60 ms, cached=2752 | **58 ms, cached=2752** |
| **換尾巴（RAG）** | **437 ms, cached=0** | **89.7 ms, cached=2560** |

RAG **快 4.9x**，重用 0 → 2560（stride 512 向下取整），而完全重複沒有退步 —— 這是設計時
的硬條件，不能為了 RAG 犧牲它。

**正確性**：重用 2560 個 token 的回答與同一問題在冷引擎上算出的**逐字相同**（`rag_correct.py`，
3 個問題）。錯的遞迴狀態不會當掉，只會產生流暢而錯誤的文字，所以這一項比時間更重要。

端到端（`e2e_concurrency.py --shape rag`，Qwen3.5-27B，700 字文件）：

| conc | TTFT p50 | TTFT p99 | ITL p50 | TPS |
|---|---|---|---|---|
| 1 | 93.61 | 99.42 | 37.03 | 26.3 |
| 4 | 262.99 | 264.29 | 39.77 | 92.4 |
| 16 | 736.74 | 742.30 | 44.36 | 289.3 |

改動前 RAG 的行為與冷 prefill 相同（`cached=0`），也就是 conc=1/4/16 分別是
420.7 / 1624.7 / 4611.7 ms、TPS 23.4 / 62.1 / 110.2。對照之下 **TTFT 降 4.5–6.3x、
TPS 升 12%–162%**，且 conc=16 的 TTFT p99 從 6.4 s 降到 742 ms —— 落回 1 秒 SLA 之內。

### 實作與設計的兩處差異

1. **額外的 slot 在發佈點補配，不在四個分配點。** 只有發佈點知道 chunk 範圍，因而知道
   要幾個；在分配點預留上限會讓一份長文件餓死整批。配置是 best-effort，池子不夠只會
   少留幾個快照，不會讓請求失敗。
2. **發佈的 slot index 必須由發佈點記下來交給 forward operation。** 掛上樹時所有權就
   轉移了，allocator 不再知道自己給了哪些 slot —— 而 operation 正是告訴 GPU 往哪寫的
   東西。少了這一步，節點會持有永遠沒被寫入的 slot，之後的命中從未初始化的記憶體接續。
   這是實作時唯一一個會靜默出錯的地方。

## 驗收

`unieinfra/bench/` 的 `reuse_probe` 三列都要動：

- **換尾巴（RAG）**：`cached=0` → `≥ 2048`，437 ms → 應接近完全重複的 60 ms 量級
- **完全重複**：維持 `cached=2752`、60 ms（不能為了 RAG 犧牲它）
- **冷**：不變

再跑 `e2e_concurrency.py --shape shared`（注意：那個形狀量的是完全重複，RAG 要用
`reuse_probe` 或新增一個 shape），確認 TTFT 下降、TPS 上升。

---

## 文獻對照（讀於 2026-08-11）：我們實作的東西在論文裡叫 `Balanced`

三篇，一篇有 code。它們把上面「為什麼是 0」與「K 取多少」兩節從工程判斷變成有名字的
已知問題，並指出我們選的那個點不是最佳解。

### 1. 我們的 0 是一個退化解，不是 bug

*Sparse Prefix Caching for Hybrid and Recurrent LLM Serving*（arXiv 2605.05219，
Shirokikh & Nikolenko，2026-05）把這題形式化。給定快照集合 `C = {c_1 < … < c_M}`，
一個重疊深度為 `t` 的請求能接續的最深快照是：

```
ℓ(t; C) = max({0} ∪ {c ∈ C : c ≤ t})
```

需要重算 `r(t; C) = t − ℓ(t; C)` 個 token 的遞迴工作。

那個 `∪ {0}` 就是本文件開頭那件事的數學版：**只有一個快照、而且它坐在比 `t` 深的位置
時，集合是空的，`ℓ = 0`，整段重算。** `MambaCheckpointPublishDepth` 把唯一的快照發在
`final_len` 附近，對任何在中途分岔的請求都是子孫節點 —— 所以 RAG 形狀必然是 0，
而完全重複必然是滿的。兩個極端是同一個式子的兩端，不是兩個 bug。

論文另外證明（Remark 1）**從快照重放是輸出精確的**：遞迴更新是決定性的，從 `c` 重放
`c+1…t` 得到的狀態與從 0 全跑相同，他們跨資料集驗過 bit-exact。這與本文件
`rag_correct.py` 的逐字相同是同一個保證，可以互相佐證。

### 2. `G = 512` 等距 = 論文的 `Balanced`，而它只在重疊均勻時最佳

論文比較五種配置策略。對照本文件「K 取多少」那節：

| 論文策略 | 快照位置 | 我們 |
|---|---|---|
| No cache | — | 修改前 |
| **Balanced** | `⌊i(L+1)/(M+1)⌋` 等距 | **現在（G = 512）** |
| Block | `B, 2B, …, ⌊L/B⌋B` | — |
| Sqrt | `⌊√L⌋` 間隔 | — |
| DP-optimal | 依重疊分布解 Thm 2 | — |

**Theorem 1**：balanced 在**均勻重疊**與**最差情況**下最佳，最差重算
`⌈(L+1)/(M+1)⌉ − 1` —— 這正是本文件表格裡「G=512 → 最差重用 2048」那一欄，我們當時
是手算出來的。

**但 Fig 1 顯示真實重疊分布離均勻很遠**：QuALITY（單文件多問題，**就是我們的 RAG 形狀**）
在文件全長附近有一個尖峰；System Prompts 的質量集中在短前綴。分布集中時，把 5 個快照
均勻鋪開是在浪費預算 —— 該把它們堆在分岔點附近。

**Theorem 2** 給出精確的 O(NM) DP（convex hull trick），用歷史重疊直方圖估分布，
每 10 個請求重解一次，離線算完不上服務路徑。Thm 3/4 給了用經驗直方圖取代真分布的
plug-in 保證與漂移界。

**結果**：NarrativeQA 上 DP 版在整個 budget 範圍壓過 Balanced/Block/Sqrt/Logarithmic，
recurrent-work reduction 最高 ~3.2x，而且**預算越低差距越大**。我們 K=5 正是低預算端。

> **評測強度要打折**：RTX 2080 Super 8GB + HuggingFace Transformers（不是 serving
> engine），wall-clock 只量「代表性層組」= Qwen-3.5-0.8B 的 1 個 full attention + 3 個
> GDN 層。作者自己寫明這不能當完整模型的 serving benchmark。快照擷取還是在另一次
> **不計時**的 run 裡做的，因為 FLA 沒有暴露非最後位置的擷取 API —— 而我們的
> `gdn_chunk_prefill(output_h=True)` 已經有（見「已經確認可用的東西」），
> **這一項我們領先論文的原型。** 沒有 code release。

### 3. Marconi：我們 TODO 裡「快取池配額與 LRU 淘汰」那一格的答案

*Marconi: Prefix Caching for the Era of Hybrid LLMs*（MLSys'25 Outstanding Paper
佳作，Princeton + AWS，<https://github.com/ruipeterpan/marconi>）與上一篇**正交**：
上一篇決定「一個 entry 內部快照放哪」，Marconi 決定「**哪些 entry 值得留**」。

它量化了本文件開頭那個現象（7B 模型，Fig 3a）：

| block size | KV block 重用率 | 遞迴狀態重用率 | 差距 |
|---|---|---|---|
| 32 | 25.0% | **0.4%** | **65.3x** |
| 64 | — | — | 27.9x |
| 128 | — | 3.3% | 11.1x |

記憶體側：block size 16 時單一 SSM 層的狀態是同 block 內 Attention 層 KV 的 4 倍；
7B 模型單一 10K token 序列吃 **17.4 GB**，是同尺寸 Transformer 的 3.3 倍。
**細粒度快照會把池子打爆**，這是我們把 G 訂在 512 而不是 64 的獨立佐證。

兩個機制：

- **Admission（推測性插入）**：把前綴重用分成「純輸入」（system prompt、few-shot —— 跨
  請求共享、會分岔）與「輸入+輸出」（對話歷史 —— 只往後接、不分岔）。prefill 前先把
  輸入 token 試插進 radix tree，**只有會產生中間節點（分岔點）的位置才值得下錨**。
  最後一個 decoded token 無論如何都存。
- **Eviction（FLOP-aware）**：不只看 recency，看重用時能省多少計算、再除以佔用空間。
  刻意犧牲短序列命中率換長序列。

**結果**：token hit rate 比擴充過 hybrid 支援的 vLLM/SGLang 高 **4.5–34.4x**，
P95 TTFT 低 **36.1–71.1%**（103.3–617.0 ms）。

Marconi 明講**當時沒有任何既有系統支援 hybrid 的 prefix caching**，vLLM/SGLang 的對照組
是他們自己擴充的 —— 這與本文件開頭那則更正（vLLM 在 Qwen3.5 上 `enable_prefix_caching`
靜默解析成 False）是同一件事的兩個來源。

### 4. 第三條路：不要存狀態

*KVBuffer: IO-aware Serving for Linear Attention*（arXiv 2605.19049，Yale，2026-05）
的 §6 提出另一個位置：**遞迴狀態可以從緩衝的 K/V 重建**。所以與其快取狀態，不如快取
選定前綴的 K/V，命中時再把狀態算出來。狀態太大導致快照只能粗粒度、難以在任意位置切分
—— 那個限制在這條路上不存在。

代價是命中時要付重建的計算，而且它要求引擎有 chunkwise 的 KV 緩衝機制（該論文的主體）。

### 5. 三個位置

| | 主張 | 我們的狀態 |
|---|---|---|
| **Sparse**（2605.05219） | 存狀態，選好 entry **內部**的位置 | 已實作 `Balanced`；DP 版未試 |
| **Marconi**（MLSys'25） | 存狀態，選好留**哪些 entry** | 未實作（= TODO 的池配額／淘汰） |
| **KVBuffer** §6（2605.19049） | **不存狀態**，存 K/V 現場重建 | 未評估 |

三者不互斥。Sparse 與 Marconi 作者互相確認正交，可直接疊。

### 6. 可測候選

| 候選 | 量什麼 | 為什麼值得 |
|---|---|---|
| P1 分岔點對齊 | 把快照從等距改成「靠近實際分岔深度」，同樣 K=5 | `reuse_probe` 現在拿 2560/2690，損失 130 是 stride 向下取整；分岔深度會變時損失更大 |
| P2 重疊深度直方圖 | 在 RAG／shared 形狀上記錄實際重疊深度分布 | Thm 2 的輸入。分布若真的是尖峰，DP 相對 Balanced 的增益就是論文說的低預算大增益 |
| P3 Marconi admission | 只在會產生 radix 分岔的節點下錨，取代固定 G | 直接減少快照數；池壓力是我們把 G 訂大的唯一理由 |
| P4 FLOP-aware 淘汰 | 對照 LRU | TODO 那一格，有現成參考實作 |

### 7. 未解格

- 本文件「四家的策略」表裡我們那列仍寫「保留幾個：**一個**／RAG 重用：**無**」，
  那是修改前的狀態，與「結果（已實作）」矛盾。表要更新，但要先確認 G=512 在
  當前 build 上仍成立（見下）。
- **上面所有量測數字的時間戳是 2026-07-25 前後，而本分支此後有 41 個 commit**，
  其中包含 scheduler 優先權重排（`d1018671`，commit message 自陳未量測）。
  引用本文件任何 ms 數之前先重跑。


---

## 8. checkpoint 密度在高併發下是「重用 vs 常駐」，不是純粹的重用（實測，2026-08-17）

出貨的間距是從 `max_model_len` 推導的固定值（`_MAX_STATE_CHECKPOINTS = 2`，
`qwen35.py`）。它在 RAG c=64/128 上讓 goodput 從 0.75x 掉到 0.66x，重用從 75.5%
掉到 69.5%，所以看起來像是「短 prompt 拿不到 checkpoint」的 bug ——
MML=16384 → 間距 64 blocks，而 RAG 的 prompt 只有約 32 頁，落在一個間距之內。

**把間距改成從請求自己的長度推導（固定 K 個快照）之後，RAG 更差。** 實測：

| RAG c=64 | per-request K=2 | 固定間距（出貨值） |
|---|---|---|
| goodput | **2.49** req/s | 4.26 / 4.37 |
| P99 TTFT | **8,955 ms** | 4,681 / 4,525（SLO 5,000） |
| TPOT p50 | 30.1 ms | 39.0 ms |
| reuse | 71.0% | 69.5% |

重用確實升了 1.5 個百分點，TPOT 也更好，但 **P99 TTFT 幾乎翻倍**，跨過 SLO 的請求
變多，goodput 掉 42%。原因是 charge：固定間距給一個 32 頁的 RAG 請求 **1** 個
backed page（只有 chunk 尾），per-request K=2 給它 **2** 個。RAG 在 c=64 是
arena-bound 的，多那一個 checkpoint 換到的重用補不回常駐數。

**結構上的結論**：固定間距按「prompt 相對於 `max_model_len` 的長度」分配 checkpoint
—— 短 prompt 少、長 prompt 多 —— 而 arena 壓力正好隨「併發 × 短 prompt」上升。
per-request 的 K 讓長短 prompt 都拿到 K 個，分不開這兩者。**在這一題上固定間距是對的
抽象，per-request 不是。** agent 26k c=16 兩者持平（1.00 vs 0.99），所以這個改動是
純粹的退步，已還原。

RAG c=64/128 的 LOSS 因此不是 checkpoint 密度能解的。下一格要看的是容量本身
（`fp8 KV` 有 2x），或 vLLM 在這個形狀上到底快在哪 —— 我們 1,089 tok/s 對它 1,647，
而兩邊 TTFT p50 都在 1 秒級，差距不在准入。
