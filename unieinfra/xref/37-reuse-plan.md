# 37 — 把 GDN hybrid 的前綴重用從 44% 推向天花板：原始碼盤點與修改提案

對象：Qwen3.5-27B（GDN hybrid），radix 路徑（`SimpleMambaPool`），branch `unieai`。
本文只做原始碼分析，未跑任何 GPU 量測、未啟動任何 server、未 commit。
與 `35-hybrid-prefix-reuse.md` 的關係：35 是四家對照與文獻，本文是**針對現況原始碼的
落地計畫**，不改 35。

---

## 現況

### 已成立的事實（本文只做原始碼側的核對，不重量）

1. hybrid 的重用 = `min(KV 匹配深度, 最深可用遞迴快照深度)`。
   核對點：`hybrid_prefix_cache.cpp:191` 從 KV 終點往上找帶快照的祖先，
   `:207-216` 在找到時把 `match.device.last_node` **截斷**到快照節點，
   `:197-205` 在找不到時把兩層都設回 `root` —— 整段 KV 匹配丟棄，而且無聲。
2. `e021b555` 修的是 `MambaCheckpointPublishDepth`（`forward_events.cpp:134-165`）：
   `:138` 在 16-token KV 頁網格上算候選深度，`:162` 現在用
   `AlignMambaCacheSeqlen()` **向下取整**到 64 的 scan grain，而不再要求兩者恰好重合。
   `AlignMambaCacheSeqlen` 見 `hybrid_prefix_cache.cpp:260-263`，grain 來自
   `SchedulerConfig::mamba_cache_chunk_size`，預設 64（`scheduler/types.h:186`），
   與 `chunk_delta_h.py:36 CHUNK_SIZE = 64` 對齊。
3. `kMambaSnapshotMaxPerRequestDefault = 1`（`forward_events.cpp:208`）。
   註解說明的理由（GPU 只寫一個）在原始碼裡**確實成立，而且理由比註解寫的更多**——
   見下面問題 1，`_expand_multi_track` 目前即使被走到也會寫出錯誤形狀的 conv 快照。
4. 有一個 open defect：快取開啟時同一 prompt 的首 token logprob 與冷算差 0.04–0.07。
   **本文認為已在原始碼中找到機制，見問題 2。**

### 目前實際生效的發佈規則（cap = 1 之下）

`MambaCheckpointPublishDepths`（`forward_events.cpp:255-282`）先在**絕對**網格上
掃 `depth = stride, 2*stride, …`（`:267`），要求
`AlignMambaCacheSeqlen(depth - chunk_begin) == depth - chunk_begin`（`:271`，注意這裡是
**精確相等**，不是取整），最後無條件 `push_back(deepest)`（`:274`），
再由 `:278-280` 的 cap **保留最深的 max_count 個**。

cap = 1 ⇒ 只剩 `deepest`，也就是「貼著這個 chunk 自己的結尾、往下退一頁再對齊到 64」。
用 35 引用的 sparse-checkpointing 論文的分類法說：這既不是 `Balanced` 也不是 `Block`，
而是 `M = 1` 的退化解；而**當 stride 網格真的生效時（cap > 1），它是 `Block`
（`B, 2B, …`）而非 35 目前寫的 `Balanced`（`⌊i(L+1)/(M+1)⌋`）** —— 35 的標籤要修，
因為兩者在 Thm 1/Thm 2 下的保證不同。

`stride` 與 `cap` 可用 `TOKENSPEED_MAMBA_SNAPSHOT_STRIDE` /
`TOKENSPEED_MAMBA_SNAPSHOT_MAX` 覆寫（`forward_events.cpp:216-231`）；
`--mamba-track-interval` 這個 server arg 在這條路徑上**仍然沒有被讀**
（`forward_events.cpp:174-176`、`:214-215` 自陳）。

---

## 四個問題的答案

### 1. `_expand_multi_track` 到底做什麼、何時回 None、要怎樣才能寫滿每一個發佈深度

**它做什麼。**
`hybrid_linear_attn.py:1677-1744`。輸入是每個 extending row 的 `prefix`、`extend_lens`、
scheduler 給的 `(counts, flat_indices)` CSR 對，以及 `query_start_loc`。
它在 host 上把四個張量 `.tolist()` 出來（`:1698-1702`，**四次 device→host 同步**），
算出 flashinfer / FLA 兩種 checkpoint layout 的每序列起點（`:1706-1712`），
然後對每個 request 用 `self._publish_depths_for(...)`（`:1721-1723`）**重新推導**
C++ 當初發佈的深度列表，與 `slots` 逐位配對（`:1729`），對每一對產出
`(src, src_fla, dst, conv)` 四個扁平索引（`:1735-1739`）。
呼叫端在 `:1568-1584`；成功時 `:1584` 把 `track_mask` 清零，於是
`_compute_track_ssm_indices` / `_compute_track_conv_indices` 完全不跑，
multi 的索引就是最終使用的索引。

**何時回 None（三處）。**
- `:1724-1728` — `len(depths) != len(slots)`。兩側對「這個 chunk 值幾個深度」不同意，
  整個 step 退回單快照路徑。
- `:1741-1742` — 迴圈跑完 `dst` 是空的（每個候選都被 `:1731-1734` 的 `continue` 濾掉）。
- 呼叫端的 `use_multi` 閘（`:1561-1566`）要求
  `sum(multi_counts) > sum(track_indices >= 0)`；cap = 1 時兩者恆等，所以**這條路現在
  一次都不會被走到**。

**要讓它寫滿每一個發佈深度，缺的是四件事（依嚴重度）：**

- **(a) conv 快照的形狀是錯的。** `:1739` 每個快照只 append **一個** token index
  （`starts_l[i] + track_len - 1`），產出 1-D `[N]`；而單快照路徑的
  `_compute_track_conv_indices`（`:1660-1675`）產出的是 2-D `[N, conv_state_len]` 視窗。
  消費端 `:2797-2802` 做 `mixed_qkv_t[:, idx].transpose(0, 1)` 後直接寫進
  `conv_states[dst]`（形狀 `[N, conv_dim, state_len]`）。用 1-D 索引得到的是
  `[N, conv_dim]` —— 形狀不合，會拋錯或被 broadcast 成錯的視窗。
  **這是「單一 blocker」的真正內容：不是沒填滿，是填了也不對。**
  修法：把 `:1739` 換成與 `:1669-1675` 相同的 `arange(conv_state_len)` 視窗展開。
- **(b) `_publish_depths_for` 這面 Python 鏡子已經和 C++ 對不上了。**
  `:1746-1774` 自稱 "bit-for-bit mirror"，但：
  - `:1759` 的 `deepest = (final_len // page_size) * page_size - page_size` **沒有**做
    `e021b555` 加進 C++ 的 grain 向下取整（`forward_events.cpp:162`）。
    於是 Python 算出的 `deepest` 常常不是 64 的倍數 → `:1731` 的
    `track_len % FLA_CHUNK_SIZE` 判斷 `continue` → 該 slot **不寫**，
    但 C++ 已經把 radix 節點掛上去了 → 之後的命中從未初始化記憶體接續。
    這正是 `forward_events.cpp:190-199` 註解描述的失敗模式，而它**在修完之後仍然在程式碼裡**。
  - `:1763` 用模組常數 `MAMBA_SNAPSHOT_STRIDE_TOKENS = 512`（`:107`），
    不讀 `TOKENSPEED_MAMBA_SNAPSHOT_STRIDE`；C++ 讀（`forward_events.cpp:223-226`）。
    **一掃 stride 兩邊就靜默分家。**
- **(c) 「用 count 反推深度」這個設計本身有洞。** C++ 端
  `forward_events.cpp:446-449` 先 `DetachCheckpoints()` 拿到 `slots`，再用
  `max_count = slots.size()` **重算**一次深度；接著 `:457-481` 的迴圈裡，
  `SplitAt` 回 NULL 或節點已有 mamba 時（`:472`）**不會**呼叫
  `RecordPublishedCheckpoint`，但 `:481` 的 `++next_slot` 照跑。
  於是送到 GPU 的 `PublishedCheckpointIndices()` 可能**比 `publish_depths` 短**，
  GPU 再用這個較短的長度當 `max_count` 重算 → 得到**另一組深度** →
  位置配對錯位。修法只有一個方向：**把深度本身送過去，不要送數量**。
  `FlatForwardOperation` 已經有 CSR 形式的 `mamba_checkpoint_multi_counts` /
  `mamba_checkpoint_multi_indices`（`forward.h:111`、`:155-159`），
  再加一個平行的 `mamba_checkpoint_multi_depths` 即可，Python 端把
  `_publish_depths_for` 整個刪掉（不變式 2 從此由型別保證，而不是靠兩份程式碼自律）。
- **(d) 四次 `.tolist()`（`:1698-1702`）是每 step 的 host sync。** 在 `init_forward_metadata`
  裡，不在 48 層迴圈裡，所以是每步一次而非每步 48 次；但既然 (c) 之後深度是 C++ 送來的，
  這段可以整段改成純張量運算（`repeat_interleave` + `cumsum`），順手拿掉。

### 2. resume 有沒有把 causal conv1d 的視窗一起還原？——**確認：有。你的假設被駁回。**

**逐條追：**

- conv state 的形狀是 `(conv_dim/tp, linear_conv_kernel_dim - 1)`
  （`qwen3_5_text_base_config.py:329-332`），也就是 `state_len = kernel_size - 1`，
  與 kernel 內部的 `state_len = KERNEL_WIDTH - 1`（`causal_conv1d.py:84-86`）一致；
  kernel 讀取時以**最後一欄為最新 token**（`causal_conv1d.py:142`
  `prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok`）。
- 擷取：`_compute_track_conv_indices`（`hybrid_linear_attn.py:1660-1675`）取
  `start = query_start_loc[i] + track_len - conv_state_len`，展開
  `conv_state_len` 個連續 token，也就是**深度 D = prefix + track_len 之前的最後
  `kernel_size - 1` 個輸入**，最新的在最後一欄 —— 與 kernel 的慣例逐欄相符。
- 來源是**卷積前**的張量：`mixed_qkv_t = mixed_qkv.transpose(0, 1)`（`:2791`）在
  `causal_conv1d_fn`（`:2809`）之前被 gather（`:2797-2799`），寫入
  `conv_states[track_ssm_h_dst]`（`:2802`）。
- SSM 側在同一個 `track_len` 上取（`:1802-1803`），也就是**同一個深度**。
- 還原：`runtime_states.py:98-148 copy_mamba_states`，`:139-148` 對
  `conv_state` 與 `ssm_state` **各做一次** `fused_mamba_state_copy`，
  來源是 `mamba_cow_src_indices`（= radix 節點的快照 slot）。conv 沒有被漏掉。
- 邊界：`track_len >= FLA_CHUNK_SIZE = 64 > kernel_size - 1`（`:1546` 保證
  `track_lens > 0` 且是 64 的倍數），所以 `:1675` 的 `clamp` 不會把視窗切到別的 request。

**所以 0.04–0.07 的來源不是 conv，而是 SSM checkpoint 的 dtype。**

`gdn_chunk_prefill(output_h=True)` 在 sm_120 上走的是 **Triton/FLA** 而不是 flashinfer：
`flashinfer/gated_delta_rule.py:204` 的 `is_available()` 對 sm120 直接
`return os.getenv("TOKENSPEED_ENABLE_FI_GDN_SM120") == "1"`，**預設關閉**，
於是 `:259 if is_available()` 那個 `register_kernel` 根本不註冊。
兩條路的 checkpoint 精度不同：

| 路徑 | checkpoint buffer | 出處 |
|---|---|---|
| flashinfer sm10x/sm120（**預設關**） | **float32** | `flashinfer/gated_delta_rule.py:353-360` |
| Triton / FLA（**預設開，就是我們跑的**） | **bfloat16** | `chunk_delta_h.py:316 h = k.new_empty(B, NT, H, K, V)` |

`chunk_delta_h.py:316` 的 `h` 跟著 `k` 的 dtype（bf16），
而同一個函式的 `final_state` 是**明寫的 float32**（`:317-318`）；
kernel 內部累加器是 fp32（`:88-95 b_h1 = tl.zeros(..., tl.float32)`），
但在 `:140` 存進 `h` 時 `.to(p_h1.dtype.element_ty)` 被截成 bf16。

於是：**冷算**時遞迴狀態在 kernel 內全程以 fp32 帶過整段序列；
**resume** 時我們把一個已經被截成 bf16 的中間狀態升回 fp32
（`hybrid_linear_attn.py:3128 harvested.to(ssm_states.dtype)`，
`ssm_dtype` 預設 `float32`，見 `qwen3_5_text_base_config.py:344` 與 `env.py:293`）
再當作初始狀態。bf16 的 round-to-nearest 相對誤差上界 `2^-9 ≈ 0.2%`，
48 層 GDN 各自帶 0.2% 的狀態誤差，首 token logprob 差 0.04–0.07 的量級完全相符；
也解釋了為什麼**關快取時 bit-identical**（完全不經過 `h`）。

35 文件裡「`fi_checkpoints … float32`」那一句只對 flashinfer 那條路成立，
對我們實際跑的那條**不成立** —— 這是本次盤點對 35 的一項實質更正。

### 3. 遞迴快照的淘汰政策；快照會不會在 KV 還在的情況下被淘汰

**政策**：`MambaEvictionManager`（`mamba_eviction_manager.cpp`）。

- 候選集合是 `mamba_leaves_` —— **帶快照且沒有帶快照子節點**的節點
  （`:30-39 isMambaLeaf`）。也就是每條分支上**最深的那個快照**先被考慮。
- 排序是 LRU：`Time()` 舊者優先，平手用 `SeqId()`（`:77-80`，跨 TP rank 決定論）。
- 跳過條件只有兩個：device KV refcount > 0（`:87-89`）與 `MambaLockRef() > 0`（`:95-97`）。
- 淘汰時 `leaf->DetachMamba()`（`:106`），然後把 parent 升為新的 leaf 並**同一輪內
  繼續當候選**（`:110-118`）—— 一條分支可以被由深往淺連續剝掉。
- 入口 `EnsureCapacity`（`:123-128`），呼叫點在 `forward_events.cpp:301-315`
  （`AllocateMambaCheckpointOrEvict` / `AllocateMambaWorkingOrEvict`）與
  `forward.cpp:513-516`（每個 prefill request 預留 `2 + loadback` 個 slot）。
- 淘汰的是 **checkpoint pool**（若有設定），見 `hybrid_prefix_cache.cpp:104-107`：
  `mamba_eviction_manager_` 綁的是 `mamba_ckpt_allocator ? ckpt : working`。

**會不會 KV 還在、快照沒了 —— 會，而且是單向的。**

- KV → mamba 方向有掛勾：`OnKVEvict`（`hybrid_prefix_cache.cpp:549-564`）與
  `OnNodeDestroyed`（`:566-596`）都會 `DetachMamba()`，所以**不會**留下沒有 KV 的孤兒快照。
- mamba → KV 方向**沒有任何掛勾**：`Evict` 只做 `DetachMamba()`，
  KV 頁一頁不動。之後 `augmentMatch` 從 KV 終點往上找不到快照，就照
  `:197-205` 把整段匹配丟回 root。**那段 KV 從此只佔位、不可能再被 hybrid 命中**，
  直到它自己被 KV LRU 淘汰為止。這就是問題 3 問的那個浪費，答案是肯定的。
- 而且方向是最壞的：候選集是 **leaf**，也就是**最深**的快照先走，
  而深的快照正是對未來匹配最值錢的那個（`forward_events.cpp:276-280` 的 cap
  也是基於同一個判斷保留最深的）。淘汰政策與發佈政策對「深度價值」的認定一致，
  但淘汰順序與它相反。

### 4. 對 `MambaCheckpointPublishDepths` 最小的改動，能比現在的等距 stride 放得更好

先把座標定清楚：論文的 `Block` = 絕對網格 `B, 2B, …`，正是 `forward_events.cpp:267`
現在做的事；`Balanced` = `⌊i(L+1)/(M+1)⌋`，隨 L 伸縮，我們**沒有**實作。
Thm 1 說 Balanced 在**均勻重疊**下最佳；Fig 1 說 QuALITY（= 我們的 RAG 形狀）
的重疊分布在**文件全長附近有尖峰**。所以正確方向不是把 Block 換成 Balanced，
而是**把預算堆到分岔深度附近**。

**最小改動：把「分岔深度」當成一個必選深度餵進去。這個數字引擎已經算好了。**

`augmentMatch` 在 KV 匹配比最深快照更深時，會設
`match.mamba_branching_seqlen = AlignMambaCacheSeqlen(kv_depth * page_size)`
（`hybrid_prefix_cache.cpp:209-214`，另兩處 `:177-182`、`:252-257`）。
這個值的語意就是**「這一次重用是在哪個深度斷掉的」**。
它一路經 `forward.cpp:1034`、`forward.h:160`、`python_module.cpp:408`、
`input_buffer.py:473-487` 送到 GPU，然後在
`hybrid_linear_attn.py:3253-3260` 被歸進 `_MAMBA_KWARGS` 之後——**沒有任何地方讀它**。
它是一條已經拉通、但沒有接上負載的線。

具體最小 diff：

```cpp
// forward_events.cpp:255  簽章多一個參數
std::vector<std::int32_t> MambaCheckpointPublishDepths(..., std::int32_t branch_depth);

// :274 之前插入
if (branch_depth > chunk_begin && branch_depth < deepest &&
    hybrid_cache->AlignMambaCacheSeqlen(branch_depth - chunk_begin) == branch_depth - chunk_begin) {
    depths.push_back(branch_depth);
}
// :278-280 的 cap 改成「一定保留 deepest + branch_depth，其餘才從深到淺砍」
```

呼叫端 `InsertHybridCache`（`:434`、`:447`）多傳一個
`match_result.mamba_branching_seqlen`（它已經在 request 上）。

為什麼這是最小且方向對的：它不需要直方圖、不需要 DP、不需要離線求解，
就把 Thm 2 想解的那個「重疊分布」用**上一個請求實際斷在哪**這個一階觀測近似掉，
而且是**單一請求就閉環**（第 N 個問題斷在文件尾，第 N+1 個問題就有祖先），
而 Block 網格要靠 512 的倍數碰巧落在文件尾才行。

次小的兩個候選（若上面不夠）：
- **把網格錨在 `chunk_begin` 而非 0**：`:267` 改成
  `for (d = chunk_begin + stride; d < deepest; d += stride)`。一行。
  對第一個 chunk 完全等價，對續傳 chunk 讓 stride 描述「本次新算的量」而非絕對位置。
- **cap 改成保留 `{deepest} ∪ 最淺的 max_count-1 個`**（`:278-280`）。
  `deepest` 服務完全重複，淺的服務分岔；現在兩種都留深的，等於把預算重複投在同一件事上。
  註：`forward_events.cpp:200-203` 記錄「cap 6 與 cap 1 的命中數與命中深度相同」，
  但那是在中間深度**根本沒被寫入**的前提下量的（見問題 1），不能當作
  「中間深度沒有價值」的證據。

---

## 依序的修改提案

順序是刻意的：**先讓現有這一條路是對的，再讓它變多，最後才調位置。**
（`AGENTS.local.md`：port first, then optimize；以及每一項都要先對照參照引擎。）

### P0-1　把發佈用的 SSM checkpoint 拉回 fp32

- **改哪裡**：`chunk_delta_h.py:316`。兩種做法：
  (a) 一行 —— `output_h` 時 `h = k.new_empty(..., dtype=torch.float32)`；
  (b) 目標版 —— 另外開一個只有「要發佈的那幾個深度」那麼大的 fp32 buffer，
  在 kernel 的 `:140` 存 `h` 之外，於 `i_t == ckpt_chunk_idx` 時再存一份 fp32。
- **預期收益**：關掉 0.04–0.07 的 logprob 漂移；讓論文 Remark 1 的
  「從快照重放是輸出精確的」在我們這裡真的成立。**這是決定整條重用路徑
  「對不對」的那一項，收益不是速度而是正確性。**
- **風險**：(a) 讓每層 prefill 多寫一份 fp32 `h`（`NT × H × K × V × 4B`），
  是純頻寬成本，48 層會顯著；且 `h` 也被 `chunk_o` 消費，改 dtype 會同時改變
  **冷算**的輸出（可能變好，但等於改了 baseline，A/B 必須重跑）。
  (b) 需要動 Triton kernel 簽章，但不影響冷算路徑。**建議先用 (a) 做一次
  一小時的判定實驗（確認誤差消失），再以 (b) 出貨。**
- **量測**：`max_tokens=1` 的 logprob 對照，cache on/off，同一 prompt 重複多次。
  不要看 tok/s。

### P0-2　修 `_expand_multi_track` 的 conv 形狀，並廢掉 Python 那面鏡子

- **改哪裡**：
  1. `hybrid_linear_attn.py:1739` → 展開成 `[N, conv_state_len]` 視窗（照 `:1669-1674`）。
  2. C++ 新增 `mamba_checkpoint_multi_depths`（`forward.h:111`/`:155-159` 旁邊、
     `python_module.cpp:404` 旁邊、`input_buffer.py:494-518` 旁邊），
     由 `forward_events.cpp:472-479` 在**真的 attach 成功**時一併記錄深度。
  3. 刪掉 `_publish_depths_for`（`:1746-1774`），`_expand_multi_track` 改吃送來的深度。
  4. 順手把 `:1698-1702` 的四次 `.tolist()` 換成張量運算。
- **預期收益**：解除 cap > 1 的封鎖。它本身不提升重用，
  但 P1-1 與 P2-1 都以它為前提。
- **風險**：`:1724-1728` 那個「數量不一致就整步退回」的保險在深度直送後失去意義，
  要改成 assert；配對錯位是**靜默且會產生流暢錯字**的失敗模式，
  必須有 `TOKENSPEED_LOG_PUBLISH_DEPTH=1`（`:86-100`）與
  `TOKENSPEED_LOG_SNAPSHOT_PUBLISH=1`（`forward_events.cpp:237-240`、`:450-455`、`:467-471`）
  兩邊逐行對讀的驗收步驟，不能只看 reuse%。

### P0-3　把 stride / cap 接上真正的 server arg

- **改哪裡**：`forward_events.cpp:216-231` 的 `EnvInt` 兩個 getter 改讀
  `SchedulerConfig`；`--mamba-track-interval` 目前在這條路徑上被忽略
  （`:174-176` 自陳）。Python 端 `MAMBA_SNAPSHOT_STRIDE_TOKENS`（`:107`）
  在 P0-2 之後就不再需要，一併刪除。
- **預期收益**：可掃參數；**同時消除「用 env 掃 stride 會讓兩側靜默分家」這個地雷**
  （C++ 讀 env，Python 讀常數）。
- **風險**：低。但 `AGENTS.local.md` 的「印出 resolved 值」要照做——
  現在沒有任何一行 log 印出生效的 stride/cap。

### P1-1　分岔對齊的發佈深度（把 `mamba_branching_seqlen` 接上）

- **改哪裡**：如問題 4 的 diff；`forward_events.cpp:255-282` 與 `:434`/`:447`。
- **預期收益**：這是四個問題裡最直接指向「44% → 74.4%（batch-aware 天花板）」的一項。
  現在 cap=1 之下每個請求只在自己結尾留一個快照，對**在文件尾分岔**的下一個請求
  永遠是子孫；接上之後第二個請求就有祖先可用。
- **風險**：
  (a) 需要 cap ≥ 2（依賴 P0-2），否則新深度會被 `:278-280` 砍掉；
  (b) `branch_depth` 來自 match，跨 chunk 續傳時要確認它相對於 `chunk_begin` 的語意
      （`forward.cpp:421-425` 在沒有 match 時會拿 `tokens_this_round` 頂上，
      那個值**不是**分岔深度，接線時必須把這兩種來源分開，否則會在錯的地方下錨）；
  (c) 每請求多一個 slot，直接加重問題 3 的淘汰壓力 —— 所以 P1-2 要一起做。

### P1-2　淘汰政策：不要留下「有 KV、沒快照」的死頁

三個可獨立採用的動作，由小到大：

1. **量出來再說**：`MambaEvictionManager::Evict`（`:106`）加一條計數/trace，
   數「被淘汰的快照其 KV 仍在 device」的次數。現在完全沒有這個數字，
   而 C=32 的 40.5%（對 74.4% 天花板）**有可能整段就是這件事**。
2. **順序反過來**：候選集用 `mamba_leaves_`（`:30-39`）＝先砍最深的。
   改成優先砍**淺**的、或在 `older` 比較子（`:77-80`）裡加入深度項。
   一行等級的改動，但語意要對照 Marconi 的 FLOP-aware 淘汰再定
   （35 §6 的 P4，有現成 reference implementation）。
3. **對稱化**：`DetachMamba` 時一併把該節點以下的 KV 交還，或反過來
   標記該子樹為 hybrid-unmatchable。這個改動大，先不做。

- **預期收益**：只在高併發成立；C=1 的 82.0% 不會動。
- **風險**：把淘汰改成砍淺的，會讓長文件的深快照長期佔住 checkpoint pool；
  在 `ckpt_pool` 尚未開 int8（`SimpleMambaPool.ckpt_pool`，
  `hybrid_linear_attn.py:519-521`）之前，pool 只有 221 slot，容易反向惡化。
  **先做動作 1，拿到數字再決定 2。**

### P2-1　把 cap 提到 5–6，並讓 cap 保留 `{deepest} ∪ 最淺的幾個`

- 依賴 P0-2 全部完成、且 `TOKENSPEED_LOG_SNAPSHOT_PUBLISH` 對讀過。
- 預期收益：35 的 stride 表說 G=512 在 2690-token 前綴上最差重用 2048/2690。
- 風險：pool 壓力乘以 cap（`forward_events.cpp:286-293` 記載過 cap=6 在 64 併發
  unique-prompt 形狀上打爆 pool 的事故）。與 P1-2 綁在一起評估。

### P2-2　重疊深度直方圖（35 的 P2）＋ int8 checkpoint pool

- 直方圖：`TOKENSPEED_LOG_MATCH_DEPTH`（`hybrid_prefix_cache.cpp:146-149`、`:192-196`）
  已經印出 `kv_tokens` 與 `mamba_tokens` 兩欄，兩者的差就是每次損失的深度。
  聚合成直方圖是**純 log 分析**，不動 scheduler，是 Thm 2 DP 版的唯一缺口。
- int8：`ckpt_pool` 的掛勾已在（`hybrid_linear_attn.py:519-521`、`:2800-2807`、
  `:3126-3137`、`runtime_states.py:118-136`），開起來是 4x 容量。
  注意 `AGENTS.local.md` 記載 `--enable-int8-mamba-checkpoint` 與
  `--enable-hierarchical-cache` 目前互斥。
- 這兩項做完才談 DP-optimal；在那之前 DP 沒有輸入。

### 不做（並記錄理由）

- **改 `augmentMatch` 的結構**（35 §不變式 3、`AGENTS.local.md` 提到的 SGLang
  `MambaRadixCache` 折進 descent）：現在 `:207-216` 已經是**截斷**而不是丟棄，
  只有「整條路徑上沒有任何祖先快照」時才歸零，而那種情況下確實無狀態可用。
  折進 descent 的收益在本 build 上已經大部分被 `e021b555` 吃掉，
  改動範圍卻很大。先不做，但若 P1-1 之後仍有大量 `mamba=MISS`，回頭再議。

---

## 無法從原始碼確定的部分

1. **P0-1 的機制是否解釋了全部 0.04–0.07。** 原始碼只能證明「resume 用的是 bf16 截斷過
   的狀態，冷算不是」，量級相符但不是證明。判定實驗：`chunk_delta_h.py:316` 一行改 fp32，
   重跑 `max_tokens=1` 的 logprob 對照。**若誤差沒有消失，第二嫌疑是
   `need_h_track` 走 `output_h=True` 分支（`hybrid_linear_attn.py:2999-3012`）與
   `output_h=False` 分支（`:3090-3102`）是不同的 kernel 路徑，兩者的 `out` 可能本來就不同。**
2. **實測時 `TOKENSPEED_ENABLE_FI_GDN_SM120` 到底是不是 1。** 若當時是開的，
   走的是 flashinfer 的 fp32 checkpoint，本文問題 2 的機制就不成立。
   照 `AGENTS.local.md`「印出 resolved 值」：先確認跑那次 44% 的 server 環境變數。
3. **量到 44% 的那個 RAG 形狀的 prompt 長度與 `chunked_prefill_size`。**
   cap=1 之下唯一的快照貼著 chunk 結尾；C=1 能到 82.0% 只有在
   **prompt 被切成多個 chunk**（於是前面的 chunk 各自在文件中段留下快照）時才說得通。
   若實測是單 chunk，那 82.0% 另有來源，P1-1 的收益估計就要重算。
4. **`PublishedCheckpointIndices()` 實際上有沒有比 `publish_depths` 短。**
   問題 1(c) 是從程式碼推出來的可能性；`TOKENSPEED_LOG_SNAPSHOT_PUBLISH=1`
   的 `depth=… split=… has_mamba=…` 三欄（`forward_events.cpp:467-471`）就能證實或排除。
5. **快照被淘汰的實際頻率**（問題 3 的浪費有多大）。目前沒有任何計數器。
6. **decode 期間的第二條發佈路徑**：`model_executor.py:1140-1200`
   `_snapshot_mamba_checkpoints` 會在 page 邊界把 working state 複製到 checkpoint slot。
   它用的是 `config.block_size`（16）而不是 64 的 scan grain。
   這條路徑有沒有對應的 radix attach、深度語意是否與 prefill 側一致，本次沒有追到底，
   但它是「深度被歸檔在它不描述的位置」的另一個潛在來源，**在 P0-2 之前應該先釐清**。
7. **35 文件的兩處需要更正**（本文不改該檔）：
   (a) 「`fi_checkpoints` float32」只對 flashinfer 路徑成立，我們跑的 FLA 路徑是 bf16；
   (b) 「我們實作的是 `Balanced`」按論文定義應為 `Block`，而 cap=1 之下連 Block 都不是。
