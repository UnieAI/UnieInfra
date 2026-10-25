# 41 — Decode attention 的 KV 軸切分（flash-decoding split）五方橫向對照

版本見 `00-METHOD.md`。所有行號只對該表的 pinned commit 有效。

範圍：**decode 時把一條序列的 KV 軸切成多段、各段各自算部分 softmax、再做第二階段
歸約**這一個機制——誰決定切幾段、用什麼決定、在哪裡決定。

為什麼是這一章：`40-decode.md` 記的是 decode 走哪一顆 kernel，沒有記 kernel 拿到的
grid。gemma-4-31B 的 10 個全域層是 `32 q × 4 kv × 512`，head_dim 512 不被任何特化
kernel 接受（trtllm XQA 上限 256、FA4 上限 128），只剩 Triton 這條路，而在那條路上
**這個 grid 就是效能本身**。

---

## 0. 一句話結論

**四個參照引擎全部都切，我們固定不切。**

`mha_decode.py:915` 是 `num_kv_splits = torch.ones(...)`，`max_kv_splits = 4` 只用來
配置緩衝區。實際 grid 是 `(total_q, cdiv(q_heads, kv_group), 1)`，gemma-4 全域層在
batch 1 是 **16 個 block**，在 188 個 SM 的卡上是 **8.5%**，每個 block 依序走完 26k
context 的全部 813 個 KV block。

這不是「我們的啟發式比較保守」，是**啟發式不存在**。這顆 kernel 是從 SGLang 移植的，
kernel 收下來了，餵它 grid 的那一段沒有跟著移植。屬於 `AGENTS.local.md` 的
「capability gap 先於 efficiency gap」。

---

## 1. 步驟表

| 步驟 | UnieInfer | vLLM (Triton) | SGLang (Triton) | TRT-LLM (MMHA) | TurboMind |
|---|---|---|---|---|---|
| 切分數的上限 | `mha_decode.py:899` `max_kv_splits = 4`（常數，只配緩衝） | `triton_attn.py:56` `NUM_PAR_SOFTMAX_SEGMENTS = 16` | `server_args.py:1908` `triton_attention_num_kv_splits = 8`（HIP 為 16，`:5954`） | `params.max_seq_len_tile`（由 workspace 推） | `unified_attention_layer.cc:601` `max_split_k = min(max(1, kMaxWorkspaceTokens/token_num), max_kv_splits)` |
| 誰決定實際切幾段 | **無** —— 恆為 1 | 主機端常數 | **裝置端 Triton kernel**，逐請求 | 主機端 `multi_block_grid_setup()` | 主機端 `GetSplitCount()` |
| 決定的位置 | `mha_decode.py:915` | `triton_attn.py:154` | `metadata.py:12` `get_num_kv_splits_triton()`，由 `triton_backend.py:323` 呼叫 | `decoderMaskedMultiheadAttentionLaunch.h:109` | `utils.cc:11`，由 `decoding_template.h:50` 呼叫 |
| 依據 batch/head 的基礎 grid | — | `num_seqs`（只當作閘門） | `token_grid = num_seq × num_group × cdiv(num_head, block_h)`，`block_h = min(16, kv_group)`（`metadata.py:41-47`） | `batch_size × num_heads`（`:110`） | `grid_size`（由 CtaMap 給） |
| 依據 SM 數 | — | **否** | `device_core_count`（`triton_backend.py:228`） | `multi_processor_count`（`:110`） | `sm_count`（`decoding_template.h:50`） |
| 依據每 SM 可同時常駐幾個 CTA | — | 否 | 否 | **是** `blocks_per_sm`（`:110`） | **是** `max_active_ctas`（`utils.cc:15`） |
| 依據 context 長度 | — | 否 | **是**：`ext_device_core_count = core_count × max(log2(seq/64), 1)`（`metadata.py:37-40`） | 是（`max_seq_len_tile = min(divUp(tlength+1, seq_len_per_kv_loop), ...)`，`:127`） | 是（`tile_count = cdiv(min(cp_k_len, window), CTA_S)`，`decoding_template.h:29`） |
| 每段的工作量下限 | — | 否 | 由 `kv_chunk_size` 隱含 | **是**：`seq_len_per_kv_loop = divUp(block_size, threads_per_value) × 8`（`:113`） | 由 `tile_count` 隱含 |
| 逐請求 or 整批一個值 | 整批（都是 1） | 整批 | **逐請求**（`metadata.py:53-55`） | 整批 | 整批 |
| 大 batch 時退回不切 | — | **閘門**：`num_seqs > 128/num_kv_heads` 就走 2D kernel（`triton_attn.py:139`, `ops/triton_unified_attention.py:1041-1049`） | 由公式自然收斂到 1 | 由公式自然收斂 | 由 wave 成本模型自然收斂 |
| 是否含 prefill 就停用 | — | **是**：`max_seqlen_q > 1` 直接走 2D（`:1047`） | 分開的 extend/decode kernel | MMHA 本來就只做 decode | decoding kernel 只做 decode |

---

## 2. 分歧點

### 2.1 四家用四種不同的量來決定同一件事

| 引擎 | 決定式（化簡後） | 這個式子在說什麼 |
|---|---|---|
| vLLM | `16`，若 `batch ≤ 128/kv_heads` | 「小 batch 就切固定 16 段，大 batch 不值得」 |
| SGLang | `cdiv(SM × max(log2(seq/64),1), token_grid)`，上限 8 | 「context 越長，願意動用的 SM 越多」 |
| TRT-LLM | `divUp(SM × blocks_per_sm, batch × heads)`，且每段至少一個 KV loop | 「填滿佔用率，但不要切到每段沒事做」 |
| TurboMind | `argmin_s (α/s + β) × ceil(grid×s / (SM×ctas))`，`s` 使 wave ≤ 4 | 「切分只在不多開一個 wave 時才划算」 |

**只有 TurboMind 明確地把 wave 的離散性寫成成本**，其餘三家都是連續除法。

這一點直接解釋了本地一個實測異常：未提交的補丁按 `SM // base_blocks` 算出 12 段，
而 12 實測 **0.776 ms/layer**，8 是 **0.569**、16 是 **0.580**——12 比兩邊都差。當時
的處置是「向下取 2 的冪」，那是把現象壓掉。TurboMind 的模型說得出**為什麼**：12 段
讓 `ceil(grid × s / (SM × ctas))` 跨進下一個 wave，尾端那個 partial wave 的成本由整個
第二階段歸約等待。**2 的冪不是原因，wave 邊界才是**，而 wave 邊界只有在
`SM × max_active_ctas` 不是 2 的冪時才會與 2 的冪分道揚鑣。

### 2.2 佔用率（`blocks_per_sm` / `max_active_ctas`）：兩家用、兩家不用

TRT-LLM 與 TurboMind 都用「每個 SM 能同時常駐幾個 CTA」當乘數；vLLM 與 SGLang 都不用。

後果是方向性的：不乘佔用率 **等於假設每個 SM 只跑一個 CTA**，於是系統性地**少切**。
gemma-4 全域層 `BLOCK_DMODEL=512`、4 warps，若每 SM 能常駐 2 個 CTA，TRT-LLM 式的
公式會算出 SGLang 式公式兩倍的段數。

未提交的本地補丁用的是 `SM // base_blocks`，**沒有佔用率乘數**，所以它與 SGLang 同
一邊——而且它連 SGLang 的 `log2(seq/64)` 長度縮放都沒有。它是四家裡最保守的那一個
的再保守版。

### 2.3 逐請求 vs 整批一個值：只有 SGLang 逐請求

SGLang 的 `num_kv_splits` 是一個 **per-request 的張量**，值為
`max(cdiv(seq_len_i, kv_chunk_size_1), cdiv(seq_len_i, kv_chunk_size_2))`
（`metadata.py:53-55`），其中 `kv_chunk_size_1` 由**該批次內最長與最短序列的比值**推出
（`metadata.py:31-34`）：批次越參差，切分越積極。

其餘四家（含我們）都是整批一個值。**agent 形狀正是最參差的形狀**——同一批裡有
127k 的與 2k 的——所以這個分歧點對本專案的目標場景不是細節。

### 2.4 「不切」在四家裡都是一個會被算出來的結果，不是一個預設

vLLM 用閘門明說（`num_seqs > 128/kv_heads` → 2D），另外三家讓公式收斂到 1。
我們是反過來的：**1 是寫死的，其他值不可達**。差別在於前者在 batch 64 上
「算出來不切」，後者在 batch 1 上「也不切」——而 batch 1 正是切分價值最大的地方。

### 2.5 每段的工作量下限：只有 TRT-LLM 寫成硬條件

`seq_len_per_kv_loop = divUp(block_size, threads_per_value) × 8`（`:113`），註解寫明
「Make sure that each block at least processes one loop of kv」。切到每段不足一個
展開迴圈時，第二階段的歸約成本會蓋過第一階段省下的時間。SGLang 靠 `kv_chunk_size`
的除法隱含地做到，TurboMind 靠 `tile_count` 上限做到，vLLM 的固定 16 **沒有**這個
保護——短 context 的小 batch 在 vLLM 上會切得過細。

---

## 3. 結構差異（同功能、不同層）

| | 決定切分數的程式在哪一層 |
|---|---|
| UnieInfer | 不存在 |
| vLLM | attention **backend**（`triton_attn.py`），與 metadata 一起建，CUDA graph 捕捉時還會把閘門對齊到 capture size（`:146-152`） |
| SGLang | attention backend 呼叫，但**計算本身在 GPU 上**（一個 `(1,)` grid 的 Triton kernel），因此不需要把 `seq_lens` 搬回主機 |
| TRT-LLM | **kernel launch 函式內**（`multi_block_grid_setup`），與 grid 一起算 |
| TurboMind | 獨立的**成本模型函式**（`utils.cc`），prefill 與 decode 各自呼叫，只差 `max_wave_cnt`（8 對 4） |

SGLang 把它放到裝置端是唯一避開 D2H 同步的做法；其餘三家都在主機端，代價是需要主機
側就知道 `seq_lens`（vLLM/TRT-LLM 本來就有 CPU 副本，TurboMind 用的是 grid 大小而非
序列長度，所以也不需要）。

**如果我們要逐請求（§2.3），SGLang 的裝置端做法是唯一不引入每步同步的路。**

---

## 4. 可量測候選（依預期效益排序）

1. **切分數本身**（已量，未提交）：gemma-4 全域層、head_dim 512、26k context、batch 1，
   1 段 2.007 ms/layer → 8 段 0.568 ms/layer，**3.54x**，HBM 達成率 23% → 83%。
   這是把不存在的機制補上，不是調參。
2. **佔用率乘數**（§2.2）：把 `SM // base_blocks` 換成 TRT-LLM 式
   `divUp(SM × blocks_per_sm, base_blocks)`。需要先量 gemma-4 兩個幾何的
   `max_active_ctas`（`cudaOccupancyMaxActiveBlocksPerMultiprocessor`）。
3. **wave 成本模型**（§2.1）：用 TurboMind 的 `GetSplitCount` 取代「向下取 2 的冪」。
   假設：12 段的異常會消失，且在 `SM × ctas` 不是 2 的冪的形狀上會勝過取冪。
4. **逐請求切分**（§2.3）：SGLang 的 `get_num_kv_splits_triton` 直接移植。
   agent 形狀（批內長度差 60 倍）是它與整批單值差最多的地方。
5. **長度縮放**（SGLang `log2(seq/64)`）與 **每段工作量下限**（TRT-LLM）：
   前者在 127k context 上決定會不會切到上限，後者保護短 context 的小 batch。

前三項不改變數值語意（切分只改變歸約順序，實測與 1 段參考差 2.44e-04，bf16 的一個
ULP）；第 4 項會讓同一批不同請求走不同段數，需要另外驗證 batch-invariance 的預期。

---

## 5. 量到之後：TurboMind 的成本模型在我們的 kernel 上是錯的，而佔用率除法是對的

`unieinfer/bench/kernel_gemma_decode.py`，`SPLIT_SWEEP=1,2,4,8,11,16,24,32`，
gemma-4-31B 的兩個真實 decode 幾何，device 時間（µs/layer）：

| 幾何 | batch | ctx | s1 | s2 | s4 | s8 | s11 | s16 | s24 | s32 | 最佳 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sliding | 1 | 1024 | 49.4 | 26.8 | 16.2 | 11.7 | **11.5** | 12.4 | 13.5 | 17.7 | s11 |
| sliding | 16 | 1024 | **183.3** | 184.0 | 185.6 | 188.6 | 191.0 | 194.2 | 196.3 | 207.8 | s1 |
| sliding | 64 | 1024 | **710.6** | 717.6 | 730.0 | 747.0 | 756.5 | 766.8 | 772.2 | 813.5 | s1 |
| global | 1 | 4096 | 294.4 | 149.6 | 79.0 | 43.5 | 35.2 | 27.4 | 24.4 | **22.3** | s32 |
| global | 1 | 26000 | 2044.4 | 1019.6 | 519.7 | 270.8 | 216.7 | 165.0 | **155.9** | 157.4 | s24 |
| global | 4 | 26000 | 2048.1 | 1051.9 | 605.6 | **568.2** | 570.0 | 577.1 | 638.6 | 587.6 | s8 |
| global | 16 | 26000 | 2373.0 | **2243.0** | 2243.8 | 2246.7 | 2252.2 | 2263.1 | 2315.4 | 2302.1 | s2 |
| global | 32 | 26000 | **4452.7** | 4470.5 | 4465.9 | 4487.6 | 4512.4 | 4531.4 | 4591.7 | 4620.3 | s1 |

（batch 4 那幾列的 GB/s 超過 100%——工作集小於 128 MB 的 L2，那不是頻寬數字。
bench 現在會在該列印 `L2!` 與工作集大小。）

**不切的代價比先前記錄的大得多**：global、26k context、batch 1 是
**s1 2044 µs 對 s24 156 µs = 13.1x**，不是 3.54x。3.54x 是拿 s8 對 s1 量的，而
s8 本身就被舊啟發式的 16 上限與 2 的冪取整壓過。

### 5.1 wave 成本模型不轉移（不要移植它）

把 `utils.cc:GetSplitCount` 照抄（`ctas_per_sm=1`、`max_waves=4`）後的預測對照實測：

| 幾何 / batch | 模型給的 | 實測最佳 | 模型那一格的實測 | 代價 |
|---|---|---|---|---|
| sliding / 1 | 11 | 11 | 11.5 | **吻合** |
| global / 1 / 26k | 16 | 24 | 165.0 對 155.9 | 6% |
| global / 16 / 26k | 11 | 2 | **81.4 對 60.3** | **35%** |

`α/s` 假設切分把工作完美均分。我們的 Triton 路徑不是：切分之後 stage-2 是**另一顆
kernel**，而每段的 KV 變小時它的固定成本壓過省下的時間。TurboMind 的模型是為它自己
那顆融合 kernel 的成本結構寫的。

**這一條是「survey 之後仍要量」的實例**：四家的分歧點指出了正確的機制（要切），但
其中一家的**係數**不能直接搬。

### 5.2 對得上資料的是最單純的那條：佔用率除法

`splits ≈ clamp(SM / base_blocks, 1, cap)`，也就是 SGLang 與 TRT-LLM 都在做的除法：

| 幾何 / batch | base_blocks | 188 / base | 實測最佳 |
|---|---|---|---|
| sliding / 1 | 16 | 11.75 | 11 |
| sliding / 16 | 256 | 0.73 | 1 |
| global / 1 | 4 | 47 | 24–32（被 cap 擋住） |
| global / 4 | 16 | 11.75 | 8–11 |
| global / 16 | 64 | 2.9 | 2 |
| global / 32 | 128 | 1.5 | 1 |

**樹裡現有的啟發式已經在算這個除法**；它只有兩個偏差，而兩個都很小：

- **2 的冪取整**：sliding/1 是 11 → 8，11.7 對 11.5，**2%**；global/4/26k 是 11 → 8，
  而 8 實測反而比 11 好。**取整幾乎無代價，而先前記錄的「12 比 8 和 16 都差」在這批
  資料裡不成立**（s11 與 s16 在每一列都相鄰且平順）。
- **cap = 16**：只有在 `base_blocks` 很小又 context 很長時咬人。global/1/4096 是
  s16 27.4 對 s32 22.3（**23%**），global/1/26k 是 165 對 156（6%）。

### 5.3 端到端還剩多少：幾乎沒有

gemma-4 一步 60 層。batch 1、26k context：`50 × 11.5 + 10 × 156 = 2.1 ms`，
而整步是 45.5 ms —— attention 佔 **4.6%**。把 cap 從 16 提到 32 只省
`10 × 9 µs = 90 µs`，**佔一步的 0.2%**。

反過來看才是重點：不切的話會是 `50 × 49.4 + 10 × 2044 = 22.9 ms`，**半步**。
所以樹裡那個未提交的切分修正是**承重的**（decode 在 c=1 從約 65 ms 降到 45.5 ms），
而它已經被計入本專案目前所有的 gemma 數字。

**結論：decode 的 KV 切分這條路已經走完了。** batch ≥ 16 時 attention 本來就在
read roofline 的 96–100%，batch 1 時整步貼著權重讀取的下限。**不要再花時間在這裡**；
剩下的空間在 prefill、前綴重用與容量（fp8 KV）。
