# Gemma on TokenSpeed — 量測記錄

分支 `roy/gemma-swa-kv` 的對應工作在 vLLM 側；這份記的是 TokenSpeed 側。
機器：RTX PRO 6000 Blackwell (sm_120)，容器 `unieinfra-dev`。
`unieinfra/experiment.md` 是另一個 session 的矩陣，這份不動它。

## 硬體天花板（引擎無關，兩邊共用）

空載實測，探針在卡上超過 2 GiB 時拒絕量測：

| | 實測 | 理論 | 達成 |
|---|---|---|---|
| HBM（2 GiB copy） | 1,461.8 GB/s | — | — |
| GEMM bf16 (8192³) | 413.0 TFLOP/s | 188 SM × 2.43 GHz × 1024 = 467.8 | **88%** |
| GEMM fp8_e4m3 | 726.2 TFLOP/s | 935.6 | 78% |

**這些數字對 TokenSpeed 同樣成立**，所以換引擎不會變出空間；能變的只有「離天花板多遠」。

## 已修：Gemma 的層標籤從 text_config 讀（commit 09f124ac）

`configs/mha.py` 從頂層 `hf_config` 讀 `layer_types` 與 `sliding_window`。Gemma 3/4 是
`*ForConditionalGeneration`，兩個欄位**只在 `text_config`**，頂層都沒有 → 取到空 → pool 只
發佈一個 full-history 群組 → 60 層全部按最寬幾何收費。

```
before  parent_bytes=62914560  parents=450   token_capacity=28800
        groups={'full_attention': 1}
after   parent_bytes=10485760  parents=2709  token_capacity=173376
        groups={'full_attention': 1, 'sliding_attention_0..4': 1 each}
```

**KV token 容量 6.02x。** 等深拆分（`split_groups_to_equal_depth`）本來就在，只是
`_mha_fields` 的 SWA 路徑閘在 `config.layer_types` 非空，而它從來就是空的。

### 它的效果，以及為什麼不能只看 goodput

RAG c=64、n=64、串行相鄰：

| | 修正後 | 修正前 |
|---|---|---|
| tok/s | 480.94 | 330.59 |
| TPOT p50 | 96.08 ms | 123.30 ms |
| TTFT p50 | 6,925 ms | 4,741 ms |
| goodput | 0.06 | 0.16 |

**兩個 arm 都不滿足 SLO（ttft:5000 tpot:100），失敗只是換邊**——修正前死在 TPOT，修正後死在
TTFT。所以那不是回歸，是**這個併發沒有可行點**，兩個 goodput 都在描述失敗。正確的下一步是找
出可行點在哪個併發，而不是比較兩個失敗點的比值。

同一格重跑兩次還給出 goodput 0.42 對 0.06（TTFT 5,682 對 6,925，都貼在 5,000 門檻上）——
**貼著 SLO 邊界的格子，goodput 對微小擾動極度敏感**，不可引用。

### 併發包絡：完整曲線

RAG 形狀、n=64、SLO `ttft:5000 tpot:100`、串行相鄰、每格獨立啟動。

| c | goodput (req/s) | 通過率 | P99 TTFT | P99 TPOT | tok/s |
|---|---|---|---|---|---|
| 1 | 0.08 | 100% | — | 51.2 | 20.6 |
| 4 | 0.31 | 100% | 1,098 | 49.3 | 78.8 |
| 8 | 0.58 | 100% | 1,549 | 51.8 | 147.3 |
| 16 | 1.02 | 100% | 2,911 | 58.7 | 262.3 |
| **32** | **1.71** | **100%** | **4,414**（餘裕 12%） | 73.2 | 437.8 |
| 40 | 1.70 | 100% | 4,961（餘裕 **0.8%**） | 77.5 | 435.6 |
| 48 | 1.33 | 部分 | **5,330（超標）** | 82.0 | 428.8 |
| 64 | 0.06 | 6% | — | 96.1（中位） | 480.9 |

**操作點是 c=32。** c=40 雖然還沒超標，但只剩 39 ms 餘裕，不該當成可靠的可行點引用。

三個由這條曲線確定的事：

1. **吞吐量在 c=32 就飽和**（437.8 → 435.6 → 428.8），再加併發只是把請求堆進佇列。
   c=64 的 tok/s 只比 c=32 高 9.8%，代價是 94% 的請求違反 SLO。
   **用吞吐量選操作點會選到一個幾乎全部違反 SLO 的配置。**
2. **綁住的是 TTFT，不是 TPOT。** c=32 的 P99 TPOT 只有 73.2/100，而 P99 TTFT 已經 4,414/5,000。
   TPOT 在 c=1 空載就用掉 47.4 ms（SLO 的一半），但它從 c=1 到 c=48 只從 51 漲到 82；
   同區間 TTFT 從約 155 漲到 5,330。
3. **必須用 p99 判邊界，中位數會誤判。** c=32 的中位 TTFT 只有 1,595 ms，看起來有 3 倍餘裕；
   p99 已經 4,414。用中位數會把邊界推到 c=64 附近，而那裡只有 6% 通過。

### TTFT 惡化的機制（從引擎日誌量到）

```
修正後  #new-seq: 30, #new-token: 8192, #running-req: 64, #queue-req: 0
修正前  #new-seq: 1,  #running-req: 16
```

容量 6 倍之後，`max_num_seqs=452` 讓排程器**一步收 30 條新序列**，8,192 的每步預算被切成 30
份，每條約 140 token，而 prompt 是 1,083 token → 每條要 8 步才填完，全部交錯。

准入會計不是原因：`--admission-reserve` 預設已是最保守的 `generation-full`（TRT-LLM 式，
prompt 加完整 max_new_tokens），但 64 × 1,150 ≈ 73k 在 173,376 的池子裡綽綽有餘，再保守也
不會綁。缺的是「**同時能有幾條在預填**」這個限制——vLLM 有 `scheduler_reserve_full_isl`
（它的並行部分預填峰值只有 2、常駐 6–10），TokenSpeed 沒有。

## 與另一個 session 的發現是同一個形狀

它在 Qwen 上量到 82 次准入拒絕，82 次全部相同：三個 `linear_attention` 群組各要一個
**全空** parent（`slots=1` 讓 state parent 綁定即滿，`free=0`），而注意力群組在 82 次裡都
要求 0 個 parent、還有 7 個 slot 空著。

- 它的：**容量存在但因為 packing=1 而無法被 state 觸及**
- 我的：**容量被錯誤的 config 讀取壓縮 6 倍**

兩者都是**分配器的帳算錯**，不是排程器不夠聰明。這解釋了為什麼兩邊的排程改動都量測不出效果
——在帳算錯的前提下，排程不是那個自由度。

（它後來更正我一個推論：`block_pool.h:94` 的 `BoundGroup` 是**一個 parent 對一個群組**，所以
三個 state 群組不可能共用一個 parent；packing=2 的收益是**總量**上的，64 個請求從 192 個必須
全空的 parent 降到 96，而 `parents_needed` 在單筆記錄裡仍是 3。）

## 量測衛生：今天踩到的三個坑

1. **殭屍 benchmark 客戶端**：舊 harness 等不到 GPU 就照樣往下跑，對著不存在的埠開客戶端後
   永遠掛著。最舊的存活 2 小時 26 分，清掉後 load average 3.04 → 1.31。同一格因此量到
   616.63 與 1,065.92 兩個值（1.73x）。
2. **等待迴圈只檢查客戶端**：server 啟動階段沒有客戶端，於是我自己的兩個掃描重疊。要同時
   檢查 server，並排除容器裡數百個 `[tokenspeed::sch] <defunct>` 殭屍。
3. **`pgrep | head -1` 只殺第一個匹配**：我啟動過兩次的 runner 活了一個下來，在我把機器交給
   別人的那一刻自己開跑。停止背景 runner 後必須**重新列一次確認為 0**。

## Decode 的 roofline：頭手還剩多少

權重 62.5 GB。分母用**實測唯讀**頻寬，不是 copy、也不是規格：

```
read-only : 1523.3 GB/s   <- decode 讀權重的真分母
copy(r+w) : 1460.6 GB/s   <- .roofline.py 報的（重現了先前的 1461.8）
datasheet : 1792.0 GB/s   <- 實測唯讀只達到它的 85%
```

唯讀只比 copy 高 **4.3%**，所以先前用 copy 當分母得到的結論碰巧成立。而規格 1792 會讓每一格
的達成率偏低 15%——**它是一個沒有 kernel 達得到的數字**。每步下限 62.5/1523.3 = **41.0 ms**。

### 達成率必須用引擎報的純 decode，不是 benchmark 的端到端

```
Decode batch. #running-req: 32, page ratio: 0.65, gen throughput: 546.49
Decode batch. #running-req: 32, page ratio: 0.68, gen throughput: 519.26
Decode batch. #running-req: 32, page ratio: 0.71, gen throughput: 514.62
```

| 來源 | tok/s | 對 780 的達成 | 可得 |
|---|---|---|---|
| benchmark 端到端 | 437.8 | 58.4% | 1.71x |
| **引擎純 decode** | **514–546** | **66–70%** | **~1.45x** |

端到端的數字含 prefill 步與暖機/收尾，把它當 decode 效率會**高估缺口**。先驗過批次是滿的
（44 個排程取樣裡 31 個 `#running-req: 32`），所以缺口本身是真的。

各併發（用引擎純 decode 校正過的比例外推）：c=1 已在約 88%，**c=32 是操作點且只有 66–70%**。
**低併發沒有空間，空間在操作點的 decode。**

### decode 吞吐隨上下文衰減

上面三行是同一次跑、批次固定 32：吞吐 546 → 514，`page ratio` 0.65 → 0.71。每步要重讀的 KV
隨已產生的 token 累積變大，所以**「每步固定讀一次權重」的模型在關鍵處是錯的**——下限是常數，
每步總量不是。

推論：**同一併發、不同 OSL 的兩格量的不是同一件事**。OSL 長的那格會看起來比較差，原因與引擎
無關。任何單一的 tok/s 都只是衰減曲線上的一個取樣點。

### 待辦：那 17.6 ms 的歸屬

批次 32、546 tok/s → 每步 58.6 ms，下限 41.0 ms，缺口 **17.6 ms**。KV 流量估約 4.5 ms，
**剩約 13 ms 未歸屬**。torch profiler 走不通（這個 build 沒註冊 `/start_profile` 路由，
`VLLM_TORCH_PROFILER_DIR` 有設也沒有），下一個儀器是 `LOG_MM_TIMING`。

## 已還原：max_partial_prefills（機制在這個引擎上不會發生）

實作了「限制步末仍未完成預填的請求數」，編譯通過、接線完整、trace 也接了計數欄位。
**然後發現它永遠不會觸發**，還原於此。

### 動機是我讀錯了自己的資料

引擎日誌裡我引用的是 `#new-seq: 30, #new-token: 8192`，把 8192/30 ≈ 273 讀成「每個請求
只拿到 273 個 token，被切碎了」。回去看完整分布：

```
new-seq=31  new-token=1892 → 61 tok/seq   ← 預算沒用完
new-seq=63  new-token=3844 → 61 tok/seq   ← 預算沒用完
new-seq=11  new-token=8192 → 745 tok/seq  ← 預算用完
```

**61 = prompt 1,149 − 前綴快取 1,088**，是未命中的尾巴。那些步是幾十個請求各自**完成**一個
61 token 的預填。而 745 是**混合批次的平均**（幾條完整 1,149 + 幾條 61），不是切分。

`schedulePrefill` 是 `tokens_this_round = std::min(remaining, unscheduled)`——貪婪配置。
所以每步最多**一個**請求以未完成狀態結束（撞到預算的那個），上限在任何 ≥1 的值都不會生效。

**一個平均值不是每個成員的值。** 這個錯誤讓我實作了一個完整的機制、寫了測試、建了二進位，
全部針對一個不存在的行為。

### 另一個 session 建了我標為「未驗證」的測試

我提交時寫明測試沒被編譯過（容器無外網、無系統 gtest）。它的環境裡 googletest 已在本地，
二十分鐘內回報：

1. **編不過**——`ForwardBatch` 是 SoA，建構時就把 operations 攤平了，沒有 `.operations`
   成員。等價的是 `extend_prefix_lens.size()`（建構子只在 `PrefillOperation` 分支推入）。
2. 修好後 4 個測試過 3 個，**失敗的正是唯一有鑑別力的那個**。
3. **另外三個是空過的**——每種情況計數都是 1，所以 `<= 3` 恆真、`uncapped == cap64` 是
   1 對 1。整個套件只有一個斷言能區分任何東西，而它是紅的。

第 3 點比第 2 點重要：**一個綠的測試套件可以完全不含資訊**。

## 真正的差距在解碼執行，不在排程

兩個引擎自己報的 c=64 解碼速率（同語料同 SLO 同卡）：

| | 各次取樣 | 平均 | 對 roofline 787 |
|---|---|---|---|
| vLLM | 801, 819, 564, 863 | ~760 | **97%** |
| TokenSpeed | 346, 226, 597, 513 | ~420 | **53%** |

**約 1.8 倍，而且我們的抖動是它的三倍**（226–597 對 564–863）。vLLM 已經貼著記憶體 roofline。
排程層的任何改動都到不了這裡。

---

# 2026-08-22 — 從 API 開始的逐層歸因（gemma-4-31B-it）

工具：`.rung.py`（五階梯）、`.msweep.py`（TTFT 對 prompt 長度）、`.fixedcost.py`
（截距會不會串行化）。全部 c=1、`/v1/completions`（不走 chat template，否則「1 個
token 進」在兩個引擎上是不同的 prompt）、同一台卡、串行相鄰、每格兩輪且第二輪反序。

環境：GPU 1，`--max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 64`，
兩邊都開前綴快取。鄰居 session 在 GPU 3 跑 35B-A3B（04:33 起約 71 GB）——**這支探針對
它不敏感**：vLLM 五個階梯兩輪的重現差 <1.2%（R1 49.67/49.11、R2 2827.85/2827.78）。

## 階梯

| 階 | 比上一階多測到 | ours | vLLM | |
|---|---|---|---|---|
| R0 `GET /v1/models` | 只有 HTTP ingress | **0.18 ms** | 0.38–0.58 | **贏 2–3x** |
| R1 1 token 進 / 1 出 | ＋tokenize、一次 forward、sample、detokenize | 96.9 | **49.1** | 輸 47.8 ms |
| R2 1 token 進 / 64 出 | ＋63 個 decode step | 2982 | **2828** | |
| ↳ 每 decode token | | 45.8 | **44.1** | 輸 3.3% |
| R3 8192 進 / 1 出（每次新 prompt） | ＋tokenize 與 prefill 8192 | **1835** | 1900 | 贏 3.4% |
| ↳ prefill 速率 | | **4713 tok/s** | 4427 | 贏 6.4% |
| R4 同一個 8192 再送一次 | ＋零 —— 差額就是前綴重用 | 118 | **87.9** | 輸 30 ms |

decode roofline 是 62.5 GB / 1523.3 GB/s = **41.0 ms**，所以兩邊的 decode 都在
1.08–1.12x，**單步沒有空間**。R4 減掉各自的截距後，解 8k 快取我們用約 23 ms、vLLM 約
39 ms——**重用路徑本身我們比較快**。

**整個差距是一個與 prompt 長度、輸出長度都無關的約 45 ms 截距。**

## TTFT 對 prompt 長度：交叉點在 M ≈ 2,100

| M | ours | vLLM | ours − vLLM |
|---|---|---|---|
| 1 | 98.3 | 49.7 | +48.6 |
| 64 | 106.2 | 58.7 | +47.5 |
| 256 | 112.0 | 69.4 | +42.6 |
| 512 | 147.5 | 107.9 | +39.6 |
| 1,024 | 246.0 | 204.3 | +41.6 |
| 2,048 | 423.3 | 413.8 | +9.5 |
| 4,096 | **815.2** | 861.8 | **−46.5** |
| 8,192 | **1,824.8** | 1,888.1 | **−63.3** |

差額到 M=1024 都是同一個常數，之後被 6% 的速率優勢吃掉。**M<256 幾乎不長**
（98.3 → 112.0，192 個 token 只加 14 ms）——那段的邊際成本近乎零，因為 forward 是
權重綁定的；截距不是「prefill 慢」。

## 截距不會串行化，所以它是延遲不是吞吐稅

同時送 c 個 1-token 請求，量整批的 wall：

| c | ours wall | 每請求 | vLLM wall | 每請求 |
|---|---|---|---|---|
| 1 | 97.9 | 97.9 | 50.4 | 50.4 |
| 2 | **98.1** | 49.0 | 94.8 | 47.4 |
| 4 | 99.3 | 24.8 | 97.3 | 24.3 |
| 8 | 101.1 | 12.6 | 101.6 | 12.7 |
| 16 | 112.5 | 7.0 | 108.8 | 6.8 |
| 32 | **125.4** | 3.9 | 166.2 | 5.2 |

ours 在 c=2 的 wall 與 c=1 相同——**兩個請求進同一個 forward**，vLLM 花了兩次
（50 → 95）。到 c=32，**ours 125 ms 對 vLLM 166 ms，我們贏 25%**。

**結論：這 45 ms 只在低併發的短 prompt 上咬人。** agent 形狀（26k–127k）遠在交叉點
右邊，我們本來就贏；一般 RAG 形狀（約 1k）在 c=1 輸 20%，但併發一起來就被批次吸收。
**它是真缺陷，但不是貴的缺陷**，優先度低於下一節。

## 沿路修掉的一個真缺陷（commit fc897601）

`event_loop.py` 的 `resolve_fi_gdn_prefill_graph` 為了躲 FlashInfer GDN chunk-prefill
在 graph capture 下輸出 `!!!!!!` 的 bug，**只看環境變數就關掉 prefill CUDA graph**，
不問模型有沒有 GDN 層。全樹只有 `models/qwen3_5.py` 與 `models/kimi_k3.py` 會走到那顆
kernel，gemma-4 是 50 sliding + 10 full、**零個 linear 層**，卻每個請求都被收走 graph。
改成讀模型自己的 `layer_types`；無法判定的 config 維持今天的行為。

**這個修正在 c=1 上值 0**：R1 96.85 → 98.44（+1.6%）、R3 1835.02 → 1836.28（+0.1%）、
每 decode token 45.80 → 45.55（−0.5%）。而且**機制確實啟動了**——日誌有
`prefill breakable graph ENGAGED (text): bucket=16`、捕了 40 個 bucket——所以這是
「功能開著且無效」的陰性，不是「功能沒開」的陰性。這個閘門自己的 docstring 記的 +8.4%
是 Qwen3.5-27B RAG C=128 量的，gemma 的併發形狀還沒量。

修它的理由不是那 0%，是它拿全域旗標回答一個關於模型的問題。

## 下一個目標：decode 隨 batch 的成長

c=1 的 decode 是平手（45.8 對 44.1，都約 roofline 的 1.1x），但本檔前面那次 RAG c=64
量到 ours 約 420 tok/s 對 vLLM 約 760——每步 152 ms 對 84 ms。

算得出來為什麼這是 KV 的問題：gemma-4 每 token 每層的 KV 是 sliding 16 KiB、full
8 KiB，所以 batch 64、context 1k 時一步要讀
`50 層 × 1024 窗 × 16 KiB × 64 = 52 GB` 加 `10 × 1024 × 8 KiB × 64 = 5.2 GB`，
**KV 流量約 57 GB，與 62.5 GB 的權重同量級**。步的下限變成 (62.5+57)/1523 = **78 ms**，
vLLM 的 84 ms 是它的 **93%**，我們的 152 ms 是 **51%**。

**所以差距在 decode attention 怎麼讀分頁 KV，而且只有在 context 夠長時才看得見。**
下一輪（`.decodescale.py`）用兩個 context 長度掃 batch 1→64，短的那個應該看不到差距。

## decode 隨 batch 的成長：沒有 1.8x，兩邊都撞同一面牆

`.decodescale.py`，串流量 ITL（所以不含 prefill），unique prompt（共用前綴會讓兩臂做
不同量的 prefill，那是另一個實驗），`--max-num-seqs 64`、GMU 0.90、bf16 KV。

**短 prompt（約 120 token）**

| c | ours ITL | vLLM ITL | ours decode tok/s | vLLM |
|---|---|---|---|---|
| 1 | 46.8 | 44.8 | 20.5 | 21.4 |
| 4 | 47.3 | 44.8 | 63.5 | 79.6 |
| 16 | 51.6 | 50.0 | 194.7 | 224.7 |
| 32 | 55.9 | 54.3 | 293.6 | 336.9 |
| 64 | 70.6 | 66.0 | 380.6 | 433.4 |

**輸 3–7%（ITL）／12–14%（吞吐），不是 1.8x。** 本檔前面那個「約 420 對約 760」是把兩次
不同跑的引擎自報 `gen throughput` 拿來相除得到的，受控 A/B 下不成立。**撤回。**

**約 1,000 token 的 prompt**

| c | ours ITL | vLLM ITL | ours tok/s | vLLM tok/s | ours TTFT | vLLM TTFT |
|---|---|---|---|---|---|---|
| 1 | 46.9 | 45.2 | 17.4 | 17.4 | 545 | 579 |
| 4 | 49.0 | 47.1 | 43.7 | 41.8 | 1,744 | 2,223 |
| 16 | 60.1 | 59.0 | 50.6 | 56.0 | 8,293 | 6,467 |
| 32 | 60.9 | 61.9 | 52.3 | 54.7 | 12,695 | 11,445 |
| 64 | 61.7 | 63.1 | 55.5 | 54.2 | 25,677 | 25,950 |

**兩邊在誤差內相同**，而且都在 c≥16 之後不再成長。原因兩邊的日誌都自己說了：

```
ours  Decode batch. #running-req: 18, page ratio: 0.91
vLLM  Running: 16 reqs, Waiting: 30 reqs
```

**這面牆是 gemma-4 的物理，不是誰的缺陷。** 50 個 sliding 層每 token 16 KiB、窗 1024，
一條 1k 請求就佔約 880 MiB；27 GB 的 KV 池只裝得下約 28 條，實測常駐 16–19。
`--max-num-seqs 64` 在兩個引擎上都是一個到不了的數字。

**推論：一般場景要贏，槓桿是 fp8 KV（容量直接翻倍），不是 decode kernel。**
（vLLM 側先前量過 gemma-4 RAG c=64 的 fp8 是 +57%，而且是門檻效應不是斜坡。）

## 但在做任何優化之前：ours 在 gemma-4 上的輸出是壞的

同一組設定、同一支客戶端、同一個問題（`/v1/chat/completions`, temperature 0）：

| | 回答 |
|---|---|
| **ours** | `' de de de de de de11 singular1 singular1 single de1 single de1allyally...'` |
| **vLLM** | `'The capital of France is Paris. It is famous for its world-class art museums, iconic landmarks like the Eiffel Tower, and its reputation as a global center for fashion and gastronomy.'` |

**這不是新的。** 2026-08-01 的筆記已經記著「TokenSpeed 的 triton backend 在 gemma-4 上吐
`'を______11111'`，與 HF 參考不符，可能是真 bug，**未分診**」。三週沒有人接。

gemma-4 在 ours 上會選到**四顆**不同的 attention kernel，所以「backend」在這裡不是單數：

```
mha_decode_with_kvcache -> flashinfer_trtllm_mha_decode_with_kvcache   (sliding, head_dim 256)
mha_decode_with_kvcache -> triton_mha_decode_with_kvcache_cached       (global,  head_dim 512)
mha_extend_with_kvcache -> flashinfer_paged_mha_extend_with_kvcache
mha_extend_with_kvcache -> triton_mha_extend_with_kvcache
```

**今天量到的 ours 那一側全部是在量一條輸出錯誤的路徑**，包括 prefill 快 6.4% 那一格——
一個加速有可能正是因為它算錯了才快。在分診收斂之前，本節與上面幾節的 ours 數字都只能
當成「壞路徑的效能」，不能當成引擎的效能。

---

# 正確性分診：gemma-4 在 ours 上的輸出是壞的

`.rung.py` 那一輪照 `AGENTS.local.md` 的規定在每個配置後面印一段真實輸出，抓到這個。
**這不是新缺陷**：2026-08-01 的筆記已經記著「TokenSpeed 的 triton backend 在 gemma-4 上
吐 `'を______11111'`，與 HF 參考不符，可能是真 bug，**未分診**」。三週沒有人接，因為
`--ignore-eos` 的吞吐 harness 看不見它——每個請求都會吐滿 `out_tokens`，不管吐的是不是語言。

同一組設定、同一支客戶端、同一個問題、temperature 0：

| | 回答 |
|---|---|
| **ours** | `'額LLLLLLL額額額額額額額額額...'` |
| **vLLM** | `'The capital of France is Paris. It is famous for its world-class art museums...'` |

## 已修的第一個 bug（commit 220e4a61）：softmax scale 從來沒被傳下去

`backends/mha.py` 呼叫 `mha_extend_with_kvcache` 與 `mha_decode_with_kvcache` 時沒有傳
`softmax_scale`，而每顆 kernel 對缺席的 scale 都退回 `1/sqrt(head_dim)`。這對樹裡
`self.scaling == head_dim**-0.5` 的模型都對——除了不是的那幾個：

| | scaling |
|---|---|
| `models/gemma.py:337` | **1.0**（gemma-4；與 HF `modeling_gemma4.py:1194` 一致，qk-norm 之後 q/k 已是單位 RMS） |
| gemma-3-27b | `query_pre_attn_scalar=168` 開根號倒數，而 head_dim 是 128 |
| `models/inkling.py:467` | `1.0 / head_dim`（不是開根號） |

gemma-4 的 logits 因此被縮小 **16x**（50 個 sliding 層，head_dim 256）與 **22.6x**
（10 個 global 層，512）。`backends/inkling.py` 包的是同一個 backend，它三個呼叫點
**一直都有傳**；只有這條主路徑漏了。

**但它不是全部**：修完之後輸出從 `'額LLLL...'` 變成 `'al outalerPP dees...'`——
**變了但仍是垃圾**，所以 scale 確實有進到計算裡、確實是錯的，而且還有第二個因素。

## 已排除（逐項對照過參考，不是推測）

| 項目 | 怎麼排除的 |
|---|---|
| 四顆 attention kernel | `unieinfra/bench/kernel_gemma_correctness.py` 在 gemma-4 兩個真實幾何上逐顆對 `scaled_dot_product_attention`，decode 與 extend、ctx 512/2048，**全部 rel 2–3e-3**（bf16 雜訊），且確認它們吃 `softmax_scale=1.0` |
| kernel 的**選擇** | 三個臂：FlashInfer extend 開／關／全部強制 Triton，**輸出都是垃圾** |
| proportional RoPE 的頻率 | `build_rope_inv_freq` 與 HF `_compute_proportional_rope_parameters` 逐行相符，含「回傳整個 head_dim//2 長度、尾巴 192 個為零」這個關鍵——它與 `rotary_dim=128` 的一般 partial rotary **是不同的變換**（旋轉的維度集合、配對、頻率階梯三者全異），而我們的 docstring 早就寫明了這個陷阱 |
| RMSNorm 形式 | HF `Gemma4RMSNorm` 是標準 `x * weight`（**不是** gemma-3 的 `1+w`），我們用的也是 |
| 三明治 norm 與殘差順序 | 與 HF `Gemma4TextDecoderLayer.forward` 逐行相符 |
| `layer_scalar` | checkpoint 有這個張量，HF 在層末 `hidden_states *= layer_scalar`，我們同位置 |
| GeGLU 啟動 | `gelu_pytorch_tanh`，且對其他值直接 raise |
| embedding 縮放 | `sqrt(hidden_size)`，與 HF `embed_scale=hidden_size**0.5` 相同 |
| `attention_k_eq_v` | checkpoint 的 full 層確實沒有 `v_proj`（先前看到的是 vision tower 的），V 取 k_proj 的**原始**輸出（k-norm 與 RoPE 之前），與 HF 的物件繫結順序一致 |
| chat template | 25 個 token，與 HF `apply_chat_template` 的參考完全相同 |
| 權重載入 | 載入器對任何未寫入的參數或未認領的張量**直接 raise**，不會靜默 |

## 下一個探針：逐位置的 prompt logprob

`echo=True, max_tokens=1, logprobs=1` 回傳每個 **prompt** token 的 logprob，兩個引擎跑
同一段文字再逐位置比。設計上最鋒利的一點是**位置 1**：只有一個 token 可以 attend，
softmax 退化成單元素，attention 恆等於 V，而 RoPE 在位置 0 的角度是 0（cos=1, sin=0）
也是恆等。**所以位置 1 的結果完全不含 RoPE、也不含任何 attention 權重計算。**

- 位置 1 就不合 → RoPE 與 attention 同時出局，剩下 embedding 縮放、norms、MLP、
  o_proj、`layer_scalar`、最終 norm、LM head，或 KV 快取的 V 往返。
- 位置 1 合、之後才發散 → 是會累積的東西：KV 快取、視窗、或只有部分位置會走到的路徑。

工具：`.logprob_cmp.py`（`--fetch` / `--compare` 兩段式，因為兩個 62 GB 的伺服器不能
同時上卡）、驅動 `.lpcmp.sh`。

**在分診收斂之前，本檔所有 ours 的效能數字都只是「壞路徑的效能」。**

---

# 根因：KV 池只模型化一個 head_dim，gemma-4 有兩個

## 證據鏈

**1. 逐層對 HF 比對（`TOKENSPEED_DUMP_HIDDEN` + `.hf_ref.py` + `.hidden_cmp.py`）**

同一組 input_ids（直接取自引擎自己的 dump，所以 BOS 差異不可能冒充成壞層）：

```
embed        rel_l2 = 0.0000e+00     完全相同
layer0..4    rel_l2 ≈ 4.4e-03        bf16 雜訊
layer5       rel_l2 = 3.16e-01       <-- 第一次跳躍
layer11      1.73e-01   layer17  2.06e-01   ...   final_norm  1.26e+00
```

`[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]` 正好是 gemma-4 的十個 `full_attention` 層。
**50 個 sliding 層全對，10 個全域層全錯。**

**2. 層內拆分（`TOKENSPEED_DUMP_ATTN` + `.attn_cmp.py`），含對照**

| 層 | 幾何 | `attn_out` 對 SDPA(它自己傳入的 q,k,v) | `v` 對 v_norm(投影) |
|---|---|---|---|
| 0（sliding） | 32q × 16kv × **256** | **1.42e-03 相符** | 2.3e-03 |
| 5（full） | 32q × 4kv × **512** | **1.13 不符** | 2.4e-03 |

投影正確（`k_raw` 對 `h @ k_proj.T` = 1.6e-03）、V 正確、kernel 本身在同樣幾何的合成張量上
已對 SDPA 驗證過（`kernel_gemma_correctness.py`）、RoPE 的套用也驗證過
（`kernel_gemma_rope.py`，含 head_dim 512 的 partial rotary）。
**所以 kernel 讀到的 K/V 不是我們寫進去的那一組。**

## 機制

`kv_cache/mha.py:210` `_layer_row_view`：

```python
heads_l = self._layer_heads_per_rank(layer_id)
return buf.reshape(-1, heads_l, self.head_dim)     # <-- 池子的 head_dim，不是該層的
```

池子只有**一個** `head_dim`（`factory.py:168-169` 取 `config.num_kv_heads` 與
`config.head_dim`，也就是頂層的 16 與 256——全域層的 4 與 512 藏在
`num_global_key_value_heads` / `global_head_dim`，工廠看不到）。它的異質機制只讓
**頭數**變動：頭少的層把同樣的位元組重新解釋成更多列。

gemma-4 是第一個 **head_dim 也逐層不同**的模型：

| | 每 token 每層每個 K 的元素數 |
|---|---|
| 配置寬度（sliding） | 16 × 256 = **4096** |
| 全域層實際需要 | 4 × 512 = **2048**（配置寬度的 1/2） |
| `_layer_row_view` 給它的列 | 4 × 256 = **1024**（配置寬度的 1/4） |

**差剛好一倍。** `set_kv_buffer` 以 token 為單位的 `loc` 寫進 1024 元素的列，而
`_get_kv_cache` 把同一塊記憶體看成 `(-1, page_size, 4, 512)`，每個 token 是 2048 元素——
於是每個 token 的 K/V 都落在錯的位移上，只在那 10 層。

日誌自己印得出來：`parent_bytes=10485760 / (64 token × 10 層 × 2) = 8192 B` = 4096 個
bf16 = **16 × 256**，而六個群組共用 `page size 64`。

## 建議的修法（未實作）

`_layer_heads_per_rank` 回傳的本來就是「以配置寬度為基準的等效頭數」，不是真實頭數
（它的 docstring 已經這樣寫了）。所以：

1. 讓池子接受 `layer_kv_head_dims`，等效頭數改成
   `served_heads × dim_l // alloc_head_dim`。gemma-4：sliding 16×256/256 = **16**、
   full 4×512/256 = **8** → 列變成 8 × 256 = 2048 元素，正確。
2. gemma 的 cache recipe 供給每層的 head_dim（目前 `layer_kv_head_counts` 對 gemma 是
   `None`，只有 qwen35 與 inkling 有給）。

**必須一起驗的**：全域層每個 byte 能裝 2 倍的 token，所以 `token_capacity` 與群組的
page 記帳會變；`_layer_kv_head_counts` 的其他使用者（mxfp8 的 scale buffer 與
`quantize_store_kv_mxfp8`）在 head_dim 一致時語意不變，但要確認。這是對共用配置器的
設計改動，值得自己一輪量測，不該在一個長 session 的尾巴半途改完——這棵樹的
「impossible state」缺陷正是那樣來的。

**回歸測試**：`.attn_cmp.py` 對 layer 5 的那一列（`attn_out` 對 SDPA）就是判準，
layer 0 是對照。兩者都在同一支腳本裡，改完直接重跑。

---

# 已修：每層要拿到屬於自己寬度的 cache 列（commit 40f2478a）

## 修法

池子把每層的欄位配置在**同一個寬度**，較窄的層把同樣的位元組重新解釋成更多列。
那個重新解釋是用該層的**頭數**去除**配置的 head_dim**——而那只有在所有層共用一個
head_dim 時才等於該層的寬度。

改動有五處，但只有一個想法：**寬度要來自那一層**。

| 檔 | 改動 |
|---|---|
| `configs/mha.py` | `_per_layer_kv_geometry` 從 decoder config 讀 `num_global_key_value_heads` 與 `global_head_dim`（gemma 是 `*ForConditionalGeneration`，頂層兩個都沒有——與 `09f124ac` 同一個陷阱）。所有層共用一個幾何時回 `(None, None)`；宣告了不同幾何但層標籤蓋不滿整個堆疊時**拒絕**，因為那時無法指認哪些層是全域的 |
| `configs/mha.py` | `cache_cell_size` 取**最寬**的層。gemma-4 比頂層窄所以數值不變，但全域層更寬的模型本來會被配置不足 |
| `recipes/{setup,ordinary}.py` | 把 `layer_kv_head_dims` 與 counts 一起帶、一起切片；拒絕「只有 dims 沒有 counts」（那會讓層以配置的頭數配自己的 head_dim，比它的欄位還寬） |
| `kv_cache/mha.py` | `_layer_row_view` 改 reshape 成 `(-1, heads_l, dim_l)`；`_layer_page_tokens` 也改用**寬度**而非頭數——同一個錯誤，隔壁一個函式 |
| `factory.py` | `family=="mha"` 那一支**連 `layer_kv_head_counts` 都沒傳**，所以整套每層機制對所有 ordinary MHA 模型都是死的。前面四處改完量不到任何東西，直到這一處補上 |

## 驗證（三個層級）

**1. kernel 輸入層級**（`.attn_cmp.py`，layer 0 是同一支腳本裡的對照）

| | `attn_out` 對 SDPA(引擎自己傳入的 q,k,v) |
|---|---|
| layer 0（sliding 32×16×256） | 1.42e-03（修正前後皆同） |
| layer 5（global 32×4×512）**修正前** | **1.13** |
| layer 5 **修正後** | **1.48e-03** |

**2. 逐層對 HF**（同樣五個 id）

```
layer5      3.16e-01  ->  5.49e-03
final_norm  1.26e+00  ->  1.62e-02
```
每一層都回到 bf16 漂移的量級。

（`layer59` 那列的 9.8e-01 是比對腳本的對齊差一位：HF 的 `hidden_states[-1]` 已過
`model.norm`，我們的 `layer59` 是過 norm 之前；兩邊範數 23.96 對 723.6 就是這件事，
而 `final_norm` 對得上。）

**3. 端到端**——與 vLLM **逐字相同**（temperature 0，同一台卡串行）：

| | ours | vLLM |
|---|---|---|
| 法國首都 | `'The capital of France is Paris.'` | 相同 |
| 俳句 | `'Blue waves kiss the shore,\nEndless depths of salt and foam,\nTides drift in and out.'` | 相同 |
| 17×23 | `'...Multiply 17 by 3 (the ones place): $17 \times 3 = 51$...'` | 相同 |

## 為什麼三週沒被發現，以及現在怎麼防

啟動日誌現在會印**解析後**的幾何：

```
KV geometry: allocation 16 heads x 256, per-layer widths ['16x256', '4x512']
```

一個悄悄退回單一幾何的池子，在其他任何觀測點上都與「本來就不需要每層寬度」的池子
無法區分——這正是它藏了三週的方式。而 `--ignore-eos` 的吞吐 harness 看不見它：每個
請求都吐滿 `out_tokens`，不管吐的是不是語言。

測試 `test/runtime/test_per_layer_kv_head_dim.py`（9 例）其中一例**釘住缺陷本身**
（沒有 dims 的池子必須仍產生舊的 1/4 寬度列），所以 revert 過不了；另一例斷言配置寬度
恰為全域層的兩倍——若那個比值變成別的，就代表 row view 與配置在這個改動沒有描述到的
地方又不一致了。

## 刻意沒做的

全域層現在只佔半個配置列，那十層有一半的位元組閒置。**那是容量問題，不是正確性問題**，
值得自己一輪量測。

## 本檔前面所有 ours 的效能數字都作廢

它們是在輸出錯誤的路徑上量的。正在於修正後的建置上重跑。

---

# 回歸：`fc897601` 讓 gemma 拿回 prefill graph，但那在 GMU 0.945 起不來

agent-127k 的第一格 `ours` 直接啟動失敗：

```
ERROR - Prefill graph capture ran out of GPU memory. Free up
        --gpu-memory-utilization headroom, lower --prefill-graph-max-tokens
torch.OutOfMemoryError: Tried to allocate 70.00 MiB ... 320.00 KiB is free
```

**歷史上的 a127 grid 能在 0.945 跑，正是因為那個 GDN 閘門把 graph 關著。** 我修掉閘門
（它問的是錯的問題——gemma 一個 GDN 層都沒有），graph 於是開始捕捉，把餘裕吃掉。

值得對照的是它的效益：**在 gemma、c=1 上量到的是 0**
（R1 96.85 → 98.44、R3 1835.02 → 1836.28、每 decode token 45.80 → 45.55），而且是在
確認過 `prefill breakable graph ENGAGED` 的前提下量的。併發形狀還沒量。

所以現況是：**一個未量到效益的優化，讓一個原本可用的配置起不來。**

`prefill_graph.py:374` 對 `NotImplementedError / AttributeError / KeyError /
RuntimeError` 都會退回 eager prefill，**只有 OOM 是重新拋出**。一個「可選」的優化不該
讓伺服器起不來——但我沒有在這裡直接改，因為：

1. 部分捕捉之後繼續服務，會留下 CUDA graph mempool 拿著的記憶體，而現有的退場分支也
   沒有把它還回去；能不能安全繼續需要實測。
2. 重現這個 OOM 要一次完整啟動（GMU 0.945），而卡正被量測佔著。

**盲改共用的啟動路徑，正是這棵樹「impossible state」缺陷的來源。** 三個選項留給下一步，
按我的偏好排序：

| | 做法 | 代價 |
|---|---|---|
| A | 捕捉前先估成本，不合就跳過並印出數字 | 估計不準；但「跳過」永遠比「死掉」安全，且不會留下半個捕捉 |
| B | OOM 改成退場（清掉 captures、`empty_cache`、`disable = True`） | 需要驗證繼續服務不會因殘留 mempool 而在執行期 OOM |
| C | 撤回 `fc897601` | 閘門的邏輯是對的，撤回會把一個真缺陷放回去 |

本輪的 a127 量測用 `--prefill-graph-max-tokens 0` 跑，並在腳本裡寫明原因——**把代價記下來，
不是繞過去**。
