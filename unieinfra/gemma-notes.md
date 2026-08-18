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
