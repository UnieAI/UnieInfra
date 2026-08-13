# 80 — Kernel 記分板

目標：**每一顆 kernel 都對照參照引擎量測並優化，直到我們贏。** 這份是進度狀態，
一顆一列，量到什麼寫什麼。

量測環境：RTX PRO 6000 Blackwell（sm_120, 188 SM, HBM peak 1792 GB/s）、
容器 `unieinfra-dev`、vLLM 在 `/home/ubuntu/workspace/roy/vllm/.venv`。
方法與陷阱見 `90-DIVERGENCES.md` §3。

---

## 絕對刻度：離理論值還有多遠

記分板其餘部分是**相對**的（對 vLLM 幾倍）。相對指標有一個結構性缺陷：**它永遠
不會告訴你已經到頂了**。MoE decode 就是活例 —— eager 模式下量到「贏 2.4x」，換成
CUDA graph 後是 1.07x，而真正的資訊兩者都沒說出來：**兩邊都在 65–69% of HBM
peak，bf16 上不存在 3x 的空間**。那句話只有絕對刻度講得出來。

理論值的算法，這張卡（HBM peak 1792 GB/s、188 SM）：

- **memory-bound 的東西**（decode 的一切、elementwise、KV 讀寫）：`bytes / 1792 GB/s`
- **decode 一整步**：`模型權重 bytes / 1792 GB/s` —— 每產生一個 token，權重就得
  過一遍，這是物理下限，與 batch 無關（batch 只攤薄它）
- **compute-bound 的東西**（大 M 的 GEMM、prefill attention）：`FLOP / 卡的 bf16
  峰值`，此處尚未量到峰值，故留 `?` 而不是猜

| 項目 | 理論值 | 實測 | 達成率 | 量測條件 |
|---|---|---|---|---|
| Qwen3.5-27B decode step | **31.0 ms**（55.6 GB） | 49.6 ms | **63%** | C=32，`.cq.sh` mean ITL |
| 同上，vLLM | 31.0 ms | 53.5 ms | 58% | 同一支 harness |
| gemma-4-31B decode step | 34.9 ms（62.5 GB） | `?` | `?` | 未量 |
| Qwen3.6-35B-A3B decode step | `?`（MoE 只讀 top-8，非全部 71.9 GB） | `?` | `?` | 未量 |
| MoE decode 一層（35B-A3B, bf16, T=1） | **0.028 ms**（50.3 MB） | 0.0405 ms | **69%** | CUDA graph，輪換專家集合避開 L2 |
| 同上，vLLM `fused_experts` | 0.028 ms | 0.0433 ms | 65% | 同上 |
| 同上，eager（無 graph） | 0.028 ms | 0.0618 ms | 45% | 差額即 launch 開銷 |
| MoE gate/up 半邊 | `?` | — | 68% | 單獨量，未含 down |
| 前綴重用（RAG shape, C=32） | **74.4%**（本批次結構）／84.1%（循序到達） | 44.0% | **59%** | oracle 由該批 64 個 prompt 實算 |
| TTFT (C=32) | `?` | 1405 ms | `?` | 理論值需先定義：`重用後的 prefill token × 每 token 成本 / batch` + 排隊 |

**目前離理論值最遠的兩項是前綴重用（59%）和 decode 達成率（63%）**，而且性質相反：
前者是政策問題（該重用的沒重用，kernel 再快也補不回來），後者是 kernel 與 launch
問題。

decode 的理論值同時說死了它的改善路徑只有三條，因為 55.6 GB 每步都得讀：

1. **讀更少** —— 量化（fp8 → 27.8 GB → roofline 15.5 ms）
2. **一次讀多產出更多 token** —— 推測解碼（已量 +27~45%）
3. **把 63% 推到 90%** —— kernel 與 launch 層面

任何不屬於這三類的 decode 優化，對這個模型在這張卡上不可能有量級效果。

---

## 記分板

| kernel | 對 vLLM | 狀態 | 已整合 |
|---|---|---|---|
| Sampling (top-k/top-p) | **贏 1.6–2.8x**（batch 1–256 全區間） | ✅ 全贏 | 原本就是 |
| GDN chunk prefill | **贏 1.65x**（vLLM 在 sm_120 走同一套 Triton FLA） | ✅ 贏 | ⚙️ env var |
| Full attention prefill | 4096 **贏 7%**；1024 輸 9%；256 輸 1.7x | ⚠️ 長贏短輸 | ⚙️ env var |
| Decode attention | batch 8–32 **贏 3–32%**；batch 1 輸 1.8–2.9x；batch 128 平 | ⚠️ 大批贏小批輸 | 原本就是 |
| RMSNorm | **全區間贏**：device 1.01–1.19x、host 1.7–1.8x | ✅ 贏（**前值是錯的**） | 原本就是 |
| fused_add_rmsnorm | device 32–4096 輸 5–7%；host 贏 1.0–1.7x | ⚠️ | 原本就是 |
| SiLU-and-mul | **全區間贏**：device 1.08–1.26x、host 1.01–1.08x | ✅ 贏（**前值是錯的**） | 原本就是 |
| MoE (bf16) | 輸 1.2–8x（原本輸 2.1–21x） | ⚠️ **已改善 2.1–2.6x**，仍落後 | ⚙️ `solution="triton"` |
| MoE (NVFP4) | 未量 | ⬜ | — |
| 量化 GEMM (w4a16) | 無法量（缺 Marlin GEMM，見 `70-PORTS.md` T1） | ⬜ | — |
| RoPE | 1–32 tokens **贏 1.70–1.73x**（device）；1024 贏 3%；4096 打平 | ✅ 全區間不輸 | ✅ 已是預設 |
| KV cache 寫入 | kernel **贏 1.7–2.2x**；Triton launch 的 host 成本輸 2.0x（14.9 vs 7.5 µs） | ⚠️ decode 贏、prefill 輸 | 原本就是 |
| Dense GEMM (bf16) | device **打平**（1279 vs 1271 µs，0.6%）—— 底層同一顆 cuBLAS；host 輸（16.6 vs 8.0 µs，已從 19.0 降下來） | ⚠️ 新增一列 | 原本就是 |

### 端到端（Qwen3-8B，同一支 harness）

**TTFT / TPOT**

| | tokenspeed | vLLM | |
|---|---|---|---|
| TTFT（8 tok prompt＝固定開銷） | 25.07 ms | **15.54** | 輸 9.53 ms |
| TTFT（~2.5k tok prompt） | 175.9 ms | **143.9** | 輸 22.5% |
| TPOT（每 output token） | **11.77 ms** | 12.07 | **贏 2.5%** |

**按工作形狀（TTFT 中位數，1000 字共用前綴）** —— 每次換新 prompt 是唯一會把
prefix cache 問題藏起來的形狀，RAG／agent 的真實形狀是共用前綴 + 短尾巴：

| 形狀 | tokens | tokenspeed | vLLM | |
|---|---|---|---|---|
| cold（每次新 prompt） | 5892 | 381.0 ms | **273.2** | 輸 1.39x |
| repeat（完全命中） | 3890 | 35.0 ms | **24.4** | 輸 1.44x |
| **rag（前綴＋新尾巴）** | 4200 | 53.4 ms | **31.6** | **輸 1.69x** |
| turn（多輪累積） | 7838 | 73.5 ms | **57.8** | 輸 1.27x |

dense Qwen3-8B 的 prefix cache 是活的（repeat 35 ms vs cold 381 ms）——
`unieinfra-prefix-cache-dead-on-hybrid` 那個問題只在 hybrid 上。

### 併發曲線（`unieinfra/bench/e2e_concurrency.py`，shared 形狀，700 字前綴，64 output）

「每個 user 盡可能快」和「盡可能高併發」是同一條軸的兩端，中間那段才是真正要看的。
每個請求從自己送出的時刻起算，串流量 TTFT 與 token 間延遲：

| 併發 | ts TTFT p50 | vLLM | ts TTFT p99 | vLLM | ts ITL p50 | vLLM | ts tok/s | vLLM |
|---|---|---|---|---|---|---|---|---|
| 1 | 33.43 | **23.49** | 33.43 | **23.49** | **11.68** | 11.94 | **83.1** | 82.5 |
| 4 | **52.04** | 53.45 | **52.63** | 56.74 | **11.97** | 12.40 | **316.8** | 307.2 |
| 16 | 101.05 | **82.66** | 152.88 | **129.52** | **14.21** | 14.32 | **1019.5** | 1001.7 |
| 64 | 293.73 | **227.70** | 505.06 | **425.34** | **22.95** | 23.64 | **2314.9** | 2209.2 |
| 128 | **300.71** | 359.46 | **745.70** | 773.41 | **31.62** | 33.54 | **3517.2** | 2942.1 |

三個結論：

1. **token 間延遲（ITL）五個併發等級全贏**（1–6%）。串流開始之後，使用者看到的
   逐字速度我們在每個負載下都比較快。
2. **吞吐五個等級全贏，而且差距隨併發拉開**：+0.7%、+3.1%、+1.8%、+4.8%、
   **+19.5%**。我們的 TTFT 從 64 到 128 幾乎持平（293.7 → 300.7），vLLM 則從
   227.7 漲到 359.5 —— 高負載下我們退化得比較慢，128 併發時反超。
3. **TTFT 在低到中併發輸**：併發 1 輸 42%、16 輸 22%、64 輸 29%。這是唯一
   還明顯落後的一項，而且和單請求 rag 的缺口是同一個東西。

vLLM 這一輪跑了兩次，`gpu_memory_utilization` 0.33 與 0.88 各一次（0.33 是因為當時
另一張卡只剩 36 GB）。兩次數字幾乎完全一致，所以記憶體不是它高併發的限制 —— 這是先
查過才敢用的對照，不是假設。

**Throughput（batch 64，各 128 output token）**

| 形狀 | tokenspeed | vLLM | |
|---|---|---|---|
| unique | 377.7 output tok/s | **386.3** | 輸 2.2% |
| shared | 2303.4 output tok/s | **2573.1** | 輸 10.5% |

### 修掉的第一個：prefill attention 跑在 Triton 上（`TOKENSPEED_ENABLE_FI_PAGED_EXTEND=1`）

`mha_extend_with_kvcache` 只註冊了兩顆：trtllm-gen 那顆因為需要 tcgen05 被
capability 上界擋在 sm_103，剩下 Triton。**所以 sm_120 上每一次 prefill 都跑 Triton，
而 vLLM 跑 FlashAttention。**

1:1 移植 SGLang 的 paged prefill 路徑（`BatchPrefillWithPagedKVCacheWrapper`，
`backend="fa2"`，見 `srt/layers/attention/flashinfer_backend.py:1243 forward_extend`）。vLLM 呼叫的 `flash_attn` 套件在這個
容器裡只是一個空命名空間（只有 `cute`），flashinfer 包的是同一批 FA2 kernel。

| tokens | triton | fa2 | |
|---|---|---|---|
| 1024 | 142.9 µs | **76.8** | 1.86x |
| 4096 | 1985.1 µs | **578.1** | 3.43x |
| 8192 | 7500.8 µs | **1865.0** | 4.02x |
| 16384 | 29620.8 µs | **6476.0** | **4.57x** |

數值一致到 bf16 的 1 ulp（max|Δ| = 0.0156）。端到端：

| | Triton（起點） | FA2 | + plan | + 去多餘 step | **+ page 16** | vLLM |
|---|---|---|---|---|---|---|
| TTFT cold | 381.0 ms | 283.6 | 278.8 | 271.6 | **271.5** | 273.2 → **贏 0.6%** |
| TTFT repeat | 35.0 ms | 33.8 | 29.9 | 22.6 | **19.4** | 24.4 → **贏 26%** |
| TTFT rag | 53.4 ms | 49.0 | 45.0 | 38.0 | **32.8** | 31.6 → 輸 3.9% |
| TTFT turn | 73.5 ms | 67.2 | 63.1 | 59.3 | **57.2** | 57.8 → **贏 1.1%** |
| Throughput unique | 377.7 tok/s | 443.0 | 446.4 | 446.6 | **444.6** | 386.3 → **贏 15%** |
| Throughput shared | 2303.4 tok/s | 2210.0 | 2237.3 | 2263.5 | **2219.3** | 2573.1 → 輸 14% |

四個 TTFT 形狀裡**三個領先**，rag 從落後 1.69x 收到 3.9%。四欄修正依序是：FA2、
plan 提升、去掉多餘 step、page size 64→16（各見下節）。

**plan 提升**（`install_paged_extend_plan`）：kernel 自己 plan 時，貴的不是 `plan()`
——它是內容鍵、第 0 層之後就命中——而是為了組鍵而做的 `cache_seqlens.to("cpu")`。
那在 CUDA 上是同步，於是每一層都把前面各層剛填進去的佇列排空，正好抵銷引擎整個
CPU/GPU 重疊的設計。改成 `init_forward_metadata` 裡每個 forward 算一次（SGLang 的位置），
每種形狀省下 3.5–4 ms。

**這一版有兩個 bug，都是「看起來有效、其實 inert」**，值得記下來：

1. 第一版對 `metadata.page_table` 做 plan，但 flat KV 路徑上那是 `None`，layer 實際拿到的是
   `_select_page_table()` 從 `metadata.page_tables` 取的**每組各一張**表。plan 了一張
   kernel 永遠不會收到的表。改成每組各 plan 一次（wrapper 也按組分配，一個 wrapper
   只持有一個 plan）。
2. 修完後仍然全部落回慢路徑，只差**最後一個 bit**：backend 算 `head_dim ** -0.5`
   = 0.08838834764831845，kernel 的預設是 `1.0 / math.sqrt(head_dim)`
   = 0.08838834764831843。身份鍵用 `==` 比對浮點就此靜靜失效。現在算式對齊，且比對
   改用 `math.isclose(rel_tol=1e-12)`。

兩次都是「量到微幅改善、以為成功」——實際上快路徑一次都沒被走到。**驗證方式是在慢
路徑插 raise 再跑端到端**（命中 2 次 → 0 次），不是看數字有沒有變好。

### 找到的第二個：overlap scheduling 讓 TTFT 多付一整個 step

FA2 修完後，剩下的差距在每一種形狀上都是**同一個常數**（plan 提升前 +9.4~10.4 ms，
提升後仍有 +5.3~5.6 ms）：

| 形狀 | plan 提升前 | 提升後 |
|---|---|---|
| repeat（3890 tok，全命中） | +9.4 ms | **+5.6 ms** |
| turn（7838 tok） | +9.4 ms | **+5.3 ms** |
| cold（5892 tok） | +10.4 ms | **+5.6 ms** |

與長度無關 ⇒ 不是計算。對照實驗確認：`disable_overlap_schedule=True` 讓
TTFT 從 **29.96 → 20.04 ms**，正好是那 9.9 ms。（其中約 4 ms 事後證明是 plan 的
同步，已由 plan 提升拿回；剩下約 5.5 ms 才是排程器那一步。）

機制：`event_loop_overlap` 先 dispatch 當前 forward、再 commit 上一步結果，而
`check_finished` 看的是 `len(output_ids)` —— 只有 commit 時才增加。所以一個
`max_new_tokens=1` 的請求，在排程器還不知道它已經結束時就被多排了一個 forward。

vLLM 的作法是 `num_output_placeholders`（`v1/core/sched/async_scheduler.py:19`）：排程當下
就把「這一步會產生的 token」記帳，於是請求一排到最後一個 token 就不再被排。

**已修**，但沒有照抄 —— 照抄需要在 dispatch 當下送 Finish 事件，而
`FinishEvent::apply` 直接呼叫 `FreeRequest`
（`csrc/fsm/forward_events.cpp:718 FinishEvent::apply`），會在 forward 還在飛的時候
釋放該請求的 KV pages。vLLM 靠 `defer_block_free` / `last_sched_seq` 圍欄擋住
（`v1/core/sched/scheduler.py:1169 _update_after_schedule`），我們沒有那道圍欄。

改成**不通知排程器，只是不執行那一步**：`_rids_completing_this_step` 在 dispatch 當下
算出「這次 forward 會產生最後一個 token」的請求集合；下一輪若整個 plan 都落在這個集合
裡，就把 `forward_op` 設為 `None`。排程器什麼都沒被告知，頁面仍歸該請求所有，由 commit
時照常送出的 Finish 釋放 —— **所以不可能把飛行中 forward 正在寫的頁面釋放掉**。限定
純 decode 的 plan：chunked prefill 即使對最後一個 token 的請求也是真工作。

和 vLLM 一樣只看長度停止；EOS 停止無法在 token 產生前得知，那條路徑仍多付一個 step
（vLLM 的 placeholder 記帳同樣看不到 EOS）。

| | 之前 | **之後** | 關掉 overlap |
|---|---|---|---|
| TTFT（2 tok，1 output） | 29.96 ms | **15.85** | 16.02 |
| 8 個 output token | — | **94.93 ms** | 99.64 |

overlap 終於發揮作用：修好之前它比關掉還慢。

### 剩下的 rag 缺口：profile 說是 KV store 搬運，不是 attention 也不是派送

對**真正的部分命中**請求跑 profiler（第一次量錯了：先 warm 同一個 prompt 會把部分命中
變成完全命中，rag 就和 repeat 一樣快，等於什麼都沒量到）。用全新後綴重量，
device 合計 48.76 ms / 772 個 kernel：

| device | 次數 | kernel |
|---|---|---|
| 13.02 ms | 36x | GEMM 256x128 |
| **11.46 ms** | **36x** | **`_kv_transfer_per_layer_capped_kernel`** |
| 6.97 ms | 72x | GEMM 64x256 |
| 6.01 ms | 36x | GEMM 128x256 |
| 4.54 ms | 36x | FA2 attention |
| **2.75 ms** | **1x** | **`_kv_transfer_all_layer_cs32_kernel`** |
| 其餘全部 | | < 1 ms |

**KV transfer 合計 14.21 ms，佔 device 時間的 29%** —— vLLM 沒有這一層
（`enable_kvstore=True` 是我們的預設，把 KV 卸載到 host，部分命中時要搬回來）。
FA2 修完後 attention 只剩 4.54 ms，已經不是問題。

**方向與量都對得上**：36 次的那顆是 `load_to_device_per_layer`（host → device），
1 次的那顆是 `backup_from_device_all_layer`（device → host）。前綴 3840 token ×
8 heads × 128 dim × 2(k+v) × 2 bytes = **每層 15.7 MB**，318 µs/層 → 49 GB/s，
正是 PCIe5 x16 的 H2D 頻寬。**也就是每個 rag 請求都把整個 0.55 GB 前綴從 host 搬回
device**，而卡上還有約 68 GB 的 KV cache 空著。

機制在 `csrc/scheduler/operations/forward.cpp:390`：

```cpp
loadback_diff = match_result.NodesWithout<ResourceType::Device>();
```

匹配到的節點只要沒有 device 頁面就進 loadback。所以問題不在 loadback 本身，而在
**剛用過的前綴為什麼會變成 host-only** —— 完全相同的重送（`repeat` 形狀）沒有這個
成本（22.6 ms，還贏 vLLM），只有後綴不同時才發生。

`disable_kvstore=True` 只讓 rag 從 40.93 → 38.12 ms（2.8 ms），**不是 14 ms**：那些
搬運大多和 GEMM 重疊。所以「關掉它」不是答案，而且關掉會失去 host 層的容量。

**根因**在 `csrc/fsm/forward_events.cpp:822 WriteBackDoneEvent`，而且原本的註解就寫著
後果：「written-back cache demotes to host-only, so the next hit must load back.」
`DemoteWrittenBackDevice` 沒有任何壓力判斷 —— 每個請求完成後**無條件**把已寫回的頁面
從 device 釋放。device 層淪為暫存區、host 層才是快取。

改成只在 device 真的有壓力時才降級（可用頁面低於總量 1/8）。已寫回 host 的節點被
`EnsureCapacity` 驅逐是免費的，所以留著不花額外代價。

| | 之前 | **之後** |
|---|---|---|
| 部分命中的 device 總時間 | 48.76 ms | **36.41 ms**（−25%） |
| `_kv_transfer_per_layer_capped_kernel` | 36x、11.46 ms | **完全消失** |
| kernel 數 | 772 | 734 |
| TTFT rag | 38.0 ms | 37.7 |
| Throughput shared | 2263.5 tok/s | 2245.5 |

**GPU 工作少了 25%，但沒有轉成使用者可見的收益** —— 那些搬運本來就和 GEMM 重疊
（`disable_kvstore` 實驗早就預告了：14.2 ms 裡只有 2.8 ms 在關鍵路徑上），而
throughput 測試裡 64 個請求同時共用前綴、頁面全程 pinned，本來就不會 loadback。
依本專案的 +10% 門檻**這不算結果**；保留的理由是它嚴格少做事、輸出逐位相同。
真正該受益的是 GPU 飽和的並行 RAG 負載，那個我還沒量到。

過程中我一度誤判為正確性回歸：換 .so 後第一次跑輸出不同。連跑三次才發現那是冷 JIT
快取造成的，新舊 .so 各跑三次都回到同一個基準。**換掉編譯產物後第一次的執行不能當數據。**

### shared throughput：降級挑錯了對象（未修）

先前記的「shared 輸 10~14%」是**量測擺設造成的**：harness 先跑 `unique`（64 個不重複的
長 prompt）再跑 `shared`，前者把 device 快取塞滿。單獨跑 `shared` 時我們每個輸出長度
都贏：

| out | tokenspeed | vLLM | |
|---|---|---|---|
| 32 | **1745.0** tok/s | 1670.2 | 贏 4.5% |
| 128 | **2666.4** | 2573.8 | 贏 3.6% |
| 256 | **2888.2** | 2785.6 | 贏 3.7% |

線性拆解：decode 313.9 vs 324.3 µs/token（贏 3.3%）、prefill+開銷 0.527 vs 0.565 s
（贏 6.7%）。兩段都贏。

真正的差別在**快取有壓力之後**：我們 2666 → 2221（−17%），vLLM 2573 → 2573（不掉）。
關掉 kvstore 我們回到 2656.7，所以那 17% 全部是 kvstore。

但它**不是每請求成本**——這一點我一開始算錯了（64 × 11.46 ms ≈ 733 ms 對上實測
610 ms，那是巧合）。按請求數掃描是一個階梯而非斜率：

| N | kvstore 開 | 關 |
|---|---|---|
| 16 | 1.86 s | 1.86 s |
| 32 | 2.42 s | 2.42 s |
| 64 | 3.69 s | 3.08 s |

只有越過壓力門檻才付錢。所以問題在**降級挑錯對象**：
`WriteBackDoneEvent` 降的是「剛完成的那個請求的節點鏈」，而不是最冷的節點。於是一個
64 個請求正要用的熱共用前綴，只因為某個用過它的請求結束就被降到 host。vLLM 沒有 host
層，同樣壓力下走 LRU 驅逐，熱前綴根本不會被碰。

這也解釋了為什麼壓力閘（上一節）對單請求 rag 有效、對這裡無效：那裡從頭到尾沒有壓力。
**試過「降級改走 LRU」，失敗，已回退。**

想法是：有壓力時不要降「剛完成的這條鏈」，改叫既有的
`EnsureCapacityByEvict<Device>` 去回收。理由看起來很紮實 —— `Evict` 是
`lru_leaves_` 最舊優先，而對已寫回的節點來說「拆掉 device 那一份」本來就等於降級
（`PruneEmptyByNode` 只在節點不在 device 也不在 host 時才移除，所以 host 副本留得住），
而且它走的是驅逐的 `OnKVEvict` 簿記而不是降級的 `OnKVDeviceDemote`。

實測兩件事都不成立：

- **throughput 完全沒動**（2223.2 vs 2221.6 tok/s）
- **輸出變得不確定** —— 同一份建置、非首次執行，仍會跑出和基準不同的文字

第二點才是回退的理由。從請求完成的路徑上呼叫驅逐，會在有請求在飛的時候動樹，
`Evict` 雖然會跳過 `IsEvictable()` 為假的節點，但顯然還有我沒看懂的耦合。
**在讀懂之前不會再動這條路徑。** 回退後連跑四次輸出穩定。

下一次嘗試應該從「為什麼一條剛完成的鏈上的節點，會同時是別人正要用的熱節點」下手 ——
也就是 refcount 在什麼時機掉到 0，而不是換一套挑選順序。

**兩個可歸因的差距**

1. **~~page size 64 讓每次部分命中少重用最多 63 個 token~~ —— 已修（預設改為 16）。**
   同一個 RAG 請求，vLLM 重用 3888／3890（block_size 16），我們原本只重用 3840。

   | block_size | TTFT rag | cached | TTFT repeat | Throughput unique | shared |
   |---|---|---|---|---|---|
   | **16**（新預設） | **32.94 ms** | 3888 | **19.46 ms** | 444.6 | 2219.3 |
   | 32 | 33.78 | 3872 | 21.39 | — | — |
   | 64（舊預設） | 37.85 | 3840 | 22.62 | 446.9 | 2243.2 |

   延遲 −13~14%，throughput −0.5~1.1%（雜訊內）。**repeat 是完全命中、沒有多算任何
   token 卻也快了 3.2 ms**，所以 64 的代價不只是重用粒度，還有 page table 本身。

   沒有任何結構原因需要 64：唯一的互動是 `mamba_cache_chunk_size =
   max(FLA_CHUNK_SIZE, block_size)`，而 `FLA_CHUNK_SIZE` 本身就是 64，所以 recurrent
   側不受影響。**只在 dense 模型上驗證** —— hybrid checkpoint 在這個容器裡兩種 page
   size 都起不來（flashinfer 的 xqa kernel JIT 編譯失敗），那條路徑是靠上面的
   `max()` 論證而非量測。
2. **部分命中路徑每 token 慢 1.8x。** 以 repeat 當固定開銷基準相減：
   我們 15.1 ms／360 token = 42 µs/token，vLLM 7.2 ms／312 token = 23 µs/token。

cold prefill 的算術對得上：vLLM 的 273 ms ≈ 5892 token 的純 GEMM device 時間
（依實測 46 ms／1024 token 外推為 264 ms），也就是它貼著計算 roofline 跑。
我們的 381 ms 比同一個 GEMM 地板高出 **117 ms**，而 host launch 全部加起來只有
約 5.4 ms —— **所以那 117 ms 是 device 時間，不是派送成本**。最大的單一候選是長序列的
extend attention（4096 token 時單層 915 µs，5892 時外推每層約 1.9 ms × 36 層 ≈ 68 ms）。

---

## 環境：四顆編譯 kernel 之前全部載不進來

`objs/` 下的預編譯 `.so`（rope、marlin、routing、rmsnorm_fused_parallel）連結到
`GLIBCXX_3.4.32`，容器的系統 libstdc++ 沒有這個版本。`/workspace/roy/.libstdcxx-shim`
就是為此存在，但**唯一能讓這個容器跑起來的環境腳本 `sm120_env.sh` 沒有把它加進
`LD_LIBRARY_PATH`**（該腳本已 glob `nvidia/*/lib`，cu13 那半本來就有）。

後果不是「安靜地慢」而是「呼叫時才炸」：`cuda_embedding_rope` 是 `Priority.PERFORMANT`，
選擇層一定選它，載入卻在第一次呼叫才發生。GPU 1 上那台跑了 11 小時的 27B 服務，啟動
指令裡就手寫了一行 `export LD_LIBRARY_PATH=/workspace/roy/.libstdcxx-shim:$...` ——
別人也踩過，只是就地繞過去了。已補進 `sm120_env.sh`。補上之後
`test/thirdparty/test_cuda.py` 的 65 個測試才跑得起來，全過。

### RoPE：wrapper 從 10.51 µs 降到 7.31 µs

kernel 一直是贏的（1.38–1.46 µs vs vLLM 2.35–2.52），只是被 wrapper 蓋住。改動在
`thirdparty/cuda/rope.py`：in-place 時 `q_rope is query`，原本把同一個 view 算兩次；
`positions` 已是 int64 時仍呼叫 `.to()`；`_view_3d`/`_view_3d_value` 是每次呼叫重新
定義的閉包；模組 handle 每次重新解析。全部拿掉之後：

| tokens | ts device | vLLM device | ts host | vLLM host |
|---|---|---|---|---|
| 1 | **1.38** | 2.35 | **8.92** | 9.46 |
| 32 | **1.46** | 2.52 | **8.89** | 9.32 |
| 1024 | **6.63** | 6.80 | **9.06** | 9.28 |
| 4096 | 20.80 | 20.75 | 14.85 | 14.15 |

（1024／4096 兩列 GB/s 超過 HBM peak，兩邊都是 L2 常駐，那兩格不是頻寬比較。）
數值對獨立 fp32 參考在 bf16 的 1 ulp 內，out-of-place 與 in-place 逐位相同，int32
positions 仍正確 —— 跳過 `.to()` 沒有破壞它。

### Qwen3 接上融合 RoPE+KV 寫入（等價，但收益不在預期的地方）

`llama` / `gpt_oss` / `minimax_m2` 都走 `create_fused_set_kv_buffer_arg`，Qwen3 與整個
qwen3_5 家族沒有。接上去了，並比 `llama.py` 多做兩件事：

1. 呼叫 `select_out_cache_loc`（gpt_oss 有、llama 沒有）。對沒有 flat cache group 的
   backend 是 identity，零成本，但擋住 hybrid 那類寫錯位置的情況。
2. **不配置新的 `q_rope`**，讓 RoPE 就地改寫 q。llama 配一個 contiguous 的新張量，
   而未融合路徑傳的是 qkv 的 strided slice —— 佈局不同會改變 attention kernel 的
   末位，Qwen3-0.6B greedy decode 在近似平手的 logits 上就分岔了
   （`atmosphere.` vs `atmosphere and`）。就地改寫之後三個提示與未融合基準逐字相同，
   而且每層少一次配置。

等價性是逐位驗證的，不是靠肉眼看文字：融合與「RoPE 後另外 scatter」寫進 cache 的
k/v 與 q_rope，在 n=1/7/64/1024 × neox=True/False 全部 `torch.equal` 為真，
max|Δ| = 0.00e+00。

**但收益量不到，原因是我的前提反了。** `MHAAttnBackend.support_kv_cache_prewrite`
只在 **decode** 回傳 True，而 decode 是被 CUDA graph 捕捉的 —— host launch 成本早就
不存在了，能省的只剩 device 端一次 launch（約 1 µs × 36 層 = 36 µs，對 11.8 ms 的
step 是 0.3%）。真正有 host 成本的 prefill 反而不允許 prewrite。

Qwen3-8B、1024-token prompt、5 次重複的 A/B：

| | TTFT 中位數 | 每 token |
|---|---|---|
| 融合 | 176.23 ms | 11.774 ms |
| 未融合 | 175.98 ms | 11.829 ms |

TTFT 差 0.14%、每 token 差 0.46%，都在雜訊內（TTFT 散布 175.3–176.5）。方向對，但
**依本專案自己的門檻（+10%），這不算結果**。分支確實有執行（在該分支插 raise 驗證過），
改動保留的理由是等價、少一次配置、且讓 Qwen3 與其他模型一致 —— 不是因為它更快。

**這一格的正確目標因此變成 prefill 的 KV 寫入**：那裡 host 成本是真的（Triton launch
14.9 µs vs vLLM 7.5），而融合這條路被 backend 擋住，得從 launch 成本本身下手。

### 派送層：量到了、修了一部分、但它不是端到端的原因

`select_kernel` 即使**快取命中**也每次重做：`f"TOKENSPEED_KERNEL_OVERRIDE_{family.upper()}_{mode.upper()}"`
加一次 `os.environ.get`（0.46 µs —— `os.environ` 是每次查詢都解碼的 `_Environ`）、
`_make_cache_key` 的 `tuple(sorted(traits.items()))`（0.764 µs，9 個項目）、
`cache_get` 的雜湊（0.353 µs）。`mm` 與 `mha_extend_with_kvcache` 另外無條件建
`shape_params` dict 再 `**` 展開，而兩個消費者在未開 profiling 時都直接丟掉。

改法：環境覆寫每 (family, mode) 只查一次（`_env_override_for`，可用
`clear_env_overrides()` 清除供測試使用）；`shape_params` 延後到
`capture.enabled or ProfilingState.get().active` 才建。

| | 之前 | 之後 | torch/裸 kernel |
|---|---|---|---|
| `tokenspeed_kernel.mm`（M=1） | 18.95 µs | **16.57** | 7.98 |
| `mha_extend_with_kvcache` 開銷（128 tok） | 24.36 µs | **14.93** | — |
| 純 bookkeeping（不發 kernel） | 9.71 µs | **9.24** | — |

**但端到端沒動**：175.86 vs 176.23 ms（雜訊內）。理由是算得出來的 ——
GEMM 每層 4 次 × 36 層 × 2.38 µs + attention 36 × 9.4 µs ≈ 0.68 ms，對 176 ms 是 0.4%。
派送成本只在 kernel 比它短的時候看得見：total_q=1024 時 wrapper 開銷是 **0**（kernel 70 µs
完全蓋過），4096 時是 0.09 µs。

**所以 prefill 的 17.8% 不在派送層，也不在 GEMM kernel（device 與 torch 打平 0.6%）。**
下一步必須是對真實 prefill forward 跑 profiler 逐 kernel 歸因，不能再靠微量測推論 ——
這正是量測陷阱 3 說過的事。

## 已完成的兩項整合

### GDN chunk prefill（`TOKENSPEED_ENABLE_FI_GDN_SM120=1`）

48 層一次 prefill，host / device 兩軸：

| tokens | Triton（vLLM 也是這條） | 第一版 | **現在** |
|---|---|---|---|
| 64 | 17.9 ms | 13.8 ms | **11.4 ms** |
| 1024 | 18.8 ms | 13.9 ms | **11.6 ms** |
| 4096 | 19.0 / 28.4 ms | 13.9 / 19.7 ms | **11.6 / 19.8 ms** |

**對 vLLM 是 1.65x** —— 它在 sm_120 上走的就是這條 Triton FLA。

第二步的修法是換掉我們自己的 l2norm：Triton 的 `l2norm_fwd` 跑在 **380 GB/s，只有同
樣位元組純複製（1310 GB/s）的 29%**，22.8 µs/call；flashinfer 的 rmsnorm 是 9.4 µs。
兩者只差一個常數（`l2norm(x) == rmsnorm(x, w=1/√D)`，實測 max|diff| 1.95e-03、
cosine 0.9999996），所以直接改走 rmsnorm。l2norm 從 4.18 ms → 1.67 ms。

裸 kernel 是 6.0 ms，還剩約 5.4 ms 包裝成本。下一個明確的一步是把 q/k 的 l2norm 併成
一次呼叫（TurboMind 的 `invokeL2NormalizeQK` 就是一顆 kernel 同時做 q 和 k），但 q/k
在我們這裡是兩個獨立張量，要動到上游的 `fused_qkv_split_gdn_prefill`。

改的不是 kernel，是**繞過 flashinfer 公開 API 的 `@flashinfer_api` 裝飾器**
（0.33 ms/call × 48 層 = 15.8 ms）。樹裡原本記著「flashinfer sm120 比較慢」並
`return False` —— 那個觀察是對的，歸因錯了。

數值：max|d out| = 6.1e-05，含非零初始狀態（續段 chunk）。

### Full attention prefill（`TOKENSPEED_ENABLE_FI_RAGGED_PREFILL=1`）

| model / tokens | Triton | flashinfer | vLLM |
|---|---|---|---|
| 8B @4096 (×36) | 60.9 ms | **18.0 ms** | 19.2 ms |
| 27B @4096 (×16) | 39.5 ms | **12.5 ms** | 11.6 ms |

三個修法：**plan 快取**（plan 是 per-batch 不是 per-layer；SGLang 的
`fast_prefill_plan`、TRT-LLM 的 `_plan_params_to_wrappers` 都為此存在）、
**繞過 `run()` 的裝飾器**、**自己配置輸出張量**（讓 `run` 自己配置每次多 26%：
36 次呼叫 @256 tokens 是 0.556 vs 0.413 ms）。

**短序列還輸 vLLM 1.7x，但差距不在 kernel。** 裸 flashinfer 呼叫是 0.413 ms／36 次，
**已經比 vLLM 的 0.63 快**；整合路徑是 1.09 ms。中間那 0.68 ms（約 19 µs/call）
全在我們自己的 `mha_prefill` 派送層：signature 建構、`select_kernel`、每次呼叫的
Python glue。一次 forward 的 36 層形狀完全相同，卻各自重跑一次選擇 —— 這是架構成本，
不是 kernel 成本，要在 runtime 層解（一次 forward 解析一次）。

數值：max|diff| ≤ 1.95e-03。

---

## 參照引擎的實作方法（用在哪）

| engine | 讀到的作法 | 用在 |
|---|---|---|
| SGLang | `fast_prefill_plan` —— sync-free plan，避開 `plan()` 的 D2H 同步 | full attention 的 plan 快取 |
| TRT-LLM | `_plan_params_to_wrappers` 快取；註解明寫「plan() does device->host syncs」 | 同上，佐證診斷 |
| TurboMind | attention 用 `Arch<start, end>` 半開區間；`Sm80: Arch<800>` 上界開放，所以 sm_120 走 Sm80 路徑 | 驗證 portable kernel 該用「開放下界、不設上界」 |
| TurboMind | GDN 每層 3 launch（融合 cumsum+prepare、kkt+solve_tril、h+o） | `30-prefill.md` 的融合目標 |
| vLLM | 量測基準；小 op 的手寫 CUDA kernel 在 `_C_stable_libtorch.abi3.so`（Apache-2.0、ABI-stable） | 小 op 的移植來源候選 |

---

## 一個橫跨多顆 kernel 的型態（2026-08-09 重量後改寫）

原本這一節寫的是「我們的小 op 有一個與尺寸無關的地板，vLLM 沒有」，並列了三個地板
數字。**那三個數字量錯了，結論也反了。** 用「device 走 profiler、host 走無 event 的
enqueue 迴圈」兩把尺重量之後：

| op | 我們 device | vLLM device | 我們 host | vLLM host |
|---|---|---|---|---|
| RoPE (32 tok) | **1.46 µs** | 2.52 | **8.89 µs** | 9.32 |
| RMSNorm (32 rows) | **1.92** | 2.07 | **10.17** | 17.74 |
| SiLU-and-mul (32 rows) | **1.47** | 1.72 | **11.75** | 12.70 |
| KV cache 寫入 (32 tok) | **0.99** | 1.73 | 14.92 | **7.54** |

**每一顆 kernel 本身都比 vLLM 快**（1.08–2.2x）。地板是真的存在，但它在 **Python
wrapper**，不在 kernel：小尺寸時 host 時間大於 device 時間，GPU 在等 host。

四顆裡只有 KV 寫入的 host 真的輸，因為它是唯一還走 Triton launch 的（此機 Triton 空
launch 地板 7.7–13.9 µs，量到的 14.9 µs 幾乎全是它）。其餘三顆走編譯好的 CUDA/FFI
路徑，host 成本本來就低於 vLLM。

RoPE 的 wrapper 也曾經是輸的（10.51 µs），拆開後是 4 次 `.view()` 3.08 µs、
`positions.to(int64)` 0.81 µs、模組查找 0.24 µs，對照 FFI 呼叫本身 4.63 µs 的地板。
去掉重複的 view、跳過 no-op 轉型、把 handle 綁在模組層之後降到 **7.31 µs**，低於
vLLM 的 7.77。**wrapper 是可以修的，而且修的量級和 kernel 本身同一個數量級。**

反例（同樣重要）：full attention 短序列的 kernel **已經贏了**（裸 flashinfer 呼叫
0.413 ms／36 次 vs vLLM 0.63），輸的是我們的派送層。所以「小工作量輸」有至少兩個
不同的原因，不能一概而論。

## 五個反覆出現的量測陷阱（都踩過）

1. **量了引擎不走的路徑。** GDN 先量裸 kernel（快 2x）而非整合路徑（慢 1.7x）；小 op
   先量 `ops/*/triton.py`，但 NVIDIA 上 runtime import 的是 `ops/*/flashinfer.py`，
   Triton 版只是 AMD 分支。**先確認 runtime 實際 import 什麼再量。**
2. **GB/s 超過 HBM peak = L2 常駐，不是 kernel 快。** decode 8/4096 讀到 2736 GB/s、
   rmsnorm@1024 讀到 2097 GB/s、KV 寫入 @4096 兩邊都超過 2000 GB/s —— 那幾格不是
   頻寬比較。
3. **單獨量的片段不會相加。** MoE 的分項表說 routing 佔 0.25/1.23，照著改只買到 4%；
   attention 的 30.3 µs/call 拆出來只能歸因 19.4 µs。孤立微量測會誤導歸因，**要對
   真實路徑跑 profiler**（並記得此機 profiler 的 host 時間膨脹 3.6x，得先校準）。
4. **host 比 device 慢時，CUDA event 的差值量的是 host，不是 kernel。** RoPE 用 event
   量到 16.3 µs、看起來輸 vLLM 1.28x；profiler 說 kernel 只有 1.47 µs，我們其實贏
   1.7x。GPU 在兩個 event 之間空等 host 而已。**兩把尺分開量**：device 用 profiler
   （它的 kernel 名稱與次數是準的），host 用不含 event、不 sync 的 enqueue 迴圈。
   上面那張表整個型態的反轉就是這一條造成的。
5. **參照引擎也會掉到它自己的慢路徑。** vLLM 的 RMSNorm 現在經由 `vllm.ir.ops` 派送，
   而裸 `VllmConfig()` 預設開 inductor，CUDA 平台在編譯模式下的 op priority 是
   `["native"]` —— 量到的是 eager PyTorch 分解（8–11 個 launch、194 µs @4096），不是
   它的手寫 kernel。要 `CompilationMode.NONE` 讓 priority 變成 `["vllm_c", "native"]`。
   **贏得莫名其妙的時候，先確認對手有沒有站上場。**

---

## MoE (bf16)：目前最大的缺口

| tokens | vLLM `fused_experts` | 我們（flashinfer cutlass） |
|---|---|---|
| 1 | 0.152 ms | 3.148 ms（**21x**） |
| 32 | 0.722 ms | 3.183 ms |
| 128 | 1.096 ms | 3.193 ms |
| 1024 | 1.204 ms | 3.167 ms |
| 4096 | 1.530 ms | 3.184 ms（2.1x） |

**我們是 3.18 ms 平的，1 個 token 與 4096 個 token 同價。** 排除的原因：
flashinfer autotuner 不是元凶（`autotune(False)` 下仍是 3.13 ms）。剩下的解釋是
kernel 形狀 —— 256 個專家 top-8，T=1 只需要 8 個專家，但 cutlass grouped GEMM 對全部
256 個都付代價；vLLM 的 Triton `fused_moe` 依專家排序 token、只對有 token 的專家發
block，所以隨 T 縮放。

**這張卡上我們只有這一個 bf16 MoE kernel**：`flashinfer_trtllm_unquant` 需要
tcgen05，在 sm_120 上 `NoKernelFoundError`（閘是對的）；Triton 的 MoE 全是
fp8/mxfp4，沒有 bf16 版；`gluon_bf16` 是 AMD。

### 已完成：把既有的 Triton ragged MoE 開到 bf16

新增 `ops/moe/triton/bf16.py`，註冊 `triton_bf16_precomputed_moe_apply`
（solution `triton`），與 cutlass 版並存；**預設仍是 cutlass**
（`Priority.PORTABLE` 輸給它的 `SPECIALIZED`）。

| tokens | vLLM | cutlass（原本） | **triton bf16（新）** |
|---|---|---|---|
| 1 | 0.152 ms | 3.148 ms | **1.228 ms** |
| 32 | 0.722 ms | 3.183 ms | **1.426 ms** |
| 128 | 1.096 ms | 3.193 ms | **1.313 ms** |
| 1024 | 1.204 ms | 3.167 ms | **1.522 ms** |
| 4096 | 1.530 ms | 3.184 ms | **2.637 ms** |

數值等價：max|d| ≤ 7.6e-06、cosine 0.999987（bf16 累加差）。

**過程中撞到兩個 sm_120 專屬的障礙，都是硬體能力問題不是設定問題：**

1. **shared memory**：未約束時自動選到的 tile 要 115004 bytes，超過 sm_120 的
   101376。加了 tile 上限（`ops/moe/triton/bf16.py` 的 `_smem_guard`），比照
   mxfp4 路徑為 CDNA4 準備的同類護欄。
2. **`.tile::scatter4`**：把 `scatter_indx` / `gammas` 交給 kernel 會讓 Triton 發出
   TMA tile-scatter 指令，ptxas 直接拒絕 —— `Feature '.tile::scatter4' not supported
   on .target 'sm_120a'`，任何 tile 大小都一樣，只有單 token 例外。改成在 torch 裡
   自己做 gate 縮放與反排序，代價是多一趟 `[tokens × top_k, hidden]`。

**仍落後 vLLM 1.2–8x**，最大差距在 T=1（0.152 vs 1.228）。已拆開歸因（T=1，此時
GEMM 計算量可忽略，剩下的都是固定開銷）：

| 部分 | ms |
|---|---|
| **routing（argsort + scatter_add + metadata）** | **0.2488** |
| matmul w13 | 0.3451 |
| silu gate/up | 0.0298 |
| matmul w2 | 0.2276 |
| 手動散射 + gate + sum | 0.0410 |

兩個先前的嫌疑都排除了：**手動散射只花 0.041 ms**；**tile 護欄不是瓶頸**（關掉它
≤1024 的時間幾乎不變，而 4096 時兩次 matmul 各自都會超標，所以兩邊都需要）。

**routing 也試過了，不是瓶頸。** 加了第二個註冊
`triton_bf16_routed_moe_apply`（`routing_mode: kernel_routing`），改用
`triton_kernels.topk` 的融合排序取代那 8 個 torch op：

| tokens | precomputed（torch routing） | kernel_routing（融合） |
|---|---|---|
| 1 | 1.228 ms | 1.221 ms |
| 1024 | 1.522 ms | 1.431 ms |
| 4096 | 2.637 ms | 2.542 ms |

只買到 4–6%，T=1 完全沒變。**上面那張分項表誤導了我**：每一項是單獨量的裝置時間，
實際執行時 GPU 會重疊，加起來不等於總和，所以「routing 佔 0.25 / 1.23」這個推論
是錯的。

**剩下的成本在 grouped matmul 本身。** T=1（256 個專家只有 8 個活躍）兩次 matmul
單獨量就 0.57 ms，而 vLLM 整個 MoE 是 0.152 ms。這是 `triton_kernels.matmul` 在低
佔用率下的性質 —— 跟 cutlass 版同一類問題，只是輕得多。要再進一步得換 kernel 或讓
grid 真的跟著 ragged metadata 走，不是包裝層能解的。

`flashinfer.trtllm_bf16_moe` 帶融合 routing，但 trtllm-gen 血緣需要 tcgen05：
我們的註冊界在 10.0–10.3，sm_120 用不了（**這個閘是對的**）。

### 原本的評估：不是抄 vLLM 的檔案

先查過再說：vLLM 的 `model_executor/layers/fused_moe/fused_moe.py` 是 1794 行，且與
`vllm.envs` / `modular_kernel` / `config` / `_custom_ops` / `platforms` 深度耦合 ——
**不是乾淨的 1:1 複製**。

我們自己的 Triton MoE（`ops/moe/triton/fp8.py`）走的是另一條血緣：
`triton_kernels.matmul`（OpenAI 的 `triton_kernels` 套件，已 vendored），而它**本來就
依專家分組、隨 T 縮放** —— 正是 cutlass 版缺的性質。它目前只註冊了 fp8 / mxfp4：

| 已註冊 | weight_dtype | vendors |
|---|---|---|
| `triton_mxfp8_precomputed_moe_apply` | fp8 | nvidia |
| `triton_fp8_precomputed_moe_apply` | fp8 | amd |
| `triton_mxfp4_*_moe_apply` | mxfp4 | — |

**沒有 bf16/unquant 的註冊。** 所以工作是「把既有 kernel 開一個 bf16 註冊」，不是
移植一整個檔案 —— 需要一個 bf16 的 weight preprocessor 與 apply 包裝，加上
`weight_dtype: unquant` 的 trait。這比抄 vLLM 便宜得多，而且留在既有血緣裡。

注意 vLLM 那一組帶著自己的警告：`Using default MoE config. Performance might be
sub-optimal! Config file not found ... NVIDIA_RTX_PRO_6000_Blackwell`。它是**未調校**
的 Triton 配置就已經領先 —— 調校後差距只會更大。

## 待辦（依序）

0. ~~長序列 extend attention~~ —— **已修**（FA2 paged，見上）。剩下的是把 plan 從
   每層提到每個 forward，這是 shared throughput 退 4% 的原因。
0b. **overlap scheduling 的一個 step**（+9.4 ms，每一種形狀都付）。要做兩件事：
   in-flight 輸出記帳（vLLM 的 `num_output_placeholders`）**加上** 延後釋放 KV page
   的圍欄（vLLM 的 `defer_block_free`）。只做前者會釋放使用中的頁面。
0c. **page size 64 → 16**（或讓部分命中對齊到更細的粒度）。每次部分命中少重用最多
   63 個 token，RAG／agent 每一輪都付一次。
1. **移植 vLLM 的 Triton `fused_moe` 作為 bf16 MoE 的第二個 solution**（最大缺口）
2. **prefill 的 KV 寫入：削掉 Triton launch 的 host 成本**（14.9 → 目標 <7.5 µs）。
   kernel 已經贏 1.7–2.2x，融合那條路被 backend 擋住（見上），所以要從 launch 本身
   下手 —— 快取 launcher、或改走已編譯的 CUDA 路徑。RoPE 的 wrapper 從 10.51 削到
   7.31 µs 是同一類工作的先例。
3. `mha_prefill` 的派送層：短序列 full attention 落後的真正原因（裸 kernel 0.413 ms
   已贏 vLLM 的 0.63，整合後 1.09）。每次 forward 解析一次 selection，不要每層一次。
4. `fused_add_rmsnorm`：唯一 device 還輸的小 op（32–4096 輸 5–7%）
5. 量化 GEMM：先補完 `70-PORTS.md` T1 的 Marlin GEMM，才有東西可量
6. 端到端 sweep，翻掉兩個 env var 的預設

**已從清單移除**：RoPE（全區間不輸）、RMSNorm、SiLU-and-mul（前兩者的「輸」是量錯，
見上面的量測陷阱 4 與 5）。
