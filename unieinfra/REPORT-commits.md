# 從 sm_120 適配到 RAG —— 逐 commit 報告

範圍：`1dd493da~1..HEAD`，共 68 個 commit（含 re-fork 本身），2026-08-04 至 2026-08-10。
硬體：RTX PRO 6000 Blackwell（sm_120，HBM 1792 GB/s，L2 128 MB）。
模型：`Qwen3-8B`（dense）、`Qwen3.5-27B`（GDN hybrid）、`Qwen3.6-35B-A3B`（MoE）。

起點是**引擎在這張卡上完全跑不起來**。終點是 dense 與 hybrid 兩條路徑都能跑、
在有前綴重用的形狀上贏 vLLM，**在沒有重用的形狀上仍然輸**（見 §12）。

---

## 0. 一句話總結

| 階段 | 狀態 |
|---|---|
| 起點 | Qwen3.5-27B 在 sm_120 上**無法啟動** |
| 讓它啟動 | 3 個宣告錯誤（不是硬體不支援） |
| 讓它變快 | fp8 KV +19.0%，從 −17.9% 追到 +1.8% vs vLLM |
| 讓 RAG 能用 | 前綴重用 0 → 2560 tokens，對自己 TTFT 降 4.5–6.3x |
| 對 vLLM 的實際位置 | 重複形狀贏 1.5–2.6x、RAG 贏 1.2–1.6x、**全新 prompt 高併發輸 1.7x** |

> ## ⚠ 先讀這個：本報告之前的所有 vLLM 對照都是錯的
>
> `Qwen3_5ForConditionalGeneration` 宣告 `IsHybrid` 但沒宣告 `MambaPrefixCaching`，
> vLLM 因此把 `enable_prefix_caching` 解析成 **False，而且不發任何警告**。
> harness 什麼都沒傳 —— 所以這個 repo 裡**每一次 hybrid 對照，比的都是一個
> 前綴快取被關掉的 vLLM**。
>
> RAG 形狀，TTFT p50（conc 1/4/16）：
>
> | vLLM 設定 | TTFT p50 |
> |---|---|
> | 預設值（快取實際關閉） | 447.87 / 1620.73 / 4134.22 ms |
> | 明確傳 `enable_prefix_caching=True` | **112.76 / 358.40 / 1160.97 ms** |
>
> 受影響的錯誤宣稱：`bf156ad4` commit message 的
> "vLLM reuses nothing in either case"、`35-hybrid-prefix-reuse.md` 的
> 「RAG 兩邊都是 0」。**兩句都是錯的。**
>
> `bf156ad4` 自己的收益不變（437 → 89.7 ms 是對我們自己量的），
> 但它補的**不是 vLLM 沒有的能力，是 vLLM 開個開關就有的能力**。
>
> 已修：harness 明確傳旗標並印出 **resolved** 值而非 requested 值（`4e20e1b8`）。
> §12 的所有數字都是修正後重量的。

---

## 1. 倉庫地基（4 commits）

| commit | 做了什麼 | 效果 |
|---|---|---|
| `1dd493da` | 以**共同祖先**重新 fork upstream tokenspeed | `git rebase upstream/main` 從不可能變成一行指令 |
| `ab27cd9b` | 把 fork 的分歧分類成 upstream PR 與 plugin | — |
| `1fa7d5a4` | 允許修改 upstream 檔案，並寫明維持成本的條件 | — |
| `423338aa` | 修正 layout 表中「upstream 未被修改」的錯誤敘述 | — |

原本的 fork 是**整棵樹複製**，與 upstream 沒有共同 git 祖先：rebase 不可能，
分歧要靠人工 diff 兩個工作樹才看得見。

以真正的 fork point 量測，實際分歧是 `tokenspeed-kernel` 3381 行、runtime 3003 行、
`tokenspeed-scheduler` **0 行** —— 而天真 diff 顯示的是 165 個檔案。
絕大部分「分歧」是 upstream 自己的演進，沒有任何工具能把兩者分開。

**代價已經發生**：舊倉庫裡有兩個結果（E45 的 `r = P/N` 容量增益、以及它的 mixed-batch
量測）在任何 commit、任何 diff、任何保留的工作樹裡都找不到對應程式碼。兩者都是對著
「從工作樹複製檔案建出來的 image」量的。**沒有 commit 的結果就是沒有結果。**

---

## 2. 讓它在 sm_120 上啟動（9 commits）

upstream 跑不動 Qwen3.5-27B。三個 bug，都是同一個形狀 ——
**硬體做得到，是一行宣告把它排除掉**。

| commit | 做了什麼 | 效果 |
|---|---|---|
| `48a80ebb` | 擋掉兩個在 sm_120 上會贏得 selection 卻必定失敗的 FlashInfer kernel | 啟動不再 crash |
| `73b27362` | Triton MHA prefill 的 tile 改用**實測**的 shared memory 決定 | `OutOfResources: Required 114688, limit 101376` 消失 |
| `228886f0` | `max_shared_memory_per_sm` 改讀 torch 真正有的屬性 | 從**每張 CUDA 卡都回報 0** 變成 101376 |
| `94d0fa12` | 把 libstdc++ shim 放上 `LD_LIBRARY_PATH` | 4 個預編譯 kernel 從「呼叫時 raise」變成可用；65 個測試變成可跑 |
| `d97efd1c` | 在 engine 啟動時呼叫 `discover_plugins()` | plugin 系統從**完全無效**變成有效 |
| `57312157` | 讓 proton import 變成 optional | — |
| `b93a8a8e` | Triton MHA prefill 跳過越界的 query block | mixed batch **2.04–2.27x**；均勻批次不變 |
| `3eb83e9d` | dense Qwen3 略過它沒有參數的 KV-cache scale | NVFP4 / FP8-KV checkpoint 不再 `KeyError` |
| `4119f79c` | cutlass fused-MoE 的 routing scale 保持 float32 | self-routing 分支在 bf16/fp16 下不再失敗 |

三個啟動 bug 的細節：

1. **XQA decode 被選給它編不出來的 page size** —— traits 宣告了 sliding window、sinks、
   logit cap、LSE，唯獨沒宣告 `page_size`，而 caller 早就在傳了。
2. **TRT-LLM-gen FMHA 被選給它沒編過的 arch** —— `min_arch_version` 是開放區間，
   每一顆更新的晶片都滿足。真正需要的是 tcgen05/TMEM，sm_120 沒有。
3. **Triton tile 塞不進 shared memory** —— tile 階梯按 arch 家族分支，sm_120 依版本號是
   `is_hopper_plus`，卻帶著 sm_89 的 ~100 KB 預算。

第三個還有第二層：明顯的修法「問裝置要預算」**什麼也沒改變**，因為
`PlatformInfo.max_shared_memory_per_sm` 讀的是 `max_shared_memory_per_block`，
**torch 根本沒有這個屬性**，所以這個欄位在有史以來每一張 CUDA 卡上都是 0。
沒有別的地方讀它，所以從沒被發現（`228886f0`）。

`b93a8a8e` 的控制組才是重點：它不是通用加速，而是移除一個特定病理。
grid 的 M 維由批次中最長的 query 決定，較短的序列會被發配「每個 query index 都越界」
的 block —— 它們的 `mask_m` 全為 false、什麼都不寫，但要**走完整個 KV 迴圈**才發現。
均勻批次不變、全 decode 不變，只有 mixed batch 上 2x。

---

## 3. 讓它變快（3 commits）

| commit | 做了什麼 | 效果 |
|---|---|---|
| `996b05f4` | 讓 Triton MHA kernel 接受 fp8 KV cache | **+19.0% throughput**（206.5 → 245.8 tok/s），容量 1.92x |
| `b8a392f8` | fp8-KV 路徑上 query 保持 activation 精度 | 同上（配套） |
| `4c1ad83a` | 兩種粒度的 KV 規劃：attention page P 與 state stride N 分離 | 讓 `--mamba-track-interval` 能觸及幾何；**非** E45 的 +10.3% |

fp8 KV 原本編列為「一個 kernel port」，實際上**它是兩行宣告**。
兩個 Triton kernel 早就會載入量化 cache 並轉上去（`k.to(q.dtype)`）。
擋住它的是 `format_signatures` 對每個 role 只放一種 dtype，所以宣告
`{fp16, bf16}` 橫跨 q/k_cache/v_cache 的意思是「selection 只會提供與 query 同 dtype 的
cache」—— 而**有用的組合（高精度 query 配 fp8 cache）沒有任何 signature 能匹配**。

整體戰績：

| | tok/s | vs vLLM |
|---|---|---|
| 起點 | **完全跑不起來** | — |
| upstream tokenspeed（跑起來之後） | 205.9 | −17.9% |
| **+ fp8 KV** | **255.2** | **+1.8%** |
| vLLM 0.26 | 250.8 | — |

**+1.8% 在噪音內，不算數。** 這條 arm 是雙模振盪（211 對 273，25.1% 視窗跨度），
高模式已經比 vLLM 快 10%；要做的是把它**釘在那裡**，不是找新的空間。
驗收門檻定在 **full post-warmup mean +10%**，低於 ~7% 的差異這台儀器分辨不了。

---

## 4. Mixed prefill+decode batching（5 commits，機制已落地但關閉）

| commit | 做了什麼 | 效果 |
|---|---|---|
| `64e632d8` | 讓 MHA backend 能服務混合批次 | full-attention 半邊完成 |
| `c81fa7fa` | `--enable-mixed-batch` 改成啟動時拒絕，而非跑到一半失敗 | 失敗模式從 crash 變成明確拒絕 |
| `8ecb4b24` | 為混合批次切分 GDN state bounds | （中途嘗試） |
| `69a5f7c5` | 為混合批次切分 GDN query lengths | （中途嘗試） |
| `146c4175` | 在 recurrent 路徑上切分混合批次（**預設關閉**） | 機制在，開關 False |

這一段最有價值的產出不是程式碼，是**為什麼它沒成功**。

五次 serve-crash-read 循環，每次修好前一個錯誤、發現下一個 —— query length 陣列、
kernel 輸出 buffer 的張量對齊、state gather 的 illegal address。三者都是同一個
**放錯層**的症狀：**切分寫在 attention backend，而 backend 不擁有那些張量。**

去讀其他引擎，一個檔案一次就回答了：

| 引擎 | 切分寫在哪 | 順序 | 怎麼定址兩半 |
|---|---|---|---|
| vLLM | mixer（模型層） | decode 先 | token 維 `torch.split` |
| TRT-LLM | mixer | **prefill 先** | `torch.split`，輸出預先配置也切 |
| SGLang | mixer | **prefill 先** | `torch.split` |
| TurboMind | layer plan | decode 先 | **offset，完全不切片** |

vLLM 與 TurboMind decode-first、TRT-LLM 與 SGLang prefill-first ——
所以**順序是選擇，不是需求**。TurboMind 從不實體化 slice：兩個帶 offset 的 launch plan
指向共享 buffer，直接消掉 concat 與對齊陷阱，也就是五次循環裡的兩次。

**只讀一個引擎，這些一個都看不到。**

---

## 5. 前綴快取 —— 這一輪的主軸（4 commits）

這是 `AGENTS.local.md` 第 0 步「逐層歸因」找出來的，也是最容易被漏掉的一格：

| 探針 | 我們 | vLLM |
|---|---|---|
| HTTP ingress | 0.20 ms | 0.45 ms（我們**更快**） |
| 每 decode token | 36.4 ms | 36.9 ms（**平手**） |
| **同樣 4096 token 再送一次** | **660 ms** | **70 ms（9x 差距）** |

throughput harness 每次用全新隨機 prompt —— 刻意為之，為了讓前綴重用不污染量測 ——
而那**正好是「前綴快取完全死掉也不用付代價」的唯一形狀**。

| commit | 做了什麼 | 效果 |
|---|---|---|
| `e1852925` | mamba 快照發佈到**匹配搆得到**的深度 | 27B 重複 prompt 重用從 **0% 變成可用** |
| `46b6614b` | 寫回主機的 KV 留在裝置上，直到裝置池真的缺頁 | 部分命中請求 device time **48.76 → 36.41 ms**，loadback kernel 從 trace 消失（772 → 734 kernels）|
| `54c6b607` | KV 分頁從 64 token 改成 16 | TTFT **−13~14%**；throughput −0.5~1.1%（在重複跨度內）|
| `bf156ad4` | 保留**多個**遞迴快照，讓共享文件可重用 | **RAG 437 → 89.7 ms，重用 0 → 2560 tokens（4.9x）** |

### `e1852925` —— 快照發在一個匹配永遠到不了的深度

```
[TSDBG] InsertMamba   node=0x9874eaf0 depthTok=4096
[TSDBG] augmentMatch  kv_terminal=0x98f522e0 depthTok=4032 found_mamba=(nil)
```

完全重複的請求**必須留至少一個 token 去算**，所以它的匹配天花板比自己的結尾低一頁。
`FindLastMambaNode` 往**上**走，快照是後代不是祖先，於是 `augmentMatch` 把**整個 KV 匹配
丟掉** —— 重複形狀丟 4032 個 token，RAG 形狀丟 2944 個。兩者都恰好差一頁。

### `54c6b607` —— 沒有人分頁分得這麼粗

部分前綴命中只能重用整頁，64-token 的頁在每個邊界最多丟 63 個 token。
vLLM 預設 16，SGLang 預設 1，這裡是 64 —— **而且沒有任何東西需要 64**。

```
block_size   TTFT rag   cached   TTFT repeat   tput unique   tput shared
16           32.94 ms     3888     19.46 ms       444.6         2219.3
64           37.85        3840     22.62          446.9         2243.2
```

注意**完全重複**那一列：那裡沒有多算任何 token，卻仍然快 3.2 ms ——
所以大部分收益是**頁表成本**，不是失去的重用。

### `bf156ad4` —— 一個快照只能待在一個深度

線性注意力不能像 KV 那樣切前綴（要從 token n 續算，需要 n 之後的遞迴狀態），
所以重用是 `min(KV 深度, 最深可用快照)`。我們每個 chunk 只發佈**一個**快照。
發在請求自己結尾的請求，只服務得了完全重複：換個問題的請求匹配到分岔點，
而分岔點在快照**之上**，於是拿到零。

kernel 一行沒改。scan 本來就交出每一段狀態（2690-token 前綴有 42 個），
複製那步是純 gather/scatter；**單一值的是管線**。所以是把路加寬：
allocator 持有 vector、operation 用 `(counts, indices)` 攤平帶出、
backend 把每個請求展開成每個深度一對 (src, dst)。

快照放在 **512 token 的 stride**、每 chunk 上限 6 個：分岔時最多損失一個 stride 的重用，
而不是全部；一份長文件也吃不垮和所有在飛序列共用的池子。配置全程 best-effort ——
池子不夠只會少留快照，不會讓請求失敗。

**實作時發現設計漏掉的兩件事：**

1. **額外 slot 要在發佈點配，不是在四個分配點** —— 只有發佈點知道 chunk 範圍。
   在分配點預留上限會讓一份長文件餓死整批。
2. **發佈的 slot index 必須手動記下來交給 operation** —— 掛上樹時所有權就轉移了，
   allocator 不再知道自己給了哪些；而 operation 正是告訴 GPU 往哪寫的東西。
   少了這步，節點會持有永遠沒被寫入的 slot，之後的命中從**未初始化記憶體**接續。
   **這是整個改動裡唯一會靜默出錯的地方。**

正確性用**相等性**驗證，不是看時間：用 2560 個重用 token 算出的回答，
與同一問題在冷引擎上算的**逐字相同**。錯的遞迴狀態不會當掉，
只會流暢地接續錯誤的前綴。

---

## 6. Kernel 與 wrapper（8 commits）

| commit | 做了什麼 | 效果 |
|---|---|---|
| `936b835f` | CUDA RoPE wrapper 砍掉每次呼叫 3.2 µs 的 Python | **10.51 → 7.31 µs**，低於 vLLM 的 7.77 |
| `edd295cf` | Qwen3 全家改用融合的 RoPE+KV write | 每層少一個 launch；**端到端無可量測變化**（見下） |
| `da0619da` | 不再於每次 kernel 呼叫重新推導 dispatch 狀態 | `mm` 18.95 → 16.57 µs；`mha_extend` wrapper 24.36 → 14.93 µs；**端到端 0** |
| `de851d8d` | 加入 FA2 paged extend kernel，每個 forward 只 plan 一次 | 對 Triton **1.86x@1024 → 4.57x@16384**；RAG 形狀 **15.2x** |
| `bdbb3571` | 啟用融合的 sm_120 GDN chunk-prefill kernel | 48 層一次 prefill **17.9 → 11.4 ms**；比 vLLM 自己走的路徑快 **1.65x** |
| `a90de320` | 為 bf16 專家註冊 ragged Triton matmul | 對現行 cutlass kernel **2.1–2.6x**；仍落後 vLLM 1.2–8x |
| `4d35899c` | 用**探測**取代版本旗標來找該 arch 的 GDN kernel | 解開 4 道各自足以擋死的閘門 |
| `29f8eda0` | 為大模型高併發放大 FA2 workspace（128 → 512 MB） | 27B 在 conc≥4 不再 `Buffer overflow` |

三個值得記的細節：

**`936b835f`**：kernel 本來就贏 vLLM（1.38–1.46 µs 對 2.35–2.52），
**差距完全在 wrapper**。逐項 ablate：四個 `.view()` 花 3.08 µs、
`positions.to(int64)` 0.81 µs、重新解析 module handle 0.24 µs，
而 FFI 呼叫本身的地板是 4.63 µs。

**`da0619da` 與 `edd295cf` 都誠實記錄「端到端沒有變化」**，並寫明算術：
36 層 × (4 個 GEMM + 1 個 attention) ≈ 0.68 ms，對比 176 ms 的 TTFT。
**dispatch 成本只有在 kernel 比它還短的時候才看得見。**

**`4d35899c`** 是最值得警惕的一個。flashinfer 的 `chunk_gated_delta_rule_sm120`
一直都在、能跑、而且正確（對 Triton scan max|diff| = 3.1e-4），但四道閘門各自擋死它：

1. `if platform.is_blackwell:` —— `is_blackwell` 是 `major == 10`，sm_120 是 12，
   整個 import block 從不執行。
2. `from flashinfer.gdn_kernels import _has_blackwell_prefill` —— 0.6.15 已經不匯出這個
   符號，而且它 import 在 kernel **之前**，所以它的 ImportError 連 kernel 一起跳過。
   **一個寫成「import 版本旗標」的可用性探測，會把上游改名變成靜默的功能消失，
   而且沒有任何 log。**
3. `cuda_major >= 13` 套用到每個 arch —— 那個下界對 sm10x 是真的，對 sm_120 不是。

---

## 7. Scheduler（1 commit）

| commit | 做了什麼 | 效果 |
|---|---|---|
| `9e08f8ce` | 不為一個已經結束的請求再跑一步 | Qwen3-8B TTFT **29.96 → 15.85 ms** |

overlap 迴圈下，scheduler 要等結果 commit 才知道請求結束了，
而那發生在**下一個 forward 已經派發之後** —— 所以最後一個 token 上的請求
還會被排一步、而且真的執行。關掉 overlap 讓 TTFT 從 29.96 掉到 20.04，
而且 8 token 與 7838 token 的 prompt 差距同樣是 ~5.5 ms，
**是一步，不是計算量**。

vLLM 在排程時就數這一步會產出幾個 token（`num_output_placeholders`）。
直接照抄的話要在派發時就發 Finish，而 `FinishEvent::apply` 會呼叫 `FreeRequest`
—— **在 forward 還在寫的時候釋放 KV 頁**。vLLM 有 `defer_block_free` 圍欄，我們沒有。

所以改成**什麼都不告訴 scheduler**：記下在飛 forward 會結束哪些請求，
下一輪若計畫裡每個請求都在其中就跳過派發。頁的所有權不動，
由 commit 照常發出的 Finish 釋放 —— 這就是為什麼它**不可能**把頁從執行中的 forward 底下抽走。
限定在純 decode 計畫：chunked prefill 即使對最後一個 token 的請求也是真工作。

---

## 8. 量化（4 commits）

| commit | 做了什麼 | 效果 |
|---|---|---|
| `a1483ec5` | 補回 w4a16 Marlin 路徑的 Python 半邊 | 30 個測試；**端到端仍不可跑**（CUDA kernel 沒建） |
| `3ebb577c` | 記錄 w4a16 未校準時 ppl +7.2% | 負面結果 |
| `d4e075ce` | 記錄 w4a16 沒有可重用的既有路徑 —— scheme 是死碼 | 能力缺口，非效能問題 |
| `36e416a4` | 修正 w4a16 patch 裡一個未經驗證的宣稱 | — |

`compressed_tensors_wNa16` 匯入了**十個樹裡根本不存在的名字**，
還掛著一整檔的 `# ruff: noqa: F821` 讓 linter 閉嘴 —— 這個 scheme 是會在載入時 raise 的死碼。
`a1483ec5` 把 noqa 拿掉，**重點就是這個：檔案現在會誠實地失敗。**

---

## 9. 文件、量測基礎建設與被否定的嘗試（31 commits）

### 五引擎橫向對照（`18f1fa92`, `f7d6015f`, `55004cc2`, `8def103b`, `5f5635a7`）

`unieinfra/xref/`：API → scheduler → prefill → decode 四階段、spec decode、quant，
五個引擎逐 function / 逐 kernel 攤平成同一張表，附 `check_xref.py` 驗證每個
`file:line` 引用仍然存在（目前 330/330 通過）。19 個分歧點收在 `90-DIVERGENCES.md`。

**這是為了不再重複付同一筆學費**：跳過「survey 每個引擎」在單一 feature 上花掉五個
debug cycle；GDN mixed batching 的 survey 第一次還讀錯模組（讀 vLLM 的 mamba2 而非
GDN 路徑，兩個關鍵決策都不一樣）。

### Benchmark 套件（`44b780bd`, `9c411865`）

`unieinfra/bench/`：`e2e_concurrency.py`（`shared` / `unique` / `rag` 三種形狀）、
`e2e_throughput.py`、`e2e_shapes.py`、`e2e_sla_capacity.py`、kernel 層 harness。
輸出固定欄位：輸入總 token / 輸出總 token / 平均每請求長度 / RPS / TTFT / ITL / TPS。

`9c411865` 把 `fa` 改名成 `flashinfer_fa2`：**FlashAttention 與 FlashInfer 不是同一個東西**，
名字要指出血緣。vLLM 呼叫的 `flash_attn` 套件在這裡是個空 namespace，
所以我們用 flashinfer 包同一批 FA2 kernel 的 wrapper。

### 工程日誌（E48–E56，共 12 個 commit）

包含四筆**撤回**，每一筆都是先發表了錯的數字：

| 日誌 | 撤回了什麼 |
|---|---|
| `aa890b2b` E48b | E45 的 +10.3% **在倉庫裡找不到對應程式碼**，不可重現 |
| `fc09f69c` | sm_80 的 GEMM kernel 名稱**不是**舊指令的證據 |
| `8922ba60` E55b | GDN roofline 估計不可用（算出 159% 的 HBM） |
| `547819e1` | 修正 headline 的**正負號**，並把驗收門檻設在 +10% |

`547819e1` 那一筆最重要：原本報告寫 −0.9% 並稱之為平手，那是**尾段**平均。
兩條 arm 都是極限環，尾段落在哪個相位決定數字 —— 把尾段視窗掃過整個 run，
我們是 234–271、vLLM 是 235–265。**一台 ±7% 的儀器分辨不了 1% 的差異，
而換成完整 post-warmup 平均之後，正負號改變了。**

### 被否定的嘗試

| 嘗試 | 結果 |
|---|---|
| `--mamba-ssm-dtype bfloat16` | 246.4 對 245.8，**同一個數字**。白白降低遞迴狀態精度 |
| 縮小 attention page P（原本的容量計畫） | 會讓 state memory 漲 **13x**（478 MB → 6.2 GB）。檔案在會被改的那行往下 11 行就寫了 |
| LRU 驅動的降級（`5f5635a7`） | 無 throughput 增益 **且輸出不確定**。已回退 |
| 256-token 快照 stride | 連完全重複都壞掉（`AlignMambaCacheSeqlen(track_len) != track_len` → 什麼都沒發佈）。已回退 |

**記錄負面結果的理由**：一個量到的負值和一個勝利一樣值錢，而且重做一次的成本一樣高。

---

## 10. 量測本身學到的（這些是方法，不是結果）

每一條都是在發表過一個錯誤數字之後才加上的：

1. **CUDA event 在 host 比 device 慢時量的是 host，不是 kernel。**
   這翻轉過四個小算子的結論 —— 我們的 kernel 一直都是領先的。
2. **確認參照引擎在它的快速路徑上。** vLLM 在裸 `VllmConfig()` 下跑 eager PyTorch 的
   RMSNorm，量出 13x 的假勝利。
3. **超過 HBM 峰值的 GB/s 代表 L2，不是 kernel 快。** 128 MB L2 裝得下一整個投影權重，
   把 2.41x 變成假的 1.43x。
4. **profiler 的 host 時間是灌水的**（這台機器 3.6x）。要用獨立計時迴圈確認絕對值。
   地板也要校準：一個空的 Triton launch 在這台要 7.7–13.9 µs。
5. **換過 `.so` 或 config 之後的第一次執行是冷 JIT。** 兩次被誤判成正確性回歸。
   規則：至少跑三次。
6. **benchmark 只能找到它的 workload 形狀暴露得出來的缺口。**
   固定 prompt 建一次然後重播，會讓第 2..N 輪全變成 cache 命中 ——
   而第一輪被當成 warm-up 丟掉，於是「冷」形狀量的**全是快取**
   （2840 token 讀到 18.5 ms，真正的冷 prefill 約 130 ms）。
7. **參照引擎的預設值要查，不能假設。** vLLM 在 Qwen3.5/3.6 hybrid 上
   `enable_prefix_caching` 解析成 **False 且沒有任何警告**
   （`Qwen3_5ForConditionalGeneration` 宣告 `IsHybrid` 但沒宣告 `MambaPrefixCaching`）。
   不明確傳這個旗標，就是在跟一個關掉快取的 vLLM 比。

---

## 11. 尚未完成

| 項目 | 狀態 |
|---|---|
| Mixed prefill+decode batching | recurrent 半邊機制在但關閉；設計已確定（切在 layer、prefill-first、傳 offset）|
| MoE | vLLM 在 sm_120 上**載不了** Qwen3.6-35B-A3B，所以沒有 e2e 參照；「輸 1.2–8x」是 kernel 層的舊數字 |
| 共享前綴在快取壓力下的 throughput | 掉 17%（vLLM 0%）。原因已定位：降級挑的是**正在結束的請求的鏈**，不是最冷的節點 |
| w4a16 | Python 半邊完成，CUDA kernel 未建，端到端不可跑 |
| 兩個 feature 仍以環境變數預設關閉 | `TOKENSPEED_ENABLE_FI_PAGED_EXTEND`、`TOKENSPEED_ENABLE_FI_GDN_SM120` |
| `e2e_sla_capacity.py` | 已寫、未跑 |

---

## 12. 對 vLLM 的完整對照（2026-08-10）

Qwen3.5-27B bf16、單張 RTX PRO 6000 Blackwell（GPU 1）、同一支
`unieinfra/bench/e2e_concurrency.py`、`--out-tokens 64 --prefix-words 700 --repeats 3`。
**vLLM 明確傳 `enable_prefix_caching=True`**，並印出 resolved 值確認（見 §10 第 7 條）。
兩邊讀同一份權重檔（config.json md5 相同）。

### shape=`rag` —— 共用文件 + 每次不同的問題

| conc | 輸入總 token | 輸出總 token | in/req | out/req | RPS | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | TPS |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 · **我們** | 8712 | 192 | 2904 | 64 | 0.4 | **93.92** | **99.48** | 37.04 | 37.04 | **26.3** |
| 1 · vLLM | 8712 | 192 | 2904 | 64 | 0.4 | 112.76 | 113.77 | 37.96 | 37.97 | 25.6 |
| 4 · **我們** | 34848 | 768 | 2904 | 64 | 1.4 | **264.64** | **265.96** | 39.77 | 41.39 | **92.3** |
| 4 · vLLM | 34848 | 768 | 2904 | 64 | 1.4 | 358.40 | 362.83 | 39.98 | 41.54 | 88.8 |
| 16 · **我們** | 139968 | 3072 | 2916 | 64 | **4.5** | **738.32** | **742.95** | 44.37 | 49.76 | **289.2** |
| 16 · vLLM | 139968 | 3072 | 2916 | 64 | 3.9 | 1160.97 | 1396.67 | 49.03 | 57.93 | 250.6 |

TTFT **1.20 / 1.35 / 1.57x**，p99 在 conc=16 是 **1.88x**，TPS **+2.7 / +3.9 / +15.4%**。

### shape=`shared` —— 完全重複的 prompt

| conc | 輸入總 token | 輸出總 token | in/req | out/req | RPS | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | TPS |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 · **我們** | 8520 | 192 | 2840 | 64 | 0.4 | **61.84** | **66.96** | 37.04 | 37.04 | **26.7** |
| 1 · vLLM | 8520 | 192 | 2840 | 64 | 0.4 | 94.16 | 94.46 | 37.97 | 37.97 | 25.7 |
| 4 · **我們** | 34080 | 768 | 2840 | 64 | 1.5 | **121.65** | **131.80** | 39.74 | 39.90 | **97.3** |
| 4 · vLLM | 34080 | 768 | 2840 | 64 | 1.4 | 317.38 | 320.93 | 39.97 | 41.30 | 90.1 |
| 16 · **我們** | 136896 | 3072 | 2852 | 64 | **5.0** | **397.93** | **471.25** | 44.31 | 47.74 | **318.0** |
| 16 · vLLM | 136896 | 3072 | 2852 | 64 | 4.1 | 845.45 | 1209.52 | 46.78 | 55.71 | 263.2 |

TTFT **1.52 / 2.61 / 2.12x**，p99 在 conc=16 是 **2.57x**，TPS **+3.9 / +8.0 / +20.8%**。

### shape=`unique` —— 每個請求都是全新 prompt（沒有任何重用可拿）

| conc | 輸入總 token | 輸出總 token | in/req | out/req | RPS | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | TPS |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 · **我們** | 8082 | 192 | 2694 | 64 | 0.4 | **420.49** | **424.34** | 37.01 | 37.03 | **23.2** |
| 1 · vLLM | 8082 | 192 | 2694 | 64 | 0.4 | 462.34 | 462.41 | 37.96 | 37.96 | 22.4 |
| 4 · 我們 | 32328 | 768 | 2694 | 64 | 1.0 | 1623.93 | 1625.91 | 39.97 | 52.11 | 61.8 |
| 4 · **vLLM** | 32328 | 768 | 2694 | 64 | 1.0 | **1217.63** | **1359.96** | 44.01 | 55.07 | **65.7** |
| 16 · 我們 | 129330 | 3072 | 2694 | 64 | 1.7 | 4608.56 | 6441.28 | 74.65 | 132.60 | 109.9 |
| 16 · **vLLM** | 129330 | 3072 | 2694 | 64 | **2.0** | **2685.29** | **5395.67** | 82.16 | 112.42 | **126.5** |

**這一列我們輸。** conc=4 TTFT 慢 33%、conc=16 慢 72%，TPS −5.9% / −13.1%。
conc=1 還小贏 1.10x，所以這不是單請求的計算速度問題，是**併發下的 prefill 排程**問題 ——
和 §11 列的 mixed prefill+decode batching 是同一個未完成項。

### 怎麼讀這三張表

- **兩個引擎的 decode 速度基本相同**（ITL p50 在 conc=1 都是 37 ms 上下，差 2.4%）。
  所有差異都在 **TTFT**，也就是 prefill 與前綴重用。
- **有重用可拿的時候我們明顯較好**，而且併發越高差距越大 —— 這是本輪
  `e1852925` / `46b6614b` / `54c6b607` / `bf156ad4` 四個 commit 的直接結果。
- **沒有重用可拿的時候我們較差**，而且併發越高差距越大。這是目前最大的一個缺口，
  也是唯一一個「vLLM 有、我們沒有」的機制（混合批次）在數字上的樣子。
- 只看 `rag` 或只看 `shared` 會得到「我們全面領先」的錯誤結論；
  只看 `unique` 會得到「我們全面落後」的錯誤結論。**三個形狀都要跑。**

---

## 13. 高併發（32 / 64 / 128）—— 以及它打出來的一個回歸

只有我們這一側，vLLM 尚未在這些併發度重量，**所以這一節沒有對照，不要拿來比較**。

| shape | conc | in tok | out tok | in/req | out/req | RPS | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | TPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rag | 32 | 280896 | 6144 | 2926 | 64 | 6.4 | 1673.20 | 1918.20 | 53.71 | 67.50 | 408.6 |
| rag | 64 | 562752 | 12288 | 2931 | 64 | 8.6 | 2368.68 | 3832.84 | 74.68 | 101.84 | **550.4** |
| rag | 128 | 1129152 | 24576 | 2940 | 64 | 7.8 | 4098.87 | 13393.83 | 87.76 | 196.82 | 496.7 |
| shared | 32 | 274752 | 6144 | 2862 | 64 | 7.2 | 1112.76 | 1322.85 | 53.21 | 63.11 | 463.4 |
| shared | 64 | 550464 | 12288 | 2867 | 64 | 9.3 | 1728.34 | 3258.55 | 74.65 | 96.71 | **593.2** |
| shared | 128 | 1104576 | 24576 | 2876 | 64 | 7.3 | 3883.45 | 13997.27 | 100.67 | 273.90 | 466.6 |
| unique | 32 | 258690 | 6144 | 2695 | 64 | 2.0 | 8259.63 | 12892.75 | 144.58 | 241.54 | 126.4 |
| unique | 64 | — | — | — | — | — | — | — | — | — | **當機（見下）** |

**吞吐的膝點在 conc=64。** 128 的 TPS 反而低於 64（rag 496.7 對 550.4、shared 466.6 對
593.2），而 TTFT p99 從 3.8 秒跳到 13.4 秒。以「TTFT p99 ≤ 1 秒」為 SLA，這台單卡在
27B 上的容量在 conc=16 與 32 之間，不在 128。

### `bf156ad4` 在 conc=64 的 unique 形狀上會讓引擎當掉

```
RuntimeError: SchedulePrefillEvent: failed to allocate Mamba checkpoint slot
```

**歸因（不是推測）**：把 `kMambaSnapshotMaxPerRequest` 改回 1 重新編譯，
同一個工作負載跑得乾淨（conc=64 unique，TTFT p50 14377 ms，TPS 133.6）。
改回 6 就當。所以這是 `bf156ad4` 的回歸。

機制：`unique` 形狀永遠沒有重用，但每個結束的請求仍然留下最多 6 個快照，
而那個 slot 池同時要裝每個在飛序列的 working 與 checkpoint。強制配置沒有退路，
會直接 throw。低併發看不到，因為池子還有餘裕 —— **這是一個只有靠形狀 × 併發度
兩個維度一起掃才會出現的 bug**。

### 三次嘗試才修好，前兩次的「失敗」有一半是量錯的

**第一版：隨需 eviction。** `EnsureMambaCapacityByEvict` 早就存在而且**沒有任何呼叫者**，
看起來正是缺的那塊。實測不行：

```
[Scheduler] Retracted request ... lost tree-owned Mamba state, aborting request
```

被退回（retract）的請求，遞迴狀態掛在樹上，而 `applyRetract` 有一行註解講得很明白：

```cpp
// Once retracted, the recoverable Mamba state is tree-owned and evictable: keep no request-local slots.
```

**那個狀態被刻意標成可驅逐**，於是 LRU 選走它，請求被整個中止。

**第二版：保留閘門**（額外快照只從池子餘裕拿）。沒有修好 conc=64，也就沒有意義。

**第三版（採用）：把遞迴狀態的鎖和 KV 的 refcount 分開。**

四家怎麼做，一讀就有答案：

| 引擎 | 遞迴狀態放哪 | 被退回時誰保護它 |
|---|---|---|
| **SGLang** | 獨立 mamba pool + radix 節點 | **`mamba_lock_ref`，與 `full_lock_ref` 分開**（`mamba_radix_cache.py:84-90`）。KV 鎖整條路徑到 root，mamba 只鎖**那一個節點** —— 遞迴狀態不可切片，一次匹配只從某一個快照接續 |
| **vLLM** | **和 KV 同一個 block pool** | 同一套 refcount 與驅逐。排不進去就回傳 `num_gpu_blocks + 1`（`single_type_kv_cache_manager.py:1206`），**由 scheduler 不排它，而不是稍後 throw** |
| **TRT-LLM** | 雙緩衝 + replay 重算 | 不走前綴樹 |
| **我們（改前）** | 獨立 mamba pool | **只有 KV 的 refcount**，而被退回的請求正好沒有 KV 參考 |

缺的就是 SGLang 那半：`TreeNode::mamba_lock_ref_`，加上一個 RAII `MambaNodeLock`
由 `Retracting`/`Retracted` 持有 —— 用 guard 而不是在每個轉移點記得解鎖，
resume 與 abort 都自動釋放。鎖建立之後隨需 eviction 才安全，兩者一起才是修法。

### 驗收（乾淨 build，`--concurrency 1,4,16,64`）

| shape | conc | 改前 TTFT p50 | **改後** | 改前 TPS | **改後** |
|---|---|---|---|---|---|
| unique | 64 | **當機** | **14410.27 ms** | — | **131.7** |
| rag | 64 | 2368.68 | **2235.04** | 550.4 | **575.9** |
| shared | 64 | 1728.34 | **1469.12** | 593.2 | **645.5** |
| rag | 1 / 4 / 16 | 93.59 / 264.33 / 736.63 | 93.74 / 264.14 / 736.24 | 26.3 / 92.4 / 289.2 | 26.3 / 92.4 / 289.4 |

**當機修好，conc=64 還快了（rag +4.6%、shared +8.8% TPS），低併發完全不動。**

C++ 測試已寫（`test_mamba_eviction.cpp` 三個案例：鎖住的節點不被驅逐、不計入可驅逐數、
move 不會重複釋放），但**這個容器沒有 git，googletest 抓不下來，所以沒有在這裡執行過**。
Python 端 `test/runtime/execution` 與 `test/runtime/layers` 144 passed / 31 skipped。

### 前兩版的「退步」有一半是我量錯的

驗證第一、二版時我用 `--concurrency 16,64`，省掉了 1 和 4。**`rag` 形狀在同一個 process 裡
跨 concurrency 共用同一份文件**，所以掃 `1,4,16` 時 conc=16 已經被前兩輪暖過，
單獨從 16 開始則沒有 —— 736 對 907，23% 的差距全部來自這個順序。

我拿 907 去對 736，於是「保留閘門退步 23%」「lock+evict 掉 29% TPS」兩個結論都是假的。
後來甚至一度據此判定「HEAD 無法重現 736」，還建了 flat 與 radix 兩種乾淨 build 去找一個
不存在的 binary 差異。**用同樣參數重跑基準線就會發現，而我一次都沒做。**

規則：**改動前後必須用同一組 harness 參數**；在一個會累積快取的形狀上，
concurrency 清單本身就是參數的一部分。

---

## 附錄 A — 完整 commit 清單（時間由舊到新）

`—` 表示這個 commit 沒有可量測的效能效果，不是漏填。

| # | commit | 做了什麼 | 效果 |
|---|---|---|---|
| 1 | `1dd493da` | 以共同祖先重新 fork upstream tokenspeed | rebase 從不可能變成一行指令 |
| 2 | `ab27cd9b` | 把 fork 分歧分類成 upstream PR 與 plugin | — |
| 3 | `57312157` | proton import 改 optional | — |
| 4 | `b93a8a8e` | Triton MHA prefill 跳過越界 query block | mixed batch 2.04–2.27x；均勻批次不變 |
| 5 | `d97efd1c` | engine 啟動時載入 kernel plugin | plugin 系統從完全無效變有效 |
| 6 | `1fa7d5a4` | 允許修改 upstream 檔案並寫明維持條件 | — |
| 7 | `423338aa` | 修正 layout 表的錯誤敘述 | — |
| 8 | `228886f0` | 回報真實 shared memory 預算 | 每張 CUDA 卡都回報 0 → 101376 |
| 9 | `73b27362` | prefill tile 依實測 shared memory 決定 | `OutOfResources` 消失 |
| 10 | `48a80ebb` | 擋掉兩個必定失敗的 FlashInfer kernel | 啟動不再 crash |
| 11 | `1bdc361e` | 日誌 E48：re-fork 後可跑，落後 vLLM 16.2% | — |
| 12 | `4c1ad83a` | attention page P 與 state stride N 分離 | 讓 `--mamba-track-interval` 觸及幾何 |
| 13 | `aa890b2b` | 日誌 E48b：**撤回** E45 的 +10.3% | 該程式碼在倉庫中不存在 |
| 14 | `3ebb577c` | 日誌 E49：w4a16 未校準 ppl +7.2% | 負面結果 |
| 15 | `36e416a4` | 修正 w4a16 patch 的未驗證宣稱 | — |
| 16 | `c808eee3` | 記錄 sm_120 進度與未完成項 | — |
| 17 | `6a913132` | pre-commit hook 排除凍結的 legacy 封存 | — |
| 18 | `7f7e22c3` | 對 plugin 套用格式化 | — |
| 19 | `86d1c7d1` | 重排一個未格式化的 dict comprehension | — |
| 20 | `4119f79c` | cutlass fused-MoE routing scale 保持 float32 | self-routing 在 bf16/fp16 不再失敗 |
| 21 | `3eb83e9d` | dense Qwen3 略過它沒有的 KV-cache scale | NVFP4/FP8-KV checkpoint 不再 KeyError |
| 22 | `b12febcc` | 讓環境變數指定 smg exporter port | — |
| 23 | `0abe5e65` | 讓 upstreaming Status 表描述真實存在的分支 | — |
| 24 | `996b05f4` | Triton MHA kernel 接受 fp8 KV cache | **+19.0% throughput**，容量 1.92x |
| 25 | `b8a392f8` | fp8-KV 路徑 query 保持 activation 精度 | 同上（配套） |
| 26 | `add29d74` | 記錄 fp8 KV 實際是什麼、量到什麼 | — |
| 27 | `7b195f10` | 列出 fp8 KV 分支並標出非獨立的那個 | — |
| 28 | `2e18a3e4` | 記錄 fp8 KV +19.0% 與 harness 的誤報方式 | — |
| 29 | `d4e075ce` | 記錄 w4a16 沒有可重用路徑，是死碼 | 能力缺口 |
| 30 | `64e632d8` | MHA backend 能服務混合批次 | full-attention 半邊完成 |
| 31 | `c81fa7fa` | `--enable-mixed-batch` 改成啟動時拒絕 | crash → 明確拒絕 |
| 32 | `128b4ac2` | 日誌 E52：容量計畫方向相反、bf16 SSM 無效 | 三個負面結果 |
| 33 | `8ecb4b24` | 切分 GDN state bounds | 中途嘗試 |
| 34 | `69a5f7c5` | 切分 GDN query lengths | 中途嘗試 |
| 35 | `932c4661` | 日誌 E53：需要 forward split，不是更多 index 修補 | — |
| 36 | `146c4175` | recurrent 路徑切分混合批次（**關閉**） | 機制在，放錯層 |
| 37 | `3aceeab8` | 日誌 E54：vLLM 與 TRT-LLM 都切在 mixer | — |
| 38 | `83dc7af3` | 日誌 E54b：四家都切；TurboMind 用 offset | 順序是選擇不是需求 |
| 39 | `4bd9bebf` | 把優化流程寫進共享的版本控管檔案 | — |
| 40 | `7b0aa0ee` | sm_120 狀態報告 | — |
| 41 | `547819e1` | **修正 headline 的正負號**，門檻設在 +10% | −0.9% → +1.8%（尾段平均相位偏誤） |
| 42 | `cf63637e` | 日誌 E55：decode profile，三個「缺的」融合有兩個已存在 | — |
| 43 | `fc09f69c` | **撤回** E55：kernel 名稱不是舊指令的證據 | — |
| 44 | `8922ba60` | 日誌 E55b：**撤回** roofline 估計（算出 159% HBM） | — |
| 45 | `e1852925` | 快照發佈到匹配搆得到的深度 | 27B 重複 prompt 重用 0 → 可用 |
| 46 | `4d35899c` | 用探測取代版本旗標找 GDN kernel | 解開 4 道閘門 |
| 47 | `94d0fa12` | libstdc++ shim 上 `LD_LIBRARY_PATH` | 4 個 kernel 從 raise 變可用；RoPE 贏 vLLM 1.70x |
| 48 | `936b835f` | RoPE wrapper 砍 3.2 µs Python | 10.51 → **7.31 µs**（vLLM 7.77） |
| 49 | `edd295cf` | Qwen3 改用融合 RoPE+KV write | 每層少一 launch，**端到端無變化** |
| 50 | `da0619da` | 不再每次呼叫重推 dispatch 狀態 | wrapper 24.36 → 14.93 µs，**端到端 0** |
| 51 | `de851d8d` | FA2 paged extend，每 forward 只 plan 一次 | 對 Triton 1.86–4.57x；RAG 15.2x |
| 52 | `9e08f8ce` | 不為已結束的請求再跑一步 | 8B TTFT 29.96 → **15.85 ms** |
| 53 | `bdbb3571` | 啟用融合 sm_120 GDN chunk-prefill kernel | 48 層 17.9 → **11.4 ms**，比 vLLM 快 1.65x |
| 54 | `a90de320` | 為 bf16 專家註冊 ragged Triton matmul | 對 cutlass 2.1–2.6x；仍落後 vLLM 1.2–8x |
| 55 | `a1483ec5` | 補回 w4a16 Marlin 的 Python 半邊 | 30 測試通過；端到端仍不可跑 |
| 56 | `18f1fa92` | 五引擎橫向對照 + sm_120 kernel 計分板 | 19 個分歧點 |
| 57 | `46b6614b` | 寫回主機的 KV 留在裝置上 | 部分命中 48.76 → **36.41 ms** |
| 58 | `55004cc2` | 記錄 KV 降級發現與冷 JIT 量測陷阱 | — |
| 59 | `2d0fe6cd` | 對兩個新檔案套用格式化 | — |
| 60 | `54c6b607` | KV 分頁 64 → 16 token | TTFT **−13~14%** |
| 61 | `8def103b` | 共享吞吐缺口是降級挑錯節點 | 定位，未修 |
| 62 | `5f5635a7` | 記錄失敗的 LRU 降級嘗試 | 無增益且輸出不確定，已回退 |
| 63 | `44b780bd` | 可執行的 benchmark 套件與併發曲線 | — |
| 64 | `9c411865` | 回報完整請求形狀；FA2 依血緣改名 | — |
| 65 | `29f8eda0` | FA2 workspace 128 → 512 MB | 27B 在 conc≥4 不再 Buffer overflow |
| 66 | `f7d6015f` | 讀完四家後的 hybrid 前綴重用設計 | — |
| 67 | `bf156ad4` | 保留多個遞迴快照 | **RAG 437 → 89.7 ms，重用 0 → 2560** |
| 68 | `4e20e1b8` | vLLM 前綴快取旗標 + 更正 + 本報告 | 揭露先前所有 hybrid 對照皆無效 |
