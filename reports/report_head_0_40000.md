2026-11-14



［目錄］

## 摘要

本研究在 4 張 AMD MI350P（gfx950，GPU 間僅 PCIe、無 xGMI）上以 TP4 服務 DeepSeek-V4.1-Flash，工作負載為 AgentX 的 coding-agent 多輪軌跡（prompt 中位數 83K tokens、輸出中位數 235 tokens）。我們先以 GPU-only trace 與逐 kernel 對照找出瓶頸，接著提出並實作三項方法：（1）依 DSv4.1 注意力結構分組的 KV cache 佈局，每 token 佔用約 940 B，相對 vLLM 的約 2,090 B；（2）DSpark 投機解碼的整塊驗證（block verification），配合分塊計算、批次門檻與 TP 分片計算（shard-local），使其在 TP4 下可行；（3）prefill 的兩 micro-batch 通訊／計算重疊，並指出 ROCm 硬體佇列分配使未設 priority 的重疊完全失效。另外重寫了 prefill MoE 的 MXFP4 GEMM（M=4096 每層 −8%、M=8192 −30%）。

在雙方各取最佳設定、同為 T=1.0 取樣與無量化通訊的條件下，11/16 最終 build（mi350p-final）的每分鐘完成請求數（RPM）為 vLLM 的 1.67×（c1）、1.19×（c4）、1.28×（c16）、1.11×（c32）、1.22×（c64）與 5.88×（c128），6 格 0 錯誤、無 OUT_OF_RESOURCES；needle 6/6、GSM8K（T=0）90.45%。我們同時報告各方法的理論上限與實測差距、二十餘項未帶來端到端增益的嘗試，以及一項重要的量測教訓：ROCm 的 torch profiler 會改變 HIP graph 的提交方式，使「主機端阻塞」看起來遠比實際嚴重（第 10.2 節）。

（初版 2026-11-14；2026-11-16 更新：新增 6.8、8.1、10.2、10.3 節與附錄 B、C，並更新第 1、3.3、8–12 節與附錄 A。）

## 1. 研究動機與貢獻

長上下文 agent 工作負載有三個特性：prompt 極長但輸出短、多輪對話之間前綴高度重疊（單一請求 98% 的 prompt 可由上一輪命中）、並且大量請求同時在線。這使得推論效能取決於三件事：KV cache 能留住多少前綴、decode 每一步能產出多少 token，以及 prefill 的成本。在沒有 xGMI 的 PCIe 機器上，TP 所需的 all-reduce 又進一步放大了 prefill 與 decode 的通訊成本。

本報告的貢獻如下：

1. 一套可重現的比較方法（壓力模式、best-vs-best、重複交錯量測），並說明哪些常見設定會讓結論失真（第 3 節）。
2. 以 GPU-only trace 拆解 AgentX 負載下的時間組成，定位 all-reduce（34%）與 MoE（24%）為主要成本（第 4 節）。
3. 三項方法的設計、實作細節、理論上限與實測差距分析（第 5–7 節），以及 gfx950 kernel 的配置調整與 prefill MoE GEMM 重寫（第 8 節）。
4. 負面結果的整理，包含失敗原因；並指出 ROCm 上連 GPU-only profiler 都會改變 HIP graph 提交方式，主機端結論必須改用無 profiler 計時（第 10 節）。
5. 把整套流程整理成可重現的硬體適配 skill（附錄 C）。

## 2. 背景

### 2.1 硬體

![](/api/attachments/4edf8fdf-b394-43aa-96a7-39db519581e0)

每張 MI350P 有 128 個 CU（4 個 XCD）與 144 GB HBM。同屬 gfx950 的 MI355X 有 256 CU／8 XCD，許多現成 gfx950 kernel 的 grid 與 XCD swizzle 依 MI355X 寫死，這是第 8 節的出發點。

### 2.2 模型：DeepSeek-V4.1-Flash

- MoE：384 個 routed expert、top-6，權重 MXFP4；另有 shared expert。
- 注意力：CSA 稀疏注意力。indexer 以 MXFP4 key 對歷史打分、選出 top-k 位置（512），再由 selected attention 只對這些位置計算。另有 sliding window（128）分支與壓縮器：全域 KV 每數個 token 壓縮成一筆。
- mHC：層間的 multi-head connection，於每次 all-reduce 之後做 mix／post 運算。
- TP4：每層兩次 all-reduce（attention 輸出投影後、MoE 後），每次的資料量為 rows × 5,120 × 2 B。

### 2.3 DSpark 投機解碼

![](/api/attachments/e1d7f554-a527-4121-8535-a4e1d7ec5eb1)

目標模型一次 forward 的成本幾乎與列數無關（受權重頻寬限制），所以 decode 吞吐量 ≈ 每步接受 token 數 ÷ 每步時間。AgentX 請求的 temperature 為 1.0，驗證必須是取樣式（rejection sampling 類），輸出分布須等於目標模型。

### 2.4 服務架構

![](/api/attachments/5dd05903-294b-404f-9499-f969e786cc3f)

## 3. 實驗方法

### 3.1 工作負載

AgentX（agentx-harness 1.0.13 + AiPerf，資料集 semianalysis_cc_traces_weka_062126，seed 42）以 OpenAI chat completions 串流重放 coding-agent 軌跡。每個 concurrency lane 依序重放一段對話；後一輪 prompt = 前一輪 prompt + 回覆 + 新內容。



**表 1 c32 壓力模式 600 s 量測期間的請求統計（1,162 個請求，不含 warmup）。**

| 欄位 | p10 | p50 | p90 | 最大 | 平均 |
|---|---|---|---|---|---|
| prompt（tokens） | 29,819 | 83,175 | 217,979 | 578,948 | 112,432 |
| 輸出（tokens） | 94 | 235 | 2,059 | 14,177 | 765 |

- 106 段對話，每段中位數 8.5 輪（最多 62 輪）；主 agent 約 56%、子 agent（agent_depth = 1）約 41%。
- 每請求 prompt 的可命中比例中位數 98%；AiPerf 計算的理論前綴命中率約 95%。實際需重算的 prefill 每請求 4–8K tokens。

### 3.2 兩種跑法

**官方模式**依軌跡記錄的思考時間送下一輪（中位 1.9 s、平均 14–22 s、上限 300 s），≥ 900 s。量測發現 c ≤ 64 時 client 有 34–63% 時間在等思考時間，c16 名義 16 個並行實際只有 4–6 個在途，RPM 主要由軌跡決定而非引擎。**壓力模式**（--ignore-trace-delays）前一輪結束立即送下一輪，600 s，屬封閉迴路（closed loop）：RPM ≈ c ÷ 平均請求延遲。本報告以壓力模式衡量引擎容量。

### 3.3 公平性控制

- 取樣：UnieInfra 一律 --sampling-backend triton_full。AMD 預設的 greedy 會忽略 temperature，接受長度虛高（3.1–4.0 vs 2.8）(曾使 c4–c64 的比值從 0.82–0.91× 被誤估為 1.0–1.2×)。
- running-request 上限：vLLM recipe 限制在 CONC/4。雙方都跑 CONC/4 與 2×CONC 兩種上限，取 RPM 較佳者（best-vs-best）。
- 記憶體：vLLM GPU memory utilization 0.9；UnieInfra 最終全部並行度皆為 0.86（原因見 5.5 節），仍比 vLLM 少用記憶體。
- 通訊：不使用量化通訊（INT8 all-reduce 雖可再增益，但 vLLM 未使用）。
- 雜訊：單次 600 s 格的 run-to-run 差異可達 ±10–15%，小於 15% 的差異至少跑兩次、以 A B B A 交錯。11/15 另以同一 build、只切旗標的 OFF/ON/ON/OFF 測得：同一設定的 c4 ITL 在 7.9–9.9 ms、RPM 在 ±10% 之間依先後次序漂移（第 10.2 節），所以端到端 A/B 只用來擋明顯退化；有獨立量測證實收益且正確的改動保留並預設開啟。
- 剖析：端到端判斷只用 GPU-only trace；開啟 Python 追蹤（with_stack）的 torch profiler 會把 Python 時間放大 35–40 倍（第 10 節）。主機／launch 問題改用無 profiler 的 TOKENSPEED_GPU_STEP_TIMING=1（以 event 夾住每個 forward），因為 profiler 本身會改變 HIP graph 的提交方式（第 10.2 節）。

## 4. 基準剖析：時間花在哪裡

在 c32 壓力負載下錄製 6 秒 GPU-only trace（加入 prefill 重疊之前），GPU 忙碌 89%（5,416 / 6,061 ms）。以 kernel 名稱分類如圖：

![](/api/attachments/998671f8-3f6c-4604-ae4f-637364493886)

另一個對照是純 decode：32 個使用者共享 130K 前綴，雙方相同設定。去除偶發 all-reduce 等待後，每步中位數 UnieInfra 45.1 ms、vLLM 40.0 ms；但 GPU 忙碌時間幾乎相同（39.9 vs 39.4 ms），差距來自 GPU 閒置（5.3 vs 0.55 ms）。逐 kernel 分類後，all-reduce 雙方持平（11.96 vs 11.95 ms），MoE stage1 我們多 0.8 ms，attention 多 0.9 ms，mHC 我們少 1.2 ms。這說明 decode 的 kernel 總量已與 vLLM 相當；剩餘差距在主機端與少數 kernel。

由此得出三個可攻擊的方向：（a）高並行度時 KV 容量決定前綴命中率；（b）每步接受 token 數直接決定 decode 吞吐；（c）prefill 大小的 all-reduce 佔牆鐘時間 21%，而且受 PCIe 頻寬限制、期間計算單元閒置。

## 5. 方法一：依注意力結構分組的 KV cache

### 5.1 問題

c128 時，128 段平均 112K tokens 的對話同時在線，工作集約 1,440 萬 tokens（尚未計入輸出與子 agent 分支）。vLLM 在 0.9 記憶體下的 KV 容量為 1,456–1,855 萬 tokens（依 running-request 上限而異，CUDA graph 佔用越多容量越小），實測前綴命中率只有 23.5–70.6%，遠低於理論的 95%，每輪都要大量重算 prefill（c128 RPM 僅 13.8–31.1）。

### 5.2 分析

DSv4.1 的每層 KV 並非「每 token 一筆」：全域層每數個 token 才存一筆壓縮後的 KV，sliding-window 層每請求只需保留最近 128 個位置，壓縮器只需保留未湊滿一個壓縮單位的尾巴。若以「每 token 固定大小」配置，就會為不需要的狀態保留空間。

### 5.3 設計與實作

![](/api/attachments/7eee1412-8d2f-4299-aeb0-572e86a9ebae)

- cache group：每一種狀態（global_r1、global_r2、compressor_tail_r2、swa）是一個 cache group，有自己的區塊粒度。server log 的 Cache profile 為 global_r1 140、global_r2 45、compressor_tail_r2 8、swa 2（每 parent 的 slot 數）。
- LCM arena：所有 group 放在同一塊 arena，以各 group 區塊大小的最小公倍數（LCM block，此設定為 256 tokens／3.2 MB）為單位分配。一個請求的所有 group 以同一組 block 分配、做前綴比對、在 retraction 時一起搬移、一起釋放，所以前綴命中時所有層的狀態同時可用。
- 壓縮器尾巴：ROCm 上每頁從 2 列改為 32 列（對應的尾巴 kernel 以 ROWS 為 constexpr），13 萬 token 請求的 block table 從約 69K 欄縮到 4.3K 欄，降低每步 block table 打包成本。
- 容量由預算決定：原本 MAX_TOTAL_TOKENS 預設 16.7M，KV 受 token 上限而非記憶體限制；改為依記憶體預算（MAX_TOTAL_TOKENS = 2^30）後，0.9 時為 3,860–4,410 萬 tokens。
- cache-aware admission：等待中的請求依「裝置上已快取的前綴長度」排序，最多被超車 64 次，讓能大量命中的請求先進來，c128 命中率 0% → 65%（當時容量較小時）。

### 5.4 效果

**表 2 c128 壓力模式的 KV 容量與前綴命中率。**

| 設定 | 每 GPU token 容量 | 前綴命中 | RPM |
|---|---|---|---|
| vLLM recipe cap（32） | 1,855 萬 | 63.7% | 23.2 |
| vLLM 2× cap（256） | 1,456 萬 | 23.5% | 13.8 |
| UnieInfra，上限 33.5M | 3,355 萬 | 約 85% | 123.5 |
| UnieInfra，依預算，gmu 0.9 | 3,860 萬 | 94.2% | 132.3 |
| UnieInfra，依預算，gmu 0.86（最終） | 約 3,300 萬 | 93% | 127.1 / 131.1 |

### 5.5 理論與實際差距

- 命中率：UnieInfra 實測 93–94%，與 AiPerf 理論上限 95% 只差 1–2 個百分點，主要是 256-token 的比對粒度與每輪新尾巴。vLLM 雖然容量也大於工作集的粗估，命中率卻遠低於理論；其原因（逐出策略、頁面補齊、輸出與分支佔用）我們未讀其原始碼確認。
- 容量上限：KV 吃滿 0.9 的預算時，長 chunked prefill 偶爾使 GPU runtime 在佇列中申請 scratch 記憶體失敗（HSA_STATUS_ERROR_OUT_OF_RESOURCES，abort 訊息皆為「Available Free mem: 0 MB」）。94 次啟動中崩潰 5 次；固定 33.5M 上限的 217 次為 0 次。另發現啟動腳本以不帶值的 -e HSA_NO_SCRATCH_RECLAIM 傳遞環境變數，使 docker 刪除了映像內建的 =1；修正後在 0.9 仍會崩潰，故最終以 0.86 保留約 6 GB 餘裕。
- 適用範圍：c16–c64 時雙方容量都足以容納工作集（命中率皆 94–96%），此方法幾乎沒有貢獻。

## 6. 方法二：DSpark 的 coupled drafting 與整塊驗證

### 6.1 問題

改用取樣式驗證後，每步接受長度只有約 2.8（vLLM 約 3.3–3.5），c4–c64 落後 vLLM 9–18%。逐 token 驗證以 min(1, P/Q) 決定每個草稿 token，一旦拒絕就停止，後面的 token 即使目標模型很滿意也被丟棄。

### 6.2 Coupled drafting（先行步驟）

草稿第 s 步以 Gumbel-max 對 (base + bias)/T 取樣，使用與驗證器第 s+1 列相同的 Philox 噪聲（offset = start_pos + 2 + s）。驗證器維持「目標取樣後精確比對」，所以輸出就是目標模型的樣本；當兩分布接近時，共用噪聲使兩邊抽到同一 token 的機率趨近 1 − TV(P, Q)。實作在 DSpark block kernel 的 SAMPLE 分支，以單元測試驗證草稿噪聲與驗證池逐位元相同。同 build、同 chunk 大小下：c16 +11.6%、c32 +10.6%、c4 +2.4%、c64 +2.9%；接受長度 2.8 → 3.1–3.2。

### 6.3 整塊驗證的原理

Block verification（Sun et al., 2024）把整串草稿聯合判定：

```
p_0 = 1
p_i = min(1, p_(i−1) · P(x_i) / Q(x_i))
R_i = Σ_x max(p_i · P(x | x_≤i) − Q(x | x_≤i), 0)
h_i = R_i / (R_i + 1 − p_i)      （最後一個位置 h_n = p_n）
取 u_i ~ U(0,1)，接受最大的 i 使 u_i ≤ h_i
下一個 token ~ normalize(max(p_i · P − Q, 0))；整塊接受時 ~ P
```

p_i 是「累積信用」：P/Q > 1 的 token 會把信用往後帶，抵補先前 P/Q < 1 的 token；R_i 是在第 i 個位置停下時可用於重新取樣的殘差質量，h_i 以此分配「停」與「繼續」的機率，使輸出序列的分布等於目標模型自行取樣。論文證明此法 lossless，且期望接受長度不小於逐 token 驗證。

![](/api/attachments/613f662e-b2b6-4adf-a8bf-6302e3184eaf)

### 6.4 在 TP4 下實作的困難

殘差 R_i 需要對整個詞表（129,280）加總，所以需要草稿模型在每個位置的完整分布 Q。但 TP4 下 lm_head 是 vocab-parallel，每張 GPU 只有 32,320 個 logits，且草稿 logits 在草稿 forward 結束後即不存在。

![](/api/attachments/7af3124c-9cb2-445f-a34c-36f3b065601a)

### 6.5 實作

1. 保存草稿分布（execution/drafter/deepseek_v41_dspark.py）：在 bind_sampling_backend 時建立 DSparkBlockVerifyState，含 pool[slots, 5, local_vocab] FP32。草稿每一步把本 GPU 的 base logits 以 index_copy_ 寫入該請求的槽位。256 個請求時約 0.16 GB。
2. 重建完整 Q（sampling/block_verify.py 的 draft_logits）：驗證時對每列重建 base + Markov bigram bias（bias = embedding[prev] @ projection_local^T），把 padding 欄設為 −inf，再以 all_gather_single 收齊 4 張 GPU，得到 [bs, 5, 129,280]。
3. 向量化驗證（block_verify_chain）：對目標與草稿 logits 做 log_softmax（除以 temperature），由 P/Q 比值的累積和算 p（log 域、上限 0），由 p·P − Q 的正部算 R、h；以 u ≤ h 的最大索引得接受長度，從殘差以 Gumbel-max 取下一個 token；輸出寫成與原 verify_chain_target_sampled 相同的 predict / accept_index / accept_length 格式，下游不需修改。
4. 分塊（block_verify_chain_chunked）：每 16 個請求一塊處理。full-vocab FP32 暫存約 6 × bs × 5 × 129K × 4 B，會被 HIP graph 的記憶體池保留並從 KV 預算扣除；分塊後暫存約 41 MB。CPU 測試確認 chunk = batch 時與不分塊逐位元相同。
5. 批次門檻（sampling/backends/triton_full.py）：一步超過 128 個請求時改用原驗證。256 個請求時 all-gather 約 660 MB／步，decode ITL p90 退化到 2.3 s。graph 依批次大小分別捕捉，所以門檻在捕捉時決定路徑，執行時無額外成本。
6. TP group 綁定：wire_target 早於 bind_sampling_backend 執行，原實作在建立 state 時拿不到 TP group，warmup 時 all-gather 失敗；改為在 drafter 宣告 tp_group 並於建立 state 時傳入。

開關：TOKENSPEED_DSPARK_BLOCK_VERIFY=1（另 _MAX_BS 預設 128、_CHUNK 預設 16）。

### 6.6 效果

**表 3 只差整塊驗證開關的 AgentX 壓力測試（同一套 kernel；c16–c64 為 3 次平均）。**

| 並行度 | 關閉 RPM | 開啟 RPM | 變化 |
|---|---|---|---|
| c1 | 15.9 | 18.9 | +19% |
| c4 | 37.2 | 38.4 | +3% |
| c16 | 63.8 | 69.0 | +8% |
| c32 | 106.9 | 113.0 | +6% |
| c64 | 137.8 | 138.4 | +0% |
| c128 | 132.3 | 127.0 | -4% |

正確性：GSM8K 500 題（T=1.0）開 95.8%、關 96.2%，差距在 ±0.9 pp 的抽樣誤差內；接受長度 2.1–3.7 → 3.3–3.9。c1 的 ITL p90 從 9.4 降至 6.1 ms（vLLM 7.5 ms）。

### 6.7 理論與實際差距

- 上限：若完全 decode-bound，吞吐增益 ≈ 接受長度增幅（約 +10–30%）。c1 的 +19% 落在此範圍。
- c4–c64 只有 +2–8%，原因有三：
  - 壓力負載下約 25–35% 的 GPU 時間是 prefill，不受影響；
  - 整塊驗證本身的成本：每請求每步約 2.6 MB 的 all-gather 走 PCIe，加上 full-vocab softmax；
  - 批次越大，每步 all-gather 越大，攤提效益越低。
- c128 −4%：多數步驟超過 128 個請求而走原驗證，屬雜訊範圍。

### 6.8 Shard-local 整塊驗證（11/14）

6.5 的做法每個 chunk 要把 [16, 5, 129,280] 的 FP32 草稿 logits all-gather 到每張 GPU（約 41 MB／chunk）。c32 GPU-only trace 中這是 508 次 × 0.35 ms 的 RCCL 呼叫，暴露時間佔 3%。

改法（sampling/block_verify.py）：softmax 正規化與殘差 R_i 都是對詞表的加總，可以拆成各分片的部分和。每張 GPU 只在自己的 32,320 個詞上計算，跨卡只交換兩次小 all-gather：

1. 每個位置：本分片草稿 logits 的 log-sum-exp，以及草稿 token 落在本分片時的 logit（[bs, 5, 2]）。合併後得到全詞表的 log Q(x_i)，進而算出 p_i。
2. 每個可能的停止位置：本分片的殘差質量 Σ max(p·P − Q, 0)，以及本分片從殘差與從 P 各自的 Gumbel-max 候選（值與 token id）。下一個 token 對每種接受長度都預先選好，所以第二次交換就已帶回勝者，不需要第三次通訊。

Gumbel 噪聲對全詞表抽取後再切片，各分片不會重用同一份噪聲；輸出分布與收齊全部 logits 的版本相同。通訊量從每 chunk 數十 MB 降到每請求數十個浮點數，chunk 從 16 放大到 64。

- 正確性：CPU 統計測試與 6.5 版本的輸出分布 TV 距離 0.03–0.04，等於同樣本數的雜訊底。
- 效果（2 次，TBO 開啟）：c32 122.5 / 123.5 RPM（未分片 122.8 / 121.6）、c64 154.2 / 150.9（148.9 / 153.6），接受長度 3.6–4.2。端到端在雜訊內，但移除了大批次時 all-gather 隨 bs 線性成長的成本，最終 build 採用。
- 開關：TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1、TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64。

## 7. 方法三：prefill 的兩 micro-batch 通訊重疊

### 7.1 問題

prefill 以 8,192 tokens 為一塊，每層兩次 all-reduce、每次 8,192 × 5,120 × 2 B ≈ 84 MB。PCIe 上中位數 1.67 ms，期間計算單元閒置。c32 trace 中這類 all-reduce 共 699 次、1,295 ms，佔牆鐘時間 21%。

### 7.2 理論上限

以合成層 microbenchmark 量測，序列執行（fused Iris all-reduce + mHC post，84 programs）每層 10.57 ms。若兩次 all-reduce（約 3.4 ms）完全被計算覆蓋，下限約 7.2 ms（−32%）。在端到端層級，若 21% 的牆鐘時間能被覆蓋 87%（實測覆蓋率，見 7.5），上限約為 −18% 牆鐘時間，即 c32 約 +22% RPM。

### 7.3 設計

![](/api/attachments/4aebcf93-bef3-46e1-bd2a-25dddd91705f)

- 適用條件：純 EXTEND 步驟且 ≥ 4,096 rows（TOKENSPEED_PREFILL_TBO_MIN_ROWS）；只對 encoder 層（每個 token 都要計算的前段層）。decoder 層只算每請求尾端 rows（narrowing），資料量小，不切。
- 切點：優先選靠近中間的請求邊界；只有一個請求時，切在請求內絕對位置 128 對齊處，並避開 replay rows。
- 因果：被切開的請求，B 半視為同一請求的下一個 chunk。同一層內 A 先完成 attention 並把 sliding-window、壓縮全域 KV、壓縮器尾巴寫入 cache，B 的 attention 再讀取；與一般 chunked prefill 等價。
- 流水：每層 compute stream 依序 attn(A) → attn(B) → MoE(A) → MoE(B)；side stream 在 attn(B) 期間 reduce A 的 attention 輸出、MoE(A) 期間 reduce B 的、MoE(B) 期間 reduce A 的 MoE 輸出，B 的 MoE 輸出則與下一層 attn(A) 重疊。兩個 stream 以 event 同步，每個 micro-batch 重用自己的輸出緩衝。

### 7.4 實作

1. metadata（layers/attention/backends/specific/deepseek_v41.py）：新增 plan_prefill_micro_batches(align)，一次準備 A、B 與整體三份 forward metadata（V41ForwardState），執行時以 use_forward_state() 切換；KV 寫入位置、block table、壓縮器狀態都依各自的 rows 計算。
2. 模型（models/deepseek_v41.py）：_plan_encoder_overlap 判斷是否啟用並取得兩條通訊 lane；_encoder_layers_overlapped 實作上述流水；每層拆成 overlap_attention、overlap_ffn、overlap_finish 三個階段。
3. 通訊（distributed/comm_backend/iris.py、comm_ops.py 的 overlap_all_reduce_lanes）：每個 micro-batch 一條 Iris producer lane，於 Iris heap 預留（約多 320 MB，僅在開啟時）；pull kernel 上限 32 programs；mHC post 從 all-reduce 拆出為獨立 kernel（低 program 數時 fused 版較慢）。
4. stream priority：side stream 以 torch.cuda.Stream(priority=−1) 建立。

![](/api/attachments/1184a42c-51ff-46a7-8c5b-18016fdb9747)

第一版未設 priority，trace 顯示重疊時間為 0 ms、prefill 無改善。設為 −1 後，2,574 ms 的 all-reduce 有 2,236 ms（87%）被覆蓋。先前兩次 row-chunked FFN 重疊嘗試在單一長 prompt 上反而慢 5–15%，很可能也是同一原因。開關：TOKENSPEED_PREFILL_TBO=1（_IRIS_PROGRAMS 預設 32）。

### 7.5 效果

**表 4 prefill 重疊開關的量測（AgentX 各 2 次，順序 off/on/on/off）。**

| 量測 | 關閉 | 開啟 | 變化 |
|---|---|---|---|
| 合成層 microbenchmark（ms/層） | 10.57 | 8.04 | −24% |
| 104K prompt prefill | 5.1 s | 3.9 s | −24% |
| AgentX c16 RPM | 65.0, 66.5 | 85.0, 81.5 | +26.6% |
| AgentX c32 RPM | 104.7, 116.1 | 122.8, 121.6 | +10.7% |
| needle-in-haystack | 5/6 | 6/6, 6/6 |  |
| GSM8K（T=0） | 91.2% | 90.7% | 雜訊內 |

### 7.6 理論與實際差距

- 層級：實測 −24%，理想 −32%，達成約 75%。損失來自：87% 而非 100% 的覆蓋率、mHC post 拆成獨立 kernel、兩個半 batch 的 GEMM 與 MoE 效率略低、Iris 自旋 workgroup 仍佔部分 CU。
- c32：+10.7%，低於約 +22% 的粗估上限；decoder 層與 decode 步驟不受影響，小於 4,096 rows 的 prefill 也不啟用。
- c16：+26.6%，高於僅以 prefill 佔比推算的上限。封閉迴路下 RPM ≈ c ÷ 延遲，prefill 加速同時縮短 TTFT 與佇列等待，效果會被放大；此外關閉組兩次（65.0、66.5）偏低。此差距我們尚未完全解釋。

## 8. gfx950 kernel 與通訊的配置調整

這部分是前述方法的基礎：在 130K 共享前綴的純 decode 下，64 人每 token 時間由 30.0 ms 降到 21.6 ms，與 vLLM 相同。

表 5 主要 kernel 修改。

| 元件 | 問題 | 修改 | 效果 |
|---|---|---|---|
| CSA indexer 打分 | 依 MI355X 的 32 MiB logits 預算寫死，長 prompt 時 launch 數暴增並在 overlap 下崩潰；key 以 byte 載入 | 改用 512 MiB 預算；page stride 允許 specialize 以 16 B 向量載入、waves_per_eu 2→8；16 列單 wave tile；6 個投機列共用一次 page 載入、32×32×64 MFMA | 384×65K：942→212 µs |
| indexer top-k select | 每列一個 CTA、需兩次讀 logits | 兩段式：12-bit histogram + LDS 壓縮，16 B 載入 | 220→64 µs |
| selected attention | 每 WG 串行處理整列；query 依 FlashMLA 補到 64 heads | 維度切分 wave、每 CU 2 個 WG、FP16 global dequant、XCD remap；ROCm decode query 只補到 16 heads | 64 人：24.3→22.8 ms/token |
| dense GEMM（decode） | M ≤ 256 時 gluon kernel 16–28 µs | 移植 block32 MXFP8 GEMM（tiled + packed、fused split-K），依 128 CU 重調 | 每步省約 2.8 ms |
| grid / XCD swizzle | MXFP8 GEMM、MXFP4 MoE、DSv4 prefill、selected attention 依 256 CU / 8 XCD 配置 | 改由裝置讀取 CU 與 XCD 數 | 64 人：28.8→26.8 ms/token |
| prefill MoE（MXFP4） | E8M0 scale 逐 byte 載入，每個 K tile 等自己的 scale，B 預取被一併等掉 | stage1 三槽 LDS pipeline + dword scale；stage2 dword scale；reduce 改 M-major 與 .cg 讀取（8.1 節） | M=4096 每層 1111→1021 µs，M=8192 2239→1570 µs |
| prefill all-reduce | RCCL 168 MB 6.19 ms | PCIe 主機上改走 Iris two-shot | 104K prefill 4.15→3.67 s |
| decode all-reduce | Iris 預設 84 programs 對 1–4 MB 訊息過多 | 2 MB 以下用 32 programs | 16 人 decode −10% |
| 草稿注意力 | 寬度 134 落入 portable Triton kernel，0.77 ms/call | 補到 192 並走 gfx950 sparse kernel | 每步省 2.3 ms |

1. MoE 的頻寬上限（11/15 更正）：初版以規格書的 HBM 峰值 8 TB/s 估計 decode MoE 只達 72%。以 Triton 純讀 kernel（4 GB）實測，這台機器的 HBM 讀取峰值是 3.13 TB/s。以此為準，decode 大小（M=16–512）的 stage1 達 2.6–3.0 TB/s、stage2 達 2.5–2.8 TB/s，即實測峰值的 85–97%，decode MoE 已無優化空間。改用 vLLM 的 aiter FlyDSL a8w4 並為 128 CU 重新調參後，兩次交錯重複測試與內建 kernel 打平（c16 70.5 vs 70.05、c32 108.5 vs 112.0、c64 139.3 vs 137.6 RPM），故未採用。
2. decode all-reduce 的延遲上限：訊息約 1.5 MB，two-shot 每張 GPU 收發約 2 × 3/4 × 1.5 MB，以 55 GB/s 計約 41 µs，實測 110–130 µs。這表示主要成本是同步與來回延遲而非頻寬，也解釋了 INT8 壓縮（傳輸量減半）、one-shot 改 two-shot、與 mHC 融合三者皆無效。

### 8.1 prefill MoE GEMM 重寫（11/15）

TBO 開啟後 prefill 以 M=4096（8,192-token chunk 切半）進入 MoE。路徑為 moe_sorting → 量化 → stage1（gate/up，BLOCK_N 128、BLOCK_K 256）→ stage2（down，BLOCK_N 256、BLOCK_K 128；M ≤ 2048 用 FP32 atomic 累加，否則寫 BF16 partials 再 reduce）。

先做的設定掃描（block_m 16–128 × stage1 warps 1–8 × 排序旗標 × atomic/reduce；stage1 XCD swizzle + GROUP_SIZE_M 1–32）端到端全在雜訊內，所以直接改 kernel。根因在反組譯中可見：E8M0 scale 以 buffer_load_ubyte 逐 byte 載入、再用 v_perm 拼裝，每個 K tile 都要等自己的 scale；而 vmcnt 依發出順序退場，等 scale 時順帶把已發出的 B 預取也等掉，pipeline 實際只有一層。

| 改動 | 做法 | 旗標（預設 1，設 0 退回舊 kernel） |
|---|---|---|
| stage1 pipeline | BLOCK_M 64 時三槽 LDS ring：tile k 計算時 A[k+2]、B[k+1]、scale[k+1] 同時在途；scale 以整個 4-byte CDNA4 cell 載入、到 MFMA 前才拆；每 expert 24–320 routes 時選 64-row tile | TOKENSPEED_MOE_PREFILL_S1_V2 |
| stage2 scale | 1x2 kernel（BLOCK_M ≥ 32）以 dword 載入 scale，一個 cell 供兩個 BK128 tile | TOKENSPEED_MOE_PREFILL_S2_V2 |
| stage2 reduce | reduce 模式改 M-major tile 順序（同一 route block 的 20 個 N tile 連續執行，W2 留在 L2）；reduce 讀 partials 用 .cg | TOKENSPEED_MOE_PREFILL_S2_REDUCE_V2 |

**表 5b 整個 MoE 每層時間（單 GPU，DSv4.1 TP4 形狀 E384／H5120／I576／top-6）。**

| M（tokens） | 舊 | 新 | 變化 |
|---|---|---|---|
| 2048 | 839 µs | 806 µs | −4% |
| 4096（TBO 下的生產大小） | 1111 µs | 1021 µs | −8% |
| 8192 | 2239 µs | 1570 µs | −30% |

- 正確性：reduce 模式與舊 kernel 逐位元相同且 deterministic；atomic 模式差 5.7e-4–8.8e-4，等於舊路徑自身的雜訊。無 register spill（stage1 216 VGPR、occupancy 2；stage2 148–154 VGPR、occupancy 3）；257 個測試通過。
- decode 不受影響：M=16–512 的新舊時間相同（0–2% 內）。
- 端到端：同一 build 以旗標 OFF/ON/ON/OFF 跑 c4（無 profiler 計時），extend 步 GPU 時間 138.6 / 131.6 / 121.9 / 133.0 ms（開啟平均 −6%），decode 步不變。prefill MoE 只佔 GPU 時間約 1–3%，RPM 差異落在 ±10% 漂移內；依「獨立量測有收益且正確就保留」的原則預設開啟。
- 試過但較慢：reduce 融合進 GEMM（計數器 + buffer_wbl2，M=4096 GEMM 變 1685 µs；改 relaxed 也有 532 µs）、部分 slot 改 atomic 的 hybrid（435–460 µs vs 426）、其他 atomic tile 順序（441–473 µs）、reduce kernel 換 tile／向量化（±3 µs）。atomic／reduce 門檻移到 3584 可再省 18–31 µs，但該區間會失去 determinism，未改。

## 9. 端到端結果

**表 6 AgentX 壓力模式 RPM（各階段為累加；baseline 至 + TBO 為重複量測平均，最終 build 每格 1 次）。**

| 並行度 | baseline | + BV | + BV + prefill TBO | 最終 build（11/16） | vLLM 最佳 | 倍數（最終） |
|---|---|---|---|---|---|---|
| c1 | 15.9 | 18.9 | 17.8 | 18.4 | 11.0 | **1.67×** |
| c4 | 37.2 | 38.4 | 48.1 | 44.3 | 37.3 | **1.19×** |
| c16 | 63.8 | 69.0 | 83.2 | 81.7 | 63.9 | **1.28×** |
| c32 | 106.9 | 113.0 | 122.2 | 124.3 | 111.7 | **1.11×** |
| c64 | 137.8 | 138.4 | 151.2 | 165.4 | 135.6 | **1.22×** |
| c128 | 132.3 | 127.0 | 129.1 | 136.4 | 23.2 | **5.88×** |

最終 build（mi350p-final 839d4cdf）= + BV + prefill TBO，再加上 shard-local 整塊驗證（6.8 節）與重寫的 prefill MoE（8.1 節），UnieInfra 全部以 gmu 0.86 執行。各格 ITL：c1 4.72、c4 9.52、c16 20.66、c32 23.44、c64 38.51、c128 140.89 ms；TTFT：0.39、5.12、1.59、0.53、0.87、3.02 s。6 格 0 錯誤、無 HSA OUT_OF_RESOURCES；c128 的逐步 prefix 命中率 98–99%。

最終 build 與 + TBO 欄的差異（c4 −8%、c64 +9%）落在單格 ±10% 的漂移內：同一 build 只切 MoE 旗標的 c4 對照，四次 RPM 為 43.7 / 43.8 / 48.0 / 46.7，差異跟著先後次序而非旗標（第 10.2 節）。

![](/api/attachments/aee29991-fa92-483a-9523-539a04cb9c47)

![](/api/attachments/f56d28a3-7ffc-40fc-854a-221396d85ea3)

![](/api/attachments/83d2be0e-6050-467c-8070-0e3c64bb512c)

（上方三張圖為 11/14 的 baseline、+ BV、+ BV + TBO 三階段，未含最終 build。）

**官方模式（含思考時間、900 s）的對照**：c128 為 4.07×（126.7 vs 31.1 RPM，TTFT 1.30 vs 37.6 s），c64 為 1.085×（90.6 vs 83.5 RPM），TTFT 低 2.7 倍。c ≤ 64 時 client 大部分時間在等思考時間，引擎差異主要反映在 TTFT 而非 RPM。

**正確性（最終 build，正式旗標）**：needle-in-haystack 6/6；GSM8K（T=0，1,319 題）90.45%，TBO 版本 90.67%、無 TBO 91.21%，CI 門檻 0.90。

## 10. 負面結果

下列嘗試在微基準或 trace 上有效，但端到端沒有增益，或有明確的反效果。

**表 7　未採用的方法與原因。**

| 方法 | 量測 | 原因 |
|---|---|---|
| Attention DP4 + TP MoE | 64 人 58.3 vs 21.6 ms/token | 複製權重後的 GEMM 形狀慢 3.3×；各 DP rank 的事件迴圈不同步 |
| Expert parallelism（EP） | 同形狀 RPM −27%（cap 32）、−33%（cap 128） | PCIe 上 all-to-all 跨 socket 慢；TP-MoE 較佳 |
| decode 兩 micro-batch 重疊 | MoE 拆半 1.65×、GEMM 1.6× | decode 受權重頻寬限制，拆半等於讀兩次權重；即使 all-reduce 全藏住每步仍慢約 7 ms |
| INT8 decode all-reduce | 23.3 vs 21.6 ms/token | 延遲受限；拆掉 fused mHC post 反而變慢 |
| 只驗證前 3 個草稿 | decode bench −12%；AgentX ±6% 正負混雜 | 端到端無一致增益 |
| aiter MoE | 重複 A/B 與 gluon 打平 | 頻寬受限，調參僅 t16 −7% |
| prefill MoE 設定掃描（tile／warps／XCD swizzle） | 端到端皆在 ±5% 內 | 瓶頸在 scale 載入方式，不在設定；改為重寫 kernel（8.1 節） |
| forward 結果延後交出（主機空檔） | trace 每步 51→47 ms；AgentX c32 114.6 vs 114.45 | 純 decode 的改善未轉成混合負載吞吐 |
| hold + Iris small cap 8 MB 組合 | trace decode 步 −8%；AgentX c16 +1.2%、c32 −0.8%、c64 −3.7% | 同上；10.2 節證實 trace 上的主機空檔多為 profiler 假象 |
| TBO 門檻 1024 rows + Iris small cap 8 MB | c32／c64 各 2 次，雜訊內 | 1K–4K rows 的 prefill 切半後 GEMM 太小 |
| --enable-mixed-batch（prefill 不縮） | c32／c64 無增益 | decode 搭便車的收益被 prefill 輪變長抵消 |
| GIL switch interval 500 µs | c32 −31%、c4 −17% | 主執行緒工作被切碎 |
| 強制 HIP graph packet 提交 | graph launch 40.6 vs 39.3 ms | 「複雜 graph 退回逐節點提交」的假說不成立 |
| DEBUG_HIP_FORCE_GRAPH_QUEUES=1（graph 只用一條 queue） | 無 profiler：launch host 7.3→0.5 ms，但 decode 步 GPU 45.2→55.3 ms、RPM 128.5→122.0 | graph 內平行分支被序列化 |
| DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=2（± 單一 queue） | decode 步 49.0／47.4 ms、RPM 118.4／128.5 | 雜訊內或更差 |
| decode graph 內 RCCL 改 Iris | 9 個 ncclDevKernel 共 1.26 ms／步，其中 0.9 ms 是等其他 rank | 最多省約 0.3 ms／步；兩 rank microbench 證實 RCCL 不觸發慢路徑；未實作 |
| rows 注意力 split | 192 rows 66→69–79 µs | 更慢 |
| MoE e4m3 activation | 慢 3–10% | 較準確（相對誤差 0.047 vs 0.213）但較慢 |
| prefill graph 擴到 8,192 tokens | c32 107.2–112.3 vs 108.0 | 雜訊內；eager 啟動空檔多為 profiler 放大 |
| shadow n-gram 草稿 | 命中時每步 1.78 tokens，DSpark 3.94 | 混合使用最多 +5% |

### 10.1 Python 追蹤放大主機成本

開啟 Python 追蹤的 profiler 顯示 AgentX c32 有 20% GPU 閒置、每步 post_process 3.2 ms；無 profiler 實測 post_process 僅約 80 µs，GPU-only trace 的閒置為 6–11%。數個「主機端瓶頸」方向因此被高估。

### 10.2 GPU-only profiler 也會改變 HIP graph 的提交方式（11/15）

即使只開 GPU activity，c32 trace 仍顯示 GPU 只有 82% 忙碌（閒置 1,077 / 6,072 ms），其中每個 decode 步一次的 zero_page_fields→HtoD 空檔共 175 ms；CPU+GPU trace 中 hipGraphLaunch 的主機耗時約等於 graph 的 GPU 跨度（中位 60.7 vs 60.5 ms），看起來像「launch 阻塞整步、下一步準備全排在 graph 結束後」。單一 queue 模式下更出現每 256 個節點約 6 ms 的空檔，4 個 rank 同時發生。

讀 ROCm 7.2.4 clr 原始碼（rocvirtual.cpp）後發現：profiler 錄製時，graph 內每個 AQL packet 都會掛上 completion signal（attachSignal = timestamp_ != nullptr），預設 signal pool 只有 64 個，用完就得新建或等舊的。256 正是 DEBUG_HIP_GRAPH_BATCH_SIZE，這解釋了 6 ms 空檔與 60–100 ms 的 launch 阻塞。

改用無 profiler 的 TOKENSPEED_GPU_STEP_TIMING=1（在 execution stream 上以 event 夾住每個 forward，每 6 s 輸出 decode／extend 的 GPU 時間、launch 主機時間與 forward 間閒置）重量 c32：decode 步 GPU 45.2 ms、launch 只佔主機 7.3 ms、forward 之間 GPU 閒置僅 0.13–0.19%，主機一直跑在 GPU 前面。結論：decode 不受主機限制；先前的 hold、GIL、host prep 等修法只是重新分配同一段主機時間，難怪吞吐不變。主機端唯一剩下的方向是 prefill 的 eager launch（extend 步主機 95 ms 對 GPU 124 ms）。

### 10.3 端到端 A/B 的漂移

最終 build 對 c4c64-wave 基準的 A/B 一度顯示 c4 −10%、c16 −8%。改以同一 build 只切 prefill MoE 旗標、OFF/ON/ON/OFF 交錯的無 profiler 對照後，decode 步 GPU 時間為 24.9 / 24.4 / 21.0 / 21.5 ms，只隨先後次序變化、與旗標無關；同一設定的 ITL 在 7.9–9.9 ms 之間。該 A/B 的差異是負載漂移，不是退化。

## 11. 討論與限制

1. c4–c64 未達 1.3×（最終 1.11–1.28×）：此區間 KV 容量充足，結果取決於 decode 每步時間與 prefill 成本。decode MoE 已達實測頻寬 85–97%（8 節），decode 也不受主機限制（10.2 節），剩下的 decode 成本主要是 all-reduce（約 13% 牆鐘時間，延遲受限）。可能方向：Lamport 式低延遲 all-reduce 並融合 mHC post／RMSNorm／MXFP8 量化；降低 prefill eager launch 開銷（extend 步主機 95 ms 已逼近 GPU 124 ms，TBO 又使 kernel 數加倍）。
2. 非官方模式：壓力模式移除了思考時間，用來衡量容量；官方模式下 c ≤ 64 的 RPM 差距較小。
3. 單機、單模型：結論限於 PCIe 4×MI350P 與 DSv4.1；在有 xGMI 的機器上，通訊重疊的收益會較小。
4. 雜訊：600 s 單格 ±10–15%，同一設定連續跑也會漂移 ±10%（10.3 節）；最終 build 每格只跑 1 次，表 6 的倍數應以 ±10% 解讀。
5. 記憶體餘裕：KV 依預算長滿時有約 5% 的崩潰率，目前以 0.86 迴避。已掃描執行期 kernel 快取（7,304 個 amdgcn），有 208 個編譯產物使用 scratch（最高 _mxfp8_block32_tiled 2,024 B/lane），這些是 scratch 需求的來源；根本解法（讓 KV 預算計入 scratch，或在 KV 配置前於每個佇列預先配置 scratch）尚未實作。
6. 量測工具：GPU trace 只適合看 kernel 本身；主機／launch 相關的結論必須用無 profiler 的計時（10.2 節）。初版第 4 節與表 7 中以 trace 推得的主機空檔數字應視為上界。

## 12. 結論

在 GPU 間只有 PCIe 的 MI350P 上，長上下文 agent 推論的效能由三件事決定：KV 容量（決定高並行度的前綴命中率）、每步接受的 token 數（決定 decode 吞吐），以及 all-reduce 能否被計算覆蓋（決定 prefill 成本）。依注意力結構分組的 KV cache 讓 c128 達 5.88×；整塊驗證（含 shard-local 版本）與 prefill 兩 micro-batch 重疊讓 c16 達 1.28×、c64 達 1.22×；prefill MoE 重寫在 kernel 層有 8–30% 的收益，但端到端被 ±10% 的漂移蓋過。decode 的 MoE 已達頻寬上限、主機也不是瓶頸，c4–c64 要再往 2× 前進，下一步是 decode all-reduce 的延遲與 prefill 的 launch 開銷。

## 附錄 A：最終設定

```
TOKENSPEED_PREFILL_TBO=1                    # 方法三
TOKENSPEED_DSPARK_BLOCK_VERIFY=1            # 方法二（_MAX_BS=128）
TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1         # shard-local 整塊驗證（6.8 節）
TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64
TOKENSPEED_DSPARK_COUPLED_DRAFT=1           # coupled drafting
TOKENSPEED_CACHE_AWARE_ADMISSION=1
HIP_FORCE_DEV_KERNARG=1                     # 與 vLLM 映像預設相同
HSA_NO_SCRATCH_RECLAIM=1
TOKENSPEED_IRIS_Q8_ALLREDUCE=0              # 不使用量化通訊
# prefill MoE v2（8.1 節）預設開啟，無需設定；設 0 可退回舊 kernel：
#   TOKENSPEED_MOE_PREFILL_S1_V2 / _S2_V2 / _S2_REDUCE_V2
CHUNKED_PREFILL_SIZE=8192  MAX_NUM_SEQS=2*CONC  MAX_TOTAL_TOKENS=2^30
GPU_MEMORY_UTILIZATION=0.86（全部並行度）
--sampling-backend triton_full --enable-log-request-stats
# 量測用（預設關）：TOKENSPEED_GPU_STEP_TIMING=1
```

## 附錄 B：迭代時間線

| 日期 | 階段 | 關鍵結果 |
|---|---|---|
| 11-04–05 | baseline 與崩潰修正 | c4 全部 rank SIGSEGV；根因是 indexer 寫死 32 MiB 預算，修正後 overlap 恢復；Iris 接管 prefill all-reduce |
| 11-05–06 | 排程與移植 | c128 因 prefill-first 排程餓死 decode；decode-every-N、cache-aware admission；補回合併時遺失的 DSv4.1 改動 |
| 11-06–07 | kernel 追平 | indexer 4–5×、rows attention、16-head padding、block32 GEMM、CU／XCD 幾何；64 人 decode 30.0→21.6 ms/token（= vLLM） |
| 11-07 | KV 容量 | KV 受 token 上限而非記憶體限制；16.7M→33.5M，c128 3× |
| 11-08 | 取樣公平性 | greedy 忽略 temperature；改取樣後 c4–c64 為 0.82–0.91×；coupled drafting 補回 2–12% |
| 11-09–11 | 公平性與 best-vs-best | DP attention、EP 否決；關掉 INT8 通訊；官方模式壓不到引擎，改用壓力模式 |
| 11-12–13 | 整塊驗證與量測校正 | block verification、分塊與批次門檻；aiter／host-gap／prefill graph 三輪 A/B 無端到端增益 |
| 11-14 | prefill 通訊重疊 | stream priority 是關鍵；c16 +26.6%、c32 +10.7%；OUT_OF_RESOURCES 根因與 0.86 餘裕 |
| 11-14–15 | c4–c64 收尾 | shard-local 整塊驗證；MoE roofline（decode 已達 85–97%）；prefill MoE 重寫 M=4096 −8%、M=8192 −30% |
| 11-15–16 | 主機空檔釐清與最終 build | 「hipGraphLaunch 阻塞」為 profiler 假象；mi350p-final c1–c128 為 1.67／1.19／1.28／1.11／1.22／5.88×；GSM8K 90.45%、needle 6/6 |

## 附錄 C：分支、PR 與工具

- PR 分支：mi350p-final（839d4cdf），已推到 UnieAI/UnieInfra，PR 待建立：https://github.com/UnieAI/UnieInfra/pull/new/mi350p-final 。共 39 個 commit；除 prefill MoE kernel 外，所有改變行為的功能都預設關閉、以環境變數開啟。
  - gfx950 kernel：indexer、selected attention、block32 MXFP8 GEMM、CU／XCD 幾何、prefill MoE 重寫。
  - 通訊：PCIe 主機上 prefill 走 Iris all-reduce；選用的 INT8／FP8 two-shot（預設關）。
  - 排程與 prefill：decode-every-N、cache-aware admission、prefill 兩 micro-batch 重疊、32 列壓縮器尾巴等。
  - 投機解碼：coupled drafting、整塊驗證（分塊、批次門檻、shard-local）、first-k 驗證。
  - 工具與文件：TOKENSPEED_GPU_STEP_TIMING、.skills/unieinfra-hw-optimize。
- 本輪主要 commit（皆在 mi350p-final 上）：6361be94（prefill TBO）、49c86553（shard-local 整塊驗證）、62255315／dba84259（prefill MoE stage1／stage2，預設開）、188c0c52（硬體適配 skill）、761b8d08（stage2 reduce）、839d4cdf（GPU_STEP_TIMING）。
- 硬體適配 skill：.skills/unieinfra-hw-optimize 把本報告的流程寫成可重現步驟（bring-up、正確性門檻、基準測試、負載下剖析、槓桿目錄、A/B 雜訊控制、改 kernel 前先量 roofline、紀錄與 PR、常見陷阱），附 19 支腳本與模板，供換新硬體或交給其他模型照做。
- 實驗紀錄：experiments.json（每次迭代一筆，含否決項，#1–#250）；dashboard：https://claude.ai/artifact/L5bkHnHRbzb2thttkZhHHg 。

# 第二部分：工程細節版（與 Claude Doc 技術報告同步）

本部分與 [Claude Doc 技術報告](https://claude.ai/artifact/GZcbYZ7NVzFNCUJYnYv3XJ) 內容相同，偏工程紀錄（commit、量測細節、踩坑）；互動圖表只在 Claude Doc 版本中。更新至 2026-11-16。

## 工程版摘要

在 4 張 AMD MI350P（gfx950、PCIe、TP4）上跑 DeepSeek-V4.1-Flash 的 AgentX 工作負載，UnieInfra 以各自最佳設定對比 vLLM 最佳設定：11/16 最終 build（mi350p-final）在 c1 達 1.67 倍、c16 1.28 倍、c64 1.22 倍、c4 1.19 倍、c32 1.11 倍、c128 5.88 倍 RPM；c4–c64 仍未到 1.3 倍目標。最終 build 每格只跑一次，與 11/14 兩次平均的差異（c4 −8%、c64 +9%）落在單格 ±10% 的漂移內。

| 並行度 | UnieInfra RPM（11/16 最終 build） | 11/14 RPM（TBO，2 次平均） | vLLM 最佳 RPM | 倍數（最終） | 決定性因素 |
|---|---|---|---|---|---|
| c1 | 18.4 | 17.8 | 11.0（recipe cap） | 1.67× | prefill 與 TTFT |
| c4 | 44.3 | 48.1 | 37.3（2× cap） | 1.19× | prefill 重疊 + block verification |
| c16 | 81.7 | 83.3 | 63.9（2× cap） | 1.28× | prefill 重疊 + block verification |
| c32 | 124.3 | 122.2 | 111.7（2× cap） | 1.11× | decode 每步速度 |
| c64 | 165.4 | 151.3 | 135.6（2× cap） | 1.22× | decode 每步速度 + shard-local block verification |
| c128 | 136.4 | 131.1 | 23.2（recipe cap） | 5.88× | KV 容量 → prefix 命中 >90% vs 24–64% |

條件：AgentX 壓力模式（移除 think time）、每格 600 s、T=1.0 取樣、vLLM GPU memory utilization 0.9、UnieInfra 0.86、vLLM 取 recipe cap（CONC/4）與 2×CONC 兩者較佳者；全部 0 錯誤。最終設定為 coupled DSpark drafting + block verification + prefill 兩 micro-batch 通訊重疊（`TOKENSPEED_PREFILL_TBO=1`），全部 lossless，不使用量化通訊。

11/16 最終 build 另加上 shard-local block verification（`TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1`，chunk 64）與重寫的 prefill MoE kernel（預設開）；6 格 0 錯誤、無 `OUT_OF_RESOURCES`，needle 6/6、GSM8K（T=0，1319 題）90.45%。

三個主要結論：

- **容量決定高並行度**：UnieInfra 每 token 的 KV 約 940 B，vLLM 約 2090 B；c128 時 vLLM 放不下工作集，prefix cache 大量被逐出。
- **c4–c64 由 decode 與 prefill 每步效率決定**：GPU-only profile 顯示 all-reduce 佔 kernel 時間 37%（PCIe 頻寬受限），MoE 佔 25%；把 prefill all-reduce 藏到計算後面是 c16 跨過 1.3 倍的關鍵。
- **方法論比優化更常出錯**：greedy vs sampling、官方 vs 壓力、單次 600 s 格 ±10–15% 雜訊、profiler 放大 Python 成本 35–40 倍、profiler 改變 HIP graph 提交方式、測試意外載入不同 MoE kernel，這些都曾讓結論翻盤，下方「比較方法的演進與踩過的坑」逐一記錄。

### 完整結果圖（壓力模式 c1–c128）

兩個 vLLM 系列是灰色，UnieInfra 三個階段是由淺到深的藍色（互動圖見 Claude Doc；資料：results/agentx 壓力模式 600 s，2026-11-12 至 11-14；重複量測的格取平均，+ block 驗證 c16–c64 為 3 次、重疊 c4–c128 為 2 次；TTFT 為每格平均值，含排隊）。靜態版見第一部分第 9 節的三張圖。

TTFT 圖可以看到：在各自的最高輸出吞吐附近，UnieInfra 的平均 TTFT 大多在 0.5–4 秒；vLLM recipe cap 會讓 TTFT 拉到 30–140 秒（請求在佇列排隊），vLLM 2× cap 在 c128 崩到 189 秒。c4 的 TTFT 偏高是因為少數 40–50 萬 token 的冷 prompt 擋住了後面的請求，兩個引擎都有。

## 硬體、模型與測試方法

測試平台是一台 4×MI350P 的純 PCIe 機器，GPU 間沒有 xGMI，這個限制決定了大部分優化方向。

| 項目 | 設定 |
|---|---|
| GPU | 4× AMD MI350P（gfx950），每張 128 CU / 4 XCD、144 GB HBM；GPU0/1 在 CPU0、GPU2/3 在 CPU1，彼此只經 PCIe |
| 軟體 | ROCm 7.2.4；UnieInfra（TokenSpeed fork）於 ts-dev 容器；vLLM 官方 AMD 映像（DSv4.1 recipe，moe_backend aiter） |
| 模型 | DeepSeek-V4.1-Flash-1017，TP4；MoE 384 experts / top-6，MXFP4 權重；層間 mHC；CSA 稀疏注意力（indexer + selected attention） |
| 投機解碼 | 兩邊都用 DSpark：一次草稿 5 個 token，驗證 6 列；AgentX 請求 temperature 1.0 |
| 記憶體 | vLLM `GPU_MEMORY_UTILIZATION=0.9`；UnieInfra 最終全部 0.86（見「穩定性與正確性」） |

### AgentX 測資樣式

AgentX（agentx-harness 1.0.13 + AiPerf）回放真實的 coding-agent 對話軌跡，資料集 `semianalysis_cc_traces_weka_062126`，random seed 42，端點 `/v1/chat/completions` 串流。每個 concurrency lane 依序重放一個對話：後一個 turn 的 prompt = 前一個 turn 的 prompt + 回覆 + 新內容，所以 prefix cache 是核心。以下是 c32 壓力模式一次 600 s 量測期間的統計（1162 個請求，不含 warmup）：

| 欄位 | p10 | p50 | p90 | 最大 | 平均 |
|---|---|---|---|---|---|
| prompt 長度（tokens） | 29,819 | 83,175 | 217,979 | 578,948 | 112,432 |
| 輸出長度（tokens） | 94 | 235 | 2,059 | 14,177 | 765 |

- **對話結構**：106 個對話、每對話中位數 8.5 個 turn（最多 62）。`source_kind` 分為 `weka_main`（主 agent，約 56%）、`weka_subagent`（子 agent，`agent_depth=1`，約 41%）、`weka_flat`（單輪）。
- **前綴重用**：單一請求的 prompt 中位數 98% 可從前一輪命中；理論命中率（AiPerf `theoretical_prefix_cache_hit`）約 95%。實際需重算的 prefill 每請求 4–8K tokens。
- **每筆請求的紀錄**（`profile_export.jsonl`）：`conversation_id`、`turn_index`、`source_kind`、`agent_depth`、時間戳（credit / start / ack / end）、`input_sequence_length`、`output_sequence_length`、`usage_prompt_cache_read_tokens`、TTFT、ITL。本報告所有命中率都只算 profiling 階段（warmup 的第一輪必然是冷的）。

### 兩種跑法

| 模式 | 做法 | 用途 | 限制 |
|---|---|---|---|
| 官方（recorded gaps） | 照軌跡記錄的使用者思考時間送下一輪（中位 1.9 s、平均 14–22 s、上限 300 s），≥900 s | InferenceX 官方比較 | c≤64 時 client 有 34–63% 時間在等，RPM 由 trace 決定，壓不到引擎 |
| 壓力（`--ignore-trace-delays`） | 前一輪結束立即送下一輪，600 s | 量引擎容量 | 非官方；單格 run-to-run ±10–15% |

每個並行度雙方各跑兩種 running-request 上限：vLLM recipe 的 CONC/4 與 2×CONC，各取 RPM 較佳者（best-vs-best）。本報告的倍數預設指壓力模式。

## 比較方法的演進與踩過的坑

十次「結論翻盤」都來自量測方法，不是引擎本身；每一次修正後的規則都沉澱成之後所有數字的前提。

| # | 坑 | 影響 | 修正後的規則 |
|---|---|---|---|
| 1 | AMD 預設 `--sampling-backend greedy` 忽略請求的 temperature 1.0，而 vLLM 會取樣 | greedy 接受長度 3.1–4.0，取樣只有 2.8；先前 c4–c64 的 1.0–1.2× 改用取樣後變成 0.82–0.91× | 所有對 vLLM 的數字一律 `--sampling-backend triton_full` |
| 2 | vLLM recipe 把 running requests 限在 CONC/4 | 早期 c32 1.95× 是對手被限流；改用 2× cap 後只剩 0.86× | best-vs-best：雙方各跑兩種 cap 取較佳 |
| 3 | 官方模式按 trace 的 think time 送請求 | c16 名義 16 並行，實際只有 4–6 個在途；RPM 幾乎只反映 trace | 引擎容量用壓力模式，官方模式另表 |
| 4 | 量化通訊混入公平比較 | INT8 prefill all-reduce 曾帶來 c32 +12%、c128 +28% | 公平比較全部 quantization-free；INT8 是使用者決策項 |
| 5 | 單次 600 s 格雜訊 | 同設定兩次可差 ±10–15%（aiter c64：89.8 vs 134.0） | 任何 <15% 的判斷要 ≥2 次、交錯順序（A B B A） |
| 6 | torch profiler 的 Python tracer（with_stack） | Python 成本放大 35–40×，誤判「GPU 閒置 20%」 | 端到端判斷用 GPU-only trace；CPU 成本用無 profiler 的實測 |
| 7 | 測試腳本的 `ENGINE_PYTHONPATH` 偏差 | block verification 試驗意外載入未 commit 的 aiter MoE，KV 少 14 GB，增益被混淆 | A/B 兩臂只差一個旗標，server log 確認 `moe_backend` 與 KV 容量 |
| 8 | 啟動腳本 `-e HSA_NO_SCRATCH_RECLAIM` 沒帶值 | docker 把 image 內建的 =1 刪掉，導致執行期崩潰 | 改 `-e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}`，並用 `/proc/<pid>/environ` 核對 |
| 9 | 開 torch profiler 時 ROCm 7.2.4 替 graph 內每個 AQL packet 掛 completion signal（signal pool 64 個） | trace 上 hipGraphLaunch 看似阻塞整步 60–100 ms、單一 queue 模式每 256 節點空 6 ms；不開 profiler 實測 launch 只佔 7.3 / 45 ms，forward 間閒置 0.13–0.19% | host / launch 問題一律用 `TOKENSPEED_GPU_STEP_TIMING=1`（event 夾 forward）；GPU trace 只看 kernel 本身 |
| 10 | 同一 build、同一旗標連續跑的負載漂移 | c4 同設定 ITL 7.9–9.9 ms、RPM ±10%；final vs wave A/B 的 −8–10% 全是漂移（同 build 旗標 OFF/ON/ON/OFF 下 decode 步時只隨先後次序變） | 端到端 A/B 只用來擋明顯退化；有獨立量測證實收益且正確的改動保留並預設開啟 |

其他常態規則：只在同一 build、同一 prefill chunk 大小下比較；命中率只算 profiling 階段；c1–c128 每格都要有數字才推進；失敗的格標 `FAILED-` 保留不刪。

## 迭代時間線

十二天的工作可以分成十個階段：先讓引擎穩定、再把 kernel 追平、再用容量拉開 c128，中途兩次發現比較方法有誤，再靠 decode 的接受長度與 prefill 通訊重疊把 c4–c16 拉到 1.3×；最後兩天做 c4–c64 的收尾（shard-local 驗證、prefill MoE 重寫、host 空檔釐清）與最終 build 的驗收。

| 日期 | 階段 | 關鍵結果 |
|---|---|---|
| 11-04–05 | Baseline 與崩潰修正 | UnieInfra c4 全部 rank SIGSEGV；根因是 indexer 寫死 32 MiB 預算，修掉後 overlap 恢復；Iris 接管 prefill all-reduce |
| 11-05–06 | 排程與移植 | c128 因 prefill-first 排程饑餓 decode；decode-every-N、cache-aware admission；從 dsv41 分支搬回遺失的 prefill 優化 |
| 11-06–07 | Kernel 追平 | indexer 4–5×、rows attention、16-head padding、block32 GEMM、CU/XCD 幾何；64 人 decode 30.0 → 21.6 ms/token（= vLLM） |
| 11-07 | KV 容量 | 發現 KV 受 token 上限而非記憶體限制；16.7M → 33.5M，c128 3× |
| 11-08 | 取樣不對等 | greedy 忽略 temperature，改取樣後 c4–c64 變 0.82–0.91×；coupled drafting 補回 2–12% |
| 11-09–11 | 公平性與 best-vs-best | DP attention 否決；關掉 INT8 通訊；發現官方模式壓不到引擎，改用壓力模式 best-vs-best |
| 11-12–13 | Block verification 與量測校正 | block verification、KV 依預算；aiter / host-gap / prefill graph 三輪 A/B 均無端到端增益；GPU-only profile 指出 all-reduce 佔 37% |
| 11-14 | Prefill 通訊重疊 | stream priority 是關鍵；c16 +26.6%、c32 +10.7%；OUT_OF_RESOURCES 根因與 0.86 餘裕 |
| 11-14–15 | c4–c64 收尾 | shard-local block verification；decode MoE 已達實測頻寬 85–97%；prefill MoE 重寫每層 −8%（M=4096）至 −30%（M=8192）；hold/E2/mixed batch/MoE tuning 皆在 ±5% 雜訊內 |
| 11-15–16 | host 空檔釐清與最終 build | 「hipGraphLaunch 阻塞」是 profiler 假象；mi350p-final c1–c128 為 1.67/1.19/1.28/1.11/1.22/5.88×，GSM8K 90.45%、needle 6/6 |

（互動圖見 Claude Doc：各階段 c4–c64 對 vLLM 的倍數；c128（5.5–5.7×）未畫入。）

c4–c64 的「原本」已包含前七個階段的 kernel 與排程工作，而且仍只是持平；把 c4 與 c16 推過 1.3× 的是 prefill 重疊。c32–c64 的每步時間由 decode 主導，重疊只帶來 10% 左右。



## Kernel 層優化

Kernel 工作把 130K context decode 從明顯落後 vLLM 拉到持平：64 人 130K 共享前綴 decode 從 30.0 ms/token 降到 21.6 ms/token（vLLM 同 bench 21.6）。主線是「MI350P 不是 MI355X」：128 CU / 4 XCD，而很多 gfx950 kernel 寫死 256 CU / 8 XCD。

### Indexer（CSA 稀疏注意力的選擇器）

| 改動 | 量測 | commit |
|---|---|---|
| 修正 gfx950 CSA2 indexer 忽略 512 MiB logits 預算、寫死 32 MiB | 434K prompt 不再崩潰（原本 overlap 下 SIGSEGV），prefill 76 → 62 s | 21edd46b |
| key 載入向量化（page_stride 可 specialize）、waves_per_eu 2→8 | 96 rows×75K：2332 → 617 µs（3.8×） | 2143b539 |
| 16-row one-wave tile、依工作量分桶的 score chunk | 192×150K：2341 → 1173 µs | 27314382 |
| row-block logits（aiter 風格：6 個 spec row 共用一次 page 載入、32×32×64 MFMA） | 384×65K：942 → 212 µs；155K prefill 5.6 → 5.0 s | 1afcc3b1 |
| candidate-mode logits 上 MFMA tile | 384×2048：419 → 146 µs | 7f1b42f6 |
| 兩段式 LDS 壓縮 top-k select | 220 → 64 µs（aiter 57） | b8595975 |
| 拿掉 rows 路徑的 −inf logits 預填 | 每步少 22.5 µs/層的純浪費 | 92ef86c3 |

### Selected attention、query padding 與幾何

- **XCD/CU 從裝置讀取**（c14eb59f）：MXFP8 GEMM split-K、MXFP4 MoE swizzle、DSv4 prefill、selected attention 原本照 MI355X 配置；64 人 decode 28.8 → 26.8 ms/token。
- **rows kernel**（dim-split waves、每 CU 2 個 WG、FP16 global dequant、XCD remap，6429a22d / 7bc097ba）：384 rows 170 → 140 µs。
- **ROCm decode query 只補到 16 heads**（e8189f55，原本照 FlashMLA 補到 64）：64 人 24.3 → 22.8 ms/token。
- **DSpark 草稿注意力改走 sparse kernel**（d164aa89）：0.77 → 約 0.1 ms/call，每步省 2.3 ms。

### Dense GEMM

移植 vLLM/aiter 的 block32 MXFP8 GEMM（tiled + packed、fused split-K）並針對 128 CU 重調（83d49991）：M=96 時每次快 2–3×（wo_b 29.0 → 10.5 µs），每步省約 2.8 ms；約 5% 輸出與精確 BF16 路徑差 1 ULP。

### MoE：gluon（內建）vs aiter

MoE 在 decode 是權重頻寬受限：192 rows 時 384 個 expert 幾乎全部被啟用，每步要讀約 75 GB/GPU 的 MXFP4 權重，實測約 3.2 TB/s。vLLM 的 AMD recipe 用 aiter 的 FlyDSL a8w4。我們把 aiter backend 正式接進來（`MOE_BACKEND=aiter`），並修掉它載入權重時的 12–14 GB 記憶體碎片（預配 padded buffer、每 32 個 expert 原地 shuffle，a5ae4510）。兩次交錯重複量測後與 gluon 打平（c16 70.05 vs 70.5、c32 112.0 vs 108.5、c64 137.6 vs 139.3 RPM），為 128 CU 重調 aiter 參數也只有 t16 快 7%，所以不採用。

### MoE roofline 與 prefill MoE 重寫（11/15，預設開）

**先量 roofline 再動 kernel**：用 Triton 純讀 kernel 量到的 HBM 讀取峰值是 3.13 TB/s（不是規格書的 8 TB/s）。以此為準，decode 大小（M=16–512）的 stage1 跑到 2.6–3.0 TB/s、stage2 2.5–2.8 TB/s，即峰值的 85–97%，decode MoE 不再有優化空間。前一輪的 tile / warp / XCD swizzle 設定掃描（block_m 16–128 × warps 1–8 × GROUP_SIZE_M 1–32）端到端全在雜訊內，所以直接改 prefill kernel。

**根因**：prefill stage1 / stage2 逐 byte 讀 E8M0 scale（`buffer_load_ubyte` + `v_perm` 拼裝），每個 K tile 都要等自己的 scale；而 vmcnt 按發出順序退場，等 scale 時順帶把 B 的預取也等掉，pipeline 實際上只有一層。

| 改動 | 做法 | 旗標（預設 1，設 0 退回） |
|---|---|---|
| stage1 pipeline | BLOCK_M 64 時 3-slot LDS ring：tile k 計算時 A[k+2]、B[k+1]、scale[k+1] 同時在途；scale 以整個 4-byte CDNA4 cell 讀入、到 MFMA 前才拆；每 expert 24–320 routes 時選 64-row tile | `TOKENSPEED_MOE_PREFILL_S1_V2` |
| stage2 scale | 1x2 kernel（BLOCK_M ≥ 32）以 dword 讀 scale，一個 cell 供兩個 BK128 tile | `TOKENSPEED_MOE_PREFILL_S2_V2` |
| stage2 reduce | M > 2048 的 reduce 模式改 M-major tile 順序（同一 route block 的 20 個 N tile 連續跑，W2 留在 L2），reduce 讀 BF16 partials 用 `.cg` | `TOKENSPEED_MOE_PREFILL_S2_REDUCE_V2` |

| M（tokens） | 整個 MoE 每層（舊 → 新） | 變化 |
|---|---|---|
| 2048 | 839 → 806 µs | −4% |
| 4096（TBO 下的生產大小） | 1111 → 1021 µs | −8% |
| 8192 | 2239 → 1570 µs | −30% |

正確性：reduce 模式與舊 kernel 逐位元相同且 deterministic；atomic 模式差 5.7e-4–8.8e-4，等於舊路徑自身的雜訊。無 register spill（stage1 216 VGPR、occupancy 2；stage2 148–154 VGPR、occupancy 3），兩個測試檔共 257 passed。

端到端：同一 build 以旗標 OFF/ON/ON/OFF 跑 c4（`TOKENSPEED_GPU_STEP_TIMING=1`），extend 步 GPU 時間 138.6/131.6/121.9/133.0 ms（開啟平均 −6%），decode 步不受影響。prefill MoE 只佔 GPU 時間約 1–3%，RPM 差異落在 ±10% 漂移內；依「獨立量測有收益且正確就保留」的原則預設開啟。

試過但較慢的：reduce 融合進 GEMM（計數器 + `buffer_wbl2`，M=4096 GEMM 變 1685 µs）、部分 slot atomic 的 hybrid（435–460 µs vs 426）、其他 atomic tile 順序（441–473 µs）、reduce 向量化（±3 µs）。atomic/reduce 門檻移到 3584 可再省 18–31 µs，但該區間失去 determinism，未改。

## 通訊

純 PCIe 的 TP4 讓 all-reduce 成為最大單項成本：AgentX c32 的 GPU-only trace 中佔 kernel 時間 37%，其中 prefill 大小的全縮佔總時間 21%、decode 大小佔 13%。

### Iris all-reduce

- **prefill 大小改走 Iris**（db3956be）：168 MB 時 RCCL 6.19 ms、Iris two-shot 5.51 ms，約 55 GB/s（PCIe 上限）；104K prompt prefill 4.15 → 3.67 s。
- **小訊息的 program 上限**（0f528991）：decode 大小（1–4 MB）時 84 個 programs 比需要的慢 2–2.7×；2 MB 以下改 32 programs，16 人 11.85 → 10.6 ms/token。
- **decode 的 all-reduce 是延遲受限**：每步約 90–109 次、每次 110–130 µs。INT8 壓縮、one-shot 改 two-shot、與 mHC post 融合三種方向都量過，結果中性。

### INT8 / FP8 two-shot（有損，預設關閉）

fce19969 / 4f519968 重寫 8-bit Iris pull two-shot：全 grid quantize kernel、每 128 一個 FP32 scale、16-byte peer load。168 MB 5.12 → 2.81 ms（1.82×），相對誤差 9.3e-3，GSM8K 93.0%。壓力模式 c32 +12%、c128 +28%。因為是有損通訊、vLLM 沒有使用，公平比較一律關閉（`TOKENSPEED_IRIS_Q8_ALLREDUCE=0`）。

### 通訊與計算重疊：先失敗、再成功

- **row-chunked FFN 重疊（#32、#49）**：RCCL 或 Iris-32 放 side stream，單一長 prompt 的 prefill 反而慢 5–15%。
- **decode 兩 micro-batch（評估）**：MoE 拆半變 1.65×、GEMM 1.6×，即使 all-reduce 全藏每步仍慢 7 ms；實測重疊的總時間反而比序列長。不做。
- **prefill 兩 micro-batch（TBO，6361be94）**：只對 ≥4096 rows 的 pure EXTEND。把一個 chunk 在請求邊界或 128 對齊處切成 A、B，B 視為同一請求的下一個 chunk。每層 compute stream 依序跑 attn(A)、attn(B)、MoE(A)、MoE(B)，同時 side stream 對另一半做 all-reduce；每半有自己的 Iris producer lane（32 programs），mHC post 拆成獨立 kernel。

TBO 成功的關鍵是 stream priority：ROCm 把同 priority 的 stream 輪流分到 4 個 HW queue，side stream 跟 compute stream 落在同一個 queue 就會完全序列化。第一版 trace 重疊 0 ms，改成 `priority=-1` 後，2574 ms 的 all-reduce 有 2236 ms 被藏住。先前 #32/#49 的失敗很可能是同一個原因。

| TBO 量測 | 關 | 開 |
|---|---|---|
| 合成層 microbenchmark（ms/層） | 10.57（serial fused Iris P84） | 8.04（−24%） |
| 104K prompt prefill | 5.1 s | 3.9 s |
| AgentX 壓力 c32 RPM | 104.7, 116.1 | 122.8, 121.6（+10.7%） |
| AgentX 壓力 c16 RPM | 65.0, 66.5 | 85.0, 