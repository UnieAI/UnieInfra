2026-11-14（最後更新 2026-11-17）

［目錄］

## 摘要

本研究在 4 張 AMD MI350P（gfx950，GPU 間僅 PCIe、無 xGMI）上以 TP4 服務 DeepSeek-V4.1-Flash，工作負載為 AgentX 的 coding-agent 多輪軌跡（prompt 中位數 83K tokens、輸出中位數 235 tokens）。我們先以 GPU-only trace 與逐 kernel 對照找出瓶頸，接著提出並實作四項方法：

1. 依 DSv4.1 注意力結構分組的 KV cache 佈局：每 token 佔用約 940 B，vLLM 約 2,090 B。
2. DSpark 投機解碼的 coupled drafting 與整塊驗證（block verification）：配合分塊計算、批次門檻與 TP 分片計算（shard-local），在 TP4 下可行。
3. prefill 的兩 micro-batch 通訊／計算重疊（TBO）：並指出 ROCm 硬體佇列分配會讓未設 priority 的重疊完全失效。
4. （11/17）mixed batch 保留 prefill TBO：讓 decode 搭 prefill 那一輪，同時不失去重疊。

另外重寫了 prefill MoE 的 MXFP4 GEMM（M=4096 每層 −8%、M=8192 −30%）。

在雙方各取最佳設定、同為 T=1.0 取樣與無量化通訊的條件下，11/16 最終 build（mi350p-final）的每分鐘完成請求數（RPM）為 vLLM 的 1.67×（c1）、1.19×（c4）、1.28×（c16）、1.11×（c32）、1.22×（c64）與 5.88×（c128），6 格 0 錯誤、無 OUT_OF_RESOURCES；needle 6/6、GSM8K（T=0）90.45%。

11/17 的兩項新結果：

- **mixed batch 保留 TBO**（第 8 節）：同日交錯對照 c64 +18%（174.9 vs 148.7 RPM）、c128 159.8 RPM（先前紀錄 136.4，+17%），c1–c32 在雜訊內；c4–c128 的 ITL p99 降 31–79%，代價是 TTFT 變長。GSM8K 0.9052。建議 c32 以上開啟。
- **ROCm 版本對照**（第 11 節，可提供給 AMD）：把 ROCm user-space 從 7.2.4 換成 7.14.1 或 10.0，端到端變慢（7.14：c4 −22%、c16 −11%、c64 −6%；10.0：c4 −8%、c16 −29%、c64 +2%），但 GEMM、RCCL、launch 等單元件都持平或更快。無 profiler 計時顯示退步是 GPU 端、偶發的 prefill 變慢與停頓。production 維持 7.2.4。

我們同時報告各方法的理論上限與實測差距、二十餘項未帶來端到端增益的嘗試，以及一項重要的量測教訓：ROCm 的 torch profiler 會改變 HIP graph 的提交方式，使「主機端阻塞」看起來遠比實際嚴重（第 12.3 節）。

（初版 2026-11-14；11/16 更新：shard-local 整塊驗證、prefill MoE 重寫、profiler 對 HIP graph 的影響、A/B 漂移、時間線與分支附錄；11/17 更新：mixed batch 保留 TBO、ROCm 7.2.4／7.14.1／10.0 對照，並把原「第二部分：工程細節版」的內容併入正文對應章節。互動圖表見 [Claude Doc 技術報告](https://claude.ai/artifact/GZcbYZ7NVzFNCUJYnYv3XJ)。）

## 1. 研究動機與貢獻

長上下文 agent 工作負載有三個特性：prompt 極長但輸出短、多輪對話之間前綴高度重疊（單一請求 98% 的 prompt 可由上一輪命中）、並且大量請求同時在線。這使得推論效能取決於三件事：KV cache 能留住多少前綴、decode 每一步能產出多少 token，以及 prefill 的成本。在沒有 xGMI 的 PCIe 機器上，TP 所需的 all-reduce 又進一步放大了 prefill 與 decode 的通訊成本。

本報告的貢獻如下：

1. 一套可重現的比較方法（壓力模式、best-vs-best、重複交錯量測），並說明哪些常見設定會讓結論失真（第 3 節）。
2. 以 GPU-only trace 拆解 AgentX 負載下的時間組成，定位 all-reduce（34%）與 MoE（24%）為主要成本（第 4 節）。
3. 四項方法的設計、實作細節、理論上限與實測差距分析（第 5–8 節），以及 gfx950 kernel 的配置調整與 prefill MoE GEMM 重寫（第 9 節）。
4. ROCm 7.2.4／7.14.1／10.0 的對照與定位（第 11 節），可作為與 AMD 討論的依據。
5. 負面結果的整理，包含失敗原因；並指出 ROCm 上連 GPU-only profiler 都會改變 HIP graph 提交方式，主機端結論必須改用無 profiler 計時（第 12 節）。
6. 把整套流程整理成可重現的硬體適配 skill（附錄 C）。

## 2. 背景

### 2.1 硬體

![](/api/attachments/4edf8fdf-b394-43aa-96a7-39db519581e0)

每張 MI350P 有 128 個 CU（4 個 XCD）與 144 GB HBM；GPU0/1 在 CPU0、GPU2/3 在 CPU1，彼此只經 PCIe。同屬 gfx950 的 MI355X 有 256 CU／8 XCD，許多現成 gfx950 kernel 的 grid 與 XCD swizzle 依 MI355X 寫死，這是第 9 節的出發點。

| 項目 | 設定 |
|---|---|
| GPU | 4× AMD MI350P（gfx950），每張 128 CU / 4 XCD、144 GB HBM；GPU0/1 在 CPU0、GPU2/3 在 CPU1，彼此只經 PCIe |
| 軟體 | ROCm 7.2.4（新版本的對照見第 11 節）；UnieInfra（TokenSpeed fork）於 ts-dev 容器；vLLM 官方 AMD 映像（DSv4.1 recipe，moe_backend aiter） |
| 模型 | DeepSeek-V4.1-Flash-1017，TP4；MoE 384 experts / top-6，MXFP4 權重；層間 mHC；CSA 稀疏注意力（indexer + selected attention） |
| 投機解碼 | 兩邊都用 DSpark：一次草稿 5 個 token，驗證 6 列；AgentX 請求 temperature 1.0 |
| 記憶體 | vLLM `GPU_MEMORY_UTILIZATION=0.9`；UnieInfra 最終全部 0.86（見 5.5 與 13.1 節） |

### 2.2 模型：DeepSeek-V4.1-Flash

- MoE：384 個 routed expert、top-6，權重 MXFP4；另有 shared expert。
- 注意力：CSA 稀疏注意力。indexer 以 MXFP4 key 對歷史打分、選出 top-k 位置（512），再由 selected attention 只對這些位置計算。另有 sliding window（128）分支與壓縮器：全域 KV 每數個 token 壓縮成一筆。
- mHC：層間的 multi-head connection，於每次 all-reduce 之後做 mix／post 運算。
- TP4：每層兩次 all-reduce（attention 輸出投影後、MoE 後），每次的資料量為 rows × 5,120 × 2 B。

### 2.3 DSpark 投機解碼

![](/api/attachments/e1d7f554-a527-4121-8535-a4e1d7ec5eb1)

DSpark 的小草稿模型每步先猜 5 個 token，大模型用一次 forward 同時算 6 列（原本一列 + 每個猜測一列）的機率，再決定收下前幾個。目標模型一次 forward 的成本幾乎與列數無關（受權重頻寬限制），所以 decode 吞吐量 ≈ 每步接受 token 數 ÷ 每步時間。AgentX 請求的 temperature 為 1.0，驗證必須是取樣式（rejection sampling 類），輸出分布須等於目標模型，不能只比對「最可能的 token」。

### 2.4 服務架構

![](/api/attachments/5dd05903-294b-404f-9499-f969e786cc3f)

## 3. 實驗方法

### 3.1 工作負載

AgentX（agentx-harness 1.0.13 + AiPerf，資料集 semianalysis_cc_traces_weka_062126，seed 42）以 OpenAI chat completions 串流（`/v1/chat/completions`）重放 coding-agent 軌跡。每個 concurrency lane 依序重放一段對話；後一輪 prompt = 前一輪 prompt + 回覆 + 新內容，所以 prefix cache 是核心。

**表 1 c32 壓力模式 600 s 量測期間的請求統計（1,162 個請求，不含 warmup）。**

| 欄位 | p10 | p50 | p90 | 最大 | 平均 |
|---|---|---|---|---|---|
| prompt（tokens） | 29,819 | 83,175 | 217,979 | 578,948 | 112,432 |
| 輸出（tokens） | 94 | 235 | 2,059 | 14,177 | 765 |

- 對話結構：106 段對話，每段中位數 8.5 輪（最多 62 輪）。`source_kind` 分為 `weka_main`（主 agent，約 56%）、`weka_subagent`（子 agent，`agent_depth=1`，約 41%）、`weka_flat`（單輪）。
- 前綴重用：每請求 prompt 的可命中比例中位數 98%；AiPerf 計算的理論前綴命中率（`theoretical_prefix_cache_hit`）約 95%。實際需重算的 prefill 每請求 4–8K tokens。
- 每筆請求的紀錄（`profile_export.jsonl`）：`conversation_id`、`turn_index`、`source_kind`、`agent_depth`、時間戳（credit / start / ack / end）、`input_sequence_length`、`output_sequence_length`、`usage_prompt_cache_read_tokens`、TTFT、ITL。本報告所有命中率都只算 profiling 階段（warmup 的第一輪必然是冷的）。

### 3.2 兩種跑法

| 模式 | 做法 | 用途 | 限制 |
|---|---|---|---|
| 官方（recorded gaps） | 照軌跡記錄的使用者思考時間送下一輪（中位 1.9 s、平均 14–22 s、上限 300 s），≥ 900 s | InferenceX 官方比較 | c ≤ 64 時 client 有 34–63% 時間在等思考時間，c16 名義 16 個並行實際只有 4–6 個在途，RPM 主要由軌跡決定而非引擎 |
| 壓力（`--ignore-trace-delays`） | 前一輪結束立即送下一輪，600 s；屬封閉迴路（closed loop）：RPM ≈ c ÷ 平均請求延遲 | 衡量引擎容量 | 非官方；單格 run-to-run ±10–15% |

本報告以壓力模式衡量引擎容量，倍數預設指壓力模式。

### 3.3 公平性控制

- 取樣：UnieInfra 一律 `--sampling-backend triton_full`。AMD 預設的 greedy 會忽略 temperature，接受長度虛高（3.1–4.0 vs 2.8），曾使 c4–c64 的比值從 0.82–0.91× 被誤估為 1.0–1.2×。
- running-request 上限：vLLM recipe 限制在 CONC/4。雙方都跑 CONC/4 與 2×CONC 兩種上限，取 RPM 較佳者（best-vs-best）。
- 記憶體：vLLM GPU memory utilization 0.9；UnieInfra 最終全部並行度皆為 0.86（原因見 5.5 節），仍比 vLLM 少用記憶體。
- 通訊：不使用量化通訊（INT8 all-reduce 雖可再增益，但 vLLM 未使用）。
- 雜訊：單次 600 s 格的 run-to-run 差異可達 ±10–15%，小於 15% 的差異至少跑兩次、以 A B B A 交錯。11/15 另以同一 build、只切旗標的 OFF/ON/ON/OFF 測得：同一設定的 c4 ITL 在 7.9–9.9 ms、RPM 在 ±10% 之間依先後次序漂移（第 12.4 節），所以端到端 A/B 只用來擋明顯退化；有獨立量測證實收益且正確的改動保留並預設開啟。
- 剖析：端到端判斷只用 GPU-only trace；開啟 Python 追蹤（with_stack）的 torch profiler 會把 Python 時間放大 35–40 倍（第 12.2 節）。主機／launch 問題改用無 profiler 的 `TOKENSPEED_GPU_STEP_TIMING=1`（以 event 夾住每個 forward），因為 profiler 本身會改變 HIP graph 的提交方式（第 12.3 節）。
- 其他常態規則：只在同一 build、同一 prefill chunk 大小下比較；命中率只算 profiling 階段；c1–c128 每格都要有數字才推進；失敗的格標 `FAILED-` 保留不刪。

### 3.4 比較方法的演進與踩過的坑

十次「結論翻盤」都來自量測方法，不是引擎本身；每一次修正後的規則都沉澱成之後所有數字的前提。

| # | 坑 | 影響 | 修正後的規則 |
|---|---|---|---|
| 1 | AMD 預設 `--sampling-backend greedy` 忽略請求的 temperature 1.0，而 vLLM 會取樣 | greedy 接受長度 3.1–4.0，取樣只有 2.8；先前 c4–c64 的 1.0–1.2× 改用取樣後變成 0.82–0.91× | 所有對 vLLM 的數字一律 `--sampling-backend triton_full` |
| 2 | vLLM recipe 把 running requests 限在 CONC/4 | 早期 c32 1.95× 是對手被限流；改用 2× cap 後只剩 0.86× | best-vs-best：雙方各跑兩種 cap 取較佳 |
| 3 | 官方模式按 trace 的 think time 送請求 | c16 名義 16 並行，實際只有 4–6 個在途；RPM 幾乎只反映 trace | 引擎容量用壓力模式，官方模式另表 |
| 4 | 量化通訊混入公平比較 | INT8 prefill all-reduce 曾帶來 c32 +12%、c128 +28% | 公平比較全部 quantization-free；INT8 是使用者決策項 |
| 5 | 單次 600 s 格雜訊 | 同設定兩次可差 ±10–15%（aiter c64：89.8 vs 134.0） | 任何 < 15% 的判斷要 ≥ 2 次、交錯順序（A B B A） |
| 6 | torch profiler 的 Python tracer（with_stack） | Python 成本放大 35–40×，誤判「GPU 閒置 20%」 | 端到端判斷用 GPU-only trace；CPU 成本用無 profiler 的實測 |
| 7 | 測試腳本的 `ENGINE_PYTHONPATH` 偏差 | block verification 試驗意外載入未 commit 的 aiter MoE，KV 少 14 GB，增益被混淆 | A/B 兩臂只差一個旗標，server log 確認 `moe_backend` 與 KV 容量 |
| 8 | 啟動腳本 `-e HSA_NO_SCRATCH_RECLAIM` 沒帶值 | docker 把 image 內建的 =1 刪掉，導致執行期崩潰 | 改 `-e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}`，並用 `/proc/<pid>/environ` 核對 |
| 9 | 開 torch profiler 時 ROCm 7.2.4 替 graph 內每個 AQL packet 掛 completion signal（signal pool 64 個） | trace 上 hipGraphLaunch 看似阻塞整步 60–100 ms、單一 queue 模式每 256 節點空 6 ms；不開 profiler 實測 launch 只佔 7.3 / 45 ms，forward 間閒置 0.13–0.19% | host / launch 問題一律用 `TOKENSPEED_GPU_STEP_TIMING=1`（event 夾 forward）；GPU trace 只看 kernel 本身 |
| 10 | 同一 build、同一旗標連續跑的負載漂移 | c4 同設定 ITL 7.9–9.9 ms、RPM ±10%；final vs wave A/B 的 −8–10% 全是漂移（同 build 旗標 OFF/ON/ON/OFF 下 decode 步時只隨先後次序變） | 端到端 A/B 只用來擋明顯退化；有獨立量測證實收益且正確的改動保留並預設開啟 |
## 4. 基準剖析：時間花在哪裡

在 c32 壓力負載下錄製 6 秒 GPU-only trace（加入 prefill 重疊之前），GPU 忙碌 89%（5,416 / 6,061 ms）。以 kernel 名稱分類如圖：

![](/api/attachments/998671f8-3f6c-4604-ae4f-637364493886)

以 kernel 時間計，all-reduce 佔 37%（PCIe 頻寬受限）、MoE 佔 25%；以牆鐘時間計，prefill 大小的 all-reduce 佔 21%、decode 大小佔 13%。

另一個對照是純 decode：32 個使用者共享 130K 前綴，雙方相同設定。去除偶發 all-reduce 等待後，每步中位數 UnieInfra 45.1 ms、vLLM 40.0 ms；但 GPU 忙碌時間幾乎相同（39.9 vs 39.4 ms），差距來自 GPU 閒置（5.3 vs 0.55 ms）。逐 kernel 分類後，all-reduce 雙方持平（11.96 vs 11.95 ms），MoE stage1 我們多 0.8 ms，attention 多 0.9 ms，mHC 我們少 1.2 ms。這說明 decode 的 kernel 總量已與 vLLM 相當；剩餘差距在主機端與少數 kernel。

由此得出三個可攻擊的方向：（a）高並行度時 KV 容量決定前綴命中率；（b）每步接受 token 數直接決定 decode 吞吐；（c）prefill 大小的 all-reduce 佔牆鐘時間 21%，而且受 PCIe 頻寬限制、期間計算單元閒置。

## 5. 方法一：依注意力結構分組的 KV cache

c128 的 5.6–5.9 個倍數幾乎全部來自 KV 容量：同樣 0.9 的記憶體預算，UnieInfra 存 3,300–4,400 萬 tokens，vLLM 存 1,460–1,855 萬。

### 5.1 問題

c128 時，128 段平均 112K tokens 的對話同時在線，工作集約 1,440 萬 tokens（尚未計入輸出與子 agent 分支）。vLLM 在 0.9 記憶體下的 KV 容量為 1,456–1,855 萬 tokens（依 running-request 上限而異，CUDA graph 佔用越多容量越小），實測前綴命中率只有 23.5–70.6%，遠低於理論的 95%，每輪都要大量重算 prefill（c128 RPM 僅 13.8–31.1）。

### 5.2 分析

DSv4.1 的每層 KV 並非「每 token 一筆」：全域層每數個 token 才存一筆壓縮後的 KV，sliding-window 層每請求只需保留最近 128 個位置，壓縮器只需保留未湊滿一個壓縮單位的尾巴。若以「每 token 固定大小」配置，就會為不需要的狀態保留空間。

### 5.3 設計與實作

![](/api/attachments/7eee1412-8d2f-4299-aeb0-572e86a9ebae)

- cache group：每一種狀態（global_r1、global_r2、compressor_tail_r2、swa）是一個 cache group，有自己的區塊粒度。server log 的 `Cache profile` 為 global_r1 140、global_r2 45、compressor_tail_r2 8、swa 2（每 parent 的 slot 數）。
  - 全域層：存壓縮後的 KV，每幾個 token 一筆，長 context 的邊際成本很低。
  - sliding-window 層：每請求只留最近一段視窗，不隨 context 增長。
  - 壓縮器尾端：只存未湊滿一個壓縮單位的尾巴。
- LCM arena：所有 group 放在同一塊 arena，以各 group 區塊大小的最小公倍數（LCM block，此設定為 256 tokens／3.2 MB）為單位分配。一個請求的所有 group 以同一組 block 分配、做前綴比對、在 retraction 時一起搬移、一起釋放，所以前綴命中時所有層的狀態同時可用。
- 壓縮器尾巴：ROCm 上每頁從 2 列改為 32 列（0f528991，對應的尾巴 kernel 以 ROWS 為 constexpr），13 萬 token 請求的 block table 從約 69K 欄縮到 4.3K 欄，降低每步 block table 打包成本。
- 容量由預算決定：原本 `MAX_TOTAL_TOKENS` 預設 16.7M，KV 受 token 上限而非記憶體限制；改為依記憶體預算（`MAX_TOTAL_TOKENS` = 2^30）後，0.9 時為 3,860–4,410 萬 tokens。
- cache-aware admission（e2317c72）：等待中的請求依「裝置上已快取的前綴長度」排序，最多被超車 64 次，讓能大量命中的請求先進來，c128 命中率 0% → 65%（當時容量較小時）。

vLLM 每 token 為何多約一倍，我們沒有讀其程式碼確認。

### 5.4 效果

**表 2 c128 壓力模式的 KV 容量與前綴命中率。**

| 設定 | 每 GPU token 容量 | 前綴命中 | RPM |
|---|---|---|---|
| vLLM recipe cap（32） | 1,855 萬 | 63.7% | 23.2 |
| vLLM 2× cap（256） | 1,456 萬 | 23.5% | 13.8 |
| UnieInfra，上限 33.5M | 3,355 萬 | 約 85% | 123.5 |
| UnieInfra，依預算，gmu 0.9 | 3,860 萬 | 94.2% | 132.3 |
| UnieInfra，依預算，gmu 0.86（最終） | 約 3,300 萬 | 93% | 127.1 / 131.1 |

**表 2b 容量設定的演進（c128 壓力 RPM）。**

| 設定 | token 容量 | c128 壓力 RPM | 備註 |
|---|---|---|---|
| `MAX_TOTAL_TOKENS` 16.7M（早期預設） | 16.7M | 41.5 | KV 受 token 上限而非記憶體限制 |
| 33.5M | 33.5M | 123.5 | prefix 命中 65.8% → 85.0% |
| 47M、64M | 預算上限 | c64 OOM | runtime 空間不足 |
| 2^30（依預算），gmu 0.9 | 38–44M | 132.3 | 約 5% 啟動在執行期崩潰（13.1 節） |
| 2^30，gmu 0.86（最終） | 約 33M | 127.1, 131.1；最終 build 136.4 | 穩定；vLLM 0.9 為 18.6M |

### 5.5 理論與實際差距

- 命中率：UnieInfra 實測 93–94%，與 AiPerf 理論上限 95% 只差 1–2 個百分點，主要是 256-token 的比對粒度與每輪新尾巴。vLLM 雖然容量也大於工作集的粗估，命中率卻遠低於理論；其原因（逐出策略、頁面補齊、輸出與分支佔用）我們未讀其原始碼確認。
- 容量上限：KV 吃滿 0.9 的預算時，長 chunked prefill 偶爾使 GPU runtime 在佇列中申請 scratch 記憶體失敗（`HSA_STATUS_ERROR_OUT_OF_RESOURCES`，abort 訊息皆為「Available Free mem: 0 MB」）。94 次啟動中崩潰 5 次；固定 33.5M 上限的 217 次為 0 次。另發現啟動腳本以不帶值的 `-e HSA_NO_SCRATCH_RECLAIM` 傳遞環境變數，使 docker 刪除了映像內建的 =1；修正後在 0.9 仍會崩潰，故最終以 0.86 保留約 6 GB 餘裕（細節見 13.1 節）。
- 適用範圍：c16–c64 時雙方容量都足以容納工作集（命中率皆 94–96%），此方法幾乎沒有貢獻。

### 5.6 排程

- **decode 饑餓**（#38、cfaac28b）：非 mixed 模式下一個 round 有 prefill 就不 decode，c128 queue 永遠不空，結果 0 完成。`TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1` 讓它變成 21.4 RPM。
- **cache-aware admission**（e2317c72）：見 5.3 節；c128 命中 0% → 65%。
- **GIL 競爭**（fcbe6e2d）：ExecutionPlan.forward 在 GIL 下複製 MB 級 block table、reserve() 每次 O(table)；修掉後每步 idle 5.3 → 4.2 ms。
- **prefill chunk**：16K → 4K 在開 INT8 時 c4 ITL p99 66 → 25 ms；最終設定用 8K，8K、4K、8K+graph 8192 在 c32 上於雜訊內（108.0 / 112.9 / 107.2–112.3）。
- **mixed batch**（decode 搭乘 prefill round）：原本的實作小並行度有利、c64 不利，最終關閉；11/14 以不縮 prefill 的設定在 c32／c64 重測，仍無增益。原因是 mixed 輪會失去 prefill TBO；11/17 補上後的結果見第 8 節。

## 6. 方法二：DSpark 的 coupled drafting 與整塊驗證

在 T=1.0 取樣下，每步接受的 token 數從 2.8（獨立取樣）提升到 3.1–3.2（coupled drafting），再提升到 3.3–3.9（block verification）；兩者都是 lossless，輸出分布等於目標模型。

### 6.1 問題

改用取樣式驗證後，每步接受長度只有約 2.8（vLLM 約 3.3–3.5），c4–c64 落後 vLLM 9–18%。逐 token 驗證以 min(1, P/Q) 決定每個草稿 token，一旦拒絕就停止，後面的 token 即使目標模型很滿意也被丟棄。

### 6.2 Coupled drafting（先行步驟，5e0db812）

草稿第 s 步以 Gumbel-max 對 (base + bias)/T 取樣，使用與驗證器第 s+1 列相同的 Philox 噪聲（offset = start_pos + 2 + s）。驗證器維持「目標取樣後精確比對」，所以輸出就是目標模型的樣本；當兩分布接近時，共用噪聲使兩邊抽到同一 token 的機率趨近 1 − TV(P, Q)。實作在 DSpark block kernel 的 SAMPLE 分支，以單元測試驗證草稿噪聲與驗證池逐位元相同。同 build、同 chunk 大小下，相對獨立取樣：c16 +11.6%、c32 +10.6%、c4 +2.4%、c64 +2.9%；接受長度 2.8 → 3.1–3.2。草稿溫度倍率 0.7 / 1.0 / 1.4 測過，1.0 最好。

### 6.3 整塊驗證的原理

**一句話**：草稿提出的一整串 token，不再一個一個獨立判斷「要不要收」，而是整串一起看；後面 token 的「好」可以補貼前面 token 的「不夠好」，所以每步平均能接受更多 token，而輸出的機率分布完全不變。

- **逐 token（標準做法）**：第 i 個草稿 token 以 min(1, P/Q) 的機率收下；一旦某個被拒絕就停，後面的全部丟掉。每個 token 只看自己。
- **整塊（Sun et al., 2024）**：從第一個開始累積一個「信用」p。P/Q > 1 的 token（大模型比小模型更想選它）會把信用往上拉，後面的 token 可以用到。最後接受「滿足 u ≤ h 的最長前綴」，而不是在第一個失敗就停。

```
p_0 = 1
p_i = min(1, p_(i−1) · P(x_i) / Q(x_i))
R_i = Σ_x max(p_i · P(x | x_≤i) − Q(x | x_≤i), 0)
h_i = R_i / (R_i + 1 − p_i)      （最後一個位置 h_n = p_n）
取 u_i ~ U(0,1)，接受最大的 i 使 u_i ≤ h_i
下一個 token ~ normalize(max(p_i · P − Q, 0))；整塊接受時 ~ P
```

p_i 是「累積信用」：P/Q > 1 的 token 會把信用往後帶，抵補先前 P/Q < 1 的 token；R_i 是在第 i 個位置停下時可用於重新取樣的殘差質量，h_i 以此分配「停」與「繼續」的機率，使輸出序列的分布等於目標模型自行取樣。論文證明此法 lossless，且期望接受長度不小於逐 token 驗證。（說明用的兩 token 例子為互動圖，見 Claude Doc。）

![](/api/attachments/613f662e-b2b6-4adf-a8bf-6302e3184eaf)

### 6.4 在 TP4 下實作的困難

殘差 R_i 需要對整個詞表（129,280）加總，所以需要草稿模型在每個位置的完整分布 Q。但 TP4 下 lm_head 是 vocab-parallel，每張 GPU 只有 32,320 個 logits，且草稿 logits 在草稿 forward 結束後即不存在。

![](/api/attachments/7af3124c-9cb2-445f-a34c-36f3b065601a)

### 6.5 實作

程式：sampling/block_verify.py、sampling/backends/triton_full.py、execution/drafter/deepseek_v41_dspark.py。

1. 保存草稿分布（execution/drafter/deepseek_v41_dspark.py）：在 bind_sampling_backend 時建立 DSparkBlockVerifyState，含 pool[slots, 5, local_vocab] FP32。草稿每一步把本 GPU 的 base logits 以 index_copy_ 寫入該請求的槽位（每槽位 5 × 32K FP32）。256 個請求時約 0.16 GB（258 槽位）。
2. 重建完整 Q（sampling/block_verify.py 的 draft_logits）：驗證時對每列重建 base + Markov bigram bias（bias = embedding[prev] @ projection_local^T），把 padding 欄設為 −inf，再以 all_gather_single 收齊 4 張 GPU，得到 [bs, 5, 129,280]；每請求每步約 2.6 MB 走 PCIe。
3. 向量化驗證（block_verify_chain）：對目標與草稿 logits 做 log_softmax（除以 temperature），由 P/Q 比值的累積和算 p（log 域、上限 0），由 p·P − Q 的正部算 R、h；以 u ≤ h 的最大索引得接受長度，從殘差以 Gumbel-max 取下一個 token；輸出寫成與原 verify_chain_target_sampled 相同的 predict / accept_index / accept_length 格式，下游不需修改。
4. 分塊（block_verify_chain_chunked，23b4d606）：每 16 個請求一塊處理。full-vocab FP32 暫存約 6 × bs × 5 × 129K × 4 B，會被 HIP graph 的記憶體池保留並從 KV 預算扣除（KV 少幾 GB）；分塊後暫存約 41 MB。CPU 測試確認 chunk = batch 時與不分塊逐位元相同。
5. 批次門檻（sampling/backends/triton_full.py，9d643855）：一步超過 128 個請求時改用原驗證。256 個請求時 all-gather 約 660 MB／步，decode ITL p90 退化到 2.3 s。graph 依批次大小分別捕捉，所以門檻在捕捉時決定路徑，執行時無額外成本。
6. TP group 綁定（e966e7a5）：wire_target 早於 bind_sampling_backend 執行，原實作在建立 state 時拿不到 TP group，warmup 時 all-gather 失敗（KeyError）；改為在 drafter 宣告 tp_group 並於建立 state 時傳入。

開關：`TOKENSPEED_DSPARK_BLOCK_VERIFY=1`（另 `_MAX_BS` 預設 128、`_CHUNK` 預設 16）。

**和 coupled drafting 的關係**：兩者同時開啟。草稿端用 coupled Gumbel 噪聲取樣（讓草稿更常猜中目標會抽的 token），驗證端在 ≤ 128 個請求時用 block verification、否則用原本的「目標取樣後精確比對」。兩條路徑輸出都正確，只是每步接受數與成本不同，所以可以一步一步自動切換。

### 6.6 效果

**表 3 只差整塊驗證開關的 AgentX 壓力測試（同一套 kernel；c16–c64 為 3 次平均）。**

| 並行度 | 關閉 RPM | 開啟 RPM | 變化 |
|---|---|---|---|
| c1 | 15.9 | 18.9 | +19% |
| c4 | 37.2 | 38.4 | +3% |
| c16 | 63.8 | 69.0 | +8% |
| c32 | 106.9 | 113.0 | +6% |
| c64 | 137.8 | 138.4 | +0% |
| c128 | 132.3 | 127.0 | −4% |

（另一組只差旗標的量測：c1 +19%、c32 +8%、c16 +5%、c4 +3%、c64 +2%、c128 −4%，皆在雜訊內一致。）

正確性：GSM8K 500 題（T=1.0）開 95.8%、關 96.2%，差距在 ±0.9 pp 的抽樣誤差內；接受長度 2.1–3.7 → 3.3–3.9。c1 的 ITL p90 從 9.4 降至 6.1 ms（vLLM 7.5 ms）。

### 6.7 理論與實際差距

- 上限：若完全 decode-bound，吞吐增益 ≈ 接受長度增幅（約 +10–30%）。c1 的 +19% 落在此範圍。
- c4–c64 只有 +2–8%，原因有三：
  - 壓力負載下約 25–35% 的 GPU 時間是 prefill，不受影響；
  - 整塊驗證本身的成本：每請求每步約 2.6 MB 的 all-gather 走 PCIe，加上 full-vocab softmax；
  - 批次越大，每步 all-gather 越大，攤提效益越低。
- c128 −4%：多數步驟超過 128 個請求而走原驗證，屬雜訊範圍。

### 6.8 Shard-local 整塊驗證（11/14，49c86553）

6.5 的做法每個 chunk 要把 [16, 5, 129,280] 的 FP32 草稿 logits all-gather 到每張 GPU（約 41 MB／chunk）。c32 GPU-only trace 中這是 508 次 × 0.35 ms 的 RCCL 呼叫，暴露時間佔 3%。

改法（sampling/block_verify.py）：softmax 正規化與殘差 R_i 都是對詞表的加總，可以拆成各分片的部分和。每張 GPU 只在自己的 32,320 個詞上計算，跨卡只交換兩次小 all-gather：

1. 每個位置：本分片草稿 logits 的 log-sum-exp，以及草稿 token 落在本分片時的 logit（[bs, 5, 2]）。合併後得到全詞表的 log Q(x_i)，進而算出 p_i。
2. 每個可能的停止位置：本分片的殘差質量 Σ max(p·P − Q, 0)，以及本分片從殘差與從 P 各自的 Gumbel-max 候選（值與 token id）。下一個 token 對每種接受長度都預先選好，所以第二次交換就已帶回勝者，不需要第三次通訊。

Gumbel 噪聲對全詞表抽取後再切片，各分片不會重用同一份噪聲；輸出分布與收齊全部 logits 的版本相同。通訊量從每 chunk 數十 MB 降到每請求數十個浮點數，chunk 從 16 放大到 64。

- 正確性：CPU 統計測試與 6.5 版本的輸出分布 TV 距離 0.03–0.04，等於同樣本數的雜訊底。
- 效果（2 次，TBO 開啟）：c32 122.5 / 123.5 RPM（未分片 122.8 / 121.6）、c64 154.2 / 150.9（148.9 / 153.6），接受長度 3.6–4.2。端到端在雜訊內，但移除了大批次時 all-gather 隨 bs 線性成長的成本，最終 build 採用。
- 開關：`TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1`、`TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64`。

## 7. 方法三：prefill 的兩 micro-batch 通訊重疊（TBO）

**一句話**：把一個 prefill chunk 切成 A、B 兩半，讓一半在做 all-reduce 的時候另一半在計算，把 PCIe 通訊時間藏起來。

### 7.1 問題

prefill 以 8,192 tokens 為一塊，每層兩次 all-reduce（attention 輸出投影之後、MoE 之後），每次 8,192 × 5,120 × 2 B ≈ 84 MB。GPU 之間只有 PCIe（約 55 GB/s），中位數 1.67 ms，期間計算單元閒置。c32 trace 中這類 all-reduce 共 699 次、1,295 ms，佔牆鐘時間 21%。

### 7.2 理論上限

以合成層 microbenchmark 量測，序列執行（fused Iris all-reduce + mHC post，84 programs）每層 10.57 ms。若兩次 all-reduce（約 3.4 ms）完全被計算覆蓋，下限約 7.2 ms（−32%）。在端到端層級，若 21% 的牆鐘時間能被覆蓋 87%（實測覆蓋率，見 7.5），上限約為 −18% 牆鐘時間，即 c32 約 +22% RPM。

### 7.3 設計

![](/api/attachments/4aebcf93-bef3-46e1-bd2a-25dddd91705f)

- 適用條件：純 EXTEND 步驟且 ≥ 4,096 rows（`TOKENSPEED_PREFILL_TBO_MIN_ROWS`）；小的 prefill 切半後 GEMM 太小，得不償失。只對 encoder 層（每個 token 都要計算的前段層）；decoder 層只算每請求尾端 rows（narrowing），資料量小，不切。decode 路徑完全不動。
- 切點：優先選靠近中間的請求邊界；只有一個請求時，切在請求內絕對位置 128 對齊處，並避開 replay rows。
- 因果：被切開的請求，B 半視為同一請求的下一個 chunk。同一層內 A 先完成 attention 並把 sliding-window、壓縮全域 KV、壓縮器尾巴寫入 cache，B 的 attention 再讀取；與一般 chunked prefill 等價，只是發生在同一層內。
- 流水：每層 compute stream 依序 attn(A) → attn(B) → MoE(A) → MoE(B)；side stream 在 attn(B) 期間 reduce A 的 attention 輸出、MoE(A) 期間 reduce B 的、MoE(B) 期間 reduce A 的 MoE 輸出，B 的 MoE 輸出則與下一層 attn(A) 重疊。兩個 stream 以 event 同步，每個 micro-batch 重用自己的輸出緩衝。

### 7.4 實作

1. metadata（layers/attention/backends/specific/deepseek_v41.py）：新增 plan_prefill_micro_batches(align)，一次準備 A、B 與整體三份 forward metadata（V41ForwardState），執行時以 use_forward_state() 切換；KV 寫入位置、block table、壓縮器狀態都依各自的 rows 計算。
2. 模型（models/deepseek_v41.py）：_plan_encoder_overlap 判斷是否啟用並取得兩條通訊 lane；_encoder_layers_overlapped 實作上述流水；每層拆成 overlap_attention、overlap_ffn、overlap_finish 三個階段。
3. 通訊（distributed/comm_backend/iris.py、comm_ops.py 的 overlap_all_reduce_lanes）：
   - 每個 micro-batch 一條 Iris producer lane，A、B 可同時在途，於 Iris heap 預留（約多 320 MB，僅在開啟時配置）。
   - pull kernel 上限 32 programs：Iris 的 all-reduce 靠 GPU 上的 workgroup 輪詢（spin），會佔走計算單元。用 84 個時只能藏住 17% 的通訊，降到 32 個能藏住 62–64%。
   - mHC post 從 all-reduce 拆出為獨立 kernel：低 program 數時 fused 版較慢，改成 plain all-reduce + 獨立的 mHC post kernel。
4. stream priority：side stream 以 torch.cuda.Stream(priority=−1) 建立。

![](/api/attachments/1184a42c-51ff-46a7-8c5b-18016fdb9747)

**關鍵：stream priority**。第一版未設 priority，trace 顯示重疊時間為 0 ms、prefill 無改善。原因是 ROCm 會把同一個 priority 的 stream 輪流放進 4 個硬體佇列（HW queue），通訊 stream 跟計算 stream 剛好分到同一個佇列時，硬體會依序執行，程式上是兩個 stream，實際上完全序列化。設為 −1（較高）後它一定落在另一個佇列，2,574 ms 的 all-reduce 有 2,236 ms（87%）被覆蓋，104K prompt 的 prefill 從 5.1 s 降到 3.9 s。先前兩次 row-chunked FFN 重疊嘗試（#32、#49）在單一長 prompt 上反而慢 5–15%，很可能也是同一原因。開關：`TOKENSPEED_PREFILL_TBO=1`（`_IRIS_PROGRAMS` 預設 32），commit 6361be94。

### 7.5 效果

**表 4 prefill 重疊開關的量測（AgentX 各 2 次，順序 off/on/on/off）。**

| 量測 | 關閉 | 開啟 | 變化 |
|---|---|---|---|
| 合成層 microbenchmark（ms/層） | 10.57（serial fused Iris P84） | 8.04 | −24% |
| 104K prompt prefill | 5.1 s | 3.9 s | −24% |
| AgentX c16 RPM | 65.0, 66.5 | 85.0, 81.5 | +26.6% |
| AgentX c32 RPM | 104.7, 116.1 | 122.8, 121.6 | +10.7% |
| needle-in-haystack | 5/6 | 6/6, 6/6 |  |
| GSM8K（T=0） | 91.2% | 90.7% | 雜訊內 |

### 7.6 理論與實際差距

- 層級：實測 −24%，理想 −32%，達成約 75%。損失來自：87% 而非 100% 的覆蓋率、mHC post 拆成獨立 kernel、兩個半 batch 的 GEMM 與 MoE 效率略低、Iris 自旋 workgroup 仍佔部分 CU。
- c32：+10.7%，低於約 +22% 的粗估上限；decoder 層與 decode 步驟不受影響，小於 4,096 rows 的 prefill 也不啟用。
- c16：+26.6%，高於僅以 prefill 佔比推算的上限。封閉迴路下 RPM ≈ c ÷ 延遲，prefill 加速同時縮短 TTFT 與佇列等待，效果會被放大；此外關閉組兩次（65.0、66.5）偏低。此差距我們尚未完全解釋。

### 7.7 代價與限制

- 數值：通訊完全 lossless；attention 切成兩段、GEMM 的 batch 形狀改變，數值差異與一般 chunked prefill 相同。
- 只支援 TP（attention 與 MoE 用同一個 TP group），不支援 EP / DP。
- 只在 eager prefill 生效（chunk 大於 prefill graph 上限時）。
- 原本只對純 prefill 生效，mixed 輪會失去重疊；11/17 已擴充到 mixed 輪（第 8 節）。
## 8. 方法四：mixed batch 保留 prefill TBO（11/17）

### 8.1 問題

非 mixed 模式下，一輪有 prefill 就不跑 decode，正在 decode 的請求每遇到一個 8K prefill chunk 就要等約 0.2 s。開 `--enable-mixed-batch` 可讓 decode 搭 prefill 那一輪，但原本 mixed 輪會整輪跳過 prefill TBO（只接受純 prefill），prefill 變慢，所以先前測 mixed batch 都沒有增益（−2～−4%，見 5.6 節與第 12 節）。

### 8.2 做法（分支 `roy/mixed-tbo`，217bca75 + b9c0368f）

- mixed 輪也拆成兩個 micro-batch。切點一定落在 extend 列內：A 是前半段 prefill（純 prefill），B 是後半段 prefill 加上所有 decode 請求（mixed forward）。decode 列在 forward 裡本來就排在所有 extend 列之後，所以 B 是連續的一段。
- 每個 micro-batch 有自己的 forward mode（A 用 EXTEND、B 用 MIXED）、自己的 attention metadata，B 的 decode 列與整體窗口寫入完全相同的 cache 位置。
- 單元測試：`test_mixed_micro_batches_keep_every_decode_row_in_the_second`（cache 位置、位置編號、請求對應、`num_tokens`）；整個 `test_deepseek_v41_cache.py` 158 passed。
- 第一版有 bug：準備 A 時會覆蓋目前的 metadata，B 再從那份 metadata 讀 decode 列數就讀錯了。server 一跑 mixed 輪就報 `num_tokens disagrees`；第二個 commit 修正並加入能抓到它的測試。

### 8.3 正確性

GSM8K（CI 設定，T=0，1,319 題）同時在背景持續送 40K token 的冷 prompt（142 個），讓 GSM8K 的 decode 都落在 mixed 輪的 B 裡：**0.9052**（production 0.9045，門檻 0.90）。4 個 rank 都記錄到 `prefill TBO: first mixed window split at row 4008 of 8192`。

### 8.4 效能（同日交錯，mixed+TBO vs production）

| 並行 | RPM | 輸出 tok/s | ITL 平均 | ITL p99 | TTFT |
|---|---|---|---|---|---|
| c1 | 18.0 vs 18.9（−5%） | 256 vs 270 | 5.45 vs 5.38 ms | 29.6 vs 28.8 ms | 0.56 vs 0.44 s |
| c4 | 51.6 vs 52.8（−2%） | 770 vs 783 | 12.1 vs 17.5 ms | 74 vs 359 ms | 1.87 vs 0.92 s |
| c16（2 組平均） | 85.2 vs 87.9（−3%） | 1253 vs 1238 | 17.8 vs 20.6 ms | 83 vs 156 ms | 1.6 vs 1.2 s |
| c32 | 129.6 vs 126.3（+3%） | 1744 vs 1699 | 21.5 vs 22.6 ms | 62.5 vs 71.2 ms | 0.67 vs 0.49 s |
| c64 | **174.9 vs 148.7（+18%）** | 2276 vs 1901 | 34.7 vs 43.4 ms | 82 vs 167 ms | 1.33 vs 1.17 s |
| c128 | **159.8**（production 同日兩次都崩潰，見 13.3 節；先前紀錄 136.4，+17%） | 1786 | 80.5 ms（先前 141） | 134.5 ms | 10.4 s（先前 3.0） |

c16 的兩組：第一組 78.5 vs 92.4、第二組 91.8 vs 83.3。

- 待 decode 的請求越多，收益越大：c1–c32 在雜訊內，c64／c128 明顯提升；c64 也超過先前最佳 165.4。
- decode 不再被 prefill 卡住，c4–c128 的 ITL p99 降 31–79%。
- 代價：TTFT 變長（prefill 和 decode 共用同一輪）。
- 建議：c32 以上開 `--enable-mixed-batch`，c16 以下維持原設定。紀錄：experiments.json #255、#256。

## 9. gfx950 kernel 與通訊的配置調整

這部分是前述方法的基礎：在 130K 共享前綴的純 decode 下，64 人每 token 時間由 30.0 ms 降到 21.6 ms，與 vLLM 同 bench 的 21.6 ms 相同。主線是「MI350P 不是 MI355X」：128 CU / 4 XCD，而很多 gfx950 kernel 寫死 256 CU / 8 XCD。

**表 5 主要 kernel 修改。**

| 元件 | 問題 | 修改 | 效果 |
|---|---|---|---|
| CSA indexer 打分 | 依 MI355X 的 32 MiB logits 預算寫死，長 prompt 時 launch 數暴增並在 overlap 下崩潰；key 以 byte 載入 | 改用 512 MiB 預算；page stride 允許 specialize 以 16 B 向量載入、waves_per_eu 2→8；16 列單 wave tile；6 個投機列共用一次 page 載入、32×32×64 MFMA | 384×65K：942→212 µs |
| indexer top-k select | 每列一個 CTA、需兩次讀 logits | 兩段式：12-bit histogram + LDS 壓縮，16 B 載入 | 220→64 µs |
| selected attention | 每 WG 串行處理整列；query 依 FlashMLA 補到 64 heads | 維度切分 wave、每 CU 2 個 WG、FP16 global dequant、XCD remap；ROCm decode query 只補到 16 heads | 64 人：24.3→22.8 ms/token |
| dense GEMM（decode） | M ≤ 256 時 gluon kernel 16–28 µs | 移植 block32 MXFP8 GEMM（tiled + packed、fused split-K），依 128 CU 重調 | 每步省約 2.8 ms |
| grid / XCD swizzle | MXFP8 GEMM、MXFP4 MoE、DSv4 prefill、selected attention 依 256 CU / 8 XCD 配置 | 改由裝置讀取 CU 與 XCD 數 | 64 人：28.8→26.8 ms/token |
| prefill MoE（MXFP4） | E8M0 scale 逐 byte 載入，每個 K tile 等自己的 scale，B 預取被一併等掉 | stage1 三槽 LDS pipeline + dword scale；stage2 dword scale；reduce 改 M-major 與 .cg 讀取（9.6 節） | M=4096 每層 1111→1021 µs，M=8192 2239→1570 µs |
| prefill all-reduce | RCCL 168 MB 6.19 ms | PCIe 主機上改走 Iris two-shot | 104K prefill 4.15→3.67 s |
| decode all-reduce | Iris 預設 84 programs 對 1–4 MB 訊息過多 | 2 MB 以下用 32 programs | 16 人 decode −10% |
| 草稿注意力 | 寬度 134 落入 portable Triton kernel，0.77 ms/call | 補到 192 並走 gfx950 sparse kernel | 每步省 2.3 ms |

### 9.1 Indexer（CSA 稀疏注意力的選擇器）

| 改動 | 量測 | commit |
|---|---|---|
| 修正 gfx950 CSA2 indexer 忽略 512 MiB logits 預算、寫死 32 MiB | 434K prompt 不再崩潰（原本 overlap 下 SIGSEGV），prefill 76 → 62 s | 21edd46b |
| key 載入向量化（page_stride 可 specialize）、waves_per_eu 2→8 | 96 rows×75K：2332 → 617 µs（3.8×） | 2143b539 |
| 16-row one-wave tile、依工作量分桶的 score chunk | 192×150K：2341 → 1173 µs | 27314382 |
| row-block logits（aiter 風格：6 個 spec row 共用一次 page 載入、32×32×64 MFMA） | 384×65K：942 → 212 µs；155K prefill 5.6 → 5.0 s | 1afcc3b1 |
| candidate-mode logits 上 MFMA tile | 384×2048：419 → 146 µs | 7f1b42f6 |
| 兩段式 LDS 壓縮 top-k select | 220 → 64 µs（aiter 57） | b8595975 |
| 拿掉 rows 路徑的 −inf logits 預填 | 每步少 22.5 µs/層的純浪費 | 92ef86c3 |

### 9.2 Selected attention、query padding 與幾何

- **XCD/CU 從裝置讀取**（c14eb59f）：MXFP8 GEMM split-K、MXFP4 MoE swizzle、DSv4 prefill、selected attention 原本照 MI355X 配置；64 人 decode 28.8 → 26.8 ms/token。
- **rows kernel**（dim-split waves、每 CU 2 個 WG、FP16 global dequant、XCD remap，6429a22d / 7bc097ba）：384 rows 170 → 140 µs。
- **ROCm decode query 只補到 16 heads**（e8189f55，原本照 FlashMLA 補到 64）：64 人 24.3 → 22.8 ms/token。
- **DSpark 草稿注意力改走 sparse kernel**（d164aa89）：0.77 → 約 0.1 ms/call，每步省 2.3 ms。

### 9.3 Dense GEMM

移植 vLLM/aiter 的 block32 MXFP8 GEMM 並針對 128 CU 重調（83d49991）：M=96 時每次快 2–3×（wo_b 29.0 → 10.5 µs），每步省約 2.8 ms；約 5% 輸出與精確 BF16 路徑差 1 ULP。

### 9.4 MoE 的頻寬上限與 gluon vs aiter

MoE 在 decode 是權重頻寬受限：192 rows 時 384 個 expert 幾乎全部被啟用，每步要讀約 75 GB/GPU 的 MXFP4 權重，實測約 3.2 TB/s。

- **roofline（11/15 更正）**：初版以規格書的 HBM 峰值 8 TB/s 估計 decode MoE 只達 72%。以 Triton 純讀 kernel（4 GB）實測，這台機器的 HBM 讀取峰值是 3.13 TB/s。以此為準，decode 大小（M=16–512）的 stage1 達 2.6–3.0 TB/s、stage2 達 2.5–2.8 TB/s，即實測峰值的 85–97%，decode MoE 已無優化空間。
- **aiter**：vLLM 的 AMD recipe 用 aiter 的 FlyDSL a8w4。我們把 aiter backend 正式接進來（`MOE_BACKEND=aiter`），並修掉它載入權重時的 12–14 GB 記憶體碎片（預配 padded buffer、每 32 個 expert 原地 shuffle，a5ae4510；細節見 13.2 節）。為 128 CU 重新調參後，兩次交錯重複測試與內建 kernel 打平（c16 70.5 vs 70.05、c32 108.5 vs 112.0、c64 139.3 vs 137.6 RPM），重調 aiter 參數也只有 t16 快 7%，故未採用。

### 9.5 decode all-reduce 的延遲上限

訊息約 1.5 MB，two-shot 每張 GPU 收發約 2 × 3/4 × 1.5 MB，以 55 GB/s 計約 41 µs，實測 110–130 µs，每步約 90–109 次。這表示主要成本是同步與來回延遲而非頻寬，也解釋了 INT8 壓縮（傳輸量減半）、one-shot 改 two-shot、與 mHC 融合三者皆無效。

### 9.6 prefill MoE GEMM 重寫（11/15，預設開）

TBO 開啟後 prefill 以 M=4096（8,192-token chunk 切半）進入 MoE。路徑為 moe_sorting → 量化 → stage1（gate/up，BLOCK_N 128、BLOCK_K 256）→ stage2（down，BLOCK_N 256、BLOCK_K 128；M ≤ 2048 用 FP32 atomic 累加，否則寫 BF16 partials 再 reduce）。

先做的設定掃描（block_m 16–128 × stage1 warps 1–8 × 排序旗標 × atomic/reduce；stage1 XCD swizzle + GROUP_SIZE_M 1–32）端到端全在雜訊內，所以直接改 kernel。根因在反組譯中可見：E8M0 scale 以 buffer_load_ubyte 逐 byte 載入、再用 v_perm 拼裝，每個 K tile 都要等自己的 scale；而 vmcnt 依發出順序退場，等 scale 時順帶把已發出的 B 預取也等掉，pipeline 實際只有一層。

| 改動 | 做法 | 旗標（預設 1，設 0 退回舊 kernel） |
|---|---|---|
| stage1 pipeline | BLOCK_M 64 時三槽 LDS ring：tile k 計算時 A[k+2]、B[k+1]、scale[k+1] 同時在途；scale 以整個 4-byte CDNA4 cell 載入、到 MFMA 前才拆；每 expert 24–320 routes 時選 64-row tile | `TOKENSPEED_MOE_PREFILL_S1_V2` |
| stage2 scale | 1x2 kernel（BLOCK_M ≥ 32）以 dword 載入 scale，一個 cell 供兩個 BK128 tile | `TOKENSPEED_MOE_PREFILL_S2_V2` |
| stage2 reduce | M > 2048 的 reduce 模式改 M-major tile 順序（同一 route block 的 20 個 N tile 連續執行，W2 留在 L2）；reduce 讀 BF16 partials 用 .cg | `TOKENSPEED_MOE_PREFILL_S2_REDUCE_V2` |

**表 5b 整個 MoE 每層時間（單 GPU，DSv4.1 TP4 形狀 E384／H5120／I576／top-6）。**

| M（tokens） | 舊 | 新 | 變化 |
|---|---|---|---|
| 2048 | 839 µs | 806 µs | −4% |
| 4096（TBO 下的生產大小） | 1111 µs | 1021 µs | −8% |
| 8192 | 2239 µs | 1570 µs | −30% |

- 正確性：reduce 模式與舊 kernel 逐位元相同且 deterministic；atomic 模式差 5.7e-4–8.8e-4，等於舊路徑自身的雜訊。無 register spill（stage1 216 VGPR、occupancy 2；stage2 148–154 VGPR、occupancy 3）；兩個測試檔共 257 個測試通過。
- decode 不受影響：M=16–512 的新舊時間相同（0–2% 內）。
- 端到端：同一 build 以旗標 OFF/ON/ON/OFF 跑 c4（無 profiler 計時），extend 步 GPU 時間 138.6 / 131.6 / 121.9 / 133.0 ms（開啟平均 −6%），decode 步不變；四次 RPM 43.7 / 43.8 / 48.0 / 46.7。prefill MoE 只佔 GPU 時間約 1–3%，RPM 差異落在 ±10% 漂移內；依「獨立量測有收益且正確就保留」的原則預設開啟。commit：62255315（stage1）、dba84259（stage2）、761b8d08（stage2 reduce）。
- 試過但較慢：reduce 融合進 GEMM（計數器 + buffer_wbl2，M=4096 GEMM 變 1685 µs；改 relaxed 也有 532 µs）、部分 slot 改 atomic 的 hybrid（435–460 µs vs 426）、其他 atomic tile 順序（441–473 µs）、reduce kernel 換 tile／向量化（±3 µs）。atomic／reduce 門檻移到 3584 可再省 18–31 µs，但該區間會失去 determinism，未改。

### 9.7 通訊

純 PCIe 的 TP4 讓 all-reduce 成為最大單項成本（第 4 節）。

- **prefill 大小改走 Iris**（db3956be）：168 MB 時 RCCL 6.19 ms、Iris two-shot 5.51 ms，約 55 GB/s（PCIe 上限）；104K prompt prefill 4.15 → 3.67 s。
- **小訊息的 program 上限**（0f528991）：decode 大小（1–4 MB）時 84 個 programs 比需要的慢 2–2.7×；2 MB 以下改 32 programs，16 人 11.85 → 10.6 ms/token。
- **decode 的 all-reduce 是延遲受限**（9.5 節）：INT8 壓縮、one-shot 改 two-shot、與 mHC post 融合三種方向都量過，結果中性。
- **INT8 / FP8 two-shot（有損，預設關閉）**：fce19969 / 4f519968 重寫 8-bit Iris pull two-shot：全 grid quantize kernel、每 128 一個 FP32 scale、16-byte peer load。168 MB 5.12 → 2.81 ms（1.82×），相對誤差 9.3e-3，GSM8K 93.0%。壓力模式 c32 +12%、c128 +28%。因為是有損通訊、vLLM 沒有使用，公平比較一律關閉（`TOKENSPEED_IRIS_Q8_ALLREDUCE=0`）。
- **通訊與計算重疊：先失敗、再成功**：
  - row-chunked FFN 重疊（#32、#49）：RCCL 或 Iris-32 放 side stream，單一長 prompt 的 prefill 反而慢 5–15%。
  - decode 兩 micro-batch（評估）：MoE 拆半變 1.65×、GEMM 1.6×，即使 all-reduce 全藏每步仍慢 7 ms；實測重疊的總時間反而比序列長。不做。
  - prefill 兩 micro-batch（TBO，6361be94）：成功，見第 7 節；關鍵是 stream priority。

## 10. 端到端結果

**表 6 AgentX 壓力模式 RPM（各階段為累加；baseline 至 + TBO 為重複量測平均，最終 build 每格 1 次）。**

| 並行度 | baseline | + BV | + BV + prefill TBO | 最終 build（11/16） | vLLM 最佳 | 倍數（最終） |
|---|---|---|---|---|---|---|
| c1 | 15.9 | 18.9 | 17.8 | 18.4 | 11.0 | **1.67×** |
| c4 | 37.2 | 38.4 | 48.1 | 44.3 | 37.3 | **1.19×** |
| c16 | 63.8 | 69.0 | 83.2 | 81.7 | 63.9 | **1.28×** |
| c32 | 106.9 | 113.0 | 122.2 | 124.3 | 111.7 | **1.11×** |
| c64 | 137.8 | 138.4 | 151.2 | 165.4 | 135.6 | **1.22×** |
| c128 | 132.3 | 127.0 | 129.1 | 136.4 | 23.2 | **5.88×** |

最終 build（mi350p-final 839d4cdf）= + BV + prefill TBO，再加上 shard-local 整塊驗證（6.8 節）與重寫的 prefill MoE（9.6 節），UnieInfra 全部以 gmu 0.86 執行。各格 ITL：c1 4.72、c4 9.52、c16 20.66、c32 23.44、c64 38.51、c128 140.89 ms；TTFT：0.39、5.12、1.59、0.53、0.87、3.02 s。6 格 0 錯誤、無 HSA OUT_OF_RESOURCES；c128 的逐步 prefix 命中率 98–99%。

最終 build 與 + TBO 欄的差異（c4 −8%、c64 +9%）落在單格 ±10% 的漂移內：同一 build 只切 MoE 旗標的 c4 對照，四次 RPM 為 43.7 / 43.8 / 48.0 / 46.7，差異跟著先後次序而非旗標（第 12.4 節）。

**表 6b 各並行度的 vLLM 對照與決定性因素。**

| 並行度 | UnieInfra RPM（11/16 最終 build） | 11/14 RPM（TBO，2 次平均） | vLLM 最佳 RPM | 倍數（最終） | 決定性因素 |
|---|---|---|---|---|---|
| c1 | 18.4 | 17.8 | 11.0（recipe cap） | 1.67× | prefill 與 TTFT |
| c4 | 44.3 | 48.1 | 37.3（2× cap） | 1.19× | prefill 重疊 + block verification |
| c16 | 81.7 | 83.3 | 63.9（2× cap） | 1.28× | prefill 重疊 + block verification |
| c32 | 124.3 | 122.2 | 111.7（2× cap） | 1.11× | decode 每步速度 |
| c64 | 165.4 | 151.3 | 135.6（2× cap） | 1.22× | decode 每步速度 + shard-local block verification |
| c128 | 136.4 | 131.1 | 23.2（recipe cap） | 5.88× | KV 容量 → prefix 命中 > 90% vs 24–64% |

條件：AgentX 壓力模式（移除 think time）、每格 600 s、T=1.0 取樣、vLLM GPU memory utilization 0.9、UnieInfra 0.86、vLLM 取 recipe cap（CONC/4）與 2×CONC 兩者較佳者；全部 0 錯誤，全部 lossless，不使用量化通訊。c4 時 vLLM 另以 cap 32 重測為 45.9 RPM（同 cap 下 UnieInfra 51.9），兩邊各自最佳時 c4 為 1.13×。

c4–c64 的「原本」已包含前七個階段的 kernel 與排程工作，而且仍只是持平；把 c4 與 c16 推過 1.3× 的是 prefill 重疊。c32–c64 的每步時間由 decode 主導，重疊只帶來 10% 左右。11/17 的 mixed batch 保留 TBO 在 c64／c128 再 +17–18%（第 8 節）。

![](/api/attachments/aee29991-fa92-483a-9523-539a04cb9c47)

![](/api/attachments/f56d28a3-7ffc-40fc-854a-221396d85ea3)

![](/api/attachments/83d2be0e-6050-467c-8070-0e3c64bb512c)

（上方三張圖為 11/14 的 baseline、+ BV、+ BV + TBO 三階段，未含最終 build。互動版見 Claude Doc：兩個 vLLM 系列是灰色，UnieInfra 三個階段由淺到深的藍色；資料為 results/agentx 壓力模式 600 s，2026-11-12 至 11-14，重複量測的格取平均，+ block 驗證 c16–c64 為 3 次、重疊 c4–c128 為 2 次；TTFT 為每格平均值，含排隊。）

**TTFT**：在各自的最高輸出吞吐附近，UnieInfra 的平均 TTFT 大多在 0.5–4 秒；vLLM recipe cap 會讓 TTFT 拉到 30–140 秒（請求在佇列排隊），vLLM 2× cap 在 c128 崩到 189 秒。c4 的 TTFT 偏高是因為少數 40–50 萬 token 的冷 prompt 擋住了後面的請求，兩個引擎都有。

**官方模式（含思考時間、900 s）的對照**：c128 為 4.07×（126.7 vs 31.1 RPM，TTFT 1.30 vs 37.6 s），c64 為 1.085×（90.6 vs 83.5 RPM），TTFT 低 2.7 倍。c ≤ 64 時 client 大部分時間在等思考時間，引擎差異主要反映在 TTFT 而非 RPM。

**正確性（最終 build，正式旗標）**：needle-in-haystack 6/6；GSM8K（T=0，1,319 題）90.45%，TBO 版本 90.67%、無 TBO 91.21%，CI 門檻 0.90。完整驗證清單見 13.4 節。
## 11. ROCm 7.2.4 vs 7.14.1 vs 10.0 對照（11/17，可提供給 AMD）

**結論先講**：同一份 UnieInfra 程式碼（mi350p-final 839d4cdf）、同一台機器、同一天交錯跑，只換 ROCm user-space 函式庫：

- 換成主機 driver 對應的 **ROCm 7.14.1**（HIP 7.14）：c4 −22%、c16 −11%、c64 −6%（RPM）。
- 換成 vLLM 映像的 **ROCm 10.0**（HIP 7.15）：c4 −8%、c16 −29%、c64 +2%。
- 但單獨量各元件時，兩個新版本**幾乎全部持平或更快**（唯一例外是 10.0 的 eager 提交慢 6%）：hipBLASLt GEMM 快 5–30%、RCCL 小封包快 20%、eager launch 快 12%（7.14）。
- 投機解碼的接受長度、每個請求的 decode 速度在三套 runtime 上都相同。
- 差別在 **prefill 的穩定性**：7.2.4 同一個 prompt 每輪的 prefill 時間差不到 0.1 s；7.14 和 10.0 會不定期變慢，最多慢 50%（6.2 vs 4.1 s）和 30%（28.1 vs 21.6 s）。穩定時也慢 0–10%。這和端到端 ITL p99 翻倍、TTFT 變長的現象一致。
- 無 profiler 的 GPU 計時（11.7 節）顯示停頓發生在 **GPU 端**，主機送出每一步的時間沒有變；7.2.4 跑 4 次都沒有這個現象。根本原因還沒定位到，11.6 節列出推論與要請 AMD 協助的事項。

### 11.1 為什麼測、測什麼

- 這台機器上有三套 ROCm：
  - 主機：driver amdgpu 6.19.14，user-space 是 ROCm 7.14.1（`/opt/rocm/core-7.14`）。
  - UnieInfra 開發容器 ts-dev：ROCm 7.2.4（torch 2.14.0+rocm7.2）。
  - vLLM／SGLang 映像：ROCm 10.0.0（HIP runtime 7.15）。
- UnieInfra 一直跑在最舊的 7.2.4 上，所以要測換成和 driver 一致的 7.14、或 vLLM 用的 10.0，會不會變快。
- 名詞：7.14 和 10.0 都是 AMD TheRock 新建置系統的產物。10.0.0 是 SDK 版號，裡面的 HIP runtime 是 7.15（`hipRuntimeGetVersion` = 71526333）；7.14.1 的 HIP 是 71460850；7.2.4 是 70253211。7.14 與 10.0 的 RCCL 都是 2.30.4。

### 11.2 測試環境

| 項目 | 內容 |
|---|---|
| 硬體 | 4× AMD Instinct MI350P（gfx950，每張 128 CU／144 GB HBM），GPU 之間只有 PCIe Gen5 x16，沒有 xGMI |
| 主機 | Ubuntu 24.04.4、kernel 6.8.0-100-generic、amdgpu 6.19.14.31400000；未設 `iommu=pt`，NUMA auto balancing 開啟（RCCL 啟動時有提示） |
| 模型 | DeepSeek-V4.1-Flash-1017，TP4，MoE 權重 MXFP4，DSpark 投機解碼 |
| 程式碼 | UnieInfra mi350p-final 839d4cdf（三者相同） |
| 7.2.4（基線） | ts-dev 容器，torch 2.14.0+rocm7.2 自帶的 HIP／RCCL／hipBLASLt／rocBLAS／comgr |
| 7.14.1 | 同一容器、同一份 Python 環境，ROCm 函式庫換成主機 `/opt/rocm/core-7.14` 的版本 |
| 10.0 | 同上，函式庫換成 `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261021` 映像裡的版本 |

### 11.3 新版 ROCm 環境怎麼建

UnieInfra 需要 torch 2.14，但新版映像只有 torch 2.11。為了只改一個變數，我們沒有換 torch，而是做疊加層（overlay）：複製一份 torch 套件，把它自帶的 ROCm 函式庫換成新版，用 `PYTHONPATH` 與 `LD_LIBRARY_PATH` 讓它優先載入。腳本：`unieinfra/.rocm714/mk_overlay.sh`、`unieinfra/.rocm10/mk_overlay.sh`。

1. 複製新版的 `lib/*.so*`、`lib/hipblaslt`、`lib/rocblas`、`lib/rocm_sysdeps/lib`、`lib/llvm/lib` 的 `libLLVM.so.23.0git` 與 `libclang-cpp.so.23.0git`。
2. torch/lib 裡這些函式庫換成新版：libamdhip64、libhsa-runtime64、librccl、libhipblaslt、libhipblas、librocblas（含 kernel 目錄）、libamd_comgr、libhiprtc、librocprofiler-register、librocprofiler-sdk、libroctracer64、libroctx64、libhsa-amd-aqlprofile64、librocm_smi64、librocm-core。只換一部分會因兩套 HIP runtime 同時載入而 segfault。
3. **必須一起複製 `.kpack`**（`{blas,rccl,rand,fft}_lib_gfx950.kpack`）。TheRock 把 GPU code object 另外打包；少了它時只會看到 RCCL `invalid device function`，要開 `AMD_LOG_LEVEL=2` 才看得到 `kpack_load_code_object failed with error: 13`。
4. 上游 triton 改用 ROCm 10 映像的 3.8.0+rocm10。原本的 triton 3.8.0 靜態連結自己的 LLVM，和新版 comgr 動態載入的 libLLVM 23 衝突，在 `llvm::DenseMap` 靜態初始化時 segfault。UnieInfra 自己的 Gluon／Triton kernel 用 tokenspeed-triton（自帶 LLVM），三者相同。
5. torch 本身的 kernel 仍是 7.2 編譯器產生的 code object（載入時約 100 筆 `register fat binary failed`；常用 op、CUDA graph、torch.distributed 都測過可正常運作）。

**限制**：這是「torch 2.14（ROCm 7.2 編譯）＋ 新版 runtime 與函式庫」的組合，不是官方支援的原生建置。最乾淨的對照需要以 ROCm 7.14／10 原生編譯的 torch 2.14。

### 11.4 測試方法

**端到端**

- 負載：AgentX（agentx-harness 1.0.13 + AiPerf，資料集 semianalysis_cc_traces_weka_062126、seed 42）壓力模式（`--ignore-trace-delays`），量測 600 s，prompt 中位數約 83K tokens。指標：每分鐘完成請求數（RPM）、輸出 tok/s、ITL、TTFT。
- 三者設定完全相同，只差 `PYTHONPATH`／`LD_LIBRARY_PATH`：`TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 TOKENSPEED_DSPARK_COUPLED_DRAFT=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_IRIS_Q8_ALLREDUCE=0 CHUNKED_PREFILL_SIZE=8192 MAX_TOTAL_TOKENS=1073741824 GPU_MEMORY_UTILIZATION=0.86`，`--sampling-backend triton_full`。running-request 上限 c4／c16 為 32、c64 為 128。
- 排程：新版與 7.2.4 同一天交錯跑（新版 → 7.2.4 → 新版 → 7.2.4），每次重啟 server、冷 cache 開始。7.14 與 10.0 各自有同輪的 7.2.4 基線（腳本 `scripts/stress_r714.sh`；`stress_r10.sh`、`stress_r10b.sh`、`stress_r10c.sh`）。
- 每格各一次（n=1）。同一設定 run-to-run 差異約 ±10%（同日 7.2.4 c16 三次：92.4、83.3、81.3 RPM）。

**單元件 microbenchmark**（同一 Python 環境，各跑兩次取平均）

- `unieinfra/.rocm10/bench/micro2.py`（單 GPU）：eager kernel launch（`x.add_(1)` 連續 2 萬次）、50 個 eager op 後同步、50 節點 HIP graph replay＋同步、6 種 BF16 GEMM（hipBLASLt）、256 MiB 讀寫頻寬。
- `bench/ar4.py`（4 GPU）：torch.distributed（RCCL）BF16 all_reduce，16 KiB–64 MiB。

**長 prompt prefill**（`scripts/rocm_prefill.sh`）

- 每種 runtime 起一次 server（c4 設定），送冷 prompt：約 150K tokens × 3 輪、約 600K tokens × 2 輪（每輪不同內容，max_tokens 1），量每輪延遲。
- 每種 runtime 都測 `TOKENSPEED_PREFILL_TBO=1` 與 `=0`，用來確認 prefill 的通訊／計算重疊在新版上是否仍有效（這個重疊依賴 stream priority 把通訊 stream 分到另一條硬體佇列）。

**無 profiler 的 GPU 計時**（`scripts/rocm_prefill_t.sh`，見 11.7 節）

**其他檢查**

- 投機解碼接受長度與每請求 decode 速度：從 server log 的 `Decode batch` 行取 `avg_accept_len` 與 `gen throughput`。
- 執行期 JIT 編譯：從 server log 的 compile monitor 行計數。

### 11.5 結果

**表 11-1 端到端 RPM（括號為對同輪 7.2.4 的差異）**

| 並行 | 7.2.4（7.14 輪） | 7.14.1 | 7.2.4（10.0 輪） | 10.0 |
|---|---|---|---|---|
| c4 | 50.0 | 38.9（−22%） | 52.8 | 48.8（−8%） |
| c16 | 81.3 | 72.2（−11%） | 92.4 | 65.2（−29%） |
| c64 | 159.0 | 150.2（−6%） | 148.7 | 151.0（+2%） |

**表 11-2 端到端延遲（新版 vs 同輪 7.2.4）**

| 並行 | 版本 | ITL 平均 | ITL p99 | TTFT |
|---|---|---|---|---|
| c4 | 7.14 | 21.6 vs 19.2 ms | 470 vs 326 ms | 1.50 vs 1.52 s |
| c16 | 7.14 | 23.4 vs 19.4 ms | 230 vs 135 ms | 1.31 vs 1.21 s |
| c64 | 7.14 | 46.8 vs 39.2 ms | 214 vs 112 ms | 1.40 vs 0.86 s |
| c4 | 10.0 | 18.3 vs 17.5 ms | 259 vs 359 ms | 1.11 vs 0.92 s |
| c16 | 10.0 | 24.0 vs 19.9 ms | 150 vs 135 ms | 1.38 vs 1.13 s |
| c64 | 10.0 | 44.5 vs 43.4 ms | 130 vs 167 ms | 0.99 vs 1.17 s |

**表 11-3 單 GPU microbenchmark**

| 項目 | 7.2.4 | 7.14.1 | 10.0 |
|---|---|---|---|
| eager kernel launch | 4.6 µs | **4.0 µs** | 4.7 µs |
| 50 個 eager op＋同步 | 252 µs | **231 µs** | 265 µs |
| 50 節點 graph replay＋同步 | 100 µs | 95 µs | 97 µs |
| GEMM 16×7168×2048 | 18.9 µs | 15.5 µs | 17.1 µs |
| GEMM 64×7168×2048 | 24.4 µs | 24.5 µs | 18.5 µs |
| GEMM 256×7168×2048 | 35.8 µs | 27.4 µs | 25.5 µs |
| GEMM 4096×7168×2048 | 178 µs | 168 µs | 169 µs |
| GEMM 64×4096×7168 | 27.2 µs | 20.4 µs | 21.4 µs |
| GEMM 8192×4096×7168 | 742 µs | 704 µs | 700 µs |
| HBM 讀寫（256 MiB） | 3,070 GB/s | 3,071 GB/s | 3,073 GB/s |

**表 11-4 4-GPU RCCL all_reduce（BF16）**

| 大小 | 7.2.4（兩輪） | 7.14.1 | 10.0 |
|---|---|---|---|
| 16 KiB | 33.2–33.3 µs | 26.0 µs | 25.8 µs |
| 64 KiB | 47.4–47.8 µs | 39.4 µs | 39.0 µs |
| 256 KiB | 104–106 µs | 103 µs | 103 µs |
| 1 MiB | 65.0 µs | 64.8 µs | 65.0 µs |
| 4 MiB | 170–218 µs | 169 µs | 169 µs |
| 16 MiB | 610–669 µs | 609 µs | 611 µs |
| 64 MiB | 2,388–2,488 µs | 2,461 µs | 2,511 µs |

**表 11-5 冷 prompt prefill 延遲（每輪秒數，c4 server、無其他負載）**

| runtime | TBO | 150K × 3 輪 | 600K × 2 輪 |
|---|---|---|---|
| 7.2.4 | 開 | 3.8／3.7／3.7 | 16.9／17.0 |
| 7.2.4 | 關 | 4.6／4.6／4.6 | 20.4／20.3 |
| 7.14.1 | 開 | **6.2**／4.2／4.1 | 18.7／17.8 |
| 7.14.1 | 關 | 4.9／4.7／4.6 | 21.6／**28.1** |
| 10.0 | 開 | 3.7／3.7／4.0 | 18.2／17.3 |
| 10.0 | 關 | 4.6／4.8／5.0 | 21.0／22.8 |

**表 11-6 其他檢查（c16）**

| 項目 | 7.2.4 | 7.14.1 | 10.0 |
|---|---|---|---|
| 投機解碼平均接受長度 | 3.74–3.77 | 3.76 | 3.74 |
| 每請求 decode 速度（gen tok/s ÷ running-req） | 74.2–74.6 | 74.8 | 71.3 |
| 執行期 JIT 編譯 | 236 次／1.29 s | 236 次／1.27 s | — |

### 11.6 解讀：為什麼新版比 7.2.4 慢

**可以確定的**

- GEMM（hipBLASLt）、RCCL、kernel launch、HIP graph replay、HBM 頻寬都不是退步來源：新版持平或更快。
- 投機解碼的接受長度相同，所以取樣 kernel（上游 triton 換版）沒有改變行為；每請求 decode 速度相同，所以 decode 本身沒有變慢。
- prefill TBO 在新版上仍有效（開比關快 13–20%），stream priority 分配硬體佇列的行為沒有失效。
- 停頓在 GPU 端、不在主機端（11.7 節）。
- UnieInfra 的 TP all-reduce 主要走自寫的 Iris（GPU 直接讀寫對方記憶體），不走 RCCL，所以新版 RCCL 的改善幾乎用不到。

**觀察到的差異**

- **prefill 時間不穩定**：7.2.4 同一設定每輪差異 ≤ 0.1 s；7.14 有一輪 +50%（150K 第一輪 6.2 s）、一輪 +30%（600K 21.6 → 28.1 s）；10.0 每輪差異 0.3–1.8 s。穩定時新版也慢 0–10%。
- 端到端的退步型態與此一致：每請求 decode 速度不變，但同時在 decode 的請求變少（c16 平均 19.8–20.0 vs 21.1–21.8 個）、TTFT 變長、ITL p99 變大，代表更多時間卡在 prefill。

**推論（未證實）**

1. **新版 runtime 的記憶體配置行為不同**：例如 fine-grained／coarse-grained 的選擇、page 大小、TLB、scratch 配置與回收（我們以 `HSA_NO_SCRATCH_RECLAIM=1` 關閉回收；新版是否仍遵守）。這能解釋「同樣的 kernel 在同一個 server 裡逐漸變慢」。
2. **時脈或功耗管理不同**：同樣能造成 GPU 端逐漸變慢；需在執行中記錄時脈、功耗、溫度才能和第 1 點分開。
3. **UnieInfra 的主要 kernel 不在 microbenchmark 範圍內**：MXFP4 MoE、DSv4.1 稀疏注意力、Iris all-reduce 都是 tokenspeed-triton 編出的 Gluon kernel，交給新 runtime 載入與 launch，可能在 kernarg（`HIP_FORCE_DEV_KERNARG=1`）、fine-grained／IPC 記憶體上有行為差異，尚未逐 kernel 比對。
4. **雜訊**：端到端每格 n=1，同設定本身約 ±10%；但兩個新版本九格中有七格變慢，方向一致。

**觀察到的 runtime 問題（可回報給 AMD）**

- **ROCm 7.2.4 的 HIP graph 連續 replay 會 segfault**：同一個 50 節點 graph（`x.add_(1)`）連續 replay 約 2,000 次、中間不同步，torch 2.14+rocm7.2 在 `CUDAGraph.replay` 內 segfault；HIP 7.15 正常。
- **TheRock 的 kpack 依賴**：函式庫與 `.kpack` 分開時沒有清楚錯誤，只看到 `invalid device function`。
- **LLVM 符號衝突**：新版 comgr 以全域方式載入 libLLVM 23，任何靜態連結 LLVM 的擴充模組（例如上游 triton 3.8.0 的 libtriton.so）都會在靜態初始化時 crash。
- **production 在 c128 不穩**（7.2.4，與換版無關）：兩次 c128 分別在 indexer kernel 報 `hipErrorUnknown`、在 prefill attention OOM（GPU1 申請 1020 MiB 時 0 bytes free，gmu 0.86）。

**要請 AMD 協助的事項**

1. 提供以 ROCm 7.14／10 原生編譯的 torch 2.14 gfx950 wheel，讓我們做乾淨的 A/B。
2. 從 ROCr 1.x（7.2.4）到 TheRock（7.14／10.0），記憶體配置（fine-grained／coarse-grained、page 大小）、scratch 配置與 `HSA_NO_SCRATCH_RECLAIM`、code object 延遲載入有沒有行為變化？長時間 eager prefill 中 kernel 逐漸變慢 10–30%、偶發整步停頓 2 s，有沒有已知原因？
3. 對 Iris 這類跨 GPU 直接讀寫（IPC 記憶體＋GPU 輪詢）的通訊方式，新版對 fine-grained 記憶體或 PCIe peer access 有沒有行為改變？
4. 7.2.4 的 graph replay segfault 是否為已知問題、哪一版修正？

**決定**：production 維持 ROCm 7.2.4。兩個 overlay 保留在 `unieinfra/.rocm714`、`unieinfra/.rocm10`，可隨時重建重測。紀錄：experiments.json #254、#257、#258。

### 11.7 停頓位置：GPU 端（無 profiler 計時）

**方法**（`scripts/rocm_prefill_t.sh`）：7.2.4 → 7.14 → 7.2.4 → 7.14 交錯，各起一次 server（c4 設定、TBO 開、production 旗標），依序送 3 個約 150K 與 3 個約 600K token 的冷 prompt（每個之間停 8 s）。開 `TOKENSPEED_GPU_STEP_TIMING=1`：每個 forward 前後在執行 stream 上各記一個 event，每 6 秒回報一次。回報內容：

- 每個 forward 的 GPU 時間（`extend`）：主機送得慢也會讓它變長。
- 主機送出這一步花的時間（`extend.step_host`）。
- forward 之間的 GPU 閒置（`gap_ms`）。

這個方法不用 torch profiler，因為 profiler 會改變 ROCm 的 graph 提交方式（第 12.3 節）。

**表 11-7 冷 prompt 延遲與 extend 的 GPU 時間**

| 執行 | 150K（3 個） | 600K（3 個） | extend GPU 總和 | 超出各窗口中位數的時間 |
|---|---|---|---|---|
| 7.2.4 第一次 | 3.8／3.7／3.7 s | 16.9／16.9／16.8 s | 56.1 s | ≤ 0.03 s |
| 7.14 第一次 | 3.8／3.7／3.7 s | 16.8／16.9／16.9 s | 56.2 s | ≤ 0.04 s |
| 7.2.4 第二次 | 4.0／3.8／3.7 s | 16.9／16.9／16.8 s | 56.4 s | ≤ 0.12 s |
| 7.14 第二次 | 3.8／3.7／3.7 s | 17.1／17.9／**23.1** s | **63.8 s** | **2.25 s** |

**表 11-8 每個 6 秒窗口的 extend GPU 時間中位數（ms，依時間排列，只列 ≥ 10 步的窗口）**

| 執行 | 前段 | 600K 第 2 個 | 600K 第 3 個 |
|---|---|---|---|
| 7.2.4 第一次 | 186 200 192 192 201 236 258 | 201 233 256 | 200 233 |
| 7.14 第一次 | 195 194 192 200 234 258 | 204 235 258 | 201 234 |
| 7.2.4 第二次 | 193 193 195 201 234 257 | 202 234 257 | 201 234 |
| 7.14 第二次 | 194 193 192 203 236 258 | 205 **253 277** | **265 276 300** |

**觀察**

- 7.14 的退步是**偶發**的：第一次與 7.2.4 完全相同，第二次後段才出現。7.2.4 四次執行同位置的中位數誤差都在 ±3 ms 內。
- 退步在 **GPU 端**：同樣的 kernel 每一步逐漸變慢 10–30%，另有一步在 GPU 上多花約 2.0–2.3 s。這段時間主機送出每一步的時間一直是 127 ms，forward 之間的 GPU 閒置也只有約 5 ms。
- 那次 2 s 停頓在 **4 個 rank 的同一個窗口同時出現**：TP 下只要一張卡停住，其他卡就在 all-reduce 裡等它，所以看不出是哪一張卡先停。
- 這與端到端的現象一致：decode 速度不變，但長 prefill 偶發變慢，連帶拉長 TTFT 與 ITL p99。

**下一步**：在跑的同時用 `amd-smi metric` 記錄每張卡的時脈、功耗、溫度，把「記憶體配置」與「時脈／功耗」兩個方向分開。紀錄：experiments.json #258。
## 12. 負面結果

下列嘗試在微基準或 trace 上有效，但端到端沒有增益，或有明確的反效果。每一項都有量測數字。

**表 7 未採用的方法與原因。**

| 方法 | 量測 | 原因 |
|---|---|---|
| Attention DP4 + TP MoE | 64 人 58.3 vs 21.6 ms/token | 複製權重後的 GEMM 形狀慢 3.3×；各 DP rank 的事件迴圈不同步 |
| Expert parallelism（EP） | 同形狀 RPM −27%（cap 32）、−33%（cap 128） | PCIe 上 all-to-all 跨 socket 慢；TP-MoE 較佳 |
| decode 兩 micro-batch 重疊 | MoE 拆半 1.65×、GEMM 1.6×，重疊實測為負 | decode 受權重頻寬限制，拆半等於讀兩次權重；即使 all-reduce 全藏住每步仍慢約 7 ms |
| INT8 decode all-reduce | 64 人 23.3 vs 21.6 ms/token | 延遲受限；拆掉 fused mHC post 反而變慢 |
| RCCL 取代 Iris（PCIe 自動切換） | n=1 155 vs 202 tok/s | 小批次明顯較差 |
| Iris program cap 32（全域） | 155K prefill 5.5 → 6.8 s | prefill 損失大於 decode 增益；改為依大小切換 |
| DSpark 只驗證前 3 個草稿 | decode bench −12%；AgentX ±6% 正負混雜 | 端到端無一致增益 |
| aiter MoE（修掉碎片、調 128 CU 參數） | 重複 A/B 與 gluon 打平 | 頻寬受限，調參僅 t16 −7%；多一份 vendor 依賴 |
| prefill MoE 設定掃描（tile／warps／XCD swizzle） | 端到端皆在 ±5% 內 | 瓶頸在 scale 載入方式，不在設定；改為重寫 kernel（9.6 節） |
| stage2 reduce 融合進 GEMM / hybrid atomic | M=4096 1685 µs / 435–460 µs（現行 426） | 更慢 |
| 6-row 去重 selected attention | 384 rows 115 vs 115 µs | 重複 dequant 不是瓶頸 |
| rows 注意力 tile split | 192 rows 66 → 69–79 µs | 更慢 |
| MoE e4m3 activation | 慢 3–10%（相對誤差 0.047 vs e2m1 0.213） | 較準確但較慢；精度發現另記 |
| forward 結果延後交出（主機空檔） | trace 每步 51 → 47 ms；AgentX c32 114.6 vs 114.45、c64 −3% | 純 decode 的改善未轉成混合負載吞吐 |
| post_process 快路徑 | 每步省約 50 µs | profiler 把 Python 成本放大 35–40×，真正成本很小 |
| hold + Iris small cap 8 MB 組合 | trace decode 步 −8%；AgentX c16 +1.2%、c32 −0.8%、c64 −3.7% | 12.3 節證實 trace 上的主機空檔多為 profiler 假象 |
| TBO 門檻 1024 rows + Iris small cap 8 MB | c32／c64 各 2 次，雜訊內 | 1K–4K rows 的 prefill 切半後 GEMM 太小 |
| `--enable-mixed-batch`（prefill 不縮，未保留 TBO） | c32／c64 無增益（−2～−4%） | decode 搭便車的收益被 prefill 輪變長抵消；保留 TBO 後見第 8 節 |
| GIL switch interval 500 µs | c32 −31%、c4 −17% | 主執行緒工作被切碎 |
| 強制 HIP graph packet 提交（`DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=2`） | graph launch 40.6 vs 39.3 ms、步時不變 | 「複雜 graph 退回逐節點提交」（clr classic RunNodes）的假說不成立 |
| `DEBUG_HIP_FORCE_GRAPH_QUEUES=1`（graph 只用一條 queue） | 無 profiler：launch host 7.3 → 0.5 ms，但 decode 步 GPU 45.2 → 55.3 ms、RPM 128.5 → 122.0 | graph 內平行分支被序列化 |
| `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=2`（± 單一 queue，無 profiler） | decode 步 49.0／47.4 ms、RPM 118.4／128.5 | 雜訊內或更差 |
| decode graph 內 RCCL 改 Iris | 9 個 ncclDevKernel 共 1.26 ms／步，其中 0.9 ms 是等其他 rank | 最多省約 0.3 ms／步；兩 rank microbench 證實 RCCL 不觸發慢路徑；未實作 |
| prefill graph 擴到 8,192 tokens | c32 107.2–112.3 vs 108.0 | 雜訊內；eager 啟動空檔多為 profiler 放大 |
| shadow n-gram 草稿 | 命中時每步 1.78 tokens，DSpark 3.94 | 混合使用最多 +5% |
| ROCm 7.14.1／10.0 user-space（11/17） | 7.14：c4 −22%、c16 −11%、c64 −6%；10.0：c4 −8%、c16 −29%、c64 +2% | GPU 端偶發 prefill 變慢與停頓（第 11 節）；維持 7.2.4 |

### 12.1 量測教訓總覽

數個「主機端瓶頸」方向曾因量測工具被高估；以下三小節是細節。量測規則的完整清單見 3.4 節。

### 12.2 Python 追蹤放大主機成本

開啟 Python 追蹤的 profiler 顯示 AgentX c32 有 20% GPU 閒置、每步 post_process 3.2 ms；無 profiler 實測 post_process 僅約 80 µs，GPU-only trace 的閒置為 6–11%。數個「主機端瓶頸」方向因此被高估。

### 12.3 GPU-only profiler 也會改變 HIP graph 的提交方式（11/15）

即使只開 GPU activity，c32 trace 仍顯示 GPU 只有 82% 忙碌（閒置 1,077 / 6,072 ms），其中每個 decode 步一次的 zero_page_fields→HtoD 空檔共 175 ms；CPU+GPU trace 中 hipGraphLaunch 的主機耗時約等於 graph 的 GPU 跨度（中位 60.7 vs 60.5 ms），看起來像「launch 阻塞整步、下一步準備全排在 graph 結束後」。單一 queue 模式下更出現每 256 個節點約 6 ms 的空檔，4 個 rank 同時發生。

讀 ROCm 7.2.4 clr 原始碼（rocvirtual.cpp）後發現：profiler 錄製時，graph 內每個 AQL packet 都會掛上 completion signal（attachSignal = timestamp_ != nullptr），預設 signal pool 只有 64 個，用完就得新建或等舊的。256 正是 DEBUG_HIP_GRAPH_BATCH_SIZE，這解釋了 6 ms 空檔與 60–100 ms 的 launch 阻塞。

改用無 profiler 的 `TOKENSPEED_GPU_STEP_TIMING=1`（在 execution stream 上以 event 夾住每個 forward，每 6 s 輸出 decode／extend 的 GPU 時間、launch 主機時間與 forward 間閒置）重量 c32：decode 步 GPU 45.2 ms、launch 只佔主機 7.3 ms、forward 之間 GPU 閒置僅 0.13–0.19%，主機一直跑在 GPU 前面。結論：decode 不受主機限制；先前的 hold、GIL、host prep 等修法只是重新分配同一段主機時間，難怪吞吐不變。主機端唯一剩下的方向是 prefill 的 eager launch（extend 步主機 95 ms 對 GPU 124 ms）。

### 12.4 端到端 A/B 的漂移

最終 build 對 c4c64-wave 基準的 A/B 一度顯示 c4 −10%、c16 −8%。改以同一 build 只切 prefill MoE 旗標、OFF/ON/ON/OFF 交錯的無 profiler 對照後，decode 步 GPU 時間為 24.9 / 24.4 / 21.0 / 21.5 ms，只隨先後次序變化、與旗標無關；同一設定的 ITL 在 7.9–9.9 ms 之間。該 A/B 的差異是負載漂移，不是退化。

## 13. 穩定性與正確性

KV 依記憶體預算長滿時，約 5% 的啟動會在長 prefill 中以 `HSA_STATUS_ERROR_OUT_OF_RESOURCES` 崩潰；留約 6 GB/GPU 餘裕（gmu 0.86）後穩定，最終 build 的 c1–c128 掃描 0 次崩潰。

### 13.1 HSA OUT_OF_RESOURCES

- **共同點**：8 個崩潰的 abort 訊息都是 `Available Free mem : 0 MB`，都在 4K–16K new token 加數十萬到數百萬 cached token 的 chunked prefill 中；7/8 在 abort 前 ≤ 2 秒四個 rank 都出現 `HIPCachingAllocator ... OOM (free: 0)`。
- **機制**：AQL queue 的動態 scratch handler 配置失敗時回報這個錯誤。KV 預算 = profile 後可用記憶體 − 0.1×144 GB，這 14.4 GB 在 ready 前已被 graph、retraction pool（4 GB）、non-torch（約 8.8 GB）吃掉大半。
- **副因**：啟動腳本的 `-e HSA_NO_SCRATCH_RECLAIM` 沒帶值，docker 把 image 內建的 =1 刪掉，ROCr 因此會回收 scratch 後重配。修正後仍在 gmu 0.9、40.3M tokens 下崩潰，所以主因是記憶體用滿。
- **scratch 來源**：掃描執行期 kernel 快取（7,304 個 amdgcn），208 個編譯產物使用 scratch：`_mxfp8_block32_tiled` 最高 2,024 B/lane、`triton_mm_mxfp8_block32` 1,592、`selected_attention_rows` 932、`selected_attention` 760、`index_topk` 472、`gluon_mm_mxfp8` 432、`index_topk_select` 120。

| 設定 | 啟動次數 | 崩潰 |
|---|---|---|
| 依預算（2^30），gmu 0.9 | 94 | 5（5.3%） |
| 固定上限，gmu 0.95 | 121 | 3（2.5%） |
| 固定 33.5M，gmu 0.9 | 217 | 0 |
| 依預算，gmu 0.86 | c128 ×2 + 最終 build 6 格 | 0 |

### 13.2 aiter MoE 的記憶體碎片

舊的 preprocessor 整層 pad 再 shuffle，產生 1.26 GB / 1.26 GB / 0.63 GB×2 的暫存；釋放的大 block 被之後的小常駐 tensor 切走釘住，`empty_cache` 收不回，40 層後約 17.4 GB（單 GPU repro）。第一版修正用 `max_split_size_mb:32` 反而在真實 4-rank server 載入時 OOM；最後改成預配 padded buffer、每 32 個 expert 原地 shuffle、不在每層 empty_cache（a5ae4510），c128 連續啟動 3 次 4/4 rank 無 OOM，KV 36.9–37.6 GB（gluon 36.5）。

### 13.3 production 在 c128 的崩潰（11/17）

11/17 同日 production（mi350p-final、ROCm 7.2.4、cap 256、gmu 0.86）的 c128 基線跑了兩次都崩潰：一次在 indexer kernel 報 `hipErrorUnknown`，一次在 prefill attention OOM（GPU1 申請 1020 MiB 時 0 bytes free）。先前（11/16）同設定的 c128 跑過沒事；mixed+TBO 版本在 c128 沒有出事。原因尚未調查。

### 13.4 正確性驗證

- needle-in-haystack 6/6：每個 kernel commit、TBO 與最終 build 都有跑。
- GSM8K：T=0 的 CI 設定在 kernel 系列為 92.3–93.6%（門檻 90%）；TBO 90.7 vs 91.2%；最終 build（1,319 題）90.45%；mixed+TBO（同時有長 prefill 壓力）90.52%。T=1.0 設定用來驗 block verification（95.8 vs 96.2%）。
- prefill MoE kernel：reduce 模式與舊 kernel 逐位元相同，257 個測試通過；shard-local block verification 的 CPU 統計測試 TV 0.03–0.04（雜訊底）。
- mixed+TBO：`test_deepseek_v41_cache.py` 158 passed（含新測試）。
- prefix cache：三個 105K 文件的冷 / 續輪 / 重複請求檢查，續輪 TTFT 3.5 → 0.08 s、答案正確，重複請求輸出與冷請求相同。

## 14. 討論、限制與下一步

1. **c4–c64 未達 1.3×**（最終 1.11–1.28×）：此區間 KV 容量充足，結果取決於 decode 每步時間與 prefill 成本。decode MoE 已達實測頻寬 85–97%（9.4 節），decode 也不受主機限制（12.3 節），剩下的 decode 成本主要是 all-reduce（約 13% 牆鐘時間，延遲受限）。mixed+TBO 把 c64 再提高 18%（第 8 節）。
2. **非官方模式**：壓力模式移除了思考時間，用來衡量容量；官方模式下 c ≤ 64 的 RPM 差距較小。
3. **單機、單模型**：結論限於 PCIe 4×MI350P 與 DSv4.1；在有 xGMI 的機器上，通訊重疊的收益會較小。
4. **雜訊**：600 s 單格 ±10–15%，同一設定連續跑也會漂移 ±10%（12.4 節）；最終 build 每格只跑 1 次，表 6 的倍數應以 ±10% 解讀。
5. **記憶體餘裕**：KV 依預算長滿時有約 5% 的崩潰率，目前以 0.86 迴避（13.1 節）；c128 在 11/17 仍出現兩次崩潰（13.3 節）。
6. **量測工具**：GPU trace 只適合看 kernel 本身；主機／launch 相關的結論必須用無 profiler 的計時（12.3 節）。初版第 4 節與表 7 中以 trace 推得的主機空檔數字應視為上界。
7. **ROCm 版本**：production 停在 7.2.4；新版的退步原因待與 AMD 釐清（第 11 節）。

依把握程度排序的下一步：

1. **decode all-reduce（約 13% 牆鐘時間）**：壓縮、幾何、融合、重疊都試過；剩下的是更低延遲的同步設計（例如 vLLM nvidia/specialized_decode 的 Lamport 式 all-reduce，把 mHC post / RMSNorm / MXFP8 量化一併融合），預估每步 3–5 ms。
2. **prefill eager launch**：extend 步主機 95 ms 對 GPU 124 ms，主機已逼近 GPU；TBO 使 kernel 數加倍（約 95 kernels/層/chunk）。方向是融合小 kernel、減少每層主機工作，或把 TBO prefill 收進 graph。
3. **mixed+TBO 設為 c ≥ 32 的預設**（第 8 節）。
4. **記憶體餘裕根治**：讓 KV 預算計入 scratch，或在 KV 配置前預先配置每個佇列的 scratch，才能回到 gmu 0.9；並調查 13.3 節的 c128 崩潰。
5. **ROCm 新版退步**：執行中以 `amd-smi metric` 記錄時脈、功耗、溫度，分辨記憶體配置與時脈／功耗兩個方向（11.7 節）。
6. **atomic／reduce 門檻**：prefill MoE 在 M=2560–3584 改走 atomic 每層可省 18–31 µs，但失去 determinism，需決定是否接受。
7. **需使用者決定**：INT8 prefill all-reduce（有損，先前 c32 +12%、c128 +28%）。

## 15. 結論

在 GPU 間只有 PCIe 的 MI350P 上，長上下文 agent 推論的效能由三件事決定：KV 容量（決定高並行度的前綴命中率）、每步接受的 token 數（決定 decode 吞吐），以及 all-reduce 能否被計算覆蓋（決定 prefill 成本）。依注意力結構分組的 KV cache 讓 c128 達 5.88×；整塊驗證（含 shard-local 版本）與 prefill 兩 micro-batch 重疊讓 c16 達 1.28×、c64 達 1.22×；mixed batch 保留 TBO 讓 decode 不再被 prefill 卡住，c64 再 +18%、ITL p99 降一半以上；prefill MoE 重寫在 kernel 層有 8–30% 的收益，但端到端被 ±10% 的漂移蓋過。decode 的 MoE 已達頻寬上限、主機也不是瓶頸，c4–c64 要再往 2× 前進，下一步是 decode all-reduce 的延遲與 prefill 的 launch 開銷。ROCm user-space 升級到 7.14／10.0 目前反而變慢，原因在 GPU 端，已整理給 AMD。
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
# prefill MoE v2（9.6 節）預設開啟，無需設定；設 0 可退回舊 kernel：
#   TOKENSPEED_MOE_PREFILL_S1_V2 / _S2_V2 / _S2_REDUCE_V2
CHUNKED_PREFILL_SIZE=8192  MAX_NUM_SEQS=2*CONC  MAX_TOTAL_TOKENS=2^30
GPU_MEMORY_UTILIZATION=0.86（全部並行度）
--sampling-backend triton_full --enable-log-request-stats
# 建議（11/17，第 8 節）：c ≥ 32 另加 --enable-mixed-batch（需 mixed-tbo 分支）
# 量測用（預設關）：TOKENSPEED_GPU_STEP_TIMING=1
```

壓力模式 c=CONC 的完整啟動指令：

```bash
STRESS=1 TOKENSPEED_PREFILL_TBO=1 TOKENSPEED_DSPARK_BLOCK_VERIFY=1 \
TOKENSPEED_DSPARK_BLOCK_VERIFY_TP=1 TOKENSPEED_DSPARK_BLOCK_VERIFY_CHUNK=64 \
TOKENSPEED_DSPARK_COUPLED_DRAFT=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1 \
HIP_FORCE_DEV_KERNARG=1 TOKENSPEED_IRIS_Q8_ALLREDUCE=0 \
CHUNKED_PREFILL_SIZE=8192 MAX_NUM_SEQS=$((2*c)) MAX_TOTAL_TOKENS=1073741824 \
GPU_MEMORY_UTILIZATION=0.86 \
EXTRA_SERVE_ARGS="--sampling-backend triton_full --enable-log-request-stats" \
scripts/run_round.sh <tag> 600 unieinfra "$c"
```

- `HIP_FORCE_DEV_KERNARG=1` 是 vLLM 映像的預設，我們原本沒設（decode +4–8%）。
- `HSA_NO_SCRATCH_RECLAIM=1` 由啟動腳本預設帶入。
- prefill MoE v2 三個旗標預設為 1，不需設定。

## 附錄 B：迭代時間線

十三天的工作可以分成十二個階段：先讓引擎穩定、再把 kernel 追平、再用容量拉開 c128，中途兩次發現比較方法有誤，再靠 decode 的接受長度與 prefill 通訊重疊把 c4–c16 拉到 1.3×；之後做 c4–c64 的收尾與最終 build 驗收，最後補上 mixed+TBO 與 ROCm 版本對照。

| 日期 | 階段 | 關鍵結果 |
|---|---|---|
| 11-04–05 | baseline 與崩潰修正 | c4 全部 rank SIGSEGV；根因是 indexer 寫死 32 MiB 預算，修正後 overlap 恢復；Iris 接管 prefill all-reduce |
| 11-05–06 | 排程與移植 | c128 因 prefill-first 排程餓死 decode；decode-every-N、cache-aware admission；從 dsv41 分支補回合併時遺失的 prefill 優化 |
| 11-06–07 | kernel 追平 | indexer 4–5×、rows attention、16-head padding、block32 GEMM、CU／XCD 幾何；64 人 decode 30.0→21.6 ms/token（= vLLM） |
| 11-07 | KV 容量 | KV 受 token 上限而非記憶體限制；16.7M→33.5M，c128 3× |
| 11-08 | 取樣公平性 | greedy 忽略 temperature；改取樣後 c4–c64 為 0.82–0.91×；coupled drafting 補回 2–12% |
| 11-09–11 | 公平性與 best-vs-best | DP attention、EP 否決；關掉 INT8 通訊；官方模式壓不到引擎，改用壓力模式 best-vs-best |
| 11-12–13 | 整塊驗證與量測校正 | block verification、分塊與批次門檻、KV 依預算；aiter／host-gap／prefill graph 三輪 A/B 無端到端增益；GPU-only profile 指出 all-reduce 佔 37% |
| 11-14 | prefill 通訊重疊 | stream priority 是關鍵；c16 +26.6%、c32 +10.7%；OUT_OF_RESOURCES 根因與 0.86 餘裕 |
| 11-14–15 | c4–c64 收尾 | shard-local 整塊驗證；MoE roofline（decode 已達 85–97%）；prefill MoE 重寫 M=4096 −8%、M=8192 −30%；hold／E2／mixed batch／MoE tuning 皆在 ±5% 雜訊內 |
| 11-15–16 | 主機空檔釐清與最終 build | 「hipGraphLaunch 阻塞」為 profiler 假象；mi350p-final c1–c128 為 1.67／1.19／1.28／1.11／1.22／5.88×；GSM8K 90.45%、needle 6/6 |
| 11-16–17 | c4 上限、query-parallel、干擾 | c4 running-request 上限 8→32（UnieInfra 44.3→51.9，vLLM 37.3→45.9）；V4.1 query-parallel 否決；c4 GPU 87% 時間在 decode，prefill 不受 decode 干擾 |
| 11-17 | mixed+TBO 與 ROCm 對照 | mixed batch 保留 TBO：c64 +18%、ITL p99 降 31–79%；ROCm 7.14／10.0 端到端變慢，定位為 GPU 端偶發停頓 |

（互動圖見 Claude Doc：各階段 c4–c64 對 vLLM 的倍數；c128（5.5–5.7×）未畫入。）

## 附錄 C：分支、PR、工具與資料

### C.1 分支與 commit

| 分支 | 內容 | 狀態 |
|---|---|---|
| `mi350p-final` | 39 個 commit：下列 clean 版 + prefill TBO（6361be94）+ shard-local BV（49c86553）+ prefill MoE（62255315、dba84259、761b8d08）+ 硬體適配 skill（188c0c52）+ GPU_STEP_TIMING（839d4cdf） | 已 push 到 UnieAI/UnieInfra，PR #44 |
| `roy/mixed-tbo` | mixed batch 保留 prefill TBO（217bca75、b9c0368f），基於 mi350p-final | 11/17 push，待決定是否併入 |
| `mi350p-dsv41-perf` | 31 個 commit：gfx950 kernel、Iris、排程、coupled drafting、block verification（到 23b4d606） | 乾淨版本 `mi350p-dsv41-perf-clean`（f6ecc3f4）已併入 mi350p-final |
| `prefill-tbo2` | prefill 兩 micro-batch 通訊重疊（6361be94） | 已併入 mi350p-final |
| `moe-aiter-fix` | aiter MoE backend 與記憶體修正（ffbf0865、bbb32bee、a5ae4510） | opt-in，無增益，不併 |
| `hostfix-zero-pages` | forward 結果延後交出（4a3a6435） | 端到端無增益，不併 |
| `scheduler-postproc-fast` | post_process 快路徑（36101815） | 每步約 50 µs，不併 |
| `decode-attn2`、`decode-kernels2` | rows split、草稿注意力、radix top-k、e4m3 MoE | 大多否決；草稿注意力與 blocks top-k 可取用 |

11/17 另把其餘 30 個實驗分支以 `roy/<分支名>` 推到 UnieAI/UnieInfra 保存。

mi350p-final 除 prefill MoE kernel 外，所有改變行為的功能都預設關閉、以環境變數開啟：

- gfx950 kernel：indexer、selected attention、block32 MXFP8 GEMM、CU／XCD 幾何、prefill MoE 重寫。
- 通訊：PCIe 主機上 prefill 走 Iris all-reduce；選用的 INT8／FP8 two-shot（預設關）。
- 排程與 prefill：decode-every-N、cache-aware admission、prefill 兩 micro-batch 重疊、32 列壓縮器尾巴等。
- 投機解碼：coupled drafting、整塊驗證（分塊、批次門檻、shard-local）、first-k 驗證。
- 工具與文件：`TOKENSPEED_GPU_STEP_TIMING`、`.skills/unieinfra-hw-optimize`。

### C.2 硬體適配 skill

`.skills/unieinfra-hw-optimize` 把本報告的流程寫成可重現步驟（bring-up、正確性門檻、基準測試、負載下剖析、槓桿目錄、A/B 雜訊控制、改 kernel 前先量 roofline、紀錄與 PR、常見陷阱），附 19 支腳本與模板，供換新硬體或交給其他模型照做。

### C.3 腳本與資料

- 實驗紀錄：`experiments.json`（每次迭代一筆，含否決項，#1–#258）；dashboard：https://claude.ai/artifact/L5bkHnHRbzb2thttkZhHHg
- 互動版技術報告：[Claude Doc 技術報告](https://claude.ai/artifact/GZcbYZ7NVzFNCUJYnYv3XJ)
- AgentX 結果：`results/agentx/<engine>-<tag>-c<conc>-d<sec>-<time>/aiperf_artifacts/`（`profile_export.jsonl`、`profile_export_aiperf.json`）
- 主要腳本：`scripts/run_round.sh`、`run_agentx.sh`、`serve_unieinfra.sh`、`gap_check.py`（client 等待比例）、`agentx_trace_split.py`（依 forward 模式分段）、`prof_agentx_gpu.sh`（GPU-only trace）、`repeat_ab.sh`（交錯 A/B）、`prof_final.sh` + `sched_timing_summary.py`（無 profiler 計時）、`stress_final.sh`（最終 c1–c128 掃描）、`test_final.sh`（needle + GSM8K）
- 11/17 新增：`stress_mtbo2.sh`／`stress_mtbo3.sh`／`verify_mtbo.sh`（mixed+TBO 對照與 GSM8K）、`stress_r10*.sh`／`stress_r714.sh`（ROCm 端到端 A/B）、`rocm_prefill.sh`／`rocm_prefill_t.sh`（冷 prefill 與 GPU 計時）、`ab_table.py`（結果彙整）、`unieinfra/.rocm10/bench/micro2.py`、`ar4.py`（ROCm 單元件 microbenchmark）
- 主要 trace：`logs/prof-agentx-gpu-c32-*`（AgentX 負載）、`logs/prof-h2h-u/v-11112041`（與 vLLM 純 decode 對比）、`logs/prof-sched-*`（host 空檔與 graph 提交研究）
