# 90 — 分歧點彙總（證據底稿）

這一份把 `10-` 到 `60-` 六份文件裡的所有分歧點收成一張表。它是 `AGENTS.local.md` 流程第 1 步（決定做什麼）的輸入。
**行動計畫在 `70-PORTS.md`** —— 依「先 1:1 移植追平、再優化」的原則排序；本文件是它的
證據底稿，第 2 節的量測只用來建立移植前後的對照基線。

版本見 `00-METHOD.md`。所有主張都有 `file:line` 支撐，在來源文件裡。

---

## 1. 分歧點總表

「本引擎在哪一邊」欄：**孤例** = 五家只有 UnieInfra 這樣做；**少數** = 2 家；
**多數** = 3 家以上；**缺席** = UnieInfra 沒有這個東西。

| # | 分歧點 | 階段 | 誰跟誰同一邊 | 本引擎 | 預期成本方向 | 來源 |
|---|---|---|---|---|---|---|
| D1 | GDN prefill 每層 launch 數 9 vs 3 | prefill | **在 sm_120 上只有 TurboMind 是 3**；SGLang 的 3 是 SM100 專屬跑不了；UnieInfra / vLLM / SGLang(預設) 9 | 多數（但落後） | **大**：48 層 × 6 個多餘 launch × 每 launch 7.7–13.9 µs 地板 | `30-prefill.md` §2.1 |
| D2 | spec verify 的 GDN 走 prefill 路徑 | spec | SGLang 有專用 verify kernel；UnieInfra 直接轉呼叫 `forward_extend` | 孤例 | **大**：每個 verify step 付 432 個 launch | `50-spec-decode.md` §2.1 |
| D3 | hybrid 前綴匹配：折進 descent vs 事後補掛 | scheduler | SGLang 折進；UnieInfra 事後 `augmentMatch` | 孤例 | **大**：已知曾量到 0% vs 83.3% 命中率 | `20-scheduler.md` §2.4 |
| D4 | 引擎端重複 detokenize | API | 其他四家各做一次；UnieInfra 做兩次（Rust gateway + 引擎內嵌） | 孤例 | **中**：每個 output token 一次多餘的 HF decode | `10-api.md` §2.2 |
| D5 | 混批關閉時「這一步只做 prefill」 | scheduler | SGLang 同語意；vLLM / TRT-LLM 恆混；TurboMind 無此概念 | 少數 | **中**：影響 TPOT 尾延遲 | `20-scheduler.md` §2.1 |
| D6 | 搶佔挑 token 最多者 | scheduler | 其他四家挑最晚到達或依 policy | 孤例 | **中**：系統性懲罰長請求 | `20-scheduler.md` §2.3 |
| D7 | 兩層 HTTP（gateway + sidecar 代理） | API | 其他四家單層 | 孤例 | **小**（但會污染量測） | `10-api.md` §2.5 |
| D8 | 融合粒度：in_proj / conv+SiLU / L2norm | prefill | TurboMind 全融；其他家分開 | 多數（但落後） | **中**：每層 3 個可省的 launch | `30-prefill.md` §4.1 |
| D9 | 動態 grid 的雙路徑切換 | prefill | 只有 TurboMind 有 | 缺席 | **中**：varlen 不平衡時的長尾 | `30-prefill.md` §4.2 |
| D10 | GDN mixed batch 用兩條 stream | prefill | 只有 TurboMind 有 | 缺席 | **中**：prefill/decode 兩段可重疊 | `30-prefill.md` §4.3 |
| D11 | w4a16 是死碼（缺 Marlin GEMM） | quant | 其他四家都有可用的 4-bit GEMM | 孤例 | **大**（能力缺口，非效能問題） | `60-quant.md` §2.1 |
| D12 | 無 ngram drafter | spec | 其他四家都有，其中兩家還重寫成 C++/GPU | 缺席 | **中** | `50-spec-decode.md` §2.4 |
| D13 | 無樹狀 draft | spec | 其他四家都有 | 缺席 | **中**：影響 accept rate | `50-spec-decode.md` §2.3 |
| D14 | draft 階段無專用 CUDA graph | spec | 只有 SGLang 有三個專用 runner | 多數 | **小到中** | `50-spec-decode.md` §2.2 |
| D15 | 無 fp4 KV | quant | 只有 SGLang 有 | 多數 | **小**（容量題） | `60-quant.md` §1 |
| D16 | 容量准入與批次組成合併在一個迴圈 | scheduler | 只有 TRT-LLM 拆開 | 多數 | **小**（可維護性） | `20-scheduler.md` §2.2 |
| D17 | chunk 切分無 policy（等同 FCFS） | scheduler | 只有 TRT-LLM 有三種 policy | 多數 | **小到中**：TTFT 分布 | `20-scheduler.md` §2.5 |
| D18 | 串流不做 stop 前綴抑制 | API | 只有 SGLang 有 | 多數 | **小**（正確性/體感） | `10-api.md` §2.3 |
| D19 | OpenAI 協定層在編譯過的 Rust binary 裡 | API | 其他四家都在同一個 repo、同語言 | 孤例 | **不可量**（但擋住 profiler） | `10-api.md` §2.1 |
| D20 | 快照從哪個池子配 | scheduler | SGLang 獨立 int8 池；vLLM 同 KV 池但有統一 refcount 且排不進去就不排；UnieInfra 從**活池**配且 admission 不算它 | **孤例** | **大**：`failed to allocate Mamba checkpoint slot` 的根因 | `36-cache-tiering.md` §2 |
| D21 | 配不到 slot 時怎麼辦 | scheduler | vLLM 回傳 `num_gpu_blocks + 1` 讓 scheduler 不排它；UnieInfra 在 `SchedulePrefillEvent` 裡 throw | **孤例** | **大**：512 斷點剩下的那一半 | `36-cache-tiering.md` §4 |
| D22 | 快取狀態的精度 | quant | 只有 SGLang 量化成 int8（獨立池）；其餘同活狀態精度 | 多數（但容量少 4x） | **大**：~4x 快取容量 | `36-cache-tiering.md` §3 |

### 1.1 UnieInfra 結構**優於**參照引擎的地方（別改壞）

| # | 項目 | 內容 | 來源 |
|---|---|---|---|
| P1 | kernel registry 的 arch 閘同時有上下界 | `min_arch_version` **與** `max_arch_version` 都支援（`ops/attention/msa.py:96` 就用 `ArchVersion(10,0)`–`(10,3)` 把 kernel 圈在 SM100）。這正是 SGLang 缺的那一半：它用 `major >= 10` 放行 sm_120 進一個 SM100 kernel | `60-quant.md` §3.1、`30-prefill.md` §0.1 |
| P2 | decode 路徑的 GDN 已融合 | 2 launch/層（conv update + 單一融合 kernel），與 prefill 的 9 形成對比 | `40-decode.md` §2.3 |
| P3 | prefill 專用 breakable graph | 與 SGLang 同級，vLLM 只有 piecewise，TRT-LLM 只有 decode graph | `30-prefill.md` §3 |
| P4 | C++ scheduler 的 TP 決定性有明文保證 | 排序以 `Id()` 平手，理由寫在註解裡 | `20-scheduler.md` §2.6 |

---

## 2. 下一輪量測範圍

三個引擎實測（UnieInfra / vLLM / SGLang），TRT-LLM 與 TurboMind 不跑。
模型：`Qwen3-8B`（五方對照組）＋`Qwen3.5-27B`（深挖）。

### 2.1 第一優先 — 三個「大」的分歧點

| 量測 | 對應分歧 | 怎麼量 | 判準 |
|---|---|---|---|
| **M1 GDN prefill 逐 kernel** | D1、D8 | nsys 抓 `Qwen3.5-27B` 的一次 prefill，逐 kernel 表（名稱／次數／總時間／佔比／achieved GB/s）。兩組：UnieInfra、SGLang 預設(triton) | 9 個 launch 的實際佔比；作為移植後的對照基線 |
| **M2 spec verify 成本** | D2 | UnieInfra 開 MTP，比對 verify step 與純 decode step 的逐 kernel 表 | 若每 verify 多付 432 launch，spec decode 的淨效益要重算 |
| **M3 hybrid 前綴命中率** | D3 | 重複 prompt / 共享前綴 / 全新 prompt 三種形狀，量 UnieInfra vs SGLang 的實際 reuse 率 | 命中率本身就是結果，不需要換算成吞吐 |

**注意**：不要把 SGLang 的 `--linear-attn-prefill-backend cutedsl` 當成第三組。
它的閘會放行 sm_120，但 kernel 是 SM100 專屬（tcgen05 / tmem），在這張卡上跑不了 ——
見 `30-prefill.md` §0.1。唯一在 sm_120 上跑得動的融合 GDN prefill 在 TurboMind，
移植路線見 `70-PORTS.md`。

### 2.2 第二優先 — 分層歸因（跑一次，同時覆蓋多個分歧點）

沿用 `AGENTS.local.md` 第 0 步的探針階梯，三個引擎各跑一次：

| 探針 | 覆蓋的分歧點 |
|---|---|
| `GET /v1/models`（**gateway port 與 sidecar port 各一次**） | D7 |
| 1 token 進 / 1 出 | D4（開關 `enable_inline_detokenizer` 做 A/B） |
| 1 token 進 / N 出 | D4 |
| M token 進 / 1 出 | D1 |
| 同樣 M token 再送一次 | D3 |

D7 特別註記：**過去的 ingress 數字要重新確認量的是哪一跳。**

### 2.3 第三優先

| 量測 | 對應分歧 |
|---|---|
| 每步 scheduler CPU 時間（在途 1/8/32/128） | 「C++ scheduler 比較快」這個假設從沒被量過 |
| `--enable-mixed-batch` 開／關的 TTFT 與 TPOT | D5 |
| 長短混合工作負載下被搶佔的請求長度分布 | D6 |
| decode 逐 kernel 佔比（batch 1/8/32/128） | 確認 decode 打平不只在 batch 1 成立 |
| fp8 KV 三家對照 | D15 相關 |

### 2.4 不是量測題（是實作題）

- **D11 w4a16**：現在是死碼，量不了。要先決定 port TurboMind 的 u4 GEMM 還是接
  Marlin。注意 `60-quant.md` §2.2：TurboMind 的 GEMM 也沒有 sm_120 config。
- **D12 / D13 ngram 與樹狀 draft**：能力缺口。
- **D9 / D10 動態 grid 雙路徑、雙 stream**：要先有 D1 的數字才知道值不值得。

---

## 3. 量測開始前的固定檢查

每一條都是因為量錯過才寫下來的：

- token 數取 `usage.completion_tokens`，**不要數 SSE chunk**。
- 固定時長、關閉早停；震盪的 arm 用全程平均而非尾段。
- kernel 時間旁邊印 achieved GB/s；超過 HBM peak 表示working set 在 L2。
- 主機端數字不要引用 torch profiler 的 `cpu_op`（此機膨脹 3.6x），用獨立計時迴圈或
  `TOKENSPEED_LOG_MM_TIMING=1`。
- 空 Triton launch 在此機是 7.7–13.9 µs —— 任何「這個 launch 花 100 µs」都要對照它。
- C=1 的延遲不預測吞吐。
- 受測產物要來自一個 commit。

## 4. 環境備忘

| 用途 | 位置 |
|---|---|
| vLLM 執行環境 | `/home/ubuntu/workspace/roy/vllm/.venv`（vLLM 以 editable 指向 `/workspace/roy/vllm`，含 flashinfer / triton） |
| SGLang 執行環境 | `/home/ubuntu/sglang-venv`（0.5.16，需 `CUDA_HOME` 指向 venv 內 `nvidia/cu13`） |
| UnieInfra 執行環境 | 容器 `unieinfra-dev` |
| 既有 harness | `legacy/benchmark/`（`run_sweep.sh`、`launch/launch_{unieinfra,vllm}.sh`、`scripts/launch_sglang_host.sh`） |
| 結果落點 | `/workspace/roy/unieinfra-data/results/` |
