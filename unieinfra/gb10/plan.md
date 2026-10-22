# 在 GB10 上讓 TokenSpeed（TS）贏過 vLLM：研究計畫

更新：2026-10-20。範圍：單台 DGX Spark（GB10, sm_121），單 GPU，單機。
對照組：vLLM（主要）、SGLang（公開數字）。
基準模型：Qwen3.8-27B（BF16 GDN hybrid，52 GB，自帶 MTP 層）；第二階段加 MoE（Qwen3.6-35B-A3B，CI 已有）。
相關筆記：同目錄的 `kernel_zoo.md`（踩坑與優化候選，A–E 節；本計畫接在它的 E 節之後）與 `results/`（每次 level 2 的 `metrics.json` 摘要）。

---

## 0. 一頁結論

**GB10 的本質**：記憶體頻寬 273 GB/s（H100 的 1/12）、BF16 算力約 100 TFLOPS（H100 的 1/10）、CPU 和 GPU 共用同一池 128 GB。
所以 **decode 完全被頻寬綁住**（每個 step 都要把整份權重讀一遍），**prefill 被很低的算力綁住**。
這兩件事決定了所有優化的優先順序。

**今天的差距**：TS 在 GB10 上（BF16、無推測解碼）單流 4.3 tok/s；公開的 vLLM NVFP4+MTP 在同一顆模型約 25 tok/s，SGLang NVFP4+DSpark 約 34 tok/s。
差 6–8 倍，**全部來自「每個 token 要搬幾個 byte」和「每個 step 吐幾個 token」**，不是引擎架構。引擎再聰明也補不回 4 倍的 byte 數。

**所以計畫分三層，順序不能反**：

| 層 | 內容 | 目的 |
|---|---|---|
| L0 補齊（必要） | sm_121 上的 FP8 / NVFP4 權重 GEMM；MTP / DSpark 走 Triton 路徑 | 先把單流追平到 vLLM 的水準，否則後面的優勢在圖上看不見 |
| L1 放大（vLLM 在單顆小 GPU 上結構性弱的地方） | decode 不被 prefill 打斷；填滿每步「免費的 ~300 個 token 名額」（batching、tree speculation）；GDN state 成本；共用前綴只讀一次 | 讓 Pareto 曲線在 c=4–64 整段往右上推 |
| L2 統一記憶體原生（目前沒有引擎做） | 絕對記憶體預算、彈性 KV、不凍機、與其他租戶共存；零拷貝載入權重（482 s → 數十秒）；整池 prefix cache | 產品層的差異化：Spark 是開發機 / 邊緣機，這些是使用者每天撞到的痛 |

**判準**：README 慣例的 Pareto 圖（x = 每人 tok/s = 1000/TPOT，y = 總輸出 tok/s），同一台機器、同一個 `vllm bench serve`、同一個 GSM8K 門檻（≥ 0.90，CI 既有）之下，TS 曲線整段在 vLLM 右上方。

---

## 1. 機器定錨：幾個決定一切的數字

| 項目 | GB10 | 意義 |
|---|---|---|
| 記憶體頻寬 | 273 GB/s（LPDDR5X, 256-bit） | decode 一個 step 的下限 = 權重 bytes ÷ 273 GB/s |
| BF16 dense 算力 | 約 100 TFLOPS（實測 MAMF 99.8） | prefill 一個 token ≈ 2 × 參數量 FLOP，27B 模型 ≈ 54 GFLOP |
| FP8 dense | 約 190–210 TFLOPS（CUTLASS 實測 188） | 權重 FP8 → decode byte 減半、prefill 算力加倍 |
| NVFP4 GEMM | 實測 356 TFLOPS（CUTLASS sm_121a，大 shape） | byte 再減半、GEMM 再 ~1.9x；但 end-to-end 多數人只量到 1.2–1.5x FP8 |
| Shared memory / block | 101,376 B（跟 RTX 4090 一樣；B200 是 228 KB） | 為 sm_100 設計的 tile 全部塞不下；沒有 tcgen05 / TMEM / 2-SM MMA |
| Compute capability | 12.1（sm_121a） | 大多數函式庫只預編 sm_120，sm_121 常是最後被支援的；PTX 指令集有差（FP4 `cvt` 問題） |
| 記憶體 | 128 GB 共用，MemTotal ≈ 121.7 GiB，CUDA 看到 ≈ 119.7 GiB | 權重、KV、CUDA graph、JIT、OS、page cache、其他 container 全部搶同一池 |
| CPU | 20 核 Arm（10 X925 + 10 A725） | 算力太小，而且**跟 GPU 共用同一條 273 GB/s**，所以 CPU offload 計算在這台機器上沒有意義 |

### 1.1 Roofline：Qwen3.8-27B 在 GB10 上的理論上限

| 權重格式 | 每步讀的權重 | decode step 下限（85% 頻寬） | 單流 tok/s | × 推測解碼 2.5（接受率） |
|---|---|---|---|---|
| BF16（今天） | ~52 GB | 233 ms | 4.3（**實測 4.3，已到頂**） | ~11 |
| FP8 | ~27 GB | 116 ms | 8.6 | ~21 |
| NVFP4（attention / GDN 層留 FP8） | ~17 GB | 72 ms | 14 | ~35 |

| prefill | 達成 TFLOPS（估） | tok/s | 1024-token prompt |
|---|---|---|---|
| BF16（今天） | 65–70（實測 1,293 tok/s） | ~1,300 | 0.8 s |
| FP8 | ~120 | ~2,200 | 0.46 s |
| NVFP4 | ~180 | ~3,300 | 0.31 s |

> 名詞：**TPOT**（time per output token）= decode 每個 token 的間隔；**TTFT** = 送出 prompt 到第一個 token 的時間（≈ prefill 時間 + 排隊）；**prefill** = 把 prompt 整批算一次填 KV cache（算力密集）；**decode** = 一次一個 token（頻寬密集）。

### 1.2 「每個 step 有 ~300 個免費名額」

一個 decode step 不管 batch 裡有幾個 token，都要把權重讀一遍（190 ms @ BF16）。
在這 190 ms 內 GPU 的算力可以處理多少 token？
`TFLOPS × bytes/param ÷ (2 × 頻寬)` = BF16 366、FP8 366、NVFP4 326 個 token。

也就是說：**在 GB10 上，一個 step 處理 1 個 token 和處理 300 個 token，時間幾乎一樣**（KV / state 讀取另計）。
這個比值（算力 ÷ 頻寬）GB10 其實比 H100 還高，所以 batching、speculative decoding、tree speculation 在這台機器上**比在資料中心 GPU 上更划算**。
vLLM 自己的 Spark 文章建議 `--max-num-seqs 4`（「超過 4 個 TTFT 會暴增」），等於把這 300 個名額放著不用 —— 這就是我們的機會。

### 1.3 統一記憶體的六個後果

1. **沒有「VRAM 滿了」這件事，只有「整台機器記憶體滿了」**：OOM 不是一個 CUDA error，是整台機器凍住（vLLM 使用者回報 MemAvailable 一分鐘內歸零、earlyoom 救不回來、沒有 BMC 只能拔電）。
2. **cgroup 看不到 GPU 配置**：`docker --memory` 擋不住（實測 10.7 GB 的 GPU 配置，cgroup 只記了 377 MB）。
3. **page cache 跟 GPU 搶同一池**：讀完 checkpoint 留下的 page cache 讓 `cudaMemGetInfo` 以為沒記憶體了（kernel_zoo B1，已修 b5e5e71c）。
4. **任何 host↔device 拷貝都是純浪費**：同一條 DRAM 自己拷給自己，還佔兩倍空間。TS 的 KVStore host tier、retraction snapshot pool、預設 weight loader 的 H2D copy 都屬此類。
5. **CPU 配的記憶體 GPU 直接可讀**（ATS / NVLink-C2C 一致性；`deviceQuery` 報 `Integrated GPU sharing Host Memory: Yes`）：mmap 的檔案可以 `cudaHostRegister` 後直接當權重用 —— 零拷貝載入的基礎。GPU 走 ATS 讀這種記憶體的頻寬還沒人量過，是本計畫第一個要做的微實驗。
6. **`nvidia-smi` 報 N/A**：所有記憶體決策要改看 `/proc/meminfo` 的 MemAvailable。

---

## 2. 現況：我們的實測 vs. 公開的 vLLM / SGLang 數字

### 2.1 TS 在 GB10（rev 4d5ed6d0，upstream main + bring-up 修正，無 plugin，util 0.6，`results/baseline-bf16/`）

| shape | c | 總 tok/s | TTFT p50 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|
| unique 1024/128 | 1 | 4.3 | 1.04 s | 223 ms | 225 ms |
| | 16 | 48.5 | 6.2 s | 260 ms | 318 ms |
| | 64 | 90.0 | 15.4 s | 597 ms | 692 ms |
| shared 1024+256/128 | 1 | 4.4 | 0.61 s | 225 ms | |
| | 64 | 122.4 | 9.4 s | 441 ms | 490 ms |
| long 6144/128 | 1 | 3.7 | 4.75 s | 233 ms | |
| | 64 | 22.8 | **108 s** | **1,934 ms** | 2,294 ms |

FP8 KV cache（`fp8-kv` run）：吞吐 +0–7%，KV 槽位 2 倍，GSM8K 0.97 vs 0.945（200 題），沒有壞處。
Ready time：482–497 s。

兩個後續 run（`results/`）：

- `plugin-fp8kv`（rev cf5d6b15，plugin 裝好）：數字跟 `fp8-kv` 一樣（unique c=64 92.6、GSM8K 0.965）。預期中：`--attention-backend triton` 把 solution 鎖死，plugin 的 FlashInfer decode 選不到（kernel_zoo C4）。量 plugin 效益要先拿掉這個旗標。
- `prefill-proof`（rev 07696f3d）：`long` c=64 TPOT p50 1,998 ms vs `longshared` c=64（同樣的 6144-token decode context，但前綴共用、幾乎沒有 prefill）435 ms。**同一個 decode 工作量，有沒有別人的 prefill 夾在中間差 4.6 倍** —— 直接證實下面「prefill 打斷 decode」的拆解，也是 WP4 的依據。

**拆解（這決定了工作包的順序）**：

- **c=1**：TPOT 223 ms = 52 GB ÷ 233 GB/s，85% 的頻寬上限。**BF16 下已經到頂，剩下只能靠減 byte（量化）和每步多吐 token（推測解碼）。**
- **unique c=64**：597 ms ≈ 190 ms 權重 + ~400 ms prefill。穩態下每個 step 平均要吞 0.5 個新請求 × 1024 token，以 1,300 tok/s 算正好 ~390 ms。**c=64 時 TS 其實是 prefill 算力綁住的，不是 decode。**
- **long c=64**：prefill 一段 6144 token 要 ~4.7 s，預設 `enable_mixed_batch=False` → prefill 期間所有 decode 停住；TPOT p50 1.9 s、TTFT p50 108 s。kernel_zoo E 已算過：每 token 間隔 231→1934 ms 之中 88% 是 prefill 停頓。
- **shared c=64**：441 ms = 190 權重 + ~100 prefill（256 新 token）+ ~40 KV 讀取 + ~45 GDN state（FP32，每條 151 MB，c=64 讀寫 ~9.7 GB）。**hybrid 模型的 state 流量在高併發時跟權重同一個量級。**
- prefill 1,293 tok/s ≈ 70 TFLOPS ≈ BF16 峰值的 70%：**BF16 kernel 再調也不到 1.4x，要大幅提升只能換 FP8 / FP4 運算。**

### 2.2 公開數字（同一台機型，別人的量測，未自行驗證）

| 來源 | 設定 | 單流 decode | prefill | 備註 |
|---|---|---|---|---|
| NVIDIA 論壇 / helge A/B | vLLM 0.27, Qwen3.8-27B NVFP4 + MTP | ~24.5（25–30 依工作負載） | ~1,570–1,914 tok/s | 混合語言散文 MTP 接受率穩 |
| 同串 | SGLang, Qwen3.8-27B NVFP4 + **DSpark** | ~34（eval 平均 38，峰 46.7） | TTFT 0.22–0.28 s | 作者的上限推算：225 GB/s ÷ 20 GB/step × 3.3–4.7 接受 = 34–47 |
| 同串 | llama.cpp Q4_K_XL + MTP | ~27 | ~600 tok/s @14k | |
| vLLM 官方 Spark 文章 | Nemotron-3-Super-120B-A12B NVFP4 | 22.7–23.7 | 1,636–1,884 | 建議 `--max-num-seqs 4`；載入 10–15 分鐘；首請求 JIT 25 s |
| sergioamsilva 同 harness | vLLM 4-bit, 4 顆 A3B/A4B 級 MoE | 55–77 | ~6,000（小模型） | **c=8 總吞吐只有 c=1 的 3.9–5.5x**；MTP +81%（k=2 優於 4） |
| ai-muninn | vLLM, Qwen3-8B: FP8 → NVFP4 W4A4 → W4A16 | 25.7 → 38.6 → 40.9 | | **hybrid Qwen3.6-35B-A3B 上 FP8 反而比 NVFP4 快 32%：BF16 GDN 層佔 decode 時間 59%** |
| docai.hu | vLLM, Qwen3.5-122B-A10B NVFP4, MTP-2 | 25–31 | | 接受率：JSON 100%、code 94–99%、散文 63–74% |
| LMSYS | SGLang, Llama-3.1-8B BF16 | 20.5 | 7,991 | b=32：368 tok/s（18x）—— batching 本身在 GB10 是可以 scale 的 |

### 2.3 我們輸在哪、贏的空間在哪

| 面向 | 現況 | 結論 |
|---|---|---|
| 單流 | 4.3 vs 25–34 | 純 byte 問題。FP8 + MTP 可到 ~20，NVFP4 + MTP/DSpark 可到 30–40。**沒有這一步，其他全部看不見。** |
| 高併發 | TS c=64 unique 90 tok/s；vLLM 公開 c=8 只有 c=1 的 4–5x | 兩邊都被「prefill 打斷 decode」和 prefill 算力綁住。誰先讓 decode 不等 prefill，誰的曲線就往右上。 |
| 記憶體 | TS：host tier 要 103 GB、KV 曾被算成 1.2 GB（已修）；vLLM：0.9 util 要 115 GB、CUDA graph + compile 13 GB、JIT 尖峰 20–30 GB、常凍機 | 兩邊都是為「獨立 VRAM」寫的。這是沒人佔的地。 |
| 啟動 | TS 482 s；vLLM 10–15 分鐘 | 零拷貝載入可以到數十秒。開發機換模型是日常。 |
| hybrid / GDN | FP32 state、Triton 路徑、plugin 的融合 prefill 只在 ≥2 序列才用、prefill 沒 graph | Qwen 3.5/3.6/3.8 全是 hybrid；誰把 GDN decode 做好，誰在這條產品線上贏。 |

---

## 3. 策略論點

### L0：補齊（沒得選）

1. **FP8 權重 GEMM 在 sm_121 上能跑、夠快**（WP1）。
2. **NVFP4 在 sm_121 上能跑、夠快**（WP2）；W4A16（Marlin）當備案。
3. **MTP / DSpark 在 Triton / plugin 路徑上能跑**（WP3）。TS 已經有 DSPARK drafter —— SGLang 贏 vLLM 的那一招我們 code base 裡就有。

### L1：放大（vLLM 在單顆小 GPU 上做不好的）

4. **decode 永遠不等 prefill**（WP4）：從 mixed batch → 小 chunk + decode 優先 → 雙 stream 重疊（decode 吃頻寬、prefill 吃算力，兩者在 GB10 上是互補資源）。
5. **填滿免費名額**（WP3 後段）：tree speculation 目前綁 trtllm backend，要移植到 Triton / FA2；c=1 時驗證 16–32 個 draft token 幾乎零成本。
6. **GDN state 成本**（WP5）：FP32 → BF16 state（旗標已有）、融合的 GDN decode、plugin prefill 補 graph capture。
7. **共用前綴只讀一次**（WP6，kernel_zoo E 已規劃）：agent 場景（長 system prompt × 多人）c=64 decode byte −29%。

### L2：統一記憶體原生（差異化）

8. **記憶體預算語意**（WP7）：絕對 GB 預算、MemAvailable 驅動、彈性 KV、預設關掉 pinned 複本、內建不凍機 guard、與 CI / 其他服務共存。
9. **零拷貝載入權重**（WP8）：mmap + `cudaHostRegister`，權重直接住在 page cache，重啟 / 跨 container 共用。
10. **整池 prefix cache**（WP7 延伸）：128 GB 裡權重以外全部當 prefix cache，agent session 的 KV 不用重算。
11. （moonshot）**放不進 128 GB 的 MoE**（WP9）：NVMe-backed expert streaming。

### 明確不做

- **CPU offload 計算**：CPU 跟 GPU 共用同一條 273 GB/s，decode 搬去 CPU 不會多出頻寬。
- **BF16 kernel 微調**：已達峰值 70%，天花板 1.4x，而 FP8 是 2x、NVFP4 是 3–4x。
- **EXL3 / GGUF 格式**：常駐 vLLM 服務用 EXL3，但 TS 沒有這條路；NVFP4 是 NVIDIA 在 Spark 上的主推格式，checkpoint 供應最多。
- **多台 Spark TP**：先把單機做到贏。

---

## 4. 工作包

每個 WP 列：目標 / 為什麼在 GB10 特別有效 / 現有基礎（盤點自 main 原始碼，2026-10-20）/ 要做的事 / 驗收 / 風險。
排序依「預期收益 × 確定性 ÷ 工作量」。

### WP0 量測基礎建設（第一週就要有，之後每個 WP 都靠它）

- **目標**：一張圖說服所有人。同機、同 harness 的 TS vs vLLM Pareto 圖，加上 TPOT p99（「decode 有沒有被打斷」）和 TTFT。
- **現有基礎**：`unieinfra/gb10/run.sh window`（4 shapes × c=1/4/16/64、`vllm bench serve`、GSM8K、`metrics.json`）；vLLM image 已在機器上（`vllm/vllm-openai:nightly-aarch64`）；perf-chart 慣例（README Pareto 圖）。
- **要做**：
  1. `run.sh` 加 `vllm` 模式：同一個模型、同一組 shapes 跑 vLLM（NVFP4 checkpoint + MTP，照 vLLM 官方 Spark 建議設定），寫同格式 `metrics.json`。
  2. 畫圖腳本：Pareto（x = 1000/TPOT p50，y = 總 tok/s）、一條單流頻寬上限參考線；第二張：TPOT p99 vs 併發（WP4 的指標）。
  3. sm_121 kernel microbench suite（repo 只有 AMD gfx950 的；`tokenspeed_kernel.benchmark` harness 現成）：GEMM（BF16/FP8/NVFP4，M=1/16/64/1024）、MHA decode、GDN decode/prefill。配合 kernel_zoo D2：**跑完 ≠ 用到**，每次量測都印選中的 kernel 名字。
  4. 記憶體時間序列：`/proc/meminfo` MemAvailable 每秒採樣進 log（level 1 watchdog 已有，擴到 level 2），之後 WP7 要用。
  5. 衛生債（盤點發現）：`docker/Dockerfile.nvidia_gb10` 用 `sm121`、`unieinfra/gb10/Dockerfile` 用 `12.1a`，要統一（README 與 plugin CHANGELOG 已於 2026-10-20 更新）。
- **驗收**：一個指令產出 TS 與 vLLM 的兩張圖 + GSM8K。
- **需要**：一個 level 2 視窗（停常駐服務）。

### WP1 FP8 權重 GEMM 在 sm_121（最快的 2x）

- **目標**：Qwen3.8-27B-FP8（block 128×128，Qwen 官方格式）c=1 ≥ 8 tok/s（TPOT ≤ 125 ms）、prefill ≥ 2,000 tok/s、GSM8K 掉 ≤ 1 pt。
- **為什麼**：decode byte 減半、prefill 算力加倍，一次改善兩個 regime；FP8 在 sm_121 沒有 FP4 的 PTX 地雷，確定性最高。
- **現有基礎**：
  - 路徑 `runtime/layers/dense/fp8.py` → `mm(quant="mxfp8")`；候選 kernel：`flashinfer_mm_fp8_blockscale`（min 10.0 無上限，**sm_121 會被選中但沒驗證**）、`triton_mm_fp8_blockscale`（PORTABLE）。
  - plugin 有寫好但**沒接線**的 `unieinfra_mm_fp8_blockscale_sm120`（M≤256 走 Triton、以上走 FlashInfer）。
  - 活化量化 `triton_quantize_fp8_with_scale` 任何 NVIDIA 都可。
  - FP8 KV 已在 `gb10-fp8-kv` 分支修好（kernel_zoo C1）。
- **要做**：
  1. 用 WP0 的 microbench 量三個候選在 M=1/16/64/1024 的 TFLOPS 與頻寬，決定 plugin 的切換點（M≤256 這個數字要重量）。
  2. 把 plugin 的 FP8 GEMM 接上註冊（kernel_zoo C2/C5：加「預期註冊的 kernel 都在」和 trait 對照的測試）。
  3. 整模型跑 `run.sh window`，進 WP0 的圖。
  4. MoE：`triton_fp8_block_precomputed_moe_apply` 已可用；`flashinfer_trtllm_fp8` 在 `auto` 下會被選中但 plugin 筆記說 sm_120 會 raise → 排除或修。
- **風險**：FlashInfer blockscale kernel 在 sm_121 的 tile 可能超 101 KB smem（kernel_zoo A3 同型問題）。

### WP2 NVFP4 在 sm_121（追平 vLLM 單流的關鍵）

- **目標**：Qwen3.8-27B-NVFP4（RadixArk 或 NVIDIA 官方 checkpoint）c=1 ≥ 13 tok/s（無推測）、prefill ≥ 3,000 tok/s；GSM8K 掉 ≤ 2 pt。
- **為什麼**：vLLM / SGLang 在 Spark 上所有漂亮數字都是 NVFP4；不做就永遠少 1.5–2x。
- **現有基礎與阻塞（盤點）**：
  - `dense/nvfp4.py` 直接呼叫 `ops/quantization/flashinfer.py` 的 `fp4_quantize`，而那個函式只在 `is_blackwell`（major == 10）下定義，sm_12x 上是 `error_fn` → **目前 NVFP4 linear 在 GB10 必定失敗，就算 plugin 有 `unieinfra_flashinfer_quantize_nvfp4_sm120` 也沒用，因為被繞過了**。
  - GEMM 候選：`cublaslt_mm_nvfp4`（min 10.0 無上限，需要 `trtllm_kernel` wheel 有 sm_121 code，未驗證）；`flashinfer_mm_nvfp4` 綁 `is_blackwell` 沒註冊；CuTe NVFP4 限 10.0–10.7；W4A16 路徑直接 raise「requires SM100/SM103」。
  - compressed-tensors WNA16 Marlin（min cap 9.0）可用 → W4A16 備案（ai-muninn 量到 W4A16 單流反而最快，因為 c=1 純頻寬）。
  - 外部狀態：FlashInfer issue #3013（sm_120/121 NVFP4 慢）2026-05 關閉，期間合入 b12x `mm_fp4`、小 tile-N blockscaled GEMM、b12x fused MoE（PR #3051/#3066/#3152）；CUTLASS 4.4+ sm_121a 可編（issue #2800）；CUTLASS DSL 4.5.1 才含 sm_121a。
- **要做**：
  1. `dense/nvfp4.py` 改走 op registry（讓 plugin 的 quantize kernel 能被選中）；放寬 W4A16 的 arch 檢查。
  2. 三條 GEMM 路線各做 microbench：(a) `cublaslt_mm_nvfp4`（先查 wheel 有沒有 sm_121 SASS，level 1 的 SASS 檢查可重用）；(b) FlashInfer 0.7.x 的 b12x `mm_fp4`；(c) 自己用 CUTLASS sm_121a 編（論壇實測 356 TFLOPS 的配置可直接抄：TN layout、cluster 1×1×1、tile 掃描限 101 KB）。
  3. 混合精度策略：GDN / attention 層留 FP8（ai-muninn 的 hybrid 教訓），MLP 走 NVFP4；`modelopt_mixed` 已支援 per-layer 格式。
  4. MoE：`flashinfer_cutlass_nvfp4` 在 sm_121 admitted 但未測；備案 Marlin MXFP4。
- **驗收**：WP0 圖上 TS NVFP4 c=1 與 vLLM NVFP4（無 MTP）持平；prefill ≥ vLLM 的 1,900 tok/s。
- **風險**：PTX / 靜默錯誤（sm_120 kernel 在 sm_121 跑出垃圾但不報錯，有人踩過）→ 每個 kernel 先過 numerics 比對再上整模型；FlashInfer 版本漂移（runbook 已從 0.6.18 → 0.7.1rc2 → 0.7.2 跟著 main 的 pin 走，每次 rebase 都要對一次）。

### WP3 推測解碼：MTP → DSpark → tree

- **目標**：c=1 在 NVFP4 上 ≥ 30 tok/s（code/JSON ≥ 40）；c=4–16 時每 step 接受 ≥ 2 token。
- **為什麼**：1.2 節 —— 每步 300 個免費名額，c=1 只用了 1 個。公開數字：MTP +81%～2x、DSpark 讓 SGLang 贏 vLLM 40%。
- **現有基礎**：drafter 有 EAGLE3 / MTP / DFLASH / **DSPARK**；Qwen3.5 NextN 模型檔、Qwen3.8 自帶 MTP 層；ReplaySSM（GDN state 的 verify 回滾）預設開；FlashInfer `gdn_decode_mtp` 在 sm_121 admitted；Triton FLA 有 `decode_mtp` / `replay_commit`。
  - **阻塞**：tree speculation（`--speculative-eagle-topk > 1`）綁 trtllm backend（`docs/design/tree-speculation.md`），sm_12x 不能用；CI 裡所有 MTP 設定都用 trtllm，**沒有任何 MTP 設定在 Triton 路徑上跑過**。
- **要做**：
  1. 鏈式 MTP（topk=1）在 `--attention-backend triton` 上跑通：k=1/2/3，量接受率（code / JSON / 中文散文 / 英文散文 四種 prompt）與 tok/s。預期 k=2 最好（公開數字一致）。
  2. DSpark：找 / 訓 Qwen3.8-27B 的 DSpark drafter（SGLang 用 1.4B 的 block drafter，一次提 7 個），接 TS 的 DSPARK 路徑，比 MTP。
  3. tree speculation 移植到 Triton / FA2：verify 需要 cascade（同請求內 prefix + window 的合併），`attn_merge_state` 零件現成；先做 c=1，再做 batch。
  4. 併發下的策略：接受 token 數 × batch ≤ ~300 之前都是免費的 → scheduler 依 batch 大小動態調 draft 長度（c=1 用 8–16、c=64 用 1–2）。
- **驗收**：Pareto 圖左上角（c=1–4）TS 在 SGLang 公開數字（34–38）之上。
- **風險**：NVFP4 擾動 hidden state 讓 BF16 訓的 drafter 接受率掉（EAGLE3 有人量到數學 −2～12 pt）→ MTP 層跟主模型一起量化校準，或 MTP 層留 BF16（才 5 GB 級）。

### WP4 decode 永遠不等 prefill（高併發曲線的關鍵）

- **目標**：unique c=64 TPOT p50 597 → ≤ 300 ms、總吞吐 90 → ≥ 150 tok/s；long c=64 TPOT p99 2.3 s → < 0.5 s。
- **為什麼**：2.1 的拆解 —— c=64 的 TPOT 有 2/3 是等別人的 prefill。decode 吃頻寬、prefill 吃算力，**在 GB10 上這是兩種不同的資源**，理論上可以同時跑。vLLM 的解法是 `max-num-seqs 4`（放棄併發）。
- **現有基礎**：
  - `--enable-mixed-batch`（decode 優先、prefill 填剩餘預算）存在，**但 `MHAAttnBackend` 斷言不接受 mixed batch**（`paged/mha.py` L190）→ MHA / GDN-hybrid 模型今天不能用。
  - unieai 分支有 `--mixed-prefill-decode-min-load`（PRO 6000 上 Qwen3.5-27B c=128 共用前綴 754 → 1,087 tok/s；無共用前綴 −16%）。
  - C++ scheduler 的 FSM + overlap schedule（depth 1）+ breakable prefill graph（`--prefill-graph-max-tokens`）是做「兩條執行 lane」的好底子。
- **要做（三階梯，每階都量 WP0 的 TPOT p99 圖）**：
  1. **mixed batch 支援 MHA / GDN**：拿掉斷言，extend kernel 已經同時吃 prefill 和 decode（Triton 路徑 prefill 本來就走 `triton_mha_extend_with_kvcache`）；移植 unieai 的 min-load 門檻。預期：long c=64 的 p99 從 2.3 s 掉到 chunk 大小決定的值。
  2. **chunk 大小 vs 效率掃描**：`--chunked-prefill-size` 8192 → 2048 / 1024 / 512，量 prefill tok/s 損失（小 chunk 的 GEMM 效率低）與 TPOT 平穩度；可能要「動態 chunk」：有 decode 在跑時小 chunk，閒置時大 chunk。
  3. **雙 stream 重疊（研究題）**：decode 和 prefill 各自一條 CUDA stream 同時跑。階梯：(a) stream priority（decode 高、prefill 低，硬體排程器會優先派 decode 的 CTA）；(b) **green contexts**（CUDA 12.4+ `cuGreenCtxCreate`，把 SM 切成例如 8 個給 decode、40 個給 prefill）；(c) 最後才考慮兩個 process + MPS。
     - **第一個要做的微實驗**：decode kernel 只拿 8–12 個 SM 能不能吃滿 273 GB/s？（一個 copy kernel 在 green context 下掃 SM 數 vs 頻寬。）能 → 整套值得做；不能 → 停在階梯 2。
     - scheduler 改動：一個 round 裡同時發出 decode batch 和 prefill chunk，兩者的 KV 寫入分區不重疊（prefix 粒度 128 已保證 block 級隔離）。
- **驗收**：unique c=64 的 TPOT p50 ≤ 1.3 × c=1；long c=64 的 TPOT p99 ≤ 2 × p50。
- **風險**：雙 stream 時權重讀取互搶頻寬（prefill 一次 1024 token 也要讀 52 GB 權重）—— 但 prefill 一段 0.8 s 只讀一次，decode 每 0.23 s 讀一次，加總仍在預算內；真正的未知是 ATS / L2 的行為，靠微實驗。

### WP5 Hybrid（GDN）decode 成本

- **目標**：shared c=64 TPOT 441 → ≤ 350 ms；Qwen3.6-35B-A3B（CI 模型）單流 ≥ vLLM FP8 公開數字。
- **為什麼**：Qwen 3.5 / 3.6 / 3.8 全是 GDN hybrid；ai-muninn 量到 hybrid 上 GDN BF16 層佔 decode 59%，連 NVFP4 都救不了。這條產品線誰做好 GDN 誰贏。
- **現有基礎**：`--mamba-ssm-dtype bfloat16` 旗標已存在（預設 float32；BF16 state 會強制 Triton decode）；plugin 融合 GDN prefill（≥2 序列才用、head_dim 128、不在 `_DEVICE_BOUNDS_CHUNK_PREFILL` 名單 → 選中時 prefill 失去 graph capture，kernel_zoo C7）；FlashInfer `gdn_decode_step` 在 sm_121 admitted 未驗證。
- **要做**：
  1. BF16 state 的精度實驗（GSM8K + 長生成的 drift；UnieAI 有 `mamba_ckpt_dtype` 的實驗可參考），通過就改 GB10 預設。
  2. GDN decode microbench：Triton FLA vs FlashInfer 在 c=1/16/64、state fp32/bf16；c=64 目標 state 流量 9.7 → < 5 GB/step。
  3. plugin 融合 prefill 補 graph capture 名單 + 放寬 `FUSED_MIN_SEQS`，量整體（不是只量 kernel）。
  4. 對照 vLLM 的 GDN 路徑（FLA Triton），看有沒有 sm_12x 專屬的可抄之處。
- **風險**：BF16 state 的長序列精度；plugin decode 的 `prepare` hook 在 main 沒人呼叫（盤點）→ 開 CUDA graph 會 raise，要先修。

### WP6 共用前綴只讀一次（cascade attention）

- 承 kernel_zoo E：64 人共用 6k 前綴，decode 每步 ~87 GB → ~62 GB（−29%），前提是 WP4 先把 prefill 停頓拿掉（否則被遮住）。
- 參考：vLLM v1 `cascade_attention`（≥ 256 token 共用且 ≥ 8 請求才開）、FlashInfer 0.7.1rc2 `MultiLevelCascadeAttentionWrapper`；TS 內只有同請求的 tree cascade，`merge_state.cu` 可重用。
- 排在 WP4 之後；agent 工作負載（長 system prompt × 多人）是目標形狀。

### WP7 統一記憶體原生的記憶體管理（L2 的第一步，也是這台機器的日常痛點）

- **目標**：(a) 一個 `--memory-budget-gb` 絕對預算（vLLM 社群在要、還沒有）；(b) 永不凍機；(c) 跟 CI / 常駐服務 / 桌面共存；(d) 整池 prefix cache。
- **現有基礎**：`get_available_gpu_memory` 的 `is_integrated` 分支（b5e5e71c）；KVStore L2 host tier（預設開、ratio 2.0、pinned）與 retraction pool（pinned）—— 在 GB10 上是複本；`--enable-memory-saver`（TorchMemorySaver 區域）；C++ scheduler 擁有 KV block。
- **要做**：
  1. **預設策略**：`is_integrated` 時預設關 KVStore host tier 與 retraction pinned pool（kernel_zoo B2，需團隊決定，AGENTS.md 要求行為選擇明確）；文件化「在 Spark 上這些 tier 沒意義，因為 host 就是 device」。
  2. **絕對預算 + 保留量**：`--memory-budget-gb` 與 `--host-reserve-gb`（預設 20，對應 vLLM 使用者實測的凍機門檻 ~20 GiB），sizing 全部以 MemAvailable 為準；CUDA graph reserve、JIT 尖峰（FlashInfer 平行 cicc 20–30 GB）納入預算。
  3. **內建 guard**：engine 內一條執行緒每秒看 MemAvailable，低於保留量就停止接新請求 / 先 retract（TS 有 retraction 機制！）而不是讓 OOM killer 砍常駐服務（kernel_zoo B3）。
  4. **彈性 KV arena**：用 CUDA VMM（`cuMemAddressReserve` + `cuMemMap` 分塊）而非一次 `torch.zeros(arena)`，啟動時只配最小量，依 MemAvailable 成長 / 回收 → 跟 vLLM、CI container 同機也能跑；也讓「整池 prefix cache」自然成立：沒人用的記憶體就是 prefix cache。
- **驗收**：在常駐 vLLM 跑著（剩 ~10–20 GB）的情況下 TS 能起一個 8B 模型服務不凍機；預算 60 GB 時實際峰值 ≤ 60 GB（含 JIT）。
- **風險**：VMM 在整合式 GPU 的支援度；PyTorch allocator 與 VMM 的整合。

### WP8 零拷貝載入權重（482 s → 數十秒）

- **目標**：ready time < 60 s（page cache 熱時 < 20 s）；載入期間峰值記憶體 = 權重大小（不是 2 倍）。
- **為什麼**：預設 loader = prefetch 進 page cache（40 GiB 視窗）→ `safetensors load_file(cpu)` mmap → `param.copy_()` 的 pageable H2D copy。在 GB10 這是把同一條 DRAM 的資料自己拷給自己，期間 page cache + GPU 複本佔兩倍空間（52 GB 模型 → 104 GB，這就是為什麼 KV 曾被算成 1.2 GB）。vLLM 一樣慢（10–15 分鐘）。開發機天天換模型，這是看得見的產品差異。
- **現有基礎**：`platform.register_host_tensor_for_gpu_access`（cudaHostRegister 包裝）已存在，DeepSeek-V4.1 Engram host table 和 PLE lookup 已用這招；`--load-format instanttensor`（直接 I/O，aarch64 要 sdist 編，仍是拷貝，半價）。
- **要做**：
  1. **微實驗（最先做，也是 WP4/WP9 的前提）**：GPU 讀 (a) `cudaMalloc`、(b) mmap + `cudaHostRegister`、(c) `cudaMallocManaged`、(d) (b) 加 2 MB hugepages / `MAP_POPULATE` 四種記憶體的頻寬（一個 reduction kernel）。如果 (b)/(d) ≥ 90% 的 (a)，整套成立。
  2. `--load-format mmap`（只在 `is_integrated` 允許）：參數 storage 直接指向 mmap 區域；`mlock` 避免被 page cache 淘汰；`process_weights_after_loading` 需要可寫複本的層（線上量化、repack）退回一般路徑。
  3. 跨重啟 / 跨 container 共用：同一個檔案的 page cache 只有一份 → 兩個 TS process（CI + dev）載同一模型不用兩份記憶體。
  4. 備案：InstantTensor aarch64 sdist 進 runbook image。
- **風險**：ATS 讀 host-registered 記憶體的 TLB 成本（4 KB page）—— hugepages 解；page cache 被回收時的 page fault 延遲 —— mlock 解；safetensors 的 tensor 對齊（通常 8 byte，GEMM 需要 16–256 byte 對齊）—— 先掃 checkpoint 的 offset。

### WP9（moonshot）放不進 128 GB 的 MoE

- Qwen3.5-397B-A17B NVFP4 ≈ 220 GB，active 17B → 每 token 只讀 ~10 GB。mmap 全部 expert 到 NVMe-backed page cache，熱 expert 留在記憶體、冷 expert 從 NVMe（~5 GB/s）補。vLLM 做不到；llama.cpp 有 lazy mode 的雛形（PLE 表）。
- 只在 WP8 的微實驗通過後開始；先量 expert 命中率（agent 工作負載 expert 分佈很偏）。

---

## 5. 時程與里程碑

| 階段 | 週 | 工作包 | 里程碑（可量化） |
|---|---|---|---|
| A 追平 | 1–3 | WP0、WP1、WP3 步驟 1（鏈式 MTP）、WP5 步驟 1（BF16 state） | 第一張 TS vs vLLM 同機 Pareto；TS FP8+MTP c=1 ≥ 15 tok/s；TPOT 到 c=16 平穩 |
| B 超車 | 4–9 | WP2、WP3 步驟 2（DSpark）、WP4 階梯 1–2、WP7 步驟 1–3 | TS NVFP4+MTP/DSpark c=1 ≥ 30；unique c=64 ≥ 150 tok/s 且 TPOT ≤ 300 ms；絕對預算旗標；與常駐服務共存不凍機 |
| C 差異化 | 10–18 | WP4 階梯 3（雙 stream）、WP8、WP6、WP3 步驟 3（tree）、WP7 步驟 4 | ready < 60 s；long c=64 TPOT p99 < 0.5 s；shared 6k c=64 decode byte −25%；blog / README 圖 |
| D moonshot | 之後 | WP9 | 單機跑 397B-A17B，c=1 ≥ 15 tok/s |

**決策閘**：
- 第 3 週末：WP1 的 FP8 c=1 沒到 7 tok/s → 代表 sm_121 GEMM 有結構問題，先修 kernel 再往下。
- 第 5 週末：WP8 微實驗 (b)/(d) 頻寬 < 80% 的 cudaMalloc → WP8 降為 InstantTensor 備案、WP9 取消。
- 第 6 週末：WP4 微實驗 decode 用 ≤ 12 SM 吃不到 ≥ 85% 頻寬 → 階梯 3 改成「小 chunk + 動態 chunk」收尾。

**機器時間**：每個 level 2 實驗需要停常駐服務的視窗（`gb10-host` 筆記）；WP0 之後把 vLLM A/B 也排進同一個視窗。WP7 做完之後，大部分實驗可以在 8B 模型上與常駐服務共存，不再需要視窗。

---

## 6. 風險與未知（照嚴重度）

1. **sm_121 的 NVFP4 kernel 供應鏈**：cuBLASLt wheel 有沒有 sm_121 SASS、FlashInfer b12x 路徑版本、CUTLASS DSL 版本；靜默錯誤風險 → 每個 kernel 先過 numerics。
2. **ATS 頻寬未知**：WP4 階梯 3、WP8、WP9 都押在「GPU 讀 host-registered 記憶體不比 cudaMalloc 慢多少」；第一個微實驗就要做。
3. **精度**：NVFP4 hybrid、BF16 GDN state、推測解碼與量化的交互；GSM8K 200 題太少，WP0 加長生成的 drift 測試。
4. **時間都花在跟 upstream 同步**：main 每週變（kernel_zoo C6），plugin 簽名跟不上；WP0 的 trait / kwargs AST 對照測試要先做成 CI。
5. **GB10 是共用機**：CI runner、常駐服務、網路曾斷三天；實驗要可中斷、結果要落盤（`run.sh` 已做）。
6. **同事回報的潛在 bug（2026-10-22）**：這批 bring-up 改動被指出可能有 bug，尚未指明哪一個；候選與 rebase 後發現的 scaled-FP8 缺口記在 `kernel_zoo.md` F 節。進 WP1 之前先把它釐清。
7. **單人力**：WP4 階梯 3 與 WP7 步驟 4 是 scheduler / allocator 級改動，需要 Roy 或 WMC 參與；本計畫的 A、B 階段單人可做。

---

## 7. 要問 Roy / 團隊的問題

1. unieai 分支的 `--mixed-prefill-decode-min-load` 在 PRO 6000 上的 −16%（無共用前綴）根因是什麼？是 chunk 太小還是 GEMM 效率？（決定 WP4 階梯 2 的起點）
2. plugin 的 FP8 GEMM（`unieinfra_mm_fp8_blockscale_sm120`）當初為什麼不接線？M≤256 的切換點怎麼量的？
3. `mamba_ckpt_dtype` 的 BF16 state 實驗結果在哪？（WP5 步驟 1 直接沿用）
4. FA2 on sm_12x：aarch64 + torch 2.14 的 `vllm_flash_attn` 有沒有可行的 build？（WP3 tree 的 verify 路徑選 Triton 還是 FA2）
5. 常駐服務用 EXL3 是產品需求還是歷史原因？（決定 WP2 之後要不要看 W4A16 以外的格式）
6. KVStore / retraction pinned pool 在整合式 GPU 預設關閉 —— 同意改預設嗎？

---

## 8. 參考資料

- 樹內：`unieinfra/gb10/results/`（四個 run 的 metrics.json 與表格）；`unieinfra/gb10/kernel_zoo.md`；`unieinfra/gb10/README.md`；`docs/design/{scheduler,tree-speculation,cache-concepts}.md`；plugin `unieinfra/kernel-sm120/docs/kernel-provenance.md`。
- GB10 規格：https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- vLLM 官方 Spark 文章（設定、數字、限制）：https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- Qwen3.8-27B 各引擎對照（SGLang+DSpark 34–38 tok/s）：https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257
- 同 harness 跨引擎實測（vLLM c=8 scaling、MTP +81%）：https://forums.developer.nvidia.com/t/measured-inference-benchmarks-on-a-single-dgx-spark-same-harness-across-ollama-llama-cpp-and-vllm-notes-data-published/379766
- sm_121 CUTLASS 實測（NVFP4 356 / FP8 188 TFLOPS、101 KB smem、無 tcgen05）：https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960
- FP4 在 Spark 為何不照比例 scale：https://forums.developer.nvidia.com/t/fp4-on-dgx-spark-why-it-doesnt-scale-like-youd-expect/360142
- NVFP4 = 壓縮而非算力；hybrid 上 FP8 反勝（GDN 59%）：https://ai-muninn.com/en/blog/dgx-spark-nvfp4-compression-not-compute
- FlashInfer sm_120/121 NVFP4 效能 issue（已關，含修正 PR 清單）：https://github.com/flashinfer-ai/flashinfer/issues/3013
- 122B NVFP4 + MTP-2 接受率 by 工作負載：https://docai.hu/en/blog/qwen35-122b-spark
- Qwen3.8-Flash-Next 單機（PLE 表 NVMe 分頁讀取的先例）：https://madeye.github.io/qwen38-flash-next-on-dgx-spark/
- 記憶體：vLLM 在 Spark 的記憶體去向（13 GB graph、JIT 尖峰）：https://forums.developer.nvidia.com/t/memory-creep-on-dgx-spark-where-your-128-gb-actually-goes-and-how-to-stop-it/364886
- 記憶體：cgroup 看不到 GPU 配置、凍機時間線、保留量：https://dev.to/conatusai/dgx-spark-gb10-memory-sizing-for-llm-serving-the-numbers-42b7
- 統一記憶體 / ATS / 零拷貝背景：https://blog.kubesimplify.com/day-3-the-dgx-spark-unpacked-gb10-unified-memory-sm-121-and-the-one-reason-this-hardware-exists
- LMSYS Spark 評測（SGLang batch scaling）：https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/
