# Kernel Zoo — 我們踩過的坑與優化 pattern

> 原本是私人筆記，2026-10-20 起進 git（`unieinfra/gb10/kernel_zoo.md`）交接給團隊；姊妹篇 `hardware.md`（硬體上限怎麼推算）仍是本地筆記。
> 每條：**症狀 → 根因 → 修法 → 怎麼抓**。發現新的就往下加，附 commit / log 路徑。
> 跟 `hardware.md`（硬體規格、上限怎麼算）一起是專案的 knowledge base；這份記「發生過什麼」。

---

## A. 「版本號 ≠ 能力」系列（GB10 / sm_12x 最常見）

NVIDIA 的 compute capability 是兩條產品線，不是一條階梯：sm_100/103（B200/B300，
資料中心）有 tcgen05/TMEM、~228 KB shared memory；sm_120/121（RTX PRO 6000 / GB10，
消費級）沒有 tcgen05、只有 101,376 B shared memory per block。版本號 12.x > 10.x
不代表功能更多。

### A1. 精確比對版本 → 新型號靜默掉出
- **症狀**：GB10 上 UnieAI plugin 0 個 kernel 註冊，只有一行 INFO log。
- **根因**：`if arch_version != ArchVersion(12, 0): return`，GB10 是 (12, 1)。
- **修法**：用產品線判斷 `arch_version.major == 12`；kernel 自己的
  `CapabilityRequirement` 寫 min/max 範圍。
- **怎麼抓**：`python -m tokenspeed_kernel.plugins info <name>` 數註冊數。
- **出處**：`gb10-bringup` d3c5bf1d、23181e56。

### A2. 只寫下限的 guard → 新型號誤入不支援的程式碼
- **症狀**：`ptxas: Instruction 'tcgen05.st' not supported on .target 'sm_121a'`，整個
  kernel 套件編不起來。
- **根因**：`#if __CUDA_ARCH__ >= 1000` 包住 tcgen05；sm_121 是 1210 也通過。
- **修法**：`>= 1000 && < 1100`（upstream #1186 的寫法）或 `< 1200`。
- **怎麼抓**：用 `TOKENSPEED_CUDA_ARCH=121a` 編一次，ptxas 會直接報。
- **出處**：`gb10-bringup` a4cfc101；upstream 8/21 已修。

### A3. 依版本選 tile，而不是依 shared memory 預算
- **症狀**：`OutOfResources: shared memory, Required: 114688, Hardware limit: 101376`
  （Qwen3.8 head_dim 256 的 prefill，在 autotune 階段就死）。
- **根因**：`if platform.is_hopper_plus:` → 給 sm_90 的大 tile；sm_12x 版本號 ≥ 9.0。
- **修法**：用 per-block opt-in shared memory 預算決定 tile（< 128 KB 走小 tile）。
- **怎麼抓**：測試要涵蓋 head_dim 256（原本只測 64/128，所以 level 1 沒抓到）。
- **出處**：main `gb10` 54a85c2f（移植自 unieai mha_prefill.py）。

### A4. 讀了不存在的裝置屬性 → 永遠是 0
- **症狀**：`PlatformInfo.max_shared_memory_per_sm == 0`，在所有 CUDA 裝置上。
- **根因**：`getattr(props, "max_shared_memory_per_block", 0)`，torch 沒這個名字。
  正確是 `shared_memory_per_block_optin`（101,376 on sm_12x / 232,448 on sm_90）。
- **教訓**：`getattr(..., 0)` 的預設值會把拼錯的屬性名藏起來。印出來看一次。
- **出處**：同 A3。

---

## B. 統一記憶體（GB10：CPU 和 GPU 共用 128 GB）

### B1. `cudaMemGetInfo` 不算可回收的 page cache
- **症狀**：載入 52 GB 權重後 free 從 111.8 → 7.3 GB，KV cache 只配 1.2 GB。
- **根因**：整合式 GPU 上 `mem_get_info` = `MemFree`；讀 checkpoint 留下的 page
  cache 被當成已用。實測同時刻 `mem_get_info` 62.6 GB vs `MemAvailable` 116.8 GB。
- **修法**：`props.is_integrated` 時取 `max(mem_get_info, psutil MemAvailable)`。
- **出處**：main `gb10` b5e5e71c。

### B2. Host tier 在統一記憶體上是重複佔用
- **症狀**：`Not enough Host memory for the compact Host cache: requesting 103.28 GB`。
- **根因**：KVStore（L2 host cache，預設開、ratio 2.0）+ retraction snapshot pool
  依 device cache 大小配 host 記憶體；GB10 的 host 記憶體就是 device 記憶體。
- **目前處理**：`--disable-kvstore --retraction-snapshot-ratio 0`。預設值要不要改
  給團隊決定（AGENTS.md：行為選擇要明確）。

### B3. 硬擠著跑會先殺掉常駐服務
- 常駐 vLLM（105 GB）的 oom_score 全機最高；OOM killer 先砍它。測試要用
  `docker --memory` + host `MemAvailable` watchdog，且 cgroup 不一定看得到 GPU 配置。

---

## C. Kernel 註冊 / 選擇（registry）

### C1. `format_signatures` 給所有 role 同一個 dtype → 混合精度組合不存在
- **症狀**：FP8 KV cache 開了就 `NoKernelFoundError`（q=fp8, k/v=fp8, solution triton）。
- **根因**：Triton kernel 只登記 {fp16, bf16} × 全 role 同 dtype；runtime 又把 q
  轉成 fp8。sm_12x 上能吃 fp8 q 的只有 sm_100 限定的 FA4 / trtllm-gen。
- **修法**：用 `format_signature(q=..., k_cache=..., v_cache=...)` 登記
  「bf16 q × fp8 KV」，kernel 載入 KV 後轉成 q 的 dtype 再算（unieai 996b05f4）。
- **另一半問題**：main 的 Triton kernel 寫 `q.to(k.dtype)` / `p.to(v.dtype)`，
  方向是把 Q **往下**轉成 FP8；改成 `k.to(q.dtype)`、`v.to(q.dtype)`、`p.to(q.dtype)`。
  dtype 相同時是 no-op，BF16 路徑不變。
- **runtime**：不能直接拿掉「q 轉 fp8」，sm_100 的 FA4 / trtllm-gen 就是要 fp8 q。
  改成啟動時查 registry：兩個 op 都有 fp8-q kernel 就轉，否則保持 model dtype
  （`mha_fp8_cache_query_dtype`）。
- **數值**：tl.dot 的 kernel 跟「同一份 fp8 資料轉回 bf16」逐 bit 一致；逐元素的
  非 grouped decode 差 1 bf16 ulp（2⁻⁷）。
- **出處**：main `gb10-fp8-kv` 55ca2051、0169bb28。

### C2. Plugin 的 per-module try/except 把錯誤變成「沒註冊」
- **症狀**：plugin 的 FP8 GEMM 在 sm_120 上從來沒註冊過，log 只有 warning。
- **根因**：`platform.fp8e4m3fn` 被 upstream 移除（#889）→ AttributeError → 被吞。
- **教訓**：plugin 隔離失敗是對的，但要有測試斷言「預期註冊的 kernel 都在」。

### C3. 呼叫端加了 kwarg，plugin kernel 沒跟上 → 第一次被選中才爆
- **症狀**：`unieinfra_gdn_chunk_prefill_sm120() got an unexpected keyword argument 'h_dtype'`。
- **根因**：op API 對選中的 kernel 一律傳 `h_dtype`（f162d993 加的），plugin 簽名沒更新。
- **教訓**：plugin kernel 的簽名要跟 in-tree 同 op 的 kernel 一起被測試參數化。

### C4. 強制 backend 會讓 plugin kernel 選不到
- `--attention-backend triton` 把 attention solution 鎖成 triton；plugin 的
  FlashInfer decode（priority 16）因此不會被選。量 plugin 效益前先檢查這類旗標。

### C5. 沒宣告的 trait = 什麼值都接受（最危險的一條）
- **規則**：`spec_matches_traits` 預設忽略 kernel 沒宣告的 trait。所以 trait **改名**
  或 op **新增** trait 時，舊 kernel 會默默變成「全都支援」。
- **實例**（移植 plugin 到 main 時一次抓到）：decode 宣告 `support_sinks` 但 main 查
  `sinks` → 會被選去跑有 attention sink 的模型；GDN 宣告 `head_v_dim` 但 main 查
  `value_head_dim`；FP8 量化沒宣告 `dequantize` → 會被選去做 FP8→BF16 反量化；
  MoE 沒宣告 activation / EP / bias / routing 等 15 個 trait。
- **怎麼抓**：AST 掃 in-tree 同 (family, mode) 的 kernel 宣告過哪些 trait key，
  跟 plugin 宣告的比對；缺的就是「隱性全支援」。一次性腳本在對話紀錄裡，值得做成測試。

### C6. upstream 改 op 呼叫參數，plugin 簽名沒跟上
- main 的 MHA op 對選中的 kernel 一律傳 `enable_pdl`；`register_kernel` 拿掉 `tags`
  （#1654）；GDN helper 搬到 `ops/attention/gdn/`。全部都是「第一次被選中才爆」或
  「註冊時被吞」。
- **怎麼抓**：AST 掃 op wrapper 裡 `kernel(...)` 的 kwargs，對照 plugin 函式參數。

### C7. 換掉 kernel 可能順便關掉別的最佳化
- main 只在選中的 GDN prefill kernel 屬於 `_DEVICE_BOUNDS_CHUNK_PREFILL` 時才把
  prefill 錄成 CUDA graph；plugin 的融合 GDN 不在名單內 → 被選中時 prefill 失去
  graph capture。量 plugin 效益要看整體，不能只看 kernel 本身。

---

## D. 測試 / 量測

### D1. 測試只覆蓋小 head_dim
- head_dim 256（Qwen3.5/3.8、gemma）才會撞到 shared memory 上限。新增 kernel 測試
  至少涵蓋實際要跑的模型的 head_dim。

### D2. 跑完≠用到：量測前確認機制真的啟用
- plugin 註冊數、選到的 kernel 名字、prefix cache 命中（`#cached-token`）。
  UnieAI bench README：「A timing delta is not evidence that a mechanism engaged.」

### D3. 傳上去的 binary 跟原始碼版本要一致
- rebase 後 `sync.sh` 把舊 wheel 也傳了，`run.sh` 會裝舊版 → 測錯版本。現在不一致就不傳。

### D4. 單次量測的雜訊：< 7% 不下結論
- 每格只跑一次時，理論上不該受影響的點也會差到 ±4–7%（FP8 KV 的 shared c16 總輸出
  −4.3%、plugin 的 long c4 每人速度 −6.5%），TTFT p50 在 64 人時更敏感（排隊順序）。
  要宣稱 < 7% 的改善就重複跑同一格，或用 `--save-detailed` 看分佈。

### D5. 融合 kernel 有批次門檻 → 單流量不到
- plugin 的融合 GDN prefill 在 `FUSED_MIN_SEQS = 2` 以下交回 Triton（`gdn.py`）；
  c=1 的量測根本沒用到它。量 plugin 前先算「這個形狀下一個 prefill 批次有幾條序列」。

---

## E. 優化候選（還沒做）

- **GDN state 用 FP32**（`mamba_ssm_dtype='float32'`）：Qwen3.8-27B 每條序列約
  48 層 × 48 heads × 128 × 128 × 4 B ≈ 151 MB；c=64 時每個 decode step 要讀寫
  ~9.7 GB，跟 51 GB 權重比不小。BF16 state 可省一半，但要驗精確度（UnieAI 有
  `mamba_ckpt_dtype` 的實驗）。
- **FA2 on sm_12x**：plugin 需要 aarch64、torch 2.14 相容的 `vllm_flash_attn`。
- **共用前綴的 decode 只讀一次（cascade attention）**：前綴快取讓 KV 只「存」一份，
  但 Triton decode kernel 每個請求照自己的 page table 各讀一次，64 人共用 6k 前綴
  = 同一份 KV 讀 64 次。FlashInfer 的 cascade / shared-prefix attention 對整批只讀一次；
  agent 場景（長 system prompt × 多人）正好是這個形狀。
  - repo 現況（2026-10-20 查）：所有分支都沒有跨請求的版本；main 只有 tree speculation
    的 cascade（同一請求內 prefix + window，`docs/design/tree-speculation.md`）；
    `attn_merge_state` / `merge_state.cu` 是現成的合併零件。
  - 參考實作：vLLM v1 `flash_attn.py` 的 `cascade_attention` / `use_cascade_attention`
    （共用前綴 ≥ 256 token 且 ≥ 8 請求才開；GB10 上 vllm 0.29.1rc1 image 有）；
    FlashInfer 0.7.1rc2 `MultiLevelCascadeAttentionWrapper`（多層共用）。SGLang /
    TRT-LLM / TurboMind 尚未調查。
  - 粗估：64 人共用 6k 前綴，decode 每步 ~87 GB → ~62 GB（約 −29%），前提是 decode
    沒被 prefill 打斷。
- **prefill 打斷 decode**：`enable_mixed_batch=False`（預設）時，每段 prefill（最多
  8192 token，GB10 上約 6 秒）期間所有 decode 停住。long c=64：ITL p50 428 ms 但
  p99 25 s；每 token 間隔 231→1934 ms 中約 88% 是 prefill 停頓、12% 是 KV 變多。
  - 參考：unieai 分支有 UnieAI 自己的 `--mixed-prefill-decode-min-load`（main 只有
    布林值 `--enable-mixed-batch`）；PRO 6000 上 Qwen3.5-27B c=128 共用前綴 754 →
    1087 tok/s（`unieinfra/BENCH-qwen35-27b.md`）；無共用前綴時反而 −16%。
  - **已證實**（10-20 prefill-proof run，FP8 KV + plugin）：long vs long-shared（同樣
    ~6.2k decode 上下文、每步讀一樣多 KV，prefill 少 17 倍）。64 人：平均 token 間隔
    1,928 → 431 ms，總輸出 24.5 → 115.9 tok/s；ITL 中位數 405 vs 358 ms（≈ 頻寬推算
    309 ms），所以差別全在等待（79% vs 17%）。前三組 run 的 long c64 也都是 77–79%。
  - **怎麼量**：`--save-detailed` 的 `itls`：中位數 = 正常一步，(平均 − 中位數) / 平均
    = 等 prefill 的比例。不需要 profiler 就能拆。
  - **波浪現象**：closed loop 下同一批請求一起 prefill、一起 decode、一起結束 → 單一請求
    的時間軸是「前段一直被打斷、後段順暢」。挑代表請求時不要以為後段順暢是收尾假象。
- **硬體定位**：GB10 實測 prefill 1,292 tok/s ≈ 72 TFLOPS，約為推估 BF16 峰值
  （~125 TF）的 58%，BF16 下軟體空間有限；大幅提升要靠 FP8/FP4 運算（GB10 有 FP4，
  H200 沒有）。算力 / 頻寬約為 PRO 6000 的 1/4 / 1/6.6、H200 的 1/8 / 1/17.6。
- **MTP 推測解碼**：Qwen3.8 自帶 MTP 層；GB10 上 decode 受頻寬限制，推測解碼
  是最直接的單流加速（舊 vLLM NVFP4 + MTP 紀錄 24.7 tok/s）。

---

## F. 2026-10-22 rebase 到 main（d71ea804 → 87a789d1，約 200 個 upstream commit）之後

### F1. 同一個 bug 兩邊各修一次 → rebase 時只留測試
- `54a85c2f`（Triton prefill tile 依 shared memory 選）跟 WMC 已合進 main 的 `8d9414fe`
  修同一件事，邏輯等價（他的版本多了 `shared_memory_per_block` fallback 與警告）。
  rebase 採 main 的版本，我們的 commit 縮成只剩 head_dim 256 的測試（`test_attention.py`）。
- **教訓**：開工前先 `git log origin/main -- <檔案>`，同事可能已經修了。

### F2. main 新增 per-tensor scaled FP8 KV cache（upstream #2414）→ 跟 C1 的 Triton 路徑有縫
- main 現在把「checkpoint 自帶 k_scale/v_scale ≠ 1 的 FP8 cache」當成另一個 signature
  （`scaled-fp8`，`ops/attention/mha/__init__.py::_blockscaled_signature_and_scales`），
  只有 FlashInfer trtllm-gen 的 MHA kernel 宣告它；我們 C1 的 Triton「bf16 q × fp8 KV」只服務
  unit-scale 的 dense fp8 signature。
- 後果：GB10 上 `--kv-cache-dtype fp8_e4m3` 配 BF16 checkpoint（scale = 1）照常；**配自帶
  KV scale 的 FP8 checkpoint 會在第一次 forward 才 `NoKernelFoundError`**，因為
  `mha_fp8_cache_query_dtype` 啟動時只用 dense signature 探 registry，看不到這個缺口。
- 修法候選：Triton kernel 多宣告 scaled-fp8 signature 並在 upcast 後乘 descale（FlashInfer
  那邊是 `k_descale, v_descale = fp8_kv_scales`），或讓 `mha_fp8_cache_query_dtype` 也探
  scaled signature 並在啟動時就報錯。**未修，先記著。**

### F3. FlashInfer pin 0.7.1rc2 → 0.7.2（upstream #2338）
- `tokenspeed-kernel/python/requirements/cuda.txt` 現在是 `flashinfer-python==0.7.2`；runbook
  的 base image（`unieinfra/gb10/Dockerfile`、`build_wheels.sh`）原本是 0.7.1rc2，kernel build
  會再撞 jit-cache 版本檢查（同 65f5c809 當初的問題）。已改成
  `cu130-torch-2.14.0-flashinfer-0.7.2`（Docker Hub 有），**GB10 上尚未重建驗證**。
- main 自己的 `docker/Dockerfile.nvidia_gb10`（CI 用）還停在 0.6.18 base，CI 的 kernel 編譯
  是否還過要看下一次 `/ci-run`。

### F4. 同事回報：這批改動可能有潛在 bug（2026-10-22，尚未指明是哪一個）
- 尚未定位。依 rebase 後的審視，最值得先看的三處：
  1. F2 的 scaled-fp8 縫（C1 的 `55ca2051`）。
  2. `b5e5e71c`（`is_integrated` 時 KV sizing 取 `max(mem_get_info, MemAvailable)`）：
     MemAvailable 把可回收 page cache 算進來，但 page cache 若正被另一個 process（CI、常駐
     服務）的 mmap 權重佔著，回收會逼它們重讀磁碟；在共用機上可能把別人的服務拖慢。
     KV 佔比用 `--gpu-memory-utilization 0.6` 時有緩衝，0.95 預設值時未測。
  3. `5246eab6`（trtllm-gen MHA extend 的 max_arch 11.9）：decode 那支（`flashinfer_trtllm_mha_decode`）
     仍是 min 10.0 無上限，無 plugin、預設 `mha` backend 時會在 sm_121 被選中但沒驗證過。
- 確認是哪一個之後把這條改成具體的 症狀 → 根因 → 修法。

### F5. kernel 套件重組（upstream #2317/#2323 gemm/moe 依實作分目錄、#2278 移除未引用 wrapper）
- 我們改的 21 個 .py 與 plugin 全部通過靜態 import 檢查與 py_compile（Mac 上做的，沒有 CUDA）；
  plugin 的 gemm/moe 模組本來就沒註冊。真正的驗證要等 GB10 重跑 level 1。

---

## 量測紀錄（GB10, Qwen3.8-27B BF16, 輸出 128 token）

local 副本：`bench-results/<label>/`（bench JSON、metrics.json、gsm8k report、run.log）。

| 版本 | log | gsm8k | KV token | unique c1 / c64 tok/s | shared c64 tok/s | long c64 tok/s / TTFT p50 | 備註 |
|---|---|---|---|---|---|---|---|
| baseline（main + 4 bring-up fixes，BF16 KV，無 plugin） | 20261019-175223-window-54a85c2f | — | — | 4.3 / 89.8 | — | — | TPOT c1 223 ms ≈ 84% 頻寬上限 |
| baseline-bf16 | 20261020-075427-window-baseline-bf16-4d5ed6d0 | 0.945 | 597,504 | 4.3 / 90.0 | 122.4 | 22.8 / 108 s | 跟 10-19 那次差 < 1% |
| fp8-kv | 20261020-093322-window-fp8-kv-4d5ed6d0 | 0.970 | 1,201,152 | 4.3 / 92.0 | 124.5 | 24.4 / 76 s | 容量 2.01×；速度在雜訊內，TTFT c64 shared −22% / long −29% |
| plugin-fp8kv | 20261020-112217-window-plugin-fp8kv-cf5d6b15 | 0.965 | 1,207,296 | 4.3 / 92.6 | 125.2 | 24.4 / 80 s | 6 kernel 註冊；vs fp8-kv 只有 unique c64 TTFT −18%，其餘 ±7% 內 |
| prefill-proof（FP8 KV + plugin） | 20261020-133139-window-prefill-proof-07696f3d | — | 1,201,152 | — | longshared c64 115.9 | 24.5 / 78 s | long vs longshared 證明 prefill 等待 79% vs 17%（kernel_zoo E）；long c16 19.6（上一輪 22.3，差在等待） |
