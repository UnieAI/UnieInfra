# 16 — 影像/影片 diffusion：四方橫向對照與落點

範圍：**沒有 token 的那一段生成**。一個請求不是「prefill 然後逐 token decode」，而是
對固定大小的 latent 跑 N 次去噪 forward（N ≈ 20–50），再 VAE 解碼成像素或影格。

觸發這份文件的三個模型——`MiniMaxAI/MiniMax-H3`、`Qwen/Qwen-Image-Edit`、
`black-forest-labs/FLUX.1`——**共用同一個形狀**：

```
text encoder →  DiT 去噪迴圈（flow-matching scheduler）→ VAE decode
```

| 模型 | text encoder | transformer | VAE | 任務 |
|---|---|---|---|---|
| MiniMax-H3 | `MiniMaxH3Qwen3VLHFEncoder`（Qwen3-VL） | `MiniMaxH3DiTModel` | video VAE **＋ audio VAE** | t2va / fl2va |
| Qwen-Image-Edit | `Qwen2_5_VLForConditionalGeneration` | `QwenImageTransformer2DModel` | `AutoencoderKLQwenImage` | image edit |
| FLUX.1 | T5-XXL ＋ CLIP-L | `FluxTransformer2DModel` | `AutoencoderKL` | t2i |

H3 多一條 audio 支線（`sigma_shift_scales` 對 video 是 12.0、對 audio 是 3.0），
其餘三者一致。

## 0　版本與讀了哪些模組

**不動既有 pin。** `legacy/third_party/sglang` 停在 `b61a0692`（2026-08-07），那個時間點
**還沒有** diffusion 子系統。這一輪為 diffusion 另外 clone 兩份參考，放在
`/workspace/roy/diffusion-refs/`，與 `legacy/third_party/` 的 pin 互不影響：

| 參考 | pin | 內容 |
|---|---|---|
| `diffusion-refs/sglang-main` | `05c584c` (2026-08-22) | `python/sglang/multimodal_gen/`，即 SGLang Diffusion |
| `diffusion-refs/fastvideo` | `6d6a10b` (2026-08-21) | hao-ai-lab/FastVideo，獨立的影片生成推論框架 |

## 1　哪個引擎真的做了這件事

| 引擎 | 影像/影片 diffusion | 支援 H3？ |
|---|---|---|
| **SGLang（main，pin 之後）** | **有**：`python/sglang/multimodal_gen/`，948 個 .py、311,716 行 | **有** |
| **TensorRT-LLM** | **有**：`tensorrt_llm/_torch/visual_gen/`，**139 檔** | 無（最近的同類是 LTX-2） |
| FastVideo | **有**：獨立框架，`fastvideo/{pipelines,models,attention,layers,workflow}` | 無 |
| vLLM | **沒有** —— 見下面那一節 | — |
| TurboMind | **沒有**（已查證，見下） | — |

> **更正（2026-08-22）**：這張表初稿把 TensorRT-LLM 與 TurboMind 都寫成「無」，
> 那是**假陰性**。`legacy/third_party/` 底下三份參考全是 **sparse checkout**
> （`git sparse-checkout list` 可見），稀疏樣式本來就不含任何 diffusion 路徑，
> 所以在本地 `find | grep` 得到 0 命中是必然的，跟遠端有沒有無關。
> 不動 pin 的查法：對同一個 commit 打 GitHub tree API 列全樹再篩。結果是
> TensorRT-LLM `d8b67db2` 有 139 個 `visual_gen/` 檔案，lmdeploy `a1e5674`
> 全樹 1936 筆、`truncated: false`、0 命中——後者這次才算真的成立。
> （對照 memory `a-negative-result-is-easy-to-produce-by-accident`。）

### TensorRT-LLM `visual_gen` 的形狀

比 SGLang 小一個量級，而且結構對我們特別合：

```
visual_gen/
  pipeline.py  executor.py  pipeline_registry.py  cuda_graph_runner.py  profiler.py
  attention_backend/   flash_attn4 / trtllm / cute_dsl / vanilla
  cute_dsl_kernels/blackwell/          <- 9 檔，就是我們這張卡的世代
  cache/               teacache / cache_dit（跨步快取）
  models/              ltx2(48) wan(11) cosmos3(8) flux(8) qwen_image(4+3)
  modules/             attention / rms_norm / vae
  quantization/        loader / ops
  ENGINEERING_CRITERIA.md
```

`models/ltx2/ltx2_core/audio_vae/` 是**影片＋音訊聯合生成**，和 H3 的 audio 支線同類；
H3 本身它沒有。`AGENTS.md` 要求 NVIDIA kernel 優先用 CuteDSL，而
`cute_dsl_kernels/blackwell/` 正是 CuteDSL 寫的 Blackwell kernel——這是四家裡唯一
和本樹 kernel 方針對得上的一份。

### vLLM 的 "diffusion" 是表親，不是同一件事

`vllm/config/diffusion.py` 的 docstring 第一行就寫明：*"Configuration for discrete
diffusion (dLLM) models."* 它的 `canvas_length` 是「去噪 canvas 的長度，也決定每步排
幾個投機 token」，`diffusion_gemma.py` 是一個**產生文字**的 diffusion 語言模型。

**它和影像 diffusion 只共用一個詞。** 這正是 `AGENTS.local.md` 第 2 步警告的那種誤讀
（「讀實作同一個 feature 的模組，不是它的表親」）——我第一次 grep 到 `config/diffusion.py`
時確實把它當成了證據，讀了 docstring 才發現不是。記在這裡，因為下一個人 grep 會撞到
同一個字。

## 2　SGLang Diffusion 的形狀

它**不是**長在 LLM 那條路上的一個模型檔，而是一個平行子系統，有自己的
runtime、scheduler、loader、layers、entrypoints、distributed、cache——連
`breakable_cuda_graph` 這個名字都和我們這邊的同名概念重疊。

```
multimodal_gen/
  configs/pipeline_configs/   一個模型一個 config：flux.py qwen_image.py minimax_h3.py …（30 個）
  runtime/models/             dits/ vaes/ encoders/ schedulers/ decoders/ vocoder/ upsampler/
  runtime/pipelines/          一個模型一支 pipeline
  runtime/pipelines_core/     composed_pipeline_base.py + stages/ + schedule_batch.py
  runtime/managers/           scheduler.py dynamic_batch_admission.py gpu_worker.py
```

**執行模型是「組合式的 stage 管線」**，不是 token 迴圈：

```
input_validation → text_encoding → image_encoding → latent_preparation
                 → timestep_preparation → denoising → decoding
```

`ComposedPipelineBase` 把 stage 串起來，`add_standard_denoising_stage()` /
`add_progressive_denoising_stage()` 決定去噪那一段的形狀。批次由
`runtime/managers/dynamic_batch_admission.py` 管，和 LLM 的 continuous batching 是
兩套東西。

**diffusers 只被當工具用**，不是骨幹：出現最多的匯入是
`randn_tensor`(15)、`ConfigMixin`(9)、`VaeImageProcessor`(7)、`AutoencoderKLOutput`(6)、
`SchedulerMixin`(5)。DiT / VAE / encoder 本身是原生實作。

## 3　對 UnieInfer 的落點

**先講清楚規模**：把這個子系統整份 port 進來不是一個 feature，是第二個引擎
（311k 行）。這棵樹既有的 embedding / ASR / paged KV / continuous batching 幾乎都用不上——
一個 diffusion 請求沒有 KV cache、沒有自迴歸 decode、沒有 token 輸出。

**最接近能重用的是 text encoder，但今天還不能直接用。** registry 裡有
`qwen3_vision.py`（ViT tower）與 `qwen3_omni.py`（MoE 的
`Qwen3OmniMoeForConditionalGeneration`），**沒有** dense 的
`Qwen3VLForConditionalGeneration`——而 MiniMax-H3 的 text encoder 正是後者。
FLUX 的 T5/CLIP 也沒有。（初稿寫成「已經在 registry 裡」，2026-08-21 查證後更正。）

**已決定的做法**（2026-08-21，operator 選擇）：

1. **先包 diffusers 跑通**一條端到端的 pipeline，取得可比較的基準線，再逐塊把
   attention / DiT block / VAE 換成本樹的 kernel。這符合 `port-first-then-optimize`，
   代價是新增 diffusers 這個 runtime 依賴——`AGENTS.md` 說 runtime 要 vendor-neutral，
   所以這一步要獨立提交並標明。
2. **驗證用「對 diffusers 逐步 latent」**：固定 seed 與 scheduler，比對**每一個去噪步**
   之後的 latent，以及最終 VAE 輸出的像素。理由是去噪迴圈會把誤差放大，只比最終影像
   的話一旦不合就沒有歸因線索；逐步 latent 讓「第幾步開始偏」變成可讀的訊號。

**SGLang Diffusion 在這個計畫裡的角色是地圖，不是要整份抄**：它已經走過一次
「先能跑、再逐塊換」，所以它的 `runtime/models/dits/`、`vaes/`、`schedulers/` 告訴我們
哪些塊值得換、換的順序，以及每一塊的正確形狀。

**第一個垂直切片**應該挑三個模型裡最簡單的那個（FLUX 或 Qwen-Image，兩者都是單一
image VAE、沒有 audio 支線），把 stage 管線、scheduler、loader 的骨架立起來；
MiniMax-H3 的 audio VAE 與 t2va/fl2va 雙任務留到骨架穩定之後。

**尚未確認**：FLUX.1-dev 的 repo 是 gated（抓 config 回 401），要嘛用 FLUX.1-schnell，
要嘛提供 HF token。容器內目前**沒有 diffusers**。

**MiniMax-H3 的展開設計見 `unieinfer/DESIGN-minimax-h3.md`。** 兩件在那裡查證出來、
會改變本節判斷的事：H3 的 repo **未 gated**，而且 `diffusers` **0.40.0 正式版**已經
內含 `MiniMaxH3Transformer3DModel` / `AutoencoderKLMiniMaxH3`——所以「先包 diffusers」
這條路可以釘在發行版號上，而 H3 少了 FLUX 那道 gated 的關卡。

## 4　他們怎麼量、怎麼驗（2026-08-22 補）

初稿只比了「架構長什麼樣」，沒比**方法**。而方法正是 `AGENTS.local.md` 第 0 步與
第 2 步要的東西。以下兩張表是四方的實際做法，不是我的設計。

### 4.1 歸因：一個 diffusion 請求切成哪幾層

| 引擎 | 儀器 | 切法 |
|---|---|---|
| **TensorRT-LLM** | `visual_gen/profiler.py`，`TLLM_PROFILE_VISUAL_GEN_START_STOP` | `predenoise`（text encode + latent prep）／`A-B`（指定去噪步）／`postdenoise`（VAE decode）／`all` |
| **SGLang** | `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1` + `scripts/bench_diffusion_denoise.py` | 每個 stage 一段，且**同步**後才計時 |
| FastVideo | `fastvideo/profiler.py` | torch.profiler 包整條 |
| vLLM / TurboMind | — | 不適用 |

**兩家獨立收斂到同一個切法**：`text encode + latent prep` / `denoise 第 i 步` /
`VAE decode`。這就是 diffusion 版的探針階梯，對應 `AGENTS.local.md` §0 那張
HTTP→prefill→decode 表。TRT-LLM 的 profiler docstring 還記了一個他們自己的限制：
`postdenoise` 是綁在**第一個** denoise loop 之後，所以在 LTX-2 那種兩段式 pipeline 上
它的視窗是 VAE decode 的**超集**——多段 pipeline 要隔離 decode，今天沒有任何模式做得到。
H3 是單段，這個坑對我們不成立，但抄的時候要知道它為什麼長這樣。

兩條他們寫下來的量測紀律，跟本樹既有的教訓是同一件事：

* SGLang **Native Backend Gate**：log 只要出現 `Falling back to diffusers backend` /
  `Using diffusers backend` / `Loaded diffusers pipeline`，**立刻停止**，那次的數字與
  trace 都不得當作引擎證據。對照本樹 memory
  `verify-the-reference-engine-is-on-its-fast-path`（vLLM 的 RMSNorm 在裸
  `VllmConfig()` 下跑 eager，換來 13x 的假勝）。
* SGLang：`--profile` 會顯著改變延遲，**before/after 的宣告要用非 profile 的 perf dump**。
  對照 memory `profiler-cpu-times-are-inflated`（本箱實測 3.6x 灌水）。

### 4.2 parity：怎麼證明「換掉」不是「換壞」

| 引擎 | 指標 | 門檻 |
|---|---|---|
| **TensorRT-LLM** | **LPIPS**（或等價）對**參考框架**產生的 baseline 影像/影片 | 每個 checkpoint 每個任務一個**明確門檻**；lossless 類工作對既有 golden 算 LPIPS，「差一點點可接受，差多了就是紅旗」 |
| **SGLang** | 每個 fast path 自帶**數值契約**（見 4.3），加 H3 consistency check | kernel 必須複製 eager 的 BF16 捨入邊界 |
| FastVideo | `evaluator.py` / `metric.py`，含 LPIPS | — |

TRT-LLM 還把測試分兩層（`ENGINEERING_CRITERIA.md` §5.1）：廣泛跑的**便宜 sanity**
（有產生內容、形狀合法、**影格非全黑**），與只在該把關處跑的**品質閘**
（LPIPS / VBench）。E2E 刻意跑小：**FLUX 4 步、LTX-2 8 步、256×256**。

**這推翻了本專案設計初稿的 §7。** 我原本寫「固定 seed，逐去噪步比對 latent 的
`max|Δ|` 與 cos」——四家沒有任何一家這樣驗。逐步 latent 是**歸因用的除錯工具**
（偏差從第幾步開始），不是驗收指標；驗收指標是對參考框架輸出的 LPIPS，配一層
非全黑 sanity。兩者要分開，不能拿除錯工具當門檻。

### 4.3 H3 的追平清單（SGLang 已經做過的 fast path）

SGLang 的 `existing-fast-paths.md` 明列了 H3 專屬的優化，並訂了一條規矩：
**在把某個熱點叫做「新的」之前，先用這份清單分類**。這份清單就是「先追平」的標的：

| # | fast path | 位置 | 單卡是否適用 |
|---|---|---|---|
| 1 | **indexed AdaLN modulation + gated residual 融合** | `triton/indexed_modulation.py` | **是** |
| 2 | **fused QK norm + RoPE** | `fused_inplace_qknorm_rope`，`round_norm_before_rope=True` | **是** |
| 3 | packed Ulysses QKV + output relayout | `triton/ulysses_qkv.py`, `usp_relayout.py` | 否（要 SP>1） |
| 4 | batched TP AdaLN collectives（整疊 block 一次 all-gather） | `dits/minimax_h3.py` `_can_batch_block_adaln()` | 否（要 TP>1） |
| 5 | 最終投影前先移除死的 text/padding row 再做 TP column gather | `dits/minimax_h3.py` | 否（要 TP>1） |

**這和我初稿 §8 用 FLOP 算術排出來的順序不一樣，以這張表為準。** 我排的第一名是
「換掉 self-attention kernel」——它不在他們的清單上，因為注意力本身已經是
FA/flashinfer 的函式庫呼叫；他們動的是**注意力的前處理（QK norm + RoPE 融進去）**
與**每層兩次的 indexed modulation**。單卡上真正能做的是 1 與 2，其餘三項要多卡才有意義。

兩個必須跟著抄的約束：

* 第 1 項的**數值契約**：kernel 必須複製 H3 eager 的 BF16 捨入邊界，
  「不得用數學上等價的縮並取代，除非通過 H3 consistency check」。
* 第 2 項的 `round_norm_before_rope=True` 同樣是 H3 eager 數值契約的一部分，不是效能旋鈕。

另外一條會影響任何 M1 計畫的事實：**H3 在 SGLang main 上「永遠是 eager consistency
case」**，他們的 preset 會強制關掉全域的 torch.compile 預設。所以任何「先 compile 起來」
或「先捕 CUDA graph」的計畫，在追平之前都不成立。
