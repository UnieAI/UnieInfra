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

| 引擎 | 影像/影片 diffusion |
|---|---|
| **SGLang（main，pin 之後）** | **有**：`python/sglang/multimodal_gen/`，948 個 .py、311,716 行 |
| FastVideo | **有**：獨立框架，`fastvideo/{pipelines,models,attention,layers,workflow}` |
| vLLM | **沒有** —— 見下面那一節 |
| TensorRT-LLM | 無 |
| TurboMind | 無 |

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

## 3　對 UnieInfra 的落點

**先講清楚規模**：把這個子系統整份 port 進來不是一個 feature，是第二個引擎
（311k 行）。這棵樹既有的 embedding / ASR / paged KV / continuous batching 幾乎都用不上——
一個 diffusion 請求沒有 KV cache、沒有自迴歸 decode、沒有 token 輸出。

**唯一能直接重用的是 text encoder。** Qwen2.5-VL 與 Qwen3-VL 已經在 registry 裡
（`qwen3_vision.py`、`qwen3_omni.py`、`qwen3_5.py`），FLUX 的 T5/CLIP 沒有。

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
