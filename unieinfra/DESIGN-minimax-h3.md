# MiniMax-H3 在 UnieInfra 的實作設計

> 前置閱讀：`unieinfra/xref/16-diffusion.md`（四方橫向對照與落點）。這份文件把
> 該文 §3 的落點展開成可執行的設計。方法論依 `AGENTS.local.md`：
> 先 survey（已完成）、再 port、再優化；每個里程碑都要有可驗收的數字。

## 0　事實來源與版本

寫這份設計時實際查證過的東西，全部有版本：

| 來源 | 版本 / 路徑 | 用途 |
|---|---|---|
| HF `MiniMaxAI/MiniMax-H3` | `main`，**未 gated**、非 private | checkpoint 佈局、各元件 config |
| `diffusers` | **0.40.0（已發行）** | `MiniMaxH3Transformer3DModel`、`AutoencoderKLMiniMaxH3`、`AutoencoderKLMiniMaxH3Audio` 都在裡面 |
| SGLang Diffusion | `/workspace/roy/diffusion-refs/sglang-main/python/sglang/multimodal_gen/` | stage 管線形狀、量化與部署決策的參考 |
| 本箱 | 4x RTX PRO 6000 Blackwell，各 97887 MiB | 可行性算術 |

一個對計畫影響很大的事實先講：**H3 的 diffusers 支援已經在 0.40.0 正式版裡**，
不是 git main。`model_index.json` 寫 `_diffusers_version: 0.36.0.dev0`，容易讓人
以為要跟 main，實際上 0.40.0 的 `models/transformers/__init__.py` 與
`models/autoencoders/__init__.py` 都已匯出這三個類別。也就是說
「先包 diffusers 跑通」這條路可以**釘在一個發行版號上**，不必依賴 unstable branch。

## 1　可行性：兩個硬數字

### 1.1 磁碟：144 GB

repo 全量 498 GB，但那是三份重複打包。取一份即可：

```
transformer   66.28 GB      FL2VA/        144.05 GB   (= 上面四項)
text_encoder  66.73 GB      Ref2VA/       144.05 GB
vae           10.42 GB      transformer_ref 66.28 GB
audio_vae      0.61 GB
──────────────────────
              144.04 GB
```

`/workspace` 目前可用 **799 GB**。放得下，但這是一次 144 GB 的下載，要當成一個
獨立步驟排程，不要藏在某個測試腳本裡。

### 1.2 顯存：單卡放得下，但必須分段常駐

單卡 97887 MiB ≈ 102.6 GB（十進位）。

| 組合 | 大小 | 單卡 |
|---|---|---|
| text_encoder（bf16） | 66.73 GB | 放得下 |
| transformer + vae + audio_vae | 77.31 GB | 放得下，餘 ~25 GB 給 activation |
| **三者同時** | **144.0 GB** | **放不下** |

結論是明確的：**單卡 H3 可行，但 text encoder 與 DiT 不能同時常駐**。
流程必須是「編碼文字 → 釋放 encoder → 載入 DiT → 去噪 → 換 VAE 解碼」。
diffusers 的 `enable_model_cpu_offload()` 正好就是這個語意，M1 直接用它。

順帶澄清一件容易誤讀的事：SGLang 的 `validate_quality_deployment()` 把
`quality="high"` 鎖死在 **4xH200 + sp_degree=4 + ulysses_degree=4**，並且在別的
配置上直接 `raise`。那是**品質稽核的邊界**（他們只在那個配置上測過 SSIM/PSNR），
不是「H3 需要四張 H200 才能跑」。單卡跑得動，只是慢，而且沒有人替那個配置的
品質背書——我們自己的驗收因此不能引用他們的數字，要自己對 diffusers 比。

## 2　模型形狀

### 2.1 DiT（`transformer/config.json`）

```
num_layers            50          hidden_size        5376
num_attention_heads   56          attention_head_dim 128   -> inner_dim 7168
ffn_dim               14336       text_dim           5120
in_channels           24 (video)  audio_in_channels  32
patch_size            (1, 2, 2)   time_embed_dim     2688
rope_theta            10000.0     num_refiner_layers 2
```

注意 `56 x 128 = 7168 > hidden_size 5376`：**注意力的 inner dim 比殘差流寬**，
q/k/v 無 bias。這不是筆誤，是 H3 的架構選擇，port 的時候不能照抄一般
`hidden == num_heads * head_dim` 的假設。

參數量核對（用來確認理解沒錯，也用來算成本）：

```
每層 attn   5376x7168x3 + 7168x5376        = 154.1M
每層 mlp    5376x14336 x2                  = 154.1M
每層 adaln  Linear(2688, 6x5376x3=96768)   = 260.1M      <- 吃 temb，不吃序列
                                            ────────
                                            568.3M x 50 = 28.4B
```
加上 refiner、embedder、兩個輸出頭 ≈ 33B，對得上 66.28 GB / 2 bytes。

### 2.2 三模態打包成一條序列

diffusers 0.40.0 的 `transformer_minimax_h3.py`（664 行，註解很完整）揭露的契約：

```python
forward(hidden_states,          # video latent rows
        audio_hidden_states,    # audio latent rows
        encoder_hidden_states,  # text, 5120 dim
        timestep, ...)
```

內部做的事：三路各自 patch-project → 用 `index_copy` 塞進**同一條長度
`sequence_length` 的序列** → 50 層 block 一起跑 → 最後用 `index_select` 把 video 與
audio 的 row 各自取回。每個 row 帶一個 `token_tags`（0/1/2），與 timestep index 合成

```python
adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags   # MODALITY_NUM = 3
```

也就是 **AdaLN 是「每 row 依模態查表」**，一次 `adaln_proj(temb)` 產生
`(num_timesteps * 3, hidden)` 的調變表，block 內用 `index_select` 取。這個設計讓
video 與 audio 可以走**不同的 sigma 排程**卻共用一次 forward——這正是 H3 的核心，
也是它跟一般 video DiT 不同的地方。

混精度也是契約的一部分：`_keep_in_fp32_modules` 讓兩個 patch projection、
`time_embedder`、兩個輸出頭留在 fp32，block stack 走 bf16。逐塊替換時這條界線
不能自己改，否則逐步 latent 比對會從第一步就偏。

### 2.3 VAE 與時間軸

```
video VAE   spatial_downsample [2,2,2,2,1,1] -> 16x     latent_channels 24
            temporal_downsample [1,2,2,1,1,1] -> 4x     clip_length 17
audio VAE   encoder_rates [2,4,4,5,5] -> 800x           sampling_rate 32000
            -> 40 latent frames/s                       latent_channels 32
```

`MINIMAX_H3_SUPPORTED_FPS = 24`，時長 4.0–15.0 s，官方唯一有 recipe 與參考輸出的
短邊是 **768**（SGLang 對其他短邊會 warn 一次，說「跑得出來，但沒人測過品質」）。

## 3　成本結構：注意力主導，而且對時長是平方

以 768x1344、24 fps、8 秒算一次：

```
latent  t = 192/4 = 48      h = 768/16 = 48      w = 1344/16 = 84
patch (1,2,2)  -> video rows = 48 x 24 x 42 = 48,384
audio rows = 8 x 40                         =    320
text rows                                   ~    O(100)
                                              ────────
packed sequence S                           ~ 48,800
```

每個去噪步的 FLOPs：

```
線性層   2 x 15.4B(每 token 參數，已扣掉不吃序列的 adaln) x 48.8k = 1.50 PFLOP
注意力   50 層 x 2(QK^T, PV) x 2 x S^2 x 7168                    = 3.40 PFLOP
                                                                  ───────────
                                                                   4.9 PFLOP / step
```

**注意力是線性層的 2.3 倍**，而且隨時長平方成長（15 秒會讓注意力那項再乘 3.5，
線性項只乘 1.9）。這直接決定了 §8 的替換順序：**attention backend 是第一個值得換的
塊，不是 GEMM**。

（下面這行是算術不是量測，標明：若 bf16 有效算力落在 300–500 TFLOP/s，單步約
10–16 s，50 步約 8–14 分鐘。真正的數字要等 M1 跑出來，任何引用都要用實測值。）

一個推論：**H3 不可能走同步 HTTP**。M1 的 API 就要是 job 形式。

## 4　選哪一種 checkpoint 佈局

repo 裡有兩套，不相容：

| | root（`model_index.json`） | `FL2VA/model_index.json` |
|---|---|---|
| pipeline 類別 | `MiniMaxH3ModularPipeline` | `MiniMaxH3Pipeline` |
| diffusers | 0.36.0.dev0 → **0.40.0 已支援** | 0.32.2 |
| 元件類別 | `AutoencoderKLMiniMaxH3` 等，**發行版有** | `MiniMaxH3VideoVAE` 等，**只有 SGLang 有** |
| 誰在用 | diffusers 本家 | SGLang（`model_variant="fl2va"`） |
| 任務 | — | `t2va`, `fl2va`；`sigma_shift_scales {video:12, audio:3}` |

**兩份都要，而且先要 FL2VA（2026-08-22 修正）。** 初稿只選 root，理由是它能釘在
發行版 diffusers 上。但改成「先追平」之後，第一個要跑起來的是**參考引擎**，而
SGLang 的 `MiniMaxH3Pipeline._load_config()` **只接受 `FL2VA/` 或 `Ref2VA/`**：
它從子目錄讀 `model_index.json`，再用 `MiniMaxH3ReleaseMetadata.from_model_index()`
驗 `partition`，而該函式第一行就要 `_minimax_h3` 這個 key——**root 的
`model_index.json` 沒有這個 key，直接 `ValueError`**。root 佈局在 SGLang 上載不起來。

所以：**FL2VA 是 parity baseline 的 checkpoint，root 是之後 diffusers 側的正確性
參考。** 兩份都已在磁碟上（各 144 GB，見 §10）。順帶一提，這也解決了初稿擔心的
audio sigma shift 來源問題——SGLang 從 FL2VA 的中繼資料讀到
`{'video': 12.0, 'audio': 3.0}`，已用 CPU 驗過。

但 FL2VA 的 `_minimax_h3` 中繼資料仍要讀：`sigma_shift_scales` 與任務清單是
**模型常數**，root 的 `scheduler_config.json` 只給了 `shift: 12.0`（video），audio 的
3.0 只在 FL2VA 那份中繼資料裡。這一項要在程式碼裡寫死並註明出處，不能讓它 fallback
到 video 的 12.0——那會讓音訊排程整條錯掉，而且輸出仍然是「有聲音的影片」，
沒有任何 timing 數字會顯示這件事（參見 `well-formed-is-not-the-right-quantity`）。

## 5　落地架構

### 5.1 已經接好的一段

`python/tokenspeed/runtime_select.py` 的 `resolve_serving_runtime()` **已經**會對有
`model_index.json` 的 checkpoint 回傳 `"diffusion"`；`cli/__main__.py` 的
`_announce_runtime()` 目前對它明確拒絕。也就是說辨識這一段是好的，缺的是後面。

### 5.2 新增的檔案落點

```
python/tokenspeed/runtime/diffusion/
  __init__.py
  config.py          H3 常數（sigma shift、fps、時長界線、短邊）與 checkpoint 佈局解析
  components.py      分段常駐：載入 / 釋放 / 換手，包住 diffusers 的 offload
  pipeline.py        stage 管線（見下）
  job.py             job 佇列與狀態機（submit -> running(step i/N) -> done/failed）
python/tokenspeed/runtime/entrypoints/
  diffusion_http.py  job 形式的 HTTP，見 §5.4
test/runtime/diffusion/                 CPU 可跑的形狀 / 排程 / 錯誤路徑測試
unieinfra/bench/h3_latent_parity.py     逐步 latent 對照 harness（§7）
```

**不新增第二個引擎。** `xref/16-diffusion.md` §3 已經算過：整份 port SGLang 的
`multimodal_gen` 是 311k 行。這裡要的是一個小的、獨立的子系統，跟 LLM 那條路只共用
CLI 入口與 HTTP 慣例；paged KV、continuous batching、prefix cache 一個都用不上。

### 5.3 stage 管線

沿用 SGLang 的切法（它已經走過一次「先能跑再逐塊換」，所以它的邊界是可信的），
但只保留 H3 真正需要的那幾個：

```
validate            時長 4–15 s、fps=24、短邊（768 以外 warn 一次）、任務型別
text_encoding       Qwen3-VL -> hidden_states[text_len, 5120]           <- 之後可換到本樹
visual_encoding     首幀 / 參考影像 -> keyframe cond rows（fl2va 才有）
latent_preparation  固定 seed 的 randn -> video rows / audio rows
timestep_preparation  兩條 sigma 排程：video shift 12.0、audio shift 3.0
denoising           N 步，每步一次 packed forward，兩模態各自 Euler 更新
decoding            video VAE -> 影格；audio VAE -> 32 kHz 立體聲；封裝成檔案
```

去噪那一步的更新式，照 SGLang `denoise_loop.py` 的實作（flow-matching Euler）：

```
denoised = state + (1 - t) * velocity
state    = sigma_ratio * state + (1 - sigma_ratio) * denoised     # sigma_ratio = s[i+1]/s[i]
最後一步 sigma == 0 時直接取 denoised
```

video 與 audio 各自帶自己的 `sigma_ratio`，共用同一次 forward 的兩個輸出頭。

### 5.4 HTTP：job，不是同步請求

§3 算出單次生成是分鐘等級，所以：

```
POST /v1/videos              -> 202 {"id": "...", "status": "queued"}
GET  /v1/videos/{id}         -> {"status": "running", "progress": {"step": 12, "total": 50}}
GET  /v1/videos/{id}/content -> mp4 bytes
```

形狀對齊 OpenAI 的 video job API（image 的 `/v1/images/generations` 是同步的，
不適用）。`ts serve <h3-checkpoint>` 走 `_serve_diffusion(raw_argv)`，與
`_serve_pooling` / `_serve_asr` 同一個 dispatch，滿足「同一個啟動方式」。

**一次只跑一個 job。** 分段常駐的前提是同一時刻只有一組元件在卡上；併發 job 會
讓兩個 DiT 同時要 66 GB。佇列深度是設定值，預設 1，超過就排隊——這是
`AGENTS.local.md` 講的「讓不可能的狀態在啟動時就無法表示」。

## 6　M1：第一個垂直切片與驗收

範圍刻意最小：**t2va，768 短邊，4 秒，固定 seed，單卡，全部用 diffusers 的實作**。
不做 fl2va、不做參考影像、不做量化、不做 sp/tp。

驗收條件，三個都要過：

1. `ts serve <h3>` 起得來，`POST /v1/videos` 回 202，job 走到 `done`，產出可播放的
   mp4，**影片有畫面且音訊不是靜音**——而且要人工看過一次。
   （`a-negative-result-is-easy-to-produce-by-accident`：一個全黑但檔案結構正確的
   mp4 會通過所有自動檢查。）
2. §7 的逐步 latent 比對，對 diffusers 原生 pipeline **每一步**都在容差內。
3. 記錄實測的單步時間、峰值顯存、端到端時間。這三個數字之後每次替換都要重測，
   §3 的算術估計從此作廢，只引用實測。

## 7　驗收指標（2026-08-22 改寫）

**初稿這一節寫錯了。** 我原本訂的驗收是「固定 seed，逐去噪步比對 latent 的
`max|Δ|` 與 cos」。四方 survey 後（`xref/16-diffusion.md` §4.2）沒有任何一家這樣驗。
照他們的做法分成兩層：

| 層 | 內容 | 何時跑 |
|---|---|---|
| **sanity** | 有產生內容、形狀合法、**影格非全黑**、音軌非靜音 | 每次，廣泛地跑 |
| **品質閘** | **LPIPS**（或等價指標）對**參考框架**產生的 baseline，門檻寫死 | 動到數值的每一次 |

跑小：照 TRT-LLM 的 `ENGINEERING_CRITERIA.md` §5.2，E2E 用**減少的步數與解析度**
（他們對 FLUX 用 4 步、LTX-2 用 8 步、256×256）。變體矩陣留在離線。

**逐步 latent 比對降格為除錯工具，不是驗收指標。** 它回答的是「偏差從第幾步開始」，
在品質閘紅燈之後才拿出來用。把除錯工具當門檻是我初稿的錯誤。

換 kernel 時另外還有 SGLang 訂的**數值契約**（§8 表格第 1、2 項）：
kernel 要複製 H3 eager 的 BF16 捨入邊界，不得以數學等價為由改變縮並順序。

## 8　追平清單（2026-08-22 改寫）

**初稿這一節也寫錯了。** 我用自己的 FLOP 算術排出「attention 第一、GEMM 第四」。
那是我自己想的，不是量測出來的，也不是任何引擎做過的。正確的順序見
`xref/16-diffusion.md` §4.3——SGLang 已經替 H3 做過的 fast path 清單：

| # | fast path | 單卡適用 |
|---|---|---|
| 1 | indexed AdaLN modulation + gated residual 融合 | **是** |
| 2 | fused QK norm + RoPE（`round_norm_before_rope=True`） | **是** |
| 3 | packed Ulysses QKV + output relayout | 否，要 SP>1 |
| 4 | batched TP AdaLN collectives | 否，要 TP>1 |
| 5 | 最終投影前先移除死的 text/padding row | 否，要 TP>1 |

我排第一的「換掉 self-attention kernel」不在清單上——注意力本身已經是 FA/flashinfer
的函式庫呼叫；他們動的是它的**前處理**（QK norm + RoPE 融合）與每層兩次的
**indexed modulation**。單卡上能做的是 1 與 2。

**而且順序不由這張表決定，由量測決定。** §3 那些 FLOP 數字只能當量級感，
真正的順序要等 §6 的歸因跑完（`xref/16-diffusion.md` §4.1 的三層探針：
predenoise / denoise 第 i 步 / postdenoise）。SGLang 的規矩是：把一個熱點叫做
「新的」之前，先用他們的清單分類。

一個會擋掉不少計畫的事實：**H3 在 SGLang main 上永遠是 eager consistency case**，
他們的 preset 強制關掉 torch.compile 預設。所以「先 compile」「先捕 CUDA graph」
在追平之前都不成立——設計初稿 §8 末段講 CUDA graph 收益很大的那段，作廢。

## 9　依賴與提交切分

`AGENTS.md` 要求 runtime 保持 vendor-neutral，而這個計畫的第一步就是引入
`diffusers`。處理方式：

* `diffusers` 進 **optional extra**（如 `tokenspeed[diffusion]`），不進核心 runtime 依賴。
* `runtime/diffusion/` 以外的任何地方都不 import 它；`cli/__main__.py` 的
  dispatch 用 `try/except ImportError` 包住，缺 diffusers 時給一句能照著做的錯誤訊息。
* 這一步**獨立成一個 commit**，commit message 講清楚它是暫時的骨架依賴、以及 §8 的
  替換計畫，免得三個月後有人以為它是永久設計。

提交切分：

```
1. deps: diffusers as an optional extra for the diffusion runtime
2. runtime/diffusion: component residency and the H3 stage pipeline
3. entrypoints: job-style /v1/videos for diffusion checkpoints
4. cli: ts serve dispatches diffusion checkpoints
5. bench: step-wise latent parity against diffusers for MiniMax-H3
```

每個 commit 前跑 `pre-commit run --all-files`（完整那條，不能換成較窄的 lint），
並 sign off。

## 10　未解 / 需要決定

1. ~~**144 GB 下載要不要現在做。**~~ **已完成（2026-08-22）。** root 佈局 102 檔、
   144.09 GB，逐檔比對 HF manifest 大小全部相符、無 `.incomplete`，
   四個元件的 safetensors index 與 header 都通過。落點：

   ```
   /workspace/.cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
   ```

   （`HF_HOME=/workspace/.cache`，所以可以直接用 repo id `MiniMaxAI/MiniMax-H3` 解析；
   `/workspace` 抓完後餘 664 GB。）`resolve_serving_runtime()` 對這個路徑回傳
   `"diffusion"`，§5.1 說的辨識那一段確認可用。

   checkpoint 本身印證了 §2.2 的混精度契約：`transformer` 是 BF16+F32 混合、
   `vae` 與 `audio_vae` 純 F32、`text_encoder` 純 BF16。

   **抓的時候踩到一個坑，記在這裡免得下一個人重踩**：
   `hf download <repo> --exclude "FL2VA/*" "Ref2VA/*" ...` 會把那些 pattern 當成
   **要下載的檔名**（它自己會 warn `Ignoring --exclude since filenames have been
   explicitly set`，但那行淹在進度條裡），結果是抓了正好要排除的 42 GB。
   改用 `snapshot_download(allow_patterns=[...])` 正向列舉，並且**送出前先在本地
   用同一套 fnmatch 語意算一次選中集合**，確認檔數與位元組數符合預期再跑。
2. **GPU 2 上已經有一套 diffusion 部署**：`vLLM-Omni::DiffusionWorker-0` 加一個
   53 GB 的 `python3.13`（ComfyUI），已經跑了十幾天。那張卡是第三方的、不能碰。
   值得先問清楚那是誰的、在跑什麼——如果是同一個團隊的 H3/影像部署，這份設計的
   第一個切片可能已經有現成的參考輸出可以對，省掉一輪。
3. **參考引擎已就緒，但卡上不去。** SGLang main（pin `05c584c`）裝在
   `/workspace/roy/sglang-diffusion/.venv`（專用 venv，沒碰共用的 vLLM venv），
   `sglang 0.0.0.dev1+g05c584c44` / `torch 2.13.0+cu130` / `diffusers 0.37.0`。
   跑法在 `unieinfra/bench/h3_reference_run.sh`。

   **有一個會靜默毀掉基準線的環境陷阱**：本箱系統 nvcc 是 12.0，而 flashinfer 對
   SM 12.x 要求 CUDA >= 12.9，否則 `_normalize_cuda_arch()` 直接
   `raise RuntimeError("SM 12.x requires CUDA >= 12.9")`，`TARGET_CUDA_ARCHS` 變成
   空集合——它只印兩行 `Failed to get device capability`，然後帶著退化的 flashinfer
   繼續跑。venv 內自帶 CUDA 13.3 的 nvcc，把 `CUDA_HOME` 指過去就得到
   `TARGET_CUDA_ARCHS = {(12, '0f')}`。這條寫在 `sglang-diffusion/env.sh` 裡。

4. **本箱可用的卡。** 現在 GPU 1 與 GPU 3 空著（0 有別人的 tokenspeed scheduler，
   2 是第三方）。M1 只需要一張。跑之前照規矩 `hclaim`，並且用 `nvidia-smi` 確認，
   不是只看 claim。
4. **`ts serve <whisper>` 的 dtype 解析**仍未修（fp16 模型對上 bf16 KV cache）。
   跟 H3 無關，但同一個 dispatch 底下，列在這裡不讓它被忘掉。

---

附帶更正一則前文：`xref/16-diffusion.md` §3 初稿說「唯一能直接重用的是 text encoder，
Qwen3-VL 已經在 registry 裡」。查證後不成立，該節已就地更正——registry 裡有 `qwen3_vision.py`
（ViT tower）與 `qwen3_omni.py`（MoE 的 `Qwen3OmniMoeForConditionalGeneration`），
**沒有** dense 的 `Qwen3VLForConditionalGeneration`，而 H3 的 text encoder 正是後者
（64 層、hidden 5120、GQA 8、intermediate 25600，即 Qwen3-VL-32B）。所以「重用
text encoder」是 §8 的一個項目，不是既有能力；好消息是它的 text tower 就是 dense
Qwen3，vision tower 我們有，缺的是把兩者接起來的那一層。
