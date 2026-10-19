# MiniMax-H3 追平基準線：SGLang Diffusion 單卡實測

日期 2026-08-22　卡：1x RTX PRO 6000 Blackwell（sm_120, 97887 MiB）
參考引擎：SGLang main pin `05c584c`，`sglang 0.0.0.dev1+g05c584c44` /
`torch 2.13.0+cu130` / `diffusers 0.37.0`
被測 commit（perf dump 自動記錄）：`3e469489`
跑法：`unieinfra/bench/h3_reference_run.sh {smoke,nightly} <gpu>`

## 0　這是什麼、不是什麼

H3 在四方 survey 裡**只有 SGLang 支援**（`xref/16-diffusion.md` §1），所以「追平」
的對象就是這裡的數字。

**是**：單卡、eager、`--performance-mode=memory` 的端到端分段歸因。
**不是**：可以拿去跟 SGLang 公佈值比的數字。他們的 nightly preset 是
**4 卡**（`--tp-size=2 --ulysses-degree=2`），本表是 1 卡（`tp=sp=ulysses=ring=1`，
text encoder 與 VAE CPU offload）。唯一改動的就是並行度與 residency 模式，其餘
（prompt、seed 1101、短邊、時長、步數、flow_shift 12.0 / audio 3.0、eager）照抄他們的 preset。

## 1　分段歸因

`SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`，非 profile 的 perf dump（`--profile`
會顯著改變延遲，他們自己的 skill 明講 before/after 要用非 profile 的那份）。

| stage | SMOKE 384/4.0s/8步 | | NIGHTLY 768/5.0s/50步 | |
|---|---:|---:|---:|---:|
| TextEncoding | 0.46 s | 0.4% | 0.38 s | 0.1% |
| VisualEncoding | 0.0006 s | ~0 | 0.0006 s | ~0 |
| AudioEncoding | 0.00003 s | ~0 | 0.00004 s | ~0 |
| LatentPreparation | 0.005 s | ~0 | 0.017 s | ~0 |
| TimestepPreparation | 0.001 s | ~0 | 0.0003 s | ~0 |
| **Denoising** | **55.17 s** | **52.3%** | **521.04 s** | **90.4%** |
| Decoding（video+audio VAE） | 10.31 s | 9.8% | 15.38 s | 2.7% |
| **（未歸屬）** | **39.53 s** | **37.5%** | **39.74 s** | **6.9%** |
| **total** | **105.47 s** | | **576.55 s** | |

去噪逐步：

| | 記錄步數 | step 0 | 其餘平均 | min | max |
|---|---:|---:|---:|---:|---:|
| SMOKE | 7（要求 8） | 2.68 s | **1.20 s** | 1.20 | 1.20 |
| NIGHTLY | 49（要求 50） | 12.56 s | **9.82 s** | 9.67 | 9.86 |

輸出：nightly 是 1344x768 h264 @24fps 5.167 s ＋ AAC 立體聲 32 kHz 5.175 s。

## 2　三個直接可用的發現

### 2.1 有一段固定 ~39.6 s 不在任何 stage 裡

兩次 run 的未歸屬時間是 **39.53 s 與 39.74 s**——工作量差了一個量級，這段幾乎不動。
它是固定成本，位置在 stage 計時之外，最可能是 `component_manager` 的常駐搬遷
（`memory` 模式下把 DiT 搬上卡、把 text encoder 搬下去）。

**這件事對短請求是致命的**：在 smoke 上它佔 **37.5%**，比整個 VAE decode 還大。
而 TRT-LLM 與 SGLang 的探針切法（`predenoise / denoise 第 i 步 / postdenoise`）
**都看不到它**——它不在任何一個窗口裡。要歸因得另外量。

### 2.2 逐步時間極穩，warmup 只有第一步

其餘步的 min/max 是 1.20/1.20 與 9.67/9.86——±1% 以內。這對量測是好消息：
去噪步是一個乾淨的測量對象，不需要 `benchmark-period-blocked-means` 那種
處理極限環的手法。第一步多出 1.5 s / 2.7 s，是單次 warmup。

### 2.3 我先前的 FLOP 推論，被時鐘打了折

設計初稿算出「注意力是線性層的 2.3 倍」。用兩個實測點檢查這個推論：

```
packed 序列長度   smoke  24 x 12 x 21 =  6,048 rows     (384x672,  4 s)
                nightly  30 x 24 x 42 = 30,240 rows     (768x1344, 5 s)
                比值 5.00x
每步時間          1.20 s -> 9.82 s     比值 8.18x
                縮放指數 log(8.18)/log(5.00) = 1.31
```

純線性層是 1.0、純注意力是 2.0。實測 **1.31**。若把每步時間拆成 `a·S + b·S²`，
1.31 對應 **注意力約佔 31%、線性層約佔 69%**——和我算的「注意力 2.3 倍」**方向相反**。

這是兩點擬合，中間同時變了解析度與時長，所以它**不是**「注意力只佔 31%」的證明；
它足以說明的是：**我那個 2.3:1 不能拿來排優化順序**。要知道真的花在哪，
下一步是 per-op profile（`torch.profiler` / Nsight），不是再算一次。

## 3　量測本身踩到的兩個坑

兩個都不是 H3 的問題，是 harness 的問題，記在這裡因為兩個都會產生「看起來成功的失敗」。

### 3.1 `sglang generate` 在請求失敗時仍然 exit 0

第一輪 smoke 與 nightly **兩次都 CUDA OOM**，log 裡有完整的
`torch.OutOfMemoryError` 與 `Generation failed`，而 process **exit 0**。我的 runner
信了 exit code，於是在 smoke 已經掛掉的情況下還去跑了 nightly。

三個訊號當時都在，只是我沒先看：smoke（8 步/384）101 s 對 nightly（50 步/768）
104 s——工作量差 50 倍時間沒動；`perf.json` 兩次都不存在；沒有任何輸出檔。

現在的閘（`h3_reference_run.sh`）：

| gate | 擋什麼 |
|---|---|
| 1 | 卡的空閒記憶體不足就拒跑，並印出需要的數字 |
| 2 | SGLang 的 Native Backend Gate：log 出現 `Falling back to diffusers backend` 就非零退出 |
| 3 | 掃 log 找 `Generation failed` / `OutOfMemoryError` / `Error executing request` |
| 4 | 產物必須存在且大小合理 |
| 5 | `h3_sanity.py`：非全黑、非平面、有動作、音訊非靜音 |

Gate 3 與 4 是拿第一輪那份失敗 log 實測過會開火才算數的。

### 3.2 把 4 卡 preset 的欄位原封搬到單卡

OOM 的成因是我沿用了他們 preset 的 `--performance-mode=speed`。引擎自己的錯誤
訊息裡就寫著「For single-GPU deployment, use `--performance-mode memory`,
component CPU offload, or `--dit-layerwise-offload`」。改 `memory` 之後直接就過了。

這正是 `AGENTS.local.md` 說的「不要假設參考引擎的預設解析成什麼」——差別在於
這次不是沒查預設，是**照抄了一個為不同拓撲而設的值**。

## 4　下一步

1. **per-op profile 那 9.82 s 的去噪步**，用 `torch.profiler`（非 profile 的 perf dump
   已經有了，可以對照 profiler 的灌水倍率）。目標是把 §2.3 的兩點擬合換成真的分佈。
2. **量那 39.6 s 的未歸屬時間**是不是 residency 搬遷；如果是，它是短請求上最大的
   單一目標，而且兩家引擎的探針都看不到它。
3. 上述兩件做完，才輪到 `xref/16-diffusion.md` §4.3 那張追平清單
   （單卡可做的是 indexed AdaLN modulation 與 fused QK norm + RoPE）。
   SGLang 的規矩是：把熱點叫做「新的」之前，先用他們的清單分類。
