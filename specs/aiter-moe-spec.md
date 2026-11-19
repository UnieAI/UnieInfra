# aiter MoE 整合規格 — DeepSeek-V4.1-Flash-1017 / MI350P (gfx950) / TP4

目的：讓 unieinfra（TokenSpeed, torch 2.14.0+rocm7.2, container `ts-dev`）以
與 vLLM 完全相同的方式呼叫 aiter fused MoE（a8w4 = fp8 動態活化 × mxfp4 權重）。

驗證基礎（全部可重跑，見 §F）：
- vLLM 側：container `priceless_hypatia`，vLLM 0.33.0rc1.dev320+g335dd77b0，torch 2.13.0+rocm10.0.0，
  flags `--moe-backend aiter --quantization deepseek_v4_fp8 -tp 4 --language_model_only`，
  env `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1`，**未**設 `VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4`；
  日誌 `/tmp/deepseek-server.log`。
- 引擎側：在本 session 內以 `PYTHONPATH` 直接使用 aiter wheel（未安裝）於 `ts-dev`
  (torch 2.14.0+rocm7.2, 4×MI350P) 實跑 `aiter.fused_moe`，**成功**，kernel 選擇與 vLLM 一致（§D5）。
- 模型幾何（`/data/models/DeepSeek-V4.1-Flash-1017/config.json` → `text_config`）：
  hidden 5120、moe_intermediate_size 2304、40 layers、n_routed_experts 384、num_experts_per_tok 6、
  hidden_act silu、**swiglu_limit 10.0**、routed_scaling_factor 1.5；
  `quantization_config={fp8, dynamic, weight_block [32,32], scale_fmt ue8m0, expert_dtype fp4}`。
  TP4 ⇒ 每卡 inter_dim = 2304/4 = 576 → 對齊後 **640**；hidden 5120（不變）；cu_num 128。

---

## A. `--moe-backend aiter` 到底跑哪個 MoE 實作

### A1 決策鏈（程式碼路徑）

1. `vllm/model_executor/layers/fused_moe/layer.py` 依 `--moe-backend aiter` →
   `fused_moe/oracle/mxfp4.py:select_deepseek_v4_mxfp4_moe_backend()`（L780-851）。
2. 該函式用 `_return_or_raise(weight_key=...)`（L612-628）挑 backend；DSV4 checkpoint 的
   expert 權重是 packed fp4 + ue8m0 靜態 scale ⇒ `weight_key = kMxfp4Static`。
   **沒有** `VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4` ⇒ 不會選 a4w4 變體。
3. 結果：**`Mxfp4MoeBackend.AITER_MXFP4_BF16`**。
   日誌實證（`/tmp/deepseek-server.log:77`）：
   `[mxfp4.py:626] Using 'AITER_MXFP4_BF16' Mxfp4 MoE backend.`
4. Expert 類別：`fused_moe/experts/rocm_aiter_moe.py:673 AiterMxfp4Experts`
   （`_supports_quant_scheme → [(kMxfp4Static, None), (kMxfp4Static, kMxfp4Dynamic)]`），
   prepare/finalize 用 `MoEPrepareAndFinalizeNoDPEPModular`（無 EP、無 DP）。
   日誌實證（log:149-150）：`Using MoEPrepareAndFinalizeNoDPEPModular` / `Using AiterMxfp4Experts`。
5. 最終呼叫：`rocm_aiter_ops.fused_moe(...)`（`_aiter_ops.py:3547`）→
   `_rocm_aiter_fused_moe_impl`（`_aiter_ops.py:328-422`）→ **`aiter.fused_moe(...)`**
   （`vendor/aiter-src/fused_moe.py:1084`）。

> TP4 且非 EP：`expert_mask=None`、`num_local_tokens=None`，即「全部 384 個 expert 都在本地」。

### A2 實際送進 aiter 的參數（照抄，順序即 positional 順序）

```python
aiter.fused_moe(
    hidden_states,                       # bf16 [M, 5120]  ← 未量化，AITER 內部量化
    w13,                                 # fp4x2 [384, 1280, 2560]  (2*640, 5120/2)
    w2,                                  # fp4x2 [384, 5120, 320]    (5120, 640/2)
    topk_weight,                         # float32 [M, 6]
    topk_ids,                            # int32   [M, 6]
    expert_mask=None,
    activation=ActivationType.Silu,      # hidden_act=silu
    quant_type=QuantType.per_1x32,       # QuantMethod.BLOCK_1X32
    doweight_stage1=False,
    w1_scale=w13_scale,                  # uint8/e8m0 [384*1280, 160]
    w2_scale=w2_scale,                   # uint8/e8m0 [384*5120, 24]
    a1_scale=None, a2_scale=None,        # 動態量化，AITER 自己算
    dtype=torch.bfloat16,                # output_dtype
    hidden_pad=0,                        # 5120-5120
    intermediate_pad=64,                 # 640-576
    gate_mode=GateMode.INTERLEAVE.value, # 預設值；4w4 才是 SEPARATED
    swiglu_limit=10.0,                   # moe_config.swiglu_limit
    bias1=None, bias2=None,              # DSV4 無 expert bias（bias 轉 fp32 只是保險）
    # num_local_tokens / moe_sorting_dispatch_policy / splitk / output 皆用預設
)
```

三個「必須相等」的值（寫錯就是靜默錯結果）：
- `quant_type=QuantType.per_1x32` ↔ 權重 scale 是 1×32 block（最後一維 = K/32）。
- `gate_mode=GateMode.INTERLEAVE` ↔ 權重是用 `is_guinterleave=True` shuffle 的（§B4）。
- `activation=ActivationType.Silu` ↔ `swiglu_limit=10.0`（Silu + clamp，非 SITU）。

`q_dtype_a`（活化型別）不是呼叫端指定，而是 aiter 依 `AITER_BF16_FP8_MOE_BOUND` 解析
（`fused_moe.py:999-1081 resolve_activation_dtype`）。**vLLM 在權重準備時把該變數設為 `"0"`**
（`fused_moe/oracle/mxfp4.py:1839`），因此 decode 小 batch 也用 **fp8** 活化，而不是 bf16。
實測（§D5）：bound=0、M=16 ⇒ `q_dtype_a='torch.float8_e4m3fn'`；
bound=256、M=16 ⇒ `q_dtype_a='torch.bfloat16'`（走 a16w4 kernel 家族，與 vLLM 不同）。
⇒ **引擎必須在載入權重前 `os.environ["AITER_BF16_FP8_MOE_BOUND"]="0"`，且之後不要改。**

---

## B. 載入期權重準備契約（一次性）

### B1 padding 規則

`fused_moe/oracle/mxfp4.py:854-930 mxfp4_round_up_hidden_size_and_intermediate_size`
的 ROCm 分支（L900-930）：`aiter_uses_128 = (backend == AITER_MXFP4_BF16)`；
silu/situ ⇒ `intermediate_alignment=128`、`hidden_alignment=128`。
⇒ hidden 5120（已對齊 128，不變）、per-partition inter 576 → **640**（= ceil(576/128)*128）。
`quantization/mxfp4.py:185-200 maybe_roundup_sizes` 呼叫它，權重/scale 一律先用 `torch.zeros`
配置（L222-285）⇒ **padding 列/欄全 0**，在 stage1/stage2 GEMM 中貢獻 0（安全）。
`rocm_aiter_moe.py:239-274 aiter_moe_pads` + `L385-412` 把同一組差值當 `hidden_pad`/`intermediate_pad`
傳給 aiter：`hidden_pad = 5120-5120 = 0`、`intermediate_pad = 640-576 = 64`。

### B2 shuffle 函式（**純 python，無 C++ op**，`aiter/ops/shuffle.py`）

| 對象 | 函式 | 位置 |
|---|---|---|
| w13（gate+up） | `shuffle_weight(w13.view(float4_e2m1fn_x2), is_guinterleave=True, gate_up=True)` | `ops/shuffle.py:149`，guinterleave 分支 L175-198 |
| w2（down） | `shuffle_weight(w2.view(float4_e2m1fn_x2), is_guinterleave=True, gate_up=False)` | 同上 |
| w13 scale | `shuffle_scale(s13.reshape(-1, s13.shape[-1]), E, True, True)` | `ops/shuffle.py:343` |
| w2 scale | `shuffle_scale(s2.reshape(-1, s2.shape[-1]), E, True, False)` | 同上 |

呼叫端程式碼（vLLM 原文，`mxfp4.py:1841-1881`）：
```python
os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"                       # L1839
from aiter.ops.shuffle import shuffle_scale as _shuf_s
from aiter.ops.shuffle import shuffle_weight as _shuf_w
is_guinterleave = not use_separated_a4w4                            # L1848（我們 = True）
w13_weight = Parameter(_shuf_w(w13_weight.data.view(torch.float4_e2m1fn_x2),
                               is_guinterleave=is_guinterleave, gate_up=True))
shuffled_w13_scale = _shuf_s(w13_weight_scale.reshape(-1, w13_weight_scale.shape[-1]),
                             num_experts, is_guinterleave, True)          # L1858
w2_weight = Parameter(_shuf_w(w2_weight.data.view(torch.float4_e2m1fn_x2),
                              is_guinterleave=is_guinterleave, gate_up=False))
shuffled_w2_scale = _shuf_s(w2_weight_scale.reshape(-1, w2_weight_scale.shape[-1]),
                            num_experts, is_guinterleave, False)
w13_weight.is_shuffled = True; w2_weight.is_shuffled = True          # L1880-1881
```

### B3 形狀 / dtype / contiguity（實測於 `ts-dev`，E=8 子集，TP4 幾何）

| tensor | 進入前（checkpoint→per-rank，packed uint8） | shuffle 後（aiter 實收） | dtype |
|---|---|---|---|
| w13 | `[E, 1280, 2560]` | `[E, 1280, 2560]`（同形狀，內部重排） | `torch.float4_e2m1fn_x2` |
| w13_scale | `[E, 1280, 160]` | `[E*1280, 160]`（E=8 → `[10240,160]`） | `torch.float8_e8m0fnu` |
| w2 | `[E, 5120, 320]` | `[E, 5120, 320]` | `torch.float4_e2m1fn_x2` |
| w2_scale | `[E, 5120, 20]` | `[E*5120, 24]` ← K/32=20 補到 8 的倍數 | `torch.float8_e8m0fnu` |

- checkpoint 原始（`layers.6.ffn.experts.0.*`）：`w1.weight [2304,2560] I8`、`w1.scale [2304,160] F8_E8M0`、
  `w3` 同、`w2.weight [5120,1152] I8`、`w2.scale [5120,72] F8_E8M0`。⇒ fp4 packed（2/byte）、
  scale 沿 K 每 1×32 一個 e8m0。
- shuffle 輸出需 `.contiguous()`（`shuffle_weight` 內部已 `contiguous()`；scale 由 permute 而來，
  建議顯式 `.contiguous()`）。
- scale padding：`shuffle_scale`（L343+）把 K//32 補到 8 的倍數時填 **0x7F（= 2^0=1.0）**，
  N 方向 padding 由呼叫端（`torch.zeros`→0）負責。
- 期望數值分佈：e8m0 沒有 sign/NaN，合法值 0x7F 代表 1.0；實測 per-1x32 scale 落在 120..127
  （≈2^-7..2^0）屬合理。**不要**把 scale 當 float8（e4m3）讀，它是 e8m0（只有指數）。

### B4 `is_shuffled` 與 gate/up 交錯語義

- `shuffle_weight` 在 guinterleave 分支（L175-198）把 gate/up 視為
  `(E, 2, N0, NLane, K0, KLane, KPack)`，permute 次序 `(0,2,1,4,5,3,6)`（`gate_up=True`），
  輸出形狀不變，並在回傳 tensor 上設 `is_shuffled=True`／`gate_up_interleave=gate_up`。
- 語義：**INTERLEAVE = 權重列交錯排列**（gate[0], up[0], gate[1], up[1], …）⇒ aiter stage1
  可以用單一 B-tile stream 同時算 gate/up。`gate_mode=SEPARATED` 則是「前 640 列 gate、後 640 列 up」。
- 我們這條路徑 `is_guinterleave = True` ↔ `gate_mode = INTERLEAVE`。**兩者必須一致**，
  否則不會報錯、只會輸出垃圾（vLLM 註解原文：`INTERLEAVE + fp4x2 has no tuned kernel and
  produces garbage output`——指的是 a4w4/SEPARATED 情境；“mismatch ⇒ garbage” 對兩個方向都成立）。
- kernel 名尾碼 `_gui` = gate/up interleaved weights；`_w2`/`_w3`/`_bnt0`/`_atomic` 分別是
  stage1 的 wave 數 hint 或 stage2 的原子累加模式。

### B5 引擎側最省事的作法

1. 讀 checkpoint 的 expert 權重（uint8 packed）→ `view(torch.float4_e2m1fn_x2)`。
2. 直接對 **640 / 5120 已 pad 的 buffer** 做上面 4 個 shuffle；padding 區保持 0。
3. 不要自己寫 shuffle kernel：`aiter.ops.shuffle` 是純 python（view/permute/contiguous），
   在 torch 2.14 上實測可正常執行（§D5）。

---

## C. decode 形狀實際跑哪個 gfx950 kernel

### C1 查表 key 與決策點

`aiter.fused_moe` 進 `get_2stage_cfgs`（`fused_moe.py:3349+`）。實際 runtime key（vLLM 日誌原文）：

```
('gfx950', 128, <token>, 5120, 640, 384, 6, <ActivationType.Silu: 0>,
 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float4_e2m1fn_x2',
 'QuantType.per_1x32', True, False)
```

= (gfx, cu_num, M, hidden, inter_dim_padded, E, topk, act, out_dtype, q_dtype_a, q_dtype_w,
quant_type, use_g1u1=True, doweight_stage1=False)。

- 查表檔：`AITER_CONFIGS.AITER_CONFIG_FMOE_FILE`（`fused_moe.py:3515`）＝
  `$AITER_CONFIG_FMOE` 或 `<aiter>/configs/tuned_fmoe.csv`，並與
  `<aiter>/configs/model_configs/*tuned_fmoe*.csv` 合併（`jit/core.py:233 get_config_file`），
  合併結果寫到 **`/tmp/aiter_configs/tuned_fmoe.csv`**（`jit/core.py:646`，需要 /tmp 可寫）。
  實測：合併後 3135 行，含 column `kernelName1/kernelName2`；**沒有任何一列命中我們的 key**
  （vLLM 與引擎兩邊都印 `no tuned FlyDSL config …, using heuristic FlyDSL fallback`）。
- 為什麼不是 asm/CK：`get_2stage_cfgs` L3384 `cktile_mxfp4_unsafe = q_dtype_w==fp4x2 and
  inter_dim % 256 != 0`；我們 inter_dim=640 ⇒ 640%256=128 ≠ 0 ⇒ True，CK-Tile 分支（L4100-4140）
  被跳過；`_can_reroute_mxfp4_to_flydsl`（L3318）要求 `q_dtype_a==bf16`（我們是 fp8）⇒ False。
  最終走 **FlyDSL heuristic 分支（L4270-4372）**，`MOEMetadata(block_m=tile_m, ksplit=-1, run_1stage=False)`。

### C2 實測 kernel 名（vLLM 日誌 log:167-201，與引擎實跑一致）

| M（token 數） | stage1 `kernelName1` | stage2 `kernelName2` |
|---|---|---|
| 1–1024（含 16/64/512） | `flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui` | `flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_bnt2` |
| 2048 | `flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w3_bnt0_gui` | `flydsl_moe2_afp8_wfp4_bf16_t64x128x128_atomic` |
| 8192 | `flydsl_moe1_afp8_wfp4_bf16_t128x128x256_w2_bnt0_gui` | `flydsl_moe2_afp8_wfp4_bf16_t128x128x128_atomic` |

命名文法（`ops/flydsl/moe_kernels.py:190 get_flydsl_kernel_params`）：
`flydsl_moe{1,2}_{a_type}_{w_type}_{out_type}_t{tile_m}x{tile_n}x{tile_k}[_{wN|_bnt2|atomic}][_gui]`。
⇒ **decode（c1–c128）就是 `t32x128x256_w2_gui` + `t32x128x128_atomic_bnt2` 這一組**，
tile_m=32、block_m=32、ksplit=-1、`run_1stage=False`（2-stage）。

### C3 a4w4（fp4×fp4 asm）可達性

**不可達（在我們的設定下）。** 條件：(a) `gate_mode=SEPARATED`（只有設
`VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4=1` 才會，`rocm_aiter_moe.py:409-435`）；
(b) M ≥ `_SWIGLU_MXFP4_BF16_BOUND`(256) 且活化解析成 fp4。
我們 `gate_mode=INTERLEAVE` + `AITER_BF16_FP8_MOE_BOUND=0` ⇒ 活化固定 fp8 ⇒ asm fmoe、
CK 2-stage、`moe_gemm_a8w4`/Triton 都不會被選到。
（若之後要試 a4w4：還必須改成 `shuffle_weight(..., is_guinterleave=False)` 且把
 `gate_mode=SEPARATED` 一起改，並確認 M≥256，否則會退化成 bf16 或垃圾輸出。）

---

## D. 從 unieinfra 呼叫的最小自足 python

### D0 需要的套件（`ts-dev` 目前**都沒有**）

從 `priceless_hypatia` 的 `/usr/local/lib/python3.12/dist-packages/` 取 4 個東西（總約 2.9 GB）：

| 套件 | 大小 | 必要性 |
|---|---|---|
| `aiter/` (amd_aiter 0.1.26) | 2.3 G | 本體，含 133 個 prebuilt `jit/*.so` |
| `aiter_meta/` | 335 M | **import 期必要**（`aiter/__init__.py:137 → ops/attention.py:15 → csrc.cpp_itfs.pa`）；亦提供 `hsa/`、`3rdparty/`（若需 JIT 重編） |
| `flydsl/` + `flydsl.libs/` (0.4.0) | 268 M + 2.3 M | **MoE kernel 本體**（runtime 編譯） |
| `aiter_worker_limits.py` | 8 K | `csrc/cpp_itfs/utils.py:25` 直接 import，缺它 `import aiter` 就失敗 |

裝法（不改 repo、不污染 site-packages 亦可）：整包放到某目錄 + `PYTHONPATH=<dir>` 即可，
**實測在 torch 2.14.0+rocm7.2 上可 import 並執行**（§D5）。

### D1 imports（與 vLLM 等價）

```python
import os
os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"   # 必須在任何 aiter 呼叫前、權重準備前

import torch
from aiter.fused_moe import fused_moe            # 注意：aiter.fused_moe 是 module，不是頂層屬性
from aiter import ActivationType, QuantType
from aiter.ops.flydsl.moe_common import GateMode # GateMode 不在 aiter 頂層
from aiter.ops.shuffle import shuffle_scale, shuffle_weight
```
（`import aiter` 後 `aiter.fused_moe` 是 module ⇒ 不可寫 `aiter.fused_moe(...)`，實測會 AttributeError。）

### D2 一次性（載入權重時）

```python
E, H, I_PAD = 384, 5120, 640          # I_PAD = ceil(576/128)*128
fp4, e8m0 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu

# w13_u8: uint8 [E, 2*640, 2560]  (前 2*576 列來自 checkpoint，其餘 0)
# s13_u8: uint8 [E, 2*640, 160]   (e8m0；K/32=160，無需再 pad)
# w2_u8 : uint8 [E, 5120, 320]    (前 576/2=288 欄有效)
# s2_u8 : uint8 [E, 5120, 20]     (576/32=18 有效，其餘填 0x7F 或 0 皆可)
w13 = shuffle_weight(w13_u8.view(fp4), is_guinterleave=True, gate_up=True).contiguous()
s13 = shuffle_scale(s13_u8.view(e8m0).reshape(-1, 160), E, True, True).contiguous()
w2  = shuffle_weight(w2_u8.view(fp4),  is_guinterleave=True, gate_up=False).contiguous()
s2  = shuffle_scale(s2_u8.view(e8m0).reshape(-1, 20),  E, True, False).contiguous()
# 期望：w13 [E,1280,2560] fp4 / s13 [E*1280,160] e8m0 / w2 [E,5120,320] fp4 / s2 [E*5120,24] e8m0
```

### D3 每個 step（per token）

```python
out = fused_moe(
    hidden_states,           # bf16 [M, 5120]，**不要**自己量化
    w13, w2,
    topk_weight.float(),     # [M, 6] float32
    topk_ids.int(),          # [M, 6] int32（topk 索引，值 < 384）
    expert_mask=None, activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
    doweight_stage1=False, w1_scale=s13, w2_scale=s2,
    dtype=torch.bfloat16, hidden_pad=0, intermediate_pad=64,
    gate_mode=GateMode.INTERLEAVE.value, swiglu_limit=10.0,
)
# out: bf16 [M, 5120]（已含 topk_weight 加權；routed_scaling_factor 1.5 由呼叫端乘）
```
- `a1_scale`/`a2_scale` 省略（=None）⇒ **aiter 內部做動態 fp8 量化**，不需呼叫端提供 fp8 活化或 scale。
- `output=`/`num_local_tokens=`/`moe_sorting_dispatch_policy=` 省略即得 vLLM 行為。
- topk 的 softmax/sqrtsoftplus + noaux_tc 的 bias 修正、routed_scaling_factor **不在** aiter 內，
  由模型層算好 `topk_weight`/`topk_ids` 再傳入（與 vLLM `AiterMxfp4Experts` 相同分工）。
- shared expert（`n_shared_experts=1`）**不在此呼叫**：DSV4 由模型層另做 dense GEMM。

### D4 活化格式的明確答案

**必須傳 bf16 活化。** aiter 這條路徑內部把 bf16 → fp8_e4m3（per-token dynamic scale）+ 排序
（`fused_sort_quant`，`fused_moe.py:1735-1830`）再進 stage1；
`q_dtype_a` 由 `resolve_activation_dtype`（`fused_moe.py:999-1081`）依 `AITER_BF16_FP8_MOE_BOUND`
決定，vLLM 以 `"0"` 強制 fp8。**呼叫端不需要、也不應該先量化活化**；
若要自己預量化，必須改傳 `q_dtype_a=torch.float8_e4m3fn` 並給 `a1_scale/a2_scale`
（本 spec 不走這條）。

### D5 實測（在 `ts-dev`，torch 2.14.0+rocm7.2，4×MI350P，E=8/16 的子集 smoke test）

```
shapes [8,1280,2560] [10240,160] [8,5120,320] [40960,24] fp4 / e8m0
[aiter] [fused_moe] using 2stage default for ('gfx950', 128, 16, 5120, 640, 8, 6, Silu, bf16, float8_e4m3fn, float4_e2m1fn_x2, per_1x32, True, False)
[aiter] no tuned FlyDSL config … using heuristic FlyDSL fallback (kn1='flydsl_moe1_afp8_wfp4_bf16_t32x128x256_w2_gui', kn2='flydsl_moe2_afp8_wfp4_bf16_t32x128x128_atomic_bnt2')
[aiter] import [module_moe_sorting_opus] … / [module_quant] …
OUT torch.Size([16, 5120]) torch.bfloat16 finite True absmean 79.5
```
- 全程無需安裝（`PYTHONPATH` 直用 aiter wheel）；**kernel 名與 vLLM 完全相同**。
- 冷啟（import + 權重 shuffle + 第一次 fused_moe 含 FlyDSL 編譯）**6.4 秒**；第二次同 process ≈ 0。
- `M=512, BOUND=0` → 同 `q_dtype_a=fp8`；`M=16, BOUND=256` → `q_dtype_a=bf16`（不同 kernel 家族）。
- 只載入 `module_aiter_core`、`module_moe_sorting_opus`、`module_quant` 三個 .so。
- 腳本：`specs/aiter-moe-smoke.py`（可直接 `PYTHONPATH=<dir> python` 重跑）。

---

## E. 在 torch 2.14.0+rocm7.2 跑這份 aiter (為 torch 2.13.0+rocm10.0.0 編譯) 的風險

### E1 連結事實（`readelf -d`，`vendor/aiter-src/jit/`，共 133 個 .so）

- **47/133 個 .so 連到 libtorch**（`libc10.so, libc10_hip.so, libtorch_cpu.so, libtorch.so, libtorch_python.so`）。
- **MoE 路徑用到的三個都不連 libtorch**：
  `module_aiter_core.so`、`module_moe_sorting_opus.so`、`module_quant.so`、`module_moe_fmoe_asm.so`、
  `module_moe_mxfp4_aux.so` ⇒ NEEDED 只有 `libamdhip64.so.7, libstdc++.so.6, libgcc_s.so.1, libc.so.6`。
- 反例：`module_moe_sorting.so`（非 opus 版）**連 libtorch**，且**實測載入失敗**：
  ```
  undefined symbol: _ZN3c104c29c10_cuda_check_implementationEiPKcS2_jb
  ```
  （torch 2.13→2.14 改了 `c10_cuda_check_implementation` 簽名。）
  但預設路徑**不會**用到它：`fused_moe.py:74-75`
  `AITER_USE_CK_MOE_SORTING=0`、`AITER_USE_FLYDSL_MOE_SORTING=0` ⇒ `use_opus=True`（L812）
  ⇒ 走 `moe_sorting_opus_*`（無 libtorch）。**不要**設這兩個環境變數，否則會踩到 ABI 雷。
- `libamdhip64.so.7` 在本機解析到 `/opt/rocm/lib/libamdhip64.so.7.2.70204`（ts-dev 有），
  且 torch 自帶 `/opt/venv/.../torch/lib/libamdhip64.so`（aiter 會選它，日誌有印）⇒ 無缺庫問題。

### E2 實測結論

| 檢查 | torch 2.13+rocm10.0 (vLLM container) | torch 2.14+rocm7.2 (ts-dev) |
|---|---|---|
| `import aiter` | OK | **OK**（需 `aiter_meta` + `aiter_worker_limits`） |
| `shuffle_weight` / `shuffle_scale` | OK | **OK**（純 python） |
| `module_aiter_core.so` / `module_moe_sorting_opus.so` / `module_quant.so` dlopen | OK | **OK** |
| `module_moe_sorting.so` dlopen | OK | **FAIL（undefined symbol）** |
| `aiter.fused_moe` 端到端（gfx950） | OK | **OK**，kernel 選擇一致，輸出 finite |
| FlyDSL runtime 編譯（ROCm 7.2 LLVM） | OK | **OK**（首次 ~數秒，無錯誤） |

### E3 什麼會觸發 JIT 重編（需要 hipcc/ninja/csrc）

- `jit/core.py:1076-1093 _needs_arch_rebuild`：比對 `.so` 內 offload arch 與 `get_gfx_runtime()`；
  gfx950 已在 `module_*.so` 的 device code 內（`strings` 可見 `gfx942 gfx950`）⇒ **回報 False，不重編**。
- `AITER_REBUILD=1`（`jit/core.py:37`）會強制對每個 `compile_ops` 模組丟 `ModuleNotFoundError`→
  走 `build_module`（hipcc+ninja+`aiter_meta/csrc`）。**正常部署不要設**。
- `get_module` 找不到 `.so`（例如自己 remove JIT dir）也會觸發重編；
  `jit/core.py:1071` 的訊息：`Rebuild both in one environment: AITER_REBUILD=1, or remove {get_user_jit_dir()}`。
- 真的需要重編時：`ts-dev` 有 `/opt/rocm/bin/hipcc`、`/opt/venv/bin/ninja`、
  且 `aiter_meta/csrc`+`3rdparty/composable_kernel` 已隨 wheel 提供 ⇒ 條件具備，
  但 47 個 libtorch-linked 模組要全部重編才不會踩 E1 的 ABI（成本高、非必要）。

### E4 FlyDSL：runtime 編譯（與 aiter 的 prebuilt .so 不同）

- 每個 stage 的 kernel 由 **`flydsl` 套件在第一次呼叫時編譯**：
  `ops/flydsl/kernels/tensor_shim.py:305 _run_compiled → flyc.compile(exe, *args)`，
  之後只快取在記憶體 `exe._cf`（L329 `_preload_compiled`）。
- **沒有觀察到 on-disk kernel cache**：實測跑完後 `/tmp`、`~/.cache` 均無新產物
  （只有 `~/.cache/comgr/llvmcache.timestamp` 被 touch）⇒ **每個 process 首次呼叫都要付編譯時間**，
  多卡要各自 warmup。實測冷啟總計 6.4 s（含 torch import），可接受。
- 必須安裝 `flydsl==0.4.0`（含 `flydsl.libs`，268 MB，自帶 MLIR/LLVM compiler），
  它會用自己的編譯器，不依賴 ROCm 版本對齊。

### E5 建議部署方式（風險最低）

1. 把 `aiter`/`aiter_meta`/`flydsl`/`flydsl.libs`/`aiter_worker_limits.py` 放進 image
   或專用 `PYTHONPATH` 目錄（**不要** `pip install` 進 torch 的 site-packages 以免升級衝突）。
2. 啟動時 `AITER_BF16_FP8_MOE_BOUND="0"`，**不要**設 `AITER_REBUILD`、
   `AITER_USE_CK_MOE_SORTING`、`AITER_USE_FLYDSL_MOE_SORTING`。
3. 確保 `/tmp` 可寫（`/tmp/aiter_configs/tuned_fmoe.csv` 會被寫入）。
4. 載入後對 M=1/16/64/128 各做一次 warmup（觸發 FlyDSL 編譯、填 temp buffer）。
5. 若哪天必須用 libtorch-linked 的模組（CK/asm 路徑、非 opus sorting、attention 等），
   就必須以 torch 2.14 headers 重編 ⇒ 視為獨立工作項。

---

## F. 重現指令

```bash
# 1) 從 vLLM container 取執行期套件到 ts-dev（不落地 host）
D=/usr/local/lib/python3.12/dist-packages
docker exec priceless_hypatia bash -lc "tar -cf - -C $D aiter aiter_meta flydsl flydsl.libs aiter_worker_limits.py" \
 | docker exec -i ts-dev bash -lc "mkdir -p /tmp/aitertest && tar -xf - -C /tmp/aitertest"

# 2) 跑 smoke test（M/BOUND 可調）
docker exec -i ts-dev bash -lc 'cat > /tmp/aitertest/smoke.py' < specs/aiter-moe-smoke.py
docker exec ts-dev bash -lc "cd /tmp/aitertest && M=16 AITER_BF16_FP8_MOE_BOUND=0 \
  PYTHONPATH=/tmp/aitertest python smoke.py"

# 3) 驗證 kernel 選擇與 vLLM 一致（比對 vLLM 日誌）
docker exec priceless_hypatia grep -nE "Using 'AITER|AiterMxfp4Experts|heuristic FlyDSL" /tmp/deepseek-server.log
```

## G. 尚未驗證 / 待辦（誠實標註）

- 未在 **E=384、M≥1、完整 40 層**規模下實跑（GPU 正被 vLLM server 佔用；smoke 用 E=8）。
  kernel 名的 key 只差 expert 數（384 vs 8），heuristic 對 E 不敏感，風險低但未實證。
- 未驗證吞吐（本 spec 只保證「可跑且與 vLLM 同 kernel」）。
- shared expert / gate(bias) 的模型層前處理不在本 spec 範圍（由 unieinfra 模型層負責）。
