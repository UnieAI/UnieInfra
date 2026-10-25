# 45 — bf16 MoE 在 decode 形狀下的五方橫向對照（到 kernel）

版本見 `00-METHOD.md`。所有行號只對該表的 pinned commit 有效。

範圍：**一層 bf16（未量化）MoE，token 數 T=1..8**。目標形狀取 Qwen3.6-35B-A3B：
`hidden = 2048`、`num_experts = 256`、`top_k = 8`、`moe_intermediate = 512`，外加一個恆常
啟用的 shared expert。

T=1 時這一層是**頻寬受限**的：

| | 讀取的權重 | FLOP |
|---|---|---|
| gate/up（`w1`，`E×1024×2048`）| 8 × 1024 × 2048 × 2 B = **33.5 MB** | 33.5 MFLOP |
| down（`w2`，`E×2048×512`）| 8 × 2048 × 512 × 2 B = **16.8 MB** | 16.8 MFLOP |
| 合計 | **50.3 MB** | ~42 MFLOP（含 shared 之外）|

50 MB 的權重換 42 MFLOP 的算術 —— 算術強度 ~0.8 FLOP/byte，離任何張量核心的
roofline 轉折點差三個數量級。**這一層唯一該優化的量是 achieved GB/s。**

已知量測（RTX PRO 6000，sm_120，HBM peak 1792 GB/s）：vLLM 只到 **~372 GB/s＝peak 的
21%**。本文回答的是「為什麼會這樣，以及四個參照引擎有沒有人做得不一樣」。

---

## 0. 一句話的結論

**四個參照引擎沒有任何一家在 bf16 MoE 的 expert GEMM 上切分 K。這不是四家漏看了同一件
事，而是三種不同形態的「沒有」，其中只有 TurboMind 那一種留下了可讀的理由 —— 而那個
理由在 decode 形狀下是錯的。本 repo 有一支切了 K 的候選（`ops/moe/triton/bf16_decode.py`），
但它在本文寫作時**尚未進 commit**，見 §2.8 與 U1。**

| engine | K 是否切分 | 這個「沒有」的形態 | partial 怎麼歸約 | 有無 decode 專用路徑 |
|---|---|---|---|---|
| UnieInfer（serving 路徑）| 否 | 參數不存在 | — | 無（NVIDIA bf16）|
| UnieInfer（未入帳候選）| **是，`SPLIT_K = 8`** | grid 第三軸就是 K | **fp32 buffer + 第二趟 kernel**（刻意不用 atomics）| 有，`DECODE_MAX_PAIRS = 64` 以下 |
| vLLM | 否 | **參數存在但是死碼**：`SPLIT_K` 在 signature 裡，kernel body 從不讀它 | — | **有**：`naive_block_assignment` 整個跳過 sort |
| SGLang | 否 | 參數不存在 | — | 無（只有 tuned tile）|
| TRT-LLM | 否 | **實作存在但被 `max_split_k = 1` 掐死在候選產生階段** | —（dense 路徑用 `SPLIT_K_SERIAL`）| **有但 bf16 進不去**：`min_latency_mode` 要求 `use_fp4` |
| TurboMind | 否（**唯一結構上可達**）| **kernel 註冊成 `SplitK=true`、grid.z 就是 split 軸，但被 wave 啟發式排除** | **epilogue 內的序列 semaphore chain** | 無（同一族 kernel 依 M 選 tile）|

三件值得單獨拿出來的事：

1. **vLLM 與 SGLang 都已經寫好了 split-K 的 MoE GEMM —— 在 LoRA 路徑上，用 relaxed
   atomics 歸約。** 兩家的 base expert GEMM 都沒有。同一個 repo、同一個 Triton、同一個
   問題（N 太窄、CTA 太少），解法寫好了卻沒有搬過去一條街。
2. **TurboMind 的 split-K 被排除的理由寫在原始碼裡，而且可以指出它錯在哪**：
   成本模型把 `grid_size` 乘上專家數 `num`，但那個 `num` 來自「每個專家至少配一個
   M-tile」的上界，其中 248/256 個 tile 在 T=1 會立刻 `return`。**它是拿幽靈 CTA 算出
   grid 已經飽和的。**
3. **每一家為 decode 準備的手段都跟 K 正交**（省掉 sort、swap-AB、更窄的 N tile），而且
   在 sm_120 + bf16 這一格上**逐一被閘掉**。
4. **partial 的歸約在五方之外還有第三種形態**：vLLM/SGLang 的 LoRA 用 relaxed atomics、
   TurboMind 用 epilogue 內的序列 semaphore chain、本 repo 的候選用 **fp32 buffer +
   獨立的第二趟 kernel**，並且是三者之中**唯一寫下取捨理由**的（§2.8）。

---

## 1. 步驟表

一層 MoE、T=1、bf16、NVIDIA 預設旗標。格內為 `file:line symbol`。

| # | 步驟 | UnieInfer | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 1 | MoE block | `runtime/models/qwen3_5_moe.py:170 Qwen3_5MoeSparseMoeBlock` | `model_executor/models/qwen3_moe.py:130 Qwen3MoeSparseMoeBlock` | `srt/models/qwen2_moe.py:220 Qwen2MoeSparseMoeBlock` | `_torch/models/modeling_qwen3_next.py:281 shared_expert` | `src/turbomind/models/llama/moe_ffn_layer.cc:133 Forward` |
| 2 | backend 決策 | `ops/moe/__init__.py:189 select_kernel` | `model_executor/layers/fused_moe/oracle/unquantized.py:67 is_cuda` | `srt/layers/quantization/unquant.py:567 MoeRunnerBackend` | `_torch/model_config.py:401 CUTLASS` | `src/turbomind/kernels/gemm/context.cu:118 Filter` |
| 3 | **預設落到誰** | `ops/moe/flashinfer/cutlass_unquant.py:76 priority`（**CUTLASS**）| Triton `fused_moe_kernel`（FlashInfer 見 §2.6）| Triton `fused_moe_kernel` | `_torch/modules/fused_moe/create_moe.py:82 CutlassFusedMoE` | 通用 GEMM + grouped descriptor |
| 4 | router GEMM | `runtime/models/qwen3_5_moe.py:205 gate` | `model_executor/layers/fused_moe/router/gate_linear.py:138 fp32_router_gemm_dispatch` | `srt/models/qwen2_moe.py:304 gate` | `_torch/models/modeling_qwen3_next.py:330 gate` | `src/turbomind/models/llama/moe_ffn_layer.cc:43 Gate` |
| 5 | top-k of 256 | **torch**：`runtime/layers/moe/topk.py:219 topk_weights`（`softmax`+`torch.topk`）| `model_executor/layers/fused_moe/router/fused_topk_router.py:24 topk_softmax` | `kernels/ops/moe/moe_fused_gate.py:340 grid` | `cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu:249 numBlocks` | `src/turbomind/kernels/gemm/moe_utils_v2.cu:265 MoeGateKernel_v8` |
| 6 | top-k grid @T=1 | 3 個 torch launch | `csrc/libtorch_stable/moe/topk_softmax_kernels.cu:618 topkGating`（1 CTA）| 1 CTA / 1 warp | 1 CTA × 256 thr | 1 CTA |
| 7 | sort / align | `unieinfer/kernel-sm120/unieinfra_kernel_sm120/moe.py:273 _align_blocks`（純 torch）| `model_executor/layers/fused_moe/moe_align_block_size.py:74 max_num_tokens_padded` | `kernels/aot/csrc/moe/moe_align_kernel.cu:342 small_batch_expert_mode` | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:437 blocks` | `src/turbomind/kernels/gemm/moe_utils_v2.cu:133 MoeScanKernel_v2` |
| 8 | **T=1 能否跳過 sort** | 否 | **能**：`model_executor/layers/fused_moe/fused_moe.py:1497 naive_block_assignment` | 否（單 token kernel 只接 Marlin，見 §2.3）| 否（快徑仍跑，1 CTA×32 thr）| 否 |
| 9 | gate/up GEMM | `unieinfer/kernel-sm120/unieinfra_kernel_sm120/moe.py:146 _moe_gemm_kernel` | `model_executor/layers/fused_moe/fused_moe.py:294 fused_moe_kernel` | `kernels/ops/moe/fused_moe_triton_kernels.py:324 fused_moe_kernel` | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:485 GemmUniversal` | `src/turbomind/models/llama/LlamaLinear.cu:193 Run` |
| 10 | **grid 怎麼組** | `unieinfer/kernel-sm120/unieinfra_kernel_sm120/moe.py:484 grid` | `model_executor/layers/fused_moe/fused_moe.py:795 grid` | `kernels/ops/moe/fused_moe_triton_kernels.py:810 grid` | CUTLASS persistent scheduler，`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:512 max_split_k` | `src/turbomind/kernels/gemm/scheduler_sm70.cuh:61 splits` |
| 11 | grid 的軸 | M×N（1-D flatten）| M×N（1-D flatten）| M×N（1-D flatten）| persistent，SM 數決定 | **M×N×splits（grid.z＝split）** |
| 12 | **K 迴圈** | `unieinfer/kernel-sm120/unieinfra_kernel_sm120/moe.py:235 cdiv`（序列）| `model_executor/layers/fused_moe/fused_moe.py:496 cdiv`（序列）| `kernels/ops/moe/fused_moe_triton_kernels.py:529 k_start`（序列）| CTA 內序列 | `src/turbomind/kernels/gemm/scheduler_sm70.cuh:153 k_iters`（**可分段**）|
| 13 | **split-K 開關** | 不存在 | `model_executor/layers/fused_moe/fused_moe.py:338 SPLIT_K`（**死碼**）| 不存在 | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:512 max_split_k`（**=1**）| `src/turbomind/kernels/gemm/kernel/sm80_16816_16.cu:53 add`（**`SplitK=true`**）|
| 14 | partial 歸約 | — | — | — | — | `src/turbomind/kernels/gemm/epilogue.h:402 SplitK_`（semaphore chain）|
| 15 | activation | 融進 CUTLASS | `model_executor/layers/fused_moe/fused_moe.py:1751 apply_moe_activation` | 獨立 launch | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:2111 doActivation` | `src/turbomind/models/llama/moe_ffn_layer.cc:272 Activation` |
| 16 | down GEMM | 同 9 | 同 9 | 同 9 | 同 9 | `src/turbomind/models/llama/moe_ffn_layer.cc:276 Forward` |
| 17 | top-k combine | `runtime/models/qwen3_5_moe.py:321 fused_gate_sigmoid_mul_add` | `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:714 moe_sum` | `srt/layers/moe/moe_runner/triton_utils/fused_moe.py:814`（`torch.sum`）| `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:1985 blocks`（1 CTA）| `src/turbomind/kernels/gemm/moe_utils_v2.cu:1127 invokeMoeReduce`（1 CTA）|
| 18 | shared expert | 分開（capture 時另一 stream）：`runtime/models/qwen3_5_moe.py:237 shared_expert` | 分開（aux stream ≤256 token）：`model_executor/layers/fused_moe/runner/shared_experts.py:99 should_run_shared_in_aux_stream` | Qwen3.5 分開 / Qwen3-Next 融合：`srt/models/qwen2_moe.py:247 enable_cuda_shared_expert_fusion` | 分開（aux stream）：`_torch/models/modeling_qwen3_next.py:350 maybe_execute_in_parallel` | 分開（decoder 層另一個 FFN）：`src/turbomind/models/llama/unified_decoder.cc:307 forward` |
| 19 | shared 融進同一 launch 的條件 | 無此選項 | **ROCm/AITER 限定**：`model_executor/layers/fused_moe/layer.py:88 is_fusion_moe_shared_experts_enabled` | `srt/models/qwen2_moe.py:397 _append_shared_to_topk_output` | **TRTLLM-gen + 量化限定**：`_torch/modules/fused_moe/quantization.py:1123 fuse_shared_expert` | 無此選項 |

### 1.1 T=1 的 grid 算術（由上表推導，非原始碼陳述）

| engine | tile (M/N/K) | M-tiles | gate/up CTA | down CTA | K 迴圈次數（gate/up）|
|---|---|---|---|---|---|
| vLLM（Triton 預設）| 16 / 64 / 128 | 8 | **128** | 256 | 16 |
| SGLang（本卡 tuned）| 32 / 32 / 64 | 8 | **256** | 512 | 32 |
| UnieInfer（sm120 plugin）| 32 / 32 / 64 | **249 發射 / 8 存活** | 7968 發射 / **256 有效** | — | 32 |

- vLLM：`model_executor/layers/fused_moe/fused_moe.py:794 EM` 在 `sorted_token_ids is None` 時取 `EM = num_tokens *
  BLOCK_SIZE_M = 8 × 16 = 128`；tile 來自 `model_executor/layers/fused_moe/fused_moe.py:1238 get_default_config`
  （`M ≤ 32 → BLOCK_M 16`、`M ≤ 64 → BLOCK_N 64`、`M ≤ 64 → BLOCK_K 128`）。
- SGLang：`srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py:77 max_num_tokens_padded` 的小 M 分支把緩衝夾到
  `numel × block_size = 8 × 32`；tile 讀自本卡的 tuned JSON（見 §2.6）。
- UnieInfer：`unieinfer/kernel-sm120/unieinfra_kernel_sm120/moe.py:323 max_len` 用
  **靜態最壞值** `num_routes + num_local_experts × (block_m − 1) = 8 + 256 × 31 = 7944`
  當 grid，多餘的 block 在裝置上 `return`。**vLLM 與 SGLang 都在主機端把這個數夾掉，
  UnieInfer 沒有** —— 這是一格結構差異，不是效能調校（§3）。

**128 個 CTA（vLLM）在 188 個 SM 上連一個 wave 都填不滿**，而每個 CTA 只有 16 次 K 迭代、
`num_stages = 4` 的 prefetch 深度。這就是 372 GB/s 的形狀解釋：可同時在飛的 memory
transaction 數（MLP）被 CTA 數 × stage 數卡住，跟頻寬本身無關。

---

## 2. 分歧點

### 2.1 K 不切分 —— 五方一致，理由分成三種形態

**(a) 參數根本不存在** —— UnieInfer、SGLang。

SGLang 的 `fused_moe_kernel` 從 `kernels/ops/moe/fused_moe_triton_kernels.py:529 k_start`
一路序列走到 `:650` 的單次 `tl.store`，沒有 `SPLIT_K` 常數、沒有 partial 緩衝、沒有第二趟。
UnieInfer 的 `unieinfer/kernel-sm120/unieinfra_kernel_sm120/moe.py:235 cdiv` 同構。
兩邊**都沒有任何註解或設計說明討論過 split-K**。這一格是繼承來的結構，不是決策。

**(b) 參數存在但是死碼** —— vLLM。

`model_executor/layers/fused_moe/fused_moe.py:338 SPLIT_K` 出現在 `fused_moe_kernel` 的
signature 裡（`:100` 的 `fused_moe_kernel_gptq_awq` 也有），但**在 kernel body 裡零次
引用**。K 迴圈是 `model_executor/layers/fused_moe/fused_moe.py:496 cdiv` 的單一序列迴圈，epilogue 是 `model_executor/layers/fused_moe/fused_moe.py:582 accumulator`
的普通 `tl.store`，沒有 atomics。主機端 `model_executor/layers/fused_moe/fused_moe.py:802 SPLIT_K` 每次都硬寫 1，而且
連那個 1 都不會被讀。

它為什麼在那裡：`SPLIT_K` 是 tuned-config JSON 的 schema 欄位（`configs/` 下每個 AMD
int4 檔都寫 `"SPLIT_K": 1`），並且是 `fused_moe_lora` 那支**真的有實作**的 kernel 的參數。
base kernel 只是把欄位吞掉。

**讀這一格的教訓**：grep 到 `SPLIT_K` 不等於有 split-K。這正是
`compile-proof-is-not-run-proof` 的同型錯誤 —— 符號存在、型別對、launch 成功，而語意是空的。

**(c) 實作存在但被啟發式排除** —— TRT-LLM 與 TurboMind，兩者的排除點不同。

TRT-LLM 排除在**候選產生**階段。兩個 `MoeGemmRunner` 的候選產生器都硬寫
`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:512
max_split_k`（Ampere）與 `:540 max_split_k`（TMA-WS），值都是 1；下游
`cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp:685 split_k_factor`
的迴圈 `for (split_k_factor = 2; split_k_factor <= max_split_k; ...)` 因此一次都不進去。
結果：**每一個 arch、每一個 dtype，MoE 的每個 tactic 都是 `NO_SPLIT_K`**。
同檔 `:725` 有一個真正的 small-N split-K 啟發式（`n >= multi_processor_count * 256 ? 1
: split_k_limit`），但它服務的是 dense weight-only / int8 plugin GEMM，`MoeGemmRunner`
從不經過。`StreamK` 在 `cutlass_kernels/` 整棵樹零命中。

TurboMind 排除在**成本模型**階段 —— 見 §2.2，這是全篇最值得帶走的一格。

### 2.2 TurboMind：唯一結構上可達的 split-K，以及它為什麼沒被選中

TurboMind 是五家裡唯一一個 split-K 與 grouped（MoE）在模板上**正交**的：
`Config_F16_g` 的 scheduler 同時吃 `SplitK` 與 `group_axis` 兩個參數
（`src/turbomind/kernels/gemm/arch/config_sm80_s16816.h:58 SplitK`），而 registry 把
**小 M 的 grouped bf16 tile 全部註冊成 `SplitK=true`**，包含 decode 最可能落到的那一個：

```cpp
// src/turbomind/kernels/gemm/kernel/sm80_16816_16.cu:53
c.add<C::Type< 16,  64, 128, 1, 2, 2, D, S, 3, true, 1, 1>>(); // 10
```

grid 的第三軸就是 split 數（`src/turbomind/kernels/gemm/scheduler_sm70.cuh:61 splits`），
每個 CTA 拿到自己的 k 區間（`:153 k_iters`）。

**partial 的歸約方式是五家裡唯一存在的一種，而且不是 atomics 也不是第二趟 kernel：**

```cpp
// src/turbomind/kernels/gemm/epilogue.h:402
if (SplitK_ && splits > 1) {
    int* barrier = &param.locks[tile_id];
    sem_wait(barrier, tile_offset.z, threadIdx.x == 0);
    const MatrixData p = resolve<Dtype, kMode>(param.partials, tile_offset.w);
    Reduce(tmp_C, p, tile_offset.z == 0, is_last, cs0, pred);
    const int post_id = is_last ? 0 : tile_offset.z + 1;
    sem_post(barrier, post_id, threadIdx.x == 0);
    if (!is_last) { return; }
}
```

也就是 **epilogue 內的序列 semaphore chain**：split `z` 等 `z-1` 放行、累加、放行 `z+1`，
最後一個才真的寫出去。代價是 partial 緩衝只要一份而不是 `splits` 份
（`src/turbomind/kernels/gemm/kernel_impl.h:224 kSerial`，`if constexpr (!kSerial)` 才乘
`splits`），好處是 32 MB 的固定 workspace 就夠用（`src/turbomind/kernels/gemm/gemm.h:16 kPartialsSize` `kPartialsSize = 32<<20`）。
**代價是 split 之間完全序列化 —— 這在 decode 尤其要注意：它買到的是更多 CTA 去發
memory request，不是更多算術平行度，而序列鏈把 epilogue 的延遲乘上 splits。**

**它為什麼沒被選中** —— `src/turbomind/kernels/gemm/context.cu:221 grid_size`：

```cpp
const int   grid_size    = tiled_shape_m * tiled_shape_n * splits * num;
const int   full_waves   = grid_size / concurrency;
const int   residue      = grid_size % concurrency;
const float partial_wave = (float)cdiv(residue, sm_count_) / info.max_active_ctas;
const float waves        = full_waves + partial_wave;

if (splits > 1 && waves > param.max_waves) {
    break;
}
```

`max_waves` 預設 10（`src/turbomind/kernels/gemm/tuner/params.h:14 max_waves`）。
`num` 是本地專家數 = 256。而 `tiled_shape_m` 來自
`src/turbomind/kernels/gemm/scheduler_sm70.cuh:77 gemm_shape_`：

```cpp
tiles_[i] = ((gemm_shape_[i] / tile_shape[i] >> log_unit[i]) + gemm_shape_[3]) << log_unit[i];
// 註解：overwrite dynamic axis <- estimated upper bound
```

即 **M-tiles = ⌈rows/CTA_M⌉ + E**：每個專家無論有沒有拿到 token，都先配一個 M-tile。
T=1 時真正有工作的只有 8 個，其餘的在
`src/turbomind/kernels/gemm/gemm_universal.h:114` 立刻 `return`（"empty tile"）。

**所以成本模型是拿幽靈 CTA 在算 wave 數。** `grid_size` 同時被「+E 的 M-tile 上界」和
「× num」放大，`waves` 在 `splits = 1` 就已經遠超 10，迴圈在 `splits = 2` 就 `break`，
唯一被送出去的候選是 `splits = 1`。對 **dense** decode GEMM（`num = 1`、
`tiled_shape_m = 1`）`waves ≪ 10`，split-K 候選正常產生 —— **那正是這套機制實際服務的路徑**。

換句話說：TurboMind 有能力對 MoE 切 K、有歸約機制、有 autotuner 會真的去量
（`src/turbomind/kernels/gemm/gemm.cu:177 Measure`），只是候選從來沒被產生出來給它量。
**這是全篇唯一一個「不做的理由」寫在原始碼裡的地方，而那個理由在 decode 形狀下站不住。**

補一格閘門：SM90 的 bf16 grouped 路徑把 split-K **硬關**了 ——
`src/turbomind/kernels/gemm/kernel_impl_sm90_bf16.h:354 GetMaxSplits` 直接 `return 1;`，
`Launch` 的 `splits` 參數在 body 裡從不讀。所以上面那條 `SplitK=true` 的註冊只在
SM70/75/80 這一族有效；而 `src/turbomind/kernels/gemm/arch.h:53` 把 `karch == 800`
判為相容 `Sm120` —— **sm_120 走的正是那一族**，split-K 在架構上是可達的。

### 2.3 split-K 已經在隔壁寫好了：vLLM 與 SGLang 的 LoRA 路徑

兩家的 **MoE-LoRA** kernel 都有完整的 split-K，用 **relaxed atomics** 歸約：

```python
# vllm: lora/ops/triton_ops/fused_moe_lora_op.py:1162
tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")
```
```python
# sglang: kernels/ops/moe/fused_moe_lora_kernel.py:206
tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")
```

vLLM 的 grid 把 `split_k` 直接乘進 block 數（`lora/ops/triton_ops/fused_moe_lora_op.py:1221
grid`），k 區間由 `:975 grid_k`（`tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)`）切開。

**它解的是同一個問題**：LoRA shrink 的 N = lora rank，很窄，所以 N 方向沒有平行度可拿 ——
跟 decode MoE 的 N = 1024 / 每 CTA 只有 8 個 M-tile 是同一個病。

而且 vLLM 的預設值揭露了一條梯度：

| op | split_k 預設 | 引用 |
|---|---|---|
| dense LoRA `shrink` | **64（batch < 128）/ 8** | `lora/ops/triton_ops/utils.py:221 split_k` |
| `fused_moe_lora_*_shrink` | **1**（kernel 支援，預設不用）| `lora/ops/triton_ops/utils.py:247 split_k` |
| base bf16 expert GEMM | **無實作** | `model_executor/layers/fused_moe/fused_moe.py:338 SPLIT_K` |

**同一個 repo，同一個 decode batch，dense 路徑切 64 份、MoE-LoRA 路徑寫好了不切、
expert GEMM 連寫都沒寫。** 這三格擺在一起，比任何一格單獨看都有資訊量。

vLLM 的 LoRA 小批次 kernel 另外有一個五家裡唯一的東西：**SM 數感知的 persistent grid**，
且理由寫在註解裡（`lora/ops/triton_ops/fused_moe_lora_op.py:786-790`，
"enough to saturate the SM array once `M_grid * num_slices` reaches ~`SM_count`"），
grid 是 `min(work_total, 2 * sm_count)`。**base MoE 的 grid 從不看 SM 數。**

### 2.4 每一家為 decode 準備的手段都跟 K 正交 —— 而且逐一被閘掉

| 手段 | 誰有 | 在 sm_120 + bf16 這一格 |
|---|---|---|
| 跳過 sort（`sorted_token_ids = None`，一個 block 一個 token）| vLLM：`model_executor/layers/fused_moe/fused_moe.py:1497 naive_block_assignment` | **生效**（`1×8×4 = 32 ≤ 256`，條件即 T ≤ 8）|
| 跳過 permute + 不做 finalize（`min_latency_mode`）| TRT-LLM：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4260 use_fp4` | **被擋**：`TLLM_CHECK(use_fp4 == true)`；且 Python 端無人設定 |
| swap-AB（把 token 放到 N 軸）| TRT-LLM：`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:614 swap_ab_configs`；vLLM：`model_executor/layers/fused_moe/utils.py:474 enable_swap_ab` | vLLM **被擋兩次**：`is_device_capability(90)`（Hopper 限定）且只在 `use_fp8_w8a8` 開（`model_executor/layers/fused_moe/fused_moe.py:761 SWAP_AB`）|
| 單 warp 的 single-token align | SGLang：`kernels/ops/moe/moe_align_single_token.py:32 moe_align_single_token` | **只接 Marlin**（量化路徑），bf16 沒有對應捷徑 |
| 本卡 tuned tile 表 | SGLang（見 §2.6）| **生效** |

vLLM 的 `naive_block_assignment` 值得單獨讀，因為它是五家裡唯一真正的
「decode 形狀改變 kernel 結構」：

```python
# model_executor/layers/fused_moe/fused_moe.py:1497
naive_block_assignment = (
    expert_map is None
    and num_tokens * top_k_num * 4 <= global_num_experts
    and not (...)
)
```

`1 × 8 × 4 = 32 ≤ 256` 成立到 T = 8 為止 —— **這個條件就是 decode 的定義**。命中後
`moe_align_block_size` 整個不跑，`expert_ids` 就是 `topk_ids.view(-1)`，kernel 走
`model_executor/layers/fused_moe/fused_moe.py:404 offs_token` 的分支：每個 block 的 `offs_token` 只有第 0 個 lane 有效，其餘被
mask 掉。代價是 `BLOCK_SIZE_M = 16` 的 M 維只有 1 列有用（15/16 的 MMA 浪費），
但在頻寬受限的形狀下這不是成本 —— **省下的兩個 align kernel launch 才是**。

SGLang 對照組：`kernels/aot/csrc/moe/moe_align_kernel.cu:342 small_batch_expert_mode`
的快徑條件是 `(topk_ids.numel() < 1024) && (num_experts <= 64)`，**256 專家直接出局**，
所以 T=1 仍然要付兩個 kernel（`:366` 2 CTA、`:386` 1 CTA）純 launch 開銷。

### 2.5 routing 在 T=1 一律是 1 個 CTA，差別在有沒有 kernel

top-8 of 256 在四家都是**一個單 CTA 的 launch**：vLLM
`csrc/libtorch_stable/moe/topk_softmax_kernels.cu:618 topkGating`（E=256 有專門特化）、
SGLang `kernels/ops/moe/moe_fused_gate.py:340 grid`（1 program / 1 warp）、TRT-LLM
`cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu:249 numBlocks`（1 CTA×256 thr，8 個
warp 只有 1 個活）、TurboMind `src/turbomind/kernels/gemm/moe_utils_v2.cu:265
MoeGateKernel_v8`（1 block）。**四家都是 launch 延遲地板，不是計算成本。**

UnieInfer 是唯一一個**沒有** routing kernel 的：`runtime/layers/moe/topk.py:219
topk_weights` 走 `torch_native_fused_topk`，`F.softmax` + `torch.topk` + 除法 =
**三個 torch launch**。以本 repo 記錄的空 Triton launch 地板 7.7–13.9 µs 計，
這一格每層多付兩次那個地板 × 層數。

TurboMind 有一個五家獨有的硬限制值得記下：`src/turbomind/kernels/gemm/moe_utils_v2.cu:634 dispatch-667` 的 if-chain 把
（experts, top_k）組合寫死，E=256/top_k=8 落進 `experts <= 512` 那一格，
於是 kernel **以 512 個專家的模板實例化去跑 256 個專家**，logits 的工作量翻倍；
不在表內的組合直接 `TM_CHECK` 失敗（`:671`）。

### 2.6 sm_120 + bf16 這一格的閘門（跨引擎）

這一節是 `sm120-closed-allowlist-tables` 那條記憶在 MoE 上的複驗。三個新的實例：

1. **vLLM 的 unquantized backend 優先序在 sm_120 上把 FlashInfer 排在 Triton 前面。**
   `model_executor/layers/fused_moe/oracle/unquantized.py:67 is_cuda` 的清單是
   `[FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, TRITON, BATCHED_TRITON]`，而把 FlashInfer
   降級的那兩行只對 `is_device_capability_family(90)` 生效（`:77`）。逐一檢查：
   - `FLASHINFER_TRTLLM` 要求 `is_device_capability_family(100)`
     （`model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py:55`）→ **sm_120 出局**。
   - `FLASHINFER_CUTLASS` **明確包含 120**
     （`model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:139`
     `or p.is_device_capability_family(120)`）→ **只要 flashinfer 裝了就會選它，不是 Triton。**

   **這件事直接框住那個 372 GB/s 的數字**：它是 Triton 的數字還是 FlashInfer CUTLASS 的
   數字，取決於量測當時 `has_flashinfer_cutlass_fused_moe()` 回什麼。啟動時會 log
   `Using ... Unquantized MoE backend out of potential backends: [...]`
   （`model_executor/layers/fused_moe/oracle/unquantized.py:200 _make_log_backend`）—— **引用那個數字之前先把這行 log 貼出來。**
   這正是 `verify-the-reference-engine-is-on-its-fast-path` 要求的那一步。

2. **vLLM 在本卡本形狀沒有 bf16 的 tuned tile 表，SGLang 有。**
   vLLM 的檔名由 `model_executor/layers/fused_moe/fused_moe.py:1034 get_config_file_name`
   組出 `E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json`
   （bf16 無 dtype 後綴）—— `configs/` 下**不存在**（該卡只有 fp8 `block_shape` 版本），
   於是 `model_executor/layers/fused_moe/fused_moe.py:1050 get_moe_configs` 走到
   `warning_once("Using default MoE config. Performance might be sub-optimal!")`，
   落到 `model_executor/layers/fused_moe/fused_moe.py:1238 get_default_config` 的啟發式。

   SGLang 有這張表（`triton_3_6_0/E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json`，
   另有 `_down` 版本），M=1 的答案是 `32/32/64`、`GROUP_SIZE_M 32`、4 warps、4 stages。
   **N tile 32 比 vLLM 預設的 64 窄一半，CTA 數多一倍** —— 這是 SGLang 對同一個 MLP
   飢餓問題給出的答案，方向與 split-K 相同（增加在飛的請求數），手段不同（切 N 而非切 K）。

   UnieInfer 的 sm120 plugin 直接把這張表抄了過來，並在註解裡寫明出處：
   `unieinfer/kernel-sm120/unieinfra_kernel_sm120/moe.py:394 _BF16_TABLE`，M=1 那列
   `(1, 32, 32, 64, 32, 4, 4)`。

3. **vLLM 的 `get_default_config` 對小 M 的成文決策，方向與 split-K 相反。**
   `model_executor/layers/fused_moe/fused_moe.py:1321` 附近的註解：
   「Small batches benefit from longer reduction (larger K tile), while large batches
   prefer more output parallelism.」於是 `M ≤ 64` 時 `BLOCK_K = 128`（大 K tile），
   `M ≤ 64` 時 `BLOCK_N = 64`。**把 K tile 開大等於讓單一 CTA 獨佔更長的 K，
   正是 split-K 的反面。** 這是五家裡唯一一段直接談到「小 batch 要怎麼處理 K」的文字，
   而它選了相反的方向 —— 且沒有附上量測。

### 2.7 shared expert：沒有人在 NVIDIA + bf16 上把它融進同一次 launch

五家在**結構上**都把 shared expert 當成獨立的 dense FFN，差別只在要不要疊 stream：

- **vLLM**：`model_executor/layers/fused_moe/runner/shared_experts.py:99
  should_run_shared_in_aux_stream` —— CUDA 且 token 數 ≤
  `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`（預設 **256**，`envs.py:267`）時走
  `MULTI_STREAM_OVERLAPPED`。**T=1 一定命中**，shared expert 與 routed experts 併行。
  真正融進 grouped GEMM 的路徑存在，但 `model_executor/layers/fused_moe/layer.py:88
  is_fusion_moe_shared_experts_enabled` 是 **ROCm/AITER 限定**。
- **TRT-LLM**：`_torch/models/modeling_qwen3_next.py:350 maybe_execute_in_parallel`，
  aux stream 併行，combine 用 Triton epilogue
  （`_torch/modules/fused_shared_expert.py:70 fused_sigmoid_gate_mul_add`）。
  融合版只存在於 TRTLLM-gen + 量化（`_torch/modules/fused_moe/quantization.py:1123
  fuse_shared_expert`），`CutlassFusedMoE` 沒有。
- **SGLang**：**唯一一家真的在 NVIDIA 上融合的**，但只對 Qwen3-Next
  （`srt/models/qwen2_moe.py:397 _append_shared_to_topk_output`，shared 變成第 9 個
  expert slot）。Qwen3.5-MoE 不傳 `enable_cuda_shared_expert_fusion`
  （`srt/models/qwen2_moe.py:247`），退回 dual-stream（`:513 forward_normal_dual_stream`）。
- **TurboMind**：`src/turbomind/models/llama/unified_decoder.cc:307 forward` ——
  MoE 的 `Forward` 與 `Combine` 被**故意拆開**，dense FFN 夾在中間跑，`Combine` 再把
  routed 結果累加上去。同一條 stream，靠指令級重疊。
- **UnieInfer**：`runtime/models/qwen3_5_moe.py:237 shared_expert` 獨立 MLP，
  `:292` 的 `stream_fork` **只在 `get_is_capture_mode()` 為真時**分岔 —— 也就是
  **只有 CUDA graph 捕捉路徑才併行，eager 路徑是序列的**。這一格與 vLLM（無條件、
  以 token 數為門檻）不同，值得記進 §3。

### 2.8b 已入帳、自動生效，但端到端沒有變化（2026-08-16）

§2.8 的候選現在有 `register_kernel`、十個測試、與一個實測的交接門檻。三件事在
量測過程中被推翻，都記在這裡，因為每一件都會讓下一個人重犯：

**一、「split-K」不是勝出的原因。** 三輪掃描（tile、warps/stages、窄 N 形狀）之後
最快的組態是 **`_SPLIT_K = 1`**——K 完全不切。贏過 grouped GEMM 的是 matvec 的結構
本身，不是切 K。檔名留著是歷史，調參數的人不要去動 SPLIT_K 找空間。

**二、eager 下掃參數量不到東西。** 一次空的 Triton 啟動在這台機器是 14.8 us，這支
kernel 有四次啟動＝59 us 的固定成本壓在 25 us 的 kernel 上。tile 之間 10% 的差異
完全被蓋住。**必須在 CUDA graph 捕捉下掃**——引擎的 decode 本來就是這樣跑的。

**三、固定路由會讓工作集常駐 L2。** T=1 只碰 8 個專家＝50.3 MB，而這張卡有 128 MB
L2；用同一組專家重複量，graph replay 報出 2175 GB/s＝**121% of HBM peak**。
量測必須讓每次迭代路由到不同專家（32 組 × 8 = 256，覆蓋整個 1.61 GB）。

修正上述三點之後的數字（Qwen3.6-35B-A3B 形狀，hidden 2048 / 256 experts /
top-8 / intermediate 512，T=1，輪替專家組）：

| | eager | CUDA graph（引擎實際跑法） |
|---|---|---|
| FlashInfer CUTLASS（vLLM 也是這條） | 141.3 us / 356 GB/s / 20% peak | 60.8 us / 827 GB/s / 46% peak |
| **本 kernel** | **85.8 us / 587 GB/s / 33%** | **46.7 us / 1078 GB/s / 60%** |
| 倍數 | **1.65x** | **1.30x** |

roofline 是 28.1 us（50.3 MB ÷ 1792 GB/s）。**參數空間已經榨乾**——三輪掃描的前六名
都落在 46–48 us。演算法層面算過三條路，融合歸約省 0.4%、persistent kernel 省 ~1%、
存取形狀已掃遍，因為 50.3 MB 的權重讀取不可壓縮（算術強度 0.8 FLOP/byte）。

**真正的槓桿是批次大小**：同一支 kernel 在 T=64 達到 88% of peak，因為同一份權重
分攤給 64 個 token。**MoE 的效率本質是批次問題，而批次被准入卡在 1–2**
（見 `46-fp8-moe-reachability.md` §6）。兩條線在這裡合流。

**端到端量測：沒有變化，而且路上錯了兩次。**

| 35B agent C=16 | 基準 | 含此改動 |
|---|---|---|
| tok/s | 160.08 | 159.64 |
| goodput/s | 0.27 | 0.27 |
| 首字 p50 | 14,947 ms | 16,241 ms |

kernel 的 1.30x 沒有兌現，原因就是上面那段批次分析：35B 在 C=16 時每步只有 1–2 個
請求在解碼，MoE 的 decode 呼叫佔整體時間極小，瓶頸壓倒性在 prefill 與准入排隊
（首字 16 秒對門檻 10 秒）。**這個 kernel 要等批次變大才會兌現。**

路上兩個錯，第一個是設計層級的：

1. **把它註冊成競爭 solution，registry 就在所有形狀選了它**——包括 8192 token 的
   prefill，端到端掉到 **49.9 tok/s**（基準 160.1）。形狀門檻完全沒機會執行，因為
   **選擇在 plan 建立時發生一次**，不是每次呼叫。正確做法是**不註冊**，只透過
   `cutlass_unquant` 內部那條有 token 門檻的轉交進入。`test_is_not_registered_as_a_solution`
   釘住這一點。
2. **診斷錯方向**：先懷疑 `PAIRS: tl.constexpr` 造成每步 JIT 重編譯（那確實是缺陷，
   token 數當編譯期常數會讓每個不同的常駐請求數觸發一次完整編譯，已改成執行期
   參數），但修完數字一模一樣。真正的病因是第 1 點。分辨方法很簡單：把門檻設 0
   關掉轉交再量一次——數字不動就代表病因不在轉交。

**自動選擇，不需要旗標。** `flashinfer_cutlass_unquant_moe_apply` 在
`tokens <= _MAX_DECODE_TOKENS`（16，實測交叉點：1.48/1.57/1.38/1.22/1.10/1.05/1.00/
0.96 對應 1/2/4/8/16/32/64/128 tokens）時轉交，並以 `gate_first=False` 讀
CUTLASS preprocessor 已經互換過的 w13 兩半——**兩條路徑共用一份權重**。形狀不支援
時明確拋錯並退回 GEMM。

**順手修掉一個會靜默損壞的缺陷。** `_reduce_silu` 的載入沒有遮罩，而 partials 是
`torch.empty` 的未初始化記憶體：任何 intermediate 或 hidden 不是 `_BLOCK_RED`(128)
倍數的模型都會讀進垃圾，**而且每次不同**。35B 的 512 剛好整除所以躲過了。現在
`decode_moe_body` 會在進 kernel 前拒絕這些形狀。這個缺陷是**測試抓到的、不是量測
抓到的**——第一版測試用 hidden=256/inter=64，立刻就不可重現。

---

### 2.8 本 repo 的 split-K 候選（**未入帳**：讀取時仍是未追蹤檔）

`ops/moe/triton/bf16_decode.py` 在本文寫作期間出現在工作樹上，**尚未進任何 commit**。
依 `E48b` 的規矩它的數字還不能引用為已入帳，但它的**結構**是本文最缺的那一格，所以記下來。

**它是五方對照裡唯一真的把 K 切開的 MoE GEMM。** grid 的第三軸就是 split：

```python
# ops/moe/triton/bf16_decode.py:172
_mv_splitk[(pairs, triton.cdiv(n, _BLOCK_N), _SPLIT_K)](
```

`_SPLIT_K = 8`（`ops/moe/triton/bf16_decode.py:59 _SPLIT_K`），每個 program 只走
`K // SPLIT_K` 的區間（`:94 k_per_split`）。與其他四家的對照：

| | grid 軸 | T=1 的 CTA 數（gate/up）|
|---|---|---|
| vLLM / SGLang / UnieInfer plugin | M×N | 128 / 256 / 256 |
| 這支候選 | **pairs × N × SPLIT_K** | 8 × 16 × 8 = **1024** |

也就是說它不是把 tile 調小，而是**多出一整個平行軸**：8 個 (token,expert) pair × 16 個
N block × 8 個 K split。這正是 §2.3 裡 vLLM/SGLang 已經在 LoRA 上用過、但沒搬到 expert
GEMM 的那個手法。

**歸約用 buffer 而非 atomics，理由寫在 docstring 裡** ——
這是三種歸約形態裡唯一有成文取捨的：

> `ops/moe/triton/bf16_decode.py:39` — 「The partials are reduced through a buffer rather
> than `atomic_add`: atomics make the sum order depend on which program finishes first,
> and a forward pass that is not bit-reproducible cannot be used to validate anything
> downstream of it.」

partials 是 `(SPLIT_K, pairs, n)` 的 fp32 緩衝（`:171 part`），由兩支獨立的第二趟 kernel
歸約：`:116 _reduce_silu`（順手做掉 `silu(gate) * up`，因為 w13 把 gate 與 up 併在同一個
輸出維，兩者本來就在同一個 partials 緩衝裡）與 `:144 _reduce_weighted`（同時把 split-K
與 token 的 top-k 兩層和一次做完，並乘上 router 權重）。

**launch 數的取捨也是成文的**：`:43` 記錄 partials 只有 256 KB 對上 33.5 MB 的權重，
「what costs is the number of launches, not their bytes」；先前用 torch 歸約 + 獨立
activation 是 6 個 launch / 0.098 ms，把兩件事折進歸約的 epilogue 後是 **4 個 launch /
0.060 ms**。

它與四家的另一個結構差異：**完全沒有 sort / align 這一步**。`pairs = tokens * topk`
直接當 grid 的第一軸，expert id 由 `ids_ptr + pid_m` 取（`:89`），權重按 checkpoint
原始 layout 讀（`:196`），連 vLLM `naive_block_assignment` 那個「M tile 只有 1 列有效」
的浪費都沒有 —— 因為它根本不做 M tile。

兩處值得帶進下一輪的自陳限制：

- `ops/moe/triton/bf16_decode.py:57 DECODE_MAX_PAIRS`（= 64）是與 grouped-GEMM 路徑的
  交叉點，註解明寫 **「The crossover was not swept」**。→ M3 的直接輸入。
- `:97` 記了一個踩過的陷阱，與本 repo 既有的量測紀律同型：`BLOCK_K` 不整除
  `k_per_split` 時，未加 mask 的版本**不會 fault、時間也正常，但回傳 1.48e+38**，
  只有數值檢查抓得到。**時間數字無法區分正確與錯誤的 kernel** —— 這條與
  `a latency number cannot tell a working decode from a broken one` 是同一課。

---

## 3. 結構差異（同功能、反結構）

| 功能 | 多數作法 | 例外 | 後果 |
|---|---|---|---|
| MoE GEMM 的 K 平行度 | 四個參照引擎一律不切，CTA 內序列走完整個 K | TurboMind 結構上可達但被啟發式排除；**本 repo 未入帳的 `ops/moe/triton/bf16_decode.py:172` 真的切了** | decode 的 MLP 由 CTA 數 × stage 數決定，四家都沒有旋鈕可調 |
| split-K 的 partial 歸約 | 四家在 MoE 上不存在 | 三種形態：TurboMind **epilogue 內序列 semaphore chain**（`src/turbomind/kernels/gemm/epilogue.h:402 SplitK_`）／vLLM·SGLang 在 **LoRA** 上用 relaxed atomics／本 repo 候選用 **fp32 buffer + 第二趟 kernel**（`ops/moe/triton/bf16_decode.py:116 _reduce_silu`）| 序列鏈省 workspace 但延遲乘上 splits；atomics 快但**不可位元重現**；第二趟 kernel 可重現但多一個 launch。**只有第三種寫下了取捨理由** |
| 小 M 的 sort | 都跑，只是緩衝夾小 | vLLM `naive_block_assignment` **整個跳過**（T ≤ 8）| 省兩個 launch；代價是 M tile 只有 1/16 有效，在頻寬受限形狀下不是成本 |
| align 緩衝的尺寸 | vLLM / SGLang 在**主機端**夾到 `numel × block_size` | **UnieInfer 用靜態最壞值 `8 + 256×31 = 7944`**，多發 241 個 M-block 讓它們在裝置上 `return` | 7968 個 CTA 發射、256 個有效；換到的是免去一次 D2H sync（`unieinfer/kernel-sm120/unieinfra_kernel_sm120/moe.py:281 moe_align_block_size` 的註解寫明這是刻意的） |
| routing top-k | 四家都有專用 kernel（1 CTA）| **UnieInfer 用 `F.softmax` + `torch.topk`**（`runtime/layers/moe/topk.py:219`）| 每層多兩個 torch launch，以本 repo 的 7.7–13.9 µs 地板計 |
| shared expert 併行 | 以 token 數或 capture 狀態為條件開 aux stream | **UnieInfer 只在 CUDA graph capture 時分岔**；TurboMind 不用 stream，把 dense FFN 夾在 `Forward`/`Combine` 之間 | eager 路徑上 UnieInfer 是序列的，vLLM 不是 |
| 小 M 的 K tile 方向 | SGLang 切窄 N（32）增加 CTA | **vLLM 加大 K tile（128）**，成文理由見 `model_executor/layers/fused_moe/fused_moe.py:1321 block_n` | 兩者方向相反，且都沒有附量測 |
| tuned tile 表覆蓋本卡 | SGLang 有 bf16 的；UnieInfer 抄了 SGLang 的 | **vLLM 沒有**，落到啟發式並印警告 | 引用 vLLM 的 MoE 數字前必須看那行警告有沒有出現 |

---

## 4. 可量測候選（下一輪的輸入）

| 候選 | 量什麼 | 為什麼值得 |
|---|---|---|
| **M1** vLLM 到底跑了哪個 backend | 貼出 `model_executor/layers/fused_moe/oracle/unquantized.py:200 _make_log_backend` 那行 log，確認是 TRITON 還是 FLASHINFER_CUTLASS；兩種都量一次 | **372 GB/s 這個數字目前沒有 backend 歸屬**。`model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:139 is_device_capability_family` 明確含 120，所以裝了 flashinfer 的機器上預設不是 Triton。這一格不釐清，後面所有比較都懸空 |
| **M2** split-K 的三種歸約在本卡的差 | 以 `ops/moe/triton/bf16_decode.py` 為底，第二趟 kernel（現況）vs relaxed atomics vs TurboMind 式序列 semaphore chain，掃 splits ∈ {1,2,4,8,16} | 三種歸約各自只在一個路徑上出現過，沒有人在同一形狀下比過。**atomics 版要同時報位元重現性**，否則就是拿掉了 `:39` 那條理由才換來的速度 |
| **M3** 切 N vs 切 K | 固定 T=1，把 `BLOCK_N` 從 64 掃到 16（SGLang 的方向）對上 `_SPLIT_K` 從 1 掃到 16（split-K 方向），同時印 GB/s；順帶把 `ops/moe/triton/bf16_decode.py:57 DECODE_MAX_PAIRS` 的交叉點掃出來（註解自陳未掃）| 兩個方向都在增加在飛的請求數。**如果切 N 就能拿到大部分收益，split-K 的複雜度就不值得** —— 這個否證比實作更省時間 |
| **M4** `naive_block_assignment` 值多少 | vLLM T=1..16，跨過 `T×8×4 ≤ 256` 的邊界（T=8→9）量 per-layer 時間 | 這是五家唯一的 decode 結構切換，邊界就在我們的工作點上。**若跨過邊界有可見跳變，UnieInfer 直接抄這一條的投報率高於 split-K** |
| **M5** UnieInfer 的 7944 元素 align 代價 | 把 `unieinfer/kernel-sm120/unieinfra_kernel_sm120/moe.py:323 max_len` 換成 vLLM/SGLang 的主機端夾法，量 T=1 的 per-layer 差 | 7968 發射 / 256 有效。免去 D2H sync 是刻意的，但沒有量過那次 sync 值不值 241 個空 CTA |
| **M6** routing 的三個 torch launch | UnieInfer 換成 SGLang 的 `moe_fused_gate` 或 vLLM 的 `topk_softmax`，量 per-layer | 唯一一家沒有 routing kernel。scoreboard 記的 0.2488 ms 已含 sort，這一格要單獨拆出來 |
| **M7** TurboMind 的 wave 啟發式能不能被推翻 | 把 `src/turbomind/kernels/gemm/context.cu:221 grid_size` 的 `num` 換成**實際非空專家數**（T=1 時 8）再跑 autotuner，看它會不會自己選出 `splits > 1` | 五家唯一有 autotuner 會真的量的路徑。**如果放寬後它自己選了 split-K，那就是最強的實作依據**；如果它量完仍選 1，那也是一個同等份量的結論 |
| **M8** shared expert 併行的條件 | UnieInfer eager 路徑（非 capture）T=1 的 shared expert 是否與 routed 重疊 | `runtime/models/qwen3_5_moe.py:292 stream_fork` 只在 capture 時分岔，vLLM 以 token 數為門檻。eager 路徑上這是白給的重疊 |

**排序建議**：M1 先做（它決定其餘比較的基準是否成立），然後 M3（它可能否證整個
split-K 方向，成本最低），再 M4。M7 是四個參照引擎裡唯一可能給出「別人量過」的答案的實驗。

---

## 5. 未解格（`?`）

| # | 未解的事 | 為什麼沒解掉 |
|---|---|---|
| U1 | **split-K 候選的狀態與數字** | 檔案存在（`ops/moe/triton/bf16_decode.py`，見 §2.8）但**是未追蹤檔，不在任何 commit 裡**；`git log` 近期無 MoE commit。依 `E48b`／`e45-r-p-over-n-not-reproducible`，**結構可以引用，數字不能算已入帳**。另外它 docstring 記的是 **834 GB/s（47%）對上 vLLM 342 GB/s**（`ops/moe/triton/bf16_decode.py:34`，整層、含兩次 matmul 與兩次歸約），與口頭流傳的「1220 GB/s」不是同一個量 —— 後者應為 gate/up 半邊。**引用前要先講清楚量的是哪一段。** 序列化到 serving 路徑後還要重量一次 |
| U1b | 這支候選有沒有 `register_kernel` 進 registry | 檔內只有 `decode_moe_body`，**沒有 `register_kernel`**，也沒有 `ops/moe/__init__.py` 的引用 → 目前 serving 路徑跑不到它。要進 §2.6 的選擇機制還需要 traits + priority，且要贏過 `ops/moe/flashinfer/cutlass_unquant.py:76 priority`（PERFORMANT）|
| U2 | UnieInfer 預設（CUTLASS）路徑的 grid 與 K 迴圈 | `ops/moe/flashinfer/cutlass_unquant.py:100` 呼叫 flashinfer 的 `cutlass_fused_moe`，**kernel 在 wheel 裡，不在本 repo**。已知的只有行為：`80-KERNEL-SCOREBOARD.md` §MoE 記錄 T=1 3.148 ms ≈ T=4096 3.184 ms（形狀完全平坦） |
| U3 | UnieInfer `--moe-backend triton` 那支的 `@triton.jit` 與 grid | `ops/moe/triton/bf16.py:54` 從外部 wheel `tokenspeed-triton-kernels` import `matmul`，該套件未 vendored 也未安裝在本機。主機端只看得到被釘死的 tile（`ops/moe/triton/bf16.py:62 _smem_guard`，`bm=64/bn=128/bk=128`，M=1 與 M=4096 同一組） |
| U4 | TRT-LLM 的 FINALIZE epilogue 內部怎麼歸約 | `cpp/tensorrt_llm/cutlass_extensions/` **不在 sparse checkout 內**。只能確定 `moe_kernels.cu` 全檔沒有 `atomicAdd`（只有 `:2308` 的 `atomicMax`，用於 FP4 amax） |
| U5 | TRT-LLM 在 autotuner 沒跑過時的預設 tactic | 該預設在 `cpp/tensorrt_llm/thop/moeOp.cpp`，**該目錄不在 sparse checkout 內** |
| U6 | TurboMind 在本卡實際選到哪個 kernel / tile / splits | dispatch cache 是 warm-up 時建的（`src/turbomind/turbomind.cc:361 WarmUp`）或用 `TM_GEMM_IMPORT` 匯入，**repo 裡沒有 cache 檔**。只能確定候選集合與排除規則 |
| U7 | TurboMind 目標模型走 `noaux_tc` 還是 `MoeGate_V2` | 由 `MoeWeight::topk_method`（`src/turbomind/models/moe_weight.h:91`）決定，是權重/設定屬性，讀不出來 |
| U8 | SGLang `srt/layers/moe/topk.py:1911 _post_process_topk_ids` 在無 EPLB 預設下是否真的零 launch | 未逐行確認；不影響 split-K 結論 |
| U9 | 這一輪**沒有任何量測** | 依 `00-METHOD.md`，本系列只做原始碼對照。§4 是下一輪的輸入 |

### 5.1 需要注意的版本差異

- **SGLang 這個 checkout 是 fork，MoE 已被重構過**：Triton kernel 搬到
  `python/sglang/kernels/ops/moe/`，upstream 的
  `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` **在此版本不存在**。
  引用 SGLang MoE 的行號時要注意這一點。
- **vLLM 的 `model_executor/layers/fused_moe/` 與 upstream 合流點 `7cc2e8e74fa0` 完全相同**
  （`git diff 7cc2e8e74fa0 HEAD -- vllm/model_executor/layers/fused_moe/` 為空）。
  本文引用的 `SPLIT_K` 死參數與 `SWAP_AB` 都是 **upstream vLLM 的程式碼，不是 UnieAI fork 的**。
- `40-decode.md` 第 9 列把 UnieInfer 的 MoE 指到 `runtime/moe/`，**那個目錄只有 EPLB 與
  expert-location**；真正的層在 `runtime/layers/moe/`，kernel 在
  `tokenspeed-kernel/.../ops/moe/`。下次修訂 `40-decode.md` 時應更正。
