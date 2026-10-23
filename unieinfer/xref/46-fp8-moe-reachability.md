# 46 — FP8 block-scale MoE 在 sm_120 上到底有沒有路

版本見 `00-METHOD.md`。本文回答一個具體問題：Qwen3.5-122B-A10B-FP8
（`hidden=3072`、`num_experts=256`、`top_k=8`、`moe_intermediate=1024`、
block-wise fp8 `[128,128]`）在單機兩張 RTX PRO 6000（sm_120）上，
**MoE 這一層有沒有任何一條可執行的路**。

量測日 2026-08-14。所有結論可用 `unieinfer/bench/moe_probe.py` 重跑——它把該形狀
從模型裡拆出來，單卡、不載權重、每個 backend 幾秒鐘回答。

---

## 0. 一句話的結論

**「sm_120 上沒有 fp8 block-scale MoE kernel」是我們這棵樹的事實，不是這張卡的事實。**
vLLM 在同一台機器、同一個模型上跑得起來，選的是 DEEPGEMM。我們也有 DeepGEMM，
但它只以 DeepEP 形態存在，所以單機 TP 根本呼叫不到。差距是管線，不是 kernel。

診斷過程中錯了兩次，兩次都是**讀了宣告而不是讀了實際解析出來的值**：

1. 先以為是「TRT-LLM-Gen 在小 M 沒有 kernel」。它在 M=8、M=128 一樣失敗，
   而且它選到的 kernel 名字尾是 `_sm100f`——是為資料中心 Blackwell 編的。
2. 再以為是「vLLM 退回 CUTLASS，所以該移植 `cutlass_moe.py`」。那句
   "Falling back to CUTLASS" 是**警告訊息的字面**；引擎實際印的是
   `Using DEEPGEMM Fp8 MoE backend`。而且 `cutlass_moe_mm` 的 C++ 派發只有
   sm90 與 sm100 兩支，120 落到 `NOT_IMPLEMENTED`，那個檔案裡也**完全沒有
   `block_shape`**。照它移植會是雙重的白工。

---

## 1. 五方對照：誰有 fp8 block-scale MoE，跑在什麼架構上

| 引擎 | 路徑 | sm_120 可用 | 備註 |
|---|---|---|---|
| **vLLM** | `experts/deep_gemm_moe.py`（非 EP） | **是** | 實測選中；`triton_deep_gemm_moe.py` 是選擇器 |
| vLLM | `experts/cutlass_moe.py` | 否 | `cutlass_moe_mm` 只派發 sm90/sm100；無 `block_shape` |
| vLLM | `experts/triton_moe.py` | 是（未實測） | 純 Triton，與架構無關的保底路徑 |
| **本 repo** | `moe/deep_gemm/deepep_fp8.py` | 否（結構性） | 宣告 `supports_all_to_all_ep`，無 group 即 `ValueError` |
| **本 repo** | `moe/deep_gemm/contiguous_fp8.py` | **是**（本次新增） | 見 §3 |
| 本 repo | `moe/flashinfer/trtllm_fp8.py` | 否 | GEMM runner 選到 `_sm100f` cubin，所有 M 皆失敗 |
| 本 repo | `moe/flashinfer/cutlass_fp8.py` | 否 | 上游明講 `FP8 block scaling not yet implemented for Blackwell` |

宣告 `weight_dtype: fp8` 的 solution 全樹**只有三個**（上表後三列）。
`gluon`、`marlin`、`triton`、`cuda`、`native` 都沒有 fp8 MoE 路徑——`triton` 之所以
被排除是這個原因，不是別的。

### 1.1 一個值得抄的設計習慣，和一個不該抄的結論

vLLM 對這個模型在 Blackwell 上**主動關掉** DeepGemm：

```python
_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES = {"qwen3_5_text", "qwen3_5_moe_text"}
# is_device_capability_family(100) or is_device_capability_family(120)
# 理由：E8M0 scale 格式在此架構上掉精度
```

我們的模型 `text_config.model_type` 正是 `qwen3_5_moe_text`，判斷也涵蓋 120 家族。
**但那個開關關的是 `quant_config.use_deep_gemm`，不是 MoE backend 選擇器**——同一次
啟動裡，config 在 21:03:40 印出停用警告，worker 在 21:04:09 仍選了 DEEPGEMM。
兩個開關，不要混為一談。

精度顧慮本身是真的，而且**我們這條路已經處理了它**：`deepep_fp8` 的 weight
preprocessor 在 `deep_gemm_requires_ue8m0()` 為真時把 checkpoint 重新量化成
UE8M0 並發警告。新的 contiguous 路徑共用同一個 preprocessor，所以行為一致。

---

## 2. 為什麼 DeepEP 那條在單機兩卡上不可達

`deepep_fp8.py` 的 `_get_dispatcher` 從 plan 取 `deepep_group`，取不到就丟
`ValueError: DeepEP MoE plan is missing deepep_group`。單機 TP 的 plan 裡沒有這個
group，也不該有——DeepEP 是跨 rank 的 all-to-all，兩張卡共用一個 process 時沒有
那條線。

但它的**計算核心與 EP 無關**：`dispatch_b()` 到 `combine_a()` 之間只吃五樣東西——
fp8 tokens、block scales、routing ids、routing weights、每專家 row 數——吐一列 bf16。
DeepEP 只是那五樣的來源之一。

---

## 3. 本次新增：`moe/deep_gemm/contiguous_fp8.py`

在本地產生那五樣，重用同一個 permute／GEMM／unpermute 核心與同一個 weight
preprocessor，**不長出第二份數值實作**：

| DeepEP 提供 | 非 EP 版怎麼來 |
|---|---|
| fp8 tokens + scales | `per_token_group_quant_fp8`（UE8M0 時走 `_ue8m0` 變體，因為 scatter 要 `[tokens, groups]` fp32 而非 packed int32） |
| 每專家 row 數 | `torch.bincount(topk_ids)` 後向上補齊到 128 |
| routing ids／weights | 呼叫端已提供 |
| combine | 不需要；`deepep_gather` 直接就是加權歸約 |

刻意不做的：無 capacity 上限、無 masked GEMM、無 low-latency 模式。那三樣在
DeepEP 版存在是為了界定跨 rank 流量，這裡沒有那個成本。

**一個安靜的坑**：補齊必須只補「有 token 路由到」的專家。全部補的話仍然會跑，
但單一 decode token 會配到 256×128 列而不是 8×128，只會表現成一個吞吐之謎。
`test_deep_gemm_contiguous.py` 釘住這一點。

---

## 4. 工具鏈：兩個把「有路」偽裝成「沒路」的環境問題

新路徑第一次跑出來的是 `NVCC compilation failed`，而不是 kernel 錯誤。兩層：

**4.1 兩份 CCCL 混用。** 環境同時裝了 `nvidia/cu13/include/cccl/cuda/std/`（nvcc 自己
加）與 `nvidia/cuda_cccl/include/cuda/std/`（只有 CPATH 到得了）。`sm120_env.sh` 用
`nvidia/*/include` 全域塞 CPATH，於是同一個編譯單元的 `<cuda/std/utility>` 來自新的、
`<cuda/std/version>` 來自舊的，`_LIBCUDACXX_HAS_SPACESHIP_OPERATOR` 在使用處未定義：

```
cccl/cuda/std/utility:58:39: error: missing binary operator before token "("
```

這行讀起來像原始碼 bug，實際是 include 路徑 bug。修法是把 `cuda_cccl` 排除在
CPATH 外——nvcc 自己會帶 CCCL。

**4.2 nvcc 與其標頭版本不一致。** cu13 wheel 裡 **nvcc 是 13.2、CUDART 標頭是 13.3**，
CCCL 直接拒編：

```
__cccl/cuda_toolkit.h:41: error: "CUDA compiler and CUDA toolkit headers are incompatible"
```

那個標頭自己為「CTK 比編譯器新」留了 `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK`，
差一個 minor，用它。wheel 修好就該把旗標拿掉，`sm120_env.sh` 的註解寫明了。

**兩處都在 `sm120_env.sh`，是全域改動**，所以「不影響既有 Qwen 模型」必須實測而非
推論。

---

## 5. 附帶發現：多卡的前提是 P2P，而這台機器只有兩對

跑 122B 與 35B 的兩張卡組態時撞到的，與 MoE 無關但同一條路上：

```
CUDART error: peer access is not supported between these two devices
```

`nvidia-smi topo -m` 說明一切——**只有 (0,1) 與 (2,3) 是 `PIX`（同一個 PCIe switch）
可以 P2P，跨組的四對都是 `NODE`（經 host bridge）不行**。vLLM 在 (1,3) 上跑得起來，
因為它偵測到沒有 P2P 就關掉 custom all-reduce；我們的 collectives 與 CUDA graph
capture 都預設有 P2P。

兩個缺陷，兩處都修了：

1. **啟動時不檢查。** 錯誤發生在權重載完之後的 graph capture 裡，訊息是一句
   `CUDART error`，既不說是哪兩張卡，也不說哪一對可行。`engine.py` 的
   `_check_peer_access` 現在在載入前擋下，並把**可用的配對列出來**——照的是
   `AGENTS.local.md` 記的那個習慣（vLLM 對 `max_num_seqs` 直接報出該用的數字）。
2. **logits all-gather 硬要 symmetric memory。** `logits_processor.py` 的
   `_init_all_gather_state` 檢查了平台、world size、跨節點、vocab 對齊，就是沒檢查
   互連。RSAG kernel 走 multimem 指令，buffer 必須以 symmetric memory rendezvous，
   在 PCIe 上直接 `invalid device ordinal`。現在失敗會退回一般 all-gather 並記錄
   真正的原因，且**按 key 快取**——否則每一層都會重試一次注定失敗的 rendezvous。

順帶：這條也解釋了為什麼本輪第一次的兩張卡 sweep 全臂失敗——我把 GPU 從原本的
`0,1` 改成 `1,3` 以避開別的工作，正好選中不能 P2P 的一對。**是操作錯誤，不是引擎
迴歸**，但引擎不該用那種訊息告訴我。

---

## 6. 為什麼 35B 在併發下輸掉：循環狀態的檢查點間隔

與 MoE 無關，但是同一輪量測挖出來的，而且比 MoE 重要得多。

**症狀。** 35B-A3B 的 agent 階梯從 C=16 起全輸（0.31x / 0.67x / 0.47x / 0.50x），
RAG 在 C=64 之後也輸（0.75x / 0.72x）。C=1 反而小贏（1.10x）。

**歸因，逐層排除。** 兩個引擎跑同一支 MoE kernel（`FlashInfer CUTLASS
Unquantized`），所以 MoE 不是差距來源。我方每字 10.7 ms 對 vLLM 47.5 ms、前綴重用
63.2% 對 32.1%——解碼和快取都更好——但首字 18.1 s 對 3.4 s，在 TTFT ≤ 10 s 的門檻下
幾乎每個請求光在首字就出局。極低的 TPOT 配極高的 TTFT 是「同時常駐請求太少」的
特徵：`#running-req: 1` 出現 190 次。

**逐步縮小。** 排程器 trace（`TOKENSPEED_LOG_PLAN_TRACE` 觸發檔）顯示：
- 5,208 個有請求的步驟裡，只有 37 步是 `token budget exhausted`——不是預算問題
- 候選名單有 16 個請求，其中 14–15 個是 `Submitted`，**每一個都被
  `schedulePrefillFirstChunk` 拒絕**，decode 那側零拒絕
- 准入拒絕 39,227 次，形狀一致：`unscheduled=6388 gen_reserve=256 matched=24576`

**根因。** `AdmissionPlanner::fits()` 對每個 group 各自無條件進位地累加 parent 需求，
而拒絕當下的實際數字是：

```
g0(need=52 free=0 slots=1)  g1(need=52 free=0 slots=1)
g2(need=52 free=0 slots=1)  g3(need=52 free=14 slots=16)
parents_needed=159  empty_parents=55
```

四個 group 各需要 52 個 block，但三個的 `cache_blocks_per_lcm_block` 是 **1**。
per-group 的每 block 位元組數解釋了為什麼：

| group | bytes/block | packing |
|---|---|---|
| `full_attention` | 2,621,440 | 16 |
| `linear_attention_0/1/2` | 21,463,040 | **1** |

parent 是 41,943,040 bytes，裝一個 20.5 MB 的循環狀態 block 就放不下第二個，
**每個 parent 浪費 47%**。而 `_solve_packing` 沒有錯——`_packing_by_group_ratio`
已經先用絕對位元組比例算過，孤立 group 保留該值，1 是正確的推導結果。

**真正的成本在檢查點間隔。** `qwen35.py` 的 recipe 讓每個 linear attention 層在
**每一個 cache block** 存一份完整的 `ssm` + `conv` 快照，而 block 是 128 token——
等於檢查點間隔 128。一個 6,656 token 的 chunk 因此要 52 × 20.5 MB × 3 = **3.2 GB
的狀態快照**，224 個 parent 只夠一個請求 prefill。

**修法，由淺到深：**

1. **拉長檢查點間隔**（每 N 個 block 存一次）。parent 需求直接除以 N；以 N=4
   （間隔 512）估，上面那格的 `parents_needed` 從 159 降到約 42，併發從 1 變 5。
   代價是前綴命中時要從較早的檢查點往前重算。
2. **讓 parent 尺寸配合 20.5 MB**，回收那 47%。只有約 2x，且動到 arena 幾何。
3. **讓循環 group 用自己的 block 粒度**。C++ 的 `KvCacheManager` 本來就是
   per-manager 持有 `cache_block_tokens_`，架構通；卡住的是 Python plan 只有一個
   全域 `logical_block_tokens`（`plan.py:77/225/267`）。前綴匹配的 page hash 是以
   單一 page size 計算的，改成 per-group 需要一併處理——而那條路徑已經出過一次
   「快照比任何匹配都深一頁」的事故（見 `unieinfra-prefix-cache-dead-on-hybrid`）。

**U4** — 上述 1 或 3，帶量測。這是目前所有 35B 敗績的單一共同原因。

---

## 6.1 五方對照：循環狀態一個請求要佔多少

先做的是量測，再做的對照。四家在關鍵點上一致，只有本 repo 不同。

| 引擎 | 每請求的狀態 | 狀態的前綴重用 | 依據 |
|---|---|---|---|
| **vLLM** `align` | **2 個 block** | 一份對齊檢查點 | `MambaSpec.max_memory_usage_bytes` |
| vLLM `all` | `ceil(max_model_len / block)` | 完整，每 block 一份 | 同上；**存在但非預設** |
| vLLM `none` | 1 個 block | 無 | 同上 |
| **SGLang** | **1 個 active slot** | 獨立的 int8 壓縮檢查點池，按 radix node | `mamba_checkpoint_pool.py` |
| **TRT-LLM** | **1 個 slot** | 無 | `mamba_cache_manager.py`：`size=(layers, max_batch_size) + state_shape` |
| **TurboMind** | 1 份狀態，按層×頭切塊 | 無 | `GatedDeltaNetLayer.cc`：`num_blocks_ = layer_groups * head_groups` |
| **本 repo** | **`ceil(len / 128)`**（實測 52） | 完整，每頁一份 | `qwen35.py` recipe + `entry_stride_tokens=1` |

**分歧本身就是結論。** 四家的每請求狀態都是 O(1)；我們是 O(序列長度)。而我們實作的等同於 vLLM 的 `all` 模式——vLLM 保留它作為選項，實測這個模型時選的是 `align`
（log: `Mamba cache 'align' mode`）。

**SGLang 那一欄揭露的結構才是重點**：它把 *active*（全精度、每個在跑的請求一個、kernel
直接用）和 *checkpoint*（radix 擁有、int8 壓縮、閒置、命中時 copy-on-write 反量化回
active）分成兩個池。檢查點因此**不佔准入容量**。

對照本 repo 的 [[lcm-arena-replaced-separate-pools]]：我們把兩個池合併進一個 LCM
arena。合併本身讓「存檢查點」和「准入新請求」開始爭搶同一份容量——這才是
§6 那 52 個 block 的來源。**問題不是存了檢查點，是檢查點和活躍狀態從同一個池扣。**

### 修法選項，依風險排序

1. **把循環 group 的准入需求改成每請求常數**（vLLM `align` 的語意）。改的是准入
   計算，不動 arena 幾何，也不動前綴匹配的粒度。檢查點數量另外界定。
2. **把檢查點移出准入路徑**（SGLang 的結構）。最完整，也最大：要讓檢查點池獨立
   配置並可壓縮，等於部分回退 LCM arena 的合併。
3. ~~把 base block 設成各 group 的最小公倍數~~ —— **不要**。它把每請求需求從 52 降到
   13，仍然是 O(len)，而且會動到 page hash 的前綴匹配粒度，正是
   [[unieinfra-prefix-cache-dead-on-hybrid]] 出過事故的地方。實測 `entry_stride_tokens=4`
   直接被排程器擋下：`cache group block tokens must divide the scheduler cache block
   tokens`——group 只能比排程器更細，不能更粗。

---

## 6.2 實測：准入修好了，執行端接不住

依 §6.1 的對照做了最小版本——讓狀態群組只在檢查點與活躍尾端配置實體區塊
（`SwaManager::Acquire` 稀疏化，`AdmissionBlocksNeededFor` 讓准入看到同一個數）。
`--state-entry-stride-tokens` 從 server arg 一路接到 manager，預設 1 等同原行為，
409 個 C++ 測試釘住語意。

**排程器那半成立。** stride=4 在 35B agent C=16 上：

| | stride=1 | stride=4 |
|---|---|---|
| 每個狀態群組需要的區塊 | 52 | **13** |
| `parents_needed` | 159 | **42** |
| 准入拒絕次數 | 44,458 | **3,875** |
| `#running-req` | 幾乎恆為 1 | **7** |

**執行那半不成立。** 引擎放行 `#new-seq: 4 / #running-req: 7` 之後隨即 abort。
原因在 `KvCacheManager::BlockTablePageIds`：

```cpp
ids.push_back(block_ref ? ResolveKernelPageId(block_ref->Location()) : 0);
```

null 區塊映射到 **page id 0**，而 page 0 是 null 佔位頁。稀疏表因此不是讓 kernel
拿不到 id，而是**把每個未配置位置的寫入都導向同一頁**——多個請求並行時就是資料競爭。

**這正是 SGLang 把 active 池保持稠密的理由。** 循環狀態是每一步都在寫的活躍狀態，
不是只在頁邊界才寫；稀疏只能存在於 checkpoint 那一側。要讓稀疏 active 表可用，
寫入端必須知道「在下一個檢查點之前繼續寫回上一個檢查點的頁」，那是 GDN 層的改動，
不是排程器的。

**U5** — 二選一：
1. 讓執行端支援稀疏狀態表（GDN 寫入端 + `BlockTablePageIds` 的空洞語意）。
2. 改採 SGLang 結構：active 池稠密、每請求一個 slot；checkpoint 池分離、可壓縮、
   不佔准入容量。繞開整類空洞問題，但等於部分回退 [[lcm-arena-replaced-separate-pools]]。

現況已保留在樹裡且預設關閉（stride=1），既有量測不受影響。

---

## 7. 待辦（U 編號承 `70-PORTS.md`）

- **U1** — 跑通後量測 122B 的四種並行組態（`70-PORTS.md` 的 122B 條目）。
- **U2** — `triton_moe.py` 這條與架構無關的保底路徑尚未移植。DeepGEMM 若在某個
  模型上真的掉精度，那是唯一的退路，目前我們沒有退路可退（§1 表最後兩列都死）。
- **U3** — backend 選擇失敗時整個模型死。vLLM 列 12 個候選逐一篩，我們選中一個就
  沒有第二次機會。這是 §0 那兩次誤判之所以昂貴的原因：錯誤訊息永遠指向被選中的
  那一個，不會告訴你其他家的狀況。
