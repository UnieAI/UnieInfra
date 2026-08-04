# Design: TokenSpeed-Kernel 設計原則 → 跨引擎可移植優化參考 (lmdeploy/sglang/tokenspeed/TileRT → vLLM)

> 綱:TokenSpeed-Kernel 設計原則 blog (pytorch.org, 2026-06-25, AMD Triton 团队 + TokenSpeed 团队)。
> 對照來源(工作區實地掃描):lmdeploy TurboMind C++、sglang、tokenspeed、TileRT、vLLM(2026-07-24 版, v1 引擎)。
> 目標讀者:要把 vLLM arch 改得跟 tokenspeed 一樣好擴充、但功能用 vLLM 的工程師。
> 落地位置:`docs/design/kernel-system-design-principles.md`(與既有 `kernel-plugin-contract.md` / `scheduling-improvements.md` 互補;並在 `unieinfra-architecture.md` 的 extension map 加一行指向它)。

---

## 1. 設計原則(blog 綱)

TokenSpeed-Kernel 把「kernel」當成一等子系統,而非隱藏的快速路徑集合。三個實踐原則:

1. **多硅片支援必須是基礎性的** — kernel 系統直接理解平台能力,而不是把硬體檢查散落成條件判斷。同一 op 對不同硅片有多個 solution,全部透過選擇系統競爭。
2. **可移植性與高效能並存** — 新模型先有一條可移植路徑(Triton)盡快跨硅片跑起來,再逐步採用更優化 kernel(AMD 用 Gluon、NVIDIA 用 CuteDSL、供應商包裝器為補充)。兩條路並列,不是互斥。
3. **kernel 快速迭代需要護欄** — 從想法到落地路徑要短:精簡依賴、獨立 benchmark/profiling、不改模型碼即可註冊試驗 kernel。同結構也給 AI agent 的 kernel 開發清晰工作邊界(試一個 kernel → 驗證 → bench → 註冊,不動模型碼)。

### 分層架構
- **公共 API**(平台/方案無關):`mha_prefill` / `mha_decode_with_kvcache` / `moe_apply` … runtime 描述「算子問題」(張量、格式、模型特徵、執行約束),不指定「AMD/Triton kernel」。
- **註冊表 + 選擇器**:後端用 `@register_kernel` 宣告 op 家族/模式、solution 名、平台能力需求、張量簽名、特徵、優先級。選擇器依 `CapabilityRequirement(min/max_arch, required_features, vendors)` + `FormatSignature` + `PlatformInfo(interconnect pcie/nvlink)` 過濾→排名→回傳可執行 callable,並快取解析結果(固定 模型/平台/dtype/特徵 下選擇穩定)。可強制 override solution/精確 kernel 供 debug/bench。
- **樹外插件**:`plugins/` 用同一裝飾器註冊,自配優先級,與內建並列競爭;不必 fork。
- **獨立發布**:kernel 包可脫離完整服務棧單獨裝用。**`tokenspeed-kernel-amd` 已被 vLLM 採用**(blog 結論),證明此架構能反向餵養 vLLM。

### 職責邊界(blog 原話)
- runtime 負責:模型執行、排程元資料、頁表、路由狀態。
- kernel 負責:算子 API、後端註冊、選擇、數值計算、benchmark、profiling。
- 平台相關效能優化保留在平台專用 kernel,不散落模型碼。

---

## 2. 工作區實作對照(實地掃描,精簡表)

| 子系統 | 來源實作 | vLLM 現況(2026-07-24) | 移植價值 |
|---|---|---|---|
| **kernel 選擇層** | tokenspeed-kernel `ops/<family>`(triton+cuda+cute_dsl+flashinfer+deep_gemm+trtllm+gluon,**每 family 保證有 Triton portable**)+`registry.py`/`selection.py`(`SelectionObjective.PORTABILITY`)/`platform.py`(sm120 bug)/`signature.py`+`plugins/`;兩層:Level-1 ops family + Level-2 attention 演算法 registry | `v1/attention/selector.py` **硬編碼** `AttentionSelectorConfig` NamedTuple + `resolve_obj_by_qualname`,非 capability/feature 框架 | **高**(解鎖 sm120/AMD;有藍圖 + sm120 設計文件) |
| **C++ FSM + 型別安全 KV reuse 排程** | tokenspeed-scheduler `csrc/{fsm,cache,scheduler}`:`forward/cache/pd` 三組 state machine + `KvCacheManager`/`KvCacheCoordinator`/`cache_group` + mamba/full_attn/swa manager;**編譯期型別強制 KV 安全重用** | 排程全 Python(`v1/core/sched/scheduler.py`),無編譯期型別保證,競態靠執行期 assert | **架構級/長期高** |
| **GEMM 自調優 + DispatchCache** | TurboMind `kernels/gemm/{registry.cu,dispatch_cache.h,gpu_metric.cu}`:自寫 sm70–120 + cublas fallback,Export/Import tune 結果,實測 L2/MMA 吞吐做啟發式 | 仰賴 cutlass/cublas/flashinfer,無統一 dispatch cache/離線 tune 匯出 | **中**(sm120/消費級卡) |
| **async 多 phase 管線** | TurboMind `engine.cc` inbound/outbound 佇列 + `phases>1` BatchData slot 重疊 host scheduling↔device forward;BatchOp kPrepare/kForward/kFetch/kUpdate 明確 contract;「device-content 只在 executor thread、scheduler 不知位元含義」 | vLLM 有 `async_scheduling`(`engine/core.py:227` + `gpu_model_runner` 大量分支) | **低-中**(概念已有;可借邊界設計減競態面) |
| **VLM partial-sibling multimodal prefix trie** | TurboMind `engine/prefix_trie.h` Search/Find + `LogicalBlock::partial` sibling edge + AdmitPrompt 在圖片起點 block fold image fingerprint 進 cumulative key + `cache_prompt=all/auto` | `block_hashes` + `mm_hash`(`encoder_cache_manager`)但 **partial-sibling 結構與 fingerprint 折進 trie key 較弱** | **中-高**(VLM 多輪命中) |
| **SSM 狀態 checkpoint pool + prefix restore** | TurboMind `block_trie.py` StateManager(runtime/checkpoint 分池)+ 兩階段 pin/copy/release + ancestor chain verify + `prefix_cache_decode_state_interval`;sglang `mamba_checkpoint_pool.py` + `hi_mamba_radix_cache.py` | `mamba_cache_mode=align` + hybrid coordinator(`has_mamba_layers`)但**未見獨立 checkpoint pool + LRU + restore-from-cached-state**(偏重算/converged hit) | **高**(GLM-4.7/DeepSeek-V4 多輪 TTTC) |
| **HiCache 多層 + per-pool 命中策略 + prefetch 逾期** | sglang `hicache_storage.py` PoolName(KV/MAMBA/SWA/INDEXER/DeepSeekV4 C4/C128 + 各自 _STATE)+ `PoolHitPolicy`(ALL_PAGES for DSA / TRAILING_PAGES for Mamba/SWA)+ `PoolTransfer` 統一 device↔host↔storage + `PrefetchTimeoutConfig`;`hybrid_cache_controller.py` per-layer done-counter | `kv_offload/tiering/`(CPU primary + secondary + p2p/nixl)但**無 SSM-state/SWA/DSA indexer 細分 pool,prefetch 逾期較簡** | **高** |
| **TBO:MoE dispatch 重疊 CPU 視窗** | sglang `batch_overlap/two_batch_overlap.py` 兩 batch 重疊 CPU(DeepEP/Mooncake/Mori/Nixl dispatch)與 GPU forward + token-distribution threshold 自適應 | `async_scheduling` 重疊 scheduler↔worker 但**未把 MoE all2all dispatch 重疊進第二 batch CPU 視窗** | **中-高**(大 EP MoE CPU-bound) |
| **DFlash/Spec V2/多層 draft/frozen-KV MTP** | sglang `speculative/{dflash_worker_v2,multi_layer_eagle_*,frozen_kv_mtp_*,external_corpus_manager}` + cpp_ngram | `v1/spec_decode/{eagle,dflash,...}` 已合併 dflash,但 frozen-KV MTP + 多層 draft + 外部 corpus 較完整 | **中** |
| **outlines jump-forward** | sglang `constrained/outlines_jump_forward.py` | `v1/structured_output`(xgrammar/guidance/outlines/lm_format_enforcer)但 **jump-forward(跳多 token)較弱** | **中**(結構化生成吞吐) |
| **DeepEP/Mooncake/Mori/Nixl 多 dispatcher** | sglang `layers/moe/token_dispatcher/` | `elastic_ep` + `all2all` 但 dispatcher 多樣性/RDMA 整合較窄 | **中**(大規模 EP) |
| **DSA/Quest sparse** | sglang `mem_cache/sparsity/algorithms/{deepseek_dsa,quest}` + `sparse_coordinator` | `mla/*_sparse.py` + `sparse_swa` 已覆 MLA sparse | **低**(已追上 MLA sparse;通用 Quest/DSA 邊際) |
| **TileRT(tile-level runtime)** | `third_party/tilert`:**閉源預編 binary wheel**,pin 8×B200 + CUDA13.2 + Python3.12 + torch2.11+cu130(README 硬 ABI);Python 層僅 `pd_vllm/` PD-disaggregation adapter + weight 轉換 + profiles;核心 tile-runtime = 編譯器驅動拆 tile-level task 跨 device 重疊 compute/I/O/comm | — | **不移植**(閉源 B200-only;設計文件 §2 明說不可跑/不能 port;僅作未來低延遲路線概念參考) |

**vLLM 已追上、不必抄**:EPLB、elastic_ep、kv_offload tiering 基本架構、async_scheduling、hybrid SSM coordinator、turboquant、DeepSeek-V4 sparse MLA、dflash、tokenspeed MLA backend(`vllm/v1/attention/backends/mla/tokenspeed_mla.py`)、**tool_parser 46 個(含 `inkling`/`deepseekv4_engine`/`glm47_moe`/`kimi_k2`)**、**MTP 模型覆蓋 10+(openpangu/nemotron_h/mimo/qwen3_5/exaone/ernie/step3p5/longcat + deepseek_eagle3)**。

---

## 3. 可移植優先清單(依 ROI / 與設計原則契合度)

| # | 項目 | 來源 | 契合原則 | vLLM 落點 | ROI |
|---|---|---|---|---|---|
| 1 | capability 兩層 kernel framework(每 family 保證 Triton portable + `CapabilityRequirement`/`PlatformInfo`/`SelectionObjective.PORTABILITY` + 樹外 plugins) | tokenspeed-kernel | 1+2+3 | 取代 `v1/attention/selector.py` 硬編碼 → 新 `v1/kernel/` 子層 | **高**(解鎖 sm120/AMD) |
| 2 | SSM 狀態 prefix cache + restore(獨立 checkpoint pool+LRU+兩階段 pin/copy/release+ancestor verify+`decode_state_interval`) | TurboMind `block_trie.py` / sglang `mamba_checkpoint_pool.py` | 2(portable:重算;perf:restore) | `v1/core/single_type_kv_cache_manager.py`(mamba_cache_mode) | **高** |
| 3 | HiCache per-pool 命中策略 + SSM-state 分層 + prefetch 逾期(`PoolHitPolicy.TRAILING/ALL` + PoolName 細分 + `PrefetchTimeoutConfig`) | sglang `hicache_storage.py` + `hybrid_cache_controller.py` | 2 | `v1/kv_offload/tiering/` | **高** |
| 4 | **多深度 MTP + per-depth KV repair + lookback stash + draft 端套 grammar mask**(見 §4A) | tokenspeed `drafter/mtp.py` | 3(快速試驗 draft 不改模型碼) | `v1/spec_decode` 加 `MultiDepthMTPProposer` + draft mask | **中-高**(接受率,見 §4A) |
| 5 | VLM partial-sibling multimodal prefix trie(image fingerprint 折進 trie key + partial sibling edge) | TurboMind `prefix_trie.h` + `AdmitPrompt` | 2 | `v1/core/block_pool` + `kv_cache_utils` block hash | **中** |
| 6 | TBO:MoE dispatch 重疊進 CPU 視窗 | sglang `two_batch_overlap.py` | 2 | `v1/engine/core.py` async_scheduling + worker | **中** |
| 7 | outlines jump-forward(結構化輸出) | sglang `outlines_jump_forward.py` | 2 | `v1/structured_output` | **中** |
| 8 | C++ FSM + 型別安全 KV reuse 排程 | tokenspeed-scheduler | 1+3(邊界清晰) | 長期:`v1/core/sched` 逐步型別化;短期先抽 cache FSM 介面 | **架構級/長期高,短期成本高** |
| 9 | **tool call 約束上移到 logits/grammar 層**(見 §4B) | 方向(非直接抄;tokenspeed SMG 無源碼) | 2 | `v1/structured_output` 接 tool JSON schema | **中**(準確度,見 §4B) |

> 註:#1 與既有 `kernel-plugin-contract.md`(agent→kernel pipeline)互補:本項是「runtime↔kernel 選擇層」,落地後 agent 產出的 kernel 透過同一 registry 參與競爭,接上 `kernel-plugin-contract` 的 promotion 介面。

---

## 4. MTP 與 toolcall 準確度調查(實作差異 + 修正方向)

> 命題:vLLM 有 MTP 與 tool_parser,但**實作方式會傷害 MTP 接受率與工具呼叫準確度**。實地確認兩條機制並與 tokenspeed 對照。
> 共通根因(記下):兩條都指向同一架構分歧 —— **vLLM 傾向「先讓模型自由生成,再後處理/事後過濾導正」**(draft 不 mask + 事後 `validate_tokens` 前綴裁剪;tool call 文本層後處理 + partial JSON);**tokenspeed(推斷)傾向「在產生端(logits)就約束,讓每個 token 一產出即合法」**(draft 經 logits_processor/mask;tool call 走 logits 層 grammar)。後者對「準確度 + spec 接受率」都更友好,代價是 logits mask 計算成本與 grammar 編譯。

### 4A. MTP / spec decode × structured output(影響接受率 + 統計正確性)

**vLLM 實作鏈(實地確認):**
1. **draft 端不套 grammar mask** — `vllm/v1/spec_decode/llm_base_proposer.py:428 _greedy_sample` 與 `_sample_draft_tokens` 直接 `compute_logits().argmax()`,只過 `vocab_mapping.constrain_draft_logits`(`v1/spec_decode/vocab_mapping.py:152`,詞表對齊,**非 schema**);`is_masked_token_mask` / `is_rejected_token_mask`(`llm_base_proposer.py:221-238`)是 KV slot / parallel-padding 管理,**非 grammar**。
2. **grammar mask 只在 target 端套** — `vllm/v1/worker/gpu_model_runner.py:4566 apply_grammar_bitmask` 只在 target forward 的 logits;`rejection_sampler.apply_logits_processors`(`v1/sample/rejection_sampler.py:289`)把 `sampling_metadata.allowed_token_ids_mask` 套到 `target_logits`(`masked_fill_(token_mask, -inf)`)。→ **target 驗證端合法,draft 提案端不受 schema 約束**。
3. **事後前綴裁剪式過濾** — `vllm/v1/core/sched/scheduler.py:2069 與 2098` 對 draft `spec_token_ids` 呼叫 `metadata.grammar.validate_tokens(spec_token_ids)`;`validate_tokens`(`v1/structured_output/backend_types.py:49`,各 backend 實作:`backend_outlines.py:146`/`backend_guidance.py:186`/`backend_xgrammar.py:173`/`backend_lm_format_enforcer.py:60`)回傳「**輸入 token 的前綴中合法部分**」,非法自第一個非法 token 起全砍,再用 `-1` padding 補回原長度(`scheduler.py:2095-2102`)。

**傷害機制:**
- 多深度 MTP 一條 draft 鏈只要一個 token 不合 schema → 整條後續 draft 全丟(即使後續 draft 恰好合法)→ 該 step 接受數歸零化。
- 與 rejection sampling 統計正確性不一致:`rejection_sample`(`v1/sample/rejection_sampler.py:394`)比的是 `draft_probs`(**未 mask** 分佈) vs `target_probs`(**已 mask** 分佈)→ 機率比在有 schema 約束時失真 → recovered token 偏離「target 在 schema 下該有的分佈」。tool/JSON 場景(每位置可用 token 集劇變)放大此失真。

**tokenspeed 對照(實地推斷,工作區版 stripped 故部分為推斷):**
- `python/tokenspeed/runtime/execution/drafter/mtp.py:_sample_step_tokens` 優先取 `logits_output.next_token_ids`(`LogitsProcessor` 的 pre-sampled ids),否則 `sampling_argmax(next_token_logits)`;tokenspeed 有 `runtime/sampling/custom_logit_processor.py`(dill 序列化 callable 套 logits)→ **draft 取樣經過 logits_processor,可在 draft 竓套 mask**,每個 draft 一產出即合法。
- 加 `drafter/mtp.py` **per-depth KV slot repair + lookback stash + extend catch-up**:`_decode_shifted_ids` 處理 `src < accept` 讀本輪 verify 輸出、否則讀本輪 draft `d_m`、負 src(lookback)overlay;被拒/repaired draft 有 KV 槽修補,不必整鏈重算 → 多深度連續性更高。
- 兩者合起來 → 評價「MTP 接受率高於 vLLM」結構上可信。

**可移植修正(分階):**
- **短期(低改動,直接提升接受率)**:在 `llm_base_proposer._greedy_sample` / `_sample_draft_tokens` 把 `sampling_metadata.allowed_token_ids_mask` 也套到 draft logits(draft 竓 mask),取代僅靠事後 `validate_tokens` 前綴裁剪。draft 合法 → 不整鏈斷;rejection 兩端同 schema → 統計正確性對齊。
- **中期**:把 `validate_tokens` 的「前綴裁剪」改成「逐 token 驗證,只丟非法那幾個、保留後續合法」(需 grammar 支援 non-prefix 驗證;現 contract 為 prefix-only,要改 `backend_types.py:49` contract 與各 backend 實作)。
- **長期/可選**:移植 `drafter/mtp.py` per-depth KV repair + lookback stash(多深度連續性)。
- ⚠️ 限制:工作區 tokenspeed 為 SMG+gRPC stripped 版,`custom_logit_processor` 是否真套 grammar mask 未在工作區證實(只看到抽象基底 + dill 機制);接受率數字來自網路評論,工作區無 baseline。移植前先量 vLLM baseline(spec+structured 接受率),再驗證「draft 端 mask」的改善。

### 4B. toolcall / tool_parser 準確度(影響 Toolcall-15 100% pass)

**非架構缺口,是增量解析正確性問題。**
- vLLM 已有 46 個 tool_parser(含 `inkling`/`deepseekv4_engine`/`glm47_moe`/`kimi_k2`),覆蓋完整;`tool_parsers/streaming.py` + `abstract_tool_parser.py` + `utils.py`(`partial_json_loads`/`is_complete_json`/`partial_tag_overlap`)為公共增量基礎設施。
- 準確度風險集中於 **增量 stream 解析**(文本層後處理,非 logits 層):
  - `hermes_tool_parser.py` 用 `partial_tag_overlap`(跨 token 的 tag 重疊偵測)+ `is_complete_json` + `_compute_args_diff`(用 `len(self.streamed_args_for_tool[index])` 做字串 prefix-diff)+ `_extract_tool_name`(`re.search(r'"name"\s*:\s*"([^"]+)"')`)。
  - `streaming.py` 用 `partial_json_parser` + `_bracket_level_state`(逐字元跨 token 計算 `{}` 深度,處理 string/escape 跨 token)+ `filter_delta_text`(trim trailing tool-list delimiters)。
- **失準模式**:① tag 跨多 token 切片時 `partial_tag_overlap` 偵測邊界錯(hold back 長度失準);② JSON bracket / string / escape 跨 token 時 `_bracket_level` 計算錯;③ 多 tool call 並存時 `streamed_args_for_tool[index]` 的 index 對齊錯;④ spec decode 一次吐多 token 時 `delta_text` 非單 token,增量 diff 的「逐 delta」假設破;⑤ tool name 用正則從「partial JSON」抓,partial 時可能抓到截斷名。
- 根因:這些都是 **「token→text decode 後才做字串邊界偵測」** 的設計;vLLM tool_parser 是**文本層後處理**,不是 **logits/詞表層約束**(grammar mask 才是 logits 層)。

**tokenspeed 對照(工作區版無 tool_parser 源碼):**
- 工作區 tokenspeed **無 tool_parser / ToolCall 實作**(`grep -rln "tool_parser\|ToolCall\|tool_call" python/tokenspeed` 全空);架構是 SMG gateway + gRPC engine(`cli/serve_smg.py` spawn smg gateway + gRPC engine),工具呼叫邏輯在 **SMG / 閉源前端**,不在 Python 套件。
- 評價的「Toolcall-15 100% pass」很可能來自 **SMG 前端的工具呼叫解析**(推測:logits 層 grammar 約束 + token-id 層狀態機 / 非增量文本後處理,或更穩的 partial JSON 策略),而非 vLLM 風格的文本層後處理。
- **無源碼可考** → 本項無法精確移植,只能給「方向」與「vLLM 自家失敗模式調查」。

**可調查 / 可做的(不必抄 tokenspeed):**
1. 先拿 vLLM 對應 parser(依 benchmark 模型選:Inkling → `inkling_tool_parser.py`;DeepSeek-V4 → `deepseekv4_engine_tool_parser.py`;GLM → `glm47_moe_tool_parser.py`)跑 Toolcall-15,**分類失敗模式**:tag 跨切片?JSON bracket 跨切片?多 call index 錯?spec decode 多 token delta?partial tool name?
2. 修正方向(依失敗模式):
   - **首選(治本)**:對 tool call 用 **grammar/structured output 在 logits 層強制 JSON schema** —— 把 tool 的 JSON schema 編成 grammar,走 `v1/structured_output` 的 bitmask 在 logits 端強制;token 一產出即合法,文本層只剩「取出欄位」,邊界問題消失。vLLM 已有此基礎設施(`apply_grammar_bitmask` + 各 backend),只是 **tool path 未預設接上 structured_output**。落點:tool request 帶 schema 時,自動 compile 成 grammar 並設 `structured_output_config`,複用 §4A 的 draft-mask 修正(同一條 logits 層約束)。
   - **次選(治標,給無 grammar 的模型)**:對 stream 增量改用 **token-id 層的 tool call 狀態機**(而非 text 層 bracket 計數);或對 spec decode 的多 token delta 改「先累積完整 tag/JSON 再解析」而非逐 delta diff。
   - **配合 §4A**:若開 spec decode + tool,務必把 draft 端也套 grammar mask(§4A 短期修正),否則 draft 產生非法 tool token → 事後前綴裁剪 → 接受率崩 + 增量解析可能在「被丟棄的 draft token」上誤觸發 partial 解析。
- **若無法取得 SMG toolcall 源碼,本項以「方向 + vLLM 自家失敗模式調查」收斂,不改架構**(vLLM 已有 46 parser,不是 arch 缺口)。

---

## 5. 落地步驟(決策完整,可直接交工程師)

### Step A — 在 vLLM 引入 kernel 選擇層(#1)
- 新增 `vllm/v1/kernel/{registry.py,selection.py,platform.py,signature.py}`(直接仿 `tokenspeed_kernel` 對應檔;MIT 授權相容,仿寫介面無衝突,直接 copy 程式碼須保留 notice)。
- 改 `vllm/v1/attention/selector.py`:保留 `AttentionSelectorConfig` 為「請求描述」,改為向 `select_kernel(family="attention", signature=..., capability=...)` 查詢;舊硬編碼分支降級為內建註冊項。
- 第一波只遷 **attention** 一個 family(風險最小),e2e 確認後再遷 gemm/moe/sampling。
- 保證每 family 至少一個 Triton 後端(`SelectionObjective.PORTABILITY`),先解 sm120(參考 `docs/design/tokenspeed-sm120-support.md` §4 platform.py bug + §5 Step A–D)。
- 驗收:Qwen3.5/Llama 在 sm80/sm90 行為不變;sm120 能跑(目前不能);AMD 走 `tokenspeed-kernel-amd`(已採用)。

### Step B — SSM 狀態 prefix cache(#2)+ HiCache 分層(#3)
- 在 `v1/core/single_type_kv_cache_manager.py` 為 `MambaSpec` 加 `MambaCheckpointPool`(仿 sglang `mamba_checkpoint_pool.py` + TurboMind `StateManager`):runtime slots vs checkpoint slots 分池,LRU over ready unpinned,`prefix_cache_decode_state_interval` 控 decode 存粒度。
- restore 兩階段:scheduler pin → worker copy `checkpoint→runtime` → 排程釋放 pin(仿 TurboMind README SSM 段 + ancestor chain verify)。
- 在 `v1/kv_offload/tiering/` 加 `PoolName`/`PoolHitPolicy`/`PrefetchTimeoutConfig`(仿 sglang `hicache_storage.py`):SSM/SWA 用 `TRAILING_PAGES`,DSA 用 `ALL_PAGES`。
- 驗收:GLM-4.7 / DeepSeek-V4 hybrid 多輪 prefix 命中率 + TTTC(須先建 baseline)。

### Step C — 多深度 MTP + draft 端 grammar mask(#4,含 §4A 短期修正)
- 短期先做 §4A「draft 端套 mask」:`llm_base_proposer._greedy_sample/_sample_draft_tokens` 套 `allowed_token_ids_mask` 到 draft logits;驗收 = spec+structured 接受率 baseline 對比。
- 中期:`v1/spec_decode/` 加 `multi_depth_mtp.py`(仿 `drafter/mtp.py`):`_decode_shifted_ids` 視窗座標 + per-depth KV slot + lookback stash + extend catch-up;註冊 `--speculative-algorithm multi-depth-mtp`。
- 驗收:先量 vLLM baseline MTP 接受率,再對照 tokenspeed(評價數字來自網路,工作區無實測環境)。

### Step D — VLM partial-sibling trie(#5)+ TBO MoE(#6)+ jump-forward(#7)
- #5:`block_pool`/`kv_cache_utils` 在 image 起點 block fold `mm_hash` 進 cumulative `BlockHash` + partial-block sibling 結構(仿 TurboMind `prefix_trie.h`)。
- #6:`engine/core.py` async_scheduling + worker 把 MoE all2all dispatch 進第二 batch CPU 視窗(仿 sglang `two_batch_overlap.py`,token-distribution threshold 自適應開關)。
- #7:`v1/structured_output` 加 jump-forward(仿 sglang `outlines_jump_forward.py`)。

### Step E — Toolcall 準確度(§4B,先調查再決定)
1. 取 vLLM `inkling_tool_parser.py` / `deepseekv4_engine_tool_parser.py` 跑 Toolcall-15,分類失敗模式(tag 跨切片 / JSON bracket 跨切片 / 多 call index / spec 多 token delta / partial tool name)。
2. 首選修正:tool request 帶 JSON schema 時自動 compile 成 grammar,走 `v1/structured_output` logits bitmask;複用 §4A draft-mask(同一 logits 層約束)。
3. 次選(無 grammar 模型):token-id 層 tool 狀態機 / spec 多 token 先累積再解析。
4. **若無法取得 tokenspeed/SMG toolcall 源碼,本項擱置架構變更,僅做 vLLM 自家 parser 失敗模式調查 + 治標修正**(vLLM 已有 46 parser,不是 arch 問題)。

### Step F — 長期:C++ FSM 排程(#8)
- 短期不改語言;先在 `v1/core/sched` 抽 cache/forward FSM 介面(仿 tokenspeed `fsm/forward_states`+`cache_states` 的狀態/事件分離),把 Python 排程的隱性狀態機顯式化。
- 長期再評估是否移植 tokenspeed-scheduler C++ 核心(與 `scheduling-improvements.md` roadmap 合併)。

---

## 5.5 已實測驗證 / 已被推翻的部分(2026-07-25 補)

本文件寫成後,以下項目已在 `unieinfra-vllm/`(我們 vendored 的同一棵樹)上實地驗證,結論記於
`docs/experiments-log.md`:

- **§4A 三條程式碼宣稱:全部確認。** draft 端只經 `vocab_mapping.constrain_draft_logits`(詞表對齊,
  非 schema);`apply_grammar_bitmask` 全檔僅出現一次於 `gpu_model_runner.py:4568`(target 端);
  `validate_tokens` 位於 `backend_types.py:49`,前綴裁剪語意。→ **Step C 的短期修正(draft 端套 mask)
  是有根據的,已列為獨立工作項。**
- **§1 原則一在上游有反例,且會當機。** vLLM 的 `ll_bf16` CuteDSL router GEMM 以
  `has_device_capability(90)` 開放式 gate,收下 sm_120 後在 warmup 直接殺掉 EngineCore(E37)——
  正是本文件所述「不散落硬體判斷」要防的失敗模式,**在 vLLM 內部真實存在**。
- **MTP 的前提被修正(E39)。** 先前假設「網路上的 MTP 增益我們看不到,是因為量錯場景」——**錯**。
  實測 MTP 在 SpecBench/GSM8K 給 **+92%~117%(C=1)**,而在我們自 E1 起使用的 RandomDataset 對照組
  上同樣給 **+101%**。真正原因是 **E1–E38 從未啟用 MTP**。調節變數是**併發度而非語料**
  (增益 92→65→39 隨併發衰減)。
- **acceptance 不可再當作加速代理指標(E39)。** 隨機語料的 accepted length **最高**(3.39–3.65/4)、
  SpecBench 最低,但吞吐增益最好的是 GSM8K,其 acceptance 低於隨機語料。§4A 與 Step C 的驗收應以
  **吞吐與正確性**為準,不以 acceptance 為準。
- **結構化輸出成本已量(E39c)**:grammar mask 在 C=1 為 +0.0% TPOT、C=16 為 +2.7%,合法率 0%→100%。
  → Step E「把 tool schema 編成 grammar」的成本side已有實數,不再是未知。

- **Step A 已落地(E43)。** `unieinfra-vllm/vllm/v1/attention/unieinfra_selection.py` 把
  `platforms/cuda.py::_get_backend_priorities()` 的逐架構分支改寫成 22 條能力註冊項,
  `AttentionSelectorConfig` 保留為請求描述不變。**在 sm_80/86/89/90/100/120 六個模擬架構上
  150/150 個 shape 與上游決策相同**,吞吐為平手(C=1 -0.01%、C=32 +0.08%,皆在雜訊底線內)。
  三點與本文件的假設不同,值得記下:
  (a) **反模式的位置寫錯了** —— §2 表格指向 `v1/attention/selector.py`,但該檔不做決策,
  硬編碼樹在 `vllm/platforms/cuda.py`,而且每個 platform(cuda/rocm/xpu/cpu)各有一份。
  (b) **registry 每個 spec 只有一個靜態 priority,但上游的「順序」隨架構變**。所以
  「sm_100 偏好 FlashInfer」必須寫成**同一 backend 的第二條註冊**(收窄的 capability + trait),
  而不是一條註冊加分支。可行,也更清楚,但排名清單裡同一 backend 會出現兩次。
  (c) **`validate_configuration()` 不是架構分支,不該被取代** —— 它是每個 backend 用約 14 個
  `supports_*` 宣告自己的支援面,加上自由形式的 `supports_combination()`。這些已**由 class 機械式
  推導成 registry traits**(不會漂移),但該方法仍是最終權威:選擇層走排名清單,回傳第一個
  自己接受該請求的 backend。150 shape 掃描中需要往下走 22 次,殘量正是 `block_size` 的
  `MultipleOf` 語意與 `supports_combination`。另外 `FlashInferMLABackend.supports_combination`
  會呼叫 `get_current_vllm_config()` —— **capability 查詢不純,依賴引擎全域狀態**。

仍未量測、§6 的警告依然成立:**真正的多輪 prefix 重用**在 E40b 已補測(單輪取樣器的限制已解除);
本節其餘未量項目見 `docs/experiments-log.md`。

### 5.5.1 §4A「短期修正」的實作結論(E42,實作後修正本文件)

Step C 的短期修正已**實作並實測**(`vllm/v1/spec_decode/draft_grammar.py`,
`UNIEINFRA_DRAFT_GRAMMAR_MASK=1`)。實作過程推翻了 §4A 對「怎麼做」的三個假設,列在這裡以免
後人重蹈:

1. **`sampling_metadata.allowed_token_ids_mask` 不是 grammar。** §4A 寫「把
   `allowed_token_ids_mask` 套到 draft logits」——該欄位其實是 `SamplingParams.allowed_token_ids`
   的**靜態 per-request token 白名單**,由 `rejection_sampler.apply_logits_processors` 施加,與
   structured output **完全是兩條路**。grammar 走的是 `GrammarOutput.grammar_bitmask` →
   `apply_grammar_bitmask`。兩者不可互換。
2. **既有 bitmask 無法直接餵給 drafter,因為沒有一列是正確的狀態。** scheduler 送出的 bitmask 每個
   結構化請求有 `1 + num_spec_tokens` 列,填在狀態 `S_t`(上一步結束後)並沿**上一步的 draft**
   往前捲。drafter 需要的是 `S_{t+1} = S_t + 本步實際吐出的 token`,再每個 draft 位置各推進一次。
   **shipped bitmask 沒有任何一列等於這個狀態**,所以 draft 端的 mask 需要的是 **FSM 本身**,
   不是 bitmask。
3. **FSM 只有在同進程 executor(`UniProcExecutor`,TP=1)下才拿得到。** grammar 掛在 engine-core
   端的 `Request.structured_output_request` 上;多進程 executor 下 worker 根本看不到它。本實作因此
   在 engine core 啟動時把 scheduler 參考交給 model runner 的 masker,並在多進程情形**明確關閉並
   警告**,而不是靜默不作用。

另外兩個實作陷阱(都會真的壞掉,不是理論風險):

- **rollback ledger 少記一列 = 請求直接 500。** masker 必須推進 FSM(先吐出的 token、再每個 draft)
  再**全部 rollback**。第一版把「已終止 / 拒絕 draft」的列從追蹤清單移除卻沒回滾,結果
  `scheduler.update_from_output` 立刻噴
  `grammar rejected tokens [...] Terminating request` 並終止該請求。FSM 失步不會安靜降級。
- **`max_rollback_tokens` 只夠 k,不夠 2k。** `XgrammarBackend` 以
  `max_rollback_tokens=num_speculative_tokens` 建 matcher(夠 scheduler 自己 advance+rollback)。
  draft 端 mask 需要 `吐出(≤k+1) + draft(≤k−1) = 2k`,必須加寬。

**結論**:§4A 對*機制*的描述是對的(E40 也證實 `validate_tokens` 真的砍掉約 40% 的 draft 鏈),
但對*後果*的推論不成立——被砍掉的 draft 正是 target 本來就會拒絕的那些(E40),所以補上 draft 端
mask 的預期收益接近零。詳見 `docs/experiments-log.md` 的 E40 與 E42。

---

## 6. 假設與限制
- vLLM 版本 = 工作區 `third_party/vllm`(2026-07-24, v1 引擎,含 turboquant/V4/async_scheduling/EPLB)。
- tokenspeed 工作區版 = SMG+gRPC 架構,**無 tool_parser 源碼**;MTP `drafter/mtp.py` 在但無實測接受率環境;`custom_logit_processor` 是否套 grammar mask 未證實。
- 評價數字(ctx_pp/pp2048/tg128 vs vLLM、Toolcall-15 100%、GSM8k 0.97、MTP 接受率高)**來自網路評論,工作區無重現**;移植前須自建 baseline。
- TileRT 閉源 B200-only,**排除移植**,僅作「tile-level 重疊 runtime」概念參考。
- 授權:tokenspeed-kernel MIT;vLLM/lmdeploy/sglang Apache-2.0 —— 仿寫介面無授權衝突,直接 copy 程式碼須保留 notice。
- 本文件為設計參考;實際落地依 §5 步驟,每步先建 baseline 再驗收。
