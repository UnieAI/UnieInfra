# 15 — encoder-only（embedding）與 encoder-decoder（Whisper）：四方橫向對照

範圍：**沒有 KV 成長的那一段 forward**。兩種形狀：

- **encoder-only / embedding**：單發 prefill，輸出 pooled 向量，沒有 decode 步。
- **encoder-decoder（Whisper）**：encoder 單發算出固定長度輸出，decoder 每一步
  對它做 cross-attention；cross KV 寫一次、讀 N 次。

對照對象：vLLM、SGLang、TensorRT-LLM、TurboMind（LMDeploy）。TurboMind 這一輪幾乎整欄
是 `—`，**這本身就是發現**，見 §5。

## 0　版本與讀了哪些模組

pin 沿用 `00-METHOD.md`。三個參照 repo 的 HEAD 與 pin 一致（已核對）：
`sglang b61a0692`、`TensorRT-LLM d8b67db2`、`lmdeploy a1e5674`。

**vLLM 例外**：工作樹 HEAD 是 `b244b2bb96`，比 pin `4ad14044b254` 多**一個** commit
（`mamba: stop charging retained align snapshots as private working set`），動到
`config/cache.py`、`engine/arg_utils.py`、`layers/mamba/abstract.py`、
`v1/core/single_type_kv_cache_manager.py`、`v1/kv_cache_interface.py`。
本文所有行號以**工作樹（HEAD）**為準，這樣 `check_xref.py` 才驗得過（68/68 通過）。
被那個 commit 觸及的兩個檔案，pin 上的對應行號另記於此，讀 pin 時用這組：

| 符號 | HEAD | pin `4ad14044b254` |
|---|---|---|
| `CrossAttentionSpec` | `v1/kv_cache_interface.py:725` | `:717` |
| `CrossAttentionManager` | `v1/core/single_type_kv_cache_manager.py:1381` | `:1369` |
| `raise NotImplementedError(...)` | `:1441` | `:1429` |

（以 `git show 4ad14044b254:<file>` 逐一核對過。）其餘 vLLM 檔案未被該 commit 觸及，
pin 與 HEAD 行號相同。

讀的是實作該 feature 的模組本身，不是表親：Whisper 讀各家的 whisper 模型檔＋
cross-attention 的 cache manager，embedding 讀 pooler ＋ encoder-only 的 attention 分支，
不拿一般 prefill 頂替。

## 1　步驟表 A — encoder-only / embedding（單發、pooled 輸出）

| 步驟 | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|
| HTTP 路由 | `entrypoints/pooling/embed/api_router.py:35 create_embedding()` | `srt/entrypoints/http_server.py:1713 "/v1/embeddings"` | `tensorrt_llm/serve/openai_server.py:280 self.embedding_batcher` | `lmdeploy/serve/openai/endpoints/auxiliary.py:24` → **400 Unsupported by turbomind** |
| serving handler | `entrypoints/pooling/embed/serving.py:42 ServingEmbedding` | `srt/entrypoints/openai/serving_embedding.py:30 OpenAIServingEmbedding` | `tensorrt_llm/serve/encode_batcher.py:46 EncodeBatcher` | — |
| 請求物件 | `PoolingParams`（`v1/request.py:85`） | `EmbeddingReqInput`（`srt/managers/io_struct.py`） | `_QueuedRequest`（`serve/encode_batcher.py:41`）→ 一般 `LlmRequest` | — |
| engine 入口 | `vllm/v1/engine/async_llm.py:803 encode()`（與 `generate()` 分開） | 同一條 `Scheduler` 迴圈，`run_batch` 內分岔 | 同一條 `PyExecutor` 迴圈 | — |
| 「這是 encoder」的判定 | `vllm/config/model.py:1779 attn_type` → `"encoder_only"` | `ServerArgs.is_embedding`（`srt/server_args.py:580`） | `model_config.is_generation == False` | — |
| 非因果 attention | `MMEncoderAttention`（`model_executor/models/whisper.py:109 WhisperEncoderAttention` 同一個層家族） | `srt/models/bert.py` 走 `flash_attn_varlen_func` raw K/V；whisper encoder 走 `torch.nn.functional.scaled_dot_product_attention`（`srt/models/whisper.py:123 scaled_dot_product_attention`） | **同一個 `Attention` 類別**，改傳 `PredefinedAttentionMask.FULL`（`_torch/models/modeling_bert.py:135 attention_mask`） | — |
| KV cache 配置 | 空 `kv_cache_spec` → **0 個 group**（`v1/core/kv_cache_utils.py:1114 is_kv_cache_type_attention_free`、`:1724`） | **照常配滿整個 paged pool**（`srt/model_executor/model_runner.py:808 alloc_memory_pool`，無 `is_embedding` 分支） | **完全不建 KV manager**（`_torch/pyexecutor/py_executor_creator.py:906`、`_torch/pyexecutor/_util.py:1271 assert`） | — |
| prefix cache | `is_prefix_caching_supported` → False（`config/model.py:1855`） | `disable_radix_cache = True`（`srt/server_args.py:3728`） | 無 manager 即無 | — |
| chunked prefill | `is_chunked_prefill_supported` → False（`config/model.py:1805`） | `chunked_prefill_size = -1`（`srt/server_args.py:3729`） | context-only，單發 | — |
| pooling 在哪 | `model_executor/layers/pooler/`（`model_executor/layers/pooler/abstract.py:16 Pooler`），runner 呼叫 `v1/worker/gpu_model_runner.py:3387 model.pooler()` | `srt/layers/pooler.py:168 Pooler` / `:47 pool_hidden_states`，模型 forward 內 | 模型內 `_torch/models/modeling_bert.py:152 BertPooler` | — |
| 「不要 decode」怎麼表達 | `Request.max_tokens = 1`（`v1/request.py:105`），有 pooler 輸出就 `FINISHED_STOPPED`（`v1/core/sched/scheduler.py:1638`） | `run_batch` 的 `else` 分支回 `EmbeddingBatchResult`（`srt/managers/scheduler.py:3761`），不進 sampler | **換掉 sampler**：`EarlyStopSampler`（`_torch/pyexecutor/sampler/sampler.py:243`） | — |
| 專屬批次策略 | 無（靠一般 continuous batching） | `enable_tokenizer_batch_encode = True`（`srt/server_args.py:3731`） | **有**：hold window `0.005s` + `max_queue_size 2048` 的 dynamic batcher（`serve/openai_server.py:271-272`） | — |

## 2　步驟表 B — encoder-decoder（Whisper）

| 步驟 | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|
| 模型檔 | `model_executor/models/whisper.py`（另有 `whisper_causal.py` 變體） | `srt/models/whisper.py` | `_torch/models/modeling_whisper.py` | `lmdeploy/pytorch/models/whisper.py` — **只有 encoder**（`:13 WhisperAttention`、`:79 WhisperEncoderLayer`），唯一使用者是 `pytorch/models/interns1_pro_time_series.py:13 WhisperEncoderLayer` |
| encoder 由誰排程 | **走多模態 encoder 路徑**：`v1/core/sched/scheduler.py:225 EncoderDecoderCacheManager`；註解明說「Text-only encoder-decoder models are implemented as multi-modal models for convenience」（`:212`） | 同樣借多模態：`encoder_lens` / `encoder_cached` 由 `srt/managers/schedule_batch.py:2198 prepare_encoder_info_extend()` 準備 | **encoder 與 decoder 各跑一個 iteration**（`_torch/pyexecutor/py_executor.py:780-789`） | — |
| encoder 是否可切塊 | 否，`is_chunked_prefill_supported` False（`config/model.py:1836`） | 否，「the encoder part should be considered as a whole」（`srt/managers/schedule_batch.py:2233 encoder_len`） | 否 | — |
| encoder self-attn 用不用 KV | 不用（`model_executor/models/whisper.py:192 AttentionType`） | 不用，直接 SDPA（`srt/models/whisper.py:123 scaled_dot_product_attention`） | 不用 | 不用（`lmdeploy/pytorch/models/whisper.py:70` SDPA） |
| **cross KV 放哪** | **自己的 spec/group**：`CrossAttentionSpec`（`v1/kv_cache_interface.py:717`，pin 行號）＋ `CrossAttentionManager`（`v1/core/single_type_kv_cache_manager.py:1369`，pin 行號），註冊於 `:1528` | **併進同一個 KV pool**，靠 layer_id 位移當成額外的層：`decoder_cross_attn_layer_id = encoder_layers + decoder_layers + layer_id`（`srt/models/whisper.py:206-208`） | **獨立的 cross pool**：`ResourceManagerType.CROSS_KV_CACHE_MANAGER`，建置時把預算拆成 self / cross（`_torch/pyexecutor/_util.py:1865-1873 _split_kv_cache_budget_for_cross`） | — |
| cross KV 配置時機 | 一次靜態配置，量由 encoder token 數決定（`v1/core/kv_cache_coordinator.py:166-175`；scheduler 端 `v1/core/sched/scheduler.py:887 num_encoder_tokens`） | 與 decoder 同一段 `out_cache_loc`，prefill 時一起配 | 第一次 context 後 `skip_cross_kv_projection = True`（`_torch/pyexecutor/model_engine.py:3379`） | — |
| cross KV 前綴重用 | **明確禁止**：`find_longest_cache_hit` 直接 `raise NotImplementedError`（`v1/core/single_type_kv_cache_manager.py:1441 raise`），`get_num_common_prefix_blocks` 回 0 | 靠 `encoder_cached` 旗標避免重算，不做跨請求共用 | 有 `encoder_num_cached_tokens_per_seq`，但同樣是 per-request | — |
| 記憶體上限怎麼算 | `CrossAttentionSpec.max_memory_usage_bytes` 用 `max_num_encoder_input_tokens`（Whisper=1500）（`v1/kv_cache_interface.py:733 max_encoder_len`） | 併在 `max_total_num_tokens` 內 | cross 預算獨立切一份（同上） | — |
| decode 期間的 metadata | 一般 attn metadata，backend 依 spec 分辨（`v1/attention/backends/cpu_attn.py:161 is_cross_attention`） | 一般 RadixAttention | `attn_metadata.cross` 子物件＋穩定步快取（`_torch/pyexecutor/model_engine.py:3395 cross_attn_metadata`） | — |
| 已知限制 | 與 KV connector 互斥（`v1/core/sched/scheduler.py:133 assert`） | — | `pp_size > 1`、piecewise CUDA graph、spec decode + CUDA graph 皆 NotImplemented（`_torch/pyexecutor/py_executor.py:789 is_encoder_decoder`） | — |
| ASR 的輸出契約 | — | `/v1/audio/transcriptions`：`srt/entrypoints/openai/serving_transcription.py`、`streaming_asr.py` | — | — |

## 3　分歧點

### 3.1　非因果 encoder pass 住在哪（Q1）

三種決策，沒有兩家相同：

- **TRT-LLM = 同一條路徑加旗標**。`BertAttention` 直接繼承 `Attention`，只在呼叫時傳
  `PredefinedAttentionMask.FULL`。代價最小，但也代表 encoder 仍走完整 attention 前置。
- **SGLang = 模型內另開分支**。whisper encoder 直接呼叫 `scaled_dot_product_attention`，
  bert 走 FA varlen；attention backend 另有 `fa_skip_kv_cache` 旗標整條繞開分頁快取。
- **vLLM = 型別化**。`attn_type` 是一個 derived property（`config/model.py:1779`），
  `encoder_only` / `encoder_decoder` / `attention_free` 各自是一等公民，
  chunked prefill 與 prefix caching 兩個能力由它**推導**，不是各處 if。

後果：vLLM 的能力矩陣集中在一個 property，加新形狀時只改一處；SGLang 的判斷散在
server_args、backend、模型三層；TRT-LLM 最省事但沒有「這個模型不能切塊」這種可查詢的宣告。

### 3.2　沒有 KV 時 cache manager 怎麼辦（Q2）

**三家三種答案，差距很大**：

| | 做法 | 後果 |
|---|---|---|
| TRT-LLM | `is_generation == False` → **根本不建 KV manager**（`_torch/pyexecutor/py_executor_creator.py:906 is_generation`），並用 `assert` 把「只有生成模型才建 KV」寫死（`_torch/pyexecutor/_util.py:1271 assert`） | 最乾淨；embedding server 不佔任何 KV 顯存 |
| vLLM | 空 spec → **0 個 group**，`get_kv_cache_groups` 回空 list（`v1/core/kv_cache_utils.py:1724 is_kv_cache_type_attention_free`），coordinator 換成 `KVCacheCoordinatorNoPrefixCache`，其 docstring 明寫「supports arbitrary numbers of KV cache groups (**including 0 groups**)」（`v1/core/kv_cache_coordinator.py:378 KVCacheCoordinatorNoPrefixCache`） | 零 group 是**一等公民**，不是特例補丁 |
| SGLang | **照樣配滿整個 paged pool**：`alloc_memory_pool` 無條件執行，`mem_cache/` 與 `schedule_batch.py` 內找不到任何 `is_embedding` 分支；快路徑只是「不寫也不讀」 | 顯存被整池吃掉卻永遠不用；每個 token 仍配 slot |

這一格是本輪最值得量的分歧（見 §6）。

### 3.3　cross KV 放哪（Q3）

- **vLLM 與 TRT-LLM 同一邊：cross KV 自成一池**。vLLM 給它專屬 spec + manager + group，
  TRT-LLM 給它專屬 resource manager 並在建置期就把預算切開。兩家都明確拒絕讓 cross KV
  進入前綴重用（vLLM 直接 `raise`）。
- **SGLang 在另一邊：折進一般 KV 路徑**，用 layer_id 位移把 cross 當成「第
  `encoder_layers + decoder_layers + i` 層」塞進同一個 pool。

後果：SGLang 省掉一整套 manager，但 cross KV 與 decoder KV 共用同一組驅逐與計量，
無法單獨為「寫一次讀 N 次」的資料訂不同策略；vLLM/TRT-LLM 多一套結構，換到獨立預算與
「這塊不可共用」的靜態保證。

### 3.4　pooling 與輸出契約（Q4）

pooling 的**位置**三家不同：vLLM 抽成獨立層家族（`layers/pooler/`，seqwise/tokwise 分開）
並由 runner 呼叫；SGLang 放在 `layers/pooler.py` 由模型自己呼叫；TRT-LLM 直接寫死在模型檔內
（`BertPooler`，只做 CLS）。

「不要 token streaming」的**表達方式**也三家不同，見步驟表 A 最後兩列：
vLLM 用 `max_tokens = 1` 讓 pooling 請求偽裝成一步生成；SGLang 在 `run_batch` 分岔出
`EmbeddingBatchResult`；TRT-LLM 換掉 sampler（`EarlyStopSampler`）。
**vLLM 的做法最省結構**——排程器完全不需要知道 pooling 存在。

### 3.5　批次策略（Q5）

只有 **TRT-LLM 為 embedding 流量另做一套批次**：HTTP 層的 `EncodeBatcher`，5ms hold window、
token 預算、佇列滿回 429，模組刻意不依賴 torch 以便 CPU 單測（`serve/encode_batcher.py:14-21`）。
其餘三家靠既有的 continuous batching；SGLang 只在 tokenizer 層加了
`enable_tokenizer_batch_encode`，讓一個 list 型 embeddings 請求原子送出。

值得注意的是動機：`EncodeBatcher` 的 docstring 明說是在模仿 Triton 的 dynamic batcher。
embedding 流量是大量短而獨立的請求，**排隊聚合的收益來自 hold window，不是來自排程器**。

## 4　結構差異（同功能、反結構）

1. **「encoder-decoder 是多模態的一種」**——vLLM 與 SGLang 都把純文字 enc-dec 掛在多模態
   機制上（vLLM 的註解直說是 for convenience；SGLang 的 encoder 長度欄位叫
   `im.num_image_tokens`，Whisper 的音訊也走這個欄位）。TRT-LLM 反過來，enc-dec 是
   executor 迴圈層級的一等狀態（encoder 與 decoder 各一個 iteration）。
2. **能力宣告 vs 現場判斷**——vLLM 把「能不能 chunked prefill / prefix cache」做成可查詢的
   derived property；SGLang 是在 `server_args` 的 post-process 裡**改寫使用者的旗標**
   （`is_embedding` 一旦成立就強制關掉 radix cache 與 chunked prefill）。後者在 log 之外沒有
   任何地方可以查到「為什麼這個 server 沒有前綴快取」。
3. **pooling 的歸屬**——vLLM 放在 runner（模型只提供 `pooler`），TRT-LLM 放在模型內部。
   前者換 pooling 策略不動模型，後者反之。

## 5　TurboMind 整欄是 `—`

- `/v1/embeddings` 是**已註冊的 stub**，回 400 `Unsupported by turbomind`
  （`lmdeploy/serve/openai/endpoints/auxiliary.py:24-29`，tag 就叫 `unsupported`）。
- `lmdeploy/pytorch/models/whisper.py` **只有 encoder**（attention + encoder layer），
  沒有 decoder、沒有 cross-attention、沒有 cache；唯一使用者是把它當音訊前端的
  `interns1_pro_time_series.py`。
- `src/turbomind/models/` 下只有 `llama` 與 `qwen3_5vit`，沒有任何 enc-dec 結構。

所以「四方對照」在這兩個 feature 上實際是三方。**要照抄一份 enc-dec 設計，TurboMind 不是選項。**

## 6　可量測候選

1. **SGLang 的 embedding KV 池是不是純浪費**（§3.2）。同一個 embedding 模型，量
   `max_total_num_tokens` 對應的顯存與實際 fa_skip 路徑的使用量。若確認整池閒置，
   我們就不要抄這個結構——直接照 TRT-LLM/vLLM 的零池。
2. **hold-window 批次對 embedding goodput 的效果**（§3.5）。TRT-LLM 的 5ms 是個常數，
   來源是 Triton 慣例而非測量。這正是 `derived-constant-vs-its-derivation` 那類：
   要抄就得自己量一次曲線（1ms / 5ms / 20ms × 併發），不要照搬數字。
3. **cross KV 獨立池 vs 折進主池**（§3.3）。同一個 Whisper 形狀下，量兩種配置的
   併發上限與 TTFT。這決定我們要不要為 cross 開第三個 `family`。

## 6.1　sm_120 上的 cross-attention KV：實測到的限制

為 Whisper 的 cross-attention 找「decode 時不複製 encoder KV」的路徑時，
`tokenspeed_kernel.ops.attention.flash_attn` 的三條候選在這張卡上實測（bf16、H=4、
D=64、ENC=128、BS=3）：

| 路徑 | 結果 |
|---|---|
| `cu_seqlens_q` + `cu_seqlens_k` 分開（cross-attn 的基本形狀） | **可用**，是正確性基準 |
| `page_table`（配 `cu_seqlens_k`） | `AssertionError: page_table is not supported with cu_seqlens_k` |
| `page_table` + `seqused_k`（分頁 decode 的標準 API） | `AssertionError: **Paged KV not supported on SM 12.0 in this PR**` |
| `gather_kv_indices` | `AssertionError: gather_kv_indices is only supported with qv` |
| `flash_attn_with_kvcache` | 在此容器是未註冊的 placeholder，呼叫即拋 |

也就是說 **cross-attention 的分頁／索引讀取在 sm_120 上沒有可用的 kernel 路徑**，
decode 時只能把該請求的 encoder KV 複製成連續 buffer。這不是設計選擇，是 kernel 現況；
換卡或換 kernel 版本就該重測，不要把「必須複製」寫成永久前提。
（同一形狀的教訓見 memory 的 `sm120-closed-allowlist-tables`。）

## 7　對 UnieInfra 的落點（含對交接前提的一處修正）

**修正**：交接說明寫「`python/tokenspeed/runtime/models/` 沒有任何 whisper/bert/embed/encoder
相關模型」。以檔名 grep 是對的，但按能力看不是——我們**已經在服務 ASR**：

- `python/tokenspeed/runtime/models/qwen3_asr.py:49 Qwen3ASRForConditionalGeneration`
  （另有 `qwen3_audio.py`、`qwen3_omni.py`、`qwen3_vision.py`、`moonvit.py`）。
- 但它是 **decoder-only + audio prefix** 架構：audio tower → `MultimodalEmbedder`
  （`runtime/multimodal/embedder.py:281`）把音訊特徵接進 LLM 的 input embedding，之後就是
  一般因果 decode（`runtime/models/qwen3_asr.py:180 forward` 末尾直接呼叫 `self.language_model(...)`）。
  **沒有 cross-attention。**
- 非因果 encoder attention **已經有了**：`runtime/layers/attention/mm_encoder_attention.py`
  （566 行，triton / fa3 / fa4 / cuDNN 四個 backend，全部 `causal=False`、不碰 KV），
  等價於 vLLM 的 `MMEncoderAttention`。

所以三個問題的實際落點是：

- **Q1（非因果 encoder pass）：我們這邊已解決**。embedding 模型的 forward 可以直接復用
  `mm_encoder_attention.py`，不需要新的 attention 工作。
- **Q2（沒有 group 可綁）**：arena 的綁定在 `tokenspeed-scheduler/csrc/cache/core/block_pool.h:179`
  ——`parent.bound_group = group_id`，`bound_group` 是 `std::optional`，空池時 reset（`:136`）。
  也就是說「沒有 group」在資料結構上**可表達**，但唯一入口 `AcquireBlocks(group_id, ...)`
  （`:62`）一定要一個 group_id。結論與 vLLM 的零 group 一致：
  **embedding 請求根本不該進 cache coordinator**，而不是「進去但綁一個假 group」。
  這條路不需要動共用的 `.so`。
- **Q3（cross KV）**：`PagedCacheGroupSpec` 已經有 `family: "history" | "state"` 這個維度
  （`runtime/layers/attention/kv_cache/recipes/spec.py:36`）。cross KV 是第三種 family：
  寫一次、不成長、不參與前綴比對。vLLM 與 TRT-LLM 都投給「獨立池」，SGLang 投給「折進主池」；
  我們的 arena 是 LCM 共用 parent，折進主池會讓 cross 與 history 搶同一批 parent，
  傾向跟 vLLM/TRT-LLM 同邊——但這一步會動到 C++ 側，屬於本箱共用資產，先量 §6.3 再做。
- **Q4（pooling 與輸出契約）**：我們的 API 層目前沒有 `/v1/embeddings`、沒有
  `/v1/audio/transcriptions`（`runtime/entrypoints/control_server.py` 的路由清單），
  runtime 內也沒有任何 pooler（唯一的 "pooling" 命中是 `runtime/configs/qwen3_config.py:72` 講 GQA
  mean-pooling 的 docstring）。抄 vLLM 的結構最省：pooler 獨立成層、runner 呼叫、
  請求用 `max_tokens = 1` 表達「一步就結束」，排程器完全不必知道 pooling 存在。
- **Q5（批次）**：先不做專屬 batcher。TRT-LLM 的 5ms 是慣例常數，值得量但不值得照抄。

**順序建議**：embedding（encoder-only）先做，Whisper 後做。理由是 embedding 這條路整條
都在 Python 側（API + pooler + 一個不進 coordinator 的請求型別），不動共用的
`tokenspeed_scheduler` `.so`；Whisper 的 cross-attention 一定會動 C++ 的 group/family，
而那份 artefact 本箱其他 session 也在用。
