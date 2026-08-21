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

## 6.2　移植後的實測結果（2026-08-21）

whisper-large-v3-turbo、bf16、sm_120、triton backend，LibriSpeech dummy clip 0：

```
truth      MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
reference  ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
ours       ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
```

逐字相同，token id 亦相同。重跑：`unieinfra/bench/whisper_transcribe_parity.py`。

第一版這個檢查用隨機噪音當輸入，參考本身只吐一個 token——「通過」但幾乎沒有證據力。
**ASR 的正確性檢查一定要用真實語音**，否則測不到多步 decode，也測不到 cross-attention
在幾十步之後是否還對。

移植過程中被迫處理、而且不是 Whisper 特有的三件事：

1. `chunked_prefill_size` 的預設 8192 對 `context_len=448` 的模型代表 19 個併發請求，
   但 request pool 只有 `max_num_seqs+1` 個 slot，autotune 的 dummy batch 會索引越界。
   已改成啟動期夾到 `context_len × max_num_seqs` 並印出兩個數字。
2. autotune 的合成 batch 原本不傳 `req_pool_indices`，讀 per-request 狀態的模型會拿不到；
   `make_dummy_batch` 其實已經填好了，補傳即可。
3. **CUDA graph 捕捉期間不能有 `.item()`**。cross-attention 的 `max_seqlen_q` 第一版用
   `query_lengths.max().item()`，那是每個 decode step 一次 host 同步，捕捉時直接違法。
   flash-attn 只拿它排程，給上界即可。

## 6.3　cross-attention 讀取路徑的優化（2026-08-21）

§6.1 記的是「sm_120 沒有按索引讀 KV 的 kernel，所以要複製」。實測之後，正確的結論是
**根本不需要那份複製**：gather 存在只是為了讓該批次的 KV 連續，但一個 decode step 的
cross-attention 成本由「讀多少 KV」決定，不是由「每個請求那一列 query」決定。所以改成對
**整個 slot buffer** 做 attention，把批次的 query 按 pool index 散射進全寬 buffer——
零複製、零額外記憶體、形狀固定（因此 CUDA graph 捕捉得了）。

量了兩個層級，而**兩者的差距本身就是重點**：

| 量測範圍 | large-v3-turbo bs=16 |
|---|---|
| 只看 kernel（gather + attention） | 1.864 → 0.247 ms，**7.55x** |
| 整個 cross-attention block（含 layer norm、q/out 投影、散射） | 1.348 → 0.755 ms，**1.79x** |

**該引用的是 block 的數字。** kernel-only 的 7.55x 沒有撐過周圍那些投影：小批次時整個
block 是 launch-bound（此機空 kernel launch 7.7–13.9 µs），拿掉頻寬成本在那裡買不到東西。
這正是 memory 裡 `eager-microbench-overstates-captured-ops` 的形狀。

完整曲線（block 層級）：

```
large-v3-turbo  bs=1   0.671 → 0.766 ms  0.88x    large-v3  bs=1   5.006 → 5.777 ms  0.87x
                bs=8   0.746 → 0.774 ms  0.96x              bs=8   5.905 → 5.951 ms  0.99x
                bs=16  1.348 → 0.755 ms  1.79x              bs=16 10.920 → 5.936 ms  1.84x
```

padded 路徑的時間**與 bs 無關**（~0.755 ms），gather 則隨 bs 成長——成長的那一段就是複製。
低佔用率時 padded 反而慢約 13%，因為要為一個請求掃過所有 slot。所以**兩條路徑都留著、
依批次佔用率選**（`bs*2 >= slots`，平手點約在半滿）。這是操作點的取捨而非排名，
同 memory 的 `tokenspeed-wins-only-at-high-concurrency`。

兩條路徑都做過端到端轉譯驗證（`--max-num-seqs 1` 走 padded、較大值走 gather），
輸出都與 HF 逐字相同。重跑：`unieinfra/bench/whisper_cross_attn_paths.py`。

## 6.4　embedding：列表請求被拆進兩個 prefill step（2026-08-21）

§3.5 記的分歧點是「只有 TRT-LLM 為 embedding 流量另做批次」。在自己身上量之後，
發現我們有一個更基本的問題，而且它在單一請求的延遲數字裡完全看不見。

前端對列表型請求是**逐條 tokenize 後立刻送出**（`async_llm.py:_handle_batch_request`）。
每個 `await` 都讓出事件迴圈，scheduler 撿走已到達的請求就開跑，於是一個列表請求被拆進
兩個 prefill step。對 embedding 而言那是純損失——呼叫端本來就要等最後一條。

**用預先 tokenize 的輸入當對照組**（`input_ids=` 不可能被 tokenize 拆開），機制立刻現形：

```
n     文字       input_ids   ratio          修正後文字   ratio
 4   20.7 ms      20.4 ms    1.01x            20.7 ms    1.01x
 6   38.4 ms      20.7 ms    1.85x  ← 斷崖    21.3 ms    1.01x
 8   39.1 ms      21.1 ms    1.85x            21.8 ms    1.03x
16   41.1 ms      23.1 ms    1.78x            23.7 ms    1.01x
32   44.8 ms      30.8 ms    1.45x            31.6 ms    1.03x
```

修正是整批 tokenize 完再一起送（SGLang 對 embedding 模型強制的
`enable_tokenizer_batch_encode` 就是這件事）。修正後文字路徑在每個 n 都與對照組
一致（1.01–1.03x），**表示被修掉的是機制而不只是症狀**。最高 1.80x。

**不套用到生成請求**：那裡先送第一條能降低它的 TTFT，串流呼叫端在其餘到達前就能用。

回歸探針：`unieinfra/bench/embedding_batch_shape.py`（門檻 1.15x，超過即失敗）。

## 6.5　embedding 省掉 prefill 的 KV 寫入：在這台目前拿不到（2026-08-21）

§1 記了 SGLang 對 embedding 模型有 `fa_skip_kv_cache`——pooling 請求不會回讀自己的 KV，
所以那次寫入是純浪費。照著做了，但**在這個 backend 上拿不到**，原因值得記。

MHA backend 的 extend 有兩種模式。**postwrite** 會先用傳入的 raw k/v 算出 attention、
之後才寫頁，所以那次寫入可以省。**prewrite** 反過來：先寫頁、再從頁讀回來算，
省掉寫入就等於讀未初始化的記憶體。

模式由 kernel plan 決定：只有在該 dtype／traits 有 **PERFORMANT 優先級的 prefill kernel**
時才選 postwrite（`tokenspeed_kernel/ops/attention/__init__.py:3276`）。這台 bf16 + triton
解析成 **prewrite**，所以省不掉。

**這是被守門擋下來的，不是被量出來的。** 第一版直接把 `save_kv_cache=False` 傳下去，
守門立刻報 `extend_mode='prewrite'`——如果沒有那個檢查，就會 attend 到未初始化的頁，
回傳形狀正確、範數正確、但數值錯誤的向量，下游沒有任何東西攔得住。

目前的行為是偵測到就退回寫入並記錄一次（不是 crash：一個用不上的優化不該讓伺服器停擺）。
開關 `--embedding-skip-kv-write` 預設開著，實際生效與否取決於有沒有 postwrite kernel。
**要讓它生效的前提是為 bf16 註冊一支 PERFORMANT 的 prefill kernel**，那是另一件事。

## 6.6　每請求約 16–18 ms 的固定延遲地板（2026-08-21）

追 Whisper decode 的過程中，注意到兩個不相干的數字撞在一起：embedding 的 8-token
單一請求是 19.3 ms，Whisper 的固定成本（encoder 跑 1500 frames + prefill）是 23.0 ms。
不同模型、差了兩個數量級的工作量，卻都落在 20 ms 附近。

用小 13 倍的模型驗證：

```
Qwen3-0.6B   5-token prompt,  1 token out    17.3 ms
             500-word prompt, 1 token out    17.8 ms   ← prompt 大 100 倍，只多 0.5 ms
             每 token ≈ 1.60 ms，固定 ≈ 15.7 ms
```

**這個地板與模型大小和 prompt 長度都無關**，所以不是計算。而且 `e2e_latency`（引擎內部
量的，17.6 ms）與 client 的 wall clock（18.0 ms）幾乎相同，所以也不是 Python 橋接。

**它是管線延遲，不是 CPU 工作量**——證據是它會被分攤掉：32 條 embedding 請求總共
31.6 ms（每條約 1 ms），而 1 條要 19.5 ms。

影響面：
- **吞吐**（大量並發）：不受影響，已經攤掉了。
- **延遲**（單請求）：這是主成本。Whisper 一段 23-token 轉譯共 46 ms，其中 23 ms 是它；
  embedding 單一短請求 19.3 ms，其中約 16 ms 是它。

**用既有開關二分**（Qwen3-0.6B、5-token prompt、1 token out、p50）：

```
baseline               17.50 ms
overlap off            16.84 ms   -0.66
stream_interval=8      17.58 ms   +0.08
inline detok off       17.40 ms   -0.10
prefix cache off       17.26 ms   -0.24
prefill graph 512      17.33 ms   -0.35
enforce_eager          30.43 ms   +12.93   <- 唯一大幅移動的
```

**一個被推翻的假設**：既然 `enforce_eager` 貴 12.9 ms、而 prefill 在這棵樹預設是 eager
（prefill graph 是 opt-in），我原本推論那 16 ms 是 eager prefill 的 kernel launch 開銷。
**開了 prefill graph 只省 0.35 ms，假設不成立。**

`enforce_eager` 同時關掉 decode graph，所以它動到的是**每輪迴圈的成本**，不是某一次
forward。最符合全部數據的模型是：

    延遲 ~= (請求要活過的 scheduler 輪數) x (每輪成本)

這同時解釋了為什麼地板跟著層數走（每輪成本）而不跟 prompt 長度走，以及為什麼並發時會被
攤掉（多個請求共用同一批輪次）。

### 定位：整個地板都在 `prefill_ms`

不需要新增量測點——引擎自己的 `--enable-log-request-stats` 已經在量了，先前只是沒去讀
它的 log（它走 `logger.info`，預設 log level 看不到）。Qwen3-0.6B、5-token prompt、
1 token out：

```
queue_ms     0.4–0.7 ms    created -> scheduled（前端 hop + 准入）
prefill_ms  15.4–16.4 ms   scheduled -> prefill 完成          <- 地板全在這
ttft_ms     15.8–17.1 ms
CLIENT wall 17.2–18.3 ms   （所以 client 橋接約 1 ms）
```

**不是排隊、不是准入、不是回程**，是 scheduled 到 prefill 完成之間那一段。

### 一個要撤回的判讀

上面表中「prefill graph 512 只省 0.35 ms」**不能當成負面結果**：那一輪的
`prefill breakable graph: captured buckets ...` 日誌行從未出現，也沒有 capture failed，
**graph 根本沒有被捕捉過**。旗標確實有進到 resolved config
（`prefill_graph_max_tokens=2048`、`disable_prefill_graph=False`），所以是被
`runtime/execution/prefill_graph.py:289 self.disable` 那組條件中的某一個靜默擋掉的，尚未查出是哪一個。

所以正確的狀態是**不確定**，不是「prefill graph 沒用」。這正是 memory 裡
`a-negative-result-is-easy-to-produce-by-accident` 那條：先證明機制有 engage，
再相信 null。

### 根因：GDN 守門對每一個模型都關掉 prefill graph

先讓「被停用」自己說原因（`runtime/execution/prefill_graph.py:293` 現在會印出是哪一個
條件命中），一跑就出來了：

```
prefill breakable graph disabled: disable_prefill_graph
```

而 server_args 的 `disable_prefill_graph` 是 False。真正動手腳的是
`runtime/engine/event_loop.py:227 resolve_fi_gdn_prefill_graph`：它的 docstring 說
「當 flashinfer GDN chunk-prefill kernel 在使用時關掉 prefill graph」，**但它從來沒有
檢查模型是否真的有 GDN 層**——只看兩個環境變數，然後無條件關掉。

Qwen3-0.6B 是完全沒有 linear attention 的稠密模型，也被關了。用環境變數繞過後：

```
TOKENSPEED_ENABLE_FI_GDN_SM120=0
  prefill breakable graph: captured buckets [16, 32, ..., 2048] (segments=57)
  prefill breakable graph ENGAGED (text): bucket=16
  p50 17.3 ms -> 8.98 ms
```

**單請求延遲約 2 倍**，而且這次有 `ENGAGED` 日誌證明機制生效——正是先前那個假陰性缺的。

修法是讓守門精確：檢查 text config 的 `layer_types` 有沒有 linear attention 層。
沒有那種層，GDN kernel 就不可能在使用。兩個方向都驗過：

| 模型 | 守門 | prefill graph | 輸出 |
|---|---|---|---|
| 稠密 Qwen3-0.6B | 不觸發 | 捕捉並 ENGAGED | 正確 |
| Hybrid Qwen3.5-2B | **仍然觸發** | 停用 | 正確（非垃圾） |

hybrid 那個方向是危險方向：GDN + prefill graph 會回傳垃圾但吞吐數字更好，沒有 gate
攔得住，所以守門必須繼續生效。

**注意**：`engine.server_args.disable_prefill_graph` 從前端讀是 False，因為那個改寫發生在
scheduler 子行程裡。要確認實際狀態，看 scheduler 的日誌行，不要看前端的欄位。

**影響面遠大於 embedding／Whisper**：這台上每一個稠密與 MoE 模型的 prefill 都在跑 eager。

### embedding 一度沒有受惠——已解決（2026-08-21）

用**實際的修法**（不是環境變數繞過）重量，先前出現不對稱：

```
dense     Qwen3-0.6B           17.3 -> 8.71 ms   2.0x
embedding Qwen3-Embedding-8B   19.3 -> 19.78 ms  無改善
```

原因是 `runtime/execution/prefill_graph.py` 的停用條件裡有 `not model_runner.is_generation`
——**prefill graph 對 pooling 模型被明確排除**，embedding 仍在跑 eager prefill。

**這個排除比機制本身的需求更嚴。** capture 捕的是內層 transformer stack、回傳 hidden
states（`_run_inner`）；generation 與 pooling 的差別只在 `replay()` 之後跑的那個 tail
（logits processor 對上 pooler），**capture 本身與模式無關**。所以排除條件換成兩個更窄的：
pooling 模型預設進來、`--no-pooling-prefill-graph` 可關；而 logits processor 與 pooler
兩者皆無的模型仍然沒有 tail 可跑，繼續排除。`replay()` 分支到
`text_model.pool(ctx, hidden_states)`，就是 `BaseCausalLM.forward` 原本的 eager tail。

過程中還有一個會靜默吃掉整個效果的細節：`_capture_bucket` 取 decode wrapper 的
capture stream 必須用 `getattr`。那個 wrapper 只有在自己 capture 時才會建 `.stream`，
而它對 pooling 模型不 capture，直接屬性存取會在 capture 失敗處理裡丟 AttributeError，
**每個 bucket 都靜默 fallback 回 eager**。

量到的結果（Qwen3-Embedding-8B、單一短請求、暖機後 20 次的 p50/min）：

```
--pooling-prefill-graph      p50 14.84 ms   min 14.66 ms
--no-pooling-prefill-graph   p50 19.75 ms   min 19.41 ms
```

**1.33x，而且兩個方向都有機制證據**：ON 那臂印出 `captured buckets [16, 32, ...]` 與
`prefill breakable graph ENGAGED (text): bucket=16`，OFF 那臂印出
`disabled: pooling model and --no-pooling-prefill-graph`。

正確性是**在 graph 確實 engage 的同一次執行裡**重驗的——先前那次 parity 的
`breakable graph` 日誌行數為 0，所以不能算數：對 HF reference 的 cosine
`[0.999864, 0.999882, 0.999883]`，與 eager 路徑相同。

注意 embedding 的 1.33x 小於 dense 的 2.0x。兩者不是同一個量：dense 是 0.6B、17.3 ->
8.71 ms，embedding 是 8B，模型大 13 倍，被 graph 消掉的 launch 開銷佔比自然較小。



## 6.7　encoder-only 架構落地：BERT / RoBERTa-XLM-R / GTE-v1.5（2026-08-21）

先前這台上每一個 pooling 模型都是「decoder 外掛 pooler」（Qwen3-Embedding）：因果 attention、
分頁 KV pool、LAST reduction。BERT 家族是另一種形狀——雙向 attention、完全沒有 KV、
一次 forward 就結束。

**「BERT / GTE / BGE」是三個架構，不是同一個架構的三個 checkpoint：**

| HF `architectures` | 代表 checkpoint | 位置編碼 | MLP |
|---|---|---|---|
| `BertModel` | `bert-base`、`BAAI/bge-*-en-v1.5`、`thenlper/gte-*` | learned absolute | 2 層 post-LN |
| `XLMRobertaModel` / `RobertaModel` | `intfloat/multilingual-e5-*`、`BAAI/bge-m3` | absolute，起點是 `pad_token_id + 1` | 同上 |
| `NewModel` | `Alibaba-NLP/gte-*-en-v1.5` | **RoPE + NTK scaling** | **gated（GeGLU）** |

RoBERTa 那個位移就是兩家 forward 的全部差異，也是為什麼那些 config 的
`max_position_embeddings` 是 514 而不是 512。

### 第三個架構裡兩個「照 config 寫就會錯」的地方

1. **`hidden_act` 是 `"gelu"`，但 checkpoint 出的是 `mlp.up_gate_proj`——MLP 其實是 GeGLU。**
   vLLM 在 config 層改寫這個欄位（`model_executor/models/config.py:391
   config.hidden_act = "geglu"`）；checkpoint 自己的 reference 實作
   （`Alibaba-NLP/new-impl` `modeling.py` `NewGatedMLP`）殊途同歸，它只把
   `ACT2FN[hidden_act]` 套在 gate 那一半。照字面讀 config 會得到一個非 gated 的 MLP，
   而且**對著一半的權重載入得乾乾淨淨**。
   注意 checkpoint 的排列是 `[up; gate]`，與本樹 fused `gate_up` 的慣例相反。
2. **`rope_scaling {"type": "ntk", "factor": 2.0}` 縮放的是 base 頻率**，所以它改變
   *每一個* 位置，不是只改訓練窗口之外的位置。短 prompt 的測試抓不到漏掉它。
   本樹的 `get_rope` 沒有 `"ntk"` 這個 scaling type，加一個會動到每個 decoder 共用的
   rotary 層，所以 cache 放在模型檔內（`NTKScalingRotaryCache`）。

   實際的縮放是兩步，而且**不是條件式的**：reference 建了兩次 cache，第一次用
   `max_position_embeddings`（守門是嚴格 `>`，不觸發），第二次用
   `max_position_embeddings * factor`（觸發）。第二次的結果才是服務時用的：

       base' = base * factor        inv_freq = 1 / base'^(2i/d)
       inv_freq /= factor^(2/d)

### attention 不需要新工作

§7 Q1 早就記過：非因果 encoder pass 在本樹已解決。三個架構都走
`mm_encoder_attention` 的 varlen backend——Whisper encoder 用的同一條 pool-free 路徑。
GTE 的 RoPE 也不需要新東西，`MultimodalEncoderAttention` 本來就收
`rotary_pos_emb_cos` / `rotary_pos_emb_sin`。

### 批次佈局取自 `gather_ids`，不取自 attention backend

encoder-only 模型沒有任何 attention 層在驅動那個 backend，而 `ctx.gather_ids` 已經是
`BaseCausalLM.pool` 拿來 reduce 的那個 tensor，所以 attention 與 pooling **不可能**
對「某個請求的 row 在哪裡」有不同看法。`gather_ids` 在 extend forward 上就是
`cumsum(extend_lens) - 1`，所以 `cat([0, gather_ids + 1])` 直接**就是** cu_seqlens。

`max_seqlen` 給的是上界（`input_num_tokens`）而不是精確值：精確值要 `.item()`，
那是一次 device sync，在捕捉中的 CUDA graph 裡非法（Whisper decoder 撞過同一條）。
varlen kernel 只拿它決定 grid 大小、再用 per-request 長度遮罩，所以高估只是空 block，
不會算到錯的 row。

### 對 HuggingFace 的 parity（bf16、sm_120、triton、四條不等長 prompt 打成一個 batch）

```
bge-small-en-v1.5      cls  + normalize      worst cosine 0.999916
multilingual-e5-base   mean + normalize      worst cosine 0.999947
gte-base-en-v1.5       cls  + 不 normalize    worst cosine 0.999939
```

pooling 模式是**由命令列明確傳入**而不是從 checkpoint 讀，所以這同時測到 engine 自己的
`resolve_pooling_config` 有沒有得到同一個答案。三個都對，包含 **GTE 沒有 Normalize
module** 這一項——一個永遠做 L2 正規化的 server 會對它回傳一個格式完美的錯向量。

### prefill graph：問題不在缺 buffer，在於 attention 沒有被標成 break（2026-08-21）

一開始的結論是「不能開」：這個 stack 在**被捕捉的區域內**讀 `ctx.gather_ids` 取批次佈局，
而 graph 捕捉時用的是 `make_dummy_batch` 的 context，捕捉完就丟掉
（`execution/prefill_graph.py` `self._ctx = None`），沒有東西每次 replay 刷新它。開著 graph
重驗 parity 掉到 0.295（bge）／0.683（e5）——**格式完美、算在錯的 row 上的向量**。

那個歸因是對的，但**開出的藥方（做一個每次 replay 刷新的批次佈局 buffer）是多餘的**。
這棵樹已經有正確答案，而且每個 decoder 都在用：`@break_point`
（`execution/breakable_cuda_graph.py`）。被它標記的 sequence-mixing 方法在**每次 replay
以 eager 重跑**，周圍 token 形狀的計算（norm、投影、MLP）才被捕捉；而且
`_record_break` 對參數的處理有兩個「活的」例外：

1. **tensor 參數別名到持久儲存**（static input buffer／pool-pinned 的 segment 中間值），
   所以 replay 時帶的是新值；
2. **`ForwardContext` 參數每次 replay 按 identity 重新綁定到 ambient context**
   （`active_forward` / `current_forward_ctx`）——docstring 寫得很直接：
   「so break bodies read live `ctx` fields exactly like the eager path」。

所以正確的修法是把 `EncoderSelfAttention.forward` 標成 `@break_point` 並讓它收 `ctx`，
在 break body 內才算 `cu_seqlens`。encoder 之所以壞掉，不是 graph 少了什麼設施，
**是它沒有像其他模型那樣把 attention 標成 break**。

一個實作細節值得記：rope 的 cos/sin **逐層重算**，沒有提到迴圈外。兩種寫法 eager 都對，
但在捕捉下，提到外面等於把 segment 0 產生的 tensor 交給第 11 個 break，而 break 參數
別名 pool 儲存的安全前提是「捕捉的 segment 會在 break 讀取前改寫該位址」——那只對
**緊鄰 break 的前一個 segment**成立。逐層重算維持這個前提，代價是 11 次額外的
`index_select`。cos/sin 也拆成兩個參數而不是一個 tuple，否則 tuple 會被當成不透明物件
整個 pin 住 pool block。

### 量到的結果，以及倍率為什麼縮水

```
--pooling-prefill-graph      p50 5.70 ms   ENGAGED (bucket=80)
--no-pooling-prefill-graph   p50 7.83 ms   disabled
```

**1.37x**，兩個方向都有機制證據，而且**兩臂的 cosine 逐位相同**
（`[0.999959, 0.999964, 0.999969, 0.999916]`），所以 graph 沒有動到任何數值。

先前記的 **2.03x 要撤回**。那個數字是在 attention 被**錯誤地捕捉進 graph** 的情況下量到的
——它快，正是因為它錯：把佈局凍結進 graph 就省掉了每次 replay 的 attention launch。
正確的版本讓 attention 保持 eager，於是每層多一次 kernel launch 與一次 graph segment 邊界，
獎品從 2.03x 縮到 1.37x。**這是「快」與「對」互斥時的實際價碼，不是實作沒調好。**

`supports_prefill_graph` 這個宣告式退出留在 graph 端但目前無人使用：encoder 家族在
attention 變成 break 之前需要它，下一個需要的家族應該找到這個接縫而不是重新發明。

### 一個在 reference 那一側的陷阱

`AutoModel.from_pretrained`（transformers 5.12）從 meta 具現化模型再用 checkpoint 填，
而 `persistent=False` 註冊的 buffer **依定義不在任何 checkpoint 裡**，也沒有人重跑
建構子，所以它們拿到的是當時那塊記憶體的內容。GTE 上三個全死：

```
rotary_emb.cos_cached   全 0
rotary_emb.sin_cached   全 0
rotary_emb.inv_freq     -1.09e30, 4.47e-41, ...
embeddings.position_ids -1.07e12, -9.4e11, -59873, ...
```

`cos = sin = 0` 讓 reference 的 `apply_rotary_pos_emb` 回傳
`q*0 + rotate_half(q)*0`——它把每一個 query 和 key 歸零，attention 退化成對 V 的
均勻平均。**輸出的形狀與 norm 幾乎不變。** 拿一個*正確*的實作去比，得到 cosine 0.31，
讀起來完全像是自己的模型壞了。只修 cos/sin 到 0.73，剩下的是那個垃圾 `position_ids`。

修法不是逐個打補丁，是換一條建構路徑：`from_config` 會真的跑建構子，再明確
`load_state_dict`。`_assert_buffers_live` 是「這次真的有效」的檢查，而不是等下一個
靜默變體。這條與 `gemma4-no-backend-sm120` 正好是一體兩面：那條說「叫自己的模型壞掉
之前先對 HF reference」，這條說 HF reference 本身也是會靜默壞掉的東西。


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
  所以站 vLLM/TRT-LLM 那邊。**已完成**（§6.2、§6.3），而且結果比原本預想的省：實作成
  arena 外的一塊 per-slot buffer，第三種 family 根本不需要，C++ 側一行未動。
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
