# 10 — API layer 五方橫向對照

版本見 `00-METHOD.md`。所有行號只對該表的 pinned commit 有效。

範圍：從 socket 收到 HTTP 位元組，到 SSE chunk 寫回 socket。**不含** scheduler 之後
的任何東西（那是 `20-scheduler.md`）。

---

## 0. 程序拓撲 — 先看這個，步驟表才讀得懂

同一件事（tokenize、detokenize、stop 判定）在五個引擎裡跨的程序邊界不同，
這決定了哪些成本是 IPC、哪些是純 Python、哪些根本不在這個 repo 裡。

```
UnieInfer   client ─HTTP→ [smg gateway (Rust)] ─gRPC→ [AsyncLLM 主程序] ─ZMQ→ [Scheduler ×TP]
                            tokenize                    inline detokenize      stop 判定
                            detokenize                  (重複做一次)
                            SSE/OpenAI 協定

vLLM        client ─HTTP→ [API server 程序] ─ZMQ→ [EngineCore 程序]
                            tokenize                     排程 + forward
                            detokenize + stop 判定
                            SSE/OpenAI 協定

SGLang      client ─HTTP→ [http_server + TokenizerManager 主程序] ─ZMQ→ [Scheduler ×TP]
                            tokenize                                    stop 判定
                            SSE/OpenAI 協定                              ↓ ZMQ
                                        ↑───────── ZMQ ───── [DetokenizerManager 程序]

TRT-LLM     client ─HTTP→ [openai_server 程序] ─(IPC 或 in-proc)→ [PyExecutor]
                            tokenize                                 排程 + forward
                            detokenize + 協定  ← 預設同程序；
                                                 num_postprocess_workers>0 時外移到 PostprocWorker 程序池
                                                 stop 判定在 executor 側

TurboMind   client ─HTTP→ [api_server + AsyncEngine 同一個 Python 程序] ─pybind→ [C++ Engine 執行緒]
                            tokenize                                             stop 判定（GPU kernel）
                            detokenize（incremental）
                            SSE/OpenAI 協定
```

**四個獨立的觀察**：

1. **只有 UnieInfer 的 OpenAI 協定層不在被測的 repo 裡**，而且是一個編譯過的 Rust
   binary（`smg_rs.abi3.so`，套件 `tokenspeed-smg` 1.8.0）。這一層的每一格都無法引用
   原始碼；能引用的只有它與引擎之間的 gRPC servicer。
2. **只有 UnieInfer 有兩層 HTTP**（smg gateway 在主 port，control sidecar 在 port+1），
   而且 sidecar 對 `/v1/*` 是**反向代理回 gateway**（`runtime/entrypoints/control_server.py:344` →
   `_proxy_request`）。打哪個 port 決定量到的 ingress 是一跳還是兩跳。
3. **只有 TurboMind 全程單一 Python 程序 + 同程序 C++ 執行緒**，沒有任何 IPC。
4. **只有 SGLang 把 detokenize 放在獨立程序**；vLLM / TurboMind 放在 HTTP 程序，
   TRT-LLM 預設放在 HTTP 程序（可外移），UnieInfer 放在 Rust gateway（且引擎端又做一次）。

---

## 1. 步驟表

`—` = 這個引擎沒有這一步。`(bin)` = 編譯產物，無原始碼可引用。

### 1.1 進場（request → engine）

| # | 步驟 | UnieInfer | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 1 | HTTP 伺服器 | smg gateway `(bin)` | `entrypoints/openai/api_server.py` (FastAPI) | `srt/entrypoints/http_server.py` (FastAPI) | `serve/openai_server.py:841 register_routes` (FastAPI) | `serve/openai/api_server.py` (FastAPI) |
| 2 | `/v1/chat/completions` 路由 | smg `(bin)`；sidecar 代理 `runtime/entrypoints/control_server.py:344` | `entrypoints/openai/chat_completion/api_router.py:53 create_chat_completion` | `srt/entrypoints/http_server.py:1702 openai_v1_chat_completions` | `serve/openai_server.py:1424 openai_chat` | `serve/openai/endpoints/chat_completions.py:205 chat_completions_v1` |
| 3 | 協定物件驗證 | smg `(bin)` | pydantic `entrypoints/openai/chat_completion/protocol.py` | `srt/entrypoints/openai/protocol.py` | `serve/openai_protocol.py` | `serve/openai/protocol.py` |
| 4 | chat template 套用 | smg `(bin)` | `renderers/base.py:1034 render_chat` / `:1070 render_chat_async` | `srt/entrypoints/openai/chat_encoding.py:194` | `serve/chat_tokenization.py:92 render_chat_request_for_tokenizer` | `serve/core/async_engine.py`（`chat_template` + `Tokenizer`） |
| 5 | **tokenize** | **smg `(bin)`（Rust）**；引擎端因此跳過 | `v1/engine/input_processor.py:242 process_inputs` | `srt/managers/tokenizer_manager.py:998 _tokenize_one_request` | `serve/chat_tokenization.py:114 tokenize_chat_request_for_serving` | `serve/core/async_engine.py:124 self.tokenizer` |
| 6 | 取樣參數正規化 | `smg_grpc_servicer/tokenspeed/servicer.py:1016 _sampling_params_from_proto` | `renderers/params.py` + `SamplingParams` | `srt/sampling/sampling_params.py` | `_torch/pyexecutor/llm_request.py:1118 convert_wordlist` 等 | `serve/core/generation_config.py`；`:409 convert_stop_bad_words_to_ids` |
| 7 | multimodal 前處理 | `smg_grpc_servicer/tokenspeed/servicer.py:1102 _mm_inputs_from_proto`（gateway 已切好，走 shm/RDMA） | `renderers/inputs/` | `srt/managers/multimodal_processor.py` | `serve/openai_server.py`（`VisualGen` 分支） | `serve/processors/multimodal.py` |
| 8 | 內部 request 物件建構 | `smg_grpc_servicer/tokenspeed/servicer.py:912 _build_generate_req` → `GenerateReqInput` | `v1/engine/async_llm.py:280 add_request` → `EngineCoreRequest` | `srt/managers/tokenizer_manager.py:765 generate_request` → `TokenizedGenerateReqInput` | `_torch/pyexecutor/llm_request.py:1118` → `LlmRequest` | `turbomind/turbomind.py:701 async_stream_infer` → `ModelRequest::InputParam` |
| 9 | **跨程序邊界** | **gRPC**（`smg_grpc_servicer/tokenspeed/servicer.py:158 Generate`，proto `tokenspeed_scheduler.proto`） | **ZMQ + msgpack**（`v1/engine/core_client.py:1121 AsyncMPClient.add_request_async`） | **ZMQ**（`srt/managers/tokenizer_manager.py:1561 _send_one_request`） | IPC 或同程序（`executor/proxy.py` / `rpc_proxy.py`）；預設由 `LLM` 決定 | **無**（pybind11 呼叫同程序 C++：`turbomind/turbomind.py:602 model_inst`） |
| 10 | 送進 engine 佇列 | `runtime/engine/async_llm.py:322 _send_one_request` | `v1/engine/core.py` EngineCore 的 input queue | `srt/managers/scheduler.py` recv loop | `_torch/pyexecutor/executor_request_queue.py:122 enqueue_request` | `src/turbomind/engine/gateway.cc:31 Gateway::push` |

### 1.2 出場（engine → response）

| # | 步驟 | UnieInfer | vLLM | SGLang | TRT-LLM | TurboMind |
|---|---|---|---|---|---|---|
| 11 | engine → 前端的輸出幀 | ZMQ → `runtime/engine/async_llm.py:797 handle_loop` | ZMQ → `v1/engine/async_llm.py:637 _run_output_handler` | ZMQ → `srt/managers/detokenizer_manager.py:166 event_loop` → 再 ZMQ 給 TokenizerManager `:2215 handle_loop` | callback → `executor/result.py` | C++ callback → `turbomind/turbomind.py:447 _func` |
| 12 | **stop token / stop string 判定** | **Scheduler 程序**：`runtime/engine/request.py:297 check_finished` | **API server 程序**：`v1/engine/output_processor.py:639` + `v1/engine/detokenizer.py:86,107` | **Scheduler 程序**：`srt/managers/schedule_batch.py:1467 _check_token_based_finish` / `:1519 _check_str_based_finish` | **Executor 側**：`_torch/pyexecutor/llm_request.py` 的 `py_stop_words_list` | **GPU kernel**：`src/turbomind/generation/stop_criteria.cc:58 StopCriteria::Forward` → `src/turbomind/kernels/stop_criteria_kernels.cu:59 invokeStopWordsCriterion_v2` |
| 13 | incremental detokenize | **兩次**：smg `(bin)` + `runtime/engine/detokenizer.py:253 IncrementalDetokenizer`（`enable_inline_detokenizer` 預設 `True`，`runtime/utils/server_args.py:111`） | `v1/engine/detokenizer.py:167 FastIncrementalDetokenizer` / `:250 SlowIncrementalDetokenizer` | `srt/managers/detokenizer_manager.py:430 handle_batch_token_id_out`（獨立程序） | `serve/postprocess_handlers.py:216 chat_stream_post_processor` | `serve/core/async_engine.py:668 detokenize_incrementally` |
| 14 | delta 文字 / 增量差分 | smg `(bin)` | `entrypoints/openai/chat_completion/serving.py:408 chat_completion_stream_generator` | `srt/entrypoints/openai/serving_chat.py:1443 _handle_streaming_request` | `serve/postprocess_handlers.py:216`（`token_ids_diff`） | `serve/openai/endpoints/chat_completions.py` |
| 15 | SSE 封裝 | smg `(bin)` | 同 14 | `srt/entrypoints/openai/sse_utils.py` | `serve/openai_server.py:1433 chat_stream_generator` | 同 14 |
| 16 | finish_reason / usage | `smg_grpc_servicer/tokenspeed/servicer.py:1483 _complete_response`（proto 欄位） | `v1/engine/output_processor.py:576 process_outputs` | `srt/managers/tokenizer_manager.py:1704 _wait_one_response` | `serve/postprocess_handlers.py:63 _ctx_usage_for_postproc` | `serve/core/async_engine.py:44 GenOut` |
| 17 | metrics | 引擎端 `--enable-metrics`，暴露在 **sidecar port**（gateway 沒有 `/metrics`） | `v1/metrics/`（預設開） | `srt/managers/scheduler_components/metrics_reporter.py` | `_torch/pyexecutor/perf_metrics_manager.py` | `lmdeploy/serve/core/async_engine.py` |

---

## 2. 分歧點

### 2.1 tokenize 與 detokenize 各自在哪個程序 — 五家四種答案

| engine | tokenize | detokenize | 兩者同程序？ |
|---|---|---|---|
| UnieInfer | Rust gateway | Rust gateway **＋引擎內嵌各一次** | 是（但做兩次） |
| vLLM | API server | API server | 是 |
| SGLang | TokenizerManager（主程序） | DetokenizerManager（**另一個程序**） | 否 |
| TRT-LLM | API server | API server（可選外移到 worker 池） | 是 |
| TurboMind | API server | API server | 是 |

**SGLang 是唯一把 detokenize 拆成獨立程序的**，代價是每個 token 多一次 ZMQ 來回
（Scheduler → DetokenizerManager → TokenizerManager），好處是 detokenize 不佔 HTTP
程序的 GIL。UnieInfer 繼承了 SGLang 的程序骨架，但把 detokenizer 收回主程序內嵌
（`runtime/entrypoints/engine.py:583` 的註解明說「no separate subprocess」），**同時**前面又多了
一個會自己 detokenize 的 Rust gateway。

### 2.2 UnieInfer 的重複 detokenize

gRPC 回傳的是 **token id**，不是文字：
`smg_grpc_servicer/tokenspeed/servicer.py:1460 _chunk_response` 填的是 `GenerateStreamChunk.token_ids`，
`smg_grpc_servicer/tokenspeed/servicer.py:1483 _complete_response` 填的是 `GenerateComplete.output_ids`。
smg gateway 拿到 id 之後自己 detokenize 成 OpenAI 的 `delta.content`。

但引擎端 `enable_inline_detokenizer` 預設 `True`（`runtime/utils/server_args.py:111`），
而 `skip_tokenizer_init` 預設 `False` 且 `cli/serve_smg.py` 不會設它。因此
`runtime/engine/output_processor.py` 走的是 detokenize 分支，每個 token 都在引擎主程序
做一次 HF tokenizer decode，**結果只被 `_generated_output_ids`（`smg_grpc_servicer/tokenspeed/servicer.py:1429`）
丟掉 token id 之外的部分**。

唯一需要引擎端文字的是 stop string 判定（`runtime/engine/request.py:344`）—— 那是在
**Scheduler 程序**做的，用的是 Scheduler 自己的 detokenize 狀態，不是主程序這一份。

→ 這是一個明確的、可量的重複工作。列入 `90-DIVERGENCES.md`。

### 2.3 stop 判定的位置 — 三種完全不同的答案

- **vLLM 在 HTTP 程序、在 detokenizer 裡**（`v1/engine/detokenizer.py:86,107`）：stop string 是
  detokenize 的副產品，不需要額外解碼，但 engine 會多跑到 detokenizer 發現為止。
- **SGLang / UnieInfer 在 Scheduler 程序**（`srt/managers/schedule_batch.py:1519` /
  `runtime/engine/request.py:297`）：scheduler 能立刻停掉這一列，代價是 scheduler 迴圈裡有
  `tokenizer.decode`（SGLang `srt/managers/schedule_batch.py:1514` 明確呼叫）。
- **TurboMind 在 GPU 上**（`src/turbomind/kernels/stop_criteria_kernels.cu:59`）：完全不需要 CPU 解碼，
  但只能比對 **token id 序列**。`serve/openai/endpoints/chat_completions.py:236` 的 API 文件因此寫
  「Only accept stop words that's encoded to one token idex」—— 限制在協定層，
  kernel 本身其實支援多 token 的 stop word（`src/turbomind/kernels/stop_criteria_kernels.cu:51` 是逐 token
  回走比對）。

SGLang 額外做了一件其他四家都沒做的事：`Req.check_match_stop_str_prefix`
（`srt/managers/schedule_batch.py:1438`）在 **串流輸出前**檢查尾巴是否是某個 stop string 的**前綴**，
是的話就先不吐 —— 避免把之後要被截掉的字送出去。呼叫點在
`srt/managers/scheduler_components/output_streamer.py:384`。

### 2.4 誰有 postprocess worker 池

只有 TRT-LLM。`serve/openai_server.py:756 postproc_worker_enabled` 讀
`num_postprocess_workers`，**預設 0**（`executor/postproc_worker.py:48`），也就是預設行為
與 vLLM 相同（在 HTTP 程序做）；設 >0 時 detokenize + 協定組裝整批外移到
`executor/postproc_worker.py:60 PostprocWorker` 程序池。其他四家沒有這個選項。

### 2.5 兩層 HTTP

UnieInfer 的 `tokenspeed serve` 同時起 smg gateway（主 port）與 control sidecar
（port+1）。sidecar 的 `/v1/*` 是**代理回 gateway**（`runtime/entrypoints/control_server.py:344` →
`:147 _proxy_request`，用 aiohttp 再開一條連線），只有 `/health`、`/abort`、
`/get_model_info` 這類走 gRPC 直連引擎。

→ **量 ingress 時打錯 port 會多量一整跳 aiohttp 代理。** 既有的
`unieinfra.launch.json` 用的是主 port（gateway），是對的；但 `--enable-metrics` 的
`/metrics` 只在 sidecar port —— 也就是說「同一個 base_url 同時拿 completion 和 metrics」
在這個引擎上是不成立的。

### 2.6 SSE chunk ≠ token

五家都可能在一個 SSE chunk 裡帶多個 token（spec decode 接受多 token、
`stream_output` 批次），也都可能送出空 chunk（TRT-LLM 的
`_ensure_stream_metadata`、SGLang 的 stop-prefix 抑制）。

→ 量測時一律以 `usage.completion_tokens` 為準，不要數 chunk。這條規則是因為數
chunk 曾把 spec decoding 的結果整個反轉過。

---

## 3. 結構差異（同功能、反結構）

| 功能 | 多數作法 | 例外 | 後果 |
|---|---|---|---|
| OpenAI 協定層 | 與引擎同一個 repo、同一個語言 | UnieInfer：獨立 Rust binary，跨 gRPC | 協定層的成本無法用 Python profiler 看到；也無法在這個 repo 裡改 |
| detokenize 的次數 | 一次 | UnieInfer：兩次（gateway + 引擎內嵌） | 每 token 一次多餘的 HF decode |
| stop 判定 | CPU、字串比對 | TurboMind：GPU、token id 比對 | TurboMind 的 stop 不需要解碼，但表達力受限於 token 邊界 |
| 串流抑制 | 產生就送 | SGLang：尾巴是 stop 前綴時先不送 | 少一次「送出去又要收回」的正確性問題 |
| tokenizer 的所有權 | HTTP 層持有 | UnieInfer：gateway 透過 `GetTokenizer` RPC（`smg_grpc_servicer/tokenspeed/servicer.py:811`）從引擎下載 zip 再自己建 | gateway 可以跑在沒有模型檔的機器上 |

---

## 4. 可量測候選（下一輪的輸入）

| 候選 | 量什麼 | 為什麼值得 |
|---|---|---|
| A1 引擎端重複 detokenize | 開 / 關 `enable_inline_detokenizer`（或 `skip_tokenizer_init`）的每 token 主機端成本 | 已確認是純重複工作；影響每一個 decode token |
| A2 兩層 HTTP | 打 gateway port vs sidecar port 的 `GET /v1/models` 與 1-in-1-out | 確認過去的 ingress 數字量的是哪一跳 |
| A3 detokenize 的程序位置 | UnieInfer / vLLM / SGLang 三家在高並發下 detokenize 佔 HTTP 程序 CPU 的比例 | SGLang 外移、其他內含，這是設計上的取捨，不是 bug |
| A4 stop 判定成本 | 有 / 無 stop string 的 scheduler 迴圈時間（UnieInfer、SGLang） | 兩家都在 scheduler 程序裡呼叫 `tokenizer.decode` |
| A5 gRPC vs ZMQ 邊界 | 單一請求的送出→引擎收到延遲 | UnieInfer 是五家唯一用 gRPC 的；protobuf 序列化成本未知 |

---

## 5. 未解格（`?`）

無。本文件所有格子皆為 `file:line`、`—` 或 `(bin)`。

`(bin)` 的格子（smg gateway 的 6 格）不是「沒讀到」，是**沒有原始碼可讀**：
`smg` 套件在容器 `unieinfra-dev` 的
`/usr/local/lib/python3.12/dist-packages/smg/smg_rs.abi3.so`，Python 端只有
`cli.py` / `router.py` / `serve.py` 這幾支薄封裝。若要補這幾格，需要
`tokenspeed-smg` 的原始碼倉庫。
