# 讓 UnieInfra 服務 embedding 模型（encoder-only / pooled 輸出）

日期：2026-08-21。範圍：**能力缺口**，不是效能題。對照文件：
`unieinfra/xref/15-encoder-embedding.md`（四方 survey，70/70 引用通過 `check_xref.py`）。

## 1　做到哪裡

**引擎路徑完成並驗證**：`EmbeddingReqInput` → tokenize → C++ scheduler → 單次 prefill
→ pooling → `BatchEmbeddingOut` → client，全程走通。

**HTTP `/v1/embeddings` 沒有做，而且不在這個 repo 裡**，見 §5。

驗證（Qwen3-Embedding-8B、bf16、sm_120、triton backend、GPU 0）：

| 檢查 | 結果 |
|---|---|
| 每條 prompt 與 HF `transformers` 的餘弦 | **0.999864 / 0.999882 / 0.999875** |
| 最大逐元素 \|diff\| | 2.78e-3（bf16 量級） |
| 跨句相似度結構（我們 / 參考） | 中英「巴黎」兩句 0.8034 / 0.8035；與貓那句 0.284 / 0.283 |
| 形狀、範數 | (3, 4096)、單位範數 |
| 生成模型回歸（Qwen3-0.6B） | 文字連貫、16 tokens、`is_embedding=None`、prefix cache 仍開、overlap 仍開 |
| CPU 單元測試 | `test/runtime/test_embedding_pooling.py` 16 passed |

**範數和形狀不是證據**。錯的 reduction、錯的 row、漏掉 normalize，三者都會回傳一個
形狀正確的單位向量。所以驗證用的是對同一份 checkpoint 跑 HF 的逐句餘弦**加上**跨句
相似度矩陣——後者才抓得到「整體偏移但每句餘弦仍高」這種情況。
重跑：`unieinfra/bench/embedding_parity.py`。

## 2　設計選擇（每一條都對得上 survey 的某一格）

| 決定 | 跟誰同一邊 | 理由 |
|---|---|---|
| pooling 抽成獨立層（`runtime/layers/pooler.py`），模型只持有 `pooler` | vLLM | 換 pooling 策略不必動模型檔；TRT-LLM 把 CLS 寫死在模型裡，反過來 |
| server 級的 embedding 模式，不是 per-request | 四家都是 | 沒有任何一家在同一個 server 同時服務兩種模式 |
| pooling 請求**不進 cache coordinator 的前綴比對** | vLLM（0 group）、TRT-LLM（不建 manager） | arena 的 `AcquireBlocks(group_id,…)` 一定要 group_id；「進去綁一個假 group」是錯的方向 |
| LAST pooling 直接重用 `ForwardContext.gather_ids` | — | 那就是 logits processor 用的「每個請求最後一個 token」索引，一份真相 |
| 啟動期強制關掉三個東西（見 §3） | vLLM / SGLang 都關前兩個 | 第三個是我們自己的結構問題，見 §4 |

## 3　啟動期被強制關掉的東西

`ModelConfig._resolve_embedding_server_args` 會改寫 server args 並各印一行說明：

1. **prefix caching → off**。pooling 請求不會回讀自己的 KV。
2. **chunked prefill → 開到整段 prompt 一定塞得下**（不是關掉）。這棵樹**沒有**
   SGLang 的 `-1 = disabled` 慣例，`chunked_prefill_size` 直接決定 input buffer 大小；
   第一版照抄 `-1` 得到 `Trying to create tensor with negative dimension -1`。
   若 `--max-prefill-tokens < context_len` 就直接拒絕啟動，並在訊息裡給操作者該用的數字。
3. **overlap schedule → off**。理由見 §4。
4. `--speculative-algorithm` 直接拒絕：沒有 token 可以 draft。

另外把 `is_embedding` / `pooling_type` / `pooling_normalize` 三個 tri-state **回填成
解析後的值**，這樣從執行中的 server 讀回去看到的是結果不是請求。

## 4　過程中撞到的三個真問題

### 4.1　C++ scheduler 不用 `max_new_tokens` 當停止條件

`MaxNewTokens()` 在整個 C++ 樹裡只出現在准入預留一處
（`tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp:316`）。請求要停，只能靠
Python 送 Finish 事件。而 **overlap 排程會在前一步的結果提交之前就排好下一步**，
所以一個「prefill 完就結束」的 pooling 請求，會被多排一個 decode step。

症狀是 `pooling requires ForwardContext.gather_ids ... forward_mode=2 bs=3
num_extends=0`。這個數字是我把診斷資訊寫進例外訊息後跑一次拿到的——猜了兩輪都不對，
把狀態印出來一次就結案。

修法是關掉 overlap：pooling server 每個請求只有一步，overlap 本來就沒有東西可以重疊。

### 4.2　繼承來的 embedding scaffolding 是死碼

`io_struct.py` 早就有 `EmbeddingReqInput` / `TokenizedEmbeddingReqInput` /
`BatchEmbeddingOut`，前端 `output_processor` 也早就有 `{"embedding": ...}` 的分支。
但 `is_generation_model()` 是無條件 `return True` 的樁，所以：

- `input_processor.py` 的 embedding 閘門永遠 raise，
- `model_runner.py` 的 `get_embedding=True` 永遠不會被設，
- `prefill_graph.py` 的 `not is_generation` 分支是死的。

**scaffolding 存在不等於路徑活著。** 這與 `60-quant.md` 記的 w4a16 死碼是同一個形狀。

### 4.3　pooling checkpoint 沒有 lm_head

Qwen3-Embedding-8B 的權重是 `embed_tokens.weight` + `layers.*`，**沒有 `lm_head`**。
現在 pooling 模型不建 lm_head 與 logits processor（照 `encoder_only` 的既有作法），
順帶省下一個沒有權重會去填的 vocab 尺寸投影。連帶要處理的是
`dp_sampling` 從 logits processor 推導 topology，對 pooling 模型必須整段跳過。

## 5　沒做的部分（範圍邊界，不是遺漏）

**`/v1/embeddings` 走不通，因為 HTTP 面不在這個 repo。** `tokenspeed serve` 的
OpenAI 介面由外部 wheel `tokenspeed-smg`（Rust gateway）加上
`tokenspeed-smg-grpc-servicer` 提供，後者的檔頭明寫：

> The proto field set is intentionally minimal — generative LLM serving, …;
> **no Embed** / PD-disaggregated / LoRA / hidden states / classifier outputs.

所以要對外提供 `/v1/embeddings`，需要在那兩個 repo 動 proto 與 servicer，這一輪沒有碰。
現在可用的入口是 in-process 的 `Engine.encode(prompt=...)`。

其他明確沒做的：encoder-only 架構（BERT / GTE / BGE）尚未支援——目前這條路是
「因果 LLM 當 embedder」，KV 照常配、只是不 decode。真正的 encoder-only 需要
`15-encoder-embedding.md` §3.2 的零 group 路徑，那是下一步。Whisper 的 cross-attention
（D24）也還沒動，它會需要 C++ 的第三種 cache family。

## 6　改了哪些檔

```
新增  python/tokenspeed/runtime/layers/pooler.py
新增  test/runtime/test_embedding_pooling.py          16 tests, CPU only
新增  unieinfra/bench/embedding_parity.py             HF 對照，含 --min-cosine 門檻
改    configs/model_config.py        pooling 偵測、能力宣告、啟動期守門
改    utils/server_args.py           --is-embedding / --pooling-type / --pooling-normalize
改    models/base/causal_lm.py       resolve_pooler / pool()，pooling 時不建 lm_head
改    execution/model_executor.py    _forward_step 的 pooling 分支、D2H 分流、dp_sampling
改    execution/types.py             PoolerForwardResult / output_embeddings
改    execution/cuda_graph_wrapper.py  pooling 時停用 decode graph
改    engine/generation_output_processor.py  post_process_embedding_forward_op
改    engine/request_handler.py      handle_embedding_request（不走 PD/session 那條）
改    engine/input_processor.py      兩個方向的請求閘門
改    engine/output_processor.py     EmbeddingReqInput 沒有 stream 欄位
改    entrypoints/engine.py          Engine.encode
```

## 7　量測候選（尚未量）

`15-encoder-embedding.md` §6 列了三個。與這一輪最相關的是第二個：TRT-LLM 在 HTTP 層
用 5ms hold window 的 dynamic batcher 聚合 embedding 流量，而那個 5ms 是抄 Triton 的
慣例、不是量出來的。要抄結構可以，**不要抄那個數字**——自己掃 1/5/20ms × 併發。
第一個（SGLang 的 embedding KV 池是不是純浪費）同樣值得量，因為我們選了不配池的那一邊，
應該有一個數字支撐這個選擇而不是只有結構論證。
