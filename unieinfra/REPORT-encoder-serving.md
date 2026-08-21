# UnieInfra 服務 embedding 與 Whisper：能力、優化、與擋住的路

日期：2026-08-21。對照文件：`unieinfra/xref/15-encoder-embedding.md`（四方 survey +
所有實測，72/72 引用通過 `check_xref.py`）。

> 這份取代原本的 `REPORT-embedding-serving.md`。那一版寫在 Whisper 完成之前，
> 裡面說 cross-attention「會需要 C++ 的第三種 cache family」——**事後證明不需要**，
> C++ 一行未動。

## 1　交付狀態

| 能力 | 狀態 | 證據 |
|---|---|---|
| Embedding（pooled 向量） | **可服務** | 對 HF 逐句餘弦 0.999864–0.999883 |
| Whisper（encoder-decoder ASR） | **可服務** | 轉譯與 HF **逐字相同**、token id 相同 |
| `/v1/embeddings`、`/v1/audio/transcriptions` | **端點未做** | 引擎端已備妥；見 §5 |

「端點未做」不等於不支援：引擎路徑完整且驗證過，缺的是 repo 外那兩個 wheel 的
proto 與 handler。

重跑證據：`unieinfra/bench/embedding_parity.py`、`whisper_transcribe_parity.py`。

## 2　支援哪些模型

Embedding 走的是「**因果 LLM 當 embedder**」。三道關卡：

1. 架構在 registry 裡（38 個）
2. 模型類別繼承 `BaseCausalLM`（pooling 掛在它的 `pool()`）——38 個裡 **23 個**符合：
   Qwen2 / Qwen3 / Qwen3-MoE / Qwen3.5、Llama、DeepSeek V3/V3.2/V4、Gemma4、
   GLM-MoE-DSA、GPT-OSS、MiniMax M2/M3、LongCat 等
3. checkpoint 宣告 pooling（`1_Pooling/config.json` 的 last/cls/mean），或 operator
   明給 `--pooling-type`

不符合的 15 個是 NextN／draft head，以及包一層 `language_model` 的複合多模態模型
（Whisper、Qwen3-ASR、Kimi、Inkling）。

**尚未支援 encoder-only 家族**（BERT / GTE / BGE / RoBERTa / ModernBERT）：那需要
非因果的模型類別加上零 KV group 的路徑（xref §3.2）。這是真正的缺口，不是設定問題。

## 3　設計上的分歧點，以及我們站哪一邊

| 決定 | 跟誰同一邊 | 理由 |
|---|---|---|
| pooler 獨立成層、runner 呼叫 | vLLM | 換策略不動模型檔 |
| pooling 請求不進 cache coordinator | vLLM（0 group）、TRT-LLM（不建 manager） | arena 入口一定要 group_id；綁假 group 是錯方向 |
| cross KV 在 paged arena 之外 | vLLM、TRT-LLM | 寫一次、不成長、不前綴比對；**實作成 per-slot buffer，不需要新 cache family，C++ 未動** |
| Whisper 的 paged 層數 = `decoder_layers` | 都不是 | SGLang 配 `encoder_layers + 2*decoder_layers`（turbo 是 40 個 slot，32 個永不索引） |

## 4　優化：三項有效、兩項此路不通、一項判讀撤回

所有優化都有開關，可獨立 A/B（`ServerArgs`，會出現在 resolved config）：

```
--embedding-batch-tokenize / --no-...      預設 on
--embedding-skip-kv-write / --no-...       預設 on（目前無效，見下）
--cross-attn-path {auto,padded,gather}     預設 auto
--pooling-prefill-graph / --no-...         預設 on
```

**有效**

| 項目 | 結果 |
|---|---|
| **Prefill graph 守門過寬**（對每個模型都關） | dense 單請求 **17.3 → 8.71 ms（2.0x）** |
| **讓 pooling 模型也能捕捉 prefill graph** | embedding 單請求 **19.75 → 14.84 ms（1.33x）**；cosine 與 eager 相同 |
| Whisper cross-attention 去掉每步 gather | 高佔用 **1.79x**（turbo）／**1.84x**（large-v3）；低佔用 0.87x，故依佔用率選路 |
| Embedding 列表請求整批 tokenize | **最高 1.80x**；關掉開關斷崖回來且探針 FAIL（雙向驗證） |

cross-attention 那項要注意：**只看 kernel 是 7.55x，看整個 block 只有 1.79x**。
小批次時 block 是 launch-bound，該引用的是 block 的數字。

**此路不通（原因已定位，附日期）**

- **省掉 prefill 的 KV 寫入**：kernel plan 在 bf16+triton 選 prewrite（先寫頁再讀回），
  不是 postwrite。**被守門擋下來的，不是被量出來的**——沒有那個檢查，會回傳範數正確
  但數值錯誤的向量。
- **fp8 cross KV**：`FP8 is only supported on SM100 (compute capability 10.x)`。
  連同 `page_table`、`gather_kv_indices`，這是 sm_120 上第三道關著的門。

**判讀撤回**：先前記的「prefill graph 只省 0.35 ms」不算負面結果——那一輪
graph 從未被捕捉（日誌行沒出現），所以狀態是**不確定**。

## 5　端點：缺什麼

OpenAI 介面由 repo 外的 `tokenspeed-smg`（Rust gateway）加
`tokenspeed-smg-grpc-servicer` 提供，後者檔頭明寫：

> The proto field set is intentionally minimal — generative LLM serving, …;
> **no Embed** / PD-disaggregated / LoRA / hidden states / classifier outputs.

要對外提供 `/v1/embeddings` 與 `/v1/audio/transcriptions`，需在那兩個 repo 動 proto 與
servicer。目前可用入口是 in-process 的 `Engine.encode(...)` 與帶 audio 的
`GenerateReqInput`。

## 6　下一步，按證據強度

1. **encoder-only 架構**（BERT/GTE/BGE）：真正的能力缺口。
2. **hold-window 聚合**（TRT-LLM 的 5ms）：治的是不同請求之間，與 §4 第 2 項不重疊；
   要等端點做起來、有真並發流量才量得準。那個 5ms 是慣例常數，**抄結構不要抄數字**。

（原本列在此處第 1 項的「讓 pooling 模型也能用 prefill graph」**已完成**，見 §4 與
xref §6.6：排除條件從 `not model_runner.is_generation` 收窄成
`--no-pooling-prefill-graph` 加「logits processor 與 pooler 皆無」，`replay()` 分支到
pooler。）
