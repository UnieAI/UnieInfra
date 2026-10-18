---
name: model-bring-up
description: "Read this skill when a new model or a new version of a supported model has to run in UnieInfra/tokenspeed — phrasings like '支援 DeepSeek V4.1', '做 X 的適配', 'port this model', 'vLLM 有我們沒有', 'new checkpoint 跑不起來'. It is the order to follow from 'checkpoint exists' to 'correct and gated': check whether upstream already has it, diff the reference implementation against the supported version, inventory hardcoded assumptions in the engine, measure whether a quantization-format change is lossy before choosing a kernel path, validate layer by layer against the reference, then truncated model, then full model with temperature-0 token agreement and an accuracy gate. Use model-structure-optimization and engine-decode-parity afterwards for speed."
---

# Model bring-up

Bring-up is mostly a reading problem. Most of the time is lost writing code
that already exists somewhere, or debugging the full model when one layer was
wrong.

## 1. Check upstream before writing anything

UnieInfra descends from upstream tokenspeed; other engines may also have it.

```sh
git fetch upstream
git log --oneline upstream/main -i --grep='<model>\|<feature>'
git ls-tree -r --name-only upstream/main | grep -i '<model>'
```

DeepSeek-V4.1 is the cautionary example: a port from vLLM was being planned
when `upstream/main` turned out to have shipped V4.1 a month earlier
(`models/deepseek_v41*.py`, CI on the 1017 checkpoint, Hopper-specific
indexer), along with the removal of the V4 path it replaced. The right move was
to merge upstream and optimize on top. Also survey vLLM, SGLang, TRT-LLM for
the same model (UNIEINFRA.md process step 2); their disagreements are the
findings, and their hardware fallbacks tell you what they could not make fast
(vLLM 0.31 on sm90 runs V4.1 dense layers as Marlin W8A16 and dequantizes the
whole ratio-1 compressed prefix per prefill chunk — both are opportunities).

If upstream has it, merge (see §6) instead of porting.

## 2. Diff the reference against the version you support

The model card's `inference/model.py` (or HF modeling file) is ground truth.
Diff it against the previous version's reference, then the configs:

```sh
diff old/inference/model.py new/inference/model.py
diff <(jq -S . old/config.json) <(jq -S . new/config.json)
```

List every architectural delta as a table: what changed, which engine module
it hits, whether a kernel changes. V4 → V4.1 produced: compress ratios 4/128 →
2/1; cross-layer sharing of compressed KV and top-k (`kv_source_layers`,
`index_source_layers`); a two-level candidate top-k; Engram n-gram embeddings;
lagged hyper-connection pre-mix and no `hc_head`; per-head q-norm removed;
window KV quantized whole incl. RoPE tail at block 32; compressed KV FP4
group-16 with E4M3 scales; FP8 weight blocks 128 → 32 (ue8m0); hash-routed
layers removed; image-token routing bias `bias_vl`; nested `text_config`;
draft MoE with its own expert count. Dump the checkpoint's tensor names, dtypes
and shapes too (read the safetensors headers) — names the loader silently drops
are a common silent bug (V4's `gate.bias_vl` was never loaded).

## 3. Inventory the engine's hardcoded assumptions

Grep the engine's implementation of the old version for every literal that the
diff touched: ratio values, layer kinds, `== 4`, `== 128`, block sizes, head
counts, `num_hash_layers`, kernel template constants (`HEAD_DIM=512`), cache
group ids, recipe ratio tables, quant block sizes in warmups. Rank by effort.
The dangerous ones are the silent folds (`max(1, ratio)` turning a new meaning
of `1` into "window only").

## 4. Measure before choosing a kernel path

When the new version changes a numeric format, measure on the real weights
whether mapping it onto an existing fast path is lossy, instead of assuming:

- V4.1 FP8 32×32 ue8m0 blocks → 128×128 (DeepGEMM sm90 recipe): ≤0.01% of
  elements changed, relative error ~1e-7 on sampled matrices. Lossless in
  practice, so no new GEMM kernel is needed.
- Count bytes per component (routed experts, dense, embedding tables, draft)
  from the safetensors headers before planning placement: V4.1-Flash is 286 GiB
  without Engram plus 189 GiB of Engram tables — it does not fit two 141 GB
  GPUs, and at TP4 the tables fit on GPU (~47 GiB/rank).

## 5. Validate bottom-up

1. **Per-layer parity** against the reference module on real weights (a few
   GB per layer), random tokens, seq ~2k: exact where the op is discrete (top-k
   indices, candidate masks, expert selection), bounded where FP8 rounding
   steps flip (attention ~1e-2 relative is normal; MoE routing amplifies it, so
   compare blocks with reference attention injected).
2. **Truncated model** (first N layers) through the real engine path: paging,
   chunked prefill, prefix hit, mixed batch, CUDA graphs, speculative decode.
3. **Full model**: temperature-0 token agreement with a reference engine on a
   fixed prompt set (report agreed tokens / total, not "looks fine"), plus the
   accuracy gate (GSM8K ≥ 0.90 for DeepSeek-class; full set on checkpoint
   change). The corrupt arm can look faster — never benchmark before this.

## 6. Merging upstream instead of porting

- Work in a separate `git worktree`; never in a tree with someone's
  uncommitted work.
- Partition conflicts by area (scheduler C++, engine/execution,
  attention/models, kernels) and resolve in parallel; only one actor touches
  the index.
- Upstream's architecture wins. For each fork commit touching a conflicted
  file decide: obsolete upstream (cite upstream file:line), re-implement
  minimally on the new structure, or follow-up. Write the decision down per
  commit; that ledger is the merge report.
- Expect upstream to delete things the fork still serves (the V4.1 merge
  removed DeepSeek-V4, MHA mixed batching, the fork's Gemma-4 path). List them
  for the owner before merging back.
- The runtime image must match upstream's pinned kernel packages (flashinfer,
  flash_mla, deep_gemm); an old image fails at import, which looks like a merge
  bug.
- The model registry imports every model module: one module that no longer
  imports breaks startup for every model. Run the registry import as the first
  test after a merge.

## 7. Bring-up on a smaller GPU than upstream tested

Upstream may only have run the model on bigger parts (V4.1: B200/B300 at 180+
GB, H20 at TP8). On H200 TP4 it OOMed during weight post-processing, with or
without host-resident Engram tables. Log allocated memory before every module's
post-load hook (tokenspeed: `TOKENSPEED_TRACE_POSTPROCESS_MEM=1`) and look for
growth that the hook's own allocation does not explain. The V4.1 cause: the
checkpoint loader called the MoE re-layout while its local name->Parameter map
was still alive, so every layer's original experts stayed resident — twice the
expert weights at peak (~63 GiB per rank). The isolated hook freed correctly;
only the in-process frame held the references. Regression test: replace the hook
with one that drops the module's reference and assert via `weakref` + `gc` that
nothing else keeps the old Parameter alive.
