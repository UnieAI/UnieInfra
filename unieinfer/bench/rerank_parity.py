"""Cross-encoder reranker parity: our scores against HuggingFace's.

A reranker is the one shape in this family whose output is a *score*, and a
score is easier to get plausibly wrong than an embedding: any monotone
distortion still ranks the documents the same way on an easy example, so a
run can look right while the model is wrong. This compares the raw logits
elementwise and also reports the induced ranking, because a broken model can
match one and not the other.

The pair's segment ids ride packed in the high bits of the token ids -- this
engine has no second per-token tensor to carry them -- so the tokenization
happens here and the request is submitted as ``input_ids``.

    docker exec unieinfra-dev bash -lc '
      source legacy/benchmark/scripts/sm120_env.sh
      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/roy/UnieInfra/python \
        python3 unieinfer/bench/rerank_parity.py --model <path>'
"""

import argparse

import numpy as np

QUERY = "What is the capital of France?"
DOCUMENTS = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany.",
    "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    "Cats are small domesticated carnivorous mammals.",
    "France is a country in Western Europe whose capital city is Paris, "
    "which sits on the river Seine and has been the seat of government "
    "since the tenth century.",
]

TOKEN_TYPE_SHIFT = 30


def tokenize_pairs(model: str):
    """(query, document) pairs, with segment ids packed into the token ids."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    packed, plain = [], []
    for document in DOCUMENTS:
        encoded = tokenizer(QUERY, document)
        ids = encoded["input_ids"]
        # Absent for models with type_vocab_size <= 1 (RoBERTa, GTE): those
        # separate the halves with their separator token instead.
        types = encoded.get("token_type_ids") or [0] * len(ids)
        packed.append([i | (t << TOKEN_TYPE_SHIFT) for i, t in zip(ids, types)])
        plain.append((ids, types))
    return tokenizer, packed, plain


def ours(model: str, packed, dtype: str) -> np.ndarray:
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = Engine(
        model=model,
        dtype=dtype,
        max_model_len=512,
        gpu_memory_utilization=0.25,
        max_num_seqs=8,
        disable_kvstore=True,
        attention_backend="triton",
    )
    sa = engine.server_args
    print(
        f"  resolved is_embedding={sa.is_embedding} "
        f"pooling_type={sa.pooling_type} normalize={sa.pooling_normalize}",
        flush=True,
    )
    out = engine.encode(input_ids=packed)
    if isinstance(out, dict):
        out = [out]
    scores = np.array([o["embedding"] for o in out], dtype=np.float32)
    engine.shutdown()
    return scores


def reference(model: str, plain, dtype: str) -> np.ndarray:
    import glob
    import runpy

    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForSequenceClassification

    # from_config, not from_pretrained: transformers materialises from meta
    # and never re-runs the constructor, so buffers registered
    # persistent=False arrive as whatever was in that memory. On
    # gte-multilingual-reranker that is ``position_ids``, and indexing the
    # rope cache with garbage is a device-side assert *in the reference*.
    # Same trap as the embedding bench's _build_reference; see
    # unieinfer/xref/15-encoder-embedding.md §6.7.
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    hf = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)
    state: dict = {}
    shards = sorted(glob.glob(f"{model}/*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under {model}")
    for shard in shards:
        state.update(load_file(shard))
    missing, unexpected = hf.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if "position_ids" not in m]
    if real_missing:
        raise RuntimeError(
            f"reference is missing {len(real_missing)} weights, e.g. {real_missing[:5]}"
        )
    hf = hf.to("cuda", getattr(torch, dtype)).eval()
    runpy.run_path("unieinfer/bench/encoder_embedding_parity.py")[
        "_assert_buffers_live"
    ](hf)

    scores = []
    with torch.no_grad():
        for ids, types in plain:
            batch = {
                "input_ids": torch.tensor([ids], device="cuda"),
                "attention_mask": torch.ones(
                    1, len(ids), dtype=torch.long, device="cuda"
                ),
            }
            if getattr(hf.config, "type_vocab_size", 0) > 1:
                batch["token_type_ids"] = torch.tensor([types], device="cuda")
            scores.append(hf(**batch).logits[0].float().cpu().numpy())
    return np.stack(scores).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument(
        "--max-rel-diff",
        type=float,
        default=0.05,
        help="Relative to the score's own magnitude. Raw logits here run to "
        "|10|, where one bf16 ulp is 0.0625 -- an absolute tolerance tight "
        "enough for a small score is unmeetable for a large one. 0.05 is set "
        "from measurement, not taste: bge-reranker-base lands at 3.8e-2 in "
        "bf16 and 3.4e-3 in fp32, so --dtype float32 is the discriminator "
        "between accumulated precision and a real defect.",
    )
    args = parser.parse_args()

    _, packed, plain = tokenize_pairs(args.model)
    print(f"pair lengths    {[len(p) for p in packed]}", flush=True)
    print("engine arm:", flush=True)
    a = ours(args.model, packed, args.dtype)
    print("reference arm:", flush=True)
    b = reference(args.model, plain, args.dtype)

    print(f"shape           {a.shape}")
    print(f"ours            {np.round(a.reshape(-1), 4).tolist()}")
    print(f"reference       {np.round(b.reshape(-1), 4).tolist()}")
    scale = np.maximum(np.abs(b), 1.0)
    rel = float((np.abs(a - b) / scale).max())
    print(f"max |diff|      {float(np.abs(a - b).max()):.3e}")
    print(f"max rel diff    {rel:.3e}")
    # The ranking is what a reranker is actually for, and it can survive a
    # wrong model on an easy query -- so it is reported next to the values,
    # not instead of them.
    rank_ours = np.argsort(-a.reshape(-1)).tolist()
    rank_ref = np.argsort(-b.reshape(-1)).tolist()
    print(f"ranking ours    {rank_ours}")
    print(f"ranking ref     {rank_ref}")

    if rel > args.max_rel_diff:
        raise SystemExit(f"FAIL: max rel diff {rel:.3e} > {args.max_rel_diff}")
    if rank_ours != rank_ref:
        raise SystemExit(f"FAIL: ranking differs {rank_ours} vs {rank_ref}")
    print(f"PASS: max rel diff {rel:.3e} <= {args.max_rel_diff}, ranking identical")


if __name__ == "__main__":
    main()
