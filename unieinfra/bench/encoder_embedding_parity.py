"""Encoder-only embedding parity: BERT / RoBERTa-XLM-R / GTE-v1.5 vs HuggingFace.

Separate from ``embedding_parity.py`` because that one is hard-wired to a
decoder's contract -- LAST pooling, always normalized. The encoder family is
where those two assumptions stop holding together: BGE and GTE pool CLS,
multilingual-e5 pools MEAN, and GTE-v1.5 ships **no** Normalize module, so a
server that always L2-normalizes returns the wrong vector for it while still
returning a perfectly well-formed one.

The pooling mode and normalization are therefore passed on the command line
rather than read from the checkpoint. That makes this a test of two things at
once: the model implementation, and whether the engine's own
``resolve_pooling_config`` reached the same answer the checkpoint's
``1_Pooling/config.json`` and ``modules.json`` state. The engine's resolved
values are printed so a disagreement is visible rather than cancelling out.

Usage (one card, both arms sequentially):

    docker exec unieinfra-dev bash -lc '
      source legacy/benchmark/scripts/sm120_env.sh
      CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/roy/UnieInfra/python \
        python3 unieinfra/bench/encoder_embedding_parity.py \
        --model <path> --pooling cls --normalize'
"""

import argparse

import numpy as np

PROMPTS = [
    "The capital of France is Paris.",
    "Cats are small domesticated carnivorous mammals.",
    "A much longer sentence, written so that the batch is genuinely ragged "
    "and the varlen cu_seqlens path has to place more than one distinct "
    "sequence length correctly rather than accidentally agreeing on a "
    "uniform one.",
    "巴黎是法國的首都。",
    # Long enough that ModernBERT's local layers actually clip. Its window is
    # 64 tokens each side, so anything under ~129 tokens leaves every token
    # inside every other token's window and the local and global layers
    # differ only by their rope theta -- a sliding-window bug would score a
    # clean 0.9999 on the four short prompts above.
    " ".join(
        f"Sentence number {i} carries its own distinct content so that "
        f"positions far apart cannot be confused with positions nearby."
        for i in range(12)
    ),
]


def ours(
    model: str, prompts: list[str], dtype: str = "bfloat16", **engine_kwargs
) -> np.ndarray:
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = Engine(
        model=model,
        dtype=dtype,
        max_model_len=512,
        gpu_memory_utilization=0.25,
        max_num_seqs=8,
        disable_kvstore=True,
        attention_backend="triton",
        **engine_kwargs,
    )
    sa = engine.server_args
    print(
        f"  resolved is_embedding={sa.is_embedding} "
        f"pooling_type={sa.pooling_type} normalize={sa.pooling_normalize}",
        flush=True,
    )
    out = engine.encode(prompt=prompts)
    if isinstance(out, dict):
        out = [out]
    vectors = np.array([item["embedding"] for item in out], dtype=np.float32)
    engine.shutdown()
    return vectors


def _assert_buffers_live(hf) -> None:
    """Fail if any buffer looks like uninitialised memory.

    Non-persistent buffers are the reference's weak point (see
    ``_build_reference``), and every way they go wrong is silent: a zeroed
    rotary cache makes attention position-blind, a garbage ``position_ids``
    indexes that cache at random rows. Both keep the output's shape and
    roughly its norm, so only an explicit check catches them.
    """
    import torch

    # Buffers whose correct value *is* all zeros. BERT's ``token_type_ids``
    # is ``torch.zeros(1, max_position_embeddings)`` -- segment 0 for every
    # token, which is what a single-sequence embedding request means. Zero
    # there is a value, not a missing one.
    legitimately_zero = ("token_type_ids",)

    for name, buf in hf.named_buffers():
        f = buf.float()
        if torch.isnan(f).any() or torch.isinf(f).any():
            raise RuntimeError(f"reference buffer {name!r} contains nan/inf")
        if any(tail in name for tail in legitimately_zero):
            continue
        if float(f.abs().max()) == 0.0:
            raise RuntimeError(
                f"reference buffer {name!r} is all zeros -- almost certainly "
                "an uninitialised non-persistent buffer, not a real value"
            )
        # A real cos/sin cache, position table or inv_freq is O(1)-O(1e4).
        if float(f.abs().max()) > 1e9:
            raise RuntimeError(
                f"reference buffer {name!r} holds |{float(f.abs().max()):.3e}|, "
                "which is uninitialised memory rather than a value this "
                "model computes"
            )


def _build_reference(model: str, dtype):
    """Construct the reference so its non-persistent buffers actually exist.

    ``AutoModel.from_pretrained`` materialises the module tree from meta and
    fills it from the checkpoint. Buffers registered with
    ``persistent=False`` are in no checkpoint by definition, and nothing
    re-runs the constructor that computed them, so they arrive as whatever
    was in that memory. On GTE-v1.5 (transformers 5.12) all three come back
    dead: ``rotary_emb.cos_cached``/``sin_cached`` all zeros, and
    ``rotary_emb.inv_freq`` and ``embeddings.position_ids`` filled with
    values like -1.07e30.

    The visible effect is nil -- ``cos = sin = 0`` makes
    ``apply_rotary_pos_emb`` return ``q*0 + rotate_half(q)*0``, zeroing every
    query and key, and attention degenerates into a uniform average over V.
    The embedding keeps its shape and very nearly its norm. Scoring a
    *correct* implementation against it gives ~0.3 cosine and reads as a bug
    in the implementation.

    ``from_config`` runs the real constructor, so the buffers are computed;
    the weights are then loaded explicitly. ``_assert_buffers_live`` is the
    check that this actually worked, rather than the next silent variant.
    """
    import glob

    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    hf = AutoModel.from_config(config, trust_remote_code=True)

    state: dict[str, torch.Tensor] = {}
    shards = sorted(glob.glob(f"{model}/*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under {model}")
    for shard in shards:
        state.update(load_file(shard))

    missing, unexpected = hf.load_state_dict(state, strict=False)
    # ``position_ids`` is a buffer, and ``pooler.*`` is the classification
    # head an embedding checkpoint does not ship. Neither absence means the
    # reference is incomplete.
    ignorable = ("position_ids", "pooler.")
    real_missing = [m for m in missing if not any(i in m for i in ignorable)]
    if real_missing:
        raise RuntimeError(
            f"reference is missing {len(real_missing)} weights, e.g. "
            f"{real_missing[:5]}"
        )
    if unexpected:
        print(f"  reference ignored {len(unexpected)} extra keys: {unexpected[:3]}")

    hf = hf.to("cuda", dtype).eval()
    _assert_buffers_live(hf)
    return hf


def reference(
    model: str,
    prompts: list[str],
    pooling: str,
    normalize: bool,
    dtype: str = "bfloat16",
) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    hf = _build_reference(model, getattr(torch, dtype))

    vectors = []
    with torch.no_grad():
        for text in prompts:
            # One prompt per forward: no padding, so the reduction cannot
            # silently include a pad row -- which is the exact failure MEAN
            # pooling has when a reference is batched carelessly.
            batch = tokenizer(text, return_tensors="pt").to("cuda")
            states = hf(**batch).last_hidden_state[0].float()
            if pooling == "cls":
                pooled = states[0]
            elif pooling == "mean":
                pooled = states.mean(dim=0)
            elif pooling == "last":
                pooled = states[-1]
            else:
                raise ValueError(pooling)
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            vectors.append(pooled.cpu().numpy())
    return np.stack(vectors).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=["ours", "reference", "both"], default="both")
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean", "last"],
        required=True,
        help="What the checkpoint's 1_Pooling/config.json says. Passed "
        "explicitly so a wrong auto-resolution shows up as a failure "
        "instead of being mirrored by the reference.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="Whether modules.json lists a Normalize module. GTE-v1.5 does "
        "not; BGE and e5 do.",
    )
    parser.add_argument("--ours-npy", default=".enc_ours.npy")
    parser.add_argument("--reference-npy", default=".enc_ref.npy")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Both arms. float32 separates a precision gap from a real one: "
        "a cosine that stays put in fp32 is a bug, one that closes was noise.",
    )
    args = parser.parse_args()

    if args.arm in ("ours", "both"):
        print("engine arm:", flush=True)
        np.save(args.ours_npy, ours(args.model, PROMPTS, dtype=args.dtype))
    if args.arm in ("reference", "both"):
        print("reference arm:", flush=True)
        np.save(
            args.reference_npy,
            reference(args.model, PROMPTS, args.pooling, args.normalize, args.dtype),
        )
    if args.arm != "both":
        other = "reference" if args.arm == "ours" else "ours"
        print(f"saved one arm; rerun with --arm {other}")
        return

    a = np.load(args.ours_npy)
    b = np.load(args.reference_npy)
    cosine = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    unit_a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    unit_b = b / np.linalg.norm(b, axis=-1, keepdims=True)

    print(f"shape           {a.shape}")
    print(f"cosine vs HF    {[round(float(x), 6) for x in cosine]}")
    print(f"norms ours      {np.round(np.linalg.norm(a, axis=-1), 4).tolist()}")
    print(f"norms ref       {np.round(np.linalg.norm(b, axis=-1), 4).tolist()}")
    print(f"max |diff|      {float(np.abs(a - b).max()):.2e}")
    print(f"pairwise ours   {np.round(unit_a @ unit_a.T, 4).tolist()}")
    print(f"pairwise ref    {np.round(unit_b @ unit_b.T, 4).tolist()}")

    worst = float(cosine.min())
    if worst < args.min_cosine:
        raise SystemExit(f"FAIL: worst cosine {worst:.6f} < {args.min_cosine}")
    print(f"PASS: worst cosine {worst:.6f} >= {args.min_cosine}")


if __name__ == "__main__":
    main()
