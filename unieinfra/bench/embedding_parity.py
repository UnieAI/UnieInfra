"""Embedding parity: UnieInfra's pooled vectors against a HuggingFace reference.

A pooled embedding is the failure mode nothing downstream catches: a wrong
reduction, a wrong row, or a missing normalization all return a well-formed
unit vector of the right shape. Norms and shapes are therefore not evidence.
This script produces the evidence -- per-prompt cosine against the same
checkpoint run through `transformers`, plus the pairwise similarity structure,
which catches a systematic offset that a per-prompt cosine could still pass.

Run both arms on the same card, sequentially (the reference needs the GPU too):

    docker exec unieinfra-dev bash -lc '
      cd /workspace/roy/UnieInfra && source legacy/benchmark/scripts/sm120_env.sh
      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/roy/UnieInfra/python \
        python3 unieinfra/bench/embedding_parity.py \
        --model /workspace/models/Qwen/Qwen3-Embedding-8B'

Measured on Qwen3-Embedding-8B, bf16, sm_120, triton backend: cosine
0.99986 / 0.99988 / 0.99988, max elementwise |diff| 2.8e-3 -- bf16 agreement.
"""

import argparse

import numpy as np

PROMPTS = [
    "The capital of France is Paris.",
    "Cats are small domesticated carnivorous mammals.",
    "巴黎是法國的首都。",
]


def ours(model: str, prompts: list[str], **engine_kwargs) -> np.ndarray:
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = Engine(
        model=model,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.25,
        max_num_seqs=8,
        disable_kvstore=True,
        attention_backend="triton",
        **engine_kwargs,
    )
    sa = engine.server_args
    # The resolved values, not the requested ones: an embedding server turns
    # three flags off at startup and reading back the request would hide that.
    print(
        f"  resolved is_embedding={sa.is_embedding} "
        f"pooling_type={sa.pooling_type} normalize={sa.pooling_normalize}"
    )
    print(
        f"  resolved prefix_caching={sa.enable_prefix_caching} "
        f"chunked_prefill_size={sa.chunked_prefill_size} "
        f"overlap_disabled={sa.disable_overlap_schedule} "
        f"skip_kv_write={sa.embedding_skip_kv_write}"
    )
    out = engine.encode(prompt=prompts)
    if isinstance(out, dict):
        out = [out]
    vectors = np.array([item["embedding"] for item in out], dtype=np.float32)
    engine.shutdown()
    return vectors


def reference(model: str, prompts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    hf = AutoModel.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda")
    hf.eval()

    vectors = []
    with torch.no_grad():
        for text in prompts:
            # One prompt per forward: with no padding the last row is
            # unambiguous, so this cannot silently pool over a pad token.
            batch = tokenizer(text, return_tensors="pt").to("cuda")
            last = hf(**batch).last_hidden_state[0, -1]
            vectors.append(
                torch.nn.functional.normalize(last.float(), p=2, dim=-1).cpu().numpy()
            )
    return np.stack(vectors).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=["ours", "reference", "both"], default="both")
    parser.add_argument(
        "--no-skip-kv-write",
        action="store_true",
        help="Restore the prefill KV write. The vectors must be identical "
        "either way -- nothing reads those pages back, so if the answer "
        "moves, the assumption behind the optimization is wrong.",
    )
    parser.add_argument("--ours-npy", default=".embed_ours.npy")
    parser.add_argument("--reference-npy", default=".embed_ref.npy")
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.999,
        help="Fail below this. bf16 through two different stacks lands ~0.9999.",
    )
    args = parser.parse_args()

    if args.arm in ("ours", "both"):
        print("engine arm:")
        np.save(
            args.ours_npy,
            ours(
                args.model,
                PROMPTS,
                embedding_skip_kv_write=not args.no_skip_kv_write,
            ),
        )
    if args.arm in ("reference", "both"):
        print("reference arm:")
        np.save(args.reference_npy, reference(args.model, PROMPTS))
    if args.arm != "both":
        print(
            f"saved one arm; rerun with --arm {'reference' if args.arm == 'ours' else 'ours'}"
        )
        return

    a = np.load(args.ours_npy)
    b = np.load(args.reference_npy)
    cosine = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    unit_a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    unit_b = b / np.linalg.norm(b, axis=-1, keepdims=True)

    print(f"shape           {a.shape}")
    print(f"cosine vs HF    {[round(float(x), 6) for x in cosine]}")
    print(f"max |diff|      {float(np.abs(a - b).max()):.2e}")
    print(f"pairwise ours   {np.round(unit_a @ unit_a.T, 4).tolist()}")
    print(f"pairwise ref    {np.round(unit_b @ unit_b.T, 4).tolist()}")

    worst = float(cosine.min())
    if worst < args.min_cosine:
        raise SystemExit(f"FAIL: worst cosine {worst:.6f} < {args.min_cosine}")
    print(f"PASS: worst cosine {worst:.6f} >= {args.min_cosine}")


if __name__ == "__main__":
    main()
