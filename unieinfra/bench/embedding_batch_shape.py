"""Does a list-valued embedding request land in one prefill step?

Embedding traffic is many short independent texts, so the shape that matters
is a batch of them. The failure this catches is invisible in a single-request
latency number: if the frontend tokenizes and submits one item at a time,
each ``await`` hands control back to the event loop, the scheduler picks up
whatever has arrived, and the list gets split across prefill steps.

The probe is the comparison, not the absolute number: the same prompts
submitted as text against the same prompts pre-tokenized and submitted as
``input_ids``. Pre-tokenized input cannot be split by tokenization, so it is
the control. A ratio near 1.0 at every n means the text path is not being
split; a cliff means it is.

Measured on Qwen3-Embedding-8B (8-token prompts, p50 of 8 runs):

    n      before      after     control (input_ids)
     4    20.7 ms    20.7 ms     20.5 ms
     6    38.4 ms    21.3 ms     21.0 ms     <- the cliff, 1.80x
     8    39.1 ms    21.8 ms     21.3 ms     1.79x
    16    41.1 ms    23.7 ms     23.4 ms     1.73x
    32    44.8 ms    31.6 ms     30.6 ms     1.42x

Run:

    docker exec unieinfra-dev bash -lc '
      cd /workspace/roy/UnieInfra && source legacy/benchmark/scripts/sm120_env.sh
      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/roy/UnieInfra/python \
        python3 unieinfra/bench/embedding_batch_shape.py \
        --model /workspace/models/Qwen/Qwen3-Embedding-8B'
"""

import argparse
import time

DEFAULT_TEXT = "The capital of France is Paris."
BATCH_SIZES = (1, 2, 4, 6, 8, 12, 16, 24, 32)
REPS, WARMUP = 8, 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument(
        "--no-batch-tokenize",
        action="store_true",
        help="Turn the optimization off. The cliff should come back -- that "
        "is what says the switch and the attribution are both real.",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.15,
        help="Fail if the text path is this much slower than pre-tokenized "
        "input at any batch size -- that gap is a split, not compute.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine

    token_ids = AutoTokenizer.from_pretrained(args.model)(args.text).input_ids
    print(f"prompt tokens: {len(token_ids)}")
    print(f"batch_tokenize: {not args.no_batch_tokenize}")

    engine = Engine(
        model=args.model,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.60,
        max_num_seqs=args.max_num_seqs,
        embedding_batch_tokenize=not args.no_batch_tokenize,
        disable_kvstore=True,
        attention_backend="triton",
    )

    def bench(n: int, pre_tokenized: bool) -> float:
        call = (
            {"input_ids": [list(token_ids)] * n}
            if pre_tokenized
            else {"prompt": [args.text] * n}
        )
        for _ in range(WARMUP):
            engine.encode(**call)
        samples = []
        for _ in range(REPS):
            started = time.perf_counter()
            engine.encode(**call)
            samples.append((time.perf_counter() - started) * 1000)
        samples.sort()
        return samples[len(samples) // 2]

    print(f"{'n':>3} {'text ms':>9} {'pre-tokenized':>14} {'ratio':>7}")
    worst = 0.0
    for n in BATCH_SIZES:
        if n > args.max_num_seqs:
            continue
        text_ms, control_ms = bench(n, False), bench(n, True)
        ratio = text_ms / control_ms
        worst = max(worst, ratio)
        print(f"{n:3d} {text_ms:9.1f} {control_ms:14.1f} {ratio:6.2f}x")
    engine.shutdown()

    if worst > args.max_ratio:
        raise SystemExit(
            f"FAIL: text path is {worst:.2f}x the pre-tokenized path; a "
            "list request is being split across prefill steps"
        )
    print(f"PASS: worst ratio {worst:.2f}x <= {args.max_ratio}")


if __name__ == "__main__":
    main()
