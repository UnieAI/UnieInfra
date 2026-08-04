# SPDX-License-Identifier: Apache-2.0
"""E40 — freeze a BFCL request set to JSONL so every arm sees identical traffic.

The vendored :class:`vllm.benchmarks.datasets.BFCLDataset` already does the hard
part: it downloads the per-category BFCL files, translates BFCL's schema dialect
(``"type": "dict"``) into real JSON Schema, and emits OpenAI-format
``tools``/``tool_choice``. We call it once, materialise the result, and every
later arm replays the same frozen list — so a difference between arms can never
be a difference in sampled prompts.

Output JSONL, one record per request::

    {"id": str, "messages": [...], "tools": [...], "tool_choice": "auto"}

Usage (inside the ``unieinfra-vllm-build`` container)::

    python3 e40_prepare_bfcl.py --tokenizer /workspace/models/Qwen/Qwen3-8B \
        --num-prompts 256 --out /workspace/roy/bench-data/bfcl_256.jsonl
"""

from __future__ import annotations

import argparse
import json

from vllm.benchmarks.datasets.datasets import BFCLDataset
from vllm.tokenizers import get_tokenizer

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--categories",
        default="simple,live_simple,multiple",
        help="comma-separated BFCL categories",
    )
    args = p.parse_args()

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=True)
    # dataset_split is required by the HuggingFaceDataset base but unused by
    # BFCLDataset, whose load_data() is a no-op (categories are resolved in
    # sample()). Pass a placeholder.
    ds = BFCLDataset(
        dataset_path=BFCL_REPO, dataset_split="train", random_seed=args.seed
    )
    reqs = ds.sample(
        tokenizer=tokenizer,
        num_requests=args.num_prompts,
        categories=[c for c in args.categories.split(",") if c],
    )

    n_tools = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(reqs):
            ov = r.request_overrides or {}
            rec = {
                "id": f"bfcl-{i}",
                "messages": r.chat_messages,
                "tools": ov.get("tools", []),
                "tool_choice": ov.get("tool_choice", "auto"),
                "prompt_len": r.prompt_len,
            }
            n_tools += len(rec["tools"])
            f.write(json.dumps(rec) + "\n")

    print(
        f"wrote {len(reqs)} requests to {args.out} "
        f"(mean {n_tools / max(1, len(reqs)):.2f} tools/request)"
    )


if __name__ == "__main__":
    main()
