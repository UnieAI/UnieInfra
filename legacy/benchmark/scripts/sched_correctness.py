#!/usr/bin/env python3
"""Greedy-output equivalence check for the E36 scheduling arms.

The prefill gate only changes *when* a request's prefill is scheduled, never
what is computed, so every arm must emit byte-identical greedy continuations.
This is the check that says so rather than assuming it: run it against each arm
under load (so the gate actually fires) and diff the JSON.

Usage:
    sched_correctness.py <base-url> <out.json> [concurrency]
"""

from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "The capital of France is",
    "Write a Python function that reverses a string:",
    "Explain in one sentence why the sky is blue:",
    "List the first eight prime numbers:",
    "def quicksort(arr):",
    "The three laws of motion are",
    "SELECT name FROM users WHERE",
    "Translate to French: 'good morning, how are you?'",
]


def complete(base_url: str, prompt: str, max_tokens: int = 64) -> str:
    """Request one greedy completion.

    Args:
        base_url: Server root, e.g. ``http://127.0.0.1:8011``.
        prompt: The prompt text.
        max_tokens: Tokens to generate.

    Returns:
        The generated text.
    """
    body = json.dumps(
        {
            "model": "bench-model",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["choices"][0]["text"]


def main() -> None:
    base_url, out = sys.argv[1], sys.argv[2]
    conc = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    # Repeat the prompt set so there is real queue pressure; the gate never
    # engages on an idle engine (`underloaded` release).
    work = [PROMPTS[i % len(PROMPTS)] for i in range(conc)]
    with ThreadPoolExecutor(max_workers=conc) as pool:
        texts = list(pool.map(lambda p: complete(base_url, p), work))
    result = {p: t for p, t in zip(work, texts)}
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    print(f"wrote {out} ({len(result)} distinct prompts)")


if __name__ == "__main__":
    main()
