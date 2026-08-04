#!/usr/bin/env python3
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Greedy output capture / comparison for speculative-decoding equivalence.

Speculative decoding must be **output-preserving under greedy sampling**: the
verified token stream has to match plain autoregressive greedy token for token.
This script captures one JSON of greedy generations from a running
OpenAI-compatible server, and (with ``--compare``) diffs it against a previously
captured baseline, reporting the first divergent token index per prompt.

Usage
-----
Capture::

    python3 spec_equivalence_check.py --base-url http://host:port \
        --model bench-model --out baseline.json

Compare::

    python3 spec_equivalence_check.py --base-url http://host:port \
        --model bench-model --out mtp.json --compare baseline.json

Arguments
---------
--base-url : OpenAI-compatible server root (no trailing ``/v1``).
--model    : served model name.
--out      : path to write this run's capture JSON.
--compare  : optional baseline capture JSON to diff against.
--max-tokens : greedy tokens to generate per prompt (default 128).

Returns
-------
Exit code 0 when every prompt matches (or when only capturing); 1 on any
divergence, which is a speculative-decoding correctness bug.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the n-th Fibonacci number:",
    "In one paragraph, explain why the sky appears blue.",
    "List the first ten prime numbers, separated by commas:",
    "Q: If a train travels 120 km in 1.5 hours, what is its average speed?\nA:",
    "Translate to French: 'The weather today is beautiful and warm.'",
    "def quicksort(arr):",
    "The three laws of thermodynamics are:",
]


def generate(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Run one greedy completion and return its text plus token ids."""
    body = json.dumps(
        {
            "model": model,
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
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.load(resp)
    choice = payload["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return {
        "prompt": prompt,
        "text": choice["text"],
        "tokens": logprobs.get("tokens") or [],
        "meta": payload.get("meta_info", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="bench-model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    results = [generate(args.base_url, args.model, p, args.max_tokens) for p in PROMPTS]
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"[equiv] captured {len(results)} greedy generations -> {args.out}")

    if not args.compare:
        for r in results:
            print(f"  {r['prompt'][:40]!r} -> {r['text'][:90]!r}")
        return 0

    with open(args.compare) as fh:
        baseline = json.load(fh)

    mismatches = 0
    for base, cur in zip(baseline, results):
        if base["text"] == cur["text"]:
            print(f"  MATCH  {base['prompt'][:48]!r}")
            continue
        mismatches += 1
        # first differing character index, plus a token-level index when the
        # server returned per-token strings.
        idx = next(
            (i for i, (a, b) in enumerate(zip(base["text"], cur["text"])) if a != b),
            min(len(base["text"]), len(cur["text"])),
        )
        tok_idx = None
        if base["tokens"] and cur["tokens"]:
            tok_idx = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(base["tokens"], cur["tokens"]))
                    if a != b
                ),
                min(len(base["tokens"]), len(cur["tokens"])),
            )
        print(f"  DIFFER {base['prompt'][:48]!r} char@{idx} token@{tok_idx}")
        print(f"    baseline: {base['text'][max(0, idx - 30):idx + 60]!r}")
        print(f"    current : {cur['text'][max(0, idx - 30):idx + 60]!r}")

    print(f"[equiv] {len(results) - mismatches}/{len(results)} identical")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
