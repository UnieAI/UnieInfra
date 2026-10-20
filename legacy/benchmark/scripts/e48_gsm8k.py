# SPDX-License-Identifier: Apache-2.0
"""E48 — GSM8K accuracy against a running OpenAI-compatible server.

fp8 KV cache is lossy.  E44 projects roughly +30% throughput from halving KV
bytes, and the owner needs the accuracy cost alongside that number, because
"+30% for free" and "+30% for two points of GSM8K" are different product
decisions.  Eyeballing a generation cannot tell those apart; this can.

Design notes that matter for the comparison being meaningful:

* **Paired.**  Every arm sees the identical prompt list in the identical order,
  greedy (`temperature=0`), so per-question outcomes can be compared directly.
  The aggregate accuracy of 1319 questions carries a ~1.0pp standard error,
  which is too coarse to resolve a 1-2pp regression on its own -- but the
  *paired* view (how many questions flipped right->wrong vs wrong->right) is
  far more sensitive, and is what `e48_summarize.py` reports.
* **Two extractors.**  `strict` requires the `#### <number>` form the few-shot
  exemplars demonstrate; `flexible` takes the last number in the completion.
  A KV-quantization regression that only perturbs formatting shows up as a
  strict-minus-flexible gap rather than as a reasoning loss, and that is worth
  being able to see.
* **No thinking mode.**  This is the raw completions endpoint with a 5-shot
  prefix, not the chat template, so Qwen3's reasoning block never opens and the
  measurement is of the model+cache, not of a chat scaffold.

Usage:
    python e48_gsm8k.py --host 127.0.0.1 --port 8448 --model bench-model \
        --out results.json [--limit 1319] [--concurrency 32]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _to_float(tok: str) -> float | None:
    try:
        return float(tok.replace(",", "").rstrip("."))
    except ValueError:
        return None


def gold_answer(ans: str) -> float | None:
    """GSM8K gold answers always end in `#### <number>`."""
    if "####" not in ans:
        return None
    return _to_float(ans.split("####")[-1].strip())


def extract_strict(text: str) -> float | None:
    """The `#### <number>` form the few-shot exemplars demonstrate."""
    if "####" not in text:
        return None
    tail = text.split("####")[-1]
    m = NUM_RE.search(tail)
    return _to_float(m.group()) if m else None


def extract_flexible(text: str) -> float | None:
    """Last number anywhere in the completion (lm-eval's flexible-extract)."""
    matches = NUM_RE.findall(text)
    return _to_float(matches[-1]) if matches else None


def build_prompts(limit: int, n_shot: int) -> tuple[list[str], list[float]]:
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")

    # Fixed exemplars: the first `n_shot` train rows.  Deterministic across arms
    # and across sessions, which is the only property that matters here.
    shots = "".join(
        f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n"
        for i in range(n_shot)
    )
    rows = test if limit <= 0 or limit >= len(test) else test.select(range(limit))
    prompts = [shots + f"Question: {r['question']}\nAnswer:" for r in rows]
    golds = [gold_answer(r["answer"]) for r in rows]
    return prompts, golds


def complete(url: str, model: str, prompt: str, max_tokens: int, retries: int = 3):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
            # The 5-shot prefix ends every exemplar with a blank line, so the
            # model signals "done" by starting the next `Question:`.
            "stop": ["Question:", "\n\n\n"],
        }
    ).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=900) as resp:
                d = json.load(resp)
            return d["choices"][0]["text"]
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"completion failed after {retries} attempts: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="bench-model")
    ap.add_argument("--limit", type=int, default=1319)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/completions"
    prompts, golds = build_prompts(args.limit, args.n_shot)
    print(
        f"[e48-gsm8k] {len(prompts)} questions, {args.n_shot}-shot -> {url}", flush=True
    )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        texts = list(
            pool.map(
                lambda p: complete(url, args.model, p, args.max_tokens),
                prompts,
            )
        )
    elapsed = time.time() - t0

    per_q = []
    n_strict = n_flex = 0
    for i, (text, gold) in enumerate(zip(texts, golds)):
        s, f = extract_strict(text), extract_flexible(text)
        ok_s = gold is not None and s is not None and abs(s - gold) < 1e-4
        ok_f = gold is not None and f is not None and abs(f - gold) < 1e-4
        n_strict += ok_s
        n_flex += ok_f
        per_q.append(
            {
                "i": i,
                "gold": gold,
                "strict": s,
                "flex": f,
                "ok_strict": bool(ok_s),
                "ok_flex": bool(ok_f),
                "text": text,
            }
        )

    n = len(prompts)
    out = {
        "label": args.label,
        "n": n,
        "n_shot": args.n_shot,
        "strict_acc": n_strict / n,
        "flexible_acc": n_flex / n,
        "elapsed_s": elapsed,
        "per_question": per_q,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(
        f"[e48-gsm8k] {args.label}: strict {n_strict}/{n} = {100 * n_strict / n:.2f}%  "
        f"flexible {n_flex}/{n} = {100 * n_flex / n:.2f}%  ({elapsed:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
