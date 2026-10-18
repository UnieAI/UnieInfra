#!/usr/bin/env python3
"""Temperature-0 agreement probe against the vLLM reference.

For each raw corpus prompt (no chat template) it:
  1. asks vLLM's /tokenize for the exact prompt token IDs vLLM will use, and
  2. generates temperature-0 text via /v1/completions with that same prompt.

The engine side is then fed token IDs [1] verbatim so both engines see
byte-identical inputs; only the greedy continuation is compared.

Usage (inside vllm-dsv4):
    python3 agree_vllm.py <prompts.json> <out.json> [max_new_tokens]
"""

from __future__ import annotations

import json
import sys
import urllib.request

COMP = "http://127.0.0.1:8889/v1/completions"
TOK = "http://127.0.0.1:8889/v1/tokenize"
MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 128


def post(url, body):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


def tokenize(prompt):
    try:
        j = post(TOK, {"model": "dsv4", "prompt": prompt})
    except Exception:
        j = post(TOK.replace("/v1/tokenize", "/tokenize"), {"prompt": prompt})
    return j["tokens"]


def complete(prompt):
    j = post(
        COMP,
        {
            "model": "dsv4",
            "prompt": prompt,
            "max_tokens": MAX_NEW,
            "temperature": 0,
        },
    )
    ch = j["choices"][0]
    return ch["text"], j.get("usage", {}).get("prompt_tokens"), ch.get("finish_reason")


def main():
    prompts = json.load(open(sys.argv[1]))
    result = {}
    for i, p in enumerate(prompts):
        ids = tokenize(p)
        text, pt, fin = complete(p)
        result[str(i)] = {"text": text, "ids": ids, "n_in": pt, "finish": fin}
        print(f"vllm[{i}]: n_in={pt} gen={len(text)} fin={fin} text0={text[:40]!r}")
    json.dump(result, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
