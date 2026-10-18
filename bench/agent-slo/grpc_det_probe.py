#!/usr/bin/env python3
"""Drive the engine gRPC scheduler directly for a temperature-0 determinism check.

Bypasses the OpenAI gateway (which the smg Rust binary materializes) so the
sample exercises the exact same engine core as the DSPARK benchmark. Sends the
same text prompt N times and reports longest-common-prefix between runs.

Usage (in engine container):
    DET_HOST=127.0.0.1:<grpc-port> DET_PROMPT_IDX=5 DET_REPS=6 \
        python3 bench/agent-slo/grpc_det_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import uuid

sys.path.insert(0, "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/inference")
import smg_grpc_proto.generated.tokenspeed_scheduler_pb2 as pb  # noqa: E402
import smg_grpc_proto.generated.tokenspeed_scheduler_pb2_grpc as pbg  # noqa: E402
import grpc  # noqa: E402

HOST = os.environ.get("DET_HOST", "127.0.0.1:37189")
PROMPT_IDX = int(os.environ.get("DET_PROMPT_IDX", "5"))
REPS = int(os.environ.get("DET_REPS", "6"))
MAX_TOKENS = int(os.environ.get("DET_MAX_TOKENS", "64"))
CORPUS = os.environ.get(
    "DET_CORPUS",
    "/workspace/roy/UnieInfra/bench/agent-slo/corpora/swe_agent_nebius_sess16x16_b.jsonl",
)


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def run(stub, input_ids, idx):
    req = pb.GenerateRequest()
    req.request_id = f"det-{idx}-" + uuid.uuid4().hex[:8]
    req.tokenized.input_ids.extend(input_ids)
    req.sampling_params.max_new_tokens = MAX_TOKENS
    req.sampling_params.temperature = 0.0
    req.sampling_params.skip_special_tokens = True
    req.stream = True
    outs = []
    chunks = 0
    for resp in stub.Generate(req, timeout=600):
        chunks += 1
        for t in resp.chunk.token_ids:
            outs.append(t)
        if resp.HasField("complete"):
            # complete echoes the last chunk only; tokens come from chunk stream.
            pass
    return outs, chunks


def main():
    rows = [json.loads(l) for l in open(CORPUS)]
    prompt = rows[PROMPT_IDX]["prompt"]
    probe_ids = json.loads(os.environ.get("DET_INPUT_IDS", "null"))
    if probe_ids is not None:
        input_ids = probe_ids
    else:
        # The gateway materializes tokenization; the engine gRPC takes raw IDs.
        # Reuse HuggingFace tokenizer ranks so the sequence is self-contained.
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
            use_fast=True,
            local_files_only=True,
        )
        input_ids = tok.encode(prompt, add_special_tokens=False)
    input_ids = input_ids[:128]

    channel = grpc.insecure_channel(HOST)
    stub = pbg.TokenSpeedSchedulerStub(channel)
    all_out = []
    for i in range(REPS):
        outs, chunks = run(stub, input_ids, i)
        all_out.append(outs)
        print(f"run{i}: len={len(outs)} chunks={chunks} first8={outs[:8]}")

    for i in range(1, REPS):
        print(f"vs run0: LCP={lcp(all_out[0], all_out[i])}")

    opens = {}
    for i, o in enumerate(all_out):
        key = tuple(o[:6])
        opens.setdefault(key, []).append(i)
    print(f"distinct opens: {len(opens)}")
    self_eq = all(
        tuple(o[:8]) == tuple(all_out[0][:8]) for o in all_out[1:]
    )
    print(f"self-eq (first8 all same): {self_eq}")


if __name__ == "__main__":
    main()