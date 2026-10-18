#!/usr/bin/env python3
"""Temperature-0 agreement probe against the engine gRPC scheduler.

Feeds the engine the EXACT prompt token IDs obtained from the reference vLLM
(/tokenize) so both engines see byte-identical inputs; compares the resulting
greedy token streams.

Usage (inside roy-h200g):
    python3 agree_engine.py <engine_input.json> <out.json> [max_new_tokens]
where engine_input.json is a JSON list of token-id lists.
"""
from __future__ import annotations

import json
import os
import sys
import uuid

sys.path.insert(0, "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/inference")
import grpc  # noqa: E402
import smg_grpc_proto.generated.tokenspeed_scheduler_pb2 as pb  # noqa: E402
import smg_grpc_proto.generated.tokenspeed_scheduler_pb2_grpc as pbg  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

HOST = os.environ.get("DET_HOST", "127.0.0.1:37189")
MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 128


def run(stub, input_ids, idx):
    req = pb.GenerateRequest()
    req.request_id = f"agree-{idx}-" + uuid.uuid4().hex[:8]
    req.tokenized.input_ids.extend(input_ids)
    req.sampling_params.max_new_tokens = MAX_NEW
    req.sampling_params.temperature = 0.0
    req.sampling_params.skip_special_tokens = True
    req.stream = True
    outs = []
    for resp in stub.Generate(req, timeout=900):
        for t in resp.chunk.token_ids:
            outs.append(t)
    return outs


def main():
    tokens = json.load(open(sys.argv[1]))
    tok = AutoTokenizer.from_pretrained(
        "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        use_fast=True,
        local_files_only=True,
    )
    channel = grpc.insecure_channel(HOST)
    stub = pbg.TokenSpeedSchedulerStub(channel)
    result = {}
    for i, ids in enumerate(tokens):
        outs = run(stub, ids, i)
        text = tok.decode(outs, skip_special_tokens=True)
        result[str(i)] = {"tokens": outs, "text": text, "n_in": len(ids)}
        print(f"engine[{i}]: n_in={len(ids)} gen={len(outs)} text0={text[:40]!r}")
    json.dump(result, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()