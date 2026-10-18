#!/usr/bin/env python3
"""Isolated self-consistency: call ONE prompt K times, nothing else in flight."""
from __future__ import annotations

import collections
import json
import sys
import uuid

sys.path.insert(0, "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/inference")
import grpc  # noqa: E402
import smg_grpc_proto.generated.tokenspeed_scheduler_pb2 as pb  # noqa: E402
import smg_grpc_proto.generated.tokenspeed_scheduler_pb2_grpc as pbg  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

HOST = "127.0.0.1:37189"
MAX_NEW = 128


def run(stub, input_ids, idx, k):
    req = pb.GenerateRequest()
    req.request_id = f"iso-{idx}-{k}-" + uuid.uuid4().hex[:8]
    req.tokenized.input_ids.extend(input_ids)
    req.sampling_params.max_new_tokens = MAX_NEW
    req.sampling_params.temperature = 0.0
    req.sampling_params.skip_special_tokens = True
    req.stream = True
    outs = []
    try:
        for resp in stub.Generate(req, timeout=900):
            for t in resp.chunk.token_ids:
                outs.append(t)
    except Exception as e:
        return [f"ERR:{type(e).__name__}:{e}"]
    return outs


def main():
    idx = int(sys.argv[1])
    k = int(sys.argv[2])
    tokens = json.load(open("engine_input.json"))
    tok = AutoTokenizer.from_pretrained(
        "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        use_fast=True, local_files_only=True,
    )
    ch = grpc.insecure_channel(HOST)
    stub = pbg.TokenSpeedSchedulerStub(ch)
    seen = collections.Counter()
    allouts = []
    for i in range(k):
        outs = run(stub, tokens[idx], idx, i)
        text = tok.decode(outs, skip_special_tokens=True)
        key = tuple(outs)
        seen[key] += 1
        allouts.append(list(outs))
        print(f"  rep{i}: gen={len(outs)} head8={list(outs[:8])} text0={text[:30]!r}")
    uniq = {tuple(o): tok.decode(o, skip_special_tokens=True) for o in seen}
    print(f"idx={idx} n_in={len(tokens[idx])} reps={k} distinct={len(uniq)}")
    for u in uniq:
        print(f"   uniq: gen={len(u)} head={uniq[u][:60]!r}")
    with open("iso_out.json", "w") as f:
        json.dump({"idx": idx, "n_in": len(tokens[idx]), "outs": allouts}, f)
    # first-diff positions across distinct streams
    keys = [tuple(o) for o in allouts]
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            if keys[a] == keys[b]:
                continue
            d = next((i for i, (x, y) in enumerate(zip(keys[a], keys[b])) if x != y),
                     min(len(keys[a]), len(keys[b])))
            print(f"  diff rep{a} vs rep{b} at pos {d}  a={list(keys[a][max(0,d-2):d+3])} b={list(keys[b][max(0,d-2):d+3])}")


if __name__ == "__main__":
    main()