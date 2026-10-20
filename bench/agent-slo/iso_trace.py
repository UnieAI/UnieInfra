#!/usr/bin/env python3
"""Isolated self-consistency that ROTATES the engine qkv trace per rep so each
rep gets a clean per-rank file, plus records per-rep token output."""

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

HOST = "127.0.0.1:37189"
MAX_NEW = 128
QKV = "/tmp/qkv_hashes.txt"
FLASH = "/tmp/flashmla_hashes.txt"
ATTN = "/tmp/attn_hashes.txt"
IDX = "/tmp/indexer_hashes.txt"
LOGITS = "/tmp/logits_hashes.txt"


def rotate(tag):
    for p in (QKV, FLASH, ATTN, IDX, LOGITS):
        if os.path.exists(p):
            os.replace(p, f"{p}.{tag}")


def run(stub, input_ids, rep):
    req = pb.GenerateRequest()
    req.request_id = f"isoT-{rep}-" + uuid.uuid4().hex[:8]
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
    except Exception as e:  # noqa: BLE001
        return [f"ERR:{type(e).__name__}:{e}"]
    return outs


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tokens = json.load(open("engine_input.json"))
    tok = AutoTokenizer.from_pretrained(
        "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        use_fast=True,
        local_files_only=True,
    )
    rotate("pre")  # stash any leftover from prior manual runs
    ch = grpc.insecure_channel(HOST)
    stub = pbg.TokenSpeedSchedulerStub(ch)
    allouts = []
    for rep in range(reps):
        rotate(f"rep{rep}")  # move the previous rep's trace aside before this one
        outs = run(stub, tokens[idx], rep)
        rotate(f"rep{rep}")  # move this rep's trace aside
        text = tok.decode(outs, skip_special_tokens=True)
        allouts.append(list(outs))
        nq = (
            sum(1 for _ in open(f"{QKV}.rep{rep}"))
            if os.path.exists(f"{QKV}.rep{rep}")
            else -1
        )
        print(
            f"  rep{rep}: gen={len(outs)} head8={list(outs[:8])} "
            f"text0={text[:30]!r} qkvlines={nq}",
            flush=True,
        )
    uniq = {tuple(o): tok.decode(o, skip_special_tokens=True) for o in allouts}
    print(
        f"idx={idx} n_in={len(tokens[idx])} reps={reps} distinct={len(uniq)}",
        flush=True,
    )
    for u in uniq:
        print(f"   uniq: gen={len(u)} head={uniq[u][:60]!r}", flush=True)
    with open("iso_trace_out.json", "w") as f:
        json.dump({"idx": idx, "n_in": len(tokens[idx]), "outs": allouts}, f)
    keys = [tuple(o) for o in allouts]
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            if keys[a] == keys[b]:
                continue
            d = next(
                (i for i, (x, y) in enumerate(zip(keys[a], keys[b])) if x != y),
                min(len(keys[a]), len(keys[b])),
            )
            print(
                f"  diff rep{a} vs rep{b} at pos {d}  "
                f"a={list(keys[a][max(0, d - 2):d + 3])} b={list(keys[b][max(0, d - 2):d + 3])}",
                flush=True,
            )


if __name__ == "__main__":
    main()
