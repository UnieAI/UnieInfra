#!/usr/bin/env python3
"""Concurrent gRPC vision benchmark for the UnieInfer DeepSeek-V4 engine.

The OpenAI gateway (smg) has no DeepSeek-V4 vision processor registered, so the
engine is driven over the gRPC scheduler interface directly, exactly like the
reference ``inference`` preprocessor leaves the request: the prompt token
sequence already carries the expanded sentinel block ``vocab_size + types``
(no image-placeholder token is used), and the multimodal item boxes the ViT
patch tensor plus the block layout (``n_vit_h``/``n_vit_w``/``types``/``perm``)
and the placeholder offsets.

This is the engine-arms side of the vision benchmark. It runs the same request
shape as the ``vllm bench serve --enable-multimodal-chat`` reference run
(1 image + ~64 text prompt tokens, ``output_len`` generated tokens), uses one
shared prebuilt request layout reused across the concurrent requests, and
reports TTFT / TPOT / total-token throughput.

Usage (in-engine container, GPU 0/1 free):
    IMG_GRPC_HOST=127.0.0.1:<grpc-port> IMG_CONC=32 IMG_NREQ=64 IMG_MAX_TOKENS=64 \
        IMG_TEXT_TOKENS=64 python3 bench/bench_dsv4_vision_grpc.py
"""

from __future__ import annotations

import io
import json
import os
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import grpc
import torch
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/inference")
import image_processor as ref_ip  # noqa: E402
import smg_grpc_proto.generated.common_pb2 as cpb  # noqa: E402
import smg_grpc_proto.generated.tokenspeed_scheduler_pb2 as pb  # noqa: E402
import smg_grpc_proto.generated.tokenspeed_scheduler_pb2_grpc as pbg  # noqa: E402

CKPT = "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
HOST = os.environ.get("IMG_GRPC_HOST", "127.0.0.1:56749")
IMG_START_POS = 3
IMG_CONC = int(os.environ.get("IMG_CONC", "32"))
IMG_NREQ = int(os.environ.get("IMG_NREQ", "64"))
IMG_MAX_TOKENS = int(os.environ.get("IMG_MAX_TOKENS", "64"))
VOCAB_SIZE = 129280


def tensor_data(t: torch.Tensor, *, dtype: str) -> pb.TensorData:
    if t.dtype == torch.bfloat16:
        t = t.float()
    t = t.contiguous()
    raw = t.detach().cpu().numpy().tobytes()
    return pb.TensorData(shape=list(t.shape), dtype=dtype, inline=raw)


def build_request(tokenizer) -> tuple[pb.GenerateRequest, int]:
    """Build one vision request; returns (request, expected_sequence_len)."""
    img = Image.new("RGB", (417, 389))
    px = img.load()
    for y in range(img.height):
        for x in range(img.width):
            px[x, y] = (
                (x * 7 + y * 3) % 255,
                (x * 5 + y * 11) % 255,
                (x * 13) % 255,
            )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    record = {"data": buf.getvalue()}

    import json as _json

    with open(f"{CKPT}/config.json") as f:
        raw = _json.load(f)
    args = type(
        "Args",
        (),
        {
            "vision_downsample_ratio": raw.get("vision_downsample_ratio", 3),
            "vision_patch_size": raw.get("vision_patch_size", 14),
            "vision_max_n_token": raw.get("vision_max_n_token", 384),
            "vision_max_wh_ratio": raw.get("vision_max_wh_ratio", None),
            "vision_min_pixels": raw.get("vision_min_pixels", 4 * 28 * 28),
        },
    )()
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ref_ip.load_image(record, args)
    types, perm = ref_ip.build_image_block(n_llm_h, n_llm_w, IMG_START_POS)
    block_len = int(types.numel())

    prefix_ids = tokenizer.encode(
        "Look at this picture. ".ljust(48), add_special_tokens=False
    )
    question = "Describe exactly what you see in one short paragraph."
    suffix_ids = tokenizer.encode(question, add_special_tokens=False)
    block_ids = (VOCAB_SIZE + types).tolist()
    input_ids = prefix_ids + block_ids + suffix_ids
    img_start = len(prefix_ids)

    mm = pb.MultimodalInputs()
    item = mm.items.add()
    item.modality = cpb.IMAGE
    item.content_hash = b"dsv4-vision-bench"
    item.encoder_input.CopyFrom(tensor_data(patches, dtype="float32"))
    item.model_specific_tensors["n_vit_h"].CopyFrom(
        tensor_data(torch.tensor([n_vit_h], dtype=torch.int64), dtype="int64")
    )
    item.model_specific_tensors["n_vit_w"].CopyFrom(
        tensor_data(torch.tensor([n_vit_w], dtype=torch.int64), dtype="int64")
    )
    item.model_specific_tensors["types"].CopyFrom(
        tensor_data(types.to(torch.int64), dtype="int64")
    )
    item.model_specific_tensors["perm"].CopyFrom(
        tensor_data(perm.to(torch.int64), dtype="int64")
    )
    ph = item.placeholders.add()
    ph.offset = img_start
    ph.length = block_len

    req = pb.GenerateRequest()
    req.request_id = "vision-bench-" + uuid.uuid4().hex[:8]
    req.tokenized.input_ids.extend(input_ids)
    req.sampling_params.max_new_tokens = IMG_MAX_TOKENS
    req.sampling_params.temperature = 0.0
    req.sampling_params.skip_special_tokens = True
    req.stream = True
    req.mm_inputs.CopyFrom(mm)
    return req, len(input_ids)


def one(stub, sem, req_template, res, idx):
    sem.acquire()
    t0 = time.perf_counter()
    req = pb.GenerateRequest()
    req.CopyFrom(req_template)
    req.request_id = "vision-bench-%04d-" % idx + uuid.uuid4().hex[:8]
    ttft = None
    outs: list[int] = []
    finish = None
    err = None
    try:
        for resp in stub.Generate(req, timeout=1800):
            if ttft is None and resp.HasField("chunk") and resp.chunk.token_ids:
                ttft = (time.perf_counter() - t0) * 1000
            if resp.HasField("chunk"):
                outs.extend(resp.chunk.token_ids)
            # The ``complete`` frame only echoes the final chunk's output_ids,
            # so token counts must be accumulated from the chunk stream. Its
            # finish_reason is the authoritative end-of-generation signal.
            if resp.HasField("complete") and resp.complete.finish_reason:
                finish = resp.complete.finish_reason
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:300]
    finally:
        sem.release()
    e2e = (time.perf_counter() - t0) * 1000
    ntok = len(outs)
    res[idx] = {
        "ms": e2e,
        "ttft": ttft,
        "ntok": ntok,
        "finish": finish,
        "tpot": (
            (e2e - ttft) / max(1, ntok - 1) if ttft is not None and ntok > 1 else None
        ),
        "err": err,
    }


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    req_template, seq_len = build_request(tokenizer)
    print(
        f"host={HOST} conc={IMG_CONC} nreq={IMG_NREQ} max_tokens={IMG_MAX_TOKENS} "
        f"seq_len={seq_len} image_seq_tokens={int(req_template.mm_inputs.items[0].placeholders[0].length)}"
    )

    chan = grpc.insecure_channel(HOST)
    stub = pbg.TokenSpeedSchedulerStub(chan)

    # batch warmup on the SAME layout (different content_hash is not possible
    # here since the layout is shared); this warms ViT/aligner/cudagraph only,
    # so the timed run measures steady-state serving, not first-touch compile.
    warm_n = 4
    for _ in range(warm_n):
        r = pb.GenerateRequest()
        r.CopyFrom(req_template)
        r.request_id = "warm-" + uuid.uuid4().hex[:8]
        r.sampling_params.max_new_tokens = 8
        for _ in stub.Generate(r, timeout=600):
            pass

    sem = threading.Semaphore(IMG_CONC)
    res: dict[int, dict] = {}
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=IMG_CONC) as ex:
        futs = [
            ex.submit(one, stub, sem, req_template, res, i) for i in range(IMG_NREQ)
        ]
        for f in futs:
            f.result()
    dur = time.perf_counter() - t_start

    rows = [res[i] for i in sorted(res)]
    ok = sum(1 for r in rows if r["ntok"] > 0 and not r["err"])
    ttfts = [r["ttft"] for r in rows if r["ttft"] is not None]
    tpots = [r["tpot"] for r in rows if r["tpot"] is not None]
    total_tok = sum(r["ntok"] for r in rows)
    print(f"n={IMG_NREQ} ok={ok} duration={dur:.2f}s")
    print(
        f"TTFT mean={statistics.mean(ttfts):.1f} med={statistics.median(ttfts):.1f} "
        f"p99={sorted(ttfts)[int(0.99*(len(ttfts)-1))]:.1f}"
    )
    if tpots:
        print(
            f"TPOT mean={statistics.mean(tpots):.1f} med={statistics.median(tpots):.1f} "
            f"p99={sorted(tpots)[int(0.99*(len(tpots)-1))]:.1f}"
        )
    print(
        f"out_tok/s={total_tok/dur:.2f} total_tok={total_tok} "
        f"avg_tok/req={total_tok/max(1,IMG_NREQ):.1f}"
    )
    out = os.environ.get("IMG_RESULT", "/tmp/vision_engine.json")
    json.dump(
        {"duration_s": dur, "n": IMG_NREQ, "ok": ok, "rows": rows}, open(out, "w")
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
