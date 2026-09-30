#!/usr/bin/env python3
"""Drive one vision generation through the running TokenSpeed gRPC server.

Builds the request exactly the way the reference ``inference`` preprocessor
leaves it: the prompt token sequence already carries the expanded sentinel
block ``vocab_size + types`` (IMPORTANT: no image-placeholder token is used),
and the multimodal item just boxes the ViT patch tensor plus the block layout
(``n_vit_h``/``n_vit_w``/``types``/``perm``) and the placeholder offsets.

Run inside the container against a live server:
    PYTHONPATH=python:tokenspeed-kernel/python python3 bench/drive_dsv4_vision.py
"""

from __future__ import annotations

import io
import os
import sys
import sys as _sys
import time
import uuid

import grpc
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/inference")
import image_processor as ref_ip  # noqa: E402
import smg_grpc_proto.generated.common_pb2 as cpb  # noqa: E402
import smg_grpc_proto.generated.tokenspeed_scheduler_pb2 as pb  # noqa: E402
import smg_grpc_proto.generated.tokenspeed_scheduler_pb2_grpc as pbg  # noqa: E402

CKPT = "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
HOST = "127.0.0.1:56749"
IMG_START_POS = 3  # block start position inside the expanded stream (arbitrary)


def tensor_data(t: torch.Tensor, *, dtype: str) -> pb.TensorData:
    # bfloat16 has no numpy mapping in this torch; float in, bf16 out on the
    # engine is lossless for a bf16-origin tensor, so widen for the wire.
    if t.dtype == torch.bfloat16:
        t = t.float()
    t = t.contiguous()
    raw = t.detach().cpu().numpy().tobytes()
    return pb.TensorData(shape=list(t.shape), dtype=dtype, inline=raw)


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    vocab_size = 129280

    # --- image -> patches / grid / sentinel block (reference preprocessor) ---
    img = Image.new("RGB", (417, 389))
    px = img.load()
    solid = os.environ.get("IMG_SOLID")
    if solid:
        r, g, b = tuple(int(v) for v in solid.split(","))
        color = (r, g, b)
        for y in range(img.height):
            for x in range(img.width):
                px[x, y] = color
    else:
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
    # A dummy args namespace mirroring config.json + the defaults generate.py uses.
    import json

    with open(f"{CKPT}/config.json") as f:
        raw = json.load(f)
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
    tokens_in_block = int(perm.numel())
    print(
        f"patch grid n_vit={n_vit_h}x{n_vit_w} n_llm={n_llm_h}x{n_llm_w} "
        f"block_len={block_len} image_tokens={tokens_in_block}"
    )

    # --- prompt: text + expanded sentinel block + text (no placeholder token) ---
    prefix_ids = tokenizer.encode("Look at this picture.", add_special_tokens=False)
    question = os.environ.get("IMG_QUESTION", "What is the dominant color?")
    suffix_ids = tokenizer.encode(question, add_special_tokens=False)
    block_ids = (vocab_size + types).tolist()
    input_ids = prefix_ids + block_ids + suffix_ids
    img_start = len(prefix_ids)

    mm = pb.MultimodalInputs()
    item = mm.items.add()
    item.modality = cpb.IMAGE
    item.content_hash = b"dsv4-vision-test-0001"
    # The ViT patch tensor rides as the encoder input (float32 on the wire; the
    # servicer casts it to the model dtype). No placeholder token is used.
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
    # placeholder_token_id deliberately left unset: the sequence uses the
    # sentinel block directly, not a placeholder token.

    req = pb.GenerateRequest()
    req.request_id = "vision-test-" + uuid.uuid4().hex[:8]
    req.tokenized.input_ids.extend(input_ids)
    req.sampling_params.max_new_tokens = 16
    req.sampling_params.temperature = 0.0
    req.sampling_params.skip_special_tokens = True
    req.stream = False
    req.mm_inputs.CopyFrom(mm)

    chan = grpc.insecure_channel(HOST)
    stub = pbg.TokenSpeedSchedulerStub(chan)

    # --- control: text-only on the same server must still generate ---
    txt = pb.GenerateRequest()
    txt.request_id = "text-probe-" + uuid.uuid4().hex[:8]
    txt.tokenized.input_ids.extend(
        tokenizer.encode("What is 2+2?", add_special_tokens=False)
    )
    txt.sampling_params.max_new_tokens = 8
    txt.sampling_params.temperature = 0.0
    txt.sampling_params.skip_special_tokens = True
    txt.stream = False
    txt_resp = []
    try:
        for resp in stub.Generate(txt, timeout=60):
            txt_resp.append(resp)
    except Exception as e:  # noqa: BLE001
        print("TEXT RPC ERR:", repr(e)[:400])
        return 1
    txt_out = []
    for resp in txt_resp:
        if resp.HasField("chunk"):
            txt_out.extend(resp.chunk.token_ids)
        if resp.HasField("complete") and resp.complete.output_ids:
            txt_out = list(resp.complete.output_ids)
    print(f"TEXT probe: gen_tokens={len(txt_out)} => {tokenizer.decode(txt_out)!r}")
    if not txt_out:
        print("WARNING: text probe produced no tokens; server baseline is broken")

    t0 = time.time()
    resp_list = []
    try:
        for resp in stub.Generate(req, timeout=180):
            resp_list.append(resp)
    except Exception as e:  # noqa: BLE001
        print("RPC ERR:", repr(e)[:600])
        return 1
    elapsed = time.time() - t0
    outs: list[int] = []
    finish = None
    for resp in resp_list:
        if resp.HasField("chunk"):
            outs.extend(resp.chunk.token_ids)
        if resp.HasField("complete") and resp.complete.output_ids:
            outs = list(resp.complete.output_ids)
        if resp.complete.finish_reason:
            finish = resp.complete.finish_reason
    print(
        f"elapsed {elapsed:.1f}s  prompts={len(input_ids)} "
        f"gen_tokens={len(outs)} finish={finish}"
    )
    print("output:", repr(tokenizer.decode(outs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
