#!/usr/bin/env python3
"""Token-for-token comparison of the UnieInfer DeepSeek-V4 vision tower
against the reference ``/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp``
``inference/vision.py`` + ``inference/image_processor.py``.

Loads only the vision / aligner / image-scaffold weights out of the 48-shard
checkpoint (no LM), encodes one real image on GPU, and asserts the produced
token blocks match the reference exactly (types/perm) and closely (embeddings).

Run inside the container:
    PYTHONPATH=tokenspeed-kernel/python:python python3 bench/verify_dsv4_vision.py
"""

from __future__ import annotations

import io
import json
import math
import sys
import types
from types import SimpleNamespace

import torch
from safetensors import safe_open

CKPT = "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
REF_INF = f"{CKPT}/inference"
sys.path.insert(0, REF_INF)

import image_processor as ref_ip  # noqa: E402
import vision as ref_vision  # noqa: E402

from tokenspeed.runtime.models.deepseek_v4_vision import (  # noqa: E402
    IMAGE,
    DeepseekV4VisionModel,
    build_image_block,
    llm_grid,
)


def make_config() -> SimpleNamespace:
    with open(f"{CKPT}/config.json") as f:
        raw = json.load(f)
    return SimpleNamespace(
        vision_n_layers=raw.get("vision_n_layers", 32),
        vision_dim=raw.get("vision_dim", 1024),
        vision_n_heads=raw.get("vision_n_heads", 16),
        vision_inter_dim=raw.get("vision_inter_dim", 2816),
        vision_patch_size=raw.get("vision_patch_size", 14),
        vision_rope_theta=raw.get("vision_rope_theta", 10000.0),
        vision_downsample_ratio=raw.get("vision_downsample_ratio", 3),
        dim=raw.get("hidden_size", 4096),
        vision_max_n_token=raw.get("vision_max_n_token", 384),
        vision_max_wh_ratio=raw.get("vision_max_wh_ratio", None),
        vision_min_pixels=raw.get("vision_min_pixels", 4 * 28 * 28),
        vocab_size=raw.get("vocab_size", 129280),
    )


def weight_sources() -> dict[str, str]:
    with open(f"{CKPT}/model.safetensors.index.json") as f:
        idx = json.load(f)
    return idx["weight_map"]


def load_ckpt_tensor(sources: dict[str, str], name: str) -> torch.Tensor:
    shard = sources[name]
    with safe_open(f"{CKPT}/{shard}", framework="pt") as f:
        return f.get_tensor(name)


def main() -> int:
    torch.manual_seed(1234)
    cfg = make_config()
    # The reference generate.py sets the default dtype to bf16 before building
    # the modules; mirror that so both sides use bf16 linear kernels.
    torch.set_default_dtype(torch.bfloat16)

    # Reference modules.
    ref_vit = ref_vision.ViT(cfg)
    ref_align = ref_vision.Aligner(cfg)
    ref_params = {
        k: torch.empty(cfg.dim, dtype=torch.bfloat16)
        for k in ("image_start", "image_pad", "image_newline", "image_end")
    }

    # Our modules (bf16 params, matching the loader's dtype context).
    ours = DeepseekV4VisionModel(cfg)
    ours.to(dtype=torch.bfloat16)
    torch.set_default_dtype(torch.float32)

    sources = weight_sources()
    vision_names = [n for n in sources if n.startswith("vision.") and "align" not in n]
    aligner_names = [n for n in sources if n.startswith("aligner.")]
    param_names = [
        n
        for n in sources
        if n in ("image_start", "image_pad", "image_newline", "image_end")
    ]

    ref_sd = ref_vit.state_dict()
    ref_align_sd = ref_align.state_dict()
    ours_sd = ours.state_dict()
    for name in vision_names:
        t = load_ckpt_tensor(sources, name)
        un = name[len("vision.") :]  # "patch_embed.proj.weight" etc
        if un in ref_sd:
            ref_vit.state_dict()[un].data.copy_(t)
        if name in ours_sd:
            ours_sd[name].data.copy_(t)
    for name in aligner_names:
        t = load_ckpt_tensor(sources, name)
        un = name[len("aligner.") :]
        if un in ref_align_sd:
            ref_align.state_dict()[un].data.copy_(t)
        if name in ours_sd:
            ours_sd[name].data.copy_(t)
    for name in param_names:
        ref_params[name].data.copy_(load_ckpt_tensor(sources, name))

    print(
        f"loaded vision={len(vision_names)} aligner={len(aligner_names)} "
        f"params={len(param_names)} tensors"
    )
    if not (vision_names and aligner_names and len(param_names) == 4):
        print("WARNING: some expected weights were not found")
        return 2

    # --- Build a real-ish image and run the reference preprocessor ---
    from PIL import Image

    img = Image.new("RGB", (417, 389))
    px = img.load()
    for y in range(img.height):
        for x in range(img.width):
            px[x, y] = ((x * 7 + y * 3) % 255, (x * 5 + y * 11) % 255, (x * 13) % 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    record = {"data": buf.getvalue()}
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ref_ip.load_image(record, cfg)
    start_pos = 12  # arbitrary block position in the expanded stream
    ref_types, ref_perm = ref_ip.build_image_block(n_llm_h, n_llm_w, start_pos)

    # Rebuild with ours and confirm identical layout.
    our_types, our_perm = build_image_block(n_llm_h, n_llm_w, start_pos)
    assert torch.equal(ref_types, our_types), "types mismatch"
    assert torch.equal(ref_perm, our_perm), "perm mismatch"
    # llm_grid round-trip.
    assert (n_llm_h, n_llm_w) == llm_grid(n_vit_h, n_vit_w, cfg.vision_downsample_ratio)

    patches_b = patches.to(torch.bfloat16)

    # Reference block.
    ref_embeds = ref_align(ref_vit(patches_b, n_vit_h, n_vit_w), n_vit_h, n_vit_w)[
        ref_perm
    ]
    rp = torch.stack(
        [
            ref_params[k]
            for k in (
                "image_start",
                "image_pad",
                "image_pad",
                "image_newline",
                "image_end",
            )
        ]
    )
    ref_block = rp[ref_types]
    ref_block[ref_types == IMAGE] = ref_embeds

    # Ours block (same layout code path as get_image_feature). The image-scaffold
    # parameters live on DeepseekV4ForCausalLM, so here we reuse the loaded
    # reference values; what we actually compare is the encoded tokens.
    our_embeds = ours.encode(patches_b, n_vit_h, n_vit_w)[our_perm]
    our_block = rp[our_types]
    our_block[our_types == IMAGE] = our_embeds

    embeds_diff = (ref_embeds.float() - our_embeds.float()).abs().max().item()
    block_diff = (ref_block.float() - our_block.float()).abs().max().item()
    print(
        f"grid n_vit={n_vit_h}x{n_vit_w} n_llm={n_llm_h}x{n_llm_w} "
        f"block_len={ref_types.numel()} img_tokens={ref_perm.numel()}"
    )
    print(f"embeds max|diff|={embeds_diff:.6e}  block max|diff|={block_diff:.6e}")
    if embeds_diff > 1e-2 or block_diff > 1e-2:
        print("FAIL: embeddings diverge from reference")
        return 1
    print("PASS: block layout identical, embeddings match reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
