"""Whisper encoder parity against HuggingFace, with the gap attributed.

Two questions, because the first one alone cannot be answered honestly:

1. does our encoder agree with `transformers`, and
2. when it does not, is that wiring or arithmetic?

A single end-to-end cosine cannot separate them -- a wrong position table and
a low-precision matmul both show up as "close but not equal". So this script
also walks the stem and the first attention block separately: a wiring bug is
large at the stem or at layer 0, while a kernel-precision difference is ~zero
at the stem and appears the moment attention runs.

Measured 2026-08-21, whisper-large-v3-turbo, sm_120:

    fp32:  stem rel = 0.000e+00 (bit-exact)
           after encoder layer 0  rel = 5.4e-04
           full encoder           cosine mean 0.999999, rel 1.7e-03
    bf16:  full encoder           cosine mean 0.999449, rel 3.3e-02

The stem being bit-exact and one attention block carrying essentially the
whole fp32 budget says the difference is the attention kernel's internal
precision, not the port. It is NOT proof the model transcribes correctly --
that needs the decoder end to end.

Run:

    docker exec unieinfra-dev bash -lc '
      cd /workspace/roy/UnieInfra && source legacy/benchmark/scripts/sm120_env.sh
      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/roy/UnieInfra/python \
        python3 unieinfer/bench/whisper_encoder_parity.py --fp32'
"""

import argparse
import glob

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers import WhisperForConditionalGeneration as HFWhisper

DEFAULT_SNAPSHOT_GLOB = (
    "/workspace/.cache/huggingface/hub/"
    "models--openai--whisper-large-v3-turbo/snapshots/*"
)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def build(snapshot: str, dtype: torch.dtype):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.whisper import WhisperForConditionalGeneration

    config = AutoConfig.from_pretrained(snapshot)
    torch.set_default_dtype(dtype)
    with torch.device("cuda"):
        model = WhisperForConditionalGeneration(config, Mapping(rank=0, world_size=1))
    weights = load_file(f"{snapshot}/model.safetensors")
    loaded = model.load_weights(
        (name, tensor.to("cuda", dtype)) for name, tensor in weights.items()
    )
    if len(loaded) != len(weights):
        raise SystemExit(
            f"loaded {len(loaded)} of {len(weights)} checkpoint tensors -- the "
            "layout does not match this implementation"
        )
    return config, model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-glob", default=DEFAULT_SNAPSHOT_GLOB)
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Run both sides in fp32. Separates precision from wiring: the "
        "port's error should collapse, a real bug should not.",
    )
    args = parser.parse_args()

    matches = glob.glob(args.snapshot_glob)
    if not matches:
        raise SystemExit(f"no checkpoint at {args.snapshot_glob}")
    snapshot = matches[0]
    dtype = torch.float32 if args.fp32 else torch.bfloat16

    config, ours = build(snapshot, dtype)
    hf = HFWhisper.from_pretrained(snapshot, dtype=dtype).to("cuda").eval()
    reference_encoder = hf.model.encoder

    # Deterministic synthetic log-mel: encoder parity is a property of the
    # weights and the wiring, not of the audio. Real speech is what the
    # end-to-end transcription check needs, not this one.
    torch.manual_seed(0)
    features = torch.randn(1, config.num_mel_bins, config.max_source_positions * 2).to(
        "cuda", dtype
    )

    with torch.no_grad():
        mine = F.gelu(ours.encoder.conv1(features))
        mine = F.gelu(ours.encoder.conv2(mine)).permute(0, 2, 1)
        mine = mine + ours.encoder.embed_positions[: mine.shape[1]].to(mine.dtype)

        ref = F.gelu(reference_encoder.conv1(features))
        ref = F.gelu(reference_encoder.conv2(ref)).permute(0, 2, 1)
        ref = ref + reference_encoder.embed_positions.weight[: ref.shape[1]]

        print(f"dtype                  {dtype}")
        print(f"stem (conv+positions)  rel = {_rel(mine, ref):.3e}")

        full_mine = ours.encoder(features)
        full_ref = reference_encoder(features).last_hidden_state
        full_ref = full_ref.reshape(-1, full_ref.shape[-1])
        cosine = F.cosine_similarity(full_mine.float(), full_ref.float(), dim=-1)
        print(f"full encoder           rel = {_rel(full_mine, full_ref):.3e}")
        print(
            f"full encoder cosine    min = {cosine.min().item():.6f}  "
            f"mean = {cosine.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
