"""Whisper end-to-end: audio in, transcript out, against a HF reference.

This is the only check that actually says Whisper works. Encoder parity says
the encoder is wired right; loading 587 tensors says the checkpoint maps; a
clean engine start says nothing at all about output. Only a transcript that
matches the reference exercises multi-step decode and cross-attention across
every step.

Use real speech. A noise clip makes the reference emit a single token, and
matching one token would "pass" while telling you almost nothing -- the first
version of this check did exactly that.

Measured 2026-08-21, whisper-large-v3-turbo, bf16, sm_120, triton backend,
LibriSpeech dummy clip 0:

    truth      MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD ...
    reference  ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
    ours       ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'

Getting a clip (the container has no `datasets`, so extract on the host):

    python3 -c "
    import glob, pathlib, pyarrow.parquet as pq
    p = glob.glob('~/.cache/huggingface/hub/datasets--hf-internal-testing--'
                  'librispeech_asr_dummy/snapshots/*/clean/*.parquet')[0]
    row = pq.read_table(p).slice(0, 1).to_pylist()[0]
    pathlib.Path('clip.flac').write_bytes(row['audio']['bytes'])
    print(row['text'])"

Run:

    docker exec unieinfra-dev bash -lc '
      cd /workspace/roy/UnieInfra && source legacy/benchmark/scripts/sm120_env.sh
      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/roy/UnieInfra/python \
        python3 unieinfra/bench/whisper_transcribe_parity.py --audio clip.flac'
"""

import argparse
import glob

import torch

DEFAULT_SNAPSHOT_GLOB = (
    "/workspace/.cache/huggingface/hub/"
    "models--openai--whisper-large-v3-turbo/snapshots/*"
)
# <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
PROMPT_TOKENS = [
    "<|startoftranscript|>",
    "<|en|>",
    "<|transcribe|>",
    "<|notimestamps|>",
]


def features(processor, audio_path: str):
    import soundfile as sf

    wav, rate = sf.read(audio_path, dtype="float32")
    return processor.feature_extractor(
        wav, sampling_rate=rate, return_tensors="pt"
    ).input_features


def reference(snapshot: str, processor, feats, max_new_tokens: int) -> str:
    from transformers import WhisperForConditionalGeneration as HFWhisper

    hf = HFWhisper.from_pretrained(snapshot, dtype=torch.bfloat16).to("cuda").eval()
    with torch.no_grad():
        ids = hf.generate(
            feats.to("cuda", torch.bfloat16),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    del hf
    torch.cuda.empty_cache()
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True)


def ours(
    snapshot: str,
    processor,
    feats,
    max_new_tokens: int,
    max_num_seqs: int,
    cross_attn_path: str,
) -> str:
    from tokenspeed.runtime.engine.io_struct import GenerateReqInput
    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.multimodal.inputs import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )

    engine = Engine(
        model=snapshot,
        dtype="bfloat16",
        max_model_len=448,
        gpu_memory_utilization=0.25,
        max_num_seqs=max_num_seqs,
        cross_attn_path=cross_attn_path,
        disable_kvstore=True,
        attention_backend="triton",
    )
    request = GenerateReqInput(
        input_ids=processor.tokenizer.convert_tokens_to_ids(PROMPT_TOKENS),
        sampling_params={"temperature": 0.0, "max_new_tokens": max_new_tokens},
    )
    # Audio rides as a precomputed multimodal item, the same way the gateway
    # ships it: this engine does no feature extraction of its own.
    request.precomputed_multimodal_inputs = MultimodalInputs(
        mm_items=[
            MultimodalDataItem(modality=Modality.AUDIO, feature=feats[0].contiguous())
        ]
    )
    out = engine.llm.generate(request)
    engine.shutdown()
    return out["text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="A speech clip (flac/wav).")
    parser.add_argument("--snapshot-glob", default=DEFAULT_SNAPSHOT_GLOB)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--truth", default=None, help="Optional ground-truth text.")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="Cross-attention picks its read path by batch occupancy "
        "(bs*2 >= slots). Set this to 1 so a single request exercises the "
        "padded path; leave it larger to exercise the gather path.",
    )
    parser.add_argument(
        "--cross-attn-path",
        choices=["auto", "padded", "gather"],
        default="auto",
        help="Force one cross-attention read path so both can be verified "
        "on the same build; 'auto' picks by occupancy.",
    )
    args = parser.parse_args()

    matches = glob.glob(args.snapshot_glob)
    if not matches:
        raise SystemExit(f"no checkpoint at {args.snapshot_glob}")
    snapshot = matches[0]

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(snapshot)
    feats = features(processor, args.audio)

    # Reference first, and freed before the engine starts: both want the card.
    reference_text = reference(snapshot, processor, feats, args.max_new_tokens)
    our_text = ours(snapshot, processor, feats, args.max_new_tokens, args.max_num_seqs)

    if args.truth:
        print(f"truth      {args.truth!r}")
    print(f"reference  {reference_text!r}")
    print(f"ours       {our_text!r}")

    if our_text != reference_text:
        raise SystemExit("FAIL: transcript differs from the reference")
    print("PASS: transcript is identical to the reference")


if __name__ == "__main__":
    main()
