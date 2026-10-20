"""Whisper decode step: padded cross-attention against the gather it replaced.

The gather existed to make the batch's encoder KV contiguous for flash-attn.
It moved hundreds of MB per step at ~1150 GB/s -- of data already resident --
and grew linearly with batch size. The padded path attends over every slot
instead, scattering the batch's queries into a full-width buffer by pool
index, so nothing is copied and the shape is constant (which is also what
lets a CUDA graph capture it).

Reported per decode step for the WHOLE decoder, not just cross-attention:
the fraction that matters is what the step costs, not what the kernel costs.

Run:

    docker exec unieinfra-dev bash -lc '
      cd /workspace/roy/UnieInfra && source legacy/benchmark/scripts/sm120_env.sh
      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/roy/UnieInfra/python \
        python3 unieinfra/bench/whisper_cross_attn_paths.py'
"""

import argparse
import glob

import torch
from transformers import AutoConfig

DEFAULT_SNAPSHOT_GLOB = (
    "/workspace/.cache/huggingface/hub/"
    "models--openai--whisper-large-v3-turbo/snapshots/*"
)
ITERS, WARMUP = 30, 8


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-glob", default=DEFAULT_SNAPSHOT_GLOB)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    args = parser.parse_args()

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.whisper import (
        CrossAttentionPlan,
        WhisperDecoder,
    )

    snapshot = glob.glob(args.snapshot_glob)[0]
    config = AutoConfig.from_pretrained(snapshot)
    slots = args.max_num_seqs + 1
    config.cross_attn_slots = slots

    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        decoder = WhisperDecoder(config, Mapping(rank=0, world_size=1)).eval()

    from tokenspeed.runtime.models.whisper import CrossAttentionCache

    heads = config.decoder_attention_heads
    dim = config.d_model // heads
    cache = CrossAttentionCache(
        num_slots=slots,
        num_layers=config.decoder_layers,
        encoder_len=config.max_source_positions,
        kv_heads=heads,
        head_dim=dim,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    enc = cache.encoder_len

    print(f"decoder_layers={config.decoder_layers} encoder_len={enc} slots={slots}")
    print(f"{'bs':>3} {'gather ms':>10} {'padded ms':>10} {'speedup':>8}")
    for bs in (1, 4, 8, 16):
        if bs > args.max_num_seqs:
            continue
        idx = torch.arange(bs, device="cuda", dtype=torch.long)
        cu_q_pad = torch.arange(slots + 1, device="cuda", dtype=torch.int32)
        cu_k_pad = torch.arange(
            0, (slots + 1) * enc, enc, device="cuda", dtype=torch.int32
        )
        cu_q = torch.arange(bs + 1, device="cuda", dtype=torch.int32)
        cu_k = torch.arange(0, (bs + 1) * enc, enc, device="cuda", dtype=torch.int32)
        qbuf = torch.zeros(slots, heads, dim, device="cuda", dtype=torch.bfloat16)
        hidden = torch.randn(bs, config.d_model, device="cuda", dtype=torch.bfloat16)

        padded_plan = CrossAttentionPlan(
            cache=cache,
            padded=True,
            slots=idx,
            cu_seqlens_q=cu_q_pad,
            cu_seqlens_k=cu_k_pad,
            encoder_len=enc,
            query_buffer=qbuf,
        )

        def padded():
            for layer in decoder.layers:
                layer.encoder_attn_layer_norm(hidden)
                q = layer.encoder_attn.project_query(hidden)
                padded_plan.query_buffer.index_copy_(0, idx, q)
                out = layer.encoder_attn.attend(
                    padded_plan.query_buffer,
                    *cache.whole_buffer(layer.layer_id),
                    cu_seqlens_q=cu_q_pad,
                    cu_seqlens_k=cu_k_pad,
                    max_seqlen_q=1,
                    max_seqlen_k=enc,
                )
                layer.encoder_attn.project_out(out.index_select(0, idx))

        def gathered():
            key, value = cache.gather(idx)
            for layer in decoder.layers:
                layer.encoder_attn_layer_norm(hidden)
                q = layer.encoder_attn.project_query(hidden)
                out = layer.encoder_attn.attend(
                    q,
                    key[layer.layer_id].reshape(-1, heads, dim),
                    value[layer.layer_id].reshape(-1, heads, dim),
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=1,
                    max_seqlen_k=enc,
                )
                layer.encoder_attn.project_out(out)

        with torch.no_grad():
            g, p = timed(gathered), timed(padded)
        print(f"{bs:3d} {g:10.3f} {p:10.3f} {g / p:7.2f}x")


if __name__ == "__main__":
    main()
