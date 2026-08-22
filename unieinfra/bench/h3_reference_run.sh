#!/usr/bin/env bash
# Run MiniMax-H3 on the SGLang Diffusion reference engine and collect the
# stage-level attribution. This is the parity baseline: H3 is supported by
# exactly one of the surveyed engines (xref/16-diffusion.md §1), so its
# numbers are what "追平" means here.
#
# Adapted from their own preset -- multimodal_gen/.claude/skills/
# sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py,
# case "minimax-h3-t2va" -- with ONE deliberate change: their nightly runs
# 4 GPUs (--tp-size=2 --ulysses-degree=2); this box has one card free at
# best, so parallel degrees are 1 and the text encoder / VAE are CPU-offloaded
# to fit the DiT. Any number from here is a SINGLE-CARD number and is not
# comparable to their published 4xH200 figures.
#
# Usage:  h3_reference_run.sh smoke|nightly <gpu-id>
set -euo pipefail

MODE="${1:?usage: $0 smoke|nightly <gpu-id>}"
GPU="${2:?usage: $0 smoke|nightly <gpu-id>}"

REF=/workspace/roy/sglang-diffusion
SNAP=/workspace/.cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe
OUT="${H3_OUT_DIR:-$REF/results/$MODE-gpu$GPU-$(date +%Y%m%d-%H%M%S)}"

# --- Gate 1: refuse to start rather than OOM halfway. The DiT is 66.28 GB
# bf16; with the text encoder and VAEs offloaded it still needs headroom for
# a ~48k-row packed sequence. Fail closed with the number the operator needs.
NEED_MIB=78000
FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
       | awk -F', *' '{print $1-$2}')
if [ "$FREE" -lt "$NEED_MIB" ]; then
  echo "REFUSING: GPU $GPU has ${FREE} MiB free, need ${NEED_MIB} MiB." >&2
  echo "  (H3 DiT is 66.28 GB bf16; text_encoder and VAEs are CPU-offloaded here.)" >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv >&2
  exit 2
fi

mkdir -p "$OUT"
# shellcheck disable=SC1091
source "$REF/env.sh"

case "$MODE" in
  # Cheap tier first, per TRT-LLM ENGINEERING_CRITERIA.md §5.2 ("reduced
  # denoising steps and resolution are typical"). Proves it runs before a
  # full run is spent.
  smoke)   SHORT_EDGE=384; DURATION=4.0; STEPS=8  ;;
  nightly) SHORT_EDGE=768; DURATION=5.0; STEPS=50 ;;
  *) echo "unknown mode $MODE" >&2; exit 64 ;;
esac

cat > "$OUT/config.json" <<JSON
{
  "task": "t2va",
  "conditions": [],
  "target": {"short_edge": $SHORT_EDGE, "aspect_ratio": "16:9", "duration_seconds": $DURATION},
  "audio_flow_shift": 3.0,
  "flow_shift": 12.0,
  "num_inference_steps": $STEPS
}
JSON

# Their preset's prompt and seed, kept verbatim so the output is comparable
# to the reference the nightly case produces.
PROMPT="At night, while their owner sleeps in a bedroom, three cats march in loudly playing tiny brass instruments, then abruptly file out."
printf '%s' "$PROMPT" > "$OUT/prompt.txt"

set +e
sglang generate \
  --model-path "$SNAP" \
  --model-variant=fl2va \
  --config "$OUT/config.json" \
  --prompt-file-path "$OUT/prompt.txt" \
  --seed 1101 \
  --gpu-ids "$GPU" \
  --num-gpus=1 --tp-size=1 --sp-degree=1 --ulysses-degree=1 --ring-degree=1 \
  --performance-mode=speed \
  --enable-torch-compile=false \
  --text-encoder-cpu-offload=true \
  --vae-cpu-offload=true \
  --perf-dump-path "$OUT/perf.json" \
  --output-file-path "$OUT/out" \
  2>&1 | tee "$OUT/run.log"
RC=${PIPESTATUS[0]}
set -e

# --- Gate 2: SGLang's own Native Backend Gate. A run that fell back to
# diffusers is not evidence about their engine, and its numbers must not be
# kept. Their skill makes this a hard stop; so does this script.
if grep -qiE "Falling back to diffusers backend|Using diffusers backend|Loaded diffusers pipeline" "$OUT/run.log"; then
  echo "NATIVE BACKEND GATE TRIPPED -- this run fell back to diffusers." >&2
  echo "Numbers and traces in $OUT are NOT SGLang evidence. Fix backend selection first." >&2
  exit 3
fi

echo "exit=$RC  results in $OUT"
exit "$RC"
