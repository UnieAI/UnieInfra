#!/usr/bin/env bash
# UnieInfra benchmark harness — concurrency sweep against a running OpenAI endpoint.
#
# Measures TTFT / TPOT / throughput / RPS at concurrency 1,16,32,64,128,256 using
# `vllm bench serve` (the InferenceMAX-standard client), saving ALL raw results
# (per-run + per-request via --save-detailed). Engine-agnostic: point it at any
# OpenAI-compatible server (UnieInfra, vLLM, SGLang, LMDeploy).
#
# Usage:
#   ENGINE=unieinfra BASE_URL=http://127.0.0.1:8000 MODEL=Qwen3-8B \
#   VLLM_PY=/home/ubuntu/vllm-bench-venv/bin/python \
#   ./run_sweep.sh
set -euo pipefail

ENGINE="${ENGINE:?set ENGINE=unieinfra|vllm|sglang|lmdeploy}"
BASE_URL="${BASE_URL:?set BASE_URL=http://host:port}"
MODEL="${MODEL:?set MODEL=<served-model-name or path>}"
VLLM_PY="${VLLM_PY:-python3}"                       # a python with vllm installed (for `vllm bench`)
TOKENIZER="${TOKENIZER:-$MODEL}"                    # local tokenizer path (client-side); MODEL is the served name
ENDPOINT="${ENDPOINT:-/v1/completions}"
CONCURRENCIES="${CONCURRENCIES:-1 16 32 64 128 256}"
ILEN="${ILEN:-1024}"                                # random input length
OLEN="${OLEN:-256}"                                 # random output length
PROMPTS_PER_CONC="${PROMPTS_PER_CONC:-8}"           # num_prompts = concurrency * this (steady state). FIXED across runs — do not vary when comparing.
NUM_PROMPTS_CAP="${NUM_PROMPTS_CAP:-1024}"          # cap so high concurrency doesn't balloon
WARMUPS="${WARMUPS:-8}"                             # warmup requests before EACH measured level (excluded from metrics)
SEED="${SEED:-0}"                                   # fixed seed -> identical prompt set every run
FLUSH="${FLUSH:-1}"                                 # flush server cache before each level (needs an OpenAI-sidecar /flush_cache)

# Timestamp is passed in (scripts can't call date in some sandboxes); default to epoch via bash.
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || echo run)}"
OUTDIR="$(cd "$(dirname "$0")" && pwd)/results/${ENGINE}/${TS}"
RAW="${OUTDIR}/raw"
mkdir -p "$RAW"

# ---- capture the engine's launch config (card, count, serve params) ----
LAUNCH_JSON="$(dirname "$0")/results/${ENGINE}.launch.json"
if [ -f "$LAUNCH_JSON" ]; then cp "$LAUNCH_JSON" "${OUTDIR}/launch.json"; fi

# ---- capture environment for reproducibility ----
{
  if [ -f "${OUTDIR}/launch.json" ]; then echo "--- launch config (card / count / serve params) ---"; cat "${OUTDIR}/launch.json"; echo; fi
  echo "engine=$ENGINE"; echo "base_url=$BASE_URL"; echo "model=$MODEL"
  echo "endpoint=$ENDPOINT"; echo "concurrencies=$CONCURRENCIES"
  echo "input_len=$ILEN output_len=$OLEN prompts_per_conc=$PROMPTS_PER_CONC"
  echo "--- gpu ---"; nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv 2>/dev/null || true
  echo "--- vllm client ---"; "$VLLM_PY" -c "import vllm;print('vllm',vllm.__version__)" 2>/dev/null || true
} > "${OUTDIR}/env.txt"

echo "[sweep] engine=$ENGINE model=$MODEL -> $OUTDIR"
for C in $CONCURRENCIES; do
  NP=$(( C * PROMPTS_PER_CONC )); [ "$NP" -gt "$NUM_PROMPTS_CAP" ] && NP=$NUM_PROMPTS_CAP; [ "$NP" -lt 16 ] && NP=16
  # --- hygiene: flush server cache so this level doesn't ride on a prior level's cached prefixes ---
  if [ "$FLUSH" = "1" ]; then curl -sf -X POST "${BASE_URL}/flush_cache" >/dev/null 2>&1 || curl -sf "${BASE_URL}/flush_cache" >/dev/null 2>&1 || curl -sf -X POST "${BASE_URL}/reset_prefix_cache" >/dev/null 2>&1 || true; fi
  echo "[sweep] concurrency=$C  num_prompts=$NP  warmups=$WARMUPS  seed=$SEED ..."
  "$VLLM_PY" -m vllm.entrypoints.cli.main bench serve \
    --backend openai \
    --base-url "$BASE_URL" --endpoint "$ENDPOINT" \
    --model "$MODEL" --tokenizer "$TOKENIZER" \
    --dataset-name random --random-input-len "$ILEN" --random-output-len "$OLEN" \
    --num-prompts "$NP" --max-concurrency "$C" --request-rate inf \
    --num-warmups "$WARMUPS" --seed "$SEED" \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed \
    --result-dir "$RAW" --result-filename "c${C}.json" \
    > "${RAW}/c${C}.stdout.log" 2>&1 \
    || { echo "[sweep] concurrency=$C FAILED (see ${RAW}/c${C}.stdout.log)"; continue; }
done

# ---- summarize all raw JSON into one CSV ----
python3 "$(dirname "$0")/summarize.py" "$RAW" "${OUTDIR}/summary.csv" "$ENGINE"
echo "[sweep] DONE. raw: $RAW   summary: ${OUTDIR}/summary.csv"
cat "${OUTDIR}/summary.csv"
