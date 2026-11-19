#!/bin/bash
# GSM8K accuracy check, same settings as unieinfra's AMD CI
# (test/ci/eval/deepseek-v4.1-flash-dspark-evalscope-gsm8k-amd.yaml, threshold 0.90).
# Usage: run_gsm8k.sh <engine-label>
set -eo pipefail
ENGINE=${1:?engine label}
ROOT=/home/smc/workspace/roy
OUT=$ROOT/results/gsm8k/$ENGINE-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"; cd "$OUT"
HF_TOKEN=$(cat ~/.cache/huggingface/token) "$ROOT/.venv-eval/bin/evalscope" eval \
  --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --api-url http://127.0.0.1:8000/v1 --api-key EMPTY_TOKEN \
  --datasets gsm8k --dataset-hub huggingface \
  --dataset-args '{"gsm8k":{"dataset_id":"openai/gsm8k"}}' \
  --eval-batch-size 32 \
  --generation-config '{"do_sample":false,"temperature":0.0,"max_tokens":512}' \
  --work-dir "$OUT" 2>&1 | tee "$OUT/eval.log"
