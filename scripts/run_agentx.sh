#!/bin/bash
# Run the InferenceX AgentX benchmark (agentx-harness 1.0.13) against a live server.
# Usage: run_agentx.sh <engine-label> <CONC> [DURATION_SECONDS]
#   DURATION_SECONDS omitted = the agentx preset (one-hour profile, InferenceX default).
# Dataset/seed follow inferencex-e2e/docs/agentx-standalone.md (full corpus, seed 42).
set -eo pipefail
ENGINE=${1:?engine label (vllm|sglang|unieinfra)}
CONC=${2:?concurrency}
DURATION=${3:-}
ROOT=/home/smc/workspace/roy
SERVER_URL=${SERVER_URL:-http://127.0.0.1:8000}
MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017
TOKENIZER=/data/models/DeepSeek-V4.1-Flash-1017
DATASET=semianalysis_cc_traces_weka_062126
TAG=${TAG:-}
# OFFICIAL mode keeps the preset's own profile length; the output directory still
# has to say which setting produced it, so it is labelled -dofficial.
DURLABEL=${DURATION:+-d$DURATION}
if [[ "$DURATION" == "official" ]]; then DURATION=""; DURLABEL=-dofficial; fi
OUT=$ROOT/results/agentx/${ENGINE}${TAG:+-$TAG}-c${CONC}${DURLABEL}-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"

# OFFICIAL mode (DURATION empty): pass only the caller inputs from
# inferencex-e2e/docs/agentx-standalone.md and let the agentx preset supply its
# own profile length and warmup -- "the agentx preset supplies a one-hour profile,
# ten warmup requests per lane". Overriding either one deviates from the
# documented recipe, so the sweep runs with neither flag set.
EXTRA=()
if [[ -n "$DURATION" ]]; then
  EXTRA+=(--benchmark-duration "$DURATION" --warmup-requests-per-lane 2)
  # The agentx scenario requires >= 900 s; shorter runs are smoke tests only
  # (the harness marks them submission_valid=false).
  if (( DURATION < 900 )); then EXTRA+=(--unsafe-override); fi
fi
# STRESS=1: replay the corpus without the recorded user think time, so every
# lane sends its next turn as soon as the last one finishes and the server is
# saturated. Not a valid AgentX submission (the scenario forbids it).
if [[ "${STRESS:-0}" == 1 ]]; then
  EXTRA+=(--ignore-trace-delays)
  [[ " ${EXTRA[*]} " == *" --unsafe-override "* ]] || EXTRA+=(--unsafe-override)
fi
case "$ENGINE" in
  vllm*) EXTRA+=(--server-metrics "$SERVER_URL/metrics") ;;
  sglang*) EXTRA+=(--server-metrics "$SERVER_URL/metrics") ;;
esac

echo "$OUT"
"$ROOT/.venv-agentx/bin/agentx" profile \
  --url "$SERVER_URL" --endpoint /v1/chat/completions \
  --model "$MODEL" --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --concurrency "$CONC" \
  --public-dataset "$DATASET" \
  --random-seed 42 \
  --ui-type simple \
  --output-artifact-dir "$OUT/aiperf_artifacts" \
  "${EXTRA[@]}" \
  2>&1 | tee "$OUT/aiperf.log"
