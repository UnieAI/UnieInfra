#!/usr/bin/env bash
# High-concurrency driver: mirrors run_sweep.sh's vllm-bench invocation exactly,
# but sweeps C=128/192/256/320 and slices the server log per level (running-req,
# queue-req, page ratio) for root-cause evidence. Read-only wrt engines.
set -uo pipefail
ENGINE="${ENGINE:?}"; BASE_URL="${BASE_URL:?}"; MODEL="${MODEL:?}"
VLLM_PY="${VLLM_PY:-/home/ubuntu/vllm-bench-venv/bin/python}"
TOKENIZER="${TOKENIZER:-/workspace/models/Qwen/Qwen3-8B}"
ENDPOINT="${ENDPOINT:-/v1/completions}"
CONCURRENCIES="${CONCURRENCIES:-128 192 256 320}"
ILEN="${ILEN:-1024}"; OLEN="${OLEN:-256}"
PROMPTS_PER_CONC="${PROMPTS_PER_CONC:-8}"; NUM_PROMPTS_CAP="${NUM_PROMPTS_CAP:-1024}"
WARMUPS="${WARMUPS:-8}"; SEED="${SEED:-0}"; FLUSH="${FLUSH:-1}"
SERVER_LOG="${SERVER_LOG:-}"          # host-visible path to the engine's stdout log (for slicing)
OUTDIR="${OUTDIR:?}"; RAW="${OUTDIR}/raw"; mkdir -p "$RAW"
SLICES="${OUTDIR}/server_slices"; mkdir -p "$SLICES"

echo "[hc] engine=$ENGINE -> $OUTDIR"
{ echo "engine=$ENGINE base_url=$BASE_URL model=$MODEL"; echo "conc=$CONCURRENCIES ilen=$ILEN olen=$OLEN ppc=$PROMPTS_PER_CONC cap=$NUM_PROMPTS_CAP warmups=$WARMUPS seed=$SEED";
  echo "--- gpu at start ---"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; } > "${OUTDIR}/env.txt"

for C in $CONCURRENCIES; do
  NP=$(( C * PROMPTS_PER_CONC )); [ "$NP" -gt "$NUM_PROMPTS_CAP" ] && NP=$NUM_PROMPTS_CAP; [ "$NP" -lt 16 ] && NP=16
  if [ "$FLUSH" = "1" ]; then curl -sf -X POST "${BASE_URL}/flush_cache" >/dev/null 2>&1 || curl -sf "${BASE_URL}/flush_cache" >/dev/null 2>&1 || curl -sf -X POST "${BASE_URL}/reset_prefix_cache" >/dev/null 2>&1 || true; fi
  N0=0; [ -n "$SERVER_LOG" ] && [ -f "$SERVER_LOG" ] && N0=$(wc -l < "$SERVER_LOG")
  echo "[hc] C=$C NP=$NP ..."
  GPUSTART=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';')
  "$VLLM_PY" -m vllm.entrypoints.cli.main bench serve \
    --backend openai --base-url "$BASE_URL" --endpoint "$ENDPOINT" \
    --model "$MODEL" --tokenizer "$TOKENIZER" \
    --dataset-name random --random-input-len "$ILEN" --random-output-len "$OLEN" \
    --num-prompts "$NP" --max-concurrency "$C" --request-rate inf \
    --num-warmups "$WARMUPS" --seed "$SEED" --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir "$RAW" --result-filename "c${C}.json" \
    > "${RAW}/c${C}.stdout.log" 2>&1 || { echo "[hc] C=$C FAILED"; }
  echo "gpu_at_c${C}_start=$GPUSTART" >> "${OUTDIR}/env.txt"
  if [ -n "$SERVER_LOG" ] && [ -f "$SERVER_LOG" ]; then
    N1=$(wc -l < "$SERVER_LOG"); sed -n "$((N0+1)),${N1}p" "$SERVER_LOG" > "${SLICES}/c${C}.serverlog"
  fi
done
python3 /workspace/roy/UnieInfra/benchmark/summarize.py "$RAW" "${OUTDIR}/summary.csv" "$ENGINE" || true
echo "[hc] DONE $OUTDIR"
