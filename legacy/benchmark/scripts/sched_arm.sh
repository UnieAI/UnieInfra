#!/usr/bin/env bash
# One benchmark arm for the scheduling study (E36): launch a vLLM server on a
# single GPU, wait for readiness, run a concurrency sweep, then kill exactly the
# PIDs this script started and verify the GPU is back to 0 MiB.
#
# Arms are distinguished only by `SCHED_CLS` / `SCHED_OBJECTIVE`; every vLLM
# flag is identical across arms so the torch.compile cache key (which covers
# `VllmConfig`, not the environment -- see E34) is shared and all arms replay
# byte-identical compiled graphs. Only the scheduling policy differs.
#
# Usage:
#   LABEL=e36_throughput_r1 SCHED_CLS=... SCHED_OBJECTIVE=throughput \
#   CONCURRENCIES="1 16 32 64 128 256" ./sched_arm.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LABEL="${LABEL:?set LABEL=<run label>}"
GPU="${GPU:-3}"
PORT="${PORT:-8011}"
MODEL="${MODEL:-/home/ubuntu/models/Qwen/Qwen3-8B}"
ENGINE="${ENGINE:-vllm}"
MODEL_PATH="$MODEL"   # keep the on-disk path; MODEL is overridden to the served name below
CONCURRENCIES="${CONCURRENCIES:-1 16 32 64 128 256}"
LOGDIR="${LOGDIR:-/tmp/uiwork}"
mkdir -p "$LOGDIR"
SRVLOG="${LOGDIR}/${LABEL}.server.log"

echo "[arm] $LABEL  sched_cls=${SCHED_CLS:-<stock>} objective=${SCHED_OBJECTIVE:-<n/a>}"

used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 100 ]; then echo "[arm] GPU $GPU not clean ($used MiB) -- refusing"; exit 1; fi

MODEL="$MODEL" PORT="$PORT" GPU="$GPU" \
  SCHED_CLS="${SCHED_CLS:-}" SCHED_OBJECTIVE="${SCHED_OBJECTIVE:-}" \
  setsid bash "$REPO/benchmark/launch/launch_vllm.sh" > "$SRVLOG" 2>&1 &
SRVPID=$!
echo "[arm] server pgid=$SRVPID log=$SRVLOG"

for _ in $(seq 1 120); do
  if grep -q "Application startup complete" "$SRVLOG" 2>/dev/null; then break; fi
  if grep -qE "Traceback|ValueError|RuntimeError|EngineDeadError" "$SRVLOG" 2>/dev/null; then
    echo "[arm] server failed to start; tail:"; tail -40 "$SRVLOG"; kill -- -"$SRVPID" 2>/dev/null || true; exit 1
  fi
  sleep 5
done
grep -q "Application startup complete" "$SRVLOG" || { echo "[arm] timeout"; kill -- -"$SRVPID" 2>/dev/null || true; exit 1; }

# Record what the policy actually derived -- evidence, not assumption.
grep -h "unieinfra_scheduler:" "$SRVLOG" | head -5 || true

set +e
ENGINE="$ENGINE" BASE_URL="http://127.0.0.1:${PORT}" MODEL="bench-model" \
  TOKENIZER="$MODEL_PATH" VLLM_PY=/home/ubuntu/vllm-bench-venv/bin/python \
  CONCURRENCIES="$CONCURRENCIES" TS="$LABEL" \
  bash "$REPO/benchmark/run_sweep.sh"
RC=$?
set -e

# Ask the engine to dump the gate histogram before we tear it down.
curl -sf -X POST "http://127.0.0.1:${PORT}/reset_prefix_cache" >/dev/null 2>&1 || true

kill -- -"$SRVPID" 2>/dev/null || true
for _ in $(seq 1 60); do
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 100 ] && break
  sleep 2
done
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
echo "[arm] $LABEL done (rc=$RC); GPU $GPU now ${used} MiB"
exit $RC
