#!/usr/bin/env bash
# E28 MTP ablation launcher.
#
# Brings up a UnieInfra OpenAI server with an IDENTICAL configuration for both
# arms of the ablation -- the ONLY difference is the --speculative-* flags -- so
# the baseline-vs-MTP delta is attributable to speculative decoding alone.
#
# It also owns process hygiene, which matters here: this box is shared with
# other agents' jobs on other GPUs, so cleanup targets ONLY servers this script
# started (matched by model path) and it refuses to launch until the target GPU
# is actually free. A stale scheduler still holding HBM silently turns the next
# launch into an OOM that looks like a model bug.
#
# Usage:
#   SPEC=off|on MODEL=/path/to/model GPU=1 PORT=8100 LOG=/path/to/log \
#     benchmark/launch/launch_mtp_ablation.sh
set -euo pipefail
MODEL="${MODEL:?set MODEL=/path/to/model}"
LOG="${LOG:?set LOG=/path/to/logfile}"
PORT="${PORT:-8100}"; GPU="${GPU:-1}"
SPEC="${SPEC:-off}"; STEPS="${STEPS:-3}"; DRAFT="${DRAFT:-4}"
GMU="${GMU:-0.75}"; MAXLEN="${MAXLEN:-4096}"; MAXSEQ="${MAXSEQ:-128}"
CONTAINER="${CONTAINER:-unieinfra-dev}"
HOST_IP="${HOST_IP:-$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CONTAINER")}"

SPEC_ARGS=""
if [ "$SPEC" = "on" ]; then
  SPEC_ARGS="--speculative-algorithm MTP --speculative-num-steps $STEPS --speculative-num-draft-tokens $DRAFT"
fi

# --- 1. kill only OUR previous server, by the process GROUP we recorded ---
# `tokenspeed serve` fans out into an smg gateway (Rust, owns the HTTP port), a
# python servicer, and a detached `scheduler_0` worker that holds the HBM.
# Killing by cmdline pattern reaches only one of them: the survivors keep the
# port answering (so readiness checks pass against a dead engine) and keep the
# GPU allocated (so the next launch OOMs). Track the setsid process group id
# instead and kill the whole group -- that is exactly our processes and nobody
# else's, which matters on a box shared with other agents' jobs.
PGID_FILE="/tmp/e28_mtp_ablation.pgid"
docker exec "$CONTAINER" bash -lc "
  if [ -f '$PGID_FILE' ]; then kill -9 -\$(cat '$PGID_FILE') 2>/dev/null || true; rm -f '$PGID_FILE'; fi
" >/dev/null 2>&1 || true
for _ in $(seq 1 36); do
  USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$USED" -lt 2000 ] && break
  sleep 5
done
USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$USED" -ge 2000 ]; then
  echo "[launch] ABORT: GPU $GPU still holds ${USED} MiB (someone else's job?). Not launching." >&2
  exit 1
fi

# --- 2. launch detached ---
docker exec -d "$CONTAINER" bash -lc "
source benchmark/scripts/sm120_env.sh
export CUDA_VISIBLE_DEVICES=$GPU
${EXTRA_ENV:-}
setsid nohup tokenspeed serve '$MODEL' \
  --served-model-name bench-model --host 0.0.0.0 --port $PORT \
  --dtype bfloat16 --gpu-memory-utilization $GMU --max-model-len $MAXLEN \
  --attention-backend triton --sampling-backend greedy \
  --max-num-seqs $MAXSEQ --max-cudagraph-capture-size $MAXSEQ \
  --mamba-full-memory-ratio 3.0 --no-enable-prefix-caching \
  --trust-remote-code --disable-kvstore $SPEC_ARGS \
  > '$LOG' 2>&1 &
SPID=\$!
sleep 2
# setsid makes the child its own session/group leader, so its PGID is the
# handle for every process the server later forks (gateway + scheduler).
{ ps -o pgid= -p \$SPID 2>/dev/null | tr -d ' ' | grep . || echo \$SPID; } > '$PGID_FILE'"

# --- 3. wait for readiness, failing fast and loudly on a startup crash ---
for _ in $(seq 1 120); do
  if curl -sf "http://${HOST_IP}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[launch] READY spec=$SPEC model=$MODEL url=http://${HOST_IP}:${PORT}"
    grep -o "Scheduler config:.*" "$LOG" | head -1 | grep -o \
      "max_batch_size=[0-9]*\|mamba_pool_total_chunks=[0-9]*\|num_device_pages=[0-9]*" || true
    exit 0
  fi
  if grep -q "startup failed\|hit an exception" "$LOG" 2>/dev/null; then
    echo "[launch] FAILED spec=$SPEC -- see $LOG" >&2
    exit 1
  fi
  sleep 5
done
echo "[launch] TIMEOUT waiting for $MODEL on port $PORT" >&2
exit 1
