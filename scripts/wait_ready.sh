#!/bin/bash
# Wait until the server actually generates: a 1-token chat completion must succeed.
# (/v1/models alone is not enough: unieinfra's gateway keeps answering it after
# the engine behind it has died.) Usage: wait_ready.sh [timeout_s] [launch_log]
# Second arg = this launch's server log (unieinfra writes one per launch): then the
# probe only counts once that log shows the engine ready, and a fatal start error
# aborts immediately instead of burning the whole timeout. Without it, a crashed
# c64 cap=16 start (2026-11-09 03:14, OOM at RCCL init) spun for 2260 s and then
# reported a false READY off the *next* cell's server on the same port.
T=${1:-1800}; URL=${SERVER_URL:-http://127.0.0.1:8000}; LOG=${2:-}
FATAL='Segmentation fault|HIP failure|startup failed|engine never reached SERVING'
READY_RE='engine ready: max_total_num_tokens'
BODY='{"model":"deepseek-ai/DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
log_ready=0
for ((s=0; s<T; s+=10)); do
  if [[ -n $LOG ]]; then
    if grep -qE "$FATAL" "$LOG" 2>/dev/null; then
      echo "SERVER DIED after ${s}s: $(grep -E "$FATAL" "$LOG" | tail -1 | cut -c1-200)"
      exit 1
    fi
    if (( ! log_ready )) && grep -qE "$READY_RE" "$LOG" 2>/dev/null; then
      log_ready=1
    fi
    (( log_ready )) || { sleep 10; continue; }
  fi
  if curl -sf -m 60 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" \
       | grep -q '"choices"'; then
    echo "READY after ${s}s"; exit 0
  fi
  sleep 10
done
echo "NOT READY after ${T}s"; exit 1
