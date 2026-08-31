#!/bin/bash
# No-replay agent/SLO matrix: one fresh server per cell, engine warmed on a
# corpus that shares no prefix with the measured one, target corpus run ONCE.
#
#   MODEL=/path/to/model ENGINE=ours|vllm GPU=1 CONCS="8 16 32" ./run_matrix.sh
#
# Why the protocol looks like this: a concurrency ladder that reruns the same
# prompts on one server measures REPLAY, not reuse. Measured on this harness,
# three passes over the same 112 prompts moved the prefix hit rate
# 0% -> 59.5% -> 71.3% -> 81.5% -> 90.7% -> 93.0% -> 94.4% -> 95.2%, and the
# same cell's wall time by 51.7%. Every number below is a first pass.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:?set MODEL to a model directory}"
ENGINE="${ENGINE:-ours}"
GPU="${GPU:-0}"
CONCS="${CONCS:-8 16 32}"
NREQ="${NREQ:-256}"
MML="${MML:-32768}"
GMU="${GMU:-0.90}"
MAXSEQ="${MAXSEQ:-452}"
DATA="${DATA:-$HERE/corpora/swe_smith_agent_sess32x8.jsonl}"
WARM="${WARM:-$HERE/corpora/swe_agent_nebius_sess16x16.jsonl}"
OUT="${OUT:-$HERE/results}"
mkdir -p "$OUT"

# The warmup corpus must not share a prefix with the measured one, or the
# "warmup" is a cache preload and every number after it is replay.
python3 - "$DATA" "$WARM" <<'PY' || exit 1
import json, sys
def load(p): return [json.loads(l)["prompt"] for l in open(p)][:20]
a, b = load(sys.argv[1]), load(sys.argv[2])
def lcp(x, y):
    n = min(len(x), len(y)); i = 0
    while i < n and x[i] == y[i]: i += 1
    return i
worst = max(lcp(x, y) for x in a for y in b)
print(f"warmup/measurement longest shared prefix: {worst} chars")
if worst > 256:
    sys.exit("REFUSING: warmup corpus shares a prefix with the measured corpus")
PY

free_gpu () {
  for _ in 1 2 3; do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" |
      while read -r p; do kill -9 "$p" 2>/dev/null; done
    sleep 12
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    if [ "${u:-99999}" -lt 2000 ]; then
      sleep 8
      u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
      [ "${u2:-99999}" -lt 2000 ] && return 0
    fi
  done
  echo "GPU $GPU still busy" >&2; return 1
}

pick_port () {
  python3 - <<'PY'
import socket
for p in range(20500, 21000):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); print(p); break
    except OSError:
        s.close()
PY
}

for c in $CONCS; do
  free_gpu || exit 1
  PORT=$(pick_port)
  LOG="$OUT/$ENGINE-c$c.log"
  if [ "$ENGINE" = ours ]; then
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup \
      python3 -m tokenspeed.cli serve "$MODEL" --port "$PORT" --served-model-name bench \
        --max-model-len "$MML" --gpu-memory-utilization "$GMU" --max-num-seqs "$MAXSEQ" \
        --enable-metrics --enable-cache-report >"$LOG" 2>&1 </dev/null &
  else
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup \
      python3 -m vllm.entrypoints.cli.main serve "$MODEL" --port "$PORT" --served-model-name bench \
        --max-model-len "$MML" --gpu-memory-utilization "$GMU" --max-num-seqs "$MAXSEQ" \
        --enable-prefix-caching >"$LOG" 2>&1 </dev/null &
  fi
  # Ready means it GENERATES, not that /v1/models answers: the gateway
  # registers the route before a worker is attached.
  ok=0
  for _ in $(seq 150); do
    t=$(curl -s -m 25 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"bench","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":8,"temperature":0}' 2>/dev/null)
    case "$t" in *'"content"'*) ok=1; break;; esac
    sleep 8
  done
  [ "$ok" = 1 ] || { echo "$ENGINE c=$c never became ready; see $LOG" >&2; continue; }
  python3 "$HERE/drive.py" "http://127.0.0.1:$PORT" bench "$DATA" "$c" "$NREQ" \
          "$OUT/$ENGINE-c$c.json" "$WARM"
done
free_gpu || true
echo "results in $OUT"
