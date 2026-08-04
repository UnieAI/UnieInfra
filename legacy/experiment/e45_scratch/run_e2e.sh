#!/usr/bin/env bash
set -eo pipefail
GPU=""
for _ in $(seq 1 480); do
  for i in 3 2 1; do
    used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -lt 3000 ]; then GPU=$i; break; fi
  done
  [ -n "$GPU" ] && break
  sleep 15
done
[ -z "$GPU" ] && { echo "NO FREE GPU"; exit 1; }
echo "[e45-e2e] GPU $GPU"
exec env GPU=$GPU /workspace/roy/UnieInfra/benchmark/scripts/run_e45_e2e_ab.sh "$@"
