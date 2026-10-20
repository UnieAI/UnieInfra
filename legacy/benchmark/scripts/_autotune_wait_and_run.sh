#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Wait for an idle GPU (never GPU 0 -- another user's), then run the autotune
# validation on it. Kernel timing is only trustworthy on an uncontended device,
# and running on a busy one would also perturb whoever is already there.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${REPO}/benchmark/results/autotune"
mkdir -p "${OUT_DIR}"
DEADLINE=$(( $(date +%s) + ${WAIT_SECONDS:-5400} ))

pick_idle_gpu() {
  # Only 1/2/3 are candidates. Require 0 MiB so the box is genuinely ours.
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', *' '$1 != 0 && $2 == 0 { print $1; exit }'
}

GPU=""
while [ -z "${GPU}" ]; do
  GPU="$(pick_idle_gpu)"
  [ -n "${GPU}" ] && break
  if [ "$(date +%s)" -ge "${DEADLINE}" ]; then
    echo "TIMED OUT waiting for an idle GPU" | tee "${OUT_DIR}/STATUS"
    exit 3
  fi
  sleep 30
done

echo "claimed GPU ${GPU} at $(date -u +%FT%TZ)" | tee "${OUT_DIR}/STATUS"

run() {
  local arm="$1"; shift
  echo "=== arm=${arm} $* ==="
  python3 "${REPO}/benchmark/scripts/autotune_validate.py" \
    --device "${GPU}" --arm "${arm}" \
    --out "${OUT_DIR}/${arm}.json" "$@"
}

{
  run gated
  echo "RC_GATED=$?"
  run forced
  echo "RC_FORCED=$?"
} 2>&1 | tee "${OUT_DIR}/validate.log"

echo "done at $(date -u +%FT%TZ) on GPU ${GPU}" | tee -a "${OUT_DIR}/STATUS"
