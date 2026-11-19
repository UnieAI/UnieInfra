#!/bin/bash
# One official-window (3600 s) A/B cell for our engine, to test a scheduling knob
# against the plain official cell at the same concurrency and cap.
#
# Motivation: at c64 cap16 the official run is 93% decode-busy, yet its ITL
# (p90 62 ms) is far worse than the same cap in the 900 s screen (20.6 ms). The
# difference is arrivals: the official window keeps the trace delays, so a 121K
# prompt is prefilled in ~30 chunked rounds while other lanes are decoding, and
# outside mixed mode those prefills run instead of decode rather than beside it.
#
# Usage: official_ab.sh <tag> [conc]        tag e.g. mixoff16, decoff16
#   CAP=16 MAX_NUM_SEQS=16 EXTRA_SERVE_ARGS=... TOKENSPEED_...=1 official_ab.sh ...
set -uo pipefail
ROOT=/home/smc/workspace/roy
TAG=${1:?tag, e.g. mixoff16}
CONC=${2:-64}
CAP=${CAP:-16}
LOG=$ROOT/logs/official-ab-$(date -u +%Y%m%dT%H%M%SZ).log

# A second server here would fight the sweep for the GPUs and produce a cell that
# measures contention instead of batching, so refuse rather than race.
used=$(docker exec -u root ts-dev rocm-smi --showmeminfo vram 2>/dev/null \
  | grep -oP 'Used Memory \(B\): \K\d+' | sort -n | tail -1)
if [[ ${used:-0} -ge 20000000000 ]]; then
  echo "GPUs are busy (${used} B in use); wait for the official sweep to finish" >&2
  exit 1
fi

echo "=== $(date -u +%FT%TZ) $TAG official c$CONC cap$CAP, extra='${EXTRA_SERVE_ARGS:-}' log $LOG"
env MAX_NUM_SEQS="$CAP" \
  "$ROOT/scripts/run_round.sh" "$TAG" official unieinfra "$CONC" 2>&1 | tee -a "$LOG"
echo "=== $(date -u +%FT%TZ) AB DONE (expect dir unieinfra-$TAG-c$CONC-dofficial-*)"
