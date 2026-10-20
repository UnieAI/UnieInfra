#!/usr/bin/env bash
# Poll until a card is genuinely free, claim it ONCE, run smoke, and on
# success run nightly under the SAME claim -- releasing between the two would
# put the card back in the pool on a shared box.
# GPU 2 is a third party's and is never a candidate.
#
# Usage:  HSESSION=<your-session> unieinfra/bench/h3_wait_and_run.sh
set -uo pipefail
. /workspace/roy/.owner_lib.sh
# A committed script must not hardcode one session's identity: an
# unattributable claim says the card is busy but not who to ask, which is most
# of a claim's value gone.
export HSESSION="${HSESSION:?set HSESSION to your session name before running}"
NEED=78000
OUTDIR="${H3_LOG_DIR:-/workspace/roy/sglang-diffusion/results}"
LOG="$OUTDIR/h3_runs.log"
RUN="$(cd "$(dirname "$0")" && pwd)/h3_reference_run.sh"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for a card with >= ${NEED} MiB free (candidates: 0 1 3)"
tick=0
while :; do
  for g in 0 1 3; do
    free=$(nvidia-smi --id=$g --query-gpu=memory.total,memory.used \
           --format=csv,noheader,nounits 2>/dev/null | awk -F', *' '{print $1-$2}')
    [ -z "${free:-}" ] && continue
    [ "$free" -lt "$NEED" ] && continue
    owner=$(howns $g 2>/dev/null)
    # howns prints "free (no claim -- ...)", not a bare "free"; an exact match
    # here skipped every free card for two minutes before this was caught.
    case "$owner" in
      free*) ;;
      *) say "GPU $g has ${free} MiB free but howns says: $owner -- skipping"; continue ;;
    esac
    say "GPU $g free (${free} MiB), claiming"
    hclaim $g "H3 smoke+nightly on SGLang reference (parity baseline)" >>"$LOG" 2>&1
    trap "hrelease $g >/dev/null 2>&1" EXIT
    free2=$(nvidia-smi --id=$g --query-gpu=memory.total,memory.used \
            --format=csv,noheader,nounits | awk -F', *' '{print $1-$2}')
    if [ "$free2" -lt "$NEED" ]; then
      say "GPU $g lost the race (${free2} MiB) -- releasing, keep waiting"
      hrelease $g >/dev/null 2>&1; trap - EXIT; continue
    fi

    say "=== SMOKE on GPU $g (384 short edge, 4.0 s, 8 steps)"
    bash "$RUN" smoke $g >>"$LOG" 2>&1
    rc=$?
    say "smoke exit=$rc"
    if [ $rc -ne 0 ]; then
      say "smoke failed -- NOT running nightly; releasing GPU $g"
      hrelease $g >>"$LOG" 2>&1; trap - EXIT
      exit $rc
    fi

    say "=== NIGHTLY on GPU $g (768 short edge, 5.0 s, 50 steps), same claim"
    bash "$RUN" nightly $g >>"$LOG" 2>&1
    rc=$?
    say "nightly exit=$rc"
    hrelease $g >>"$LOG" 2>&1; trap - EXIT
    exit $rc
  done
  tick=$((tick+1))
  if [ $((tick % 20)) -eq 0 ]; then
    say "still waiting; free MiB: $(for g in 0 1 3; do nvidia-smi --id=$g --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', *' -v g=$g '{printf "gpu%s=%d ", g, $1-$2}'; done)"
  fi
  sleep 30
done
