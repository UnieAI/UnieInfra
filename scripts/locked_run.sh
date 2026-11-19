#!/bin/bash
# Usage: scripts/locked_run.sh <owner> <purpose> <expected> <command...>
# Waits for logs/GPU_LOCK to be absent (and no other round running), takes it, runs, releases.
cd /home/smc/workspace/roy
OWNER=$1; PURPOSE=$2; EXPECT=$3; shift 3
while true; do
  if [ ! -e logs/GPU_LOCK ] && ! pgrep -f "[r]un_round.sh" >/dev/null; then
    sleep 90
    if [ ! -e logs/GPU_LOCK ] && ! pgrep -f "[r]un_round.sh" >/dev/null; then
      ( set -o noclobber; echo "owner=$OWNER($PURPOSE) since=$(date -u +%FT%TZ) release=$EXPECT" > logs/GPU_LOCK ) 2>/dev/null && break
    fi
  fi
  sleep 60
done
"$@"
rm -f logs/GPU_LOCK
