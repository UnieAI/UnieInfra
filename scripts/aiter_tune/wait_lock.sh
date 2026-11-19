#!/bin/bash
# Wait until GPU_LOCK is gone and GPU 0 has no process, then take the lock atomically.
L=/home/smc/workspace/roy/logs/GPU_LOCK
while true; do
  if [ ! -e $L ] && amd-smi process -g 0 2>/dev/null | grep -q "No running processes"; then
    if ( set -o noclobber; echo "owner=aiter-tune since=$(date -u +%FT%TZ)" > $L ) 2>/dev/null; then
      echo "LOCK TAKEN $(date -u +%FT%TZ)"; cat $L; exit 0
    fi
  fi
  sleep 30
done
