#!/bin/bash
# Hand off from the broken first official queue to the corrected one.
#
# The first queue started before run_round.sh was fixed: its opening vLLM c64 cell
# died the moment the server became ready, because `local budget=...` at script
# level fails and `set -u` then turned the arithmetic on the literal "official"
# into a fatal unbound variable. Its second cell (our c64 half) is already running
# against the fixed script, so this watcher waits for that cell to report done,
# stops the old queue, releases the GPUs and lets test_official2.sh take over.
set -uo pipefail
ROOT=/home/smc/workspace/roy
LOG=$ROOT/logs/official.log

until grep -q "OFFICIAL unieinfra c64 cap16 done" "$LOG"; do sleep 30; done
echo "=== $(date -u +%FT%TZ) handoff: c64 half finished, stopping the old queue" >> "$LOG"
# Killing the queue script alone would leave its in-flight round alive: the next
# cell's server boot starts the instant this one reports done, so a stalled
# run_round would keep port 8000 and collide with part 2.
pkill -f "scripts/test_official.sh"
sleep 2
pkill -f "scripts/run_round.sh"
sleep 2
pkill -f "[.]venv-agentx/bin/agentx"
sleep 3
"$ROOT/scripts/stop_servers.sh" >> "$LOG" 2>&1
for _ in $(seq 1 90); do
  used=$(docker exec -u root ts-dev rocm-smi --showmeminfo vram 2>/dev/null \
    | grep -oP 'Used Memory \(B\): \K\d+' | sort -n | tail -1)
  [[ ${used:-0} -lt 20000000000 ]] && break
  sleep 10
done
echo "=== $(date -u +%FT%TZ) handoff: vram free (${used:-?} B), starting part 2" >> "$LOG"
exec "$ROOT/scripts/test_official2.sh"
