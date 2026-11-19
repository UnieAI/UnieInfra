#!/bin/bash
# Run one AgentX benchmark round: for each engine x concurrency, start a fresh
# server (AgentX guidance: restart per concurrency point), run AgentX, stop it.
# Usage: run_round.sh <round-tag> <duration-s> "<engines>" "<concs>"
#   e.g. run_round.sh baseline 900 "unieinfra vllm sglang" "1 4 16"
# Must run with docker access (sg docker -c ...).
set -uo pipefail
TAG=${1:?round tag}; DURATION=${2:?duration}; ENGINES=${3:?engines}; CONCS=${4:?concs}
ROOT=/home/smc/workspace/roy
LOG=$ROOT/logs/round-$TAG.log
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

# A queued chain may name cells it cannot know the value of yet. A
# logs/skip-<tag> file retires one without editing the chain that queues it.
if [[ -f $ROOT/logs/skip-$TAG ]]; then
  echo "=== $(date -u +%FT%TZ) SKIP $TAG (logs/skip-$TAG exists)" | tee -a "$LOG"
  exit 1
fi

# Collect the crashed launch's server log next to the round log.
dump_fail_log() {
  local engine=$1 conc=$2
  if [[ $engine == unieinfra ]]; then
    cp "$ROOT/logs/unieinfra-server.log" "$ROOT/logs/fail-$TAG-$engine-c$conc.log" 2>/dev/null
  else
    docker logs bench-server > "$ROOT/logs/fail-$TAG-$engine-c$conc.log" 2>&1
  fi
}

for engine in $ENGINES; do
  for conc in $CONCS; do
    echo "=== $(date -u +%FT%TZ) $engine c$conc: starting server" | tee -a "$LOG"
    "$ROOT/scripts/stop_servers.sh"
    # A launch that refuses to start (e.g. GPUs still held by the previous cell)
    # must fail here: waiting on its port afterwards only burns the timeout.
    if ! "$ROOT/scripts/serve_$engine.sh" "$conc" >> "$LOG" 2>&1; then
      echo "=== $(date -u +%FT%TZ) $engine c$conc: SERVER FAILED TO START (launch aborted)" | tee -a "$LOG"
      dump_fail_log "$engine" "$conc"
      continue
    fi
    # unieinfra writes one log per launch; wait_ready uses it to tell a dead
    # start (and a foreign server on the same port) from this launch coming up.
    ready_log=""
    [[ $engine == unieinfra ]] && ready_log=$ROOT/logs/unieinfra-server.log
    if ! "$ROOT/scripts/wait_ready.sh" 3000 "$ready_log" >> "$LOG" 2>&1; then
      echo "=== $engine c$conc: SERVER FAILED TO START" | tee -a "$LOG"
      dump_fail_log "$engine" "$conc"
      continue
    fi
    # DURATION=official defers to the agentx preset (one-hour profile, ten
    # warmup requests per lane) per inferencex-e2e/docs/agentx-standalone.md,
    # passing no duration/warmup override at all.
    # run_agentx.sh interprets the literal "official" itself (no duration or
    # warmup flag at all) and labels the output directory -dofficial.
    # `local` is a function-only builtin; at script level it fails and, with
    # `set -u`, arithmetic on a non-numeric duration aborts the whole round.
    budget=5400
    if [[ "$DURATION" == "official" ]]; then budget=14400; else budget=$(( DURATION + 5400 )); fi
    echo "=== $(date -u +%FT%TZ) $engine c$conc: running agentx ${DURATION}" | tee -a "$LOG"
    run_log=$ROOT/logs/agentx-$TAG-$engine-c$conc.log
    # Watchdog: dataset load + warmup + drain fit well within 40 min beyond the
    # profile; a point that runs longer means the server hung (e.g. a dead TP rank).
    timeout -k 60 "$budget" \
      env TAG=$TAG "$ROOT/scripts/run_agentx.sh" "$engine" "$conc" "$DURATION" > "$run_log" 2>&1
    rc=$?
    # run_agentx.sh prints its output directory on the first line.
    out_dir=$(head -1 "$run_log")
    # An aborted run (warmup failure, server hang) still exits 0 from agentx and
    # leaves no export; without this check such a cell looks like a completed one.
    if [[ ! -f "$out_dir/aiperf_artifacts/profile_export_aiperf.json" ]]; then
      echo "=== $(date -u +%FT%TZ) $engine c$conc: NO EXPORT (rc=$rc) -> $out_dir -- cell INVALID, no result" | tee -a "$LOG"
    fi
    echo "=== $(date -u +%FT%TZ) $engine c$conc: done (rc=$rc) -> $out_dir" | tee -a "$LOG"
  done
done
"$ROOT/scripts/stop_servers.sh"
echo "=== $(date -u +%FT%TZ) ROUND $TAG COMPLETE" | tee -a "$LOG"
