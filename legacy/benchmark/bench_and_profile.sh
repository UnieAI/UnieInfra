#!/usr/bin/env bash
# One-command "change -> run -> profile -> know what to fix next" loop.
#
# Wraps the EXISTING harness (does not reimplement it):
#   benchmark/launch/launch_<engine>.sh -> benchmark/run_sweep.sh -> benchmark/summarize.py
#   -> benchmark/profile_hot_decode.py -> benchmark/trajectory.py
#
# Usage:
#   bench_and_profile.sh <label> [engine] [--dry-run]
#
#   <label>   results/<engine>/<label>/ -- also configs/<label>.env if present (sourced
#             before launch, so "make a change -> new label -> one command" works: drop
#             the env knobs for the change into configs/<label>.env, then run this).
#   [engine]  unieinfra (default) | vllm  -- which launch/launch_<engine>.sh to use.
#             (sglang/lmdeploy: start the server yourself, then run with BASE_URL set
#              and --dry-run so this script only does the sweep-adjacent bookkeeping;
#              see benchmark/scripts/launch_sglang_host.sh for sglang.)
#   --dry-run runs ONLY steps (c) trajectory refresh + (d) delta/regression report,
#             against whatever is ALREADY in results/<engine>/<label>/. Use this to
#             validate the harness or re-read a past run without touching a GPU.
#
# Steps:
#   (a) launch the engine (backgrounded), wait for it to answer, run the aligned
#       sweep (prefix-cache ON, flush between levels, fixed seed/warmups) into
#       results/<engine>/<label>/, then stop the engine.
#   (b) run a light steady-state decode profile (batch-1, batch-128) into
#       benchmark/profiles/<label>/ -- set PROFILE=0 to skip.
#   (c) refresh benchmark/results/TRAJECTORY.md + trajectory.csv over ALL saved runs.
#   (d) print this label's delta vs the previous unieinfra label (whatever ran right
#       before it) and vs the fixed baseline (default "clean"), with a REGRESSION
#       flag if any concurrency got worse vs baseline.
#
# Config (env vars, or configs/<label>.env -- env vars win if both set):
#   MODEL, GPU, PORT, DTYPE, GMU, MAXLEN, ATTN, MAX_NUM_SEQS, MAX_CGRAPH,
#   DISABLE_RADIX, VLLM_ATTENTION_BACKEND (vllm engine), CONTAINER,
#   ILEN, OLEN, PROMPTS_PER_CONC, WARMUPS, SEED, FLUSH, CONCURRENCIES,
#   VLLM_PY (client-side python with `vllm` installed, for `vllm bench serve`),
#   BASELINE (default "clean"), PROFILE (default 1).
#   Any other knob launch_<engine>.sh grows in the future (e.g. QUANTIZATION,
#   KV_CACHE_DTYPE, ENABLE_ADAPTIVE_BATCH) can already be set here -- this script
#   only passes environment through, it does not hardcode the launch scripts' flags.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  echo "usage: $(basename "$0") <label> [engine] [--dry-run]" >&2
  echo "  engine: unieinfra (default) | vllm" >&2
  exit 2
}

[ $# -ge 1 ] || usage
LABEL="$1"; shift
ENGINE="unieinfra"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage ;;
    unieinfra|vllm) ENGINE="$arg" ;;
    *) echo "[bench_and_profile] unrecognized argument: $arg" >&2; usage ;;
  esac
done

case "$LABEL" in
  ''|*/*) echo "[bench_and_profile] label must be non-empty and contain no '/': $LABEL" >&2; exit 2 ;;
esac

CONFIG_FILE="${BENCH_DIR}/configs/${LABEL}.env"
if [ -f "$CONFIG_FILE" ]; then
  echo "[bench_and_profile] sourcing ${CONFIG_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
fi

BASELINE="${BASELINE:-clean}"
CONTAINER="${CONTAINER:-unieinfra-dev}"
MODEL="${MODEL:-/workspace/models/Qwen/Qwen3-8B}"
GPU="${GPU:-1}"
PORT="${PORT:-8000}"
VLLM_PY="${VLLM_PY:-/home/ubuntu/vllm-bench-venv/bin/python}"
PROFILE="${PROFILE:-1}"

echo "[bench_and_profile] label=${LABEL} engine=${ENGINE} dry_run=${DRY_RUN}"

if [ "$DRY_RUN" != "1" ]; then
  case "$ENGINE" in
    unieinfra) LAUNCH_SCRIPT="${BENCH_DIR}/launch/launch_unieinfra.sh" ;;
    vllm)      LAUNCH_SCRIPT="${BENCH_DIR}/launch/launch_vllm.sh" ;;
    *) echo "[bench_and_profile] no launch script wired for engine=${ENGINE}; start it yourself and re-run with --dry-run" >&2; exit 2 ;;
  esac

  # ---- (a) launch + sweep ----
  echo "[bench_and_profile] (a) launching ${ENGINE} on gpu=${GPU} port=${PORT} ..."
  MODEL="$MODEL" GPU="$GPU" PORT="$PORT" "$LAUNCH_SCRIPT" &
  SERVER_PID=$!

  cleanup() {
    echo "[bench_and_profile] stopping ${ENGINE} server (pid=${SERVER_PID}) ..."
    kill "$SERVER_PID" 2>/dev/null || true
    # best-effort: docker exec forks its own process tree, so also try to kill
    # the served process inside the container directly.
    docker exec "$CONTAINER" pkill -f "serve.*${PORT}" 2>/dev/null || true
  }
  trap cleanup EXIT

  BASE_URL="http://127.0.0.1:${PORT}"
  echo "[bench_and_profile] waiting for ${BASE_URL} to come up (up to 6 min) ..."
  UP=0
  for _ in $(seq 1 180); do
    if curl -sf "${BASE_URL}/v1/models" >/dev/null 2>&1; then UP=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[bench_and_profile] server process exited early" >&2; break; }
    sleep 2
  done
  [ "$UP" = "1" ] || { echo "[bench_and_profile] ${BASE_URL} never came up; aborting" >&2; exit 1; }

  echo "[bench_and_profile] (a) sweeping -> results/${ENGINE}/${LABEL}"
  LOCAL_TOKENIZER="$MODEL"
  ENGINE="$ENGINE" BASE_URL="$BASE_URL" MODEL="bench-model" TOKENIZER="$LOCAL_TOKENIZER" \
    VLLM_PY="$VLLM_PY" TS="$LABEL" \
    ILEN="${ILEN:-1024}" OLEN="${OLEN:-256}" \
    PROMPTS_PER_CONC="${PROMPTS_PER_CONC:-8}" WARMUPS="${WARMUPS:-8}" SEED="${SEED:-0}" \
    FLUSH="${FLUSH:-1}" CONCURRENCIES="${CONCURRENCIES:-1 16 32 64 128 256}" \
    "${BENCH_DIR}/run_sweep.sh"

  # ---- (b) light profile -- MUST run before (not after) stopping the server:
  # --mode torch drives the server's own /start_profile,/stop_profile endpoints,
  # so the server has to still be up.
  if [ "$PROFILE" = "1" ]; then
    echo "[bench_and_profile] (b) profiling steady-state decode (bs=1,128) -> profiles/${LABEL}"
    python3 "${BENCH_DIR}/profile_hot_decode.py" --label "$LABEL" --engine "$ENGINE" \
      --model "$MODEL" --container "$CONTAINER" --base-url "$BASE_URL" --port "$PORT" \
      || echo "[bench_and_profile] WARN: profile step failed (non-fatal, sweep data is still saved)"
  else
    echo "[bench_and_profile] (b) PROFILE=0 -> skipping profile step"
  fi

  trap - EXIT
  cleanup
else
  echo "[bench_and_profile] --dry-run: skipping (a) launch+sweep and (b) profile;"
  echo "[bench_and_profile]            only refreshing the trajectory view over existing results/."
fi

# ---- (c) refresh TRAJECTORY.md + trajectory.csv, and print this label vs baseline ----
echo
echo "[bench_and_profile] (c) refreshing TRAJECTORY.md (baseline=${BASELINE}) ..."
python3 "${BENCH_DIR}/trajectory.py" --baseline "$BASELINE" --focus "$LABEL"

# ---- (d) vs previous label ----
PREV_LABEL="$(python3 "${BENCH_DIR}/trajectory.py" --list-runs \
  | grep '^unieinfra/' | sed 's#^unieinfra/##' \
  | awk -v l="$LABEL" '$0==l{print prev; exit} {prev=$0}')"

echo
if [ -n "$PREV_LABEL" ]; then
  echo "[bench_and_profile] (d) === ${LABEL} vs previous label (${PREV_LABEL}) ==="
  python3 "${BENCH_DIR}/trajectory.py" --baseline "$PREV_LABEL" --focus "$LABEL" --no-write
else
  echo "[bench_and_profile] (d) no earlier unieinfra label found before '${LABEL}' -- nothing to diff against"
fi

echo
echo "[bench_and_profile] DONE. See ${BENCH_DIR}/results/TRAJECTORY.md"
