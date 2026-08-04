#!/usr/bin/env bash
# Two-engine A/B at one or more concurrencies, with the contract's gates.
#
# Arms are run sequentially, not concurrently. They sit on different GPUs, so
# the GPUs would not contend, but both clients are pure-Python SSE readers and
# a shared host under two of them can limit the client rather than the server —
# which is the failure counter_agreement exists to catch. Sequential removes
# the question instead of testing for it.
#
# Usage: run_ab.sh <outdir> <conc> [conc ...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${1:?usage: run_ab.sh <outdir> <conc> [conc ...]}"
shift
CONCS=("$@")
[ ${#CONCS[@]} -gt 0 ] || { echo "no concurrencies given" >&2; exit 2; }

# arm name -> "base_url gpu_index metrics_url"
# TokenSpeed's gateway serves prometheus on its own port, so the metrics URL is
# not derivable from the base URL and has to be stated per arm.
declare -A ARMS=(
  [ts]="http://127.0.0.1:8100 2 http://127.0.0.1:9100/metrics"
  [vllm]="http://127.0.0.1:8101 3 http://127.0.0.1:8101/metrics"
)

ILEN="${ILEN:-1024}"
OLEN="${OLEN:-256}"
MAX_SECONDS="${MAX_SECONDS:-600}"
STAGGER_S="${STAGGER_S:-20}"
MODEL_NAME="${MODEL_NAME:-bench-model}"

mkdir -p "$OUTDIR"

for conc in "${CONCS[@]}"; do
  for arm in ts vllm; do
    read -r base gpu metrics <<<"${ARMS[$arm]}"
    out="$OUTDIR/${arm}_c${conc}.json"
    echo "=== $arm @ C=$conc -> $out ==="
    BASE="$base" GPU_IDX="$gpu" METRICS_URL="$metrics" MODEL_NAME="$MODEL_NAME" \
      CONC="$conc" ILEN="$ILEN" OLEN="$OLEN" \
      MAX_SECONDS="$MAX_SECONDS" STAGGER_S="$STAGGER_S" \
      OUT="$out" \
      python3 "$HERE/steady_bench.py" 2>&1 | tail -5
    # Let the engine drain and its KV pool settle before the next arm starts.
    sleep 20
  done

  echo "--- stationarity C=$conc (both arms must read settled) ---"
  python3 "$HERE/stationarity.py" 60 "$OUTDIR/ts_c${conc}.json" "$OUTDIR/vllm_c${conc}.json" || true
  echo "--- period-blocked comparison C=$conc ---"
  python3 "$HERE/ladder.py" 60 "$OUTDIR/ts_c${conc}.json" "$OUTDIR/vllm_c${conc}.json" || true
done
