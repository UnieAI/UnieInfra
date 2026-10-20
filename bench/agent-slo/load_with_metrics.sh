#!/bin/bash
# Run one measured cell at a given concurrency while sampling the scheduler's own
# gauges, so the throughput number arrives with the diagnosis that explains it.
#
#   bench/agent-slo/load_with_metrics.sh PORT SERVED_MODEL CONC DATA WARM OUTDIR
#
# The server must already be up with METRICS_PORT set and `--enable-metrics`.
# The sample loop runs for a fixed window around the measured run (400 s by
# default, DRIVE_SAMPLE_SECONDS to change it) so the CSV also covers the ramp-down
# and the WARM/measured split is visible as a step in the counters.
set -uo pipefail
PORT=${1:?port}
MN=${2:?served model name}
CONC=${3:?concurrency}
DATA=${4:?corpus}
WARM=${5:?warmup corpus}
OUT=${6:?out dir}
MPORT=${METRICS_PORT:-8117}
WINDOW=${DRIVE_SAMPLE_SECONDS:-400}

REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
[ -x "$PY" ] || PY=python3
DRIVE=$REPO/bench/agent-slo/drive.py
SAMPLER=$REPO/bench/agent-slo/step_metrics.py

export ENDPOINT=${ENDPOINT:-completions}
export TOKENIZER=${TOKENIZER:-$MN}
mkdir -p "$OUT"

echo "== warmup $CONC x 256"
"$PY" "$DRIVE" "http://127.0.0.1:$PORT" "$MN" "$WARM" "$CONC" 256 "$OUT/warm.json" "$WARM" \
    > "$OUT/warm.log" 2>&1
tail -1 "$OUT/warm.log"

echo "== sampling http://127.0.0.1:$MPORT/metrics at 20 Hz"
"$PY" "$SAMPLER" "http://127.0.0.1:$MPORT/metrics" "$OUT/metrics.csv" "$WINDOW" 20 &
SAMP=$!
sleep 2

echo "== measured $CONC x 256 from $DATA"
"$PY" "$DRIVE" "http://127.0.0.1:$PORT" "$MN" "$DATA" "$CONC" 256 "$OUT/measured.json" "$WARM"
sleep 2
kill $SAMP 2>/dev/null
wait $SAMP 2>/dev/null

# Report the step accounting the counters support: iterations per second, tokens
# per iteration, and the split of iterations by tokens carried, which is how a
# prefill step is told from a decode step without a trace.
"$PY" - "$OUT" <<'EOF'
import csv
import statistics
import sys

out = sys.argv[1]
rows = list(csv.DictReader(open(f"{out}/metrics.csv")))
live = [r for r in rows if float(r["num_requests_running"]) > 0]
print(f"== metrics: {len(rows)} samples, {len(live)} with a non-empty batch")
if len(live) < 2:
    sys.exit(0)
i0, i1 = live[0], live[-1]
dt = float(i1["t"]) - float(i0["t"])
iters = float(i1["iteration_tokens_total_count"]) - float(i0["iteration_tokens_total_count"])
tokens = float(i1["iteration_tokens_total_sum"]) - float(i0["iteration_tokens_total_sum"])
run = [float(r["num_requests_running"]) for r in rows]
wait = [float(r["num_requests_waiting"]) for r in rows]
kv = [float(r["kv_cache_usage_perc"]) for r in rows]
print(f"   {dt:.1f}s  {iters:.0f} iterations  {iters / dt:.2f} iter/s  {1000 * dt / iters:.1f} ms/iter")
print(f"   {tokens / iters:.0f} tokens/iter (of the scheduler's own budget)")
print(f"   running p50 {statistics.median(run):.0f} max {max(run):.0f}"
      f" | waiting p50 {statistics.median(wait):.0f} max {max(wait):.0f}"
      f" | kv p50 {statistics.median(kv):.2f} max {max(kv):.2f}")
buckets = sorted((k for k in rows[0] if k.startswith("iteration_tokens_total_le")),
                 key=lambda k: float(k[len("iteration_tokens_total_le"):].replace("+Inf", "inf")))
if buckets:
    first = {k: float(i0[k]) for k in buckets}
    last = {k: float(i1[k]) for k in buckets}
    total = last[buckets[-1]] - first[buckets[-1]]
    print(f"   steps by tokens carried ({total:.0f} steps):")
    prev_bound, prev = 0.0, 0.0
    for k in buckets:
        bound = float(k[len("iteration_tokens_total_le"):].replace("+Inf", "inf"))
        count = last[k] - first[k]
        label = f"({prev_bound:.0f}, {bound:.0f}]" if bound != float("inf") else f"({prev_bound:.0f}, inf)"
        print(f"     {label:>18} {count:7.0f}  {100 * count / total if total else 0:5.1f}%")
        prev_bound, prev = bound, count
EOF