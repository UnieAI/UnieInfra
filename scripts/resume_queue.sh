#!/bin/bash
# Remaining queue. vLLM was still rising at c64 (c32 111.8 -> c64 125.3 RPM),
# so find its peak: c128 at max-num-seqs 64/128, then c256 at the best seqs
# if c128 beat c64. Then the official UnieInfra cells.
ROUND_PID=${1:?pid of the running stressseq64 round}
cd /home/smc/workspace/roy
while kill -0 "$ROUND_PID" 2>/dev/null; do sleep 20; done
STRESS=1 VLLM_MAX_NUM_SEQS=128 scripts/run_round.sh stressseq128 300 vllm "128"
best=$(python3 - <<'PY'
import glob, json
best = (0, 0)
for s in (64, 128):
    for d in sorted(glob.glob(f"results/agentx/vllm-stressseq{s}-c128-*")):
        try:
            j = json.load(open(f"{d}/aiperf_artifacts/profile_export_aiperf.json"))
        except OSError:
            continue
        rpm = j["request_throughput"]["avg"] * 60
        best = max(best, (rpm, s))
print(f"{best[1]} {best[0]:.1f}")
PY
)
read seqs rpm <<<"$best"
echo "=== $(date -u +%FT%TZ) vllm c128 best: max-num-seqs $seqs at $rpm RPM (c64 big-seqs: 125.3)" | tee -a logs/round-stressseq128.log
if [[ "$seqs" != 0 ]] && python3 -c "import sys; sys.exit(0 if float('$rpm') > 125.3 else 1)"; then
  STRESS=1 VLLM_MAX_NUM_SEQS=$seqs scripts/run_round.sh "stressseq$seqs" 300 vllm "256"
fi
scripts/run_round.sh comm 900 unieinfra "64 1 4 16 32"
echo "ALL DONE $(date -u +%FT%TZ)" >> logs/round-comm.log
