#!/bin/bash
# After the overlap bench: install the cache-aware scheduler and stress c128/c64.
cd /home/smc/workspace/roy
until [ -s /tmp/claude-1000/-home-smc-workspace-roy/550416c1-af79-45ad-94e9-d75d9ed2d978/tasks/bghywu07n.output ] && ! pgrep -f bench_overlap_cu >/dev/null; do sleep 30; done
sleep 10
docker exec -u root ts-dev bash -c '/opt/venv/bin/python3 -m pip install --no-deps --force-reinstall /tmp/sched-wheel-lpm/tokenspeed_scheduler-0.1.41-cp312-cp312-linux_x86_64.whl' > logs/sched-install-lpm.log 2>&1
export TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 TOKENSPEED_CACHE_AWARE_ADMISSION=1
STRESS=1 MAX_NUM_SEQS=128 EXTRA_SERVE_ARGS="--prefill-coalesce-ms 100" scripts/run_round.sh lpm 300 unieinfra "128 64"
echo "LPM DONE $(date -u +%FT%TZ)" >> logs/round-lpm.log
