#!/bin/bash
# After the official queue: install the scheduler with
# TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS and stress-test it at c128 and c64.
cd /home/smc/workspace/roy
until grep -q "ALL DONE" logs/round-comm.log; do sleep 30; done
docker exec -u root ts-dev bash -c '/opt/venv/bin/python3 -m pip install --no-deps --force-reinstall /tmp/sched-wheel/tokenspeed_scheduler-0.1.41-cp312-cp312-linux_x86_64.whl' > logs/sched-install.log 2>&1
echo "=== $(date -u +%FT%TZ) installed scheduler with decode interleave knob" >> logs/round-dec1.log
STRESS=1 TOKENSPEED_DECODE_EVERY_PREFILL_ROUNDS=1 scripts/run_round.sh dec1 300 unieinfra "128 64"
echo "DEC1 DONE $(date -u +%FT%TZ)" >> logs/round-dec1.log
