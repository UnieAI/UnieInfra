#!/bin/bash
# GPU profile of the attention-DP server during 16-user 130K decode: starts only once every rank is decoding.
cd /home/smc/workspace/roy
L=logs/dp_prof16.log
rm -rf logs/prof-dp16
echo "== $(date -u +%T)" > $L
python3 scripts/bench_decode_shared.py --n 16 --max-tokens 6000 --seed 7 --warm 8 2>&1 | tail -1 | sed 's/^/dp16-prof /' >> $L &
for i in $(seq 1 120); do
  n=$(sed 's/\x1b\[[0-9;]*m//g' $(readlink -f logs/unieinfra-server.log) | grep "Decode batch" | tail -4 | grep -oE "#running-req: [0-9]+" | grep -cE "running-req: ([3-9]|[1-9][0-9])")
  [ "$n" -ge 4 ] && break; sleep 2
done
echo "decoding on all ranks after ${i} polls" >> $L
sleep 6
curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' -d '{"output_dir": "/logs/prof-dp16", "activities": ["GPU"]}' >> $L
sleep 4; curl -s -X POST localhost:8000/stop_profile >> $L; wait
echo "DPPROF DONE $(date -u +%T)" >> $L
