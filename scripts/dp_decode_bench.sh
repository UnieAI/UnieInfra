#!/bin/bash
# Decode speed on the attention-DP server: the same document is cached on every rank first (--warm 8), then measured.
cd /home/smc/workspace/roy
L=logs/dp_decode2.log
echo "== $(date -u +%T)" > $L
python3 scripts/bench_decode_shared.py --n 16 --max-tokens 512 --seed 7 --warm 8 2>&1 | tail -1 | sed 's/^/dp16 /' >> $L
python3 scripts/bench_decode_shared.py --n 64 --max-tokens 1024 --seed 7 --warm 8 2>&1 | tail -1 | sed 's/^/dp64 /' >> $L
echo "DPDECODE2 DONE $(date -u +%T)" >> $L
