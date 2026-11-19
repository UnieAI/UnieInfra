#!/bin/bash
cd /home/smc/workspace/roy
until grep -q "vllm c128: done" logs/round-final.log; do sleep 20; done
echo "FINAL MATRIX DONE (unieinfra cells skipped: optimization first) $(date -u +%FT%TZ)" >> logs/round-final.log
