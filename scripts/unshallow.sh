#!/bin/bash
cd /home/smc/workspace/roy/unieinfra
for i in $(seq 1 20); do
  git fetch --unshallow && { echo UNSHALLOW_OK; git log --oneline | wc -l; exit 0; }
  sleep 10
done
echo UNSHALLOW_FAILED
