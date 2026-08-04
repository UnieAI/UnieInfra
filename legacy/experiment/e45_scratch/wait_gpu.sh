#!/usr/bin/env bash
# Poll for a GPU with < THRESH MiB used. Prints the index. GPU 0 is off limits.
THRESH=${THRESH:-2000}
while true; do
  for i in 1 2 3; do
    used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -lt "$THRESH" ]; then echo "$i"; exit 0; fi
  done
  sleep 20
done
