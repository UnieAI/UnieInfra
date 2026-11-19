#!/bin/bash
# Run pytest inside ts-dev against the .wt-up worktree (engine + its own kernel package).
docker exec -u root -w /workspace/.wt-up \
  -e PYTHONPATH=/workspace/.wt-up/python:/workspace/.wt-up/tokenspeed-kernel/python:/workspace/.wt-up/tokenspeed-kernel-amd/python:/workspace/.vendor \
  -e LD_LIBRARY_PATH=/opt/rocm/lib -e HF_HUB_OFFLINE=1 \
  ts-dev python -m pytest -q -x -p no:cacheprovider "$@"
