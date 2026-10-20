#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 UnieAI
#
# Build the three TokenSpeed wheels for GB10 on any arm64 Docker host -- an
# Apple-silicon Mac works, no GPU needed. Run from the repository:
#
#   unieinfra/gb10/build_wheels.sh            # -> $GB10_WHEELS (default below)
#
# This exists so the GB10 never runs nvcc while its resident service holds most
# of the shared memory: the CUDA compile happens here, and the GB10 only
# installs the result (run.sh check). It compiles the same sources the
# Dockerfile does, from the committed HEAD, inside the same base image, so the
# wheels match that image's Python, torch and CUDA exactly.
set -euo pipefail

BASE_IMAGE="lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.7.1rc2"
OUT="${GB10_WHEELS:-$HOME/.cache/unieinfra-main/wheels}"
# Each nvcc job on the heavier units (marlin MoE, the TRT-LLM fusions) holds a
# few GB; keep jobs x ~3 GB under the Docker VM's memory.
MAX_JOBS="${MAX_JOBS:-3}"

REV="$(git rev-parse --short HEAD)"
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
git archive --format=tar HEAD -- . ':(exclude)legacy' | tar -x -C "$CTX"
rm -rf "$OUT" && mkdir -p "$OUT"

echo "building wheels for $REV with MAX_JOBS=$MAX_JOBS -> $OUT"
docker run --rm --platform linux/arm64 \
  -v "$CTX:/src" -v "$OUT:/out" \
  -e MAX_JOBS="$MAX_JOBS" \
  -e TOKENSPEED_KERNEL_BACKEND=cuda \
  -e TOKENSPEED_CUDA_ARCH=121a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PIP_BREAK_SYSTEM_PACKAGES=1 \
  --user root "$BASE_IMAGE" bash -lc '
    set -euo pipefail
    apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      libssl-dev pkg-config >/dev/null
    /usr/local/cuda/bin/ptxas --help | grep sm_121a >/dev/null
    python3 -m pip install -q cmake ninja
    cd /src
    python3 -m pip wheel --no-deps --no-build-isolation -w /out ./tokenspeed-kernel/python
    python3 -m pip wheel --no-deps -w /out ./tokenspeed-scheduler
    python3 -m pip wheel --no-deps --no-build-isolation -w /out ./python
    [ ! -d unieinfra/kernel-sm120 ] ||
      python3 -m pip wheel --no-deps --no-build-isolation -w /out ./unieinfra/kernel-sm120'

echo "$REV" >"$OUT/.gb10-rev"
ls -lh "$OUT"
