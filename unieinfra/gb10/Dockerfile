# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 UnieAI
#
# TokenSpeed on NVIDIA GB10 (DGX Spark: sm_121, aarch64, 128 GB unified memory).
# Build on the GB10 itself, from the repository root:
#
#   docker build -f unieinfra/gb10/Dockerfile -t unieinfra:gb10-main .
#
# Same recipe as docker/Dockerfile.nvidia, but on the runner image whose
# flashinfer-jit-cache matches the flashinfer-python pin (0.7.1rc2; the 0.6.18
# image fails the kernel build on flashinfer's version check), and with native
# kernels for sm_121a only: the default list (9.0a/10.0a) has no SASS a GB10
# can run, and an `a` target is not forward compatible across minor versions.
FROM lightseekorg/tokenspeed-runner:cu130-torch-2.14.0-flashinfer-0.7.1rc2

USER root

ARG MAX_JOBS=16
ARG CUDA_ARCH_LIST="12.1a"

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    FLASHINFER_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl-dev \
        libopenmpi-dev \
        libnuma1 \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Triton JIT-compiles most kernels here at runtime. Fail the build now, not the
# first forward pass, if the toolkit's ptxas cannot target sm_121a.
RUN uname -m && \
    ${TRITON_PTXAS_PATH} --version | tail -1 && \
    ${TRITON_PTXAS_PATH} --help | grep sm_121a >/dev/null

COPY . /workspace

RUN python3 -m pip install cmake ninja && \
    MAX_JOBS=${MAX_JOBS} \
    FLASHINFER_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
    TOKENSPEED_KERNEL_BACKEND=cuda \
    python3 -m pip install tokenspeed-kernel/python/ --no-build-isolation && \
    python3 -m pip install tokenspeed-scheduler/ && \
    python3 -m pip install ./python --no-build-isolation && \
    python3 -c "import importlib.metadata as m; \
print({p: m.version(p) for p in ('tokenspeed', 'tokenspeed_kernel', \
'tokenspeed-scheduler', 'torch', 'flashinfer-python')})"

CMD ["/bin/bash"]
