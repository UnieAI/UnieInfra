#!/bin/bash
# Retry pulls: docker sometimes picks unreachable IPv6 CloudFront endpoints.
for img in \
  lightseekorg/tokenspeed-runner-amd:rocm7.2.4-pytorch-2.13.0-gfx950 \
  lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261021 \
  vllm/vllm-openai-rocm:nightly-335dd77b09161c5d63139b4e7ae51bb9928305b1@sha256:3c634cc2c51eda7eb840a236b1f09a5ecf6a4b0cb92c4b8486e2158770507291; do
  for i in $(seq 1 40); do
    echo "=== $img attempt $i"
    docker pull -q "$img" && break
    sleep 5
  done
done
echo ALL_DONE
docker images
