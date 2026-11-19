#!/bin/bash
cd /home/smc/workspace/roy
exec .venv-tools/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --local-dir models/DeepSeek-V4.1-Flash-1017 --max-workers 16
