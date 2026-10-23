# Environment for the SGLang Diffusion reference engine (pin 05c584c, 2026-08-22).
# CUDA_HOME is load-bearing: the system nvcc is 12.0, and flashinfer refuses to
# resolve any SM 12.x target below CUDA 12.9 ("SM 12.x requires CUDA >= 12.9"),
# so without this it cannot determine TARGET_CUDA_ARCHS at all and every
# flashinfer path silently degrades. With it: TARGET_CUDA_ARCHS = {(12, '0f')}.
export CUDA_HOME=/workspace/roy/sglang-diffusion/.venv/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_DISABLE_VERSION_CHECK=1          # required by their own skill
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1     # stage-level attribution
export HF_HOME=/workspace/.cache
source /workspace/roy/sglang-diffusion/.venv/bin/activate

# Tracked in UnieInfer as unieinfer/bench/h3_reference_env.sh so the baseline in
# REPORT-h3-parity-baseline.md is reproducible from a commit; the copy under
# /workspace/roy/sglang-diffusion/ is a symlink to this file.
