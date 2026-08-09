# Source this before building/running UnieInfra on sm_120 in the dev container.
# See docs/guides/build-and-run.md section 4 for the rationale behind each shim.
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CPATH=$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/include 2>/dev/null | tr '\n' ':')$CPATH
export LIBRARY_PATH=$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')$LIBRARY_PATH
# The prebuilt kernels in tokenspeed_kernel/thirdparty/cuda/objs (rope, marlin,
# routing, rmsnorm_fused_parallel, ...) are linked against GLIBCXX_3.4.32, which
# the container's system libstdc++ does not carry. Without this every one of
# them fails to dlopen -- and silently, because the ops fall back or raise only
# at the first call. `/workspace/roy/.libstdcxx-shim` holds a newer libstdc++
# for exactly this; it must come before the system one.
export LD_LIBRARY_PATH=/workspace/roy/.libstdcxx-shim:/workspace/jeff/uniebench/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH
