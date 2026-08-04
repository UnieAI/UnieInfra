# Source this before building/running UnieInfra on sm_120 in the dev container.
# See docs/guides/build-and-run.md section 4 for the rationale behind each shim.
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CPATH=$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/include 2>/dev/null | tr '\n' ':')$CPATH
export LIBRARY_PATH=$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')$LIBRARY_PATH
export LD_LIBRARY_PATH=/workspace/jeff/uniebench/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH
