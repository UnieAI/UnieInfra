# Source this before building/running UnieInfra on sm_120 in the dev container.
# See docs/guides/build-and-run.md section 4 for the rationale behind each shim.
export FLASHINFER_DISABLE_VERSION_CHECK=1
# `nvidia/cuda_cccl` is excluded on purpose. Two CCCL copies are installed --
# cu13 nests its own under `include/cccl/cuda/std`, which nvcc adds itself,
# while cuda_cccl puts one at `include/cuda/std`, which only CPATH can reach.
# With both visible a single translation unit takes `<cuda/std/utility>` from
# nvcc's copy and `<cuda/std/version>` from CPATH's older one, so
# `_LIBCUDACXX_HAS_SPACESHIP_OPERATOR` is undefined where it is used and every
# DeepGEMM JIT compile fails with "missing binary operator before token" --
# which reads like a source bug and is an include-path bug.
# `nvidia/cuda_runtime` is excluded for a second, separate reason. It is the
# CUDA 12 wheel; `nvidia/cu13` is the CUDA 13 one, and both ship a top-level
# `vector_types.h`. CUDA 13 added the 32-byte-aligned vector types
# (`double4_32a`, `longlong4_32a`) and flashinfer's bundled CCCL uses them in
# `cuda/__ptx/ptx_helper_functions.h`. With both wheels on CPATH, CUDA 12's
# `cuda_runtime.h` pulls in its own sibling `vector_types.h` by relative path --
# which beats CPATH order, so putting cu13 first does not help (measured: 19
# errors either way, 0 with cuda_runtime removed). Only head_dim 512 kernels
# instantiate the code that needs those wide types, so this stayed invisible
# until gemma-4's ten global-attention layers were routed to flashinfer: the
# server died at capture with `the global scope has no "double4_32a"`, which
# reads like a flashinfer bug and is an include-path bug.
# cu13 carries every header cuda_runtime does except four texture/surface ones,
# which no GEMM or attention kernel here includes.
export CPATH=$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/include 2>/dev/null | grep -v '/cuda_cccl/' | grep -v '/cuda_runtime/' | tr '\n' ':')$CPATH
# The cu13 wheel ships nvcc 13.2 against CUDART 13.3 headers, and CCCL refuses
# the pair outright ("CUDA compiler and CUDA toolkit headers are
# incompatible"). The header names this flag for exactly that case -- a CTK
# newer than the compiler ships. One minor version apart; revisit if the wheel
# is ever fixed, and drop the flag rather than carrying it forever.
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"
# The nvidia pip wheels ship only sonames (libcudart.so.12, .so.13) and no
# unversioned libcudart.so, so anything that JIT-compiles with `-lcudart` --
# flashinfer's xqa decode kernel among them -- dies at link time with
# "cannot find -lcudart". `/workspace/roy/.cudart-shim` holds that one symlink,
# pointing at the .so.12 that matches the container's CUDA 12.9 torch build.
# Without it the hybrid checkpoints cannot start at all.
export LIBRARY_PATH=/workspace/roy/.cudart-shim:$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')$LIBRARY_PATH
# The prebuilt kernels in tokenspeed_kernel/thirdparty/cuda/objs (rope, marlin,
# routing, rmsnorm_fused_parallel, ...) are linked against GLIBCXX_3.4.32, which
# the container's system libstdc++ does not carry. Without this every one of
# them fails to dlopen -- and silently, because the ops fall back or raise only
# at the first call. `/workspace/roy/.libstdcxx-shim` holds a newer libstdc++
# for exactly this; it must come before the system one.
# CUDA 13 runtime libs. These used to be pinned to /workspace/jeff/uniebench's
# venv -- another user's directory. When it went away every launch died with
# "OSError: libcublas.so.13: cannot open shared object file" from
# trtllm_kernel's ctypes.CDLL, which reads as a broken tree rather than a
# missing path. Resolve the first candidate that actually exists instead.
_ts_cu13=""
for _c in /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib \
          /workspace/weiminc/UnieInfra/.container_venv/lib/python3.12/site-packages/nvidia/cu13/lib \
          /workspace/weiminc/unieai_branch/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13/lib \
          /workspace/jeff/uniebench/.venv/lib/python3.12/site-packages/nvidia/cu13/lib \
          /tmp/cu13libs/nvidia/cu13/lib; do
  if [ -e "$_c/libcublas.so.13" ]; then _ts_cu13="$_c"; break; fi
done
if [ -z "$_ts_cu13" ]; then
  echo "sm120_env.sh: no libcublas.so.13 found; trtllm_kernel will fail to load" >&2
fi
export LD_LIBRARY_PATH=/workspace/roy/.libstdcxx-shim${_ts_cu13:+:$_ts_cu13}:$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH
unset _ts_cu13 _c
