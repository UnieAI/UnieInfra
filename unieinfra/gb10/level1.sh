#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 UnieAI
#
# Level-1 GB10 check: does TokenSpeed install, detect sm_121, load its sm_121a
# kernels and pass its kernel tests? No model is loaded. Runs inside the
# unieinfra:gb10 container started by `run.sh check`, with the source tree at
# /src. Every step runs even if an earlier one fails; the summary lists each.
set -uo pipefail

RESULTS=()
step() {
  local name="$1"
  shift
  echo
  echo "=================== $name"
  if timeout "${STEP_TIMEOUT:-1200}" "$@"; then
    RESULTS+=("PASS  $name")
  else
    RESULTS+=("FAIL  $name (exit $?)")
  fi
}

KDIR="$(python3 -c 'import os, tokenspeed_kernel as k; print(os.path.dirname(k.__file__))')"
OBJS="$KDIR/thirdparty/cuda/objs"

step "device is sm_121" python3 -c '
import torch
cap = torch.cuda.get_device_capability(0)
print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), cap)
assert cap == (12, 1), cap'

step "platform detection" python3 -c '
from tokenspeed_kernel.platform import current_platform
p = current_platform()
print(p.vendor, p.arch_version, p.device_name, p.sm_count, "SMs,",
      p.max_shared_memory_per_sm, "B smem,", sorted(p.sm_features | p.runtime_features))
assert str(p.arch_version) == "12.1"'

# Each prebuilt kernel library must carry sm_121 SASS: an sm_120a or sm_100a
# cubin loads without complaint and fails only at the first launch.
# cuobjdump is not in the runtime image; run.sh mounts the host toolkit's.
CUOBJDUMP="${CUOBJDUMP:-cuobjdump}"
export CUOBJDUMP
step "AOT kernels carry sm_121 code" bash -c '
command -v "$CUOBJDUMP" >/dev/null || { echo "no cuobjdump ($CUOBJDUMP)"; exit 1; }
n=0; bad=0
for so in $(find "'"$OBJS"'" -name "*.so"); do
  n=$((n + 1))
  archs=$("$CUOBJDUMP" --list-elf "$so" 2>/dev/null | grep -o "sm_[0-9]*a\?" | sort -u | tr "\n" " ")
  case "$archs" in *sm_121*) ;; *) bad=$((bad + 1)); echo "  no sm_121: ${so#'"$OBJS"'/} [$archs]";; esac
done
echo "$n libraries, $bad without sm_121"
[ "$n" -gt 0 ] && [ "$bad" -eq 0 ]'

step "AOT kernels load" python3 -c '
import glob, tvm_ffi
libs = sorted(glob.glob("'"$OBJS"'/**/*.so", recursive=True))
bad = []
for so in libs:
    try:
        tvm_ffi.load_module(so)
    except Exception as e:
        bad.append((so, e))
for so, e in bad:
    print("  FAILED", so.split("/objs/")[-1], type(e).__name__, str(e)[:200])
print(len(libs), "libraries,", len(bad), "failed to load")
assert libs and not bad'

step "kernel plugins" python3 -m tokenspeed_kernel.plugins list

# The UnieAI sm_12x plugin, when this tree carries it.
if python3 -c "import unieinfra_kernel_sm120" 2>/dev/null; then
  step "UnieAI plugin registers on sm_121" bash -c '
  python3 -m tokenspeed_kernel.plugins info unieinfra_sm120 | tee /tmp/plugin.txt
  [ "$(grep -c "priority=" /tmp/plugin.txt)" -gt 0 ]'
  step "UnieAI plugin tests" python3 -m pytest -q -rfs /src/unieinfra/kernel-sm120/tests
fi

# The kernel tests' conftest puts ../python (the source tree, which has no
# compiled kernels) first on sys.path. Run a copy whose ../python does not
# exist, so they exercise the installed wheel. --rootdir keeps that conftest
# (and its `device` fixture) in scope for the tests under ops/.
rm -rf /tmp/ktest && mkdir -p /tmp/ktest && cp -r /src/tokenspeed-kernel/test /tmp/ktest/
for t in \
  test_plugins.py test_selection.py \
  ops/test_layernorm.py ops/test_activation.py ops/test_kvcache.py \
  ops/test_sampling_triton.py ops/test_quantization.py ops/test_gemm.py \
  ops/test_attention.py ops/test_attention_gdn.py \
  test_extend_split_vo_gate.py test_gemma4_extend_head_dim.py; do
  [ -f "/tmp/ktest/test/$t" ] || continue
  step "kernel tests: $t" python3 -m pytest -q -rfs -p no:cacheprovider \
    --rootdir /tmp/ktest/test "/tmp/ktest/test/$t"
done

step "tokenspeed runtime imports" python3 -c '
import tokenspeed
from tokenspeed.runtime.entrypoints.engine import Engine
print("tokenspeed", tokenspeed.__file__)'

echo
echo "=================== SUMMARY"
printf '%s\n' "${RESULTS[@]}"
