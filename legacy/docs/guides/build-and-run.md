<!--
  UnieInfra — Build & Run guide (English, canonical).
  Chinese translation: build-and-run.zh.md  (keep both in sync).
-->

# Build & Run UnieInfra on NVIDIA Blackwell (sm_120 / RTX PRO 6000)

This guide walks you, step by step, from a fresh clone to a **correct,
high-concurrency generation run** of the UnieInfra engine on an NVIDIA **sm_120**
GPU (RTX PRO 6000 / GB202, the same silicon as the RTX 5090). Every command is
copy-pasteable; every non-obvious step explains *why*, so you can adapt it to a
different box.

> **Where this fits.** UnieInfra targets *all* hardware tiers — datacenter NVIDIA
> (H100/H200/B200), Blackwell workstation (sm_120), AMD, and future NPUs. This
> guide covers the **sm_120 workstation tier**, which is the current bring-up
> target. Datacenter kernels (tcgen05/CuTe, for B200/H200) live behind the same
> capability registry and are selected automatically on that hardware — see
> `docs/design/unieinfra-architecture.md`.

**Contents**
1. [What you need](#1-what-you-need)
2. [Get the code](#2-get-the-code)
3. [Start the Blackwell dev container](#3-start-the-blackwell-dev-container)
4. [Set the runtime environment](#4-set-the-runtime-environment)
5. [Build & install the three packages](#5-build--install-the-three-packages)
6. [Run: correctness (single request)](#6-run-correctness-single-request)
7. [Run: throughput (high concurrency)](#7-run-throughput-high-concurrency)
8. [What works today / known gaps](#8-what-works-today--known-gaps)
9. [Troubleshooting](#9-troubleshooting-every-error-we-hit)

---

## 1. What you need

| Requirement | Detail |
|---|---|
| GPU | One or more NVIDIA **sm_120** cards (RTX PRO 6000 / RTX 5090). `nvidia-smi --query-gpu=compute_cap --format=csv` should report `12.0`. |
| Driver / CUDA | Driver that supports the CUDA 12.9+ runtime (Blackwell codegen needs **nvcc ≥ 12.8**). |
| Container runtime | Docker with the NVIDIA container toolkit (`--gpus` support). |
| Disk | ~30 GB free for the image + build + a model. |
| A model | A dense or GDN checkpoint on disk. This guide uses **Qwen3-8B** (dense, bf16). See [§8](#8-what-works-today--known-gaps) for MoE status. |

> **Shared box?** Check who is using each GPU first —
> `nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv`.
> Pick a **free** GPU with `CUDA_VISIBLE_DEVICES` and never kill another user's
> process.

---

## 2. Get the code

```bash
git clone <your-unieinfra-remote> UnieInfra
cd UnieInfra
# UnieInfra vendors the engine as first-party code (python/, tokenspeed-kernel/,
# tokenspeed-scheduler/). Reference engines (vllm, sglang, ...) are submodules;
# you do NOT need them to build or run:
#   git submodule update --init third_party/vllm   # only if you want to compare
```

---

## 3. Start the Blackwell dev container

sm_120 codegen needs **nvcc ≥ 12.8**, which is *not* present on many hosts. The
simplest path is a prebuilt Blackwell dev image that already has nvcc 12.9 +
`torch 2.11.0+cu129` + the `tokenspeed-*` wheels:

```bash
docker run -d --name unieinfra-dev --gpus all --ipc=host \
  -v "$PWD":/workspace/UnieInfra -w /workspace/UnieInfra \
  --entrypoint bash nctu6/unieinfra-autobuild-dev:cuda-12.9.1-blackwell_only-4ad1404 \
  -lc 'sleep infinity'

# sanity: 4 sm_120 cards, nvcc knows sm_120, torch sees them
docker exec unieinfra-dev bash -lc 'nvidia-smi -L; nvcc --list-gpu-code | grep sm_120; \
  python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_capability(0))"'
# expect: ... sm_120 ...  and  2.11.0+cu129 (12, 0)
```

*Why a container:* it pins the exact Blackwell-capable toolchain (nvcc 12.9,
torch 2.11 cu129) so you are not fighting host CUDA versions. Everything below
runs **inside** `unieinfra-dev`.

---

## 4. Set the runtime environment

The vendored source tracks the engine's `HEAD`, which pins **cu13** kernel wheels
(`tokenspeed-deepgemm`, `tokenspeed-trtllm-kernel`), while the image is cu12.9.
Three environment shims bridge that gap. Put them in a snippet you can `source`:

```bash
cat > /workspace/UnieInfra/scripts/sm120_env.sh <<'EOF'
# 1) HEAD's flashinfer-python (0.6.15) is newer than the image's jit-cache — skip the check.
export FLASHINFER_DISABLE_VERSION_CHECK=1
# 2) cuBLAS headers ship via pip wheels here, not in the CUDA toolkit include dir.
export CPATH=$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/include 2>/dev/null | tr '\n' ':')$CPATH
export LIBRARY_PATH=$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')$LIBRARY_PATH
# 3) The cu13 kernel wheels need cu13 runtime libs (libcudart.so.13, libcublas.so.13)
#    on a cu12.9 base. Point at a complete cu13 lib dir available on this box.
export LD_LIBRARY_PATH=/workspace/jeff/uniebench/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$(ls -d /usr/local/lib/python3.12/dist-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH
EOF
```

> These shims are a **temporary** consequence of building HEAD against a
> pre-built image. The clean fixes (planned) are: pin a self-consistent dev image,
> and make datacenter-kernel imports lazy so sm_120 never loads cu13 wheels it
> won't call. Until then, `source scripts/sm120_env.sh` before every build/run.

---

## 5. Build & install the three packages

UnieInfra is three installable packages: the **kernel** library, the C++
**scheduler**, and the **engine**. Install them in this order.

### 5.1 Kernel — build native kernels for sm_120

```bash
docker exec unieinfra-dev bash -lc '
  cd /workspace/UnieInfra && source scripts/sm120_env.sh
  export TOKENSPEED_CUDA_ARCH=120a          # <-- build for sm_120 (not the B200 default)
  pip install tokenspeed-proton             # small dep the image lacks
  pip install -e tokenspeed-kernel/python --no-build-isolation
'
```
*Why `TOKENSPEED_CUDA_ARCH=120a`:* the build defaults to B200 (`100a/103a`). The
17 hand-written native kernel groups are Blackwell-generic (`__CUDA_ARCH__ ≥
900/1000`, **not** `tcgen05`), so they compile cleanly for sm_120.

### 5.2 Scheduler — C++ (scikit-build-core + OpenSSL)

```bash
docker exec unieinfra-dev bash -lc '
  cd /workspace/UnieInfra && source scripts/sm120_env.sh
  pip install scikit-build-core nanobind cmake ninja "tokenspeed-spdlog==1.15.1"  # build deps
  apt-get update -qq && apt-get install -y -qq libssl-dev                          # CMake find_package(OpenSSL)
  pip install -e tokenspeed-scheduler/ --no-build-isolation
'
```

### 5.3 Engine — Python (skip the unpublished P/D dep)

```bash
docker exec unieinfra-dev bash -lc '
  cd /workspace/UnieInfra && source scripts/sm120_env.sh
  pip install -e python/ --no-deps --no-build-isolation   # --no-deps: skip tokenspeed-mooncake (P/D only, unpublished)
  # install the remaining runtime deps except mooncake:
  python3 - <<PY
import re
d=re.findall(r'"([^"]+)"', re.search(r"dependencies\s*=\s*\[(.*?)\]", open("python/pyproject.toml").read(), re.S).group(1))
open("/tmp/reqs.txt","w").write("\n".join(x for x in d if not x.startswith(("tokenspeed-kernel","tokenspeed-scheduler","tokenspeed-mooncake"))))
PY
  pip install -r /tmp/reqs.txt
'
```

### 5.4 Verify the whole stack imports

```bash
docker exec unieinfra-dev bash -lc '
  cd /workspace/UnieInfra && source scripts/sm120_env.sh
  python3 -c "from tokenspeed.runtime.entrypoints.engine import Engine; print(\"UnieInfra stack import OK\")"
'
```

---

## 6. Run: correctness (single request)

```bash
docker exec unieinfra-dev bash -lc '
  cd /workspace/UnieInfra && source scripts/sm120_env.sh
  export CUDA_VISIBLE_DEVICES=1                              # a FREE gpu
  export UNIEINFRA_SMOKE_MODEL=/workspace/models/Qwen/Qwen3-8B
  export UNIEINFRA_ATTN_BACKEND=triton                       # portable attention on sm_120
  export UNIEINFRA_EAGER=0                                   # CUDA graph ON (5x faster than eager)
  python3 scripts/m1_forward_smoke.py
'
```
Expected — a coherent completion, e.g.:
```
[smoke] === OUTPUT ===
{'text': ' Paris. The capital of Italy is Rome. The capital of Spain is Madrid. ...'}
[smoke] generate(): 133 tok / 1.573s = 84.6 tok/s (eager=False)
[smoke] DONE — forward+decode succeeded on sm_120
```

---

## 7. Run: throughput (high concurrency)

Single-request tok/s is the wrong metric for a serving engine — its value is
**continuous batching under concurrency**:

```bash
docker exec unieinfra-dev bash -lc '
  cd /workspace/UnieInfra && source scripts/sm120_env.sh
  export CUDA_VISIBLE_DEVICES=1 UNIEINFRA_SMOKE_MODEL=/workspace/models/Qwen/Qwen3-8B
  export UNIEINFRA_ATTN_BACKEND=triton UNIEINFRA_N=128 UNIEINFRA_MAX_NEW=128
  python3 scripts/m1_concurrency_smoke.py
'
```
Reference numbers (Qwen3-8B, one RTX PRO 6000, CUDA graph on):

| Scenario | Throughput |
|---|---|
| batch=1, eager | ~16.8 tok/s |
| batch=1, CUDA graph | ~84.6 tok/s |
| **128 concurrent, CUDA graph** | **~7,540 tok/s aggregate** (≈58.9 tok/s/req) |

The ~89× jump from batch=1 to 128 concurrent is the continuous-batching win — the
number that matters for RPS-under-SLO.

---

## 8. What works today / known gaps

| Model class | sm_120 status |
|---|---|
| **Dense** (Qwen3-8B, Llama) | ✅ runs, correct output, high concurrency |
| **GDN hybrid** (Qwen3.5/3.6 attention) | ✅ attention path is portable (Triton) |
| **MoE bf16 unquantized** (Qwen3.6-35B-A3B) | ⚠️ **blocked** — no sm_120 kernel yet for `moe.apply` unquant bf16 (flashinfer MoE is datacenter-only; the Triton MoE covers only quantized weights). Fix in progress: port vLLM's Apache-2.0 `fused_experts` Triton kernel. See `docs/design/tokenspeed-sm120-support.md` §9. |
| **MLA / DSA** (DeepSeek, GLM-5) | ❌ datacenter-only (tcgen05); use the B200/H200 tier. |

---

## 9. Troubleshooting (every error we hit)

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: tokenspeed_triton.profiler` | `tokenspeed-proton` not installed | `pip install tokenspeed-proton` |
| `flashinfer-jit-cache … does not match flashinfer version` | HEAD's flashinfer newer than image's jit-cache | `export FLASHINFER_DISABLE_VERSION_CHECK=1` |
| `fatal error: cublas_v2.h: No such file` (build) | cuBLAS headers are in pip wheels, not the toolkit | add `nvidia/*/include` to `CPATH` (§4) |
| `ImportError: libcudart.so.13 / libcublas.so.13` | cu13 kernel wheels on a cu12.9 base | put a cu13 lib dir on `LD_LIBRARY_PATH` (§4) |
| `BackendUnavailable: scikit_build_core.build` (scheduler) | build backend missing under `--no-build-isolation` | `pip install scikit-build-core nanobind cmake ninja tokenspeed-spdlog` |
| `CMake Error … Could NOT find OpenSSL` (scheduler) | no system OpenSSL dev headers | `apt-get install -y libssl-dev` |
| `No matching distribution … tokenspeed-mooncake` (engine) | P/D-only dep is unpublished | install engine with `--no-deps`, then deps minus mooncake (§5.3) |
| `RuntimeError: An attempt has been made to start a new process…` | your run script has no `if __name__ == "__main__":` guard; the engine uses `spawn` | wrap the body in `if __name__ == "__main__": main()` |
| `TypeError: init(...) Expected 7 but got 8 arguments` in MoE | flashinfer version skew hitting the datacenter MoE kernel | set `moe_backend="triton"` (dense/GDN) — MoE bf16 is the known gap (§8) |
| `NoKernelFoundError: moe.apply … solution 'triton' … weight_dtype unquant` | no bf16 unquant Triton MoE exists yet | the [§8](#8-what-works-today--known-gaps) gap — pending the vLLM MoE port |
| GPU shows large memory used but not by you | shared box — another user's process | pick a free GPU via `CUDA_VISIBLE_DEVICES`; never kill others' PIDs |

---

---

## 10. Build `unieinfra-vllm` (the new engine base) on sm_120

Sections 1–9 build the **retiring** engine (`python/tokenspeed/` +
`tokenspeed-kernel/` + `tokenspeed-scheduler/`). This section builds
**`unieinfra-vllm/`**, UnieInfra's engine base — a hard fork of vLLM at commit
`d02df748`, vendored as source. See `unieinfra-vllm/README.md` for what it is
and `unieinfra-vllm/PROVENANCE.md` for exactly how it differs from upstream.

The two engines are independent. Building one does not touch the other.

### 10.1 Why a separate container

`unieinfra-vllm` and any stock `vllm` both own the `vllm` import name, so they
cannot share a `site-packages`. The `nctu6/unieinfra-autobuild-dev` image
**ships a prebuilt vLLM in `dist-packages`**, which would shadow ours in
confusing ways. Build in a throwaway container of the same image with that
package removed, so the environment provably contains exactly one `vllm`:

```bash
docker run -d --name unieinfra-vllm-build --gpus all --ipc=host \
  -v /workspace:/workspace -w /workspace/roy/UnieInfra \
  --entrypoint bash nctu6/unieinfra-autobuild-dev:cuda-12.9.1-blackwell_only-4ad1404 \
  -lc 'sleep infinity'

docker exec unieinfra-vllm-build bash -lc '
  pip uninstall -y vllm                      # the image ships one; ours replaces it
  python3 -c "import vllm" || echo "good: no vllm in this env"
'
```

### 10.2 One-time environment fixes

The base image is a *runtime* image, so several build-time pieces are simply
absent. Install them once:

```bash
docker exec unieinfra-vllm-build bash -lc '
  apt-get update -qq && apt-get install -y -qq git      # CMake FetchContent clones CUTLASS
  pip install "cmake>=3.26.1" "setuptools-scm>=8" "setuptools-rust>=1.9.0"
  git config --global --add safe.directory "*"          # repo is uid 1000, container is root

  # nvcc dev symlinks: the image ships libnvrtc.so.12 but not the unversioned
  # libnvrtc.so that CMake FindCUDAToolkit looks for.
  cd /usr/local/cuda/lib64 && ln -sf libnvrtc.so.12 libnvrtc.so \
    && ln -sf libnvrtc-builtins.so.12.9 libnvrtc-builtins.so

  # Rust toolchain for the tool-call parser extension (vllm/_rust_tool_parser).
  curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain 1.95 --profile minimal
'
```

### 10.3 Build

```bash
docker exec -d unieinfra-vllm-build bash -lc '
  export PATH="$HOME/.cargo/bin:$PATH"

  # CUDA headers/libs live in pip wheels here, not /usr/local/cuda. Same shim as
  # benchmark/scripts/sm120_env.sh -- but EXCLUDING nvidia/cu13, whose CUDA 13
  # headers sort before the 12.x ones and would shadow the 12.9 toolchain.
  NV=/usr/local/lib/python3.12/dist-packages/nvidia
  export CPATH="$(ls -d $NV/*/include | grep -v /cu13/ | tr "\n" :)$CPATH"
  export LIBRARY_PATH="$(ls -d $NV/*/lib | grep -v /cu13/ | tr "\n" :)$LIBRARY_PATH"

  export TORCH_CUDA_ARCH_LIST="12.0a"          # sm_120 only; the default list triples build time
  export VLLM_VERSION_OVERRIDE="0.1.0+vllm.d02df748"
  export MAX_JOBS=32
  export CMAKE_BUILD_TYPE=Release

  cd /workspace/roy/UnieInfra/unieinfra-vllm
  nice -n 19 pip install --no-build-isolation --no-deps -v -e . 2>&1 | tee /tmp/build.log
'
```

Three flags carry weight:

- **`--no-build-isolation`** — the build must compile against the *installed*
  torch (2.11.0+cu129). Isolation would download upstream's pinned `torch==2.13.0`
  into a temporary env and build against that instead.
- **`--no-deps`** — upstream's `requirements/cuda.txt` also pins `torch==2.13.0`,
  which pip would install over the container's torch. Upstream's own escape hatch
  is `python3 use_existing_torch.py`, but that *rewrites the requirements files*,
  which would put a spurious diff in a vendored tree whose whole value is being
  byte-comparable to the upstream snapshot. `--no-deps` gets the same result
  without touching a file. The base image already carries vLLM's runtime deps.
- **`-e` (editable)** — the installed engine *is* the repo tree. Read-code equals
  run-code by construction, which is the entire reason this fork exists.

**On a shared box:** this compiles 398 targets, 126 of them CUDA. Check
`nvidia-smi` first, keep `nice -n 19`, and suspend it
(`pkill -STOP -f "cicc|cudafe|nvcc|ninja|cc1plus"`, resume with `-CONT`) if
someone starts a benchmark. Numbers taken next to a full-throttle nvcc are not
numbers.

### 10.4 Verify

```bash
docker exec unieinfra-vllm-build bash -lc '
  # (a) the import resolves INSIDE the vendored tree, not to any installed vllm
  python3 -c "import vllm, vllm.version; print(vllm.__file__, vllm.__version__)"
  # (b) every loaded vllm.* module is ours
  python3 - <<PY
import sys, vllm, vllm.model_executor.layers.utils
bad = [m for m,v in sys.modules.items()
       if m.split(".")[0]=="vllm" and getattr(v,"__file__",None)
       and "/unieinfra-vllm/" not in v.__file__]
print("leaked modules:", bad or "none")
PY
  # (c) the UnieInfra kernel arm is live
  python3 -c "
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
print(dispatch_unquantized_gemm().__name__)"   # -> unieinfra_unquantized_gemm
'
```

### 10.5 Troubleshooting (every error this build hit, in order)

Each row is a real failure from the first successful bring-up, not a
hypothetical. All five are **environment** problems; none required a source
change.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `RuntimeError: Cannot find CMake executable` | The base image is a runtime image — no `cmake` binary, despite `cmake --version` working in the older `unieinfra-dev` container (installed there by hand). | `pip install "cmake>=3.26.1"` |
| 2 | `CMake Error … could not find git for clone of cutlass-populate` | No `git` in the image; CMake `FetchContent` clones CUTLASS, vllm-flash-attn, FlashMLA at configure time. | `apt-get install -y git` |
| 3 | `fatal: detected dubious ownership in repository at '/workspace/roy/UnieInfra'` → `metadata-generation-failed` | The repo is owned by uid 1000; the container runs as root. `setuptools-scm`'s file finder shells out to git even when `VLLM_VERSION_OVERRIDE` makes the *version* static. | `git config --global --add safe.directory "*"` |
| 4 | `CUDA_nvrtc_LIBRARY (ADVANCED)` set to `NOTFOUND`, `-- NVRTC: Not Found` | The image has `libnvrtc.so.12` but not the unversioned `libnvrtc.so` dev symlink that `FindCUDAToolkit` looks for. | `ln -sf libnvrtc.so.12 /usr/local/cuda/lib64/libnvrtc.so` (same for `libnvrtc-builtins`) |
| 5 | `fatal error: cublas_v2.h: No such file or directory` (×~40 translation units in `csrc/libtorch_stable/`) | CUDA libraries are pip wheels under `nvidia/*`, not in `/usr/local/cuda/include`. Exactly the case `benchmark/scripts/sm120_env.sh` exists for. | Put `nvidia/*/include` on `CPATH` and `nvidia/*/lib` on `LIBRARY_PATH` — **but filter out `nvidia/cu13/`**, whose CUDA 13 headers sort before the 12.x ones and would shadow the 12.9 toolchain. |

Notably **absent** from this list: any sm_120 *codegen* failure. `TORCH_CUDA_ARCH_LIST="12.0a"` compiled without a single arch-gating error — unlike the retiring engine, whose sm_120 bring-up cost E23/E27/E28/E30. vLLM's CMake already computes per-kernel arch subsets (`FA2_ARCHS`, `FA3_ARCHS`, …) and simply omits kernels sm_120 cannot host.

---

*Keep this guide in sync with its Chinese version (`build-and-run.zh.md`). If you
change one, change both.*
