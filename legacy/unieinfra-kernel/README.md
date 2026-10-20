# unieinfra-kernel

UnieInfra's **capability-based kernel selection layer**, plus the first-party /
KDA-generated / community kernels that register into it. The package is
base-independent: it runs on the tokenspeed base, and it runs inside **stock
vLLM** as a plugin with no vLLM source changes.

## The two halves

```
unieinfra_kernel/
  selection/         # the moat: registry + selection + platform + signature
    registry.py      #   KernelSpec, KernelRegistry, Priority, register_kernel
    selection.py     #   select_kernel + capability/trait/objective/origin scoring
    platform.py      #   PlatformInfo, CapabilityRequirement, ArchVersion
    signature.py     #   role-indexed tensor FormatSignature
  boundary.py        # the ONE import site for the above (see below)
  dispatch.py        # dense-linear façade -> select_kernel, as a torch custom op
  __init__.py        # importing the package triggers discovery
  _discovery.py      # recursively import ops/ kda/ contrib/ so kernels self-register
  ops/       <family>/<solution>.py   # first-party hand-written / SOTA-ported
  kda/       <family>/<solution>.py   # KDA-generated kernels land here
  contrib/   <family>/...             # community contributions
  contracts/ <family>.py              # frozen KernelContract per op
  integrations/vllm/ # docs + re-export of the vLLM adapter
unieinfra_vllm_plugin.py  # the vllm.general_plugins entry point, deliberately
                          # OUTSIDE the package: vLLM loads it in every process,
                          # and a run that does not want us must import nothing
```

The selection core imports only `torch` and the standard library — no engine, no
vendor library — which is what lets it move between bases untouched.

Drop a `@register_kernel(...)`-decorated file anywhere under `ops/`, `kda/`, or
`contrib/` and it is registered and selectable on next import — zero edits to
the registry or selection core.

## Why a separate package

- `tokenspeed-kernel` is a *fork of an upstream base*. Keeping UnieInfra's own
  kernels out of that tree means upstream syncs stay clean and every UnieInfra
  kernel is attributable to its origin.
- The selection core has to survive a change of engine base. Owning it here, as
  an importable distribution with no engine dependency, is what makes that
  possible.

## One registry, whichever base you are on

`unieinfra_kernel.boundary` picks the registry once per process:

| Situation | `boundary.BACKEND` |
|---|---|
| `tokenspeed_kernel` is importable (tokenspeed base) | `tokenspeed` — register into *its* singleton, so a runtime that only knows the `tokenspeed-kernel` boundary (per `AGENTS.md`) transparently sees our kernels |
| It is not (vLLM base, bare Python) | `standalone` — the vendored core above |
| `UNIEINFRA_KERNEL_REGISTRY=standalone\|tokenspeed` | forced |

There is never more than one registry in a process.

## Running inside vLLM

```bash
pip install -e unieinfra-kernel
vllm serve /models/Qwen3-8B --quantization unieinfra
```

`unieinfra` is a **dispatch selector, not a numeric format** — weights stay in
whatever format the checkpoint ships and weight loading is inherited unchanged.
The `vllm.general_plugins` entry point installs the config in every vLLM
process; with the flag absent, not one `unieinfra_kernel` module is imported and
vLLM behaves exactly as upstream.

Measured on Qwen3-8B bf16 / sm_120 / vLLM 0.25.1: **batch-1 decode +4.2%** with
the E33 decode GEMV selected for `qkv_proj`, `o_proj` and `down_proj`. See
`docs/design/vllm-kernel-plugin.md` and E34 in `docs/experiments-log.md`.

### Quantized checkpoints

The same flag serves fp8 and NVFP4. On a quantized checkpoint the plugin
**delegates the entire checkpoint path** — parameter creation, the weight
loader, post-load processing, MoE, KV-cache scales — to the vLLM quantization
config that owns the format, and overrides exactly one method,
`LinearMethodBase.apply`:

```bash
# NVFP4 (ModelOpt checkpoint). sm_120 needs the CUTLASS FP4 GEMM and a
# non-FlashInfer attention backend; both arms of a comparison need the same two.
vllm serve nvidia/Qwen3-8B-NVFP4 --quantization unieinfra \
    --linear-backend cutlass --attention-backend TRITON_ATTN

# fp8, quantized online from a bf16 checkpoint (vLLM's own `--quantization fp8`)
UNIEINFRA_VLLM_QUANT=fp8 vllm serve /models/Qwen3-8B --quantization unieinfra
```

What the registry adds on a decode step is a **weight-only** GEMV: the
activation stays bf16 while the quantized weight is streamed once and
dequantized in-register (`unieinfra_decode_gemv_fp8_triton` = W8A16,
`unieinfra_decode_gemv_nvfp4_triton` = W4A16). Above `M = 2` nothing matches and
dispatch calls the engine's own scaled GEMM, so those shapes are bit-identical
to stock vLLM. See E38 in `docs/experiments-log.md` for the measured deltas.

Environment knobs:

| Variable | Default | Meaning |
|---|---|---|
| `UNIEINFRA_VLLM_QUANT` | `auto` | `auto` reads the checkpoint's `quantization_config`; `dense`/`fp8`/`nvfp4` force a mode. `fp8` on a bf16 checkpoint gives vLLM's online per-tensor fp8. |
| `UNIEINFRA_VLLM_LAYERS` | `all` | Comma-separated prefix substrings that opt in, or `all`. The trait gate does the real confinement. |
| `UNIEINFRA_KERNEL_LOG_SELECTION` | `0` | Log every distinct `(M, N, K) -> kernel` decision. |
| `UNIEINFRA_KERNEL_SELECTION_DUMP` | *(unset)* | Path prefix; writes `<prefix>.<pid>.json` with the selection record and kernel counters. Needed to see inside vLLM's worker process. |
| `UNIEINFRA_VLLM_DISABLE` | `0` | Load the plugin but claim no layer (A/B control arm). |
| `UNIEINFRA_DECODE_GEMV` | `1` | `0` keeps the dense kernel registered but makes it decline, so the wrapper delegates to `F.linear`. |
| `UNIEINFRA_DECODE_GEMV_FP8` / `_NVFP4` | `1` | `0` **unregisters** the quantized kernel, so dispatch finds nothing and the engine's own GEMM runs — the clean control arm. |

## The `origin` axis

Every kernel carries an `origin` (`vendor` | `first_party` | `kda` | `contrib`).
It is the **lowest-weight selection tiebreaker**: it only decides an otherwise
exact `(oracle, objective, priority)` tie. Because all in-tree `tokenspeed`
kernels default to `origin="vendor"`, this axis can never reorder the existing
corpus — it only lets a UnieInfra kernel win a tie it would previously have lost
to registration order. Use **priority** (not origin) to actually change a
winner. See `docs/design/kernel-plugin-contract.md`.

## Usage from Python

```python
import unieinfra_kernel  # noqa: F401  -> discovery runs, kernels self-register
```

On the tokenspeed base the engine wires this in automatically (see
`tokenspeed/runtime/execution/model_runner.py`); a missing package is tolerated.
