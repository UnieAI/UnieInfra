<!-- UnieInfra — hardware-universal, extreme-performance LLM inference with a vLLM interface -->

# UnieInfra

**A hardware-universal, extreme-performance LLM inference engine with a vLLM-compatible interface.**

UnieInfra pairs the usability of vLLM with an engine core engineered to push the
whole latency–throughput frontier — formally, to **maximize requests/sec under
SLO** (`max RPS s.t. TTFT_p99 ≤ X ∧ TPOT_p99 ≤ Y`) — across a wide range of
accelerators.

> Status: **early bring-up.** This repository is a hard fork of
> [TokenSpeed](https://github.com/lightseekorg/tokenspeed) (MIT), being turned
> into UnieInfra's first-party foundation. See `NOTICE` for attribution and
> `README.tokenspeed.md` for the upstream engine's original README.

## Why this shape

- **vLLM UX, decoupled from performance.** The engine, scheduler, models, and
  kernels are separable — a vLLM-compatible frontend sits on top; performance
  lives underneath.
- **A capability-based, pluggable kernel layer is the spine.** Every op has
  multiple backend *solutions* (`ops/<family>/<solution>`) selected by hardware
  capability, with an autotune path that benchmarks candidates and picks the
  fastest. This is purpose-built for **agent-driven kernel optimization** —
  generated kernels drop in as new solutions and win selection on merit, with no
  manual tuning.
- **Portability tiers.** NVIDIA datacenter (native) → NVIDIA sm_120 workstation
  → AMD → (future) Ascend / Qualcomm / other NPUs.

## Architecture & design

- `docs/design/unieinfra-architecture.md` — component stack (① frontend →
  ⑦ comm) and the extension map: where each thing we build lands.
- `docs/design/tokenspeed-sm120-support.md` — first hardware bring-up:
  RTX PRO 6000 (sm_120) support.

## Layout

```
python/               engine (AsyncLLM, scheduler client, models, execution)
tokenspeed-kernel/    capability-based kernel registry + per-op solutions   ← the spine
tokenspeed-scheduler/ C++ FSM control plane (P/D roles, admission, batching)
tokenspeed-kernel-amd/ AMD (ROCm) kernels
docs/design/          UnieInfra design docs
third_party/          reference submodules: vllm, modular, tilert (not vendored)
```

## License

MIT — see `LICENSE`. UnieInfra is an independent project and is not affiliated
with the LightSeek Foundation; see `NOTICE`.
