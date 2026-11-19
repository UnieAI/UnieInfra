## Iris two-shot threshold knob and ROCm version A/B tooling

Stacked on #44 (`mi350p-final`).

1. **`TOKENSPEED_IRIS_TWOSHOT_MIN_BYTES`** pins the producer-direct two-shot cutover for a whole run (0 = always two-shot; a value above the largest payload = never). Decode-sized reductions never reach the default thresholds, so this is the only way to measure the other geometry there. Malformed values are ignored with a warning. New test `tokenspeed-kernel/test/amd/ops/test_iris_two_shot_threshold.py` (5 passed).
2. **ROCm version A/B in the hardware skill** (`.skills/unieinfra-hw-optimize`):
   - `scripts/rocm_overlay.sh`: build a torch overlay whose bundled ROCm user-space (HIP, HSA, RCCL, hipBLASLt, rocBLAS, comgr, profiler libs) comes from another ROCm tree, so one Python environment can A/B two runtimes.
   - `scripts/rocm_micro.py` / `scripts/rocm_allreduce.py`: launch, graph replay, BF16 GEMM, bandwidth and 4-GPU RCCL microbenchmarks.
   - `references/10-rocm-versions.md`: method, required steps (swap all runtime libs together, ship `.kpack`, LLVM-matched upstream triton, how to read the loaded HIP version) and the MI350P results for ROCm 7.2.4 vs 7.14.1 vs 10.0; three new pitfalls in `09-pitfalls.md`.

Verified: overlay built from ROCm 7.14.1 reports HIP 71460850 and runs the microbenchmark (GEMM 256×7168×2048 27.3 µs).

