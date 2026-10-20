# MoE path — Qwen3.6-35B-A3B, batch 1, sm_120

## Question

The torch profiler put ~40% of GPU time at batch 1 in the MoE path and 1.9% in
attention, which chose the target. It cannot say *why* the MoE path is slow, and
the answer decides the work: a DRAM-bound kernel wants fewer round trips
(fusion), an occupancy-bound one wants different launch geometry, a
latency-bound one wants neither.

## Setup

| | |
|---|---|
| model | `/workspace/models/Qwen/Qwen3.6-35B-A3B` (256 experts, 30 linear + 10 full attention layers) |
| workload | `scratchpad/prof_moe_workload.py`, batch 1, 16 decode steps after warm-up |
| ncu | `2025.2.1`, `--section SpeedOfLight/MemoryWorkloadAnalysis/Occupancy`, `-k regex:fused_moe\|MoeFCGemm\|topkGating\|finalizeMoeRouting\|expandInput`, `--launch-skip 20 --launch-count 8` |
| container | `unieinfra-ncu` (created for this: same image, `--cap-add=SYS_ADMIN`) |

**`enforce_eager=True`.** vLLM captures decode into CUDA graphs and kernels
inside a captured graph are awkward to attribute. Eager changes the schedule, so
these numbers describe **per-kernel efficiency, not end-to-end share**.

**No `quantization="unieinfra"`.** The profiling container resolves vLLM from
the editable tree but not `unieinfra_kernel`, so that method is unregistered
here. It does not affect the kernels under the lens — `expandInputRows`,
`finalizeMoeRouting` and `topkGating` are TRT-LLM/vLLM's own MoE path. It does
change the dense-linear path, which reverts from `_decode_gemv_kernel` to
torch's. **Do not read dense-linear numbers off this run.**

## Result

| kernel | duration | SM % | DRAM % | occupancy |
|---|---|---|---|---|
| `expandInputRowsKernel` | 489.8 / 479.1 us | **13.0 / 13.3** | **84.7 / 85.0** | 85.2 |
| `finalizeMoeRoutingKernel` | 410.5 / 409.2 / 410.7 us | **20.3 / 20.4 / 20.4** | **90.7 / 91.0 / 90.6** | 80.8 |
| `topkGating` | 26.5 / 26.9 / 26.6 us | 68.7 / 68.2 / 68.1 | 19.9 / 19.6 / 19.8 | 79.4 |

Repeat launches agree to within 2%, so these are kernel properties rather than
sampling noise.

## Diagnosis

**The two longest kernels are pure data movement, and they are DRAM-bound.**
`expandInputRows` and `finalizeMoeRouting` run at 85-91% of DRAM throughput
while using 13-20% of the SMs. They are the gather and scatter around the expert
GEMMs: tokens are permuted into per-expert contiguous buffers, the GEMMs run,
and the results are permuted back.

`topkGating` is the opposite and is not worth attention: 68% SM, 20% DRAM, and
26 us against their ~450.

Note this contradicts a first reading taken from a truncated log, which reported
"Mem Pipes Busy 68%, not DRAM-bound". That was `topkGating` — the only kernel of
the three that is not DRAM-bound. Reading one kernel's metrics as the section's
is an easy mistake when the tail of a report is all that survives.

## INVALIDATED — this run profiled the wrong configuration

The recommendations below were derived from this run's ranking and **do not
hold**. Reconciled against the deployment-config torch profile (CUDA graphs,
`--quantization unieinfra`), same model, same batch size, 241.27 ms total:

| kernel | ms | % of GPU time |
|---|---|---|
| `fused_moe::run_global` | 50.27 | **20.8** |
| `MoeFCGemm` x2 | 26.72 | 11.1 |
| `topkGating` | 5.99 | 2.5 |
| `finalizeMoeRouting` | 5.30 | **2.2** |
| `expandInputRows` | 2.42 | **1.0** |

The gather/scatter pair holds **3.2%** of GPU time in deployment, not the
leading share this run showed. Eliminating them entirely caps out at 3.2%, so
the fusion below is worth hours at most, not days.

The two concessions made to get ncu running -- `enforce_eager` and dropping
`quantization="unieinfra"` -- are exactly what invalidated it. Without the
UnieInfra dispatch the MoE path routes through `MoeFCGemm` + expand/finalize;
with it, through `fused_moe::run_global`, a kernel that was in the `-k` regex
and never appeared in a sampled launch because that path was not taken.

**The real target is `fused_moe::run_global` at 20.8%**, and profiling it
requires ncu to work with CUDA graphs and with `unieinfra_kernel` resolvable in
the profiling container. Both are environment problems, not kernel problems.

The measurements themselves stand: `expandInputRows` and `finalizeMoeRouting`
genuinely are DRAM-bound data movement at 85-91% of peak. That is a true
statement about those kernels and a misleading one about this workload.

## Recommendation, ranked — SUPERSEDED, see above

1. **Fuse the gather into the first expert GEMM and the scatter into the
   second.** Removes two full DRAM round trips of the activation tensor. The
   kernels being DRAM-bound is what makes this worth doing: saved traffic
   converts directly to time, with no second bottleneck waiting behind it. This
   is what production fused-MoE kernels do, so the design is not speculative —
   the work is making it exist for this shape on sm_120.

2. **Occupancy on the two movement kernels.** 80-85% achieved against 100%
   theoretical; ncu's rule estimates ~20% local speedup. Secondary — closing an
   occupancy gap on a kernel already at 90% of DRAM peak recovers little.

3. **Nothing on `topkGating`.** 26 us, compute-bound, 5% of the trio's time.

## Open

The share of end-to-end time these kernels hold **in the deployment
configuration** is unmeasured. This run is eager and unquantized; the
torch-profiler run (cudagraph, `--quantization unieinfra`) put
`fused_moe::run_global` at 50.3 ms against `expandInput` 2.4 ms and `finalize`
5.3 ms, a very different ranking. Reconciling those two views is the next step,
and it decides whether the fusion above is worth days or hours: an efficiency
finding on a kernel that holds 3% of the time is not a throughput finding.
