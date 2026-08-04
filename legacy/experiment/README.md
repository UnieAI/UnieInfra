# `experiment/` — not part of the shipped project

Everything here is research residue: superseded code, raw measurement output, and
profiler captures. It is kept because it is the evidence behind claims in
`docs/experiments-log.md`, not because it is maintained.

| Path | What it is |
|---|---|
| `legacy/` | The retiring tokenspeed-derived engine (`tokenspeed-kernel`, `-kernel-amd`, `-mla`, `-scheduler`, the Python runtime, its tests). Moved with `git mv`, history preserved. Still holds things not yet extracted — leaf-first eviction with root-path ref-count pinning, the capturable-grammar-inside-the-CUDA-graph trick, and the RAII KV-ownership design lesson. Do not delete until those are absorbed or explicitly abandoned. |
| `nsys_out/` | Old Nsight Systems captures and derived CSVs. |
| `results/` | Raw per-request benchmark JSON (moves here once no agent is writing to it). The owner requires all raw kept — it is the evidence base for every performance number. |
| `profiles/` | Nsight Compute / Systems captures. Large and regenerable; `.ncu-rep`/`.nsys-rep`/`.sqlite` are gitignored. |

The shipped project is everything **outside** this directory:
`unieinfra-vllm/` (the engine), `unieinfra-kernel/`, `unieinfra-scheduler/`,
`docs/`, `benchmark/` (the harness itself), `docker/`, `scripts/`, and
`third_party/` (reference submodules).
