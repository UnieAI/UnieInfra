# TokenSpeed on GB10 (DGX Spark)

Bring-up and test tooling for running this tree on NVIDIA GB10: compute
capability **12.1** (sm_121), aarch64 Grace CPU, 128 GB of memory shared by CPU
and GPU. Upstream TokenSpeed lists only sm90/sm100/sm103/sm107 as supported, so
this measures where `main` stands on sm_121 rather than assuming it works.

## What GB10 needs from the build

- Native kernels must be compiled for `sm_121a`. The default list targets B200,
  and an `a` target is not forward compatible across minor versions.
- Triton JIT-compiles through `TRITON_PTXAS_PATH`, the toolkit's CUDA 13 ptxas,
  which knows sm_121a.
- The aarch64 host is not a problem: upstream already ships for Grace (GB200),
  the base image is multi-arch, and the pinned kernel dependencies have aarch64
  or pure-Python wheels on PyPI.

## Running it

```sh
# on the workstation, from this repository
unieinfra/gb10/sync.sh                      # HEAD -> gb10-unieai:~/unieinfra-main/src

# on the GB10
R=~/unieinfra-main/src/unieinfra/gb10/run.sh
$R prep      # pull base + bench images, download Qwen3.8-27B
$R check     # level 1: kernels and kernel tests, no model
tmux new -s gb10
$R window    # level 2: serve, smoke test, benchmark
```

Logs go to `~/unieinfra-main/logs/<time>-<step>-<rev>/`.

**Building the image.** With the GB10's resident service stopped, `run.sh`
builds `Dockerfile` on the host (it refuses below 48 GB free). With the service
running, build the wheels on any arm64 Docker host instead -- an Apple-silicon
Mac works, no GPU needed -- with `build_wheels.sh`, and `sync.sh` ships them;
`run.sh` then only installs them, in a memory-capped container.

**Level 1, `check`.** Runs `level1.sh` in a container capped at
`GB10_CHECK_MEM_GB` (5 GB), with no network, under a watchdog that kills it if
host `MemAvailable` drops below `GB10_GUARD_MB` (1.5 GB) -- a cgroup cap may not
see GPU allocations on unified memory. It checks that the device is sm_121, that
every prebuilt kernel library carries sm_121 SASS and loads, lists kernel
plugins, and runs a set of kernel tests. Safe beside the resident service.

**Level 2, `window`.** Stops the resident service if it is running, and then
restarts it on every exit path; a service stopped by hand stays stopped. Serves
the model (default Qwen3.8-27B, a BF16 GDN hybrid) with `tokenspeed serve` on
`127.0.0.1:18080` with the same engine flags as the GB10 CI model tests, asks
one temperature-0 question twice with thinking off, then load-tests the server with `vllm bench serve` (from the
vLLM image already on the box) at concurrency 1/4/16/64 on two shapes: unique
1024-token prompts, and a shared 1024-token prefix plus 256 new tokens.

Knobs: `GB10_MODEL_REPO`, `GB10_SERVE_ARGS`, `GB10_MEM_UTIL` (default 0.6),
`GB10_CONCURRENCY`, `GB10_BENCH_MIN_PROMPTS`, `GB10_PORT`, `GB10_SERVICE`,
`GB10_CHECK_MEM_GB`, `GB10_GUARD_MB`, `GB10_BENCH_IMAGE`.

The image is tagged `unieinfra:gb10-dev`, not `unieinfra:gb10-main`, which the
GB10 CI runs its tasks in.

The test server binds 127.0.0.1 on its own port on purpose: the host runs a
cloudflared tunnel, and anything on the service's port would be published
through it while the service is down.

## Not covered

- `nvidia-smi` reports `memory.total` as `[N/A]` on GB10; the engine falls back
  to `torch.cuda.mem_get_info()`, which sees the whole unified pool.
- The UnieAI sm_12x kernel plugin lives on the `unieai` branch
  (`unieinfra/kernel-sm120`); this tree does not carry it.
