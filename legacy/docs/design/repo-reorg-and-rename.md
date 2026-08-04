# Design: Repo Reorg + Rename (`tokenspeed` → `unieinfra`) — Migration Plan

Status: **Plan only — not executed.** Zero files moved, renamed, or edited by
this doc's authoring pass (investigation was read-only: `grep`/`ls`/`find`/`git`
only). · Date: 2026-07-25 · Owner: royshih

> This is a design/runbook doc, not a changelog entry. It describes a migration
> that has **not happened yet**. Do not treat any path in this document as
> current until the corresponding commit (see [§ 7](#7-recommended-sequencing))
> has actually landed — check `git log` / the tree, not this file, for ground
> truth on "did it happen."

No `.zh.md` mirror: `docs/design/*` has no bilingual pairing precedent today
(checked — `kernel-plugin-contract.md`, `scheduling-improvements.md`,
`tokenspeed-sm120-support.md`, `unieinfra-architecture.md` are all English-only).
User-facing docs (`README`, `docs/support-matrix.md`, `benchmark/README.md`) do
pair with `.zh.md`; this internal engineering runbook does not need to.

---

## 0. Preconditions for execution (not for this doc)

This doc only *documents* the plan. Before anyone executes it:

1. Quiet tree: no other agent/human has uncommitted or in-flight edits under
   `python/`, `tokenspeed-*/`, `unieinfra-kernel/`, `test/`, `benchmark/`,
   `docs/`, `scripts/`, `.github/`. At the time this doc was written, a
   separate subagent was actively editing engine source — that work must be
   committed (or the migration deferred until it is) before execution starts.
2. `git status` clean, on a dedicated branch (not `main`, not `unieinfra-base`
   directly — see [§ 7](#7-recommended-sequencing)).
3. No GPU needed for any step below — this is a pure source/text migration.
   Verification's "engine smoke" step is CPU-import-only unless the executor
   chooses to also run it on GPU hardware (optional, not required to validate
   the migration itself).

---

## 1. Blast radius (measured, read-only)

All counts gathered with `grep`/`find`/`ls` against the working tree at commit
`65af681`, excluding `third_party/` (git submodules — out of scope, not ours)
and `.git/`, `build/`, `__pycache__/`, `.pytest_cache/` artifacts.

| Category | Count | Evidence / command |
|---|---|---|
| `import`/`from tokenspeed…` statements | **1825** across **272 files** | `grep -rlE '^\s*(import\|from)\s+tokenspeed' python/ \| wc -l` → 272; `grep -rE '^\s*(import\|from)\s+tokenspeed' python/ \| wc -l` → 1825 (verified, matches the pre-measured figure). |
| `TOKENSPEED_*` env var identifiers (unique) | **105** | `grep -rhoE 'TOKENSPEED_[A-Z0-9_]+' . --exclude-dir=third_party --exclude-dir=.git ... \| sort -u \| wc -l`. These are a **user-facing contract** — anyone with a shell script or systemd unit exporting `TOKENSPEED_KERNEL_BACKEND` etc. breaks silently if we just delete the names. |
| Files/dirs with `tokenspeed` in the *name* (not content) | **42**, excl. `third_party/` | `find . -iname "*tokenspeed*" -not -path "./third_party/*" -not -path "./.git/*"`. Breakdown: `.github/` 11, `tokenspeed-scheduler/` 7+1(dir itself), `test/` 6, `tokenspeed-kernel/` 5+1, `python/` 3, `assets/` 2, `tokenspeed-mla/` 1+1, `tokenspeed-kernel-amd/` 1+1, `docs/` 1, `README.tokenspeed.md` 1. |
| Files containing the string `tokenspeed` (any case) | **1019** | `grep -rIl "tokenspeed" . --exclude-dir={third_party,.git,build,__pycache__,.pytest_cache} \| wc -l`. This is the honest "how big is this" number — most of it is the 272-file import blast radius plus test fixtures/CI configs. |
| `pyproject.toml` files (first-party) | 6: `python/pyproject.toml` (`name="tokenspeed"`, `[project.scripts] tokenspeed / ts`), `tokenspeed-kernel/python/pyproject.toml` (`name="tokenspeed_kernel"`), `tokenspeed-kernel-amd/pyproject.toml`, `tokenspeed-mla/pyproject.toml`, `tokenspeed-scheduler/pyproject.toml` (`name="tokenspeed-scheduler"`), `unieinfra-kernel/pyproject.toml` (`name="unieinfra_kernel"` — **already correctly named**, added this cycle). | Read directly. |
| `[tool.setuptools.packages.find]` | `python/pyproject.toml`: `include = ["tokenspeed*"]` — a glob, must become `include = ["unieinfra*"]`. | Read directly. |
| `[project.scripts]` (console entry points) | `tokenspeed = "tokenspeed.cli:main"`, `ts = "tokenspeed.cli:main"` | `python/pyproject.toml` lines ~93-95. No other first-party `pyproject.toml` in this repo declares `console_scripts`/`entry_points` (checked all 6). |
| `.isort.cfg` | `known_first_party=tokenspeed` — must become `unieinfra` (and arguably also list `unieinfra_kernel` if isort should treat it as first-party too — currently it does *not* list `tokenspeed_kernel` either, so leave kernel packages out, consistent with existing convention). | Read directly. |
| `.pre-commit-config.yaml` | **0** `tokenspeed` references. isort/black/nbstripout hooks are name-agnostic. No edit needed here beyond what `.isort.cfg` already covers. | Read directly. |
| `.gitmodules` | **0** `tokenspeed` references (submodules are `vllm`, `tilert`, `modular`, `sglang`, `lmdeploy`, `kernel-design-agents`, `ncu-report-skill`, `mlsys2026-flashinfer-contest`, `TensorRT-LLM`). Not in scope. | Read directly. |
| `.github/workflows/*.yml` | **28 files** reference `tokenspeed`/`TokenSpeed`/`TOKENSPEED` (out of ~24 workflow files — some hit multiple times, plus `.github/CODEOWNERS` and 2 `ISSUE_TEMPLATE` files). Several workflow **filenames** encode the name directly: `release-tokenspeed.yml`, `release-tokenspeed-kernel.yml`, `release-tokenspeed-kernel-amd.yml`, `release-tokenspeed-kernel-rocm.yml`, `release-tokenspeed-scheduler.yml`, `update-tokenspeed-kernel-amd-dependency.yml`, `update-tokenspeed-kernel-amd-version.yml`, `update-tokenspeed-kernel-flashinfer.yml`, `update-tokenspeed-kernel-mla.yml`, `update-tokenspeed-mla-version.yml`, `update-tokenspeed-smg-version.yml`. | `ls .github/workflows/`, `grep -rl tokenspeed .github/`. |
| `.github/CODEOWNERS` | 4 path rules keyed on old paths: `/python/tokenspeed/runtime`, `/tokenspeed-kernel`, `/tokenspeed-kernel-amd`, `/tokenspeed-mla`, `/tokenspeed-scheduler`. Every one must be updated to the new `src/...` paths or CODEOWNERS silently stops applying (GitHub does not error on stale paths — it just matches nothing). | Read directly. |
| `docker/` | 3 references across `Dockerfile.nvidia`, `Dockerfile.amd`, `Dockerfile.release`: base image tags `lightseekorg/tokenspeed-runner*` (external, not ours to rename), and `pip install tokenspeed-kernel/python/`, `pip install tokenspeed-scheduler/`, `./tokenspeed-kernel-amd` (these are **path** references into this repo's own directories — must track the `git mv` destinations). | `grep -n tokenspeed docker/*`. |
| `scripts/` | Directory itself is not tokenspeed-named. Contains `tcp_nodelay_preload.c` / `libtcp_nodelay_preload.so` — the **LD_PRELOAD TCP_NODELAY shim**, located at build time by `python/tokenspeed/cli/_proc.py:_repo_scripts_dir()` via a **hardcoded "3 levels up + `scripts`"** path computation (`dirname(__file__)/../../../scripts`). See [§ 6 risk register](#6-risk-register) — this is a structural coupling, not a text reference, and it survives a same-depth package rename but **breaks silently** (falls back to "no shim, no crash") if `scripts/` is renamed without updating the literal `"scripts"` string in `_proc.py`. | `Read` on `python/tokenspeed/cli/_proc.py` lines 34-121. |
| `benchmark/` | 17 content references; also `benchmark/launch/launch_unieinfra.sh` **already uses the `unieinfra` name** for the JSON launch-config it writes (`results/unieinfra.launch.json`, `"engine":"unieinfra"`) while its comment and payload still shell out to the `tokenspeed serve` CLI verb. This file is a preview of the target state and a concrete example of what the CLI-name rename must keep working. | `ls benchmark/launch/`, `Read` on `launch_unieinfra.sh`. |
| `test/` | **195** files reference `tokenspeed`. Largest concentrations: `test/runtime/` (80, incl. `test/runtime/models` 17, `test/runtime/distributed` 13), `test/ci/eval` (20), `test/ci_system` (9), `test/cli` (8). Also **5 fixture directories are literally named `tokenspeed`**, siblings to `vllm`/`sglang`/`trtllm` engine-comparison dirs: `test/random_benchmark/tokenspeed/`, `test/long_context_benchmark/tokenspeed/`, `test/agentic_benchmark/{glm5.2,minimax_m3,kimi_k2.5}/tokenspeed/`. These name **our own engine** in a benchmark-harness fixture tree and should become `unieinfra/` under the rename — this is a plain `git mv` per directory, but every CI YAML/script under `test/ci/*` and `test/agentic_benchmark/*/agentic_bench.sh` that hardcodes the `.../tokenspeed/...` path must move with it. | `grep -rl tokenspeed test/ \| wc -l`; `find test -type d -iname tokenspeed`. |
| `docs/` | 17 files reference `tokenspeed`/`TokenSpeed`, incl. `docs/design/tokenspeed-sm120-support.md` (filename itself), `docs/.vitepress/config.mts` (`base: "/tokenspeed/"`, GitHub links to `lightseekorg/tokenspeed`), `docs/package.json` (`"name": "tokenspeed-docs"`). | `grep -rl tokenspeed docs/`. |
| String-literal module paths (not `import`/`from`) | **2** confirmed: `"tokenspeed.runtime.cache.storage.mooncake_store.mooncake_store"` (dynamic import string in `backend_factory.py:178`) and `"tokenspeed.runtime.models"` (`package_name` used for pkgutil-style discovery in `registry.py:93`). Both must be rewritten alongside the mechanical import rename — a `sed`/`ruff` pass over `import`/`from` statements alone **will miss these**. | `grep -rn '"tokenspeed\.\|'"'"'tokenspeed\.' python/tokenspeed --include='*.py'`. |
| `setproctitle()` process names | 3: `"ts-serve"` (`cli/serve_smg.py:662`), `"tokenspeed::data_parallel_controller"` (`runtime/engine/data_parallel_controller.py:352`), `"tokenspeed::scheduler_{dp_rank}"` (`runtime/engine/event_loop.py:1929`). These are what shows up in `ps`/`top` on a running box — an operational/observability surface, not just cosmetics. | `grep -rn setproctitle\( python/tokenspeed`. |
| `LD_PRELOAD` / `TCP_NODELAY` shim | 1 first-party mechanism, spans `scripts/tcp_nodelay_preload.c` + `scripts/libtcp_nodelay_preload.so` (built artifact) + `python/tokenspeed/cli/_proc.py` (consumer, gated by env var `TOKENSPEED_GATEWAY_TCP_NODELAY`). See `scripts/` row above and [§ 6](#6-risk-register). | `grep -rln LD_PRELOAD . ...`; `Read` on `_proc.py`. |
| `PACKAGE_LIST` / dependency-name literals | `python/tokenspeed/env.py:PACKAGE_LIST` hardcodes the string `"tokenspeed"` as the first entry (used for `importlib.metadata.version()` lookups, i.e. **this literal must equal the installed PyPI distribution name**, not a Python import path). `test/test_env_package_list.py` independently hardcodes the *set* `{"tokenspeed", "tokenspeed-kernel", ...}` and asserts it equals `PACKAGE_LIST` reconciled against `pyproject.toml` — this test will fail the instant the rename lands unless updated in the same commit. | `Read` on `env.py` lines 42-44; `Read` on `test_env_package_list.py`. |
| External PyPI deps that **cannot** be renamed | `tokenspeed-kernel>=0.1.3.dev0`, `tokenspeed-mooncake>=0.3.11.post20260527` (as given) — **plus, newly found**: `tokenspeed-smg==1.8.0.post20260720`, `tokenspeed-smg-grpc-proto==0.4.14.post20260720`, `tokenspeed-smg-grpc-servicer==0.7.0.post20260720`, and (build-only) `tokenspeed-spdlog==1.15.1` pulled in by `tokenspeed-scheduler/pyproject.toml`'s `[build-system].requires`. None of `smg`/`smg-grpc-proto`/`smg-grpc-servicer`/`spdlog` have any source directory in this repo (not in `third_party/`, not top-level) — they are closed/external prebuilt packages we only consume. Two of them additionally **bake `tokenspeed` into their own internal module paths**, which our code imports by that literal name: `from smg_grpc_proto.generated import tokenspeed_scheduler_pb2` / `tokenspeed_scheduler_pb2_grpc` (`control_server.py`), and the default engine module string `"smg_grpc_servicer.tokenspeed"` (`cli/_proc.py:_ENGINE_MODULE_DEFAULT`). Renaming these would require the upstream/vendor to cut new releases — out of scope, and not something this repo controls. | `grep -rn tokenspeed-smg python/pyproject.toml`; `grep -rn tokenspeed_scheduler_pb2\|smg_grpc_servicer python/tokenspeed`. |
| `tokenspeed_scheduler` — prebuilt binary | `python/tokenspeed_scheduler` wraps a compiled extension. The C++ binding hardcodes the module name at the source level: `NB_MODULE(tokenspeed_scheduler_ext, m)` in `tokenspeed-scheduler/bindings/python_module.cpp:108`. Consumers: `runtime/cache/executor/{memory_executor,host_executor,flat_memory_executor,storage_executor}.py`, `runtime/configs/paged_cache_spec.py` (also reads `tokenspeed_scheduler.FLAT_KVCACHE` — **currently `False`, i.e. this checkout's prebuilt `.so` was built without `TOKENSPEED_FLAT_KVCACHE`**, per the task's stated fact and confirmed by the accessor code), `runtime/engine/event_loop.py`, `runtime/engine/scheduler_utils.py`, `runtime/pd/{decode_executor,prefill_executor,kv_events}.py`, `runtime/layers/attention/kv_cache/mha.py`. Renaming means: edit `NB_MODULE(...)` in C++, edit `tokenspeed-scheduler/pyproject.toml` `wheel.packages`, **rebuild the `.so`** (`tokenspeed-scheduler/build/tokenspeed_scheduler_ext.cpython-312-x86_64-linux-gnu.so` currently exists as a build artifact), then update every one of the ~13 importing files. Recommend **deferring** this rename to a follow-up (see [§ 3](#3-the-rename-boundary)). | `Read` on `python_module.cpp`; `grep -rn tokenspeed_scheduler python/tokenspeed`. |
| Asset filenames | `assets/banner/tokenspeed-banner.png`, `assets/perf/tokenspeed-kimi-k2.5-performance.png` — not currently referenced from `README.md` (grepped, zero hits), so likely referenced from `docs/` or dangling. Low priority, cosmetic. | `find assets`; `grep -n assets/ README.md` (no hits). |
| `README.tokenspeed.md` | Referenced by name in exactly 3 places: `README.md:15`, `NOTICE:12`, `CHANGELOG.md:22`, all as "the preserved upstream README, see this file." | `grep -rn README.tokenspeed .`. |
| `.skills/` | 2 files (`bisect-triton-release.md`, `optimize-amd-gpu-kernel.md`), no `tokenspeed` in filenames; skimmed, out of scope for this pass (agent-tooling convention, not source org). | `ls .skills/`. |

**Sanity check on the pre-measured numbers**: the 1825/272 import figure is
confirmed exactly. The task brief's "82" for env vars undercounts because it
grepped `runtime/` as a top-level path (doesn't exist — `runtime/` lives at
`python/tokenspeed/runtime/`) and silently skipped it; the correct repo-wide
unique count is **105**, reproducible via the command in the table.

---

## 2. Target layout

```
src/unieinfra/          <- python/tokenspeed/          (first-party engine, package tokenspeed -> unieinfra)
src/unieinfra-kernel/   <- unieinfra-kernel/            (first-party kernels, ALREADY correctly named unieinfra_kernel)
src/kernel/             <- tokenspeed-kernel/           (forked base; package name STAYS tokenspeed_kernel)
src/kernel-amd/         <- tokenspeed-kernel-amd/       (forked base component; stays tokenspeed-named)
src/mla/                <- tokenspeed-mla/              (forked base component; stays tokenspeed-named)
src/scheduler/          <- tokenspeed-scheduler/        (forked base; prebuilt ext, rename deferred — see §3)
bench/                  <- benchmark/
tests/                  <- test/
docs/                   unchanged (already correctly rooted)
tools/                  <- scripts/  **only if** the LD_PRELOAD shim lookup in
                                       python/…/cli/_proc.py is updated in the
                                       SAME commit (see decision below)
docker/                 unchanged
third_party/            unchanged (git submodules)
assets/, .github/, .skills/   unchanged (see decisions below)
```

### Rationale per move

- **`python/tokenspeed/` → `src/unieinfra/`.** Two changes riding together:
  hoisting out of `python/` (the current top-level `python/` directory exists
  only to hold this one package plus its `pyproject.toml`/`LICENSE`/
  `THIRDPARTYNOTICES` — a `src/` layout is the more conventional shape for a
  multi-package monorepo where `python/` was really standing in for "the
  first-party engine package") and the identity rename. Doing both in the
  path is fine *because* the import-rewrite step (§4) already has to touch
  every one of the 272 files' `import`/`from` lines regardless of whether the
  directory also moves — one mechanical pass, not two.
- **`unieinfra-kernel/` → `src/unieinfra-kernel/`.** Pure `git mv`, no rename —
  it already carries the target identity (`unieinfra_kernel` package name).
  Moving it under `src/` groups it with the other first-party code and makes
  the "first-party vs. forked-base" split visible at the top level.
- **`tokenspeed-kernel/` → `src/kernel/`, `tokenspeed-kernel-amd/` →
  `src/kernel-amd/`, `tokenspeed-mla/` → `src/mla/`, `tokenspeed-scheduler/` →
  `src/scheduler/`.** Directory rename only — **the Python package names
  inside stay `tokenspeed_kernel` / `tokenspeed_scheduler` etc.** This is the
  collision-avoidance the task calls out explicitly: `unieinfra-kernel`
  registers *into* the forked base's `tokenspeed_kernel` registry
  (`register_kernel`/`KernelSpec`/`select_kernel`, per
  `docs/design/kernel-plugin-contract.md`). If the forked base were also
  renamed to `unieinfra_kernel`, the two packages could not coexist under
  distinct import names, and worse, `unieinfra-kernel`'s own code and docs
  already say "registers into the forked base's registry (`tokenspeed_kernel`)"
  — changing that identity is a correctness change to a system that was just
  scaffolded (`5cdf889`), not a pure rename. Directory-level rename (dropping
  the `tokenspeed-` folder prefix in favor of grouping under `src/`) is safe
  and reduces visual noise without touching the package boundary.
- **`benchmark/` → `bench/`, `test/` → `tests/`.** Short, conventional names;
  zero functional coupling found to the literal strings `benchmark`/`test` as
  directory names (no hardcoded `"benchmark/"` or `"test/"` path lookups
  analogous to the `scripts/` shim were found in the python tree — grepped).
  Straightforward `git mv`, but every CI workflow, `pytest.ini`/`pyproject.toml`
  testpaths (if any), and the 195 `test/`-referencing files' *relative* fixture
  paths need the rename to ride along.
- **`docs/`, `docker/`, `third_party/`, `assets/`** unchanged — already
  correctly named/rooted, no `tokenspeed` coupling in the directory names
  themselves (only in *contents*, handled by the text-rewrite pass).

### Decisions

- **`scripts/` → `tools/`? Recommend: rename, but only bundled with a code
  fix.** `scripts/` currently holds exactly one thing that matters
  structurally: the TCP_NODELAY LD_PRELOAD shim, located via a **hardcoded
  literal `"scripts"`** in `_repo_scripts_dir()`
  (`python/tokenspeed/cli/_proc.py`, soon `src/unieinfra/cli/_proc.py`). The
  function already computes "3 directories up from `cli/_proc.py`" to find the
  repo root — that depth is **preserved** by the `python/tokenspeed/cli/` →
  `src/unieinfra/cli/` move (still 3 levels: `cli` → `unieinfra` → `src` →
  root), so the move alone is safe. But if `scripts/` is *also* renamed to
  `tools/`, the literal string `"scripts"` inside that function must change
  to `"tools"` **in the same commit**, or the shim silently stops being found
  (the function degrades gracefully — logs a warning, serves without the
  Nagle mitigation — so this fails quiet, not loud, which makes it dangerous
  to leave unsynced). Given that coupling, recommend doing the `scripts/` →
  `tools/` rename in the **same commit** as the `_proc.py` content rewrite
  (commit 2, not commit 1 — see §7), not as a bare `git mv`. If the executor
  wants zero risk, the alternative is simply **not renaming `scripts/`** —
  it's a defensible name on its own and this doc does not treat it as
  required.
- **OSS meta files stay at root.** `README.md`, `LICENSE`, `GOVERNANCE.md`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`,
  `ACKNOWLEDGEMENTS.md`, `NOTICE`, `AGENTS.md`, `CLAUDE.md` — GitHub's own UI
  (community-profile checklist, "Insights → Community Standards") looks for
  several of these specifically at repo root (or `.github/`), so moving them
  would actively regress GitHub-native discoverability for no benefit. No move.
- **`README.tokenspeed.md`**: recommend renaming to
  `README.upstream-tokenspeed.md` (clearer that it's attribution, not a
  competing top-level README) and fixing its 3 referrers (`README.md:15`,
  `NOTICE:12`, `CHANGELOG.md:22`) in the same commit. Low blast radius (3
  files), bundle into commit 3 (docs/config pass) — not required, owner may
  prefer to leave it as-is since `NOTICE` already explains why it exists.
- **`.skills/` does not move.** It is Claude-agent tooling configuration
  (`bisect-triton-release.md`, `optimize-amd-gpu-kernel.md`), not source
  organization — same category as `.claude/`, `.github/`. No `tokenspeed`
  coupling found in its contents beyond ordinary prose.
- **`assets/` filenames**: recommend renaming
  `tokenspeed-banner.png`/`tokenspeed-kimi-k2.5-performance.png` to
  `unieinfra-banner.png`/`unieinfra-kimi-k2.5-performance.png` for identity
  consistency, but since neither is currently referenced from `README.md`
  (verified — zero hits), this is genuinely optional cosmetic cleanup with no
  functional blast radius. Owner's call; not required for migration
  correctness.

---

## 3. The rename boundary

**Renames** (first-party identity, `tokenspeed` → `unieinfra`):

- Python package: `tokenspeed` (1825 imports / 272 files) → `unieinfra`.
  Mechanical, but touches 2 string-literal module paths too (§1) that a naive
  `import`/`from`-only rewrite will miss.
- `pyproject.toml`: `name = "tokenspeed"` → `"unieinfra"`;
  `[tool.setuptools.packages.find] include = ["tokenspeed*"]` →
  `["unieinfra*"]`.
- CLI entry points: `tokenspeed = "tokenspeed.cli:main"` and
  `ts = "tokenspeed.cli:main"` → `unieinfra = "unieinfra.cli:main"` (keep
  `ts` too — it's a short alias people type constantly, no reason to break
  it; point it at the same `unieinfra.cli:main`).
- `TOKENSPEED_*` env vars (105 unique) → `UNIEINFRA_*`, **with a backward-
  compat alias shim** (see below) — this is a public OSS project and 105 is
  a real user-facing surface (deploy scripts, systemd units, Dockerfiles
  people have already written against `TOKENSPEED_KERNEL_BACKEND` etc.).
  Silently breaking all 105 in one release is hostile; the alias shim is not
  optional for a project explicitly positioning itself as OSS with an
  external user base.
- `.isort.cfg` `known_first_party`, `setproctitle()` process names (3),
  `PACKAGE_LIST` literal in `env.py`, `test_env_package_list.py`'s hardcoded
  set, `.github/CODEOWNERS` paths, workflow filenames/content, `docker/`
  install paths, `docs/.vitepress/config.mts` base path + GitHub links,
  `docs/package.json` name, benchmark/test fixture references — all follow
  the package rename mechanically (§4, §7).

**Cannot change** (external, out of this repo's control):

- `tokenspeed-kernel`, `tokenspeed-mooncake` (as given), plus **newly
  identified**: `tokenspeed-smg`, `tokenspeed-smg-grpc-proto`,
  `tokenspeed-smg-grpc-servicer`, `tokenspeed-spdlog` — all external PyPI
  packages with no source in this repo. Two of them (`smg-grpc-proto`,
  `smg-grpc-servicer`) additionally bake the literal string `tokenspeed` into
  *their own* generated Python module paths that our code imports
  (`smg_grpc_proto.generated.tokenspeed_scheduler_pb2`,
  `smg_grpc_servicer.tokenspeed`) — renaming those requires new upstream
  releases, not a decision this repo can make unilaterally.
- `tokenspeed_scheduler` — first-party C++ source (`tokenspeed-scheduler/`,
  → `src/scheduler/`), but a **prebuilt binary**. The Python-visible module
  name is hardcoded in C++ (`NB_MODULE(tokenspeed_scheduler_ext, m)`), and
  this checkout's actual `.so` was built without `TOKENSPEED_FLAT_KVCACHE`
  (`FLAT_KVCACHE=False`, confirmed via `paged_cache_spec.py`'s runtime
  accessor). Renaming this properly means: rewrite the `NB_MODULE` macro,
  update `tokenspeed-scheduler/pyproject.toml`'s `wheel.packages`, **rebuild
  the extension** (CMake + nanobind, needs a full C++ toolchain — the
  existing `build/` artifacts would need to be regenerated, not just moved),
  then update the ~13 first-party importers. That is a real engineering task
  with a build step, not a text edit. **Recommend deferring it** to a
  follow-up after this reorg lands — the directory can move to
  `src/scheduler/` now (pure `git mv`, no rebuild needed) while the Python
  module name and PyPI distribution name (`tokenspeed-scheduler`) stay as-is
  until someone schedules the rebuild.
- Forked-base kernel packages (`tokenspeed_kernel`, and by the same logic
  anything under `src/kernel-amd/`, `src/mla/` that still imports
  `tokenspeed_kernel`): **deliberately not renamed**, to avoid the
  `unieinfra-kernel` naming collision described in §2.

**Honest bottom line**: "no trace of `tokenspeed` in the repo" is **not
achievable**, and that's the right outcome, not a shortfall. The forked-base
kernel/scheduler components carry real provenance (they *are* the TokenSpeed
fork, per `NOTICE`) and keep their name; several load-bearing PyPI
dependencies are external and simply not ours to rename. What *is* achievable
and *is* the point of this migration: every piece of code and identity that
originates as **UnieInfra's own** — the engine package, its CLI, its
public-facing env-var contract, its own new kernel plugin package — carries
the `unieinfra` name consistently, and the boundary between "this is ours"
and "this is the fork we sit on" is explicit rather than accidental.

### Backward-compat env var shim — recommended, one release

Add a small shim (e.g. `src/unieinfra/_env_compat.py`, imported early from
`env.py` or the package `__init__.py`) that, for every `UNIEINFRA_*` name the
engine reads, also checks the corresponding `TOKENSPEED_*` name if the new one
is unset, and emits a `DeprecationWarning` pointing at the new name. Concretely:

```python
import os
import warnings

_ALIASES = {  # UNIEINFRA_NAME -> old TOKENSPEED_NAME, generated from the 105-entry list
    "UNIEINFRA_KERNEL_BACKEND": "TOKENSPEED_KERNEL_BACKEND",
    # ... all 105
}

def apply_env_compat_aliases() -> None:
    for new, old in _ALIASES.items():
        if new not in os.environ and old in os.environ:
            warnings.warn(
                f"{old} is deprecated, use {new} instead; will be removed in a "
                "future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            os.environ[new] = os.environ[old]
```

Generate `_ALIASES` mechanically from the 105-name grep output (§1) at
migration time, don't hand-type it. Plan to remove the shim after one
deprecation cycle (pick a version/date and put it in `CHANGELOG.md` when the
shim lands, per the repo's "OSS docs + changelog discipline" convention).

A `tokenspeed` alias *Python package* (re-exporting `unieinfra` with a
`DeprecationWarning` on import) is a separate, smaller decision — recommend
**not** doing it: nothing publishes `tokenspeed` as an installable
first-party artifact today outside this monorepo (no evidence of external
`pip install tokenspeed` consumers — the PyPI name `tokenspeed` belongs to
the upstream project this is forked from, not to us, so shipping a package
under that name would be actively confusing). The env-var shim covers the
real backward-compat surface (scripts/configs people run); the Python import
path is internal to this repo and gets rewritten everywhere in the same
commit, so there's no external consumer of `import tokenspeed` to protect.

---

## 4. Step-by-step atomic execution procedure

Ordered so the tree is never broken for long between commits (see §7 for
commit boundaries — this section is the mechanics inside each).

### 4.1 — `git mv` moves (preserve history)

Use `git mv` (not delete+recreate) so `git log --follow` still works. Order
matters only in that the destination parent (`src/`) must exist first:

```bash
mkdir -p src
git mv python/tokenspeed src/unieinfra
git mv python/pyproject.toml src/pyproject.toml   # or keep pyproject at repo root — see note
git mv python/LICENSE src/LICENSE
git mv python/THIRDPARTYNOTICES src/THIRDPARTYNOTICES
git mv unieinfra-kernel src/unieinfra-kernel
git mv tokenspeed-kernel src/kernel
git mv tokenspeed-kernel-amd src/kernel-amd
git mv tokenspeed-mla src/mla
git mv tokenspeed-scheduler src/scheduler
git mv benchmark bench
git mv test tests
# scripts -> tools: only if bundled with the _proc.py literal-string fix (§2 decision); do it in 4.2, not here, so this commit is a pure move.
rmdir python   # now empty
```

Note on `src/pyproject.toml`: verify whether any tooling (editable installs,
CI `pip install -e python/`) assumes `pyproject.toml` lives at `python/`
specifically — if so, either keep a redirect or update those call sites in
the same pass. This needs a quick `grep -rn "pip install.*python/" .github/
docker/` at execution time (not re-derived here since it's a one-line check
the executor should redo fresh right before running, in case anything shifted).

**Verification after 4.1 alone**: `git status` shows renames (`R`), not
delete+add pairs, for every moved file — confirms history was preserved.
`git log --follow -- src/unieinfra/env.py` should still show pre-move commits.

### 4.2 — Mechanical import + reference rewrite

This is the 1825-statement / 272-file pass, plus the 2 string-literal module
paths, plus every non-Python reference. Do NOT rely on a single blind
`sed s/tokenspeed/unieinfra/g` — it will over-match (e.g. `tokenspeed_kernel`,
`tokenspeed-scheduler`, `tokenspeed_scheduler_ext`, `lightseekorg/tokenspeed`
GitHub URLs, `README.tokenspeed.md`, all of which must NOT become
`unieinfra_kernel`/`unieinfra-scheduler`/etc.). Use a **word-boundary-aware,
allowlisted** pass instead:

1. **Import statements** (`import tokenspeed`, `import tokenspeed.x`,
   `import tokenspeed.x as y`, `from tokenspeed import z`,
   `from tokenspeed.x import z`, `from .x import y` relative imports that
   don't need touching, `from ..tokenspeed_something import z` — n/a, no such
   pattern found):
   ```bash
   git grep -lE '^\s*(import|from)\s+tokenspeed(\.|(\s|$))' -- 'src/*.py' 'tests/*.py' 'bench/*.py' \
     | xargs sed -i -E \
       -e 's/^(\s*)import tokenspeed(\.[A-Za-z0-9_.]*)?( as [A-Za-z_][A-Za-z0-9_]*)?/\1import unieinfra\2\3/' \
       -e 's/^(\s*)from tokenspeed(\.[A-Za-z0-9_.]*)? import/\1from unieinfra\2 import/'
   ```
   Critically, this pattern is anchored on `tokenspeed` as a **top-level**
   import root (`tokenspeed` or `tokenspeed.xxx`), which naturally excludes
   `tokenspeed_kernel`/`tokenspeed_scheduler` (different identifier, `_`
   not `.`, doesn't match `tokenspeed(\.|(\s|$))`). Verify this explicitly
   post-hoc: `git grep -E '^\s*(import|from)\s+tokenspeed_' src/ tests/ bench/`
   must return **zero** matches changed (they should all still say
   `tokenspeed_kernel`/`tokenspeed_scheduler`, untouched).
2. **String-literal module paths** (the 2 found in §1, and re-grep at
   execution time in case more exist by then):
   ```bash
   git grep -n '"tokenspeed\.' -- 'src/*.py' 'tests/*.py'
   git grep -n "'tokenspeed\." -- 'src/*.py' 'tests/*.py'
   ```
   Fix each by hand (only 2 known instances) — not worth a regex given the
   low count and the risk of a regex also mangling
   `"tokenspeed_scheduler extension"`-style prose strings that must NOT
   change (`paged_cache_spec.py` has several — these are human-readable error
   messages *about* the external `tokenspeed_scheduler` package and are
   correctly left alone).
3. **`__module__`/pickled/entry-point references**: grepped for
   `importlib.import_module\(.*tokenspeed` and `pkgutil` walks — the only
   dynamic-discovery site found is `registry.py`'s `package_name =
   "tokenspeed.runtime.models"` (covered by item 2). No evidence of pickled
   objects whose `__module__` would embed `tokenspeed.*` (this is a serving
   engine, not a long-lived pickle-based cache across versions, per the repo's
   architecture) — but the executor should grep
   `git grep -n "__module__" src/unieinfra` fresh at execution time as a
   final check, since this is exactly the kind of thing that fails silently
   months later.
4. **`TOKENSPEED_*` → `UNIEINFRA_*` env var names**: word-boundary-anchored,
   whole-identifier replace, restricted to the 105-name allowlist derived in
   §1 (do not blind-`sed` `TOKENSPEED` as a substring — it's fine here since
   all 105 hits are already full `TOKENSPEED_XXX` tokens with no partial
   overlaps observed, but generate the allowlist file first and diff before
   committing):
   ```bash
   grep -rhoE 'TOKENSPEED_[A-Z0-9_]+' . --exclude-dir={third_party,.git,build} | sort -u > /tmp/env_names.txt
   while read -r name; do
     new="UNIEINFRA_${name#TOKENSPEED_}"
     git grep -lFw "$name" -- ':!third_party' ':!*.so' | xargs sed -i "s/\b${name}\b/${new}/g"
   done < /tmp/env_names.txt
   ```
   Then wire the compat shim (§3) so `TOKENSPEED_*` still works read-only for
   one release — the shim reads the *old* names from `os.environ`, so this
   step is what makes the *code* look for the *new* names; the shim bridges
   the gap.
5. **`setproctitle()` calls**: 3 sites, hand-edit
   (`"ts-serve"` stays as-is — it's not tokenspeed-branded;
   `"tokenspeed::data_parallel_controller"` →
   `"unieinfra::data_parallel_controller"`;
   `"tokenspeed::scheduler_{dp_rank}"` → `"unieinfra::scheduler_{dp_rank}"`).
6. **`PACKAGE_LIST`** in `env.py`: `"tokenspeed"` → `"unieinfra"` (must match
   whatever the new PyPI/importlib.metadata distribution name ends up being
   — keep in lockstep with the `pyproject.toml` `name` field, step 4.3).
7. **`scripts/` → `tools/`** (if the owner takes that option): update the
   literal `"scripts"` in `_repo_scripts_dir()` (now
   `src/unieinfra/cli/_proc.py`) to `"tools"` in this same step, plus
   `git mv scripts tools`.

**Verification after 4.2**: `python -c "import ast; ast.parse(open(f).read())"`
over every touched `.py` file (syntax didn't break); then the full "zero
stale references" grep from §5.

### 4.3 — `pyproject.toml` edits

In `src/pyproject.toml` (or wherever it ends up per the 4.1 note):
- `name = "tokenspeed"` → `name = "unieinfra"`
- `[project.scripts]`: `tokenspeed = "tokenspeed.cli:main"` →
  `unieinfra = "unieinfra.cli:main"`; keep `ts = "unieinfra.cli:main"`.
- `[tool.setuptools.packages.find] include = ["tokenspeed*"]` →
  `["unieinfra*"]`.
- Dependency lines `tokenspeed-kernel>=...`, `tokenspeed-mooncake>=...`,
  `tokenspeed-scheduler>=...`, `tokenspeed-smg*`: **unchanged** (external, or
  deferred per §3).
- `[project.urls] Homepage`: update once the GitHub org/repo for UnieInfra is
  confirmed (out of scope for this doc to guess — flag as a TODO for the
  owner, don't invent a URL).

### 4.4 — Config file edits

- `.isort.cfg`: `known_first_party=tokenspeed` → `known_first_party=unieinfra`.
- `.github/CODEOWNERS`: rewrite all 4 path rules to the `src/...` destinations
  (`/src/unieinfra/runtime`, `/src/kernel`, `/src/kernel-amd`, `/src/mla`,
  `/src/scheduler`).
- `.github/workflows/*.yml`: update path references (`working-directory`,
  `paths:` triggers, artifact upload paths) for every moved directory;
  rename the 11 `tokenspeed`-named workflow files themselves
  (`release-tokenspeed.yml` → `release-unieinfra.yml`, etc. — but only for
  workflows that build **first-party** artifacts; workflows that
  build/release the still-`tokenspeed`-named forked-base packages, e.g.
  `release-tokenspeed-kernel.yml`, should probably keep their names since
  they still produce `tokenspeed-kernel`-named PyPI artifacts — **owner
  decision**, flagged here rather than assumed).
- `docker/Dockerfile.{nvidia,amd,release}`: update the in-repo path arguments
  (`pip install tokenspeed-kernel/python/` → `pip install src/kernel/python/`,
  etc.); leave the external base image tags
  (`lightseekorg/tokenspeed-runner*`) untouched — those are upstream images,
  not this repo's artifacts.
- `docs/.vitepress/config.mts`: `base: "/tokenspeed/"` → `base: "/unieinfra/"`
  (coordinate with wherever docs actually get hosted — a base-path change
  breaks all existing bookmarked/indexed doc URLs, flag for the owner to
  confirm timing), GitHub links `lightseekorg/tokenspeed` → new org/repo once
  known.
- `docs/package.json`: `"name": "tokenspeed-docs"` → `"name": "unieinfra-docs"`.
- `test_env_package_list.py`: update the hardcoded `{"tokenspeed", ...}` set
  to `{"unieinfra", ...}` **in the same commit** as the `PACKAGE_LIST`/
  `pyproject.toml` name change — this test is a tripwire that will fail
  immediately if these two drift, which is good, but only if both sides are
  edited together.

### 4.5 — Compat shim

Land the env-var alias shim from §3 (`apply_env_compat_aliases()`, called
early — e.g. top of `src/unieinfra/env.py` or package `__init__.py`, before
any `TOKENSPEED_*`/`UNIEINFRA_*` name is read anywhere else). Generate the
105-entry `_ALIASES` dict from the same grep used in §4.2 step 4, don't
hand-type it — the two must be derived from the same source list or they'll
drift.

---

## 5. Verification checklist

Run all of these after execution (none require a GPU except where noted as
optional):

1. `python -c "import unieinfra"` — succeeds, no `ModuleNotFoundError`, no
   leftover `import tokenspeed` anywhere in the import graph it touches.
2. Full test suite: `pytest tests/` (post-rename path) — or `python -m pytest`
   from repo root with whatever `testpaths`/`conftest.py` discovery the repo
   uses; confirm `tests/test_env_package_list.py` passes (it's the tripwire
   for `PACKAGE_LIST`/`pyproject.toml` sync).
3. `pre-commit run --all-files` — per `AGENTS.md`, run the **exact** command,
   not a narrower lint substitute; commit any formatter changes it makes as
   part of the migration commit(s), not separately.
4. Engine smoke test: `unieinfra serve <small-model> --port <n>` (or `ts serve
   ...`), confirm the process starts, then one `curl`/`openai` client
   generation request round-trips. This needs a model + (optionally) a GPU to
   be *meaningful*, but the **import and CLI-wiring** part of this check
   (does `unieinfra serve --help` work, does argument parsing reach
   `run_smg_from_args`) can and should be done CPU-only first, per the "NO
   GPU usage" constraint on this task — full generation smoke can be a
   GPU-available follow-up, not a blocker for the migration commit itself.
5. Benchmark harness dry-run: `bench/trajectory.py` (post-`benchmark/`→`bench/`
   rename) and `bench/bench_and_profile.sh --dry-run` if such a flag exists
   (verify at execution time — not confirmed to exist in the current script,
   check before relying on it) — confirms path rewrites in the benchmark
   scripts didn't break argument parsing / config loading.
6. Submodule integrity: `git submodule status` — all 9 submodules
   (`third_party/*`) still report clean, unmoved SHAs (this migration must
   not touch `third_party/` at all).
7. Zero stale first-party references:
   ```bash
   git grep -c 'import tokenspeed\|from tokenspeed' -- 'src/*.py' 'tests/*.py' 'bench/*.py'
   ```
   must return **no output** (zero matches) — note the pattern deliberately
   excludes `tokenspeed_kernel`/`tokenspeed_scheduler`/`tokenspeed-mooncake`
   references, which are expected to remain (§3). A second check,
   ```bash
   git grep -c 'TOKENSPEED_' -- 'src/*.py' | grep -v _env_compat.py
   ```
   should return zero outside the compat-shim file itself.
8. `git log --follow -- src/unieinfra/env.py` (or any moved file) still shows
   pre-migration history — confirms `git mv` was used correctly in §4.1, not
   delete+recreate.

---

## 6. Risk register

| Risk | What breaks | Mitigation | Rollback |
|---|---|---|---|
| `tokenspeed_scheduler` prebuilt `.so` left un-rebuilt while everything around it renames | Import errors if anyone assumes the module name also changed; or silent staleness if the `.so` in `build/` is from before the directory move and CI doesn't regenerate it | Explicitly document (this doc, §3) that the module name and PyPI dist name (`tokenspeed-scheduler`) are **not** part of this migration; keep the `src/scheduler/` directory's *contents* (pyproject `name`, `NB_MODULE` macro) untouched; CI's existing `scheduler-cpp-test.yml`/`scheduler-python-test.yml` should still build correctly if `working-directory` paths are updated to `src/scheduler/` | Single revert commit for the `git mv` if CI red; the C++/build layer isn't touched so there's nothing there to roll back |
| `LD_PRELOAD` TCP_NODELAY shim path (`_repo_scripts_dir()`) breaks silently if `scripts/`→`tools/` isn't paired with the code edit | No crash, no test failure — just a quiet loss of the Nagle/TTFT mitigation in production; the kind of regression that shows up as "TTFT got worse" weeks later with no obvious cause | Bundle the rename and the `_proc.py` literal-string fix in the same commit (§2, §4.2 step 7); add a smoke check post-migration: start the gateway and confirm `LD_PRELOAD` is set in its env (`ps eww <pid> \| grep LD_PRELOAD`) | Revert the `scripts/`→`tools/` half only (keep the rest of the migration) if this is caught late |
| `TOKENSPEED_*` env var contract broken for existing users/scripts | Anyone with `TOKENSPEED_KERNEL_BACKEND=... unieinfra serve ...` in a deploy script silently gets default behavior instead of their intended override | Compat shim (§3, §4.5) — old names still work for one release with a `DeprecationWarning`; document removal timeline in `CHANGELOG.md` | Re-add the shim if it's found missing post-release; low risk since it's additive |
| `.github/CODEOWNERS` paths go stale | Review-assignment silently stops applying to the moved directories — no error, PRs just don't get auto-assigned reviewers | Update all 4 rules in the same commit as the `git mv` (§4.1/§4.4); verify with GitHub's CODEOWNERS UI (repo Settings → shows unmatched rules) post-push | Fix-forward commit; low blast radius |
| CI workflow `paths:`/`working-directory` triggers stop matching | Workflows silently stop triggering on changes to the renamed directories (e.g. `pr-test-nvidia.yml` no longer runs when `src/unieinfra/` changes) — worse than a hard failure because CI goes quiet instead of red | Grep every workflow file for path references as part of commit 3 (§7); after pushing, open a throwaway PR touching `src/unieinfra/` and confirm the expected workflows actually fire | Revert commit 3 (config/CI-only, no source impact) if workflows misfire |
| External `tokenspeed-smg*` packages' internal module names (`tokenspeed_scheduler_pb2`, `smg_grpc_servicer.tokenspeed`) mistaken for renameable | A well-intentioned blind `sed` pass renames the import site (`from smg_grpc_proto.generated import tokenspeed_scheduler_pb2`) to reference a module that doesn't exist in the installed external package → `ImportError` at runtime, likely only hit when the control-plane gRPC path is exercised (not necessarily caught by a quick smoke test) | The word-boundary-anchored rewrite in §4.2 is scoped to `import tokenspeed`/`from tokenspeed` as a **top-level package root**, which does not match `smg_grpc_proto.generated.tokenspeed_scheduler_pb2` (different root package); still, explicitly grep for `tokenspeed_scheduler_pb2` and `smg_grpc_servicer.tokenspeed` post-rewrite and confirm both are byte-for-byte unchanged | Targeted fix-forward on `control_server.py`/`cli/_proc.py` only |
| In-flight branches/PRs against the old `python/tokenspeed/...` paths | Every open PR gets a massive, unmergeable diff / conflict storm the moment this lands on the base branch | Coordinate timing: land this migration when the tree is quiet (§0 precondition), and communicate to any PR authors to rebase immediately after; consider a short PR-freeze window | Not really "rollback"-able after the fact — this is a coordination problem, not a code problem; the mitigation is entirely about *when* to execute, hence §7's sequencing emphasis |
| Docker image build breaks | `Dockerfile.{nvidia,amd,release}` reference `tokenspeed-kernel/python/`, `tokenspeed-scheduler/`, `./tokenspeed-kernel-amd` as literal `pip install` paths — if these aren't updated to `src/kernel/python/` etc., the image build fails outright (loud, not silent — good) | Update in commit 3; test with an actual `docker build` (CPU-only, no GPU needed to validate the build stage itself — GPU only needed to *run* the resulting image) before merging | Revert commit 3 |
| Benchmark launch scripts' `results/*.launch.json` output | `benchmark/launch/launch_unieinfra.sh` already writes `"engine":"unieinfra"` — low risk here specifically, but the sibling `launch_vllm.sh` and any downstream `bench/summarize.py` parsing of `results/*.launch.json` by filename pattern must still find files at the post-rename `bench/results/` path | Verify `bench/summarize.py` (post-rename) globs `results/` relative to its own location, not a hardcoded absolute/parent-relative string tied to the old `benchmark/` name — grep at execution time | Fix-forward, contained to `bench/` |
| Test fixture directories (`test/*/tokenspeed/`) renamed without their referencing CI YAML | `test/ci/eval/*.yml`, `test/ci/perf/*.yml`, `test/agentic_benchmark/*/agentic_bench.sh` likely hardcode `.../tokenspeed/configs/...`-style paths; if the fixture dir renames to `unieinfra/` but these don't follow, CI eval/perf jobs fail to find configs | Grep `test/ci/**/*.yml` and `test/agentic_benchmark/**/*.sh` for the literal substring `/tokenspeed/` as part of commit 2's scope, not commit 1's bare `git mv` | Fix-forward; contained to `tests/` |

---

## 7. Recommended sequencing

**Precondition** (repeated from §0 for emphasis): quiet tree, no running
subagents editing `python/`/`tokenspeed-*/`/`unieinfra-kernel/`/`test/`/
`benchmark/`, everything currently in flight committed first. This migration
touches essentially the entire first-party surface — it is actively hostile
to run concurrently with any other source edit.

**Commit 1 — pure `git mv` moves, no content changes.**
Everything in §4.1. `git mv python/tokenspeed src/unieinfra`,
`git mv tokenspeed-kernel src/kernel`, etc., plus `benchmark`→`bench`,
`test`→`tests`. Nothing inside any file changes. This commit should show 100%
renames in `git status`/`git diff --stat`, zero content diffs. Purpose:
isolate "did the move preserve history correctly" from "did the rename logic
work," so if something goes wrong later it's easy to bisect which half broke
it. Also the safest commit to revert in isolation if the *content* rewrite
(commit 2) turns out to need rework — the tree is still 100% functional after
commit 1 alone (everything still says `tokenspeed` internally, just living at
new paths — actually **not** true if any code did path-relative assumptions
like `_repo_scripts_dir()`'s depth math; verify that specific invariant holds
before treating commit 1 as "safe to pause on/wait to run tests").

**Commit 2 — mechanical rename** (§4.2, §4.5). The 1825-import rewrite, the 2
string-literal module paths, the 105 env vars, `setproctitle()` calls,
`PACKAGE_LIST`, the compat shim, and (if chosen) the `scripts/`→`tools/`
pairing. This is the commit where `pytest tests/` should start passing again
(it will be broken between commit 1 and commit 2, since paths moved but
imports still say `tokenspeed`).

**Commit 3 — config/docs/CI pass** (§4.3, §4.4). `pyproject.toml`,
`.isort.cfg`, `.github/CODEOWNERS`, workflow files (path updates + optional
filename renames), `docker/Dockerfile.*`, `docs/.vitepress/config.mts`,
`docs/package.json`, `test_env_package_list.py`, `README.tokenspeed.md` (if
renamed), `assets/` filenames (if renamed). Run `pre-commit run --all-files`
at the end of this commit and fold in any formatter changes — per `AGENTS.md`,
this is mandatory before any commit, not just the last one, so realistically
run it after each of the three commits, not only at the end.

Do not squash these three into one commit: if CI or a smoke test catches a
problem, `git bisect`/`git revert` against three well-scoped commits is far
cheaper than unpicking one 2000+-file diff. Do not split them *further* than
this either — e.g. don't do the 105 env vars as a separate commit from the
1825 imports — because commit 2 is only internally consistent as a unit (a
partial rewrite where some imports say `unieinfra` and others still say
`tokenspeed` is a broken intermediate state that shouldn't be pushed).
