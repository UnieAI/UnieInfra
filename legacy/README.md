# legacy/

Everything that was in UnieInfra before it was re-forked from upstream
tokenspeed with a shared git ancestor. Nothing here is on the maintained code
path; it is kept because most of it was never committed anywhere else.

## Why this exists

The previous repository was forked from tokenspeed by **copying the tree**, not
by cloning it — its first commit is `ff37a93 Fork tokenspeed as UnieInfra
first-party base`, with no ancestor in common with upstream's 618. That made
`git rebase` impossible and divergence invisible: it could only be found by
diffing two working trees by hand.

It also meant most of the work was never tracked. At the time of the move,
`experiment/` held 15,773 files on disk and 1,335 in git. **The contents here
were recovered from the working tree, not from history** — the old repository's
own commits (preserved as `legacy-origin/unieinfra-base`) do not contain them.

## What is here

| path | notes |
|---|---|
| `experiment/` | experiment logs and profiles. `experiment/kda-sm120/docs/log.md` is the sm_120 record, E1-E47. |
| `experiment/legacy/` | the stale tokenspeed fork. The old root had `tokenspeed-kernel` and `tokenspeed-scheduler` as **symlinks** into here. |
| `benchmark/` | benchmark scripts, profiles, `.ncu-rep` traces |
| `divergence/` | what UnieAI actually changed in the fork — see below |
| `patches/` | changes to third-party sources, as patches rather than as a fork |
| `unieinfra-kernel/`, `unieinfra-scheduler/` | vLLM-era plugins, superseded by the tokenspeed plugin contract |
| `docs/`, `assets/`, `scripts/`, `docker/` | as they were |
| `third_party/` | eleven reference upstreams, as pinned submodules |
| `README-unieinfra-pre-restructure.md`, `AGENTS.md`, `CLAUDE.md`, `CHANGELOG.md` | the old root documents |

Triton compile caches (`.cache/`, ~12,000 generated `.ptx`/`.cubin`/`.ttir`
files, 472 MB) and build products (`*.so`, `*.a`, `*.o`, `__pycache__`) were
dropped. Everything else was copied verbatim.

## divergence/ — the actionable part

The fork point is upstream `50f59ea9` (2026-07-24), identified by finding the
upstream commit whose tree is closest to the fork commit's; the fork already
carried ~1805 lines of changes at that point, so no upstream tree matches
exactly. Diffing against it separates UnieAI's work from upstream drift:

| | UnieAI actually changed | naive diff vs upstream HEAD |
|---|---|---|
| `tokenspeed-kernel` | **3381 lines, 33 files** | 6051 lines, 165 files |
| `tokenspeed-scheduler` | **0 lines, 0 files** | 32 files |
| `python/` (runtime) | **3003 lines, 21 files** | — |

`tokenspeed-scheduler` was never modified — every difference there was upstream
moving on. That is the whole argument for having a shared ancestor: without one,
most of the apparent divergence was an illusion, and the part that was real
could not be told apart from it.

The 54 files in `unieai-*.diff` are the re-expression backlog. Each needs to
become either a plugin under `unieinfra/` or a pull request to upstream; none
should come back as an edit to an upstream file. Some are already obsolete —
upstream added `TraitRange`/`TraitConstraint` to `registry.py` for exactly the
kernel shape-constraint problem this fork was working around.

## patches/

`turbomind-w4a16-sm120.patch` — w4a16 GEMM for sm_120 in TurboMind
(`third_party/lmdeploy`). Verified on RTX PRO 6000 Blackwell: 2 to 222
registered kernels, 28/28 numerics, 2.3-3.3x on decode GEMMs. Recorded as a
patch rather than as a submodule pointer because the commit exists only locally
— a gitlink to it would be unfetchable for anyone else. Apply with `git am`
inside the submodule. Full write-up in `experiment/kda-sm120/docs/log.md`, E46
and E47.
