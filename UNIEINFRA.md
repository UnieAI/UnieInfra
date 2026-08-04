# UnieInfra

UnieAI's inference stack. This repository **is** upstream
[tokenspeed](https://github.com/lightseekorg/tokenspeed) plus a small amount of
UnieAI-specific work kept strictly out of the upstream tree.

New file at the repository root by design: adding a file never conflicts on
rebase, whereas editing upstream's `README.md` would conflict on every one.

## The one rule

**Never edit a file that came from upstream.**

Upstream lands roughly 2-3 commits a day across `tokenspeed-kernel/` and
`python/tokenspeed/`. Every upstream file this repository edits becomes a
conflict that has to be re-resolved on every rebase, forever. The previous
arrangement had no shared git ancestor with upstream at all, so nothing could be
rebased and the divergence was invisible until someone diffed two working trees
by hand — by then it was 3381 lines across 33 files in `tokenspeed-kernel` alone.

UnieAI work therefore has exactly two legitimate homes:

1. **`unieinfra/` — out-of-tree plugins.** Registered through
   `tokenspeed_kernel`'s entry-point contract, so upstream never has to know
   they exist. `unieinfra/kernel-sm120/` is the working example: its
   `pyproject.toml` declares

   ```toml
   [project.entry-points."tokenspeed_kernel.plugins"]
   unieinfra_sm120 = "unieinfra_kernel_sm120:register"
   ```

   and `register()` calls `tokenspeed_kernel.register_kernel(...)` in the
   `PLUGIN` priority band, which outranks upstream's own kernels without
   touching them.

2. **A pull request to upstream.** Anything that is not UnieAI-specific — an
   architecture gate that omits `sm_120`, a kernel that is admissible but
   broken, a build break — belongs upstream. Carrying it here means paying for
   it on every rebase and never getting it reviewed.

If something genuinely cannot be expressed either way, that is worth a
discussion before it is written, not a patch to an upstream file.

## Layout

| path | what |
|---|---|
| `python/`, `tokenspeed-kernel/`, `tokenspeed-scheduler/`, … | upstream tokenspeed, unmodified |
| `unieinfra/` | UnieAI plugins and tooling |
| `legacy/` | everything from before this restructure — see `legacy/README.md` |

## Staying current

```sh
git fetch upstream
git rebase upstream/main
```

`upstream` is `lightseekorg/tokenspeed`; `origin` is `UnieAI/UnieInfra`. Because
this repository descends from upstream's history, a rebase only stops on files
UnieAI actually changed — and if the rule above is being followed, that is
`unieinfra/`, `legacy/`, and this file, none of which upstream touches.

`local-tokenspeed` and `legacy-origin` are convenience remotes pointing at local
checkouts (a fast mirror, and the pre-restructure repository whose 76 commits
remain reachable as `legacy-origin/unieinfra-base`).

## Submodules

The maintained tree has none. `legacy/third_party/` records eleven read-only
reference upstreams that kernels were ported *from*, pinned to the exact commits
that were surveyed. Nothing imports them at runtime.

```sh
git submodule update --init legacy/third_party/lmdeploy   # only if you need one
```
