# UnieInfer

UnieAI's inference stack. This repository **is** upstream
[tokenspeed](https://github.com/lightseekorg/tokenspeed) plus a small amount of
UnieAI-specific work kept strictly out of the upstream tree.

## What UnieInfer optimises for

Upstream builds an LLM inference engine for agentic workloads. **UnieInfer
targets the agent itself**: a serving system whose unit of work is an agent
turn — a fixed system prompt, a set of tool descriptions, and a conversation
that grows — rather than an independent prompt.

That is a different optimisation target, not a slogan, and it changes which
number matters:

- **The prefix is the workload.** A system prompt plus tool schemas is
  identical across every request, sits at token 0, and never varies between
  sessions. Retrieval-augmented traffic only shares part of its prompt and
  changes it per query; an agent shares all of its preamble, always. Multi-turn
  compounds it — turn N reuses the whole of turns 1..N-1.
- **On a hybrid model that means recurrent state, not just KV.** Resuming a
  prefix needs the recurrent snapshot at that depth, so a fixed preamble is
  written once and read for the life of the deployment. vLLM does have hybrid
  prefix caching, but only its aligned form: on Qwen3.5/3.6 it falls back to
  `--mamba-cache-mode=align` with a warning and keeps caching on, while the
  `all` mode — full recurrent-state snapshotting — raises `NotImplementedError`
  for `qwen3_5`, `qwen3_next` and `lfm2` and exists only for Mamba1/2
  (`vllm/model_executor/models/qwen3_5.py:297`). The differentiator is
  therefore "no full state snapshot on GDN", not "no hybrid prefix cache".
- **Goodput under a latency SLO, not peak throughput.** An agent turn that
  misses its budget is worth nothing to the caller, so an arm that posts tokens
  per second while breaching TTFT is not a result. See
  `unieinfer/BENCH-qwen35-27b.md` for how that is scored here and for the case
  where the two metrics disagree completely.

Measured on Qwen3.5-27B, one RTX PRO 6000, shared-prefix traffic at concurrency
128: 2.30x vLLM 0.26 on throughput with prefix reuse at 85.8% against its 40.2%.
That advantage is a property of this shape. On prompts with no shared prefix the
two engines are within 3% up to concurrency 64 — which is the honest statement
of where the work pays and where it does not.

The agent-shaped corpora in `bench-data/` (`bfcl_256.jsonl`,
`structured_toolcall.jsonl`, `mtbench_multiturn.jsonl`) are the workloads this
positioning implies and are **not yet benchmarked**; the report lists
long-output and multi-turn shapes as unmeasured. Closing that is the next
measurement, not a claim already banked.

New file at the repository root by design: adding a file never conflicts on
rebase, whereas editing upstream's `README.md` would conflict on every one.

## The rule

**Prefer a plugin. When you must edit an upstream file, keep the edit small,
single-purpose, and exportable as a patch.**

Upstream lands roughly 2-3 commits a day across `tokenspeed-kernel/` and
`python/tokenspeed/`. Every upstream line this repository changes is a line that
can conflict on rebase. That is affordable at the current size and was not
affordable before — the previous arrangement had **no shared git ancestor** with
upstream, so nothing could be rebased and the divergence was invisible until
someone diffed two working trees by hand. By then it was 3381 lines across 33
files in `tokenspeed-kernel` alone, tangled with four months of upstream drift
that no tool could tell apart from it.

What keeps that from happening again is not abstinence, it is visibility:

```sh
git log --oneline upstream/main..main            # every UnieAI commit
git diff --stat upstream/main..main -- python tokenspeed-kernel   # every upstream line touched
```

Both of those were impossible before and are one command now. Check them; if the
second one is growing without anyone deciding it should, that is the signal.

### Order of preference

1. **`unieinfer/` — out-of-tree plugin.** Registered through
   `tokenspeed_kernel`'s entry-point contract, so upstream never has to know it
   exists and there is nothing to rebase. `unieinfer/kernel-sm120/` is the
   working example: its `pyproject.toml` declares

   ```toml
   [project.entry-points."tokenspeed_kernel.plugins"]
   unieinfra_sm120 = "unieinfra_kernel_sm120:register"
   ```

   and `register()` calls `tokenspeed_kernel.register_kernel(...)` in the
   `PLUGIN` priority band, which outranks upstream's own kernels without
   touching them.

2. **An edit to an upstream file, as one self-contained commit.** Allowed, and
   sometimes the only option — a plugin cannot fix a bug inside an upstream
   kernel. Requirements, so the commit stays cheap to carry and cheap to send:

   - one concern per commit, with the reasoning in the message, so a rebase
     conflict can be resolved by someone who was not there;
   - a test, so the next upstream change that breaks it says so;
   - no drive-by reformatting — every unrelated line is a future conflict for
     nothing.

   Such a commit is a PR that has not been sent yet. `git format-patch
   upstream/main..main -- <paths>` exports it whenever that becomes worthwhile.

3. **A pull request to upstream**, for anything not UnieAI-specific. Not a
   prerequisite for shipping — carry it here and send it when convenient — but
   the only way to stop paying for it. `unieinfer/UPSTREAMING.md` tracks what is
   worth sending and why.

The failure mode to watch for is not "we edited an upstream file". It is an
upstream edit that nobody wrote down, so nobody knows whether it is a local
workaround, a bug fix that belongs upstream, or something upstream has since
fixed on its own.

## Process for optimization work

Follow in order. `AGENTS.local.md` carries the same list so it is loaded into
every session; this copy is the one that is shared and versioned.

1. **Decide what to do.**
2. **Survey how every inference engine implements it** — vLLM, TRT-LLM, SGLang,
   TurboMind, tileRT. All of them, not one: they disagree, and the disagreements
   are the finding. On GDN mixed batching, vLLM and TurboMind split decode-first
   while TRT-LLM and SGLang split prefill-first, which is how the order turned
   out to be a choice rather than a requirement; TurboMind alone passes offsets
   instead of materialising slices, removing a whole class of alignment bug.
   Reading one engine surfaces none of that, and skipping the step has cost this
   project five debug cycles on one feature (log E54).
3. **Port and test. Passing is +10% against the other frameworks**, on the full
   post-warmup mean, arms repeated and run in both orders. Parity is not a
   result, and a delta under ~7% is unresolvable by a single 600 s arm on an
   oscillating workload — do not quote one.
4. **Then iterate on optimization.**
5. **Write it up as a report.** The deliverable is not the code alone.

Reference checkouts are in `legacy/third_party/` (sparse; see
`legacy/.gitmodules`). tileRT ships no CUDA or C++ source, so there is nothing
to read there.

## Layout

| path | what |
|---|---|
| `python/`, `tokenspeed-kernel/`, `tokenspeed-scheduler/`, … | upstream tokenspeed, plus the handful of commits listed by `git log upstream/main..main` |
| `unieinfer/` | UnieAI plugins and tooling |
| `legacy/` | everything from before this restructure — see `legacy/README.md` |

## Staying current

```sh
git fetch upstream
git rebase upstream/main
```

`upstream` is `lightseekorg/tokenspeed`; `origin` is `UnieAI/UnieInfra`. Because
this repository descends from upstream's history, a rebase replays UnieAI's
commits one at a time and stops only where an upstream change and a UnieAI
change touch the same lines. `unieinfer/`, `legacy/` and this file never
conflict, because upstream does not have them.

When a rebase does stop, read the conflicting commit's message before resolving
it — and check whether upstream has since fixed the thing it was working
around, in which case the answer is to drop the commit rather than merge it.

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
