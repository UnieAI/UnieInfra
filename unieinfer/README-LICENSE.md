# Licensing inside `unieinfer/`

This directory is **AGPL-3.0-or-later**, `Copyright (c) 2026 UnieAI`. The rest of
the repository is not, and the boundary is deliberate.

| Area | Licence | Why |
|---|---|---|
| `unieinfer/**` (bench, kernel-sm120, xref, reports) | **AGPL-3.0-or-later** | Written here, and not intended for upstream |
| `python/tokenspeed/**`, `tokenspeed-scheduler/**`, `tokenspeed-kernel/**` | MIT (LightSeek Foundation) | Upstream's files. A contributor cannot relicense someone else's work, and these are exactly the files `UPSTREAMING.md` sends back as PRs |
| Ported code, wherever it lives | The origin's licence, retained | `runtime/cache/mamba_checkpoint_pool.py` is Apache-2.0 (SGLang); `compressed_tensors_wNa16`'s marlin utilities likewise. `kernel-provenance.md` records what each kernel derives from |

**The consequence to keep in mind: an AGPL file cannot be accepted by an MIT
upstream.** Anything under `unieinfer/` is therefore permanently ours. That is
what the plugin area is for, so nothing is lost — but it means a change that
*should* go upstream must be written in the upstream tree under MIT, not moved
here for convenience.

AGPL is compatible in this direction: the combined work may include the MIT and
Apache-2.0 components above. The reverse is not true.
