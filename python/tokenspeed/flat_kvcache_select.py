# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""Which tokenspeed_scheduler extension a checkpoint needs, decided early.

The radix and flat schedulers are two different .so files, not two modes of
one. `tokenspeed-scheduler/CMakeLists.txt` turns `TOKENSPEED_FLAT_KVCACHE`
into a single `PUBLIC` compile definition, and the flat build uses it to
compile the radix insert/publish path OUT (`#if !TOKENSPEED_FLAT_KVCACHE`
around `finalizeRadixPageTableEmission` and the `hybrid_prefix_cache_->
CommitChunk` arms in `csrc/scheduler/operations/forward.cpp`), while its
`Scheduler` constructor unconditionally builds a `KvCacheCoordinator` and so
throws `MakeCoordinator requires at least one spec` on an empty group list.
Neither .so can do the other's job.

That is why this module exists and why it is a LEAF. The choice has to be
made before the first `import tokenspeed_scheduler`, which
`runtime/engine/event_loop.py` performs at module scope -- and importing
anything under `tokenspeed.runtime.configs` drags in transformers, torch,
tokenspeed_kernel and flashinfer, i.e. everything, far too late and far too
heavy for a decision that is one field of one JSON file. So: stdlib only, no
package siblings, importable from `tokenspeed/cli/__main__.py` before the
engine exists.
"""

from __future__ import annotations

import json
import os

# Which tokenspeed_scheduler extension this process wants, carried across a
# process boundary. `ts serve` runs the gateway and the engine as separate
# processes, so an env var is the one channel that reaches both. It doubles as
# the operator override.
FLAT_KVCACHE_RUNTIME_ENV = "TOKENSPEED_FLAT_KVCACHE_RUNTIME"

# Architectures whose KV geometry the uniform radix pool cannot represent.
# These raise rather than degrade on a radix ext -- Kimi-K3 in
# `preflight_kimi_k3_flat_consumers`, Gemma 4 in `_apply_gemma4_hetero_kv`
# (its full-attention layers carry half the KV bytes of its sliding ones, and
# the uniform pool refuses those writes) -- so for them the flat path is not
# an optimization, it is the only way they serve at all.
#
# Everything NOT listed here keeps the radix extension, which is the point.
# The flat ext used to be a property of the installation, so installing it for
# Gemma 4 moved Qwen3.5-27B onto the flat path too, where its
# `linear_attention` state group drives the scheduler's page grain from 16
# tokens to 400, zeroes the recurrent-snapshot pool, and takes prefix reuse on
# a RAG shape from 54.9% to 0.0% -- 590.9 -> 361.8 tok/s at concurrency 64.
FLAT_KVCACHE_REQUIRED_ARCHITECTURES = frozenset(
    {
        "KimiK3ForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "Gemma4ForCausalLM",
        "Gemma4TextModel",
    }
)


def architectures_require_flat_kvcache(architectures) -> bool:
    """True iff any architecture name needs the flat KV path."""
    return any(
        arch in FLAT_KVCACHE_REQUIRED_ARCHITECTURES for arch in (architectures or ())
    )


def model_path_requires_flat_kvcache(model_path) -> bool:
    """Same question, off a checkpoint directory's ``config.json``.

    Any failure to read or parse reads as "no": an unreadable checkpoint is
    the model loader's error to report with its own diagnostics, not this
    function's to guess at.
    """
    try:
        with open(os.path.join(os.fspath(model_path), "config.json")) as handle:
            config = json.load(handle)
    except (OSError, TypeError, ValueError):
        return False
    return architectures_require_flat_kvcache(config.get("architectures"))


def env_flat_kvcache_override() -> bool | None:
    """The operator's explicit choice, or None when they made none."""
    raw = os.environ.get(FLAT_KVCACHE_RUNTIME_ENV)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() not in ("0", "false", "off", "no")


# The operator's setting as it stood when this process started, captured
# before anything here can write to it. Reading the variable live instead
# would let our OWN write come back as "the operator asked for this", which is
# both wrong and unfalsifiable. A child process legitimately inherits the
# parent's decision through the same variable -- that is the mechanism, not a
# violation of it.
_OPERATOR_OVERRIDE: bool | None = None


def _capture_operator_override() -> None:
    global _OPERATOR_OVERRIDE
    _OPERATOR_OVERRIDE = env_flat_kvcache_override()


_capture_operator_override()


def select_flat_kvcache_ext(required: bool) -> bool:
    """Record which extension this process (and its children) should load.

    An explicit operator setting always wins, so an architecture this module
    has not learned about yet can still be served without a code change.
    Returns the decision.
    """
    gate = required if _OPERATOR_OVERRIDE is None else _OPERATOR_OVERRIDE
    os.environ[FLAT_KVCACHE_RUNTIME_ENV] = "1" if gate else "0"
    return gate


def select_flat_kvcache_ext_for_model_path(model_path) -> bool:
    """`select_flat_kvcache_ext` with the answer read off the checkpoint."""
    return select_flat_kvcache_ext(model_path_requires_flat_kvcache(model_path))
