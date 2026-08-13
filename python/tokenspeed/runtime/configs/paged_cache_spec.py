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

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def split_groups_to_equal_depth(layer_types: Sequence[str]) -> tuple[str, ...]:
    """Relabel so every cache group holds the same number of layers.

    The hybrid slab layout binds the i-th layer of EVERY group to slab i, so
    groups of unequal depth leave dead rows: with Gemma 4's 50 sliding against
    10 full, slabs 10..49 carry a sliding layer and nothing else, and an id
    allocated to the full group still reserves a row-block across all 50 slabs
    while only 10 layers use it. Measured with prefix caching off so nothing is
    shared: an 811-token request took 70 ids where its bytes are worth 57, and
    the pool ran out at 21 concurrent sequences.

    Splitting the deeper label round-robin into ``ceil(deep/shallow)``
    sub-groups makes every slab fully bound. vLLM does the same
    (``_get_kv_cache_groups_uniform_page_size``, "make the number of layers in
    each group identical", naming Gemma 3's 5:1), and so does Inkling here
    (55 sliding + 11 full -> 5 sub-groups of 11).

    Round-robin by rank rather than contiguous slices, matching both, so a
    pipeline stage holding a slice of the stack still sees every sub-group.

    Must run BEFORE the KV memory profile: the profile divides available
    memory by a per-token cell size derived from the slab count, so relabelling
    afterwards builds buffers for one layout against a token budget sized for
    another -- which allocated 3.6 GB against a 23,680-token claim when it was
    tried the other way round.

    Args:
        layer_types: Per-layer labels, as resolved from the checkpoint.

    Returns:
        Relabelled types, or ``layer_types`` unchanged when the split is a
        no-op (one label, already equal depth, or an unlabelled stack).
    """
    if not layer_types:
        return tuple(layer_types)
    counts = Counter(layer_types)
    if len(counts) < 2:
        return tuple(layer_types)
    shallow = min(counts.values())
    deep_label, deep = max(counts.items(), key=lambda kv: kv[1])
    num_groups = -(-deep // shallow)  # ceil
    if num_groups <= 1:
        return tuple(layer_types)
    out: list[str] = []
    rank = 0
    for label in layer_types:
        if label != deep_label:
            out.append(label)
        else:
            out.append(f"{label}_{rank % num_groups}")
            rank += 1
    return tuple(out)
