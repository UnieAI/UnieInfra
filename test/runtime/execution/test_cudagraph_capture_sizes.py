"""Tests for the decode CUDA graph capture-size ladder.

``get_batch_sizes_to_capture`` decides which decode batch sizes get a captured
CUDA graph. Its largest entry is a hard boundary, not a hint: a decode step
wider than that runs the eager path, and nothing else in the engine reports the
fallback. Serving AgentX with ``--max-num-seqs 80`` and
``--max-cudagraph-capture-size 40`` therefore put every 41..80-wide decode step
on the eager path, which measured 262 total tok/s and 229 ms TPOT at c64 versus
547 tok/s and 82 ms TPOT with the boundary at 80. These tests pin the boundary
arithmetic and the warning that makes the fallback visible.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tokenspeed.runtime.execution import forward_step  # noqa: E402
from tokenspeed.runtime.execution.forward_step import (  # noqa: E402
    get_batch_sizes_to_capture,
)


def _config(**overrides):
    base = dict(
        cudagraph_capture_sizes=None,
        max_num_seqs=80,
        data_parallel_size=1,
        disable_cuda_graph_padding=False,
        max_cudagraph_capture_size=160,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_default_ladder_reaches_max_num_seqs():
    bs = get_batch_sizes_to_capture(_config())
    assert max(bs) == 80
    assert bs == sorted(bs)
    assert len(bs) == len(set(bs))


def test_default_ladder_has_no_wide_gap_at_the_top():
    # The 79 rung is added precisely so max_num_seqs - 1 is not left eager.
    bs = get_batch_sizes_to_capture(_config())
    assert max(b for b in bs if b < 80) == 79


def test_cap_below_max_num_seqs_truncates_the_ladder():
    with mock.patch.object(forward_step, "logger"):
        bs = get_batch_sizes_to_capture(_config(max_cudagraph_capture_size=40))
    assert max(bs) == 40
    assert 41 not in bs


def test_cap_below_max_num_seqs_warns_with_both_numbers():
    with mock.patch.object(forward_step, "logger") as log:
        get_batch_sizes_to_capture(_config(max_cudagraph_capture_size=40))
    assert log.warning.call_count == 1
    assert "EAGER" in log.warning.call_args[0][0]
    # effective_max, max_bs, effective_max, max_bs -- a warning that does not
    # name the pair cannot be acted on.
    assert log.warning.call_args[0][1:] == (40, 80, 40, 80)


@pytest.mark.parametrize("cap", [80, 160])
def test_cap_covering_max_num_seqs_does_not_warn(cap):
    with mock.patch.object(forward_step, "logger") as log:
        bs = get_batch_sizes_to_capture(_config(max_cudagraph_capture_size=cap))
    assert log.warning.call_count == 0
    assert max(bs) == 80


def test_explicit_capture_sizes_are_truncated_by_the_cap():
    with mock.patch.object(forward_step, "logger"):
        bs = get_batch_sizes_to_capture(
            _config(
                cudagraph_capture_sizes=[8, 64, 128],
                max_cudagraph_capture_size=40,
            )
        )
    assert bs == [8]


def test_explicit_capture_sizes_gain_the_max_num_seqs_rung():
    # An explicit list that overshoots max_num_seqs must still leave the widest
    # legal decode step on a graph, or the engine loses it exactly when loaded.
    bs = get_batch_sizes_to_capture(
        _config(cudagraph_capture_sizes=[8, 64, 128], max_cudagraph_capture_size=160)
    )
    assert bs == [8, 64, 79, 80]


def test_data_parallel_shrinks_the_bound():
    bs = get_batch_sizes_to_capture(_config(max_num_seqs=64, data_parallel_size=2))
    assert max(bs) == 32


def test_padding_ladder_is_used_when_padding_is_disabled():
    bs = get_batch_sizes_to_capture(
        _config(disable_cuda_graph_padding=True, max_num_seqs=32)
    )
    assert bs == list(range(1, 33))
