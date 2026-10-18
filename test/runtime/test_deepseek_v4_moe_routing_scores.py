# Copyright (c) 2026 LightSeek Foundation
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""DeepSeek V4 MoE routing must not score experts nobody reads.

`deepseek_v4_select_experts` returns `(topk_weights, topk_ids, scores)`, where
`scores` is `sqrt(softplus(logits))` -- the DeepSeek V4 expert-score function.
A MoE plan that routes from logits itself (`support_routing` true) consumes that
tensor as `router_logits`. A plan that takes precomputed top-k weights
(`support_routing` false) consumes only the weights and ids, so for it `scores`
is a float cast plus two elementwise kernels on every layer that feed nobody.

These tests pin the selection rule, because the waste is invisible: the numbers
stay correct, only the kernel count changes.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tokenspeed.runtime.models import deepseek_v4 as model_module
from tokenspeed.runtime.models.deepseek_v4 import DeepseekV4MoE


class _TopKFormat:
    """Minimal stand-in for a TopKOutputFormat."""

    def __init__(self, bypassed: bool) -> None:
        self._bypassed = bypassed

    def is_bypassed(self) -> bool:
        return self._bypassed


class _Gate:
    """Gate whose forward returns logits and whose bias table is absent."""

    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits
        self.e_score_correction_bias = None
        self.tid2eid = None

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        del hidden_states
        return self._logits


class _Experts:
    def __init__(self, *, support_routing: bool, bypassed: bool) -> None:
        self.support_routing = support_routing
        self.topk_output_format = _TopKFormat(bypassed)


class _Config:
    num_experts_per_tok = 6
    norm_topk_prob = True


def _layer(*, support_routing: bool, bypassed: bool = False) -> Any:
    """Build just enough of a DeepseekV4MoE to reach `_select_experts`."""

    layer = DeepseekV4MoE.__new__(DeepseekV4MoE)
    layer.config = _Config()
    layer.gate = _Gate(torch.zeros(2, 256, dtype=torch.float32))
    layer.experts = _Experts(support_routing=support_routing, bypassed=bypassed)
    return layer


def _record_need_scores(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    calls: list[bool] = []

    def fake_select_experts(
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        *,
        correction_bias: torch.Tensor | None = None,
        hash_indices_table: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        need_scores: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del router_logits, top_k, renormalize
        del correction_bias, hash_indices_table, input_ids
        calls.append(need_scores)
        empty = torch.zeros(1, dtype=torch.float32)
        return empty, empty.to(torch.int32), empty

    monkeypatch.setattr(model_module, "deepseek_v4_select_experts", fake_select_experts)
    return calls


def test_a_plan_that_routes_from_topk_gets_no_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """support_routing false -> topk weights are the only routing input."""

    calls = _record_need_scores(monkeypatch)
    layer = _layer(support_routing=False)

    layer._select_experts(torch.zeros(2, 8, dtype=torch.float32), None)

    assert calls == [False]


def test_a_plan_that_routes_from_logits_still_gets_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """support_routing true -> the plan reads router_logits, so scores stay."""

    calls = _record_need_scores(monkeypatch)
    layer = _layer(support_routing=True)

    layer._select_experts(torch.zeros(2, 8, dtype=torch.float32), None)

    assert calls == [True]


def test_a_bypassed_format_still_gets_no_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bypassed backend packs its own logits, so scores stay skipped."""

    calls = _record_need_scores(monkeypatch)
    layer = _layer(support_routing=True, bypassed=True)

    layer._select_experts(torch.zeros(2, 8, dtype=torch.float32), None)

    assert calls == [False]
