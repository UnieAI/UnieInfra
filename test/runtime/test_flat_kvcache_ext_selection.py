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

"""Which scheduler extension a checkpoint asks for, and why it is per-model.

A flat-built ext used to be a property of the installation, so installing one
so Gemma 4 could serve at all moved Qwen3.5-27B onto the flat path too and
cost it 39% of its throughput at concurrency 64 (prefix reuse 54.9% -> 0.0%
on a RAG shape). The selection is now per model, and these are the properties
that has to keep.
"""

from __future__ import annotations

import json
import types

import pytest

from tokenspeed.flat_kvcache_select import (
    FLAT_KVCACHE_RUNTIME_ENV,
    architectures_require_flat_kvcache,
    env_flat_kvcache_override,
    model_path_requires_flat_kvcache,
    select_flat_kvcache_ext,
    select_flat_kvcache_ext_for_model_path,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """A fresh process, as far as the module is concerned.

    The operator's override is snapshotted at import so the module's own
    writes cannot read back as operator intent, so clearing the variable is
    not enough -- the snapshot has to be retaken too.
    """
    import tokenspeed.flat_kvcache_select as module

    monkeypatch.delenv(FLAT_KVCACHE_RUNTIME_ENV, raising=False)
    module._capture_operator_override()
    yield
    module._capture_operator_override()


def _checkpoint(tmp_path, architectures):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"architectures": architectures}))
    return tmp_path


@pytest.mark.parametrize(
    "architectures,expected",
    [
        (["Gemma4ForConditionalGeneration"], True),
        (["Gemma4ForCausalLM"], True),
        (["KimiK3ForConditionalGeneration"], True),
        # The whole point: a hybrid that the uniform pool serves correctly must
        # NOT be dragged onto the flat path.
        (["Qwen3_5MoeForCausalLM"], False),
        (["Qwen3MoeForCausalLM"], False),
        (["LlamaForCausalLM"], False),
        ([], False),
        (None, False),
    ],
)
def test_architectures_decide(architectures, expected):
    assert architectures_require_flat_kvcache(architectures) is expected


def test_reads_architectures_off_the_checkpoint(tmp_path):
    gemma = _checkpoint(tmp_path / "gemma", ["Gemma4ForConditionalGeneration"])
    qwen = _checkpoint(tmp_path / "qwen", ["Qwen3_5MoeForCausalLM"])
    assert model_path_requires_flat_kvcache(gemma) is True
    assert model_path_requires_flat_kvcache(qwen) is False


@pytest.mark.parametrize("path", ["/nonexistent/checkpoint", None, 17])
def test_unreadable_checkpoint_is_not_an_error(path, tmp_path):
    """The loader reports a bad checkpoint with its own diagnostics.

    Raising here would turn "model path has a typo" into a traceback out of
    argument parsing, before anything has said which path it tried.
    """
    assert model_path_requires_flat_kvcache(path) is False


def test_malformed_config_json_is_not_an_error(tmp_path):
    (tmp_path / "config.json").write_text("{not json")
    assert model_path_requires_flat_kvcache(tmp_path) is False


def test_selection_is_recorded_in_the_environment(monkeypatch, tmp_path):
    """The engine is a subprocess, and it imports the ext at module scope.

    An env var is the only channel that reaches it in time, so the decision
    has to land there and not merely in a module global.
    """
    import os

    gemma = _checkpoint(tmp_path / "gemma", ["Gemma4ForConditionalGeneration"])
    assert select_flat_kvcache_ext_for_model_path(gemma) is True
    assert os.environ[FLAT_KVCACHE_RUNTIME_ENV] == "1"

    qwen = _checkpoint(tmp_path / "qwen", ["Qwen3_5MoeForCausalLM"])
    assert select_flat_kvcache_ext_for_model_path(qwen) is False
    assert os.environ[FLAT_KVCACHE_RUNTIME_ENV] == "0"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("ON", True),
        ("0", False),
        ("false", False),
        ("off", False),
        ("no", False),
        ("", None),
        ("   ", None),
    ],
)
def test_operator_override_is_read(monkeypatch, raw, expected):
    monkeypatch.setenv(FLAT_KVCACHE_RUNTIME_ENV, raw)
    assert env_flat_kvcache_override() is expected


@pytest.mark.parametrize("override,required", [(True, False), (False, True)])
def test_operator_override_beats_the_architecture(monkeypatch, override, required):
    """An architecture this module has not learned yet still has to be servable.

    Without this, adding a model that needs the flat path would be a code
    change rather than a flag.
    """
    import tokenspeed.flat_kvcache_select as module

    monkeypatch.setenv(FLAT_KVCACHE_RUNTIME_ENV, "1" if override else "0")
    module._capture_operator_override()
    assert select_flat_kvcache_ext(required) is override


def test_our_own_write_is_not_mistaken_for_an_operator_override(tmp_path):
    """Two selections in one process must not stick on the first answer.

    `select_*` writes the same variable `env_flat_kvcache_override` reads, so
    without a snapshot the second call would see its own first write and
    every model after the first would inherit that model's extension.
    """
    gemma = _checkpoint(tmp_path / "gemma", ["Gemma4ForConditionalGeneration"])
    qwen = _checkpoint(tmp_path / "qwen", ["Qwen3_5MoeForCausalLM"])
    assert select_flat_kvcache_ext_for_model_path(gemma) is True
    assert select_flat_kvcache_ext_for_model_path(qwen) is False


def _model_config(architectures):
    return types.SimpleNamespace(
        hf_config=types.SimpleNamespace(architectures=architectures)
    )


def test_verify_rejects_a_radix_ext_under_a_model_that_needs_flat(monkeypatch):
    from tokenspeed.runtime.configs import paged_cache_spec

    monkeypatch.setattr(
        paged_cache_spec, "scheduler_ext_flat_kvcache_built", lambda: False
    )
    with pytest.raises(RuntimeError, match="flat KV path"):
        paged_cache_spec.verify_flat_kvcache_ext_for_model(
            _model_config(["Gemma4ForConditionalGeneration"])
        )


def test_verify_warns_when_a_model_gets_the_flat_ext_it_does_not_need(
    monkeypatch, caplog
):
    """Silent is the one thing this must not be: it reads as a 39% regression."""
    from tokenspeed.runtime.configs import paged_cache_spec

    monkeypatch.setattr(
        paged_cache_spec, "scheduler_ext_flat_kvcache_built", lambda: True
    )
    with caplog.at_level("WARNING"):
        paged_cache_spec.verify_flat_kvcache_ext_for_model(
            _model_config(["Qwen3_5MoeForCausalLM"])
        )
    assert "does not need it" in caplog.text


@pytest.mark.parametrize("built_flat", [True, False])
def test_verify_accepts_the_matching_pair(monkeypatch, built_flat):
    from tokenspeed.runtime.configs import paged_cache_spec

    monkeypatch.setattr(
        paged_cache_spec, "scheduler_ext_flat_kvcache_built", lambda: built_flat
    )
    architectures = (
        ["Gemma4ForConditionalGeneration"] if built_flat else ["Qwen3_5MoeForCausalLM"]
    )
    paged_cache_spec.verify_flat_kvcache_ext_for_model(_model_config(architectures))
