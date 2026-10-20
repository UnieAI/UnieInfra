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

"""ModelRunner must load out-of-tree kernel plugins, and must do it in time.

``tokenspeed_kernel.plugins`` is explicit by design -- the host application has
to call ``discover_plugins()``. If no host does, installing a plugin package
silently has no effect. And if a host calls it *after* the model is built, the
plugin's kernels are registered too late to be selected, which fails in the same
silent way. Both are covered here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tokenspeed.runtime.execution import model_runner as model_runner_mod
from tokenspeed.runtime.execution.model_runner import ModelRunner


@pytest.fixture
def stub_config():
    """The smallest model_config/server_args pair ``__init__`` will accept."""
    model_config = SimpleNamespace(
        is_generation=True,
        is_multimodal=False,
        mambaish_config=None,
        is_hybrid_gdn=False,
        hf_config=SimpleNamespace(sliding_window=None),
        _parse_quant_hf_config=lambda: None,
    )
    server_args = SimpleNamespace(
        device="cuda",
        mapping=None,
        moe_backend="triton",
        draft_moe_backend=None,
        kv_cache_dtype="bfloat16",
        enable_memory_saver=False,
    )
    return model_config, server_args


@pytest.fixture
def patched_runner(monkeypatch):
    """Neutralise everything ``__init__`` does except the ordering under test."""
    calls: list[str] = []

    # raising=False so that removing the call *and* its import fails on the
    # assertion below, which explains the regression, rather than on an
    # AttributeError from monkeypatch that does not.
    monkeypatch.setattr(
        model_runner_mod,
        "discover_plugins",
        lambda: calls.append("discover_plugins"),
        raising=False,
    )
    monkeypatch.setattr(
        ModelRunner,
        "load_model",
        lambda self: calls.append("load_model"),
    )
    monkeypatch.setattr(
        model_runner_mod, "global_server_args_dict_update", lambda *_: None
    )
    monkeypatch.setattr(model_runner_mod, "initialize_moe_config", lambda *_: None)
    monkeypatch.setattr(
        model_runner_mod, "TorchMemorySaverAdapter", MagicMock(create=MagicMock())
    )
    return calls


def test_model_runner_discovers_plugins(patched_runner, stub_config):
    model_config, server_args = stub_config

    ModelRunner(
        model_config=model_config,
        server_args=server_args,
        gpu_id=0,
        global_rank=0,
    )

    assert "discover_plugins" in patched_runner, (
        "ModelRunner never called discover_plugins(), so an installed kernel "
        "plugin would not be registered and the documented plugin contract "
        "would not work end to end."
    )


def test_plugins_are_discovered_before_the_model_is_built(patched_runner, stub_config):
    """Ordering is the whole point: registering after selection is a no-op."""
    model_config, server_args = stub_config

    ModelRunner(
        model_config=model_config,
        server_args=server_args,
        gpu_id=0,
        global_rank=0,
    )

    assert patched_runner.index("discover_plugins") < patched_runner.index(
        "load_model"
    ), f"expected discover_plugins before load_model, got {patched_runner}"


def test_discovery_runs_for_each_worker(patched_runner, stub_config):
    """A process builds a ModelRunner per worker (draft and target).

    ``discover_plugins()`` is idempotent -- it skips plugins it has already
    loaded -- so the call site deliberately carries no once-only guard. This
    pins that: a second runner must still call it rather than assume the first
    one did, which would break whenever workers are constructed independently.
    """
    model_config, server_args = stub_config

    for _ in range(2):
        ModelRunner(
            model_config=model_config,
            server_args=server_args,
            gpu_id=0,
            global_rank=0,
        )

    assert patched_runner.count("discover_plugins") == 2
