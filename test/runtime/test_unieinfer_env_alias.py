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

"""``UNIEINFER_*`` spellings of ``TOKENSPEED_*`` / ``UNIEINFRA_*`` variables."""

from __future__ import annotations

import os
import subprocess
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import io
import logging

import pytest
from tokenspeed_kernel import env_alias

from tokenspeed.runtime.utils.env import envs

LOGGER = "tokenspeed_kernel.env_alias"


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(env_alias, "_warned", set())


def test_new_name():
    assert env_alias.new_name("TOKENSPEED_NVTX") == "UNIEINFER_NVTX"
    assert env_alias.new_name("UNIEINFRA_GEMM") == "UNIEINFER_GEMM"
    assert env_alias.new_name("MOONCAKE_MASTER") is None
    assert env_alias.new_name("TOKENSPEED_") is None


@pytest.mark.parametrize(
    "environ, expected",
    [
        ({}, "dflt"),
        ({"TOKENSPEED_X": "old"}, "old"),
        ({"UNIEINFER_X": "new"}, "new"),
        ({"TOKENSPEED_X": "same", "UNIEINFER_X": "same"}, "same"),
        ({"TOKENSPEED_X": "old", "UNIEINFER_X": "new"}, "new"),
        ({"TOKENSPEED_X": "old", "UNIEINFER_X": ""}, ""),
    ],
)
def test_getenv_precedence(environ, expected):
    assert env_alias.getenv("TOKENSPEED_X", "dflt", environ=environ) == expected


def test_getenv_unieinfra_legacy_prefix():
    environ = {"UNIEINFRA_FA2_REGISTER_DECODE": "0"}
    assert env_alias.getenv("UNIEINFRA_FA2_REGISTER_DECODE", environ=environ) == "0"
    environ["UNIEINFER_FA2_REGISTER_DECODE"] = "1"
    assert env_alias.getenv("UNIEINFRA_FA2_REGISTER_DECODE", environ=environ) == "1"


def test_getenv_other_names_untouched():
    environ = {"MOONCAKE_MASTER": "m", "UNIEINFER_MOONCAKE_MASTER": "x"}
    assert env_alias.getenv("MOONCAKE_MASTER", environ=environ) == "m"


def test_disagreement_warns_once(caplog):
    environ = {"TOKENSPEED_X": "old", "UNIEINFER_X": "new"}
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(3):
            assert env_alias.getenv("TOKENSPEED_X", environ=environ) == "new"
    warnings = [r for r in caplog.records if r.name == LOGGER]
    assert len(warnings) == 1
    assert "UNIEINFER_X" in warnings[0].getMessage()
    assert "TOKENSPEED_X" in warnings[0].getMessage()


def test_agreement_does_not_warn(caplog):
    environ = {"TOKENSPEED_X": "v", "UNIEINFER_X": "v"}
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        env_alias.getenv("TOKENSPEED_X", environ=environ)
    assert not [r for r in caplog.records if r.name == LOGGER]


def test_is_set():
    assert env_alias.is_set("TOKENSPEED_X", environ={"UNIEINFER_X": "1"})
    assert env_alias.is_set("TOKENSPEED_X", environ={"TOKENSPEED_X": "1"})
    assert not env_alias.is_set("TOKENSPEED_X", environ={})


def test_apply_aliases_mirrors_new_onto_legacy(caplog):
    environ = {
        "UNIEINFER_A": "1",
        "UNIEINFER_B": "new",
        "TOKENSPEED_B": "old",
        "TOKENSPEED_C": "kept",
        "UNIEINFER_": "ignored",
    }
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        env_alias.apply_aliases(environ)
        env_alias.apply_aliases(environ)
    assert environ["TOKENSPEED_A"] == "1"
    assert environ["TOKENSPEED_B"] == "new"
    assert environ["TOKENSPEED_C"] == "kept"
    assert "TOKENSPEED_" not in environ
    warnings = [r for r in caplog.records if r.name == LOGGER]
    assert len(warnings) == 1 and "TOKENSPEED_B" in warnings[0].getMessage()


def test_registry_reads_new_name(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_HEALTH_CHECK_TIMEOUT", raising=False)
    monkeypatch.delenv("UNIEINFER_HEALTH_CHECK_TIMEOUT", raising=False)
    field = envs.__class__.__dict__["TOKENSPEED_HEALTH_CHECK_TIMEOUT"]
    assert field.alias == "UNIEINFER_HEALTH_CHECK_TIMEOUT"
    assert not field.is_set()
    assert field.get() == 20
    monkeypatch.setenv("UNIEINFER_HEALTH_CHECK_TIMEOUT", "7")
    assert field.is_set()
    assert envs.TOKENSPEED_HEALTH_CHECK_TIMEOUT.get() == 7
    monkeypatch.setenv("TOKENSPEED_HEALTH_CHECK_TIMEOUT", "9")
    assert envs.TOKENSPEED_HEALTH_CHECK_TIMEOUT.get() == 7
    monkeypatch.delenv("UNIEINFER_HEALTH_CHECK_TIMEOUT")
    assert envs.TOKENSPEED_HEALTH_CHECK_TIMEOUT.get() == 9


def test_registry_set_override_clear_with_new_name(monkeypatch):
    field = envs.__class__.__dict__["TOKENSPEED_HEALTH_CHECK_TIMEOUT"]
    monkeypatch.setenv("UNIEINFER_HEALTH_CHECK_TIMEOUT", "7")
    monkeypatch.setenv("TOKENSPEED_HEALTH_CHECK_TIMEOUT", "7")
    with field.override(3):
        # A value set in code is what every reader sees.
        assert field.get() == 3
        assert "UNIEINFER_HEALTH_CHECK_TIMEOUT" not in os.environ
    assert os.environ["UNIEINFER_HEALTH_CHECK_TIMEOUT"] == "7"
    assert os.environ["TOKENSPEED_HEALTH_CHECK_TIMEOUT"] == "7"
    assert field.get() == 7
    field.set(5)
    assert field.get() == 5
    field.clear()
    assert field.get() == 20
    assert "UNIEINFER_HEALTH_CHECK_TIMEOUT" not in os.environ


def test_registry_set_none_with_new_name(monkeypatch):
    field = envs.__class__.__dict__["TOKENSPEED_HEALTH_CHECK_TIMEOUT"]
    monkeypatch.setenv("UNIEINFER_HEALTH_CHECK_TIMEOUT", "7")
    try:
        field.set(None)
        assert field.get() is None
    finally:
        field.clear()


def _child_env(**extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("UNIEINFER_ALIAS_PROBE", "TOKENSPEED_ALIAS_PROBE"))
    }
    env.update(extra)
    return env


@pytest.mark.parametrize("module", ["tokenspeed", "unieinfer"])
def test_import_mirrors_for_direct_reads(module):
    # `import tokenspeed` must not pay for tokenspeed_kernel, yet must apply
    # the aliases before any direct os.environ read of TOKENSPEED_*.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import os, sys; import {module}\n"
            "assert 'tokenspeed_kernel' not in sys.modules\n"
            "print(os.environ['TOKENSPEED_ALIAS_PROBE'])\n"
            "print(os.environ['TOKENSPEED_ALIAS_PROBE_BOTH'])\n",
        ],
        env=_child_env(
            UNIEINFER_ALIAS_PROBE="a",
            UNIEINFER_ALIAS_PROBE_BOTH="new",
            TOKENSPEED_ALIAS_PROBE_BOTH="old",
        ),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["a", "new"]
    assert result.stderr.count("disagree") == 1


def test_kernel_import_mirrors_for_direct_reads():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, tokenspeed_kernel\n"
            "print(os.environ['TOKENSPEED_ALIAS_PROBE'])\n",
        ],
        env=_child_env(UNIEINFER_ALIAS_PROBE="k"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["k"]


@pytest.mark.parametrize(
    "environ, shown",
    [
        ({}, True),
        ({"UNIEINFER_DISABLE_LOGO": "1"}, False),
        ({"TOKENSPEED_DISABLE_LOGO": "1"}, False),
        ({"TOKENSPEED_DISABLE_LOGO": "1", "UNIEINFER_DISABLE_LOGO": "0"}, True),
    ],
)
def test_logo_disable_new_name(monkeypatch, environ, shown):
    from tokenspeed.cli._logo import print_logo

    monkeypatch.delenv("TOKENSPEED_DISABLE_LOGO", raising=False)
    monkeypatch.delenv("UNIEINFER_DISABLE_LOGO", raising=False)
    for key, value in environ.items():
        monkeypatch.setenv(key, value)
    stream = io.StringIO()
    print_logo(stream=stream)
    assert bool(stream.getvalue()) is shown
