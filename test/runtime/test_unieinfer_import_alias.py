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

"""``unieinfer*`` import aliases of the ``tokenspeed*`` packages."""

from __future__ import annotations

import os
import subprocess
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import importlib
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ALIAS_FILES = {
    "unieinfer": REPO / "python/unieinfer/__init__.py",
    "unieinfer_kernel": REPO / "tokenspeed-kernel/python/unieinfer_kernel/__init__.py",
    "unieinfer_scheduler": (
        REPO / "tokenspeed-scheduler/python/unieinfer_scheduler/__init__.py"
    ),
}


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_top_level_alias_is_the_real_package():
    import tokenspeed
    import unieinfer

    assert unieinfer is tokenspeed
    assert sys.modules["unieinfer"] is tokenspeed


def test_submodule_alias_is_the_same_module_object():
    import unieinfer.runtime.metrics.config as alias
    from unieinfer.runtime.metrics.config import resolve_metrics_options

    import tokenspeed.runtime.metrics.config as real

    assert alias is real
    assert sys.modules["unieinfer.runtime.metrics.config"] is real
    assert resolve_metrics_options is real.resolve_metrics_options
    assert alias.__name__ == "tokenspeed.runtime.metrics.config"
    assert importlib.import_module("unieinfer.runtime.metrics") is (
        sys.modules["tokenspeed.runtime.metrics"]
    )


def test_alias_first_import_loads_the_module_once():
    # Fresh interpreter: the alias is the first thing to touch the module.
    result = _run(
        "import sys\n"
        "import unieinfer.flat_kvcache_select as a\n"
        "import tokenspeed.flat_kvcache_select as b\n"
        "assert a is b, (a, b)\n"
        "assert a.__name__ == 'tokenspeed.flat_kvcache_select'\n"
        "same = [k for k, v in sys.modules.items() if v is a]\n"
        "assert sorted(same) == ['tokenspeed.flat_kvcache_select',"
        " 'unieinfer.flat_kvcache_select'], same\n"
        "from unieinfer import runtime_select\n"
        "assert runtime_select is sys.modules['tokenspeed.runtime_select']\n"
    )
    assert result.returncode == 0, result.stderr


def test_missing_submodule_raises_module_not_found():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("unieinfer.no_such_module")


def test_python_dash_m_unieinfer_cli_help():
    alias = subprocess.run(
        [sys.executable, "-m", "unieinfer.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    real = subprocess.run(
        [sys.executable, "-m", "tokenspeed.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert alias.returncode == 0, alias.stderr
    assert "usage" in alias.stdout.lower()
    assert alias.stdout == real.stdout


def test_kernel_alias_is_the_real_package():
    import tokenspeed_kernel
    import tokenspeed_kernel.platform as real
    import unieinfer_kernel
    import unieinfer_kernel.platform as alias

    assert unieinfer_kernel is tokenspeed_kernel
    assert alias is real


def test_scheduler_alias_is_the_real_package():
    tokenspeed_scheduler = pytest.importorskip("tokenspeed_scheduler")
    import unieinfer_scheduler
    import unieinfer_scheduler.tokenspeed_scheduler_ext as alias_ext
    from unieinfer_scheduler import Scheduler

    assert unieinfer_scheduler is tokenspeed_scheduler
    assert alias_ext is tokenspeed_scheduler.tokenspeed_scheduler_ext
    assert Scheduler is tokenspeed_scheduler.Scheduler


def _alias_block(path: Path) -> str:
    text = path.read_text()
    start = text.index("# --- alias block")
    end = text.index("# --- end alias block")
    return text[start:end]


def test_alias_blocks_are_identical():
    blocks = {name: _alias_block(path) for name, path in ALIAS_FILES.items()}
    assert len(set(blocks.values())) == 1, sorted(blocks)
    for name, path in ALIAS_FILES.items():
        target = name.replace("unieinfer", "tokenspeed", 1)
        assert path.read_text().rstrip().endswith(f'_install("{name}", "{target}")')


def test_alias_packages_ship_with_their_distributions():
    with open(REPO / "python/pyproject.toml", "rb") as f:
        include = tomllib.load(f)["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "unieinfer" in include
    with open(REPO / "tokenspeed-scheduler/pyproject.toml", "rb") as f:
        packages = tomllib.load(f)["tool"]["scikit-build"]["wheel"]["packages"]
    assert "python/unieinfer_scheduler" in packages
    # tokenspeed-kernel's setup.py uses find_packages(), which picks up any
    # directory with an __init__.py next to tokenspeed_kernel.
    assert ALIAS_FILES["unieinfer_kernel"].parent.parent == (
        REPO / "tokenspeed-kernel/python"
    )
