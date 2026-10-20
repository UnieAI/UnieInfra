# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Tests for the standalone selection core, the boundary shim and dispatch.

The standalone half has to be exercised in a *subprocess*: ``conftest.py``
deliberately makes ``tokenspeed_kernel`` importable so the pre-existing tests
keep running against the shared registry, and
:data:`unieinfra_kernel.boundary.BACKEND` is decided once at import time. A
subprocess with ``UNIEINFRA_KERNEL_REGISTRY=standalone`` is therefore the only
honest way to assert on the vLLM-side configuration.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent

# Trait vocabulary published by ``tokenspeed_kernel.ops.gemm.mm``. Our façade
# must publish exactly this set: a kernel gating on ``m_le_8`` has to see the
# same trait names on either base or it silently never matches.
# Every trait name ``tokenspeed_kernel.ops.gemm.mm`` publishes. The dispatch
# façade must speak all of them so a kernel registered against either base sees
# the same vocabulary.
_TOKENSPEED_TRAITS = {
    "m_le_8",
    "n_le_16384",
    "n_align_16",
    "k_align_16",
    "k_align_32",
    "n_align_64",
    "n_align_128",
    "k_align_64",
    "k_align_128",
    "n_min_128",
    "k_min_128",
}

# Additive extensions this package introduces. ``m_le_4`` / ``m_le_2`` are the
# finer batch-1 decode buckets needed by the quantized GEMVs, and
# ``m_min_16`` / ``m_le_256`` bound the decode-*batch* band (E45); in both cases
# the declarative gate has to match the kernel's profitability rule exactly, so
# that CUDA-graph capture -- which walks M = 1..512 -- cannot reach a shape the
# kernel would only decline at run time (E38). Purely additive: selection never
# filters on a trait a kernel does not declare, so no pre-existing kernel is
# affected.
_UNIEINFRA_EXTRA_TRAITS = {"m_le_4", "m_le_2", "m_min_16", "m_le_256", "n_le_32768"}

_EXPECTED_TRAITS = _TOKENSPEED_TRAITS | _UNIEINFRA_EXTRA_TRAITS


def _run_standalone(body: str) -> dict:
    """Run ``body`` in a subprocess forced onto the standalone registry.

    Args:
        body: Python source that ``print``s one JSON object as its last line.

    Returns:
        The decoded JSON object.
    """
    env = dict(os.environ)
    env["UNIEINFRA_KERNEL_REGISTRY"] = "standalone"
    env["PYTHONPATH"] = str(_PKG_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Never inherit a tokenspeed source path that would defeat the forcing.
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_selection_core_imports_only_torch_and_stdlib():
    """The ported core must not reach back into tokenspeed or sglang."""
    core = _PKG_ROOT / "unieinfra_kernel" / "selection"
    offenders = []
    # ``__init__.py`` is ours and only *mentions* tokenspeed in prose; the four
    # ported modules must not reference it at all, in code or comment.
    for name in ("registry.py", "selection.py", "platform.py", "signature.py"):
        text = (core / name).read_text()
        for needle in ("from sglang", "import sglang", "tokenspeed"):
            if needle in text:
                offenders.append(f"{name}: {needle}")
    assert not offenders, offenders


def test_standalone_backend_registers_kernels():
    """A bare process with no tokenspeed gets a populated standalone registry."""
    result = _run_standalone("""
        import json
        import unieinfra_kernel  # noqa: F401
        from unieinfra_kernel.boundary import BACKEND, KernelRegistry

        registry = KernelRegistry.get()
        print(json.dumps({
            "backend": BACKEND,
            "kernels": sorted(s.name for s in registry.list_kernels()),
            "operators": sorted(list(op) for op in registry.list_operators()),
        }))
        """)
    assert result["backend"] == "standalone"
    assert "unieinfra_decode_gemv_triton" in result["kernels"]
    assert "torch_linear" in result["kernels"]
    assert ["gemm", "mm"] in result["operators"]


def test_shape_traits_match_the_tokenspeed_vocabulary():
    """``shape_traits`` must publish the same trait names as ``gemm.mm``."""
    result = _run_standalone("""
        import json
        from unieinfra_kernel.dispatch import shape_traits

        print(json.dumps(sorted(shape_traits(1, 6144, 4096))))
        """)
    assert set(result) == _EXPECTED_TRAITS
    assert _TOKENSPEED_TRAITS <= set(result), "a tokenspeed trait name went missing"


@pytest.mark.skipif(
    os.environ.get("UNIEINFRA_TEST_GPU", "1") == "0", reason="GPU tests disabled"
)
def test_standalone_selection_routes_decode_shapes():
    """On a GPU box, the decode slice picks our kernel and nothing else does.

    ``m_le_8`` / ``n_le_16384`` are what confine the GEMV: ``gate_up_proj``
    (``N = 24576``) and every prefill-sized ``M`` must fall back to the
    incumbent ``torch_linear``.
    """
    result = _run_standalone("""
        import json
        import torch

        if not torch.cuda.is_available():
            print(json.dumps({"skipped": True}))
            raise SystemExit(0)

        import unieinfra_kernel  # noqa: F401
        from unieinfra_kernel.dispatch import linear, selection_log
        from unieinfra_kernel.ops.gemm.decode_gemv_triton import (
            gemv_stats,
            reset_gemv_stats,
        )

        reset_gemv_stats()
        cases = [(1, 6144, 4096), (1, 4096, 4096), (1, 24576, 4096),
                 (4, 6144, 4096), (256, 6144, 4096)]
        errors = []
        for M, N, K in cases:
            x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
            b = torch.randn(N, dtype=torch.bfloat16, device="cuda")
            got = linear(x, w, b)
            ref = torch.nn.functional.linear(x, w, b)
            errors.append((got.float() - ref.float()).abs().max().item())

        print(json.dumps({
            "skipped": False,
            "log": selection_log(),
            "stats": gemv_stats(),
            "max_abs_err": max(errors),
        }))
        """)
    if result.get("skipped"):
        pytest.skip("no CUDA device visible")

    by_shape = {(e["M"], e["N"], e["K"]): e["kernel"] for e in result["log"]}
    assert by_shape[(1, 6144, 4096)] == "unieinfra_decode_gemv_triton"
    assert by_shape[(1, 4096, 4096)] == "unieinfra_decode_gemv_triton"
    # n_le_16384 excludes gate_up_proj; m_le_8 excludes prefill.
    assert by_shape[(1, 24576, 4096)] == "torch_linear"
    assert by_shape[(256, 6144, 4096)] == "torch_linear"
    # M = 4 passes the declarative gate but the runtime profitability rule
    # declines it, so the wrapper delegates rather than launching.
    assert result["stats"]["fast"] == 2
    assert result["stats"]["delegated"] == 1
    assert result["max_abs_err"] < 0.2


def test_vllm_integration_imports_without_vllm():
    """The adapter module must import even where vLLM is absent."""
    result = _run_standalone("""
        import json
        from unieinfra_kernel.integrations.vllm import register
        from unieinfra_kernel.integrations.vllm.plugin import QUANT_NAME

        # register() swallows a missing vLLM: a kernel package must never be
        # able to stop an engine (or a plain script) from starting.
        register()
        print(json.dumps({"quant_name": QUANT_NAME}))
        """)
    assert result["quant_name"] == "unieinfra"
