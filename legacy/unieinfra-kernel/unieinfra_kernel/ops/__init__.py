# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""First-party hand-written / SOTA-ported kernels.

Layout: ``<family>/<solution>.py`` (e.g. ``moe/triton.py``, ``gemm/cute_dsl.py``)
per AGENTS.md — CuteDSL for NVIDIA, Triton Gluon for AMD, portable Triton across
vendors. Every module self-registers via ``@register_kernel(..., origin="first_party")``.
Discovered automatically by ``unieinfra_kernel._discovery``.
"""
