# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Engine adapters.

Each submodule wires the base-independent kernel registry into one host engine.
Nothing here is imported by :func:`unieinfra_kernel.discover`; an adapter is
loaded by its host (a vLLM entry point, an explicit import from the tokenspeed
runtime), so a process that does not use an engine never pays for it.
"""
