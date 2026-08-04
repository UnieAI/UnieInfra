# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Re-export of the vLLM plugin implementation.

The implementation itself lives in the top-level :mod:`unieinfra_vllm_plugin`
module, *outside* this package, so that vLLM loading the entry point does not
drag in :mod:`unieinfra_kernel` (and therefore Triton, and therefore kernel
discovery) in processes that never ask for our kernels. This module exists so
the code is still discoverable at the obvious import path.
"""

from __future__ import annotations

from unieinfra_vllm_plugin import (  # noqa: F401
    QUANT_NAME,
    discover_kernels,
    register,
)

__all__ = ["QUANT_NAME", "discover_kernels", "register"]


def __getattr__(name: str):
    """Forward ``UnieInfraConfig`` / ``UnieInfraLinearMethod`` lookups."""
    import unieinfra_vllm_plugin

    return getattr(unieinfra_vllm_plugin, name)
