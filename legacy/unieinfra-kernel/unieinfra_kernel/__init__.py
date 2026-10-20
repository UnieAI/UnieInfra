# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""UnieInfra's kernel selection layer and its first-party / KDA / community kernels.

The package has two halves:

* :mod:`unieinfra_kernel.selection` — the **base-independent** capability
  registry and selection pipeline (``registry`` / ``selection`` / ``platform``
  / ``signature``). It imports only ``torch`` and the stdlib.
* ``ops/``, ``kda/``, ``contrib/`` — the kernels themselves, each declaring its
  capability and trait gates via ``@register_kernel``.

Importing this package triggers :func:`unieinfra_kernel._discovery.discover`,
which recursively imports every kernel module under those three trees so each
one self-registers into a **single** ``KernelRegistry`` singleton. Which
singleton that is comes from :mod:`unieinfra_kernel.boundary`: the vendored
standalone core when running on vLLM, or ``tokenspeed_kernel``'s registry when
running on the tokenspeed base, so runtime code there still sees exactly one
kernel boundary.

Typical use::

    import unieinfra_kernel  # noqa: F401  -> kernels are now registered & selectable

Inside vLLM nothing needs to import it by hand: the ``vllm.general_plugins``
entry point in ``pyproject.toml`` calls
:func:`unieinfra_kernel.integrations.vllm.register` in every vLLM process.
"""

from __future__ import annotations

from unieinfra_kernel._discovery import discover

__all__ = ["discover"]

# Register everything on import. Kept side-effecting so a bare
# ``import unieinfra_kernel`` is enough for the engine and benchmarks.
discover()
