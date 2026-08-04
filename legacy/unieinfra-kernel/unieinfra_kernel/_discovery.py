# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Kernel discovery for :mod:`unieinfra_kernel`.

Importing :mod:`unieinfra_kernel` calls :func:`discover`, which recursively
imports every implementation module under ``ops/``, ``kda/``, and ``contrib/``.
Each such module runs its ``@register_kernel(...)`` decorator as an import side
effect, so the kernel self-registers into the **shared** ``tokenspeed_kernel``
``KernelRegistry`` singleton — this package deliberately does *not* create a
second registry.

Rules:

* Recurse the three implementation trees; import each ``*.py`` module.
* Skip files whose basename starts with ``_`` (``__init__``, ``_scaffold``,
  private helpers) and skip ``__pycache__``.
* Import is idempotent: Python's module cache plus a one-shot guard mean calling
  :func:`discover` repeatedly is cheap and safe.
* A single kernel module that fails to import is logged and skipped, never fatal
  — one broken experimental kernel must not take down engine startup.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from types import ModuleType

logger = logging.getLogger(__name__)

# Implementation trees scanned for self-registering kernels, in this order.
_IMPL_SUBPACKAGES: tuple[str, ...] = ("ops", "kda", "contrib")

# Entry-point group other distributions can advertise to have their kernel
# modules imported at discovery time (see pyproject).
_ENTRY_POINT_GROUP = "unieinfra_kernel.plugins"

_discovered = False


def _import_tree(package: ModuleType, errors: list[tuple[str, Exception]]) -> list[str]:
    """Recursively import every non-underscore module under ``package``.

    Returns the list of fully-qualified module names that were imported.
    """
    imported: list[str] = []
    prefix = package.__name__ + "."
    # walk_packages recurses into subpackages; onerror keeps one bad module from
    # aborting the whole walk.
    for module_info in pkgutil.walk_packages(
        package.__path__,
        prefix,
        onerror=lambda name: errors.append((name, RuntimeError("walk error"))),
    ):
        basename = module_info.name.rsplit(".", 1)[-1]
        if basename.startswith("_"):
            continue
        # Subpackages (module_info.ispkg) carry no kernels themselves; their
        # child modules are visited by the same walk, so importing the package
        # object is unnecessary. Import leaf/regular modules only.
        if module_info.ispkg:
            continue
        try:
            importlib.import_module(module_info.name)
            imported.append(module_info.name)
        except Exception as exc:  # noqa: BLE001 - isolate per-module failures
            logger.warning(
                "unieinfra_kernel: failed to import kernel module %s: %s",
                module_info.name,
                exc,
            )
            errors.append((module_info.name, exc))
    return imported


def _import_entry_points(errors: list[tuple[str, Exception]]) -> list[str]:
    """Import modules advertised on the ``unieinfra_kernel.plugins`` group."""
    imported: list[str] = []
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover - importlib.metadata always present 3.10+
        return imported

    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - <3.10 selection API
        eps = entry_points().get(_ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]

    for ep in eps:
        try:
            ep.load()
            imported.append(ep.value)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "unieinfra_kernel: failed to load plugin entry point %s: %s",
                getattr(ep, "value", ep),
                exc,
            )
            errors.append((getattr(ep, "value", str(ep)), exc))
    return imported


def discover(force: bool = False) -> list[str]:
    """Import every kernel module so it self-registers.

    Args:
        force: Re-run discovery even if it already ran in this process. Useful in
            tests that reset the registry.

    Returns:
        The list of fully-qualified module names imported during this call.
    """
    global _discovered
    if _discovered and not force:
        return []

    errors: list[tuple[str, Exception]] = []
    imported: list[str] = []

    for subpackage in _IMPL_SUBPACKAGES:
        qualified = f"{__package__}.{subpackage}"
        try:
            package = importlib.import_module(qualified)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "unieinfra_kernel: implementation tree %s missing/unimportable: %s",
                qualified,
                exc,
            )
            errors.append((qualified, exc))
            continue
        imported.extend(_import_tree(package, errors))

    imported.extend(_import_entry_points(errors))

    _discovered = True
    logger.debug(
        "unieinfra_kernel: discovery imported %d module(s), %d error(s)",
        len(imported),
        len(errors),
    )
    return imported
