# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""Build/artifact capability tier — "is there *code* for this device?".

:class:`~unieinfra_kernel.selection.platform.CapabilityRequirement` answers two
questions well: *what silicon is this* (``min_arch_version`` /
``max_arch_version`` / ``vendors``) and *what can the silicon do*
(``required_features``). It cannot answer a third question that decides just as
many selections in practice:

    **Does the binary that is actually installed contain code for this device?**

A kernel can be perfectly expressible on the hardware and still be unreachable
because the wheel on this machine was compiled without that architecture, the
optional dependency is missing, the symbol was never bound, or the JIT toolchain
in this container cannot target the part. ``min_arch_version`` says nothing
about any of that, which is why E30 had to leave ``deep_gemm_*`` and
``cuda_attn_merge_state`` unresolved: their gates were *arch* statements
standing in for *artifact* questions.

Three probe shapes were already scattered through the tree, each hand-rolled:

1. an ``inspect.signature`` probe (``"sfq" in signature(flash_attn_varlen_func)``)
   — does the installed build ship this entry point at all;
2. ``try/except ImportError`` plus an ``is error_fn`` sentinel test — did the
   optional dependency bind;
3. a :func:`~unieinfra_kernel.selection.selection.select_kernel` /registry query
   — did some other kernel actually register.

This module generalizes all three into one mechanism —
:func:`register_build_capability` — plus a fourth probe that none of them could
express: :func:`binary_targets_device`, which asks the *object file* whether it
contains SASS or forward-compatible PTX for the running device.

Contract
--------

* A build capability is a feature name in the ``build:`` namespace. It goes in
  ``required_features`` exactly like a hardware feature.
* Probes are **lazy and cached**: nothing here runs during platform detection,
  so declaring one costs nothing until a selection actually needs the answer.
* A probe that raises falls back to its declared ``default``.
* A ``build:`` feature with **no registered probe resolves to available**, with
  a one-time warning. This is deliberate and is the E28 rule: when a capability
  cannot be queried we fall back to the previously-validated behaviour, so a
  detection gap can never *narrow* what already worked on hardware we cannot
  test. A missing probe must not silently delete a datacenter kernel.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "BUILD_NAMESPACE",
    "BuildCapability",
    "binary_targets_device",
    "build_capability_available",
    "build_capabilities",
    "clear_build_capability_cache",
    "has_signature_parameter",
    "kernel_is_registered",
    "register_build_capability",
    "symbol_is_bound",
]

#: Namespace prefix that marks a feature name as a build/artifact capability
#: rather than a hardware capability.
BUILD_NAMESPACE = "build:"


@dataclass(frozen=True)
class BuildCapability:
    """A registered build/artifact capability and the probe that answers it.

    Attributes:
        name: Feature name, always in the ``build:`` namespace.
        probe: Zero-argument callable returning a truthy value when the
            capability is present. Called at most once per process (per cache
            generation) and never during platform detection.
        default: Value used when ``probe`` raises. Set this to ``True`` for a
            capability that is known-good on the hardware we cannot test, so a
            probe failure degrades to the previously-validated behaviour rather
            than silently removing the kernel.
        doc: Human-readable description, surfaced by :func:`build_capabilities`.
    """

    name: str
    probe: Callable[[], object]
    default: bool
    doc: str = ""


_REGISTRY: dict[str, BuildCapability] = {}
_CACHE: dict[str, bool] = {}
_WARNED: set[str] = set()


def register_build_capability(
    name: str,
    probe: Callable[[], object],
    *,
    default: bool,
    doc: str = "",
) -> str:
    """Register a build/artifact capability probe.

    Args:
        name: Feature name. Must start with ``build:``. Re-registering the same
            name replaces the previous probe and drops its cached answer, which
            is what lets a test swap a probe out.
        probe: Zero-argument callable returning a truthy value when the
            installed artifact provides the capability. Evaluated lazily on the
            first selection that needs it, then cached.
        default: The answer used when ``probe`` raises. Choose ``True`` when the
            capability is validated on hardware this process cannot query, so a
            probe failure never narrows a working path (the E28 rule).
        doc: Optional description for diagnostics.

    Returns:
        ``name``, so a registration can be inlined straight into a
        ``required_features`` set::

            required_features=frozenset({
                "memory:tma",
                register_build_capability(
                    "build:deep_gemm_device_code",
                    _probe, default=True,
                ),
            })

    Raises:
        ValueError: If ``name`` is not in the ``build:`` namespace.
    """
    if not name.startswith(BUILD_NAMESPACE):
        raise ValueError(
            f"build capability name must start with {BUILD_NAMESPACE!r}, got {name!r}. "
            "The prefix is what tells CapabilityRequirement to resolve the feature "
            "through the artifact-probe registry instead of the device feature set."
        )
    _REGISTRY[name] = BuildCapability(name=name, probe=probe, default=default, doc=doc)
    _CACHE.pop(name, None)
    _WARNED.discard(name)
    return name


def build_capability_available(name: str) -> bool:
    """Resolve a ``build:`` feature, evaluating and caching its probe.

    Args:
        name: Feature name in the ``build:`` namespace.

    Returns:
        The probe's answer; its ``default`` when the probe raised; or ``True``
        when no probe is registered for ``name``. The unregistered case warns
        once and answers ``True`` on purpose: an unqueryable capability must
        never narrow a path that is already validated on hardware we cannot
        reach (E28).
    """
    cached = _CACHE.get(name)
    if cached is not None:
        return cached

    capability = _REGISTRY.get(name)
    if capability is None:
        if name not in _WARNED:
            _WARNED.add(name)
            logger.warning(
                "Build capability %r has no registered probe; treating it as "
                "AVAILABLE so an undetectable artifact cannot narrow a "
                "previously-validated path. Register a probe with "
                "register_build_capability(%r, ...) to make this decidable.",
                name,
                name,
            )
        return True

    try:
        answer = bool(capability.probe())
    except Exception as exc:  # noqa: BLE001 - a probe must never break selection
        logger.warning(
            "Build capability probe %r raised %s; falling back to default=%s.",
            name,
            exc,
            capability.default,
        )
        answer = capability.default

    _CACHE[name] = answer
    return answer


def build_capabilities() -> dict[str, BuildCapability]:
    """Return every registered build capability, keyed by feature name."""
    return dict(_REGISTRY)


def clear_build_capability_cache() -> None:
    """Drop cached probe answers so the next query re-evaluates them."""
    _CACHE.clear()
    _WARNED.clear()


# ---------------------------------------------------------------------------
# Probe constructors -- the three in-tree patterns, generalized
# ---------------------------------------------------------------------------


def has_signature_parameter(load: Callable[[], object], parameter: str):
    """Probe: does the installed entry point accept ``parameter``?

    Generalizes the ``"sfq" in inspect.signature(flash_attn_varlen_func)`` and
    ``"aux_tensors" in signature(AttentionMask.apply_mask_sm100)`` checks.

    Args:
        load: Zero-argument callable returning the function to inspect. Taking a
            loader rather than the function keeps the import lazy, so a probe on
            an optional dependency costs nothing until it is queried.
        parameter: Parameter name the installed build must expose.

    Returns:
        A zero-argument probe suitable for :func:`register_build_capability`.
    """

    def probe() -> bool:
        import inspect

        return parameter in inspect.signature(load()).parameters

    return probe


def symbol_is_bound(load: Callable[[], object], *, sentinels: tuple = ()):
    """Probe: did the optional dependency actually bind this symbol?

    Generalizes the ``try/except ImportError`` plus ``x is not error_fn``
    pattern used across ``ops/**``.

    Args:
        load: Zero-argument callable returning the symbol. Raising (typically
            ``ImportError``) counts as unbound.
        sentinels: Placeholder objects that mean "not bound" — pass
            ``(error_fn,)`` to reuse the registry's own sentinel. ``None`` is
            always treated as unbound.

    Returns:
        A zero-argument probe suitable for :func:`register_build_capability`.
    """

    def probe() -> bool:
        try:
            symbol = load()
        except Exception:  # noqa: BLE001 - an unimportable symbol is simply absent
            return False
        if symbol is None:
            return False
        return not any(symbol is sentinel for sentinel in sentinels)

    return probe


def kernel_is_registered(kernel_name: str):
    """Probe: did some other kernel register under ``kernel_name``?

    Generalizes the ``select_kernel`` / registry-query pattern, which several
    ``ops/**`` entry points use to decide whether a specialized path exists.

    Args:
        kernel_name: The registered kernel's unique name.

    Returns:
        A zero-argument probe suitable for :func:`register_build_capability`.
    """

    def probe() -> bool:
        from unieinfra_kernel.selection.registry import KernelRegistry

        return KernelRegistry.get().get_by_name(kernel_name) is not None

    return probe


def binary_targets_device(
    locate: Callable[[], str],
    *,
    arch: Callable[[], tuple[int, int]] | None = None,
):
    """Probe: does this object file contain code for the running device?

    This is the question ``min_arch_version`` structurally cannot ask. A
    prebuilt ``.so`` carries a fatbinary whose architectures were fixed at
    compile time; a device newer than every embedded target is unreachable no
    matter how capable it is. ``cuobjdump`` reports both the embedded SASS
    (exact ``sm_XY`` match required) and the embedded PTX (any target ``<=`` the
    device JITs forward, so it counts).

    Args:
        locate: Zero-argument callable returning the path of the object file to
            inspect. Lazy so a missing optional artifact is not an import error.
        arch: Zero-argument callable returning ``(major, minor)`` for the device
            to test. Defaults to the detected platform's arch version.

    Returns:
        A zero-argument probe suitable for :func:`register_build_capability`.
        The probe **raises** when ``cuobjdump`` is unavailable or the file
        cannot be read, so registration's ``default`` decides the answer — keep
        that ``True`` for artifacts validated on datacenter parts.
    """

    def probe() -> bool:
        if arch is None:
            from unieinfra_kernel.selection.platform import current_platform

            platform = current_platform()
            major, minor = platform.arch_version.major, platform.arch_version.minor
        else:
            major, minor = arch()

        cuobjdump = shutil.which("cuobjdump")
        if cuobjdump is None:
            raise RuntimeError("cuobjdump not on PATH; cannot inspect device code")

        path = locate()
        target = major * 10 + minor
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [cuobjdump, "--list-elf", "--list-ptx", path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"cuobjdump failed on {path}: {result.stderr.strip() or 'unknown error'}"
            )
        return _fatbin_covers(result.stdout, target)

    return probe


def _fatbin_covers(listing: str, target: int) -> bool:
    """Whether a ``cuobjdump`` listing covers compute capability ``target``.

    Args:
        listing: Combined ``--list-elf --list-ptx`` output.
        target: Device compute capability as ``major * 10 + minor``.

    Returns:
        ``True`` when the listing names an ELF image for exactly ``target`` (an
        exact SASS match) or a PTX image for ``target`` or lower (JIT-forward
        compatible within the same major family).
    """
    import re

    elf_archs: set[int] = set()
    ptx_archs: set[int] = set()
    for line in listing.splitlines():
        for match in re.finditer(r"sm_(\d+)", line):
            value = int(match.group(1))
            if ".ptx" in line:
                ptx_archs.add(value)
            else:
                elf_archs.add(value)
    if target in elf_archs:
        return True
    # PTX JITs forward only within a major family: sm_90 PTX does not run on
    # sm_100, but sm_100 PTX runs on sm_103.
    return any(ptx <= target and ptx // 10 == target // 10 for ptx in ptx_archs)
