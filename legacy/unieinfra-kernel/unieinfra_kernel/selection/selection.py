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

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator

from unieinfra_kernel.selection.autotune import (
    AutotuneCache,
    AutotuneParams,
    Measurement,
    autotune_available,
    autotune_cache,
    measure_candidates,
    shape_bucket,
)
from unieinfra_kernel.selection.inputs import (
    get_benchmark_shapes,
    get_shape_traits_fn,
)
from unieinfra_kernel.selection.platform import PlatformInfo, current_platform
from unieinfra_kernel.selection.registry import KernelRegistry, KernelSpec
from unieinfra_kernel.selection.signature import FormatSignature
from unieinfra_kernel.selection.tuning import autotune_frozen, freeze_autotuning

logger = logging.getLogger(__name__)


def _env(suffix: str) -> str | None:
    """Read a selection env var, preferring the ``UNIEINFRA_KERNEL_`` prefix.

    The ported core keeps honouring the historical ``TOKENSPEED_KERNEL_``
    spelling so an operator's existing override scripts keep working on either
    base.

    Args:
        suffix: Variable name without the package prefix, e.g. ``"VERBOSE"``.

    Returns:
        The first non-empty value found, or ``None``.
    """
    for prefix in ("UNIEINFRA_KERNEL_", "TOKENSPEED_KERNEL_"):
        value = os.environ.get(prefix + suffix)
        if value:
            return value
    return None


__all__ = [
    "NoKernelFoundError",
    "SelectedKernel",
    "SelectionObjective",
    "SelectionStrategy",
    "ScoreBreakdown",
    "SelectionOracle",
    "AutotuneParams",
    "SelectionPin",
    "SelectionPolicy",
    "autotune_frozen",
    "autotune_log",
    "clear_autotune_log",
    "clear_kernel_pin",
    "enable_autotuning",
    "freeze_autotuning",
    "select_kernel",
    "set_kernel_pin",
    "set_selection_policy",
    "get_selection_policy",
    "register_oracle",
    "kernel_override",
    "pin_kernel",
    "load_config_overrides",
    "clear_config_overrides",
    "explain_selection",
    "spec_matches_traits",
    "ref_compatible_with_spec",
    "spec_matches_shape_traits",
    "warmup_selection",
]


class NoKernelFoundError(RuntimeError):
    """Raised when no kernel matches the requested operation."""

    pass


class SelectedKernel:
    """Result of kernel selection — a callable that also carries the kernel name."""

    __slots__ = ("name", "impl")

    def __init__(self, name: str, impl: Callable) -> None:
        self.name = name
        self.impl = impl

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.impl(*args, **kwargs)

    def __repr__(self) -> str:
        return f"SelectedKernel(name={self.name!r})"


class SelectionObjective(Enum):
    """Objectives are a closed enum — they directly control scoring logic,
    so an unknown value would silently fall through with no effect.
    """

    DEFAULT = "default"  # Balanced heuristic (priority-weighted)
    LATENCY = "latency"  # Minimize per-call latency
    THROUGHPUT = "throughput"  # Maximize tokens/second for large batches
    PORTABILITY = "portability"  # Prefer solutions that work across vendors (Triton)
    DETERMINISM = "determinism"  # Prefer bit-reproducible implementations
    DEBUG = "debug"  # Prefer readable implementations (reference, Triton)


class SelectionStrategy(Enum):
    HEURISTIC = "heuristic"  # Score-based ranking (default, instant)
    AUTOTUNE = "autotune"  # Benchmark candidates, pick fastest


@dataclass
class ScoreBreakdown:
    """Per-kernel scoring breakdown across all dimensions.

    Ranking is lexicographic on ``(oracle, objective, priority, origin)`` — the
    oracle's per-family knowledge wins first, then objective alignment, then the
    kernel's declared priority band, and finally ``origin`` as a stable,
    lowest-weight tiebreaker. ``origin`` only ever decides an otherwise-exact tie
    and, because every in-tree kernel shares ``origin="vendor"`` (rank 0), it
    cannot reorder the existing corpus — it only lets an out-of-tree kernel
    (``first_party``/``kda``/``contrib``) win a tie it would previously have lost
    to registration order.
    """

    priority: int  # [0, 20) — from KernelSpec.priority
    objective: int  # 0 or 1 — 1 if the kernel matches the requested objective
    oracle: int  # [0, 20) — per-family oracle adjustment
    origin: int = 0  # provenance rank; final tiebreaker (see _ORIGIN_RANK)

    def sort_key(self) -> tuple[int, int, int, int]:
        """Lex sort key (descending — higher is better)."""
        return (self.oracle, self.objective, self.priority, self.origin)

    def __str__(self) -> str:
        return (
            f"ora={self.oracle} obj={self.objective} "
            f"pri={self.priority} org={self.origin}"
        )


class SelectionOracle:
    """Base class for per-family selection adjustments.

    Return a score in [0, 20). 10 = neutral. Higher = better fit.
    """

    def adjust(
        self,
        spec: KernelSpec,
        platform: PlatformInfo,
        traits: dict[str, Any] | None,
    ) -> int:
        return 10


@dataclass
class SelectionPolicy:
    """Per-op selection strategy configuration."""

    # Default strategy for all ops
    default_strategy: SelectionStrategy = SelectionStrategy.HEURISTIC

    # Per-op overrides: (family, mode) -> strategy
    op_strategies: dict[tuple[str, str], SelectionStrategy] = field(
        default_factory=dict
    )

    # Autotune parameters (used when strategy is AUTOTUNE)
    autotune_params: AutotuneParams = field(default_factory=AutotuneParams)

    def get_strategy(self, family: str, mode: str) -> SelectionStrategy:
        return self.op_strategies.get((family, mode), self.default_strategy)


@dataclass
class _ConfigOverrideEntry:
    """A single override entry parsed from overrides.yaml."""

    name: str | None = None  # Exact kernel name
    solution: str | None = None  # Solution backend to match
    objective: str | None = None  # SelectionObjective value string


@dataclass(frozen=True)
class SelectionPin:
    """An ablation pin: restrict selection to kernels matching these fields.

    Unlike an override (which names one kernel and bypasses the pipeline), a pin
    *filters the candidate set* and lets normal capability / signature / trait
    filtering and ranking run on what is left. That is what a benchmark ablation
    needs: "select as usual, but only among first-party kernels" is a different
    question from "call this exact function", and only the first one is still a
    selection.

    An empty pin (every field ``None``) matches everything and is a no-op.

    Attributes:
        name: Exact kernel name.
        solution: Backend, e.g. ``"triton"`` or ``"reference"``.
        origin: Provenance, e.g. ``"first_party"``, ``"kda"``, ``"vendor"``.
    """

    name: str | None = None
    solution: str | None = None
    origin: str | None = None

    def is_empty(self) -> bool:
        """Whether this pin constrains nothing."""
        return self.name is None and self.solution is None and self.origin is None

    def matches(self, spec: KernelSpec) -> bool:
        """Whether ``spec`` satisfies every field this pin constrains."""
        if self.name is not None and spec.name != self.name:
            return False
        if self.solution is not None and spec.solution != self.solution:
            return False
        if self.origin is not None and getattr(spec, "origin", "vendor") != self.origin:
            return False
        return True

    def __str__(self) -> str:
        parts = [
            f"{field}={value!r}"
            for field, value in (
                ("name", self.name),
                ("solution", self.solution),
                ("origin", self.origin),
            )
            if value is not None
        ]
        return "pin(" + ", ".join(parts) + ")" if parts else "pin(any)"


_policy = SelectionPolicy()
_oracles: dict[str, SelectionOracle] = {}
_global_overrides: dict[tuple[str, str], str] = {}
_config_overrides: dict[tuple[str, str], _ConfigOverrideEntry] | None = None
# (family, mode) -> pin, plus ``None`` for the process-wide default pin.
_pins: dict[tuple[str, str] | None, SelectionPin] = {}


def set_selection_policy(policy: SelectionPolicy) -> None:
    """Set per-op selection strategy. Clears all cached selections."""
    global _policy
    _policy = policy
    KernelRegistry.get().clear_cache()


def get_selection_policy() -> SelectionPolicy:
    """Return the active selection policy."""
    return _policy


def enable_autotuning(
    ops: list[tuple[str, str]] | None = None,
    *,
    params: AutotuneParams | None = None,
) -> SelectionPolicy:
    """Switch selection to measurement for some (or all) operators.

    Args:
        ops: ``(family, mode)`` pairs to autotune. ``None`` makes ``AUTOTUNE``
            the default for every operator.
        params: Timing knobs; the defaults are what
            :class:`~unieinfra_kernel.selection.autotune.AutotuneParams`
            documents.

    Returns:
        The policy that was installed, so a caller can restore the old one.
    """
    policy = SelectionPolicy(
        default_strategy=(
            SelectionStrategy.AUTOTUNE if ops is None else SelectionStrategy.HEURISTIC
        ),
        op_strategies={op: SelectionStrategy.AUTOTUNE for op in (ops or ())},
        autotune_params=params or AutotuneParams(),
    )
    set_selection_policy(policy)
    return policy


def set_kernel_pin(
    pin: SelectionPin | None,
    family: str | None = None,
    mode: str | None = None,
) -> None:
    """Install an ablation pin, process-wide or for one operator.

    Args:
        pin: The pin, or ``None`` to clear.
        family: Operator family; ``None`` (with ``mode``) sets the global pin.
        mode: Operator mode.

    Raises:
        ValueError: If exactly one of ``family``/``mode`` is given.
    """
    if (family is None) != (mode is None):
        raise ValueError("pass both family and mode, or neither")
    key: tuple[str, str] | None = None if family is None else (family, str(mode))
    if pin is None or pin.is_empty():
        _pins.pop(key, None)
    else:
        _pins[key] = pin
    KernelRegistry.get().clear_cache()


def clear_kernel_pin() -> None:
    """Remove every pin."""
    _pins.clear()
    KernelRegistry.get().clear_cache()


def _parse_pin(raw: str) -> SelectionPin:
    """Parse a pin spec: ``name=``/``solution=``/``origin=``, or a bare kernel name."""
    fields: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, _, value = part.partition("=")
            key = key.strip().lower()
            if key in ("name", "solution", "origin"):
                fields[key] = value.strip()
                continue
            logger.warning("ignoring unknown pin field %r", key)
            continue
        fields["name"] = part
    return SelectionPin(**fields)  # type: ignore[arg-type]


def _active_pin(family: str, mode: str) -> SelectionPin | None:
    """Return the pin in force for an operator, env vars included.

    Precedence (highest first): ``UNIEINFRA_KERNEL_PIN_<FAMILY>_<MODE>``,
    ``UNIEINFRA_KERNEL_PIN``, a per-op programmatic pin, the global
    programmatic pin.
    """
    env_op = _env(f"PIN_{family.upper()}_{mode.upper()}")
    if env_op:
        return _parse_pin(env_op)
    env_all = _env("PIN")
    if env_all:
        return _parse_pin(env_all)
    return _pins.get((family, mode)) or _pins.get(None)


@contextmanager
def pin_kernel(
    family: str | None = None,
    mode: str | None = None,
    *,
    name: str | None = None,
    solution: str | None = None,
    origin: str | None = None,
) -> Generator[None, None, None]:
    """Scoped ablation pin — restrict selection without bypassing it.

    Use this from a benchmark harness to hold one axis fixed, e.g.::

        with pin_kernel(origin="vendor"):      # the incumbent arm
            run()
        with pin_kernel(origin="first_party"):  # our kernels only
            run()

    Args:
        family: Operator family, or ``None`` to pin every operator.
        mode: Operator mode; required when ``family`` is given.
        name: Restrict to this exact kernel name.
        solution: Restrict to this backend.
        origin: Restrict to this provenance.

    Yields:
        ``None``, with the pin in force.
    """
    key: tuple[str, str] | None = None if family is None else (family, str(mode))
    previous = _pins.get(key)
    set_kernel_pin(
        SelectionPin(name=name, solution=solution, origin=origin), family, mode
    )
    try:
        yield
    finally:
        set_kernel_pin(previous, family, mode)


def register_oracle(family: str, oracle: SelectionOracle) -> None:
    """Register a per-family selection oracle."""
    _oracles[family] = oracle


def _get_oracle(family: str) -> SelectionOracle | None:
    return _oracles.get(family)


def _parse_overrides(
    raw: dict,
) -> dict[tuple[str, str], _ConfigOverrideEntry]:
    """Parse the ``overrides`` section of the YAML config.

    Accepted formats::

        overrides:
          attention.decode:
            solution: flashinfer
          gemm.mm:
            name: gluon_gemm_mm_fp8
          moe.experts:
            objective: determinism
          norm.rmsnorm: triton_rmsnorm   # shorthand: treated as kernel name
    """
    result: dict[tuple[str, str], _ConfigOverrideEntry] = {}
    if not isinstance(raw, dict):
        return result

    for op_key, entry in raw.items():
        parts = str(op_key).split(".", 1)
        if len(parts) != 2:
            logger.warning("Invalid override key '%s' (expected 'family.mode')", op_key)
            continue
        family, mode = parts

        if isinstance(entry, str):
            result[(family, mode)] = _ConfigOverrideEntry(name=entry)
        elif isinstance(entry, dict):
            name = entry.get("name")
            solution = entry.get("solution")
            objective = entry.get("objective")
            result[(family, mode)] = _ConfigOverrideEntry(
                name=str(name) if name else None,
                solution=str(solution) if solution else None,
                objective=str(objective) if objective else None,
            )
        else:
            logger.warning("Invalid override value for '%s': %r", op_key, entry)

    return result


def load_config_overrides(path: str | os.PathLike[str] | None = None) -> None:
    """Load kernel overrides from a YAML config file.

    Args:
        path: Path to the YAML file.  If *None*, uses the
            ``UNIEINFRA_KERNEL_OVERRIDES_FILE`` env var or falls back to
            ``~/.config/unieinfra-kernel/overrides.yaml``.
    """
    global _config_overrides

    if path is None:
        env_path = _env("OVERRIDES_FILE")
        if env_path:
            path = Path(env_path)
        else:
            path = Path("~/.config/unieinfra-kernel/overrides.yaml").expanduser()
    else:
        path = Path(path)

    _config_overrides = {}

    if not path.exists():
        return

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "PyYAML not installed; cannot load overrides from %s. "
            "Install with: pip install pyyaml",
            path,
        )
        return

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception:
        logger.warning("Failed to load overrides from %s", path, exc_info=True)
        return

    if not isinstance(data, dict):
        return

    _config_overrides = _parse_overrides(data.get("overrides", {}))

    if _config_overrides:
        logger.debug(
            "Loaded %d config override(s) from %s",
            len(_config_overrides),
            path,
        )
        KernelRegistry.get().clear_cache()


def clear_config_overrides() -> None:
    """Clear loaded config overrides.

    After this call no config-file overrides are active.  Call
    :func:`load_config_overrides` again to reload from a file.
    """
    global _config_overrides
    _config_overrides = {}


def _get_config_override(family: str, mode: str) -> _ConfigOverrideEntry | None:
    """Return the config-file override for *(family, mode)*, lazily loading
    from the default path on first access."""
    global _config_overrides
    if _config_overrides is None:
        load_config_overrides()
    return _config_overrides.get((family, mode))  # type: ignore[union-attr]


def _make_cache_key(
    family: str,
    mode: str,
    format_signature: FormatSignature,
    arch: str,
    objective: SelectionObjective,
    features: frozenset[str] | None,
    traits: dict[str, Any] | None,
    solution: str | None = None,
    shape_key: tuple = (),
    pin: SelectionPin | None = None,
) -> tuple:
    """Build a hashable cache key including selection-relevant traits.

    ``shape_key`` is the autotune shape bucket and is empty on the heuristic
    path, so turning autotuning on for one operator does not change the cache
    behaviour of any other.
    """
    traits_key = tuple(sorted(traits.items())) if traits else ()
    mods_key = frozenset(features) if features else frozenset()
    return (
        family,
        mode,
        format_signature,
        arch,
        objective,
        mods_key,
        traits_key,
        solution,
        shape_key,
        pin,
    )


def _score_priority(spec: KernelSpec) -> int:
    """Priority dimension: kernel's inherent quality/maturity."""
    return max(0, min(19, spec.priority))


# Provenance ranks for the lowest-weight selection tiebreaker. Every in-tree /
# upstream kernel keeps the default ``"vendor"`` (rank 0), so this dimension is
# constant across the existing corpus and can never reorder two existing
# kernels. Out-of-tree packages set a non-vendor origin to win an exact tie:
# first-party and KDA-generated kernels outrank community contributions, which
# in turn outrank the vendor baseline.
_ORIGIN_RANK: dict[str, int] = {
    "vendor": 0,
    "contrib": 1,
    "kda": 2,
    "first_party": 3,
}


def _score_origin(spec: KernelSpec) -> int:
    """Origin dimension: provenance-based final tiebreaker.

    Returns 0 for the ``"vendor"`` default (and any unknown origin), which keeps
    the entire existing corpus on a single rank. Only used to break an exact
    ``(oracle, objective, priority)`` tie, so it never changes a selection whose
    winner was decided on any earlier dimension.
    """
    return _ORIGIN_RANK.get(getattr(spec, "origin", "vendor"), 0)


_OBJECTIVE_TAG: dict[SelectionObjective, str] = {
    SelectionObjective.LATENCY: "latency",
    SelectionObjective.THROUGHPUT: "throughput",
    SelectionObjective.PORTABILITY: "portability",
    SelectionObjective.DETERMINISM: "determinism",
    SelectionObjective.DEBUG: "determinism",
}


def _score_objective(spec: KernelSpec, objective: SelectionObjective) -> int:
    """Objective dimension: 1 if the kernel declares the matching tag, else 0.

    DEFAULT returns 0 so every kernel ties on this dimension and ranking
    falls through to oracle/priority.
    """
    tag = _OBJECTIVE_TAG.get(objective)
    return 1 if tag is not None and tag in spec.tags else 0


def _score_oracle(
    spec: KernelSpec,
    platform: PlatformInfo,
    traits: dict[str, Any] | None,
) -> int:
    """Oracle dimension: per-family domain-specific scoring."""
    oracle = _get_oracle(spec.family)
    if oracle is None:
        return 10  # Neutral when no oracle is registered
    score = oracle.adjust(spec, platform, traits)
    return max(0, min(19, score))


def _score(
    spec: KernelSpec,
    objective: SelectionObjective,
    platform: PlatformInfo,
    traits: dict[str, Any] | None,
) -> ScoreBreakdown:
    """Score a kernel across all ranking dimensions."""
    return ScoreBreakdown(
        priority=_score_priority(spec),
        objective=_score_objective(spec, objective),
        oracle=_score_oracle(spec, platform, traits),
        origin=_score_origin(spec),
    )


def _rank_by_objective(
    specs: list[KernelSpec],
    objective: SelectionObjective,
    platform: PlatformInfo,
    traits: dict[str, Any] | None,
) -> list[tuple[KernelSpec, ScoreBreakdown]]:
    """Rank kernels lexicographically by (oracle, objective, priority, origin).

    Higher is better. Oracle wins first because per-family oracles encode the
    most domain knowledge; objective alignment breaks ties next; the kernel's
    declared priority band comes after; and ``origin`` (provenance) is the
    lowest-weight tiebreaker, constant across the vendor corpus.
    """
    scored = [(spec, _score(spec, objective, platform, traits)) for spec in specs]
    scored.sort(key=lambda x: x[1].sort_key(), reverse=True)
    return scored


def _trait_value_matches(spec_values: frozenset[Any], trait_value: Any) -> bool:
    if not isinstance(trait_value, (set, frozenset)):
        trait_value = frozenset({trait_value})
    return trait_value.issubset(spec_values)


def _ispp_satisfies_alignment(spec: KernelSpec, ispp: Any) -> bool:
    alignments = spec.traits.get("ispp_alignment")
    if alignments is None:
        return True
    try:
        ispp_value = int(ispp)
    except (TypeError, ValueError):
        return False
    return any(
        int(alignment) > 0 and ispp_value % int(alignment) == 0
        for alignment in alignments
    )


def spec_matches_traits(
    spec: KernelSpec,
    traits: dict[str, Any],
    *,
    require_all_traits: bool = False,
) -> bool:
    """Return whether a spec's declared traits match the requested traits.

    Args:
        spec: Registered kernel specification to test.
        traits: Trait requirements. Values may be concrete scalars (for example,
            ``{"head_dim": 128}``) or sets/frozensets of allowed values.
        require_all_traits: When ``False`` (selection behavior), unknown traits on
            the spec are ignored. When ``True`` (reference compatibility checks),
            every requested trait must be explicitly present on the spec.
    """
    for trait_name, trait_value in traits.items():
        # ispp stands for "intermediate size per partition" and has special
        # alignment requirements that depend on the kernel's declared
        # supported alignments (if any). It is used in some MoE ops to ensure
        # the intermediate buffer sizes are compatible with the kernel's
        # requirements.
        if trait_name == "ispp":
            if not _ispp_satisfies_alignment(spec, trait_value):
                return False
            continue

        spec_values = spec.traits.get(trait_name)
        if spec_values is None:
            if require_all_traits:
                return False
            continue
        if not _trait_value_matches(spec_values, trait_value):
            return False
    return True


def ref_compatible_with_spec(ref: KernelSpec, spec: KernelSpec) -> bool:
    """Return whether a reference kernel can handle the same inputs as a test kernel.

    For each trait the reference declares, the spec must declare that same trait
    with values that fully cover the reference's required values.  Traits the
    reference does not declare are unconstrained (the reference is general with
    respect to those traits).
    """
    for trait_name, ref_values in ref.traits.items():
        spec_values = spec.traits.get(trait_name)
        if spec_values is None:
            return False
        if not ref_values.issubset(spec_values):
            return False
    return True


def spec_matches_shape_traits(spec: KernelSpec, shape: dict[str, Any]) -> bool:
    """Return whether a spec's alignment traits match a concrete shape."""
    alignment_traits: dict[str, tuple[str, int]] = {
        "n_align_16": ("N", 16),
        "n_align_64": ("N", 64),
        "n_align_128": ("N", 128),
        "k_align_16": ("K", 16),
        "k_align_32": ("K", 32),
        "k_align_64": ("K", 64),
        "k_align_128": ("K", 128),
    }
    for trait_name, (dim_name, alignment) in alignment_traits.items():
        values = spec.traits.get(trait_name)
        if values is None or True not in values:
            continue

        dim = shape.get(dim_name)
        if isinstance(dim, int) and dim % alignment != 0:
            return False

    minimum_traits: dict[str, tuple[str, int]] = {
        "n_min_128": ("N", 128),
        "k_min_128": ("K", 128),
        "m_min_16": ("M", 16),
    }
    for trait_name, (dim_name, minimum) in minimum_traits.items():
        values = spec.traits.get(trait_name)
        if values is None or True not in values:
            continue

        dim = shape.get(dim_name)
        if isinstance(dim, int) and dim < minimum:
            return False

    # Upper-bound buckets, used by decode-shaped kernels that are only valid for
    # a skinny M / narrow N slice of an operator.
    maximum_traits: dict[str, tuple[str, int]] = {
        "m_le_8": ("M", 8),
        "m_le_256": ("M", 256),
        "n_le_16384": ("N", 16384),
        "n_le_32768": ("N", 32768),
    }
    for trait_name, (dim_name, maximum) in maximum_traits.items():
        values = spec.traits.get(trait_name)
        if values is None or True not in values:
            continue

        dim = shape.get(dim_name)
        if isinstance(dim, int) and dim > maximum:
            return False

    return True


def _filter_by_traits(
    specs: list[KernelSpec],
    traits: dict[str, Any],
) -> list[KernelSpec]:
    """Filter kernels by op-specific trait compatibility."""
    return [spec for spec in specs if spec_matches_traits(spec, traits)]


def _resolve_override(
    registry: KernelRegistry,
    family: str,
    mode: str,
    format_signature: object,
    override: str,
    platform: PlatformInfo,
) -> SelectedKernel:
    impl = registry.get_impl(override)
    if impl is not None:
        return SelectedKernel(name=override, impl=impl)

    specs = registry.get_for_operator(family, mode, solution=override)
    if specs:
        kernel_name = specs[0].name
        impl = registry.get_impl(kernel_name)
        if impl is not None:
            return SelectedKernel(name=kernel_name, impl=impl)

    raise NoKernelFoundError(
        f"Override '{override}' not found for {family}.{mode} ({format_signature})"
    )


def _log_selection(
    family: str,
    mode: str,
    format_signature: object,
    winner: KernelSpec,
    scored: list[tuple[KernelSpec, ScoreBreakdown]],
    platform: PlatformInfo,
    objective: SelectionObjective,
) -> None:
    """Log selection result if verbose mode is enabled."""
    if not _env("VERBOSE"):
        return

    breakdown = next((s for spec, s in scored if spec.name == winner.name), None)
    if breakdown:
        logger.info(
            "[unieinfra_kernel] %s.%s(%s) -> %s (%s, %s)",
            family,
            mode,
            format_signature,
            winner.name,
            breakdown,
            platform.arch,
        )
    else:
        logger.info(
            "[unieinfra_kernel] %s.%s(%s) -> %s (%s)",
            family,
            mode,
            format_signature,
            winner.name,
            platform.arch,
        )


def select_kernel(
    family: str,
    mode: str,
    format_signature: FormatSignature,
    *,
    features: frozenset[str] | None = None,
    platform: PlatformInfo | None = None,
    objective: SelectionObjective = SelectionObjective.DEFAULT,
    traits: dict[str, Any] | None = None,
    solution: str | None = None,
    override: str | None = None,
    shape: dict[str, Any] | None = None,
    strategy: SelectionStrategy | None = None,
) -> SelectedKernel:
    """Select the best kernel for an operation.

    On first call for a given (family, mode, format_signature, platform,
    objective, traits, solution) combination, runs the full selection pipeline.
    Subsequent calls
    with the same arguments return the cached result — a single dict lookup.

    Args:
        family: Operator family (e.g., "attention")
        mode: Operator mode (e.g., "decode")
        format_signature: Role-indexed tensor format signature
        features: Required operator features (e.g., {"paged"})
        platform: Hardware to match (auto-detected if None)
        objective: Selection objective (see SelectionObjective enum)
        traits: Op-specific trait values that affect kernel applicability
               (e.g., {"head_dim": 128, "num_kv_heads": 8})
        solution: Restrict selection to a registered solution while preserving
            normal platform, format signature, and trait filtering.
        override: Force a specific kernel name or solution string
        shape: Concrete shape of the request, e.g.
            ``{"M": 1, "N": 6144, "K": 4096}``. Only the ``AUTOTUNE`` strategy
            reads it — it is what makes measuring on *real* shapes possible —
            and it enters the memo key as a coarse bucket
            (:func:`~unieinfra_kernel.selection.autotune.shape_bucket`), never
            as the raw dimensions.
        strategy: Force a strategy for this call, ignoring the policy. Used by
            load-time claimability probes, which must never trigger tuning.

    Returns:
        A :class:`SelectedKernel` that is directly callable and also
        exposes the winning kernel's ``name``.
    """
    platform = platform or current_platform()

    # --- Override resolution (lowest → highest priority) ---

    # 1. Config file (lowest priority)
    config_entry = _get_config_override(family, mode)
    if config_entry is not None:
        if override is None and solution is None:
            if config_entry.name:
                override = config_entry.name
            elif config_entry.solution:
                solution = config_entry.solution
        if config_entry.objective and objective == SelectionObjective.DEFAULT:
            try:
                objective = SelectionObjective(config_entry.objective)
            except ValueError:
                logger.warning(
                    "Invalid objective '%s' in overrides config for %s.%s",
                    config_entry.objective,
                    family,
                    mode,
                )

    # 2. Context-manager global overrides
    global_override = _global_overrides.get((family, mode))
    if global_override:
        override = global_override

    # 3. Environment variable
    env_override = _env(f"OVERRIDE_{family.upper()}_{mode.upper()}")
    if env_override:
        override = env_override
    registry = KernelRegistry.get()

    # Strategy is resolved before the memo key because AUTOTUNE is the only
    # strategy that reads `shape`, and we do not want a shape dimension in the
    # key of every heuristic lookup.
    if strategy is None:
        strategy = _policy.get_strategy(family, mode)
    pin = _active_pin(family, mode)

    # Fast path: check cache (skipped when override is active)
    cache_key = _make_cache_key(
        family,
        mode,
        format_signature,
        platform.arch,
        objective,
        features,
        traits,
        solution,
        shape_bucket(shape) if strategy == SelectionStrategy.AUTOTUNE else (),
        pin,
    )
    if override is None:
        cached = registry.cache_get(cache_key)
        if cached is not None:
            return cached

    if override:
        return _resolve_override(
            registry, family, mode, format_signature, override, platform
        )

    # Get candidates (same filtering for both strategies)
    candidates = registry.get_for_operator(
        family,
        mode,
        features=features,
        platform=platform,
        format_signature=format_signature,
        solution=solution,
    )
    if pin is not None:
        candidates = [spec for spec in candidates if pin.matches(spec)]

    solution_clause = f" with solution {solution!r}" if solution else ""
    if not candidates:
        raise NoKernelFoundError(
            f"No kernel found for {family}.{mode} ({format_signature})"
            f"{solution_clause} on {platform.device_name}"
        )

    if traits:
        candidates = _filter_by_traits(candidates, traits)

    if not candidates:
        raise NoKernelFoundError(
            f"No kernel found for {family}.{mode} ({format_signature})"
            f"{solution_clause} with traits {traits} on {platform.device_name}"
        )

    # Strategy dispatch
    if strategy == SelectionStrategy.AUTOTUNE:
        winner, scored = _autotune_select(
            candidates,
            family,
            mode,
            format_signature,
            platform,
            objective,
            traits,
            shape,
            _policy.autotune_params,
            registry,
        )
    else:
        scored = _rank_by_objective(candidates, objective, platform, traits)
        winner = scored[0][0]

    _log_selection(family, mode, format_signature, winner, scored, platform, objective)

    impl = registry.get_impl(winner.name)
    result = SelectedKernel(name=winner.name, impl=impl)
    registry.cache_put(cache_key, result)
    return result


# Ordered record of every autotune decision this process made, for the
# benchmark harness and the experiments log. Measurement disagreeing with the
# heuristic is the interesting output, so it is recorded rather than inferred.
_autotune_log: list[dict[str, Any]] = []


def autotune_log() -> list[dict[str, Any]]:
    """Return a copy of every autotune decision made in this process.

    Each entry carries the operator, the shape, the heuristic winner, the
    measured winner, whether they disagreed, the per-candidate median
    microseconds (``None`` for a candidate that raised and was dropped), and
    whether the answer came from the disk cache.
    """
    return list(_autotune_log)


def clear_autotune_log() -> None:
    """Drop the recorded autotune decisions."""
    _autotune_log.clear()


def _heuristic_fallback(
    candidates: list[KernelSpec],
    objective: SelectionObjective,
    platform: PlatformInfo,
    traits: dict[str, Any] | None,
    reason: str,
    family: str,
    mode: str,
    format_signature: object,
) -> tuple[KernelSpec, list[tuple[KernelSpec, ScoreBreakdown]]]:
    """Rank heuristically and say why measurement did not happen."""
    scored = _rank_by_objective(candidates, objective, platform, traits)
    logger.debug(
        "[unieinfra_kernel:autotune] %s.%s(%s): %s; using heuristic winner %s",
        family,
        mode,
        format_signature,
        reason,
        scored[0][0].name,
    )
    return scored[0][0], scored


def _autotune_select(
    candidates: list[KernelSpec],
    family: str,
    mode: str,
    format_signature: object,
    platform: PlatformInfo,
    objective: SelectionObjective,
    traits: dict[str, Any] | None,
    shape: dict[str, Any] | None,
    params: AutotuneParams,
    registry: KernelRegistry,
) -> tuple[KernelSpec, list[tuple[KernelSpec, ScoreBreakdown]]]:
    """Measure every surviving candidate on a real shape and keep the fastest.

    The heuristic ranking is still computed, for three reasons: it is the answer
    when measurement is impossible or frozen, it is the tiebreaker when two
    kernels measure within ``params.margin`` of each other, and comparing the
    two is how a disagreement gets reported instead of silently applied.

    Args:
        candidates: Specs that survived capability/signature/trait filtering.
        family: Operator family.
        mode: Operator mode.
        format_signature: The request's format signature.
        platform: Detected hardware.
        objective: The caller's objective. Honoured — the previous
            implementation hardcoded ``DEFAULT`` here and silently discarded it,
            which made ``AUTOTUNE`` strictly worse than ``HEURISTIC``.
        traits: The request's trait values.
        shape: Concrete shape to measure on. Without one there is nothing to
            time, and the heuristic answer stands.
        params: Timing knobs.
        registry: Registry to resolve implementations from.

    Returns:
        ``(winner, heuristic_scored)`` — the same pair the heuristic path
        returns, so the caller does not care which strategy ran.
    """
    scored = _rank_by_objective(candidates, objective, platform, traits)
    heuristic_winner = scored[0][0]
    by_name = {spec.name: spec for spec in candidates}

    def fallback(reason: str):
        return _heuristic_fallback(
            candidates,
            objective,
            platform,
            traits,
            reason,
            family,
            mode,
            format_signature,
        )

    if len(candidates) == 1:
        return heuristic_winner, scored
    if not shape:
        return fallback("no concrete shape supplied")

    # Consult the disk cache before deciding whether we are even allowed to
    # measure: a cached decision is exactly what makes the second startup fast,
    # and it stays valid after the freeze.
    cache = autotune_cache(platform) if params.use_cache else None
    cache_key = (
        AutotuneCache.make_key(platform, family, mode, format_signature, traits, shape)
        if cache is not None
        else ""
    )
    if cache is not None:
        record = cache.get(cache_key)
        if record is not None:
            cached_name = record.get("kernel")
            cached_spec = by_name.get(cached_name)
            if cached_spec is not None:
                _autotune_log.append(
                    {
                        "op": f"{family}.{mode}",
                        "shape": dict(shape),
                        "heuristic": heuristic_winner.name,
                        "measured": cached_name,
                        "disagreed": cached_name != heuristic_winner.name,
                        "times_us": record.get("times_us", {}),
                        "source": "disk-cache",
                    }
                )
                return cached_spec, scored
            # The cached winner is no longer a candidate (registry changed,
            # kernel removed). Fall through and re-measure.

    if autotune_frozen():
        return fallback("autotuning frozen for serving")
    if not autotune_available():
        return fallback("no CUDA device to measure on")

    impls: list[tuple[KernelSpec, Any]] = []
    for spec in candidates:
        impl = registry.get_impl(spec.name)
        if impl is not None:
            impls.append((spec, impl))
    if len(impls) < 2:
        return fallback("fewer than two runnable candidates")

    try:
        measurements: list[Measurement] = measure_candidates(
            impls, family, mode, format_signature, shape, params
        )
    except RuntimeError as exc:
        return fallback(str(exc))
    except Exception as exc:  # noqa: BLE001 - tuning must never break selection
        logger.warning(
            "[unieinfra_kernel:autotune] measurement failed for %s.%s(%s): %s",
            family,
            mode,
            format_signature,
            exc,
            exc_info=True,
        )
        return fallback("measurement raised")

    timed = [m for m in measurements if m.ok]
    if not timed:
        return fallback("every candidate failed to run")

    best = min(timed, key=lambda m: m.median_us)  # type: ignore[arg-type]
    winner = by_name.get(best.name, heuristic_winner)

    # Stability: only overturn the heuristic when the measurement is decisive.
    # Without a margin, run-to-run noise picks the winner and the disk cache
    # then freezes that coin flip for the life of the machine.
    heuristic_time = next(
        (m.median_us for m in timed if m.name == heuristic_winner.name), None
    )
    if (
        heuristic_time is not None
        and winner.name != heuristic_winner.name
        and best.median_us is not None
        and heuristic_time <= best.median_us * (1.0 + params.margin)
    ):
        winner = heuristic_winner

    times = {m.name: m.median_us for m in measurements}
    dropped = [m.name for m in measurements if not m.ok]
    if dropped:
        logger.info(
            "[unieinfra_kernel:autotune] %s.%s %s: dropped %s (raised at runtime)",
            family,
            mode,
            shape,
            ", ".join(dropped),
        )
    if winner.name != heuristic_winner.name:
        logger.info(
            "[unieinfra_kernel:autotune] %s.%s %s: measurement overrides heuristic "
            "%s -> %s (%s)",
            family,
            mode,
            shape,
            heuristic_winner.name,
            winner.name,
            ", ".join(
                f"{name}={value:.1f}us" if value is not None else f"{name}=FAILED"
                for name, value in times.items()
            ),
        )

    _autotune_log.append(
        {
            "op": f"{family}.{mode}",
            "shape": dict(shape),
            "heuristic": heuristic_winner.name,
            "measured": winner.name,
            "disagreed": winner.name != heuristic_winner.name,
            "times_us": times,
            "dropped": dropped,
            "source": "measured",
        }
    )

    if cache is not None:
        cache.put(cache_key, winner.name, times)
        cache.flush()

    return winner, scored


@contextmanager
def kernel_override(
    family: str, mode: str, kernel_name: str
) -> Generator[None, None, None]:
    """Context manager for scoped kernel override."""
    key = (family, mode)
    old = _global_overrides.get(key)
    _global_overrides[key] = kernel_name
    try:
        yield
    finally:
        if old is None:
            _global_overrides.pop(key, None)
        else:
            _global_overrides[key] = old


def explain_selection(
    family: str,
    mode: str,
    format_signature: FormatSignature,
    *,
    features: frozenset[str] | None = None,
    platform: PlatformInfo | None = None,
    objective: SelectionObjective = SelectionObjective.DEFAULT,
    traits: dict[str, Any] | None = None,
    solution: str | None = None,
) -> str:
    """Return a human-readable explanation of kernel selection.

    Example output::

        Op: attention.decode (bfloat16)
        Platform: NVIDIA H100 (sm_90)
        Objective: default
        Ranking: lex (oracle, objective, priority, origin); higher wins

        Candidates (3 matched, 5 registered):
          1. flashinfer_decode  [SELECTED]
             ora=16 obj=1 pri=14
          2. triton_decode
             ora=10 obj=0 pri=10

        Filtered out:
          - aiter_decode: vendor mismatch (requires amd)
    """
    platform = platform or current_platform()
    registry = KernelRegistry.get()

    all_specs = registry.list_kernels(family=family, mode=mode)
    candidates = registry.get_for_operator(
        family,
        mode,
        features=features,
        platform=platform,
        format_signature=format_signature,
        solution=solution,
    )

    if traits:
        candidates = _filter_by_traits(candidates, traits)

    scored = _rank_by_objective(candidates, objective, platform, traits)

    filtered_names = {s.name for s in candidates}
    filtered_out = [s for s in all_specs if s.name not in filtered_names]

    lines = [
        f"Op: {family}.{mode} ({format_signature})",
        f"Platform: {platform.device_name} ({platform.arch})",
        f"Solution: {solution or 'any'}",
        f"Objective: {objective.value}",
        "Ranking: lex (oracle, objective, priority); higher wins",
        "",
        f"Candidates ({len(scored)} matched, {len(all_specs)} registered):",
    ]

    for i, (spec, breakdown) in enumerate(scored):
        marker = "  [SELECTED]" if i == 0 else ""
        lines.append(f"  {i + 1}. {spec.name}{marker}")
        lines.append(f"     {breakdown}")

    if filtered_out:
        lines.append("")
        lines.append("Filtered out:")
        for spec in filtered_out:
            reasons: list[str] = []
            if (
                spec.capability.vendors
                and platform.vendor not in spec.capability.vendors
            ):
                reasons.append(
                    f"vendor mismatch (requires "
                    f"{', '.join(spec.capability.vendors)})"
                )
            missing = spec.capability.missing_features(platform)
            if missing:
                reasons.append(f"missing features: {', '.join(missing)}")
            if spec.capability.min_arch_version:
                if not (platform.arch_version >= spec.capability.min_arch_version):
                    reasons.append(
                        f"arch mismatch (requires "
                        f"{spec.capability.min_arch_version})"
                    )
            if format_signature and not spec.supports_format_signature(
                format_signature
            ):
                reasons.append(
                    f"format signature mismatch (supports "
                    f"{', '.join(str(d) for d in spec.format_signatures)})"
                )
            if solution and spec.solution != solution:
                reasons.append(f"solution mismatch (is {spec.solution!r})")
            reason_str = "; ".join(reasons) if reasons else "unknown"
            lines.append(f"  - {spec.name}: {reason_str}")

    return "\n".join(lines)


def warmup_selection(
    ops: list[tuple] | None = None,
    *,
    freeze: bool = False,
) -> float:
    """Pre-resolve kernel selection for explicit op signatures.

    **This is where autotuning is meant to happen.** Tuning blocks the calling
    thread and, under tensor parallelism, desynchronizes ranks into a collective
    deadlock, so it must run once at startup and never again — call this from
    model initialization, then pass ``freeze=True`` (or call
    :func:`~unieinfra_kernel.selection.tuning.freeze_autotuning` yourself) once
    warmup, profiling and CUDA-graph capture are done.

    Args:
        ops: Entries of ``(family, mode, format_signature, traits)`` or
            ``(family, mode, format_signature, traits, shape)``. The 5-element
            form is what autotuning needs: without a concrete shape there is
            nothing to measure and selection stays heuristic. Each entry must
            carry the exact ``FormatSignature`` and trait values runtime
            selection will use, or the memoised answer will not be hit.

            When ``None``, every operator that has registered benchmark shapes
            (see :mod:`unieinfra_kernel.selection.inputs`) is warmed over those
            shapes with the traits its own contract declares; operators without
            them get a single deterministic smoke resolution using one
            representative signature and no traits. The smoke path verifies the
            registry and fills a small cache sample; it does **not** warm every
            signature, trait combination or feature-specific call path.
        freeze: Freeze autotuning when the warmup finishes. Pass ``True`` from
            an engine; leave ``False`` in tools that tune repeatedly.

    Returns:
        Wall-clock seconds spent, so a caller can report the startup cost
        autotuning added.
    """
    import time as _time

    started = _time.perf_counter()

    if ops is None:
        ops = []
        registry = KernelRegistry.get()
        for family, mode in registry.list_operators():
            specs = registry.get_for_operator(family, mode)
            if not specs or not specs[0].format_signatures:
                continue
            shapes = get_benchmark_shapes(family, mode)
            traits_fn = get_shape_traits_fn(family, mode)
            if shapes and _policy.get_strategy(family, mode) == (
                SelectionStrategy.AUTOTUNE
            ):
                # Tune over the operator's own representative workloads, using
                # the trait vocabulary its kernels gate on — a trait-free warmup
                # would fill cache entries the hot path never looks up.
                for shape in shapes:
                    traits = traits_fn(shape) if traits_fn else None
                    for format_signature in sorted(specs[0].format_signatures, key=str):
                        ops.append((family, mode, format_signature, traits, shape))
                continue
            # No-arg warmup is intentionally a smoke path. Pick a stable
            # representative from the highest-priority spec; callers that need
            # comprehensive warmup should pass explicit op signatures.
            format_signature = sorted(specs[0].format_signatures, key=str)[0]
            ops.append((family, mode, format_signature, None))

    for entry in ops:
        family, mode, format_signature, traits = entry[:4]
        shape = entry[4] if len(entry) > 4 else None
        try:
            select_kernel(family, mode, format_signature, traits=traits, shape=shape)
        except NoKernelFoundError:
            logger.debug(
                "[unieinfra_kernel] warmup: no kernel for %s.%s(%s)",
                family,
                mode,
                format_signature,
            )

    if freeze:
        freeze_autotuning()
    return _time.perf_counter() - started
