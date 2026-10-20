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

from dataclasses import dataclass
from typing import Callable

__all__ = [
    "Tolerance",
    "ToleranceFn",
    "ToleranceOverride",
    "ToleranceOverrides",
    "get_family_tolerance",
    "list_family_tolerances",
    "set_family_tolerance",
]


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


ToleranceFn = Callable[..., Tolerance]
ToleranceOverride = Tolerance | ToleranceFn
ToleranceOverrides = dict[str, ToleranceOverride]


_FAMILY_TOLERANCES: dict[str, ToleranceFn] = {}


def set_family_tolerance(family: str, tolerance_fn: ToleranceFn) -> None:
    _FAMILY_TOLERANCES[family] = tolerance_fn


def get_family_tolerance(family: str) -> ToleranceFn:
    tolerance_fn = _FAMILY_TOLERANCES.get(family)
    if tolerance_fn is None:
        known = ", ".join(sorted(_FAMILY_TOLERANCES)) or "none"
        raise KeyError(
            f"No tolerance function registered for family={family!r}. Known: {known}"
        )
    return tolerance_fn


def list_family_tolerances() -> dict[str, ToleranceFn]:
    return dict(_FAMILY_TOLERANCES)
