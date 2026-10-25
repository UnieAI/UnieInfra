# Copyright (c) 2026 UnieAI
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

"""``UNIEINFER_*`` spellings of the legacy ``TOKENSPEED_*`` / ``UNIEINFRA_*``
environment variables (see UNIEINFER.md, "Name aliases").

``UNIEINFER_X`` is the new name of both ``TOKENSPEED_X`` and ``UNIEINFRA_X``.
When the new and the legacy name are both set, the new one wins, and the first
time they disagree a warning is logged.

Two mechanisms, both stdlib-only so this module stays import-cheap:

* ``getenv`` / ``is_set`` look the pair up at call time. The ``envs`` registry
  (``tokenspeed.runtime.utils.env``) reads through them.
* ``apply_aliases`` copies every ``UNIEINFER_X`` onto ``TOKENSPEED_X`` in
  ``os.environ``. ``tokenspeed_kernel`` runs it first thing on import, so the
  many direct ``os.environ.get("TOKENSPEED_...")`` reads in upstream code, and
  every child process, see the new spelling without each call site being
  edited. Values assigned to ``UNIEINFER_*`` inside the process after that
  import are only seen through ``getenv``.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import MutableMapping

NEW_PREFIX = "UNIEINFER_"
LEGACY_PREFIXES = ("TOKENSPEED_", "UNIEINFRA_")
# The prefix ``apply_aliases`` mirrors new names onto.
MIRROR_PREFIX = "TOKENSPEED_"

logger = logging.getLogger(__name__)

_warned: set[tuple[str, str]] = set()
_warned_lock = threading.Lock()

__all__ = [
    "LEGACY_PREFIXES",
    "NEW_PREFIX",
    "apply_aliases",
    "getenv",
    "is_set",
    "new_name",
]


def new_name(name: str) -> str | None:
    """``UNIEINFER_X`` for a legacy ``TOKENSPEED_X``/``UNIEINFRA_X``, else None."""
    for prefix in LEGACY_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            return NEW_PREFIX + name[len(prefix) :]
    return None


def _warn_disagree(new: str, legacy: str, new_value: str, legacy_value: str):
    with _warned_lock:
        if (new, legacy) in _warned:
            return
        _warned.add((new, legacy))
    logger.warning(
        "%s=%r and %s=%r are both set and disagree; using %s. "
        "%s is the legacy name of %s.",
        new,
        new_value,
        legacy,
        legacy_value,
        new,
        legacy,
        new,
    )


def getenv(
    name: str,
    default: str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> str | None:
    """Read legacy variable ``name``, preferring its ``UNIEINFER_`` spelling.

    ``name`` is the legacy spelling the code already uses (``TOKENSPEED_X`` or
    ``UNIEINFRA_X``); any other name is read as is.
    """
    env = os.environ if environ is None else environ
    legacy_value = env.get(name)
    alias = new_name(name)
    if alias is not None:
        new_value = env.get(alias)
        if new_value is not None:
            if legacy_value is not None and legacy_value != new_value:
                _warn_disagree(alias, name, new_value, legacy_value)
            return new_value
    return default if legacy_value is None else legacy_value


def is_set(name: str, environ: MutableMapping[str, str] | None = None) -> bool:
    """Whether ``name`` or its ``UNIEINFER_`` spelling is present."""
    env = os.environ if environ is None else environ
    alias = new_name(name)
    return name in env or (alias is not None and alias in env)


def apply_aliases(environ: MutableMapping[str, str] | None = None) -> None:
    """Copy each ``UNIEINFER_X`` onto ``TOKENSPEED_X``; the new name wins.

    Idempotent. A disagreeing legacy value is overwritten after a one-time
    warning, so direct legacy reads and child processes agree with ``getenv``.
    """
    env = os.environ if environ is None else environ
    for key, value in list(env.items()):
        if not key.startswith(NEW_PREFIX) or len(key) == len(NEW_PREFIX):
            continue
        legacy = MIRROR_PREFIX + key[len(NEW_PREFIX) :]
        current = env.get(legacy)
        if current == value:
            continue
        if current is not None:
            _warn_disagree(key, legacy, value, current)
        env[legacy] = value
