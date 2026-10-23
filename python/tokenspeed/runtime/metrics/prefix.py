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

"""Export the ``tokenspeed:`` Prometheus metrics under ``unieinfer:`` too.

Metrics are still created as ``tokenspeed:*`` and written once to the
multiprocess directory; the prefix is applied only when ``/metrics`` is
scraped, by renaming the aggregated families. ``both`` therefore costs nothing
on the serving path -- each scrape emits the families twice, which roughly
doubles the response size and the (scrape-time) formatting work.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

SOURCE_PREFIX = "tokenspeed"
NEW_PREFIX = "unieinfer"
BOTH = "both"
METRICS_PREFIX_CHOICES = (SOURCE_PREFIX, NEW_PREFIX, BOTH)
DEFAULT_METRICS_PREFIX = SOURCE_PREFIX


def output_prefixes(mode: str) -> tuple[str, ...]:
    """The prefixes ``mode`` exports, ``tokenspeed`` first."""
    if mode == BOTH:
        return (SOURCE_PREFIX, NEW_PREFIX)
    if mode in (SOURCE_PREFIX, NEW_PREFIX):
        return (mode,)
    raise ValueError(
        f"metrics prefix must be one of {', '.join(METRICS_PREFIX_CHOICES)}, "
        f"got {mode!r}"
    )


def _renamed(name: str, prefix: str) -> str:
    source = SOURCE_PREFIX + ":"
    if prefix == SOURCE_PREFIX or not name.startswith(source):
        return name
    return f"{prefix}:{name[len(source):]}"


def rename_families(families: Iterable, mode: str) -> Iterator:
    """Yield ``families`` once per exported prefix, renamed.

    Families outside the ``tokenspeed:`` namespace (process_*, python_gc_*,
    ...) are yielded once, unchanged.
    """
    from prometheus_client.metrics_core import Metric

    prefixes = output_prefixes(mode)
    for family in families:
        if not family.name.startswith(SOURCE_PREFIX + ":"):
            yield family
            continue
        for prefix in prefixes:
            if prefix == SOURCE_PREFIX:
                yield family
                continue
            renamed = Metric(
                _renamed(family.name, prefix),
                family.documentation,
                family.type,
                family.unit,
            )
            renamed.samples = [
                sample._replace(name=_renamed(sample.name, prefix))
                for sample in family.samples
            ]
            yield renamed


class PrefixedCollector:
    """A collector re-exporting ``source``'s families under ``mode``."""

    def __init__(self, source, mode: str) -> None:
        output_prefixes(mode)  # validate early
        self.source = source
        self.mode = mode

    def collect(self) -> Iterator:
        return rename_families(self.source.collect(), self.mode)


def build_metrics_registry(mode: str = DEFAULT_METRICS_PREFIX):
    """The multiprocess registry ``/metrics`` serves, exported under ``mode``."""
    from prometheus_client import CollectorRegistry, multiprocess

    if mode == SOURCE_PREFIX:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    source = CollectorRegistry()
    multiprocess.MultiProcessCollector(source)
    registry = CollectorRegistry(auto_describe=False)
    registry.register(PrefixedCollector(source, mode))
    return registry
