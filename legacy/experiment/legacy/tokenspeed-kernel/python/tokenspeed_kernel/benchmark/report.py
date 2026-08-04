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

from collections import defaultdict
from typing import Iterable

from tokenspeed_kernel.benchmark.result import BenchmarkResult

__all__ = ["format_report"]


def _shape_key(shape: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(shape.items(), key=lambda item: item[0]))


def _format_shape(shape: dict[str, object]) -> str:
    pairs = [f"{name}={value}" for name, value in sorted(shape.items())]
    return ", ".join(pairs)


def _format_float(value: float | None, width: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{width}f}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""

    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    right_aligned = {1, 2, 3, 4}
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render_row(cells: Iterable[str]) -> str:
        out: list[str] = []
        for i, cell in enumerate(cells):
            if i in right_aligned:
                out.append(cell.rjust(widths[i]))
            else:
                out.append(cell.ljust(widths[i]))
        return "| " + " | ".join(out) + " |"

    lines = [border, render_row(headers), border]
    lines.extend(render_row(row) for row in rows)
    lines.append(border)
    return "\n".join(lines)


def format_report(results: list[BenchmarkResult], *, group_by: str = "shape") -> str:
    """Format benchmark results as an ASCII table."""
    if not results:
        return "No benchmark results."

    if group_by != "shape":
        raise ValueError(f"Unsupported group_by={group_by!r}. Supported: 'shape'")

    lines: list[str] = []
    first = results[0]
    lines.append(
        f"Op: {first.op_family}.{first.op_mode} ({first.dtype}) "
        f"| Platform: {first.platform_arch}"
    )

    groups: dict[tuple[tuple[str, object], ...], list[BenchmarkResult]] = defaultdict(
        list
    )
    for result in results:
        groups[_shape_key(result.shape_params)].append(result)

    headers = ["Kernel", "p50 (us)", "p90 (us)", "p99 (us)", "TFLOPs"]
    for shape_key in sorted(groups):
        shape_results = sorted(
            groups[shape_key], key=lambda item: item.median_latency_us
        )
        shape = dict(shape_key)

        lines.append("")
        lines.append(f"Shape: {_format_shape(shape)}")

        rows: list[list[str]] = []
        for result in shape_results:
            rows.append(
                [
                    result.kernel_name,
                    _format_float(result.median_latency_us),
                    _format_float(result.p90_latency_us),
                    _format_float(result.p99_latency_us),
                    _format_float(result.tflops),
                ]
            )
        lines.append(_render_table(headers, rows))

    return "\n".join(lines)
