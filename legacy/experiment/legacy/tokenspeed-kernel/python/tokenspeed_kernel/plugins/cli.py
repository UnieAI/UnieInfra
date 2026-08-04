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

import argparse
import sys
from typing import Sequence

from tokenspeed_kernel.plugins import discover_plugins, list_plugins
from tokenspeed_kernel.registry import KernelRegistry


def _cmd_list(_: argparse.Namespace) -> int:
    plugins = sorted(list_plugins(), key=lambda p: p.name)
    if not plugins:
        print("No kernel plugins installed.")
        return 0
    print("Installed kernel plugins:")
    for p in plugins:
        version = f" ({p.version})" if p.version else ""
        head = ", ".join(p.kernel_names[:4])
        more = ", ..." if len(p.kernel_names) > 4 else ""
        print(f"  {p.name}{version} — {p.num_kernels} kernels [{head}{more}]")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    plugins = {p.name: p for p in list_plugins()}
    info = plugins.get(args.name)
    if info is None:
        print(f"Plugin {args.name!r} not loaded.", file=sys.stderr)
        return 1

    version = f" ({info.version})" if info.version else ""
    print(f"Plugin: {info.name}{version}")
    if info.package:
        print(f"Package: {info.package}")
    print("Kernels:")
    registry = KernelRegistry.get()
    for kname in info.kernel_names:
        spec = registry.get_by_name(kname)
        if spec is None:
            print(f"  {kname:32s} (no longer registered)")
            continue
        op = f"{spec.family}.{spec.mode}"
        cap = str(spec.capability) if spec.capability else "any"
        print(f"  {kname:32s} {op:20s} priority={spec.priority}  {cap}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tokenspeed_kernel.plugins")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list", help="List installed kernel plugins")
    list_p.set_defaults(func=_cmd_list)

    info_p = sub.add_parser("info", help="Show details for a single plugin")
    info_p.add_argument("name", help="Plugin entry point name")
    info_p.set_defaults(func=_cmd_info)

    # Plugin loading is explicit; discover here so the CLI sees them.
    discover_plugins()

    args = parser.parse_args(argv)
    return args.func(args)
