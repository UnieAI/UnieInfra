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

"""TokenSpeed CLI entry point."""

import argparse
import sys


def _select_scheduler_ext(raw_argv: list[str]) -> None:
    """Pick the tokenspeed_scheduler extension this model needs.

    Has to happen HERE, before the first `import tokenspeed.*`: the radix and
    flat schedulers are two separate .so files (the flat build compiles the
    radix insert/publish path out and its Scheduler constructor refuses an
    empty group list), and `runtime/engine/event_loop.py` imports the
    extension at module scope. By the time anything holds a ModelConfig the
    choice is already made.

    So read the checkpoint's `architectures` directly -- no torch, no
    transformers, no engine import -- and record the answer in the
    environment, where the engine subprocess inherits it. Anything unparseable
    leaves the default alone; the loader will report it properly later.
    """
    from tokenspeed.flat_kvcache_select import select_flat_kvcache_ext_for_model_path

    model = next((arg for arg in raw_argv if arg and not arg.startswith("-")), None)
    if model is not None:
        select_flat_kvcache_ext_for_model_path(model)


def _serve(args: argparse.Namespace, raw_argv: list[str]) -> None:
    _select_scheduler_ext(raw_argv)

    from tokenspeed.cli.serve_smg import run_smg_from_args

    run_smg_from_args(args, raw_argv)


def _bench(args: argparse.Namespace) -> None:
    from tokenspeed.bench import main as bench_main

    bench_main(args.bench_args)


def _env(args: argparse.Namespace) -> None:
    from tokenspeed.env import main as env_main

    env_main()


def _merge_traces(args: argparse.Namespace) -> None:
    from tokenspeed.cli.trace_merge import main as merge_traces_main

    merge_traces_main(args.merge_args)


def _version(args: argparse.Namespace) -> None:
    from tokenspeed.version import __version__

    print(f"TokenSpeed v{__version__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tokenspeed",
        description="TokenSpeed is a speed-of-light LLM inference engine.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Unknown flags fall through to the smg orchestrator's own splitter; we
    # don't register the engine's ServerArgs on this parser.
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the TokenSpeed inference server.",
        description="Launch the TokenSpeed inference server: the full serving "
        "stack by default (currently an smg gateway fronting a gRPC engine), "
        "or the engine alone with --headless.",
    )
    serve_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the engine only, without a frontend. An external frontend "
        "— e.g. `smg serve --backend tokenspeed --connection-mode zmq` — "
        "binds the msgpack ZMQ sockets and this engine dials in at "
        "--data-parallel-address/--data-parallel-rpc-port (default "
        "tcp://127.0.0.1:30500). Implies --zmq-msgpack and "
        "--skip-tokenizer-init.",
    )
    serve_parser.set_defaults(func=_serve)

    bench_parser = subparsers.add_parser(
        "bench",
        add_help=False,
        help="Run TokenSpeed benchmark commands.",
    )
    bench_parser.set_defaults(func=_bench, bench_args=[])

    env_parser = subparsers.add_parser(
        "env",
        help="Check environment configurations and dependency versions.",
    )
    env_parser.set_defaults(func=_env)

    merge_traces_parser = subparsers.add_parser(
        "merge-traces",
        add_help=False,
        help="Merge one or more Proton/VizTracer trace pairs onto a shared "
        "timeline.",
    )
    merge_traces_parser.set_defaults(func=_merge_traces, merge_args=[])

    version_parser = subparsers.add_parser(
        "version",
        help="Print the TokenSpeed version.",
    )
    version_parser.set_defaults(func=_version)

    args, extra_args = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.func is _bench:
        args.bench_args = extra_args
        args.func(args)
        return

    if args.func is _merge_traces:
        args.merge_args = extra_args
        args.func(args)
        return

    if args.func is _serve:
        raw = list(sys.argv[2:])
        args.func(args, raw)
        return

    if extra_args:
        parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
    args.func(args)


if __name__ == "__main__":
    main()
