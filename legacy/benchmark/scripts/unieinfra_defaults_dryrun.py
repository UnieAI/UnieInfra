# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the UnieInfra project
"""Dry-run the `unieinfra serve` defaults for a model, without starting a server.

Parses a real `unieinfra serve` command line, applies the UnieInfra defaults
layer, then builds the full `VllmConfig` -- which is where vLLM resolves
speculative decoding, prefix caching, the KV cache dtype and the offloading
connector.  No weights are loaded and no GPU memory is allocated, so this can
verify auto-degradation on a busy box.

Usage:
    python unieinfra_defaults_dryrun.py <model> [extra serve args...]
"""

import argparse
import json
import sys

from vllm.entrypoints.cli.unieinfra_defaults import (
    apply_unieinfra_defaults,
    log_report,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2

    parser = make_arg_parser(FlexibleArgumentParser(description="dryrun"))
    args = parser.parse_args(argv)
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag

    report = apply_unieinfra_defaults(args, argv0="unieinfra")
    log_report(report)

    print("\n=== UnieInfra defaults decisions ===")
    for decision in report.decisions:
        print(decision.render())

    from vllm.engine.arg_utils import AsyncEngineArgs

    engine_args = AsyncEngineArgs.from_cli_args(args)
    config = engine_args.create_engine_config()

    spec = config.speculative_config
    cache = config.cache_config
    summary = {
        "model": config.model_config.model,
        "is_hybrid": config.model_config.is_hybrid,
        "prefix_caching_enabled": cache.enable_prefix_caching,
        "kv_cache_dtype": cache.cache_dtype,
        "kv_offloading_size_gib": cache.kv_offloading_size,
        "kv_connector": (
            config.kv_transfer_config.kv_connector
            if config.kv_transfer_config
            else None
        ),
        "speculative_method": spec.method if spec else None,
        "speculative_num_tokens": spec.num_speculative_tokens if spec else None,
        "speculative_ladder": (
            spec.num_speculative_tokens_per_batch_size if spec else None
        ),
        "speculative_draft_model": (
            spec.draft_model_config.model if spec and spec.draft_model_config else None
        ),
        "attention_backend": str(
            getattr(config.model_config, "attention_backend", None)
        ),
    }
    print("\n=== resolved engine config ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
