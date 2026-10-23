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

"""``--metrics-prefix``: export the Prometheus metrics as ``unieinfer:*``."""

from __future__ import annotations

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import argparse
import multiprocessing as mp
import time
import urllib.request
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest

from tokenspeed.runtime.metrics.prefix import (
    METRICS_PREFIX_CHOICES,
    PrefixedCollector,
    output_prefixes,
)
from tokenspeed.runtime.utils.common import launch_metrics_server
from tokenspeed.runtime.utils.network import get_free_port
from tokenspeed.runtime.utils.server_args import ServerArgs


@pytest.fixture
def source_registry():
    registry = CollectorRegistry()
    Counter("tokenspeed:num_requests", "requests", registry=registry).inc(3)
    Gauge(
        "tokenspeed:num_requests_running", "running", ["model"], registry=registry
    ).labels(model="m").set(2)
    Histogram(
        "tokenspeed:e2e_request_latency_seconds",
        "latency",
        buckets=[1.0],
        registry=registry,
    ).observe(0.5)
    Counter("other_total_requests", "not ours", registry=registry).inc()
    return registry


def _expose(source, mode: str) -> str:
    registry = CollectorRegistry(auto_describe=False)
    registry.register(PrefixedCollector(source, mode))
    return generate_latest(registry).decode()


def test_output_prefixes():
    assert METRICS_PREFIX_CHOICES == ("tokenspeed", "unieinfer", "both")
    assert output_prefixes("tokenspeed") == ("tokenspeed",)
    assert output_prefixes("unieinfer") == ("unieinfer",)
    assert output_prefixes("both") == ("tokenspeed", "unieinfer")
    with pytest.raises(ValueError):
        output_prefixes("vllm")


def test_tokenspeed_mode_is_unchanged(source_registry):
    assert _expose(source_registry, "tokenspeed") == (
        generate_latest(source_registry).decode()
    )


def test_unieinfer_mode_renames_only_our_namespace(source_registry):
    body = _expose(source_registry, "unieinfer")
    assert "tokenspeed:" not in body
    assert "unieinfer:num_requests_total 3.0" in body
    assert 'unieinfer:num_requests_running{model="m"} 2.0' in body
    assert 'unieinfer:e2e_request_latency_seconds_bucket{le="1.0"} 1.0' in body
    assert "unieinfer:e2e_request_latency_seconds_count 1.0" in body
    assert "# TYPE unieinfer:num_requests_total counter" in body
    assert "other_total_requests_total 1.0" in body


def test_both_mode_exports_each_metric_twice(source_registry):
    body = _expose(source_registry, "both")
    for name in (
        "num_requests_total 3.0",
        'num_requests_running{model="m"} 2.0',
        "e2e_request_latency_seconds_sum 0.5",
    ):
        assert f"tokenspeed:{name}" in body
        assert f"unieinfer:{name}" in body
    assert body.count("other_total_requests_total 1.0") == 1
    # The renamed copy is the original byte for byte, apart from the prefix.
    old = _expose(source_registry, "tokenspeed")
    new = _expose(source_registry, "unieinfer")
    assert sorted(body.splitlines()) == sorted(
        set(old.splitlines()) | set(new.splitlines())
    )


def _parse(flags: list[str]):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser.parse_args(["--model", "test-model", *flags])


def test_cli_default_keeps_tokenspeed_prefix():
    assert ServerArgs.metrics_prefix == "tokenspeed"
    assert _parse([]).metrics_prefix == "tokenspeed"


@pytest.mark.parametrize("value", ["unieinfer", "both", "tokenspeed"])
def test_cli_accepts_prefix(value):
    assert _parse(["--metrics-prefix", value]).metrics_prefix == value


def test_cli_rejects_unknown_prefix():
    with pytest.raises(SystemExit):
        _parse(["--metrics-prefix", "vllm"])


def _write_sample_counter(multiproc_dir: str) -> None:
    import os

    os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
    from prometheus_client import Counter

    Counter("tokenspeed:prefix_probe", "probe counter").inc(7)


@pytest.fixture
def sample_multiproc_dir(tmp_path, monkeypatch):
    # Set before spawning: the child re-imports this module, and with it
    # prometheus_client, before _write_sample_counter runs.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    proc = mp.get_context("spawn").Process(
        target=_write_sample_counter, args=(str(tmp_path),)
    )
    proc.start()
    proc.join()
    assert proc.exitcode == 0
    return tmp_path


@pytest.mark.parametrize(
    "prefix, present, absent",
    [
        (None, ["tokenspeed:prefix_probe_total 7.0"], ["unieinfer:"]),
        ("tokenspeed", ["tokenspeed:prefix_probe_total 7.0"], ["unieinfer:"]),
        ("unieinfer", ["unieinfer:prefix_probe_total 7.0"], ["tokenspeed:"]),
        (
            "both",
            ["tokenspeed:prefix_probe_total 7.0", "unieinfer:prefix_probe_total 7.0"],
            [],
        ),
    ],
)
def test_metrics_endpoint_applies_prefix(sample_multiproc_dir, prefix, present, absent):
    port = get_free_port()
    args = SimpleNamespace(
        enable_metrics=True,
        metrics_reporters=None,
        metrics_host=None,
        metrics_port=port,
        host="127.0.0.1",
    )
    if prefix is not None:
        args.metrics_prefix = prefix
    server = launch_metrics_server(args)
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "metrics server did not start"
            time.sleep(0.05)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            body = r.read().decode()
    finally:
        server.should_exit = True
    for text in present:
        assert text in body
    for text in absent:
        assert text not in body
