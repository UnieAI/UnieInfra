#!/usr/bin/env python3
"""Sample the engine's own scheduler gauges while a load runs.

The question at high concurrency is why throughput stops scaling: either the
batch is not full (admission refuses work it could hold) or the step does prefill
and decode in one round only part of the time, so each decode step pays a
prefill chunk and each prefill step pays a decode round separately. Both are
visible in the counters the scheduler already exports once per iteration:

  num_requests_running          how many requests the batch actually holds
  num_requests_waiting          how many are queued behind it
  kv_cache_usage_perc           how full the cache is, i.e. what admission sees
  iteration_tokens_total        histogram of tokens per iteration; above the
                                running count means prefill rides in the step
  request_queue_time_seconds    how long a request waited before admission

The step-size histogram's buckets are kept for that reason: a step that carries
decode rows alone lands in the low buckets, a step that carries a prefill chunk
lands in the high ones, so the bucket counts say how many of the steps did
prefill and how many did decode -- without a trace. The bounds are the server's
own (1-2-5 up to its largest step), so they are discovered from the first
scrape rather than hardcoded. Other histograms keep only their sum and count.

Kill it rather than waiting out SECONDS: a line-buffered row per sample means the
file is complete whenever the sampler stops.

Usage:
    python3 bench/agent-slo/step_metrics.py URL OUT.csv SECONDS [HZ]

Writes one row per sample with both a monotonic timestamp (for the series) and a
wall-clock one, so `step_report.py` can slice the file to the measured window a
result JSON names, and drop the warmup it shares the process with.
"""

import sys
import time
import urllib.request

# Exported as gauges, one series each.
GAUGES = (
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_perc",
)

# Exported as histograms; the sum and the count give a window mean. Several
# label sets can each contribute, so they are added up.
HISTOGRAMS = (
    "iteration_tokens_total",
    "request_queue_time_seconds",
    "time_to_first_token_seconds",
)

# Histograms whose buckets are kept, as their own columns.
BUCKET_HISTOGRAMS = ("iteration_tokens_total",)

KEYS = list(GAUGES) + [f"{h}_{part}" for h in HISTOGRAMS for part in ("sum", "count")]


def scrape(url):
    with urllib.request.urlopen(url, timeout=2) as resp:
        return resp.read().decode("utf-8", "replace")


def bucket_key(name, labels):
    """Column name for one bucket of one histogram, or None if not kept."""
    if not name.endswith("_bucket"):
        return None
    short = name.rsplit(":", 1)[-1][: -len("_bucket")]
    if short not in BUCKET_HISTOGRAMS:
        return None
    marker = 'le="'
    start = labels.find(marker)
    if start < 0:
        return None
    bound = labels[start + len(marker) :].split('"', 1)[0]
    return f"{short}_le{bound}"


def discover(body):
    """Bucket columns a body offers, so the CSV header can name them."""
    found = set()
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, _ = line.rpartition(" ")
        name, _, labels = head.partition("{")
        column = bucket_key(name, labels)
        if column is not None:
            found.add(column)
    return sorted(found)


def parse(body, keys):
    out = dict.fromkeys(keys, 0.0)
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name, _, labels = head.partition("{")
        short = name.rsplit(":", 1)[-1]
        column = short if short in out else bucket_key(name, labels)
        if column is None or column not in out:
            continue
        try:
            out[column] += float(value)
        except ValueError:
            continue
    return out


def main(argv):
    url, path, seconds = argv[1], argv[2], float(argv[3])
    hz = float(argv[4]) if len(argv) > 4 else 20.0
    period = 1.0 / hz
    keys = list(KEYS) + discover(scrape(url))
    with open(path, "w", buffering=1) as f:
        f.write("t,wall," + ",".join(keys) + "\n")
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            tick = time.monotonic() - start
            wall = time.time()
            try:
                sample = parse(scrape(url), keys)
            except Exception as exc:  # the server can be mid-shutdown
                sys.stderr.write(f"scrape failed at {tick:.1f}s: {exc}\n")
                time.sleep(period)
                continue
            f.write(
                f"{tick:.3f},{wall:.3f},"
                + ",".join(f"{sample[k]:.4f}" for k in keys)
                + "\n"
            )
            time.sleep(max(0.0, period - (time.monotonic() - start - tick)))


if __name__ == "__main__":
    main(sys.argv)
