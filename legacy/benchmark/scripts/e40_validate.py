# SPDX-License-Identifier: Apache-2.0
"""E40 — classify tool-call quality from e40_toolcall_bench.py result files.

Every response falls into exactly one bucket. The buckets are chosen so that
"malformed" means something a real agent loop would actually choke on:

  ok               a tool call whose name exists and whose arguments are valid
                   JSON conforming to that tool's parameter schema
  parse_fail       the model emitted ``<tool_call>`` text that the server-side
                   parser did NOT turn into a structured tool call — the caller
                   sees prose where a call belonged
  bad_json         a structured tool call whose ``arguments`` string does not
                   parse as JSON
  unknown_name     a call naming a function that was not offered
  schema_violation arguments parse, name is known, but the JSON Schema rejects
                   them (missing required field, wrong type, bad enum, ...)
  no_call          no tool call and no ``<tool_call>`` marker: the model
                   answered in prose instead of calling anything
  truncated        hit the output-token cap mid-call (finish_reason="length")
                   with no complete call — a budget artifact, reported apart
                   from model error so it cannot inflate the malformed rate
  failed           the HTTP request itself failed

A separate, stricter counter is also reported: ``extra_args``, arguments
containing property names absent from the schema's ``properties``. Most BFCL
schemas omit ``additionalProperties``, so plain JSON Schema validation accepts
those, but a real tool would raise ``TypeError: unexpected keyword argument``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter
from typing import Any

import jsonschema

BUCKETS = [
    "ok",
    "parse_fail",
    "bad_json",
    "unknown_name",
    "schema_violation",
    "no_call",
    "truncated",
    "failed",
]


def _load_tools(data_path: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with open(data_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec.get("tools") or []
    return out


def _extra_arg_names(args_obj: Any, schema: dict) -> list[str]:
    if not isinstance(args_obj, dict):
        return []
    props = (schema or {}).get("properties") or {}
    return [k for k in args_obj if k not in props]


def classify_one(req: dict, tools: list[dict]) -> tuple[str, list[str]]:
    """Return (bucket, extra_arg_names)."""
    if not req.get("success"):
        return "failed", []

    by_name = {
        (t.get("function") or {})
        .get("name"): (t.get("function") or {})
        .get("parameters")
        or {}
        for t in tools
    }
    calls = req.get("tool_calls") or []
    content = req.get("content") or ""

    if not calls:
        if "<tool_call>" in content or "</tool_call>" in content:
            return "parse_fail", []
        if req.get("finish_reason") == "length":
            return "truncated", []
        return "no_call", []

    extras: list[str] = []
    worst = "ok"

    def _worse(a: str, b: str) -> str:
        order = ["ok", "schema_violation", "unknown_name", "bad_json"]
        return a if order.index(a) >= order.index(b) else b

    for c in calls:
        name = c.get("name")
        raw = c.get("arguments") or ""
        if name not in by_name:
            worst = _worse(worst, "unknown_name")
            continue
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except (json.JSONDecodeError, ValueError):
            # An arguments string cut off by the token cap is a budget
            # artifact, not a model error -- surface it as truncated.
            if req.get("finish_reason") == "length":
                return "truncated", []
            worst = _worse(worst, "bad_json")
            continue
        schema = by_name[name]
        extras.extend(_extra_arg_names(parsed, schema))
        try:
            jsonschema.validate(instance=parsed, schema=schema)
        except jsonschema.ValidationError:
            worst = _worse(worst, "schema_violation")
        except jsonschema.SchemaError:
            # A malformed tool schema is our dataset's problem, not the
            # model's; do not charge it to the model.
            pass
    return worst, extras


def analyse(path: str, tools_by_id: dict[str, list[dict]]) -> dict[str, Any]:
    with open(path) as f:
        res = json.load(f)
    counts = Counter()
    n_extra_reqs = 0
    extra_names = Counter()
    for req in res["requests"]:
        rid = req["request_id"]
        bucket, extras = classify_one(req, tools_by_id.get(rid, []))
        counts[bucket] += 1
        if extras:
            n_extra_reqs += 1
            extra_names.update(extras)
    n = sum(counts.values())
    judged = n - counts["failed"]
    malformed = (
        counts["parse_fail"]
        + counts["bad_json"]
        + counts["unknown_name"]
        + counts["schema_violation"]
    )
    return {
        "path": path,
        "config": res["config"],
        "metrics": res["metrics"],
        "spec_decode": res.get("spec_decode"),
        "counts": dict(counts),
        "n": n,
        "judged": judged,
        "malformed": malformed,
        "malformed_pct": 100.0 * malformed / judged if judged else 0.0,
        "call_rate_pct": 100.0 * (judged - counts["no_call"]) / judged if judged else 0,
        "ok_pct": 100.0 * counts["ok"] / judged if judged else 0.0,
        "extra_arg_reqs": n_extra_reqs,
        "extra_arg_pct": 100.0 * n_extra_reqs / judged if judged else 0.0,
        "extra_arg_names": extra_names.most_common(10),
    }


def _agg(vals: list[float]) -> tuple[float, float, float]:
    return statistics.median(vals), min(vals), max(vals)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-glob", required=True)
    p.add_argument("--data", required=True, help="frozen BFCL jsonl (for schemas)")
    p.add_argument("--out", default=None, help="markdown report path")
    args = p.parse_args()

    tools_by_id = _load_tools(args.data)
    paths = sorted(glob.glob(args.results_glob))
    if not paths:
        raise SystemExit(f"no results matched {args.results_glob}")

    rows = [analyse(p_, tools_by_id) for p_ in paths]

    # group by (arm, concurrency) across repeats
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        key = (r["config"]["arm"], r["config"]["max_concurrency"])
        groups.setdefault(key, []).append(r)

    lines: list[str] = []
    lines.append(
        "# E40 — tool-call quality (BFCL, real `tools`/`tool_choice` traffic)\n"
    )
    lines.append(
        "Median of repeats, `[min, max]` spread. A delta inside the spread is a "
        "TIE. `malformed%` = parse_fail + bad_json + unknown_name + "
        "schema_violation, over judged (non-HTTP-failed) responses.\n"
    )
    hdr = (
        "| arm | C | n | ok% | malformed% | parse_fail | bad_json | "
        "unknown_name | schema_viol | no_call | trunc | extra_arg% |"
    )
    lines.append(hdr)
    lines.append("|" + "---|" * 12)
    for (arm, c), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):

        def med(f):
            return statistics.median([f(r) for r in rs])

        lines.append(
            f"| {arm} | {c} | {rs[0]['judged']} | {med(lambda r: r['ok_pct']):.1f} | "
            f"{med(lambda r: r['malformed_pct']):.2f} | "
            f"{med(lambda r: r['counts'].get('parse_fail', 0)):.0f} | "
            f"{med(lambda r: r['counts'].get('bad_json', 0)):.0f} | "
            f"{med(lambda r: r['counts'].get('unknown_name', 0)):.0f} | "
            f"{med(lambda r: r['counts'].get('schema_violation', 0)):.0f} | "
            f"{med(lambda r: r['counts'].get('no_call', 0)):.0f} | "
            f"{med(lambda r: r['counts'].get('truncated', 0)):.0f} | "
            f"{med(lambda r: r['extra_arg_pct']):.1f} |"
        )

    lines.append("\n## Cost of the constraint\n")
    lines.append(
        "| arm | C | out tok/s | TTFT ms | TPOT ms | out len | "
        "accept len | accept rate % |"
    )
    lines.append("|" + "---|" * 8)
    for (arm, c), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):

        def sp(r, k):
            s = r.get("spec_decode")
            return s[k] if s else float("nan")

        def cell(f, fmt="{:.1f}"):
            m, lo, hi = _agg([f(r) for r in rs])
            return f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]"

        has_spec = rs[0].get("spec_decode") is not None
        accept = (
            cell(lambda r: sp(r, "acceptance_length"), "{:.3f}") if has_spec else "--"
        )
        arate = cell(lambda r: sp(r, "acceptance_rate")) if has_spec else "--"
        lines.append(
            f"| {arm} | {c} | "
            f"{cell(lambda r: r['metrics']['output_throughput'])} | "
            f"{cell(lambda r: r['metrics']['mean_ttft_ms'])} | "
            f"{cell(lambda r: r['metrics']['mean_tpot_ms'], '{:.2f}')} | "
            f"{cell(lambda r: r['metrics']['mean_output_tokens'])} | "
            f"{accept} | {arate} |"
        )

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
