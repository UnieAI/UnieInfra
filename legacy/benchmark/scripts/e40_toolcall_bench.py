# SPDX-License-Identifier: Apache-2.0
"""E40 — tool-call quality + latency client for the OpenAI Chat Completions API.

Why this is not ``unieinfra bench serve``: the vendored chat backend
(``vllm/benchmarks/lib/endpoint_request_func.py:async_request_openai_chat_completions``)
accumulates **only** ``choices[0].delta.content``. When a server-side tool parser
is configured, a tool call arrives as ``delta.tool_calls`` instead, so the
harness records an empty ``generated_text`` and tool-call quality is invisible to
it. This client assembles both channels, so every response can be classified.

It measures, per request:
  * TTFT / ITL / end-to-end latency / completion tokens (same definitions the
    vendored harness uses, so numbers are comparable to E39)
  * the assembled ``tool_calls`` (name + arguments string) and any text content
  * ``finish_reason``

and, per run, the server's speculative-decoding counters scraped from
``/metrics`` before and after (same parsing as ``benchmarks/serve.py``), so
acceptance length is measured over exactly this run's traffic.

Arms are flags, and each flips exactly one thing:
  ``--strict``   inject ``strict: true`` into every tool's ``function`` object.
                 This is the documented lever that makes ``tool_choice="auto"``
                 actually build a grammar: see
                 ``vllm/tool_parsers/structural_tag_registry.py:get_model_structural_tag``,
                 which returns ``None`` for ``auto`` unless some tool is strict.
  ``--tool-choice`` override ``auto`` (e.g. ``required``).
  ``--enable-thinking`` pass ``chat_template_kwargs={"enable_thinking": ...}``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np


@dataclass
class ReqResult:
    request_id: str
    success: bool = False
    error: str = ""
    ttft: float = 0.0
    latency: float = 0.0
    itl: list[float] = field(default_factory=list)
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None


def _apply_strict(tools: list[dict], strict: bool) -> list[dict]:
    if not strict:
        return tools
    out = []
    for t in tools:
        t = json.loads(json.dumps(t))
        fn = t.get("function")
        if isinstance(fn, dict):
            fn["strict"] = True
        out.append(t)
    return out


class _ToolCallAccum:
    """Assemble streamed ``delta.tool_calls`` fragments into whole calls.

    OpenAI streaming sends a tool call as an ``index``-keyed sequence of
    fragments: the first carries ``id``/``function.name``, later ones append to
    ``function.arguments``. Fragments for different indices may interleave.
    """

    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, Any]] = {}
        self._order: list[int] = []

    def add(self, deltas: list[dict[str, Any]]) -> None:
        for d in deltas:
            idx = d.get("index", 0)
            if idx not in self._by_index:
                self._by_index[idx] = {"id": None, "name": None, "arguments": ""}
                self._order.append(idx)
            slot = self._by_index[idx]
            if d.get("id"):
                slot["id"] = d["id"]
            fn = d.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]

    def result(self) -> list[dict[str, Any]]:
        return [self._by_index[i] for i in self._order]


async def _one_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    request_id: str,
    stream: bool,
) -> ReqResult:
    out = ReqResult(request_id=request_id)
    st = time.perf_counter()
    last = st
    acc = _ToolCallAccum()
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                out.error = f"HTTP {resp.status}: {(await resp.text())[:400]}"
                return out
            if not stream:
                data = await resp.json()
                out.latency = time.perf_counter() - st
                out.ttft = out.latency
                choice = data["choices"][0]
                msg = choice.get("message") or {}
                out.content = msg.get("content") or ""
                out.reasoning = msg.get("reasoning_content") or ""
                out.tool_calls = [
                    {
                        "id": tc.get("id"),
                        "name": (tc.get("function") or {}).get("name"),
                        "arguments": (tc.get("function") or {}).get("arguments") or "",
                    }
                    for tc in (msg.get("tool_calls") or [])
                ]
                out.finish_reason = choice.get("finish_reason")
                usage = data.get("usage") or {}
                out.output_tokens = usage.get("completion_tokens")
                out.prompt_tokens = usage.get("prompt_tokens")
                out.success = True
                return out

            buf = b""
            async for chunk in resp.content.iter_any():
                buf += chunk
                while b"\n\n" in buf:
                    raw, buf = buf.split(b"\n\n", 1)
                    for line in raw.split(b"\n"):
                        line = line.strip()
                        if not line or line.startswith(b":"):
                            continue
                        if not line.startswith(b"data: "):
                            continue
                        body = line[6:]
                        if body == b"[DONE]":
                            continue
                        now = time.perf_counter()
                        data = json.loads(body)
                        if choices := data.get("choices"):
                            ch = choices[0]
                            delta = ch.get("delta") or {}
                            content = delta.get("content")
                            reasoning = delta.get("reasoning_content")
                            tcs = delta.get("tool_calls")
                            got = bool(content) or bool(tcs) or bool(reasoning)
                            if got:
                                if out.ttft == 0.0:
                                    out.ttft = now - st
                                else:
                                    out.itl.append(now - last)
                                last = now
                            out.content += content or ""
                            out.reasoning += reasoning or ""
                            if tcs:
                                acc.add(tcs)
                            if ch.get("finish_reason"):
                                out.finish_reason = ch["finish_reason"]
                        elif usage := data.get("usage"):
                            out.output_tokens = usage.get("completion_tokens")
                            out.prompt_tokens = usage.get("prompt_tokens")
            out.tool_calls = acc.result()
            out.latency = last - st
            out.success = True
    except Exception as e:  # noqa: BLE001 — record, never abort the run
        out.error = f"{type(e).__name__}: {e}"
    return out


async def _scrape_spec_metrics(session: aiohttp.ClientSession, base: str):
    """Same counter set benchmarks/serve.py:fetch_spec_decode_metrics reads."""
    try:
        async with session.get(f"{base}/metrics") as r:
            if r.status != 200:
                return None
            text = await r.text()
    except Exception:  # noqa: BLE001
        return None
    n_drafts = n_draft_tokens = n_accepted = 0
    per_pos: dict[int, int] = {}
    found = False
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split(None, 1)
        name = parts[0].split("{")[0]
        if not name.endswith("_total"):
            continue
        found = True
        try:
            if "num_drafts" in name:
                n_drafts += int(float(parts[-1]))
            elif "num_draft_tokens" in name:
                n_draft_tokens += int(float(parts[-1]))
            elif "num_accepted_tokens_per_pos" in name:
                lbl = 'position="'
                if lbl in line:
                    s = line.index(lbl) + len(lbl)
                    pos = int(line[s : line.index('"', s)])
                    per_pos[pos] = per_pos.get(pos, 0) + int(float(parts[-1]))
            elif "num_accepted_tokens" in name:
                n_accepted += int(float(parts[-1]))
        except ValueError:
            pass
    if not found:
        return None
    return {
        "num_drafts": n_drafts,
        "num_draft_tokens": n_draft_tokens,
        "num_accepted_tokens": n_accepted,
        "per_pos": per_pos,
    }


def _build_payload(rec: dict, args) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": rec["messages"],
        "max_completion_tokens": args.output_len,
        "temperature": args.temperature,
        "stream": not args.no_stream,
        "seed": args.seed,
    }
    if not args.no_stream:
        payload["stream_options"] = {"include_usage": True}
    tools = rec.get("tools") or []
    if tools and not args.no_tools:
        payload["tools"] = _apply_strict(tools, args.strict)
        payload["tool_choice"] = args.tool_choice or rec.get("tool_choice", "auto")
    payload["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}
    return payload


async def run(args) -> dict[str, Any]:
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.num_prompts > 0:
        # cycle if the caller wants more requests than the frozen set has
        rows = [rows[i % len(rows)] for i in range(args.num_prompts)]

    base = f"http://{args.host}:{args.port}"
    url = f"{base}/v1/chat/completions"
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        # ---- warmup (not measured) ----
        if args.warmups > 0:
            sem = asyncio.Semaphore(args.max_concurrency)

            async def _warm(rec, i):
                async with sem:
                    await _one_request(
                        session,
                        url,
                        _build_payload(rec, args),
                        f"warm-{i}",
                        not args.no_stream,
                    )

            await asyncio.gather(
                *[_warm(rows[i % len(rows)], i) for i in range(args.warmups)]
            )

        spec_before = await _scrape_spec_metrics(session, base)

        sem = asyncio.Semaphore(args.max_concurrency)
        results: list[ReqResult] = [None] * len(rows)  # type: ignore[list-item]

        async def _go(i: int, rec: dict):
            async with sem:
                results[i] = await _one_request(
                    session,
                    url,
                    _build_payload(rec, args),
                    rec.get("id", str(i)),
                    not args.no_stream,
                )

        t0 = time.perf_counter()
        await asyncio.gather(*[_go(i, rec) for i, rec in enumerate(rows)])
        dur = time.perf_counter() - t0

        spec_after = await _scrape_spec_metrics(session, base)

    ok = [r for r in results if r.success]
    n_out = sum((r.output_tokens or 0) for r in ok)
    n_in = sum((r.prompt_tokens or 0) for r in ok)
    ttfts = np.array([r.ttft * 1000 for r in ok]) if ok else np.array([0.0])
    tpots = []
    for r in ok:
        if (r.output_tokens or 0) > 1:
            tpots.append((r.latency - r.ttft) / (r.output_tokens - 1) * 1000)
    tpots_a = np.array(tpots) if tpots else np.array([0.0])
    itl_vals = [x * 1000 for r in ok for x in r.itl]
    itls = np.array(itl_vals) if itl_vals else np.array([0.0])

    spec = None
    if spec_before and spec_after:
        d_draft = spec_after["num_draft_tokens"] - spec_before["num_draft_tokens"]
        d_acc = spec_after["num_accepted_tokens"] - spec_before["num_accepted_tokens"]
        d_drafts = spec_after["num_drafts"] - spec_before["num_drafts"]
        pos = {
            k: spec_after["per_pos"].get(k, 0) - spec_before["per_pos"].get(k, 0)
            for k in spec_after["per_pos"]
        }
        spec = {
            "num_drafts": d_drafts,
            "num_draft_tokens": d_draft,
            "num_accepted_tokens": d_acc,
            # acceptance_length counts the always-accepted bonus token, matching
            # benchmarks/serve.py, so it is out of (k + 1) positions.
            "acceptance_length": (1 + d_acc / d_drafts) if d_drafts else 0.0,
            "acceptance_rate": (100.0 * d_acc / d_draft) if d_draft else 0.0,
            "per_position_accept_rate": {
                str(k): (100.0 * v / d_drafts) if d_drafts else 0.0
                for k, v in sorted(pos.items())
            },
        }

    return {
        "config": {
            "label": args.label,
            "arm": args.arm,
            "strict": args.strict,
            "tool_choice": args.tool_choice,
            "no_tools": args.no_tools,
            "enable_thinking": args.enable_thinking,
            "max_concurrency": args.max_concurrency,
            "num_prompts": len(rows),
            "output_len": args.output_len,
            "temperature": args.temperature,
            "stream": not args.no_stream,
            "data": args.data,
            "seed": args.seed,
        },
        "metrics": {
            "duration_s": dur,
            "completed": len(ok),
            "failed": len(results) - len(ok),
            "total_output_tokens": n_out,
            "total_input_tokens": n_in,
            "request_throughput": len(ok) / dur if dur else 0.0,
            "output_throughput": n_out / dur if dur else 0.0,
            "mean_ttft_ms": float(ttfts.mean()),
            "median_ttft_ms": float(np.median(ttfts)),
            "p99_ttft_ms": float(np.percentile(ttfts, 99)),
            "mean_tpot_ms": float(tpots_a.mean()),
            "median_tpot_ms": float(np.median(tpots_a)),
            "mean_itl_ms": float(itls.mean()),
            "mean_output_tokens": n_out / len(ok) if ok else 0.0,
        },
        "spec_decode": spec,
        "requests": [
            {
                "request_id": r.request_id,
                "success": r.success,
                "error": r.error,
                "ttft_ms": r.ttft * 1000,
                "latency_ms": r.latency * 1000,
                "output_tokens": r.output_tokens,
                "finish_reason": r.finish_reason,
                "content": r.content,
                "reasoning": r.reasoning,
                "tool_calls": r.tool_calls,
            }
            for r in results
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8323)
    p.add_argument("--model", default="q8")
    p.add_argument("--data", required=True)
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmups", type=int, default=8)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--strict", action="store_true", help="inject strict:true on tools")
    p.add_argument("--tool-choice", default=None)
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--arm", default="")
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    res = asyncio.run(run(args))
    with open(args.out, "w") as f:
        json.dump(res, f)
    m = res["metrics"]
    print(
        f"[{args.label}] C={args.max_concurrency} "
        f"ok={m['completed']}/{m['completed'] + m['failed']} "
        f"out_tok/s={m['output_throughput']:.1f} "
        f"ttft={m['mean_ttft_ms']:.1f}ms tpot={m['mean_tpot_ms']:.2f}ms "
        f"outlen={m['mean_output_tokens']:.1f}"
    )
    if res["spec_decode"]:
        s = res["spec_decode"]
        print(
            f"       spec: accept_len={s['acceptance_length']:.3f} "
            f"accept_rate={s['acceptance_rate']:.1f}% "
            f"drafts={s['num_drafts']} draft_tok={s['num_draft_tokens']}"
        )


if __name__ == "__main__":
    main()
