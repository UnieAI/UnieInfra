# SPDX-License-Identifier: Apache-2.0
"""E40b — a REAL multi-turn client: conversation growth, not prompt repetition.

E39b could not answer whether prefix caching helps agent conversations because
this tree has no multi-turn sampler. ``MTBenchDataset`` takes ``item["turns"][0]``
and says so in its own docstring ("We create a single turn dataset for
MT-Bench"); ``ConversationDataset`` takes only ``conv[0]``/``conv[1]``. The
*data* is multi-turn -- only the sampler is not. This client reads the full
``turns`` list and drives it the way an agent loop actually runs:

    turn 1: [user_1]                                   -> assistant_1
    turn 2: [user_1, assistant_1, user_2]              -> assistant_2
    turn 3: [user_1, assistant_1, user_2, assistant_2, user_3] -> ...

so turn *k*'s prompt literally contains every previous turn, and the reusable
prefix grows monotonically. Many such sessions run concurrently; within a
session turns are strictly sequential (turn k+1 cannot start before turn k's
response exists), which is the property that makes this a conversation rather
than a batch.

Reported per TURN INDEX (not per request): TTFT, TPOT, output throughput, and
the engine's own prefix-cache hit rate, so the growth curve is visible.

Cache hit rate comes from the server's Prometheus counters
(``vllm:prefix_cache_queries_total`` / ``vllm:prefix_cache_hits_total``),
sampled before and after the measured phase, so it reflects exactly this run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np


@dataclass
class TurnResult:
    session: int
    turn: int
    success: bool = False
    error: str = ""
    ttft: float = 0.0
    latency: float = 0.0
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    text: str = ""
    itl: list[float] = field(default_factory=list)


async def _chat(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    tr: TurnResult,
) -> TurnResult:
    st = time.perf_counter()
    last = st
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                tr.error = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return tr
            buf = b""
            async for chunk in resp.content.iter_any():
                buf += chunk
                while b"\n\n" in buf:
                    raw, buf = buf.split(b"\n\n", 1)
                    for line in raw.split(b"\n"):
                        line = line.strip()
                        if not line.startswith(b"data: "):
                            continue
                        body = line[6:]
                        if body == b"[DONE]":
                            continue
                        now = time.perf_counter()
                        data = json.loads(body)
                        if choices := data.get("choices"):
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content") or delta.get(
                                "reasoning_content"
                            )
                            if content:
                                if tr.ttft == 0.0:
                                    tr.ttft = now - st
                                else:
                                    tr.itl.append(now - last)
                                last = now
                                tr.text += content
                        elif usage := data.get("usage"):
                            tr.output_tokens = usage.get("completion_tokens")
                            tr.prompt_tokens = usage.get("prompt_tokens")
                            details = usage.get("prompt_tokens_details") or {}
                            tr.cached_tokens = details.get("cached_tokens")
            tr.latency = last - st
            tr.success = True
    except Exception as e:  # noqa: BLE001
        tr.error = f"{type(e).__name__}: {e}"
    return tr


async def _scrape_cache(session: aiohttp.ClientSession, base: str):
    try:
        async with session.get(f"{base}/metrics") as r:
            if r.status != 200:
                return None
            text = await r.text()
    except Exception:  # noqa: BLE001
        return None
    q = h = 0.0
    found = False
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{")[0].split(None, 1)[0]
        try:
            val = float(line.rsplit(None, 1)[-1])
        except ValueError:
            continue
        if name == "vllm:prefix_cache_queries_total":
            q += val
            found = True
        elif name == "vllm:prefix_cache_hits_total":
            h += val
            found = True
    return {"queries": q, "hits": h} if found else None


def _load_sessions(path: str, n_sessions: int, n_turns: int, seed: int):
    """Read MT-Bench-style rows and keep the FULL turns list.

    Accepts either the HF ``philschmid/mt-bench`` parquet layout dumped to
    jsonl, or any jsonl with a ``turns`` list of strings.
    """
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            turns = rec.get("turns")
            if isinstance(turns, list) and len(turns) >= 1:
                turns = [t for t in turns if isinstance(t, str) and t.strip()]
                if turns:
                    rows.append(turns)
    if not rows:
        raise SystemExit(f"no usable multi-turn rows in {path}")
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for i in range(n_sessions):
        turns = rows[i % len(rows)]
        # MT-Bench ships 2 turns per question; to reach n_turns we continue the
        # SAME conversation with further questions, which is what an agent does
        # (the history keeps growing) rather than restarting a new prompt.
        seq = list(turns)
        j = 1
        while len(seq) < n_turns:
            seq.extend(rows[(i + j) % len(rows)])
            j += 1
        out.append(seq[:n_turns])
    return out


async def run(args) -> dict[str, Any]:
    sessions = _load_sessions(args.data, args.sessions, args.turns, args.seed)
    base = f"http://{args.host}:{args.port}"
    url = f"{base}/v1/chat/completions"
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as http:
        if args.reset_prefix_cache:
            try:
                async with http.post(f"{base}/reset_prefix_cache") as r:
                    await r.text()
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] reset_prefix_cache failed: {e}")

        cache_before = await _scrape_cache(http, base)
        sem = asyncio.Semaphore(args.max_concurrency)
        results: list[TurnResult] = []
        lock = asyncio.Lock()

        async def drive(sid: int, turns: list[str]):
            history: list[dict[str, str]] = []
            for k, user_msg in enumerate(turns):
                history.append({"role": "user", "content": user_msg})
                tr = TurnResult(session=sid, turn=k)
                payload = {
                    "model": args.model,
                    "messages": list(history),
                    "max_completion_tokens": args.output_len,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
                }
                async with sem:
                    await _chat(http, url, payload, tr)
                async with lock:
                    results.append(tr)
                if not tr.success:
                    return
                # The assistant reply becomes part of the next turn's prompt --
                # this is what makes the reusable prefix grow.
                history.append({"role": "assistant", "content": tr.text})

        t0 = time.perf_counter()
        await asyncio.gather(*[drive(i, t) for i, t in enumerate(sessions)])
        dur = time.perf_counter() - t0
        cache_after = await _scrape_cache(http, base)

    ok = [r for r in results if r.success]
    n_out = sum((r.output_tokens or 0) for r in ok)

    per_turn: dict[int, dict[str, Any]] = {}
    for k in range(args.turns):
        rs = [r for r in ok if r.turn == k]
        if not rs:
            continue
        ttfts = np.array([r.ttft * 1000 for r in rs])
        tpots = np.array(
            [
                (r.latency - r.ttft) / (r.output_tokens - 1) * 1000
                for r in rs
                if (r.output_tokens or 0) > 1
            ]
            or [0.0]
        )
        prompt_toks = [r.prompt_tokens for r in rs if r.prompt_tokens]
        cached = [r.cached_tokens for r in rs if r.cached_tokens is not None]
        per_turn[k] = {
            "n": len(rs),
            "mean_ttft_ms": float(ttfts.mean()),
            "median_ttft_ms": float(np.median(ttfts)),
            "p99_ttft_ms": float(np.percentile(ttfts, 99)),
            "mean_tpot_ms": float(tpots.mean()),
            "mean_prompt_tokens": float(np.mean(prompt_toks)) if prompt_toks else 0.0,
            "mean_output_tokens": float(np.mean([r.output_tokens or 0 for r in rs])),
            "mean_cached_tokens": float(np.mean(cached)) if cached else None,
            "cached_frac_pct": (
                100.0 * float(np.sum(cached)) / float(np.sum(prompt_toks))
                if cached and prompt_toks
                else None
            ),
        }

    cache = None
    if cache_before and cache_after:
        dq = cache_after["queries"] - cache_before["queries"]
        dh = cache_after["hits"] - cache_before["hits"]
        cache = {
            "queries": dq,
            "hits": dh,
            "hit_rate_pct": (100.0 * dh / dq) if dq else 0.0,
        }

    return {
        "config": {
            "label": args.label,
            "arm": args.arm,
            "sessions": args.sessions,
            "turns": args.turns,
            "max_concurrency": args.max_concurrency,
            "output_len": args.output_len,
            "reset_prefix_cache": args.reset_prefix_cache,
            "enable_thinking": args.enable_thinking,
            "data": args.data,
            "seed": args.seed,
        },
        "metrics": {
            "duration_s": dur,
            "completed_turns": len(ok),
            "failed_turns": len(results) - len(ok),
            "total_output_tokens": n_out,
            "output_throughput": n_out / dur if dur else 0.0,
            "mean_ttft_ms": float(np.mean([r.ttft * 1000 for r in ok])) if ok else 0.0,
        },
        "prefix_cache": cache,
        "per_turn": {str(k): v for k, v in sorted(per_turn.items())},
        "turns_raw": [
            {
                "session": r.session,
                "turn": r.turn,
                "success": r.success,
                "error": r.error,
                "ttft_ms": r.ttft * 1000,
                "latency_ms": r.latency * 1000,
                "prompt_tokens": r.prompt_tokens,
                "cached_tokens": r.cached_tokens,
                "output_tokens": r.output_tokens,
            }
            for r in results
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8324)
    p.add_argument("--model", default="q8")
    p.add_argument("--data", required=True)
    p.add_argument("--sessions", type=int, default=32)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--reset-prefix-cache", action="store_true")
    p.add_argument("--arm", default="")
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    res = asyncio.run(run(args))
    with open(args.out, "w") as f:
        json.dump(res, f)
    m = res["metrics"]
    pc = res["prefix_cache"]
    print(
        f"[{args.label}] sessions={args.sessions} turns={args.turns} "
        f"C={args.max_concurrency} ok={m['completed_turns']} "
        f"out_tok/s={m['output_throughput']:.1f} ttft={m['mean_ttft_ms']:.1f}ms "
        f"hit={pc['hit_rate_pct']:.1f}%"
        if pc
        else ""
    )
    for k, v in res["per_turn"].items():
        print(
            f"   turn {k}: ttft={v['mean_ttft_ms']:.1f}ms "
            f"prompt_tok={v['mean_prompt_tokens']:.0f} "
            f"cached={v['mean_cached_tokens']} tpot={v['mean_tpot_ms']:.2f}ms"
        )


if __name__ == "__main__":
    main()
