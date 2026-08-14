# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (c) 2026 UnieAI
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Rebuild an agent corpus from REAL Claude Code traces, preserving the real
prefix-sharing structure.

Why not a synthetic preamble
----------------------------
The first version of this script pinned one tool set in front of BFCL user
turns and produced 98.5% shared prefix at 1024 input / 128 output tokens. The
shared fraction happened to be about right. Everything else was wrong by two
orders of magnitude. Measured over 393 sessions and 98,827 requests of
SemiAnalysis's `semianalysisai/cc-traces-weka-062126` (Claude Code proxy
traces, v7, >=20 requests per session, sub-agents included):

    input tokens   median 142,016   p10 32,704   p90 549,504   max 989,824
    output tokens  median    444                p90   2,659   max  59,903
    prefix reuse   median   99.1%   p10  83.2%  p90    99.9%
    requests/session median 86      p90 618

So real agent traffic runs at an input:output ratio near 320:1, not the 1:1 of
the InferenceX random shape or the 8:1 of the first synthetic corpus. Agent
inference is a prefill-and-reuse problem; decode is a rounding error on it.

How the sharing is reproduced
-----------------------------
Not by choosing a shared fraction -- by replaying the trace's own block
structure. Each trace request carries `hash_ids`, the KV-block hashes of its
prompt at `block_size` (64) tokens per block. Identical id means identical
content. This script maps each id to a deterministic pseudo-text chunk, so a
prompt is the concatenation of its blocks' chunks and two requests share
exactly the blocks the real traffic shared -- including partial and
non-prefix sharing, which a fixed-preamble corpus cannot express.

Output lengths come from the trace's own `out` field, per request.

What this cannot reproduce here
-------------------------------
KV costs 64 KB/token on Qwen3.5-27B, and the device pool at
--gpu-memory-utilization 0.92 is 34.32 GB, i.e. 562,298 tokens. One
median-sized real request (142,016 tokens) is 8.67 GB, a quarter of the whole
pool; a p90 request (549,504) is 33.54 GB and does not leave room for anything
else. Sessions are therefore filtered by --max-isl, and the filter is reported
in the output so no arm silently claims to be running the real distribution
when it is running its lower tail.

Usage:
    python3 unieinfra/bench/make_agent_corpus.py --max-isl 32768 \
        --num-prompts 256 --out /workspace/roy/bench-data/agent_real.jsonl
"""

import argparse
import itertools
import json
import pathlib
import statistics

DEFAULT_TRACES = pathlib.Path(
    "/workspace/.cache/hub/datasets--semianalysisai--cc-traces-weka-062126"
    "/snapshots/23f152f6f0f9399a85901b89a6458def0ef16729/traces.jsonl"
)

# One deterministic word per block id. The tokenizer is not asked to agree with
# the trace's block boundaries -- what must be preserved is that equal ids
# produce equal text and different ids do not, which is what makes the prefix
# structure real.
_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu"
).split()


_UNIT_TOKENS: int | None = None


def calibrate_unit(tokenizer) -> int:
    """How many real tokens one `word0000123 ` unit costs.

    The first version assumed 2 and hard-coded `tokens_per_block // 2`
    repetitions. The real cost on this tokenizer is 6-8, so every block came
    out about 4x too long: a corpus filtered to ISL <= 32768 produced ~100k
    token prompts and the engine rejected 104 of 128 requests with "input
    exceeds the engine's maximum input length". Measure it instead.
    """
    probe = "alpha0000123 " * 16
    return max(1, round(len(tokenizer(probe).input_ids) / 16))


def block_text(block_id: int, tokens_per_block: int) -> str:
    """A stable ~`tokens_per_block`-token chunk for one KV block id."""
    w = _WORDS[block_id % len(_WORDS)]
    unit = _UNIT_TOKENS or 2
    return f"{w}{block_id:07d} " * max(1, round(tokens_per_block / unit))


def session_end_at_or_before(bounds: list[int], limit: int) -> int:
    """Largest session boundary <= limit, so no session is cut in half."""
    ok = [b for b in bounds if 0 < b <= limit]
    return max(ok) if ok else limit


def flatten(requests):
    """Sub-agent groups nest their own `requests` list."""
    for r in requests:
        if isinstance(r.get("requests"), list):
            yield from flatten(r["requests"])
        elif "in" in r:
            yield r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=pathlib.Path, default=DEFAULT_TRACES)
    ap.add_argument(
        "--max-isl",
        type=int,
        default=32768,
        help="Drop requests whose prompt exceeds this. The real median is "
        "142016 tokens = 8.67 GB of KV, a quarter of this box's pool.",
    )
    ap.add_argument("--min-isl", type=int, default=1024)
    ap.add_argument(
        "--min-osl",
        type=int,
        default=0,
        help="Keep only requests generating at least this many tokens. The "
        "real output distribution is median 444 / p90 2659 / max 59903, so a "
        "corpus selected on ISL alone can land almost entirely in the short "
        "tail and never exercise decode. Still real requests -- a different "
        "slice, not synthetic -- so quote it alongside the ISL filter.",
    )
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--scan-sessions", type=int, default=2000)
    ap.add_argument("--tokenizer", default="/workspace/models/Qwen/Qwen3.5-27B")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    global _UNIT_TOKENS
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    _UNIT_TOKENS = calibrate_unit(tok)
    print(f"tokenizer unit cost: {_UNIT_TOKENS} tokens per block word")

    rows: list[dict] = []
    session_bounds: list[int] = []
    kept_in: list[int] = []
    kept_out: list[int] = []
    reuse: list[float] = []
    dropped_too_long = 0
    with args.traces.open() as f:
        for line in itertools.islice(f, args.scan_sessions):
            session_bounds.append(len(rows))
            trace = json.loads(line)
            tokens_per_block = int(trace.get("block_size", 64))
            seen: set[int] = set()
            for req in flatten(trace["requests"]):
                ids = req.get("hash_ids") or []
                if not ids:
                    continue
                hits = sum(1 for i in ids if i in seen)
                seen.update(ids)
                if not (args.min_isl <= req["in"] <= args.max_isl):
                    dropped_too_long += req["in"] > args.max_isl
                    continue
                if req["out"] < args.min_osl:
                    continue
                rows.append(
                    {
                        "prompt": "".join(block_text(i, tokens_per_block) for i in ids),
                        "output_tokens": max(1, int(req["out"])),
                    }
                )
                kept_in.append(req["in"])
                kept_out.append(req["out"])
                reuse.append(hits / len(ids))

    # Sessions must stay CONTIGUOUS and in order. The reuse figure below is
    # computed against the trace's full history, but the engine can only hit
    # blocks that some earlier request in THIS corpus already wrote. An even
    # stride across sessions reports 97.6% and delivers 13.3%: the predecessor
    # that established each prefix is not in the file. Truncating at a session
    # boundary keeps every kept request's ancestors present.
    #
    # (Two earlier explanations for a 0%-reuse corpus were wrong and are worth
    # not repeating: it is not that long outputs lack sharing -- every output
    # bucket sits at ~99% -- and it is not stride bias. It is the ISL cap:
    # requests with out>=1024 have median input 230,848, so capping input at
    # 32k keeps only session openers, which are the requests that have no
    # ancestor by definition.)
    if not rows:
        raise SystemExit("no requests survived the filters")
    if len(rows) > args.num_prompts:
        cut = session_end_at_or_before(session_bounds, args.num_prompts)
        rows, kept_in = rows[:cut], kept_in[:cut]
        kept_out, reuse = kept_out[:cut], reuse[:cut]

    measured = [len(tok(r["prompt"]).input_ids) for r in rows[:32]]
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} requests -> {args.out}")
    print(f"  ISL filter   : {args.min_isl}..{args.max_isl} tokens")
    if args.min_osl:
        print(f"  OSL filter   : >= {args.min_osl} tokens")
    print(f"  dropped >max : {dropped_too_long} requests")
    print(f"  input tokens : median {statistics.median(kept_in):,.0f}")
    print(f"  output tokens: median {statistics.median(kept_out):,.0f}")
    print(f"  prefix reuse : median {100 * statistics.median(reuse):.1f}%")
    print(
        f"  MEASURED prompt tokens (first 32): median {statistics.median(measured):,.0f}"
        f"  max {max(measured):,}"
    )
    print(
        "  NOTE: this is the lower tail of the real distribution "
        f"(real median ISL is 142,016); quote the filter with any result."
    )


if __name__ == "__main__":
    main()
