"""How much of each request's prompt could a perfect prefix cache serve?
Uses AiPerf per-request records: for each conversation, previous turn's prompt+output is the reusable context."""
import json, sys, collections
f = sys.argv[1]
recs = [json.loads(l) for l in open(f)]
by = collections.defaultdict(list)
for r in recs:
    m, x = r["metadata"], r.get("metrics", {})
    g = lambda k: x.get(k, {}).get("value", 0) if isinstance(x.get(k), dict) else 0
    by[m["conversation_id"]].append(dict(t=m["turn_index"], ph=m.get("benchmark_phase"), p=g("usage_prompt_tokens"),
        c=g("usage_prompt_cache_read_tokens"), o=g("output_sequence_length"), depth=m.get("agent_depth"), kind=m.get("source_kind")))
tot = dict(p=0, c=0, prevctx=0, first_p=0, first_c=0, n=0, nfirst=0, shrink=0, shrink_tok=0)
kinds = collections.Counter()
for cid, rs in by.items():
    rs.sort(key=lambda r: r["t"])
    prev = None
    for r in rs:
        if r["ph"] != "profiling": prev = r; continue
        kinds[(r["kind"], r["depth"])] += 1
        tot["p"] += r["p"]; tot["c"] += r["c"]; tot["n"] += 1
        if prev is None:
            tot["first_p"] += r["p"]; tot["first_c"] += r["c"]; tot["nfirst"] += 1
        else:
            ctx = prev["p"] + prev["o"]
            if r["p"] < ctx: tot["shrink"] += 1; tot["shrink_tok"] += ctx - r["p"]
            tot["prevctx"] += min(ctx, r["p"])
        prev = r
n = tot["n"]
print(f"profiling requests {n}, conversations {len(by)}, first-seen {tot['nfirst']}; kinds {dict(kinds)}")
print(f"achieved hit {tot['c']/tot['p']*100:.2f}%  ({(tot['p']-tot['c'])/n/1e3:.2f}K uncached tok/req)")
print(f"if prev turn prompt+output were all cached: {tot['prevctx']/tot['p']*100:.2f}%  (new tokens {(tot['p']-tot['prevctx'])/n/1e3:.2f}K/req)")
print(f"first-seen requests: {tot['first_p']/1e6:.2f}M prompt tokens, hit {tot['first_c']/max(tot['first_p'],1)*100:.1f}%  = {tot['first_p']/tot['p']*100:.1f}% of all prompt tokens")
print(f"turns whose prompt is shorter than prev prompt+output (context edited/truncated): {tot['shrink']} ({tot['shrink_tok']/1e6:.2f}M tokens)")
