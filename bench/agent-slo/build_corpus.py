"""Build a multi-session agent benchmark corpus from public SWE-agent trajectories.

PROVENANCE -- reproduce the inputs with:
  hf download nebius/SWE-agent-trajectories --repo-type dataset \
     --include "data/train-00000-of-00012.parquet" --local-dir /workspace/roy/.pubdata/nebius-swe-agent
  hf download SWE-bench/SWE-smith-trajectories --repo-type dataset \
     --include "data/train-00000-of-00008.parquet" --local-dir /workspace/roy/.pubdata/swesmith
Then:
  python3 build.py nebius   .../nebius-swe-agent/data/train-00000-of-00012.parquet 16 16 12500 \
     /workspace/roy/bench-data/swe_agent_nebius_sess16x16.jsonl
  python3 build.py swesmith .../swesmith/data/train-00000-of-00008.parquet        32  8 12500 \
     /workspace/roy/bench-data/swe_smith_agent_sess32x8.jsonl
Licences: nebius/SWE-agent-trajectories CC-BY-4.0; SWE-bench/SWE-smith-trajectories MIT.

Prompt at turn k = the full conversation rendered up to and including user message k,
so turn k+1's prompt CONTAINS turn k's prompt as an exact prefix (history is resent),
which is what a real agent client does.

Length is capped by CHOOSING AN EARLIER TURN -- never by truncating text. Truncation
would make different sessions collide on identical strings and hand the cache free hits.
"""

import hashlib
import json
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

TOKN = "/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
tok = AutoTokenizer.from_pretrained(TOKN)


def norm_nebius(path, limit):
    f = pq.ParquetFile(path)
    out = []
    for b in f.iter_batches(batch_size=256):
        for r in b.to_pylist():
            msgs = []
            for m in r["trajectory"]:
                role = {"ai": "assistant"}.get(m["role"], m["role"])
                txt = m["system_prompt"] if role == "system" else m["text"]
                msgs.append((role, txt or ""))
            out.append((r["instance_id"], r["instance_id"].split("-")[0], msgs))
            if len(out) >= limit:
                return out
    return out


def norm_swesmith(path, limit):
    f = pq.ParquetFile(path)
    out = []
    for b in f.iter_batches(batch_size=256):
        for r in b.to_pylist():
            out.append(
                (
                    r["traj_id"],
                    r["instance_id"].split(".")[0],
                    [(m["role"], m["content"] or "") for m in r["messages"]],
                )
            )
            if len(out) >= limit:
                return out
    return out


def boundaries(msgs, cap):
    """Cumulative token length at each user-message boundary; keep those under cap."""
    parts = [f"[{r}]\n{t}\n\n" for r, t in msgs]
    lens = [len(x) for x in tok(parts, add_special_tokens=False)["input_ids"]]
    cum = 0
    out = []
    for i, (role, _) in enumerate(msgs):
        cum += lens[i]
        if role == "user" and cum <= cap:
            out.append((i, cum))
    return parts, out


def build(kind, src, nsess, nturn, cap, outpath, offset=0):
    trajs = norm_swesmith(src, 2400) if kind == "swesmith" else norm_nebius(src, 2400)
    picked, seen_repo = [], set()
    for tid, repo, msgs in trajs:
        if repo in seen_repo:
            continue  # one session per repo -> real diversity
        seen_repo.add(repo)
        parts, bnd = boundaries(msgs, cap)
        if len(bnd) < nturn:
            continue
        if len(picked) < offset:
            picked.append(
                (tid, parts, bnd[-nturn:])
            )  # skip already-used sessions (fresh-corpus builds)
            continue
        picked.append(
            (tid, parts, bnd[-nturn:])
        )  # LAST nturn consecutive turns under cap
        if len(picked) >= nsess + offset:
            break
    picked = picked[offset:]
    assert len(picked) == nsess, f"only found {len(picked)} sessions"

    sessions = []
    for tid, parts, bnd in picked:
        sessions.append([(tid, "".join(parts[: i + 1]).rstrip(), n) for i, n in bnd])

    rows = []  # round-robin interleave across sessions
    for k in range(nturn):
        for s in range(nsess):
            tid, prompt, ntok = sessions[s][k]
            rows.append(
                {"prompt": prompt, "session": tid, "turn": k, "n_prompt_tokens": ntok}
            )
    with open(outpath, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    h = {hashlib.sha256(r["prompt"].encode()).hexdigest() for r in rows}
    print(
        f"wrote {outpath}: {len(rows)} requests, {len(h)} unique prompts, {nsess} sessions x {nturn} turns"
    )


if __name__ == "__main__":
    build(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
        sys.argv[6],
        int(sys.argv[7]) if len(sys.argv) > 7 else 0,
    )
