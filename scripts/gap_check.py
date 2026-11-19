"""Per cell: how much of each client slot is idle waiting for trace delays vs in-flight, profiling phase only."""
import json, sys, statistics as st, collections
for path in sys.argv[1:]:
    R = [json.loads(l) for l in open(path + "/aiperf_artifacts/profile_export.jsonl")]
    R = [r for r in R if r["metadata"].get("benchmark_phase") != "warmup"]
    m = lambda r, k: r["metadata"][k]
    t0 = min(m(r, "request_start_ns") for r in R); t1 = max(m(r, "request_end_ns") for r in R)
    dur = (t1 - t0) / 1e9
    busy = sum(m(r, "request_end_ns") - m(r, "request_start_ns") for r in R) / 1e9
    by = collections.defaultdict(list)
    for r in R: by[m(r, "conversation_id")].append(r)
    gaps = []
    for rs in by.values():
        rs.sort(key=lambda r: m(r, "request_start_ns"))
        for a, b in zip(rs, rs[1:]):
            g = (m(b, "request_start_ns") - m(a, "request_end_ns")) / 1e9
            if g >= 0: gaps.append(g)
    lat = st.mean((m(r, "request_end_ns") - m(r, "request_start_ns")) / 1e9 for r in R)
    print(f"{path.split('/')[-1][:24]:24s} n={len(R)} dur={dur:.0f}s avg_inflight={busy/dur:5.1f} lat={lat:5.1f}s "
          f"gap p50={st.median(gaps):.2f}s mean={st.mean(gaps):.2f}s idle_frac={sum(gaps)/(sum(gaps)+busy):.2f}" if gaps else path)
