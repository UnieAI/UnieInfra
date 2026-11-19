"""Summarize AgentX results: one row per run directory."""
import glob, json, os, sys
# (metric, stat, header, scale): RPM = completed requests per minute.
KEYS = [("request_throughput", "avg", "RPM", 60.0),
        ("output_token_throughput", "avg", "out tok/s"),
        ("output_token_throughput_per_user", "p50", "tok/s/user p50"),
        ("time_to_first_token", "p50", "TTFT p50 ms"),
        ("time_to_first_token", "p90", "TTFT p90 ms"),
        ("inter_token_latency", "p50", "ITL p50 ms"),
        ("request_count", "avg", "reqs"),
        ("request_error_rate", "avg", "err %")]
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
print("run".ljust(44) + "".join(k[2].rjust(15) for k in KEYS))
for d in sorted(glob.glob(f"/home/smc/workspace/roy/results/agentx/{pat}")):
    f = os.path.join(d, "aiperf_artifacts/profile_export_aiperf.json")
    if not os.path.exists(f):
        print(os.path.basename(d)[:44].ljust(44) + "  (running / no result)")
        continue
    j = json.load(open(f))
    row = os.path.basename(d).rsplit("-", 1)[0][:44].ljust(44)
    for key in KEYS:
        v = j.get(key[0], {}).get(key[1])
        if isinstance(v, (int, float)) and len(key) > 3:
            v *= key[3]
        row += (f"{v:15.1f}" if isinstance(v, (int, float)) else "-".rjust(15))
    print(row)
