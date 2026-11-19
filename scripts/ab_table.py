"""Print RPM / output tps / TTFT / ITL for result dirs matching a glob: ab_table.py 'unieinfra-R10*'."""
import glob, json, os, sys
for d in sorted(glob.glob(os.path.join("/home/smc/workspace/roy/results/agentx", sys.argv[1]))):
    f = glob.glob(os.path.join(d, "**/profile_export_aiperf.json"), recursive=True)
    if not f:
        print(os.path.basename(d), "no data"); continue
    j = json.load(open(f[0]))
    g = lambda k, s="avg": (j[k][s] if isinstance(j.get(k), dict) else j.get(k))
    print(f"{os.path.basename(d)[:28]:28s} rpm {60*g('request_throughput'):6.1f} out {g('output_token_throughput'):7.1f} "
          f"ttft {g('time_to_first_token')/1000:6.2f}s itl {g('inter_token_latency'):6.2f} itl_p99 {g('inter_token_latency','p99'):6.1f}")
