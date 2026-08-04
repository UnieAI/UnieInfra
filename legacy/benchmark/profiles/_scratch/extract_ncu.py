#!/usr/bin/env python3
import json
import sys

sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.2.1/extras/python")
import ncu_report

path = sys.argv[1]
ctx = ncu_report.load_report(path)

METRICS = [
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__time_duration.sum",
    "dram__bytes.sum.per_second",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__waves_per_multiprocessor",
]

out = []
for ri in range(ctx.num_ranges()):
    r = ctx.range_by_idx(ri)
    for ai in range(r.num_actions()):
        a = r.action_by_idx(ai)
        row = {"range": ri, "action": ai, "name": a.name()}
        # full/demangled name if available
        try:
            row["name_demangled"] = a.name(ncu_report.IAction.NameBase_DEMANGLED)
        except Exception:
            pass
        for m in METRICS:
            try:
                mm = a.metric_by_name(m)
                row[m] = mm.value() if mm is not None else None
            except Exception as e:
                row[m] = f"ERR:{e}"
        # roofline / bound classification via rule results if present
        try:
            row["nvtx_ranges"] = {}
            ns = a.nvtx_state()
            for did in ns.domains():
                dom = ns.domain_by_id(did)
                row["nvtx_ranges"][dom.name()] = list(dom.push_pop_ranges())
        except Exception:
            pass
        out.append(row)

print(json.dumps(out, indent=1, default=str))
