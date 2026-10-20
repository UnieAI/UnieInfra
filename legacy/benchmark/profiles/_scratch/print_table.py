import json
import sys

d = json.load(open(sys.argv[1]))
for row in d:
    nm = row["name"][:75]
    dur = row.get("gpu__time_duration.sum") or 0
    dbw = row.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed") or 0
    sm = row.get("sm__throughput.avg.pct_of_peak_sustained_elapsed") or 0
    occ = row.get("sm__warps_active.avg.pct_of_peak_sustained_active") or 0
    l2 = row.get("lts__t_sector_hit_rate.pct") or 0
    grid = row.get("launch__grid_size")
    block = row.get("launch__block_size")
    print(
        f"{row['action']:3d} dur={dur:8.0f}ns dram%={dbw:6.2f} sm%={sm:6.2f} occ%={occ:6.2f} l2hit%={l2:6.2f} grid={grid} blk={block}  {nm}"
    )
