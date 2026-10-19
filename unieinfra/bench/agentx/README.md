# AgentX arms (DeepSeek-V4.1, 4 GPUs)

End-to-end A/B of a tokenspeed build against vLLM 0.31 on the AgentX agent
workload. Every arm launches its own server on `GPUS`, runs the cells, saves
the server log and tears down. Results land in `results/` (ignored by git).

```
./arm_vllm.sh vllm031                                   # baseline
./arm_ts.sh ours IMAGE <recipe flags>                   # see REPORT-dsv41-h200-agentx.md
CONCS="32 16 8" ./arm_ts.sh ours_b IMAGE <flags>        # same arm, cells reordered
GSM=1 ./arm_ts.sh ours_gsm IMAGE <flags>                # + GSM8K gate after the cells
./prof_arm.sh prof IMAGE /path/to/traces <flags>        # torch traces of a c16 cell
python summarize.py results vllm=vllm031,vllm031b ours=ours,ours_b,ours_c
MAX_SEQS=128 CONCS="1 64 128" ./arm_vllm.sh vllm_hi        # high-concurrency cells
CONCS="1 64 128" python summarize.py results vllm=vllm_hi ours=ours_hi
```

- `run_cells.sh` flushes the prefix cache before each concurrency cell, warms
  up on a corpus that shares no prefix, and drives
  `bench/agent-slo/corpora/swe_smith_agent_sess32x8.jsonl` (256 requests,
  at least 128 output tokens, greedy) through `bench/agent-slo/drive.py`.
- Repeat arms in different cell orders and compare medians: one run of a
  cell moves by about 1 s, and the first cell after a launch pays JIT warmup.
- A profiled cell's duration is meaningless (trace export stalls the
  server); use it for attribution only.
