## V4.1 prefill TBO: split mixed windows too

Stacked on #44 (`mi350p-final`).

A mixed round (`--enable-mixed-batch`) skipped the prefill two-micro-batch overlap, so letting decodes ride a prefill round cost more prefill time than it saved (−2 to −4%). This PR keeps the overlap in mixed rounds:

- The split always lands inside the extend rows. Micro-batch A is a pure prefill; B is the rest of the extend rows plus every decode request (decode rows follow all extend rows, so B is contiguous).
- Each micro-batch runs with its own forward mode in the context (A `EXTEND`, B `MIXED`) and its own attention metadata; B's decode rows write the same cache slots as the whole window.
- Second commit: B's decode row count is computed from the init arguments, not the current metadata (preparing A replaces it).

### Tests
- `test_mixed_micro_batches_keep_every_decode_row_in_the_second` (positions, request mapping, `num_tokens`, SWA/compressor write slots); `test/runtime/test_deepseek_v41_cache.py` 158 passed.
- GSM8K (CI config, T=0, 1,319 problems) while 40K-token cold prompts keep mixed rounds splitting: **0.9052** (baseline 0.9045, gate 0.90). All 4 ranks log `prefill TBO: first mixed window split at row 4008 of 8192`.

### Results (4× MI350P, DeepSeek-V4.1-Flash TP4, AgentX stress 600 s, same-day interleaved vs #44)

| c | RPM | ITL mean | ITL p99 | TTFT |
|---|---|---|---|---|
| 1 | 18.0 vs 18.9 | 5.45 vs 5.38 ms | 29.6 vs 28.8 ms | 0.56 vs 0.44 s |
| 4 | 51.6 vs 52.8 | 12.1 vs 17.5 ms | 74 vs 359 ms | 1.87 vs 0.92 s |
| 16 (2 pairs) | 85.2 vs 87.9 | 17.8 vs 20.6 ms | 83 vs 156 ms | 1.6 vs 1.2 s |
| 32 | 129.6 vs 126.3 | 21.5 vs 22.6 ms | 62.5 vs 71.2 ms | 0.67 vs 0.49 s |
| 64 | **174.9 vs 148.7 (+18%)** | 34.7 vs 43.4 ms | 82 vs 167 ms | 1.33 vs 1.17 s |
| 128 | **159.8** (baseline crashed twice that day; previous 136.4) | 80.5 ms | 134.5 ms | 10.4 s |

Recommendation: `--enable-mixed-batch` for c ≥ 32; c ≤ 16 is within noise and keeps a lower TTFT without it.

