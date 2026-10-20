# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""First-party ``gemm`` kernels.

Two decode bands are covered, with disjoint shape traits so they never compete:

* :mod:`unieinfra_kernel.ops.gemm.decode_gemv_triton` (``m_le_2``) — the
  decode-tuned (small-M, narrow-N) GEMV that recovers DRAM bandwidth lost by
  the generic cuBLAS ``gemvx`` template on batch-1 weight projections (E33),
  plus its fp8 and nvfp4 weight-only variants.
* :mod:`unieinfra_kernel.ops.gemm.decode_batch_gemm_triton`
  (``m_min_16``/``m_le_256``) — the decode-*batch* GEMM that recovers the
  machine E44 measured the incumbent CUTLASS tiles losing to wave
  quantization at serving concurrency (E45).

Discovery imports every non-underscore module here, so the kernels
self-register.
"""
