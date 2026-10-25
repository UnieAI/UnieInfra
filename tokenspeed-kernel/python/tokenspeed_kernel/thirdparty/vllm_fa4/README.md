# vllm_fa4

FlashAttention 4 (CuTe DSL) as vLLM v0.31.0 ships it in
`vllm/vllm_flash_attn/cute` (docker image `vllm/vllm-openai:v0.31.0`),
copied verbatim apart from the package path (`vllm.vllm_flash_attn.cute` ->
`tokenspeed_kernel.thirdparty.vllm_fa4`) and the dropped benchmark scripts.

The copy exists for one feature the `tokenspeed-fa4` wheel lacks: the Hopper
forward kernel serves head dims up to 512 (the wheel stops at 256), which
gemma-4's 512-wide global attention layers need. Its prefill there runs at
~350 TFLOPs, against ~70-130 TFLOPs for the Triton fallback.

Upstream: https://github.com/Dao-AILab/flash-attention (BSD-3-Clause, see
LICENSE) via https://github.com/vllm-project/vllm (Apache-2.0).
