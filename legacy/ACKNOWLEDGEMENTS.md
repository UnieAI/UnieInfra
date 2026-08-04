# Acknowledgements

UnieInfra builds on a broad open source ecosystem. We are grateful to the
projects and contributors whose work is acknowledged in our package-level
third-party notices.

UnieInfra's design is to **absorb** valuable components from other inference
engines as owned in-tree source rather than depend on them as libraries. That
makes correct, visible attribution more important, not less: an absorbed
component is redistributed by us, so its license travels with it and its
provenance is recorded per component. See `NOTICE` for the authoritative record.

## unieinfra-vllm

The `unieinfra-vllm` engine base is a **hard fork of
[vLLM](https://github.com/vllm-project/vllm)** (Apache-2.0) at commit
`d02df748bf9efd99022f1a062597dc3cb3808485`, vendored as source on 2026-07-25.
We do not sync upstream. The overwhelming majority of that tree is the work of
the vLLM Team and vLLM's contributors, and our thanks are proportionate: it is
by some distance the largest single body of work in this repository. Attribution
detail — the upstream license, the preserved upstream README, and the complete
list of the files we removed, renamed and modified — is in
`unieinfra-vllm/PROVENANCE.md`.

UnieInfra is an independent project and is **not affiliated with, sponsored by,
or endorsed by** the vLLM project.

## tokenspeed

The `tokenspeed` Python package acknowledges and thanks:

- `NVIDIA/TensorRT-LLM`
- `meituan-longcat/SGLang-FluentLLM`
- `vllm-project/vllm`
- `ModelTC/lightllm`
- `sgl-project/sglang`

## tokenspeed-kernel

The `tokenspeed-kernel` package acknowledges and thanks:

- `NVIDIA/TensorRT-LLM`
- `triton-lang/triton`
- `deepseek-ai/DeepEP`
- `deepseek-ai/DeepGEMM`
- `Dao-AILab/flash-attention`
- `deepseek-ai/FlashMLA`
- `Dao-AILab/fast-hadamard-transform`
- `sgl-kernel`
- `flashinfer-ai/flashinfer`

## unieinfra-kernel

The `unieinfra-kernel` package acknowledges and thanks:

- `vllm-project/vllm` (Apache-2.0) — the decode GEMV in
  `unieinfra_kernel/ops/gemm/decode_gemv_triton.py` ports *techniques* from
  vLLM's `csrc/libtorch_stable/bf16_skinny_gemm.cu`: the block-per-output-column
  mapping with no split-K, the deliberately narrow column tile (to minimize wave
  quantization), the 128-bit per-thread weight load, the evict-first weight
  cache hint, and the constexpr specialization on the token count. No code was
  copied; the techniques were re-expressed in portable Triton.
- `triton-lang/triton`

## tokenspeed-mla

The `tokenspeed-mla` package acknowledges and thanks:

- `NVIDIA/cutlass`
- `flashinfer-ai/flashinfer`

## License Notices

We follow open source license requirements by keeping package-level
`THIRDPARTYNOTICES` files with the relevant third-party notices and license
texts. For Python packages built with setuptools, these files are included in
the package metadata with:

```toml
[tool.setuptools]
license-files = ["LICENSE", "THIRDPARTYNOTICES"]
```
