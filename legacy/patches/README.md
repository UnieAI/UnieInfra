# patches/

## turbomind-w4a16-sm120.patch

w4a16 GEMM for sm_120 in TurboMind (`third_party/lmdeploy`). Verified on an
RTX PRO 6000 Blackwell: 2 to 222 registered kernels, 28/28 numerics against a
dequant reference, 2.3-3.3x on decode GEMMs. Apply with `git am` inside the
submodule. Write-ups: log E46 (port and the two gates), E47 (throughput),
E49 (accuracy of the quantization scheme).

### Correction to a claim inside this patch

The patch's own comments and commit message say the bf16 blocks matter because
"upstream forces fp16 for AWQ (turbomind/converter.py sets dtype=float16 when
model_format=='awq')". **That was written from memory and is not true** --
searching `lmdeploy/turbomind/deploy/` for any fp16 forcing keyed on AWQ returns
nothing. The patch file is left as the historical record rather than rewritten.

What is still true, and is the actual reason for those blocks: `Config_U4_d`
hardcoded `Operand_A<half>` and `Tc = half`, so the *dense* w4a16 path could not
serve bf16 activations at all. Measured, not remembered: registered dense bf16
kernels went from 0 to 58, and the numerics sweep covers bf16 at group 128 and
32 across M=1 to 4096.

### What w4a16 needs from you, and what it does itself

**Quantization is not automatic.** Produce an AWQ checkpoint first:

```sh
lmdeploy lite auto_awq <model> --w-bits 4 --w-group-size 128
```

Defaults are `calib_dataset=wikitext2`, `calib_samples=128`, `calib_seqlen=2048`,
`search_scale=False`. This step is where the accuracy cost is decided; E49
measured +7.2% perplexity for the *uncalibrated* scheme, which is the floor of
the range, not what a calibrated checkpoint gives.

**Layout conversion is automatic.** `LinearWeight::prepare()` reads
`weight_format` from the checkpoint's quant config and, for a u4 weight, calls
`GetConverters` to repack into the MMA fragment layout at load time. No flags,
and the group size comes from the checkpoint (`weight_format.block_sizes[0]`)
rather than from a launch argument. That path was already correct on sm_120
before this patch: it keys on the MMA layout with an open-ended `sm >= 80`, not
on the arch tag that the kernel registry used.
