# Inkling 2 Mini SM12x performance optimizations

This opt-in runtime mod applies to the vLLM image pinned by the
[Inkling 2 Mini DFlash2 recipes](../../recipes/inkling-2-mini-dflash2.yaml)
(`vllm 0.1.dev25855+g7cb5dae22.cu130`) and layers cleanly on top of
`mods/inkling2-dflash2`.

Each optimization is enabled via container environment variables so a single
patch supports both strictly target-lossless and higher-throughput MXFP8-dense
configurations:

| Variable | Values | Effect |
| --- | --- | --- |
| `INKLING2_OPT_DFLASH_LMHEAD` | `""`, `mxfp8`, `nvfp4` | Online-quantizes a draft-only copy of `lm_head` via Humming W4A16/W8A16. Target verification still uses the untouched BF16 `lm_head` (100% lossless to target outputs). |
| `INKLING2_OPT_DFLASH_BACKBONE` | `""`, `mxfp8`, `nvfp4` | Online-quantizes the 5-layer DFlash2 draft backbone (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`, `fc`, and fused context KV) via Humming. 100% lossless to target outputs. |
| `INKLING2_OPT_MOE` | `""`, `mxfp8_only`, `w4a16`, `w4a16_sk` | `mxfp8_only` keeps NVFP4 layers on deterministic Humming W4A4 (caching `moe_align_block_size` between `w13` and `w2`) and switches the 4 MXFP8 coda layers from Triton to Humming MXFP8 W8A16. `w4a16` / `w4a16_sk` run all 24 layers through `HummingExperts` W{4,8}A16. |
| `INKLING2_OPT_TARGET_DENSE` | `""`, `mxfp8`, `mxfp8_all` | Online block-32 MXFP8 W8A16 quantization of Inkling 2 Mini's dense BF16 attention projections (`qkvr`, `wo_ud`). `mxfp8_all` also quantizes the target `lm_head` to MXFP8 W8A16. |

Reapplying the mod to the same fresh container is safe and idempotent.
See [Inkling 2 Mini on DGX Spark](../../docs/inkling-2-mini-spark-findings.html)
for controlled single-Spark measurements.
