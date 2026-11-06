# glm53-nvidia-mtp-unquantized

Fixes MTP drafter loading for `nvidia/GLM-5.3-Flash-NVFP4` by excluding its
BF16 MTP layers from ModelOpt quantization. Layer indices come from
`num_hidden_layers` and `num_nextn_predict_layers`; they are not hard-coded.

## Why it is needed

The checkpoint omits its BF16 MTP layer from `quantization_config.ignore`.
vLLM then allocates packed NVFP4 experts and fails in `_load_w2` when loading
unpacked BF16 weights: a sharded width of 1024 does not fit a width of 512.

Safetensors headers inspected on 2026-09-22:

| Layer | Expert tensors | Scale tensors |
| --- | --- | --- |
| 45 (MTP) | 864 BF16 | None |
| 44 (target) | 864 U8, packed NVFP4 | 864 F8_E4M3 + 1728 F32 |

The helper checks for nextn layers, not a specific `model_type`, since the
drafter configuration may differ from the target. Use this mod only for a
checkpoint whose MTP weights are unquantized.

## Enable

Add the mod to the recipe:

```yaml
mods:
  - mods/glm53-nvidia-mtp-unquantized
```

Set `"moe_backend":"auto"` in the recipe's speculative configuration.
Marlin cannot run the unquantized drafter MoE. No environment variable is
required; installing the mod enables it. Reapplying the patch is a no-op.

At startup, look for `[GLM53-NVIDIA-MTP-UNQUANT]` reporting `[45]` as the MTP
layer. The helper also logs failures to derive indices, so the prefix alone
does not prove that an exclusion was applied.

## Observed result

In a 2x DGX Spark TP2 setup, the patched drafter loaded all **3/3 shards**
with no shape mismatches. Without the patch, loading failed in `_load_w2`.
No decode-throughput or acceptance measurement is available for this fix.
The original **2.3–2.5 accepted tokens per step** prediction used a separate
Spark-checkpoint MTP3 baseline of **2.428 at k=3**; it is not a measured result
for this mod.

## Rollback

Remove the mod from `mods:` and recreate the container from the original image.

## Translation scope

The source anchor is unchanged. Within the protected replacement template,
only comments are translated; inserted docstrings and runtime message literals
retain their original wording. Log prefixes, identifiers and behavior are unchanged.
