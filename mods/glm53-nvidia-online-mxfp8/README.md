# GLM-5.3-Flash online MXFP8 / NVFP4

Quantizes selected BF16 layers left unquantized by
`nvidia/GLM-5.3-Flash-NVFP4` during loading: KDA/MLA attention projections
and shared experts. Smaller weights reduce memory bandwidth requirements.
The mod is experimental and disabled by default.

Requires a vLLM build with the ModelOpt exclusion hook and online MXFP8/NVFP4
methods used by this patch. Results below use B12X backends on DGX Spark;
compatibility with other builds is not established. `--check` validates the
patch anchors without writing.

## Enable

Add `glm53-nvidia-online-mxfp8` to the recipe's `mods` list and pass these
environment variables to every worker. Select one configuration before starting
the engine.

| Variable | Values and effect |
| --- | --- |
| `VLLM_GLM53_ONLINE_MXFP8` | `attn` selects attention projections; `attn,shared` also selects shared experts. Unset or `0` disables the mod. The parser accepts `attn,shared,mtp`, but **do not use `mtp`**. |
| `VLLM_GLM53_ONLINE_MXFP8_A16` | `0`: MXFP8 activations, recommended for MXFP8. Default `1`: BF16 activations. |
| `VLLM_GLM53_ONLINE_NVFP4` | Optional override for already-selected groups: `shared` (shared experts), `oproj` (`o_proj`), `qkv` (remaining selected attention projections). Use `shared,oproj,qkv` for all three. Unset disables the override. |
| `VLLM_GLM53_ONLINE_NVFP4_A16` | Default `1`: BF16 activations (W4A16), used in the measurements. `0`: FP4 activations (W4A4), not recommended. |

MXFP8 configuration:

```bash
export VLLM_GLM53_ONLINE_MXFP8=attn,shared
export VLLM_GLM53_ONLINE_MXFP8_A16=0
unset VLLM_GLM53_ONLINE_NVFP4 VLLM_GLM53_ONLINE_NVFP4_A16
```

For the measured NVFP4 configuration, keep the MXFP8 selection above and add:

```bash
export VLLM_GLM53_ONLINE_NVFP4=shared,oproj,qkv
export VLLM_GLM53_ONLINE_NVFP4_A16=1
```

Loading logs use the unchanged prefixes `[GLM53-ONLINE-MXFP8]` and
`[GLM53-ONLINE-NVFP4]` to identify the selected quantization method.

## Measured results

2x DGX Spark, TP2+DCP2, MTP k3, 512K context limit, kernel 7.0.
Quality: `tool-eval-bench` commit `d84fce4`, 69 scenarios.
Decode and prefill rates are tokens/s.

| Configuration | Prose | Code | JSON | Prefill | Quality score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline: selected layers remain BF16 | 26.8 | 33.0 | 38.7 | 1720 | — |
| MXFP8 `attn,shared`, A16=0 | 30.5 | 37.1 | 45.2 | 1816 | 94 |
| NVFP4 `shared,oproj,qkv`, A16=1 | 34.1 | 41.0 | 47.7 | 1656 | 95 |

The NVFP4 configuration also passed needle retrieval at 225K and 451K tokens.
These are measured results for this setup, not guarantees of identical output.

**Do not use `mtp`.** Quantizing the MTP experts holds their BF16 weights and
quantized copy simultaneously during loading. The memory spike took a node
offline. Ordinary MTP k3 remains enabled in the measurements above; it does
not require the `mtp` quantization option.

**MXFP8 A16=1 reduced prefill throughput by 14% in an earlier comparison.**
Use MXFP8 A16=0; the NVFP4 measurements use its separate A16=1 setting.

## Disable / rollback

Unset all four variables and restart the engine to reload the original weights.
To remove the patch as well, remove the mod from the recipe and recreate the
container. No checkpoint files are modified.
