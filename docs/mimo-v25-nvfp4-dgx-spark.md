# MiMo-V2.5-NVFP4 on 2× DGX Spark / GB10

This document accompanies `recipes/mimo-v2.5-nvfp4.yaml` and the runtime mods:

- `mods/fix-mimo-v2-vllm`
- `mods/fix-modelopt-mixed-mxfp8`

The recipe targets [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4), the NVFP4 export of MiMo V2.5, on a two-node DGX Spark / GB10 cluster with tensor parallel size 2.

## Validated runtime shape

- vLLM with PR #41797 / `TRITON_ATTN_DIFFKV`
- `--load-format instanttensor`
- Omni architecture enabled via `MiMoV2OmniForCausalLM`
- MiMo MTP speculative decoding enabled with `num_speculative_tokens=2`
- FlashInfer-CUTLASS MXFP8 dense GEMM
- FlashInfer-CUTLASS NVFP4 MoE
- FP8 E4M3 KV cache
- prefix caching and chunked prefill
- 131072 max model length
- explicit generation defaults: `temperature=1.0`, `top_p=0.95`, `max_new_tokens=8192`

Expected startup markers:

```text
Resolved architecture: MiMoV2OmniForCausalLM
Resolved architecture: MiMoV2OmniMTPModel
Using FlashInferCutlassMxfp8LinearKernel for MXFP8 GEMM
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
Using TRITON_ATTN_DIFFKV for attention
Using fp8_e4m3 data type to store kv cache
Chunked prefill is enabled with max_num_batched_tokens=16384
```

## Why the mods are required

### `fix-modelopt-mixed-mxfp8`

Adds missing ModelOpt mixed-precision MXFP8 support needed by the NVFP4 export:

- registers a `weight_scale_inv` alias for MXFP8 scale metadata;
- adds an MXFP8 sub-config to `ModelOptMixedPrecisionConfig`;
- dispatches `MXFP8` `LinearBase` layers to `ModelOptMxFp8LinearMethod`;
- dispatches MXFP8 `RoutedExperts` for completeness.

`weight_scale_inv` is UE8M0 scale metadata in this checkpoint path. Do **not** reciprocal-invert it.

### `fix-mimo-v2-vllm`

Applies MiMo-specific runtime fixes:

- registers local `MimoV2Config` when the base vLLM build has the model class but not the config registry entry;
- preserves explicit text-only architecture overrides for diagnostics;
- filters non-text checkpoint tensors for text-only runs;
- fixes Omni vision merger bias mismatch;
- skips target-model loading of MTP and audio-tokenizer tensors that are loaded/handled separately;
- replaces blind fused-QKV TP chunking with `QKVParallelLinear.weight_loader` for the NVFP4 checkpoint's `qkv-deinterleaved` layout;
- applies the same QKV-aware loading to MTP layers;
- adds Omni `packed_modules_mapping` so fused modules resolve quant metadata;
- mirrors Omni-remapped MTP quant metadata back to draft `model.mtp.*` prefixes;
- selects `MiMoV2OmniMTPModel` when a raw MiMo config advertises `vision_config`;
- applies vLLM PR #42969 for the Qwen3XML/MiMo streaming tool parser so a
  closed `<function>` clears `current_function_name` and does not emit duplicate
  recovery braces.

The QKV loader change is critical: the NVFP4 checkpoint stores fused QKV as canonical `[Q_all][K_all][V_all]`, not TP-prepacked `[Q0 K0 V0][Q1 K1 V1]`. Blind `chunk(tp_size, dim=0)` is shape-correct but corrupts K/V rows.

## Basic validation

After launch:

```bash
curl -s http://localhost:8000/v1/models
curl -s http://localhost:8000/metrics | grep -E 'cache_config_info|spec_decode'
```

A short text prompt, a small image prompt, and tool-call requests have been validated with this recipe. Full audio/video validation is still pending.

## Benchmark commands used

Tool eval:

```bash
tool-eval-bench \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model MiMo-V2.5-NVFP4 \
  --seed 42 \
  --temperature 1.0 \
  --top-p 0.95 \
  --timeout 120 \
  --parallel 1
```

Observed result in one run: 89/100, 123/138 points, no safety warnings.

Throughput example:

```bash
llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model lukealonso/MiMo-V2.5-NVFP4 \
  --served-model-name MiMo-V2.5-NVFP4 \
  --tokenizer lukealonso/MiMo-V2.5-NVFP4 \
  --pp 2048 \
  --tg 32 \
  --depth 0 4096 8192 16384 32768 65536 98304 114688 131072 \
  --concurrency 1 2 \
  --runs 3 \
  --latency-mode generation \
  --enable-prefix-caching \
  --format json
```
