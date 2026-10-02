# MiMo fp8 KV cache mod

Runtime mod that makes `--kv-cache-dtype fp8` actually work for MiMo-V2 on
sm_121. Required by every local MiMo recipe that passes the flag
(`mimo-v2.6-flash`, `mimo-v2.6-flash-ucm`, `3x-spark-cluster/mimo-v2.6-flash-pp3`).

## What it fixes (verified against vllm main, 2026-09-22)

- `vllm/model_executor/models/mimo_v2.py`: no caller passes `cache_config`
  into `MiMoV2Attention`, so `Attention()` resolves `kv_cache_dtype` to
  `"auto"` and the CLI flag is silently ignored on all 48 target layers. The
  mod defaults the parameter to `get_current_vllm_config().cache_config`, and
  hands full-attention layers (no per-layer window) a copy with
  `sliding_window = None`: `Attention()` falls back to
  `cache_config.sliding_window` (the model's 128) otherwise. Without that copy
  (this mod before 2026-09-23) every layer attended to 128 tokens and long
  generations looped at TP=2 and PP=3; with it the six-prompt probe is clean
  (tonyd2wild does the same in patch 01).
- `vllm/v1/attention/backends/triton_attn_diffkv.py`: this is the backend
  vLLM auto-selects on sm_121 for the model's 192/128 K/V head dims. It
  rejects quantized KV (`supported_kv_cache_dtypes` = auto/bfloat16 +
  `NotImplementedError`) and never views the cache as fp8 on read. The mod
  adds `fp8`/`fp8_e4m3`, narrows the rejection to non-e4m3 quantized caches
  (the fp8 view would corrupt e5m2/nvfp4/per-token-head), and adds the fp8
  view in `forward()` — mirroring the parent Triton backend.

Same fix as patches 01 + 03 of
[tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark](https://github.com/tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark),
who validated it end-to-end on GB10 (fp8 KV pool 12.6 GiB measured, needle
tests at 100K/250K). Rationale from his README: the reshape kernel quantizes
on write and the attention kernel upcasts K/V to bf16 on load without applying
KV scales — exact for this checkpoint because it uses unit scales.

Without the mod the recipes' fp8 KV memory math (14.3 GiB for a 1M sequence,
`gpu_memory_utilization 0.87`) does not hold: layer-side dtype stays bf16
while the flag is set.

## Behavior

- Each of the two files is patched independently; already-patched files are
  skipped, so repeat application in the same container is safe.
- Fails fast with a clear message when the installed vLLM no longer contains
  the expected anchors (layout drift after an image rebuild).
- Env: `VLLM_SITE_PACKAGES` / `PYTHON_ROOT` (default
  `/usr/local/lib/python3.12/dist-packages`).

## Not covered here

- `dflash/config.json` trailing comma and drafter staging:
  `mods/mimo-v2.6-flash`.
- Fused fp8 `qkv_proj` TP sharding and MXFP4/bf16-router support: upstream
  PRs [#57508](https://github.com/vllm-project/vllm/pull/57508) /
  [#57784](https://github.com/vllm-project/vllm/pull/57784), checked by
  `mods/mimo-v2.6-flash`.

Test: `./tests/test_mimo_diffkv_fp8_kv_mod.sh`.