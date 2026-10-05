# fix-qwen3.8-ple-dtype

Injects `text_config.ple_embedding_dtype` into the
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4` checkpoint config before vLLM
reads it.

## Problem

The checkpoint quantizes its PLE n-gram embedding to NVFP4 group 16 — declared
in `quantization_config.quantized_layers` as
`model.language_model.layers.1.ple.ple_embedding.ngram_embedding` — but its
`config.json` omits `ple_embedding_dtype` from `text_config` (verified against
the HF hub revision `6909a5b`). vLLM defaults the field to `bfloat16`
(`vllm/models/qwen3_8_flash_next/config.py`), so the b12x PLE plan is built for
bf16 storage and weight loading crashes:

```
ValueError: shape mismatch for PLE shard 100 weight: expected (2500012, 160), got (2500012, 80)
```

The checkpoint tensors are `shard_N.weight` = `(rows, head_dim // 2)` uint8
(packed fp4) plus `shard_N.weight_scale` = `(rows, head_dim // 16)` float8_e4m3
and a top-level `ngram_embedding.weight_scale_2` — exactly the b12x
`nvfp4_group16` contract, so the fix is to declare `"ple_embedding_dtype":
"nvfp4"`.

## Behavior

- Resolves the revision vLLM will load via `refs/main` and patches only that
  snapshot's `config.json`, adding `"ple_embedding_dtype": "nvfp4"` derived
  from the checkpoint's own `quantization_config` (only `NVFP4` is
  translated). Stale sibling snapshots from earlier downloads are untouched.
  Falls back to patching every snapshot only when no ref file exists.
- No-op if the key is already present and consistent with the checkpoint
  quantization; refuses loudly on contradictory or unrecognized values, on
  configs without the PLE family keys, or when the model is not in the cache.
- Writes through the snapshot symlink atomically (replaces the HF blob, keeps
  the symlink), preserving the original JSON indentation.

## Environment overrides

- `HF_HOME` — cache root (default `/root/.cache/huggingface` inside the
  container; the launcher bind-mounts the host cache there).
- `FIX_QWEN38_PLE_MODEL_ID` — checkpoint to patch (default
  `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`).
- `FIX_QWEN38_PLE_REF` — ref to resolve (default `main`).

## Test

```bash
python3 tests/test_fix_qwen38_ple_dtype_mod.py -v
```

Remove this mod once the upstream checkpoint ships `ple_embedding_dtype` in
its config.json.
