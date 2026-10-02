# MiMo-V2.6-Flash drafter staging mod

Runtime mod for the `mimo-v2.6-flash` recipe (`XiaomiMiMo/MiMo-V2.6-Flash-RL`,
309B MoE / 15B active, MXFP4 experts + fp8 attention, DFlash drafter).

## What it does

Stages a loadable copy of the checkpoint's DFlash drafter at
`/workspace/MiMo-V2.6-Flash-RL-dflash`:

- vLLM cannot address a Hub subdirectory such as `org/repo/dflash` as a draft
  model, and the drafter reads `mask_embedding.pt` from its own model root, so
  the drafter needs a real local directory.
- The shipped `dflash/config.json` has a trailing comma that transformers'
  strict JSON parser rejects. The staged copy symlinks the weight files and
  writes a sanitized `config.json`.

Before staging, the mod fails fast if the installed vLLM predates the MiMo-V2
fixes the recipe relies on (upstream PRs
[#57508](https://github.com/vllm-project/vllm/pull/57508), fused fp8 `qkv_proj`
sharding for TP=2, and [#57784](https://github.com/vllm-project/vllm/pull/57784),
bf16 router + MXFP4 experts + DFlash value scale). It does not patch vLLM; on an
older image use `--apply-vllm-pr 57508 --apply-vllm-pr 57784` or rebuild.

## Environment overrides

- `MIMO_V26_MODEL_DIR`: use this local snapshot directory instead of resolving
  the Hugging Face cache.
- `MIMO_V26_MODEL_ID`: repo id to resolve (default `XiaomiMiMo/MiMo-V2.6-Flash-RL`).
- `MIMO_V26_LINK_DIR`: parent directory for the staged drafter (default `/workspace`).
- `VLLM_SITE_PACKAGES` / `PYTHON_ROOT`: vLLM install root
  (default `/usr/local/lib/python3.12/dist-packages`).

The model must already be present in the mounted Hugging Face cache
(`HF_HOME`, or `~/.cache/huggingface`); the mod never downloads.
