# Assumptions

## UCM KV offload (mods/ucm-kv-offload, recipes/mimo-v2.6-flash-ucm.yaml)

- Image `vllm-node`: Ubuntu 24.04, Python 3.12, CUDA 13.0.2, aarch64 (DGX
  Spark), `uv` present with `UV_SYSTEM_PYTHON=1` — matches the published
  `uc-manager-cuda-cu130` cp312 wheels (verified on PyPI 2026-09-21,
  version 0.7.0 is the newest stable; 0.8.0rc1 is a prerelease).
- Mods run via a separate `docker exec` before launch, so serve-time settings
  (`ENABLE_UCM_PATCH`, `--kv-transfer-config`) must live in the recipe; store
  overrides reach the mod only as container env (`launch-cluster.sh -e` or
  `.env` `CONTAINER_*`).
- `launch-cluster.sh` mounts `~/.cache/vllm:/root/.cache/vllm` unless
  `--no-cache-dirs` — the default store dir is therefore host-backed and
  persistent. Per-node local stores are correct for TP=2: each rank keeps its
  own KV shards on its own node.
- UCM 0.7.0's version-specific patch matrix covers vLLM up to 0.28.0; the
  image builds vLLM `main`. Beyond the matrix UCM logs a warning and relies on
  the stock `KVConnectorBase_V1` API. Not verified against a running cluster
  (development scope: no real deployment).
- UCM + DFlash speculative decoding + fp8 KV cache is untested upstream; the
  recipe keeps the base recipe's flags unchanged apart from the UCM additions.
- The base `mimo-v2.6-flash` recipe's comments/rationale are duplicated in the
  new recipe; keep both in sync when changing shared flags.

## 3-node PP=3 recipe (recipes/3x-spark-cluster/mimo-v2.6-flash-pp3.yaml)

- Model geometry read from the local HF snapshot `config.json` (blob in
  `~/.cache/huggingface`): 64 heads / 4 KV heads / 8 SWA KV heads / 256 routed
  experts / 48 layers / hidden 4096 / vocab 152576 — TP=3 impossible, PP=3
  divides 48 layers exactly.
- 3rd node has 108 GB (vs 121 GB): `gpu_memory_utilization: 0.87` gives a
  94.0 GiB budget there vs ~58 GiB weights per pipeline stage; vLLM sizes KV
  from the smallest worker, so the 121 GB nodes just gain slack. ~173 GB total
  weights assumed (recipe header), split ~58 GB per stage.
- No operational actions were taken on any node (the session runs on the
  108 GB node and its engine must not be restarted): validation was dry-runs
  and mocked tests only. The 3rd node's IP is deliberately not stored in any
  tracked file.
- The local HF snapshot of MiMo-V2.6-Flash-RL on this host is incomplete
  (config + a few files, no weight blobs, no `dflash/`): any real launch needs
  `hf-download.sh` first — deferred as operational (`todo.md`).
- `--distributed-executor-backend` omitted per the dev guide even though the
  older nested recipes hardcode `ray`; runner/launcher supply backends.

## Official vLLM recipe review (vllm-project/recipes @ 09dea3c3, 2026-09-22)

- `min_vllm_version: nightly`, `nightly_required: true`; official states stable
  vLLM <= 0.29.0 cannot load the mxfp4-stored weights — consistent with this
  repo's "vLLM main from 2026-09-21 or newer" image requirement, and it
  reinforces the UCM note (UCM 0.7.0 patch matrix stops at 0.28.0).
- Official `--max-model-len auto` resolves to the same 1048576 we pin
  (`max_position_embeddings` in config.json), so the explicit value stays.
- Official `--gpu-memory-utilization 0.95` is H200 dedicated-VRAM guidance;
  not applicable to unified-memory Sparks (0.87 kept, math documented).
- Official `compatible_strategies` list only TP/TEP/DEP/PD — no
  pipeline-parallel — confirming PP=3 is our own unverified experiment.
- Official recipe omits `--moe-backend marlin` and `--kv-cache-dtype fp8`:
  both are sm_121-specific adaptations, kept deliberately.

## tonyd2wild review — upstream state verified on vllm main 2026-09-22

- `mimo_v2.py`: `MiMoV2Attention` accepts `cache_config` but no caller passes
  it (dead param) -> `Attention()` resolves `kv_cache_dtype` to `"auto"` and
  ignores `--kv-cache-dtype` for the 48 target layers. DiffKV backend
  (`triton_attn_diffkv.py`) has `supported_kv_cache_dtypes = [auto,
  bfloat16]` and raises on quantized caches. Both patched by
  `mods/mimo-diffkv-fp8-kv`, anchored to that exact layout (fails fast if an
  image rebuild drifts).
- Async scheduling default resolves to ON for dflash (upstream's auto-disable
  list whitelists it); vllm#46669 (open) shows DFlash/MTP + async garbage
  outputs on MiMo at concurrency>1. Mitigation = `--no-async-scheduling` on
  the two DFlash recipes; the pp3 recipe omits spec decode, so it is out of
  scope for that bug (its async behavior under PP>1 is upstream-managed).
- DeepGEMM: our pin `a6b593d` is the pre-regression commit named in
  DeepGEMM#417 (the broken one is `8b1392b`, what his image shipped); no
  `VLLM_USE_DEEP_GEMM=0` needed alongside `--moe-backend marlin`.
- `SupportsEagle3` is present on both `MiMoV2ForCausalLM` and
  `MiMoV2OmniForCausalLM` upstream; his patch 02 is superseded. Omni builds
  its LM from `mimo_v2.py`, so that file is the single fp8-KV patch point.
- His `setup.sh` also rewrites the trailing-comma `dflash/config.json` —
  second independent source confirming the staging premise of
  `mods/mimo-v2.6-flash`.