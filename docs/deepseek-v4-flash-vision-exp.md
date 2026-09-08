# DeepSeek-V4-Flash-Vision-Exp recipe

Unified serving of `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (FP8) on 2x DGX
Spark (GB10): full-quality vision (index_topk=512, rms_norm_eps=1e-20 as
shipped) plus DSpark speculative decoding for text. Measured on 2x GB10
(TP=2, RoCE), 2026-09: structured 66.4 / prose 36.0 tok/s, 320K context,
AEB 1.00.

It depends on vLLM `local-inference-lab/vllm`
[PR #634](https://github.com/local-inference-lab/vllm/pull/634), which is
**not merged** into the b12x serving branch, so the runner image must be built
with a `--apply-vllm-pr 634` bake. The rest of the runtime dependency stack
(b12x source overlay + LMCache dev-lineage engine-driven KV) is applied at
container start by `mods/jj-ds4-vision-deps`.

## Quick start

```bash
# Discover nodes on first use
./run-recipe.sh --discover

# Build image (with PR #634 bake) + download model + launch
./run-recipe.sh deepseek-v4-flash-vision-exp --setup
```

The recipe's `build_args` already declare the build profile
(`--exp-b12x --rebuild-vllm --apply-vllm-pr 634`), so a plain
`./run-recipe.sh deepseek-v4-flash-vision-exp` will use the right image build
when `vllm-node-b12x` is missing. Weights must be present in the HF cache of
both nodes.

Run with a specific container name:

```bash
./run-recipe.sh deepseek-v4-flash-vision-exp --name vllm_dsv4f_vision
```

## Dependency stack

| Component | Patch / PR | Covered where |
|---|---|---|
| b12x | [#246](https://github.com/local-inference-lab/b12x/pull/246) generation-safe TP2 peer-push + PIECEWISE binding | `mods/jj-ds4-vision-deps` (runtime overlay) |
| b12x | [#301](https://github.com/local-inference-lab/b12x/pull/301) FP8 V4 dual-cache prefill, sparse topk 512 | `mods/jj-ds4-vision-deps` (runtime overlay) |
| b12x | [#306](https://github.com/local-inference-lab/b12x/pull/306) `rms_norm_eps=1e-20` specialization | `mods/jj-ds4-vision-deps` (runtime overlay) |
| vLLM | **[#634](https://github.com/local-inference-lab/vllm/pull/634) Vision** (NOT in branch) | **baked via `--apply-vllm-pr 634`** |
| vLLM | [#553](https://github.com/local-inference-lab/vllm/pull/553) engine-driven LMCache + expandable-CUDA segments | already in `dev/jovian-judgement` |
| vLLM | [#671](https://github.com/local-inference-lab/vllm/pull/671) fused padded-query output accounting | already in `dev/jovian-judgement` |
| LMCache | `dev` base + **#49/#50/#51/#55/#56** (engine-driven; **replaces #44**) | `mods/jj-ds4-vision-deps` (vendored install) |
| FlashInfer | `803c4664` SM120 sparse-MLA topk-512 fallback | baked in image flashinfer 0.6.18 (no-op today) |

### vLLM PR #634 (vision)

[PR #634](https://github.com/local-inference-lab/vllm/pull/634) is an open PR
targeting `local-inference-lab/vllm` `dev/jovian-judgement`. It adds:

- streaming checkpoint loader (no 157 GiB host dict)
- multimodal runtime contracts (MAX_MODEL_LEN=-1 unbounded, engine-driven
  LMCache opt-in CPU-only, direct LMCache compat mode, InstantTensor default
  loader)
- vision multi-image preprocessing
- 512-entry sparse top-k
- 3-layer DSpark drafter
- `bias_vl` vision-aware MoE routing

It is applied as a 10-commit / 40-file patch (verified clean 3-way apply onto
`dev/jovian-judgement`). The branch already contains
[#553](https://github.com/local-inference-lab/vllm/pull/553) /
[#671](https://github.com/local-inference-lab/vllm/pull/671) functionality in
equivalent form, so those do **not** need `--apply-vllm-pr`.

### Why LMCache dev + #49/#50/#51/#55/#56, not #44?

vLLM PR #634's runtime-dependency list dropped `#44` (the
`release/v0.5.2-glm52-dcp-base` lineage) in favor of five dev-lineage PRs
[#49](https://github.com/local-inference-lab/LMCache/pull/49),
[#50](https://github.com/local-inference-lab/LMCache/pull/50),
[#51](https://github.com/local-inference-lab/LMCache/pull/51),
[#55](https://github.com/local-inference-lab/LMCache/pull/55),
[#56](https://github.com/local-inference-lab/LMCache/pull/56). The dev
engine-driven path is a different implementation from #44's (no
`EngineDrivenContextPickle`; the context lives in `worker_transfer.py` as
`EngineDrivenTransferContext`). Build the dev+5 source, **not** #44 or any
`release/v0.5.x` wheel, whose engine-driven path would not match what #634's
launcher expects.

## Mods

The recipe applies two mods:

- `mods/instanttensor-hybrid-draft-loader` — keeps target loads on
  InstantTensor while selected drafts use lazy safetensors. (Pre-existing
  upstream mod.)
- `mods/jj-ds4-vision-deps` — the runtime stack (new):
  1. b12x source overlay (pure Python/JIT): [#301](https://github.com/local-inference-lab/b12x/pull/301)
     prefill, [#246](https://github.com/local-inference-lab/b12x/pull/246)
     comm/pcie, [#306](https://github.com/local-inference-lab/b12x/pull/306)
     mHC `rms_eps` allowance.
  2. LMCache dev+5 build/install from the vendored tarball
     `lmcache-dev5-src.tar.gz` (no network needed at runtime).
  3. flashinfer — no-op (baked 0.6.18 already carries the SM120 topk-512
     dispatch).

`run.sh` is idempotent and self-verifying. `verify.sh` independently checks the
three claims (mHC eps, overlay imports, LMCache markers) against the installed
environment. See `mods/jj-ds4-vision-deps/README.md` for how to rebuild the
vendored LMCache tarball.

## Notes / caveats

- `--hf-overrides '{"architectures":["DeepseekV4ForConditionalGeneration"]}'`
  is required: the upstream draft PR's config convertor computes the VL class,
  but the model loader resolves from raw `hf_config.architectures`.
- `--limit-mm-per-prompt '{"image":4}'` limits images per prompt (4).
- Weights must be on both nodes (per-node copies; rsync over the cluster link
  beats re-downloading).
- `VLLM_LOGGING_LEVEL` defaults to `INFO`: enough for `log_outputs()`. Set to
  `DEBUG` to also log the rendered prompt (`log_inputs` is DEBUG-gated).
- This stack is experimental (as are b12x builds in general); keep the repo
  updated.
