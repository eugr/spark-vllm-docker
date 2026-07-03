# b12x-min

Enables the FlashInfer b12x CuTeDSL MoE backend on the current vLLM main
wheel, without the obsolete parts of `mods/exp-b12x`. Not used by the
glm-5.2-nvfp4 recipe; kept for reference.

## Background

vLLM PR 40082 (b12x MoE + FP4 GEMM for SM120/121) merged in May 2026, so
current wheels already contain the backend — no special build needed.
`mods/exp-b12x` predates that and does two things that now break or don't
apply:

- it pins `nvidia-cutlass-dsl` to 4.4.2, but FlashInfer 0.6.14 uses 4.5-only
  APIs (`cutlass.cute.nvgpu.OperandMajorMode`), so the pin causes an
  AttributeError at startup. The sm_121 bad-PTX bug that motivated the pin
  appears fixed in cutlass-dsl 4.5.2 — stock works.
- it gates on env vars from an older env-based backend selection; current
  wheels select via `--moe-backend flashinfer_b12x`.

## What this mod does

Two small patches: removes a stale import from FlashInfer's
`blackwell_sm12x/__init__.py`, and opens the sm_121 gate in vLLM's b12x
NVFP4 linear path (PR 40080 check). Launch with:

```
run-recipe.py <recipe> -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-b12x
# recipe command: --moe-backend flashinfer_b12x
```

The env var must go through the launcher `-e` flag (container-level), not
recipe `env:` — mods run via docker exec and don't see recipe env.

## Measured outcome (GLM-5.2-NVFP4, 8x Spark TP=8, July 2026)

Works and produces correct output, but no win over marlin: 12.6 tok/s
average vs marlin's 14.0, with extra first-request latency from CuTeDSL JIT
per new shape. Decode on this cluster is bound by inter-node all-reduce
latency, so the MoE kernel choice barely matters. The recipe stays on
marlin.
