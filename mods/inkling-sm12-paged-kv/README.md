# Inkling SM12 paged-KV FA4 mod

This opt-in mod works around the FA4 startup error:

```text
Paged KV not supported on SM 12.0 in this PR
```

It is deliberately model- and architecture-specific:

- only `vllm.models.inkling.nvidia.ops.fa4_rel_attention` is patched;
- only compute capability major 12 takes the alternate dispatch;
- all other vLLM models and GPU architectures keep the image's pinned
  `vllm_flash_attn`;
- Inkling's SM12 path is forced to `num_splits=1`, matching the vendored
  bundle's documented and validated limitation.

The mod vendors the Python/CuTe DSL source from
[`SecondNatureComputing/flash-attn-4-sm120`](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120)
at commit `60117041e10fcc6f19882afd274318c755a5ef6e`. That bundle includes the
still-open upstream SM120 paged-KV work from
[`Dao-AILab/flash-attention#2348`](https://github.com/Dao-AILab/flash-attention/pull/2348).
Its BSD-3-Clause license and authors file are retained under `vendor/`.
The vendored copy also carries the two mechanical CUTLASS DSL 4.6 API
migrations from
[`vllm-project/tml-fa4@b206834`](https://github.com/vllm-project/tml-fa4/commit/b206834606ed5b5f21f8eed6b0683f528ea9cf7d):
`cute.core.ThrMma` to `cute.ThrMma`, and `cute.make_fragment` to
`cute.make_rmem_tensor`. These are required by the newer CUTLASS DSL in the
CUDA 13 vLLM image.

Use it with the existing cluster launcher:

```bash
./launch-cluster.sh \
  --apply-mod mods/inkling-sm12-paged-kv \
  exec \
  vllm serve /model \
  ...
```

The launcher applies the mod independently inside every node's container.
`run.sh` checks the installed Inkling source shape, FA4 Python dependencies,
CUTLASS DSL APIs, CUDA 12.8+, and SM12x. It imports the complete vendored
package before patching Inkling, so an incompatible bundle fails without
changing model dispatch. Reapplying the same mod is safe.

For cross-node TP on GB10, use the regular PIECEWISE CUDA-graph path and skip
FULL graph capture:

```bash
./launch-cluster.sh \
  --apply-mod mods/inkling-sm12-paged-kv \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  exec \
  vllm serve /model \
  -cc.cudagraph_mode=PIECEWISE \
  ...
```

This avoids retaining the additional FULL graph variants, which can exhaust
CUDA/NCCL graph resources during startup on unified-memory GB10 systems. It
still preserves PIECEWISE CUDA graphs. Use `--enforce-eager` only as a
diagnostic or final fallback if PIECEWISE capture also fails.

This is an experimental downstream workaround, not the same as updating
vLLM's pinned FA4 revision. Remove `--apply-mod` once paged KV for SM12 is in
the vLLM-pinned FA4 fork.
