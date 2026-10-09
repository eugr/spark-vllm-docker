# fix-glm53-nope-rope-pad

Dresses **GLM-5.3-Flash** (`glm5_next`) rope-free NoPE-MLA into the DeepSeek
`fp8_ds_mla` shape so the SM120/121 **flashinfer sparse-MLA kernels** can serve
it on DGX Spark (GB10 / sm_121). This is the exact mod the GLM-5.3-Flash-on-2x-DGX-Spark
run uses (no kernel builds, no container rebuild) — the SM120 path the vendor
recipe ships hard-asserts `pe_dim=64`, and GLM's `qk_rope_head_dim=0` violates it.

## Why

GLM-5.3-Flash is 320B/18B-active, hybrid **NoPE sparse MLA**
(`kv_lora_rank=512`, `qk_rope_head_dim=0`, **query head dim 512**, `index_topk=2048`)
+ **KDA linear attention** (34 of 45 layers) + a built-in **MTP** head. DeepSeek V4
is rope-64 / query-576. The SM120 kernels (`flashinfer/mla/_sparse_mla_sm120.py`)
accept only the DeepSeek 576-dim contract, so a rope-free MLA (query 512) dies at
`pe_dim must be 64`. Zeros are rotation-invariant and add exactly 0 to every
logit, so appending a 64-dim zero block to `q`/`k_pe` is a mathematical no-op
that makes GLM present the DeepSeek-shaped contract the kernels expect.

## Env gate

`VLLM_MLA_NOPE_PAD_ROPE=1`. The patch is applied per-boot but the rope-padding
stays **off unless the env var is set** — and only activates on layers that are
genuinely NoPE (`qk_rope_head_dim == 0`).

## What it patches (the proven 4 parts)

| # | File | Change |
|---|------|--------|
| 1 | `vllm/model_executor/layers/mla.py` | Build the inner MLAAttention with `qk_rope_head_dim += 64` when NoPE + env on, so the KV-cache spec/kernels see head **576**; zero-pad `q` and create a zero `k_pe` right before `self.mla_attn(...)` |
| 2 | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | Pass the **real** topk table width (`topk_indices_physical.shape[-1]`) instead of `attn_metadata.topk_tokens` for both `max_seq_len` and `sparse_mla_top_k` (glm5_next kpool indexer builds 2048 + tail = **2176**) |
| 3 | `flashinfer/mla/_sparse_mla_sm120.py` | Extend the `_DECODE_DSV3_2_DISPATCH` frozenset allowlist to `(h, 2176)` for `h in {8,16,32,64,128}` (a shape allowlist, not a compiled-variant limit) |
| 4 | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | Compact the 2176-wide topk table down to the compiled **2048** width in the decode path (keep top `2048-tail` ranked entries + the always-select tail; never duplicated) |

Idempotent: each part checks for a marker string before applying. `py_compile`
-verified after every write. Paths are the standard container
`/usr/local/lib/python3.12/dist-packages/...` layout.

## Notes

- **This mod is for the stock `vllm/vllm-openai:glm53-flash-arm64-cu130` image**
  where the glm5_next model impl already ships. It is env-gated so it no-ops on
  non-GLM / non-NoPE models.
- **MoE backend on sm_121**: `--moe-backend marlin` is the proven flag — the
  auto-selected FLASHINFER_CUTLASS NvFp4 backend silently corrupts math (boots
  green, garbage from token one). For the native **FP8** checkpoint (`zai-org/
  GLM-5.3-Flash`) this needs a hardware smoke test (marlin is an int4 path; the
  FP8/GPTQ path may differ). The proven runs used the NVFP4/int4 weights.
- **`--language-model-only`** is required: the multimodal processor balloons the
  API front-end to ~15.7 GB anon and gets OOM-killed.
- **Boot-time "No available shared memory broadcast block"** is usually
  FlashInfer autotuning (check worker CPU, 137–186%), not a wedge — it persists
  to `~/.cache/vllm` after the first boot.
