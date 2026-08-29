# Qwen3.8-27B — findings ledger and upstream-baseline plan

The fork's three custom qwen3.8-27b recipes (`-nvfp4` MTP, `-nvfp4-dspark`,
`-nvfp4-1m`) were retired on 2026-08-29 in favor of tuning from upstream's
`recipes/qwen3.8-27b-nvfp4-dflash2.yaml`. This file preserves what those
recipes' headers had measured, so the knowledge survives the deletion, and
records the plan for tuning the upstream baseline. The full retired recipes
are recoverable from git: `git show 9794369:recipes/qwen3.8-27b-nvfp4.yaml`
(same commit for `-dspark` and `-1m`).

Why retired rather than kept: upstream's recipe is maintained by eugr against
the image as it moves (it shipped in the same commit as the
`patch_vllm_mrv2_speculator_cudagraph_pool.py` fix), while the customs were a
parallel lineage only this fork maintained, on a different checkpoint
publisher (unsloth vs RadixArk) with tuning frozen at 2026-08-20/21, TP=2,
2-node cluster. Per the fork policy, deltas should be measured against the
upstream baseline, not carried as a separate universe.

## Findings that transfer (hardware truths, checkpoint-independent)

### sm_121 spec-decode CUDA graphs: FlashInfer cannot, TRITON_ATTN can

Measured 2026-08-20, TP=2. Under speculative decoding on FlashInfer every
multi-token verify step ran outside full CUDA graphs: vLLM logged
"CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
attention backend FlashInferBackend (support:
AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting
cudagraph_mode=PIECEWISE", and ~2.7-2.8 accepted tokens per step returned
only ~1.35x over no speculation.

Architectural on GB10, not a tuning mistake:
`FlashInferMetadataBuilder.get_cudagraph_support()` returns UNIFORM_BATCH
only when `can_use_trtllm_attention()` holds, and
`supports_trtllm_attention()` ends in `is_device_capability_family(100)` —
this hardware is sm_121, family 120, always False. TRITON_ATTN declares
`AttentionCGSupport.ALWAYS`. Both flags are required: with only
`--attention-backend` the drafter re-selects FlashInfer.

Single-stream /v1/completions, 256 greedy tokens, TP=2, unsloth checkpoint:

    no speculation ................. 19.8 tok/s
    DSpark k=7, FlashInfer ......... 22.6-26.8
    MTP k=3, FlashInfer ............ 26.8-27.7
    MTP k=8, TRITON_ATTN ........... 28.5-29.4
    DSpark k=7, TRITON_ATTN ........ 34.5
    MTP k=3, TRITON_ATTN ........... 39.4-39.6   <- best; the number to beat

Caveat for the dflash2 campaign: upstream's speculator-cudagraph-pool patch
(added 957c47e, adjusted e9cf359) postdates this finding and was tuned
together with the dflash2 recipe. Whether dflash verify steps get graphed on
FlashInfer with that patch is UNVERIFIED — measure it, do not assume the pin
is still required, and do not assume it is not.

### Triton vs FlashInfer crossover near 42k tokens of context

Measured 2026-08-20 on the 1M recipe (speculation on), TP=2:

                          FlashInfer   TRITON_ATTN
    decode @   1,510 tok     25.6         35.9      Triton +40%
    decode @ 213,574 tok     22.8         12.2      Triton -46%
    decode @ 287,519 tok      8.5          3.6      Triton -58%
    prefill @ 100,578 tok  1,623 tok/s   1,054      Triton -35%
    TTFT    @ 213,574 tok    178 s        329 s     Triton +85%

Triton step time grows ~0.83 ms per 1k tokens of context against
FlashInfer's ~0.11; they cross near 42k. Short-context serving wants Triton
(when the cudagraph point above also applies), long-context wants FlashInfer.

### FlashInfer autotune startup deadlock: not a risk on the current image

Tested 2026-08-20 against vllm-project/vllm#52291: the image's
`flashinfer_autotune()` takes the per-rank path whenever world_size > 1
(`set_autotune_process_group` does not exist in it), autotune completed
twice (~11 s tuning fp4_gemm) with decode unchanged. Leave autotune at
vLLM's default; revisit only if a future image regresses.

### KV cache is already fp8 on this checkpoint family

`--kv-cache-dtype fp8` (which upstream's recipe passes) is a no-op — the
checkpoint's KV is already fp8. Harmless; not a lever. The context-capacity
levers are gpu_memory_utilization and max_num_seqs.

## Findings tied to the unsloth checkpoint — re-verify on RadixArk

These were properties of `unsloth/Qwen3.8-27B-NVFP4` and do NOT
automatically transfer to `RadixArk/Qwen3.8-27B-NVFP4`:

- Reasoning effort worked with no flag because unsloth ships Qwen's chat
  template plus superset patches (high -> xhigh alias, developer role,
  merged leading system messages). vLLM forwards `reasoning_effort` into
  chat_template_kwargs; the template took xhigh/medium/low/high, rejected
  minimal/max with a 400, and `none` turned thinking off. Whether RadixArk's
  template does the same is the main client-contract risk of the switch.
- Sampling came from the checkpoint's own generation_config.json (Qwen's
  thinking-mode preset); `--generation-config vllm` was deliberately omitted
  because overriding drops top_k.
- Vision was live: image_token_id/vision_config in config.json, the NVFP4
  quantization left model.visual.* in bf16, and `--limit-mm-per-prompt`
  was enforced. Verified 2026-08-21 with image probes (PNG and WebP scored
  identically). Image cost floor: preprocessor shortest_edge 65536 px^2, so
  any source at or below 256x256 bills 103 prompt tokens. Whether RadixArk's
  quantization also preserves the vision tower is unverified.
- Thinking is on by default; a small max_tokens is consumed inside the
  reasoning block and returns content: null with finish_reason "length".

## Speculation cost-model findings

- MTP k=3 beat k=8 (39.4-39.6 vs 28.5-29.4) despite lower acceptance
  (3.0 vs 3.3): the MTP head drafts sequentially, so k passes cost k
  forward passes. Block-parallel drafters (DSpark, presumably DFlash2)
  have a different cost model — do not carry k conclusions across methods.
- DSpark k=7 lost to MTP k=3 on real text (prose 25.7 vs 35.2, acceptance
  1.93 vs 2.66) and won only on highly repetitive output (96.6 vs 51.1).
- The MTP head accepts exactly zero past the native 262,144-token window
  (per-position 0.000/0.000/0.000 against 3.12 acceptance at 254k) — decode
  fell 23 -> 8.5 tok/s. Categorical, cause never established; the obvious
  YaRN explanation was ruled out (for method=mtp the draft config IS the
  target config, one shared rope).

## Why the 1M recipe is not coming back

Static YaRN taxes every short request (the majority) to serve rare long
ones; past-native speculation was pure overhead (see the cliff above); a
cold 1M prefill extrapolated to ~30 minutes. Decisively:
deepseek-v4-flash-0731-1m now serves the 1M use case far better — 48 tok/s
decode at 928k context with speculation still earning (acceptance > 3),
against qwen's 8-14 tok/s past native. The niche this recipe existed for is
covered by a better recipe in the same fleet.

## The campaign: tuning upstream's dflash2 from its own defaults

Baseline: `recipes/qwen3.8-27b-nvfp4-dflash2.yaml` exactly as upstream ships
it (RadixArk checkpoint, DFlash2 k=8, util 0.7, FlashInfer, instanttensor,
async-scheduling). First launch foreground, per the GB10 launch rules.

1. Contract probes before any tuning:
   - reasoning_effort xhigh / high / none against RadixArk's template
   - vision: does the RadixArk NVFP4 checkpoint keep the tower? probe an
     image request; check config.json for vision_config / mm token ids
   - tool calls through qwen3_xml
2. Baseline throughput at defaults, single stream + concurrency 8, against
   the recorded MTP number (39.4-39.6 tok/s, TP=2, unsloth). tools/
   bench-serving.py and quality-probe.py are the harness (their presets
   still name the retired recipes; add a dflash2 preset when running).
3. A/B, one knob at a time, each against the measured baseline:
   - TRITON_ATTN pins (engine + draft) on/off — settles whether the
     cudagraph-pool patch made the pin obsolete
   - gpu_memory_utilization ladder 0.7 -> 0.80 -> 0.85 (KV pool vs OOM;
     the customs ran 0.80-0.88 without incident at TP=2)
   - --tp 4 one-off: does dense-model allreduce over the CRS812 pay?
4. Client contract deltas (served-model-name qwen3.8-27b, mm limits if
   vision survives) go in as launch-time extra args on the deployment unit,
   or as a minimal delta if they prove permanent — never by forking the
   upstream recipe wholesale again.

Record results in this file, dated, with the config they ran against.
