# sparse-mla-chunked

Fixes a silent cluster-wide hang when serving GLM-5.2 (DeepSeek Sparse
Attention models) on DGX Spark, and unlocks fast prefill.

## The problem

FlashInfer 0.6.14's SM120 sparse-MLA attention has two code paths, chosen
per call in `flashinfer/mla/_sparse_mla_sm120.py`:

- calls with **64 or fewer** query tokens go to an autotuned kernel
  (`decode_dsv3_2`) that works fine on sm_121;
- anything larger goes to a generic fallback kernel
  (`sparse_mla_sm120_paged_attention`) that **hangs on sm_121**.

When the fallback hangs, one rank never reaches its next all-reduce, the
other seven wait in NCCL forever, and every GPU in the cluster sits at ~96%
utilization at low power with no error anywhere. Any prefill batch over 64
tokens could trigger it, so the model was unusable beyond trivial prompts.

How this was pinned down: `CUDA_LAUNCH_BLOCKING=1` stack dumps on all 8
ranks showed rank 0 inside the fallback kernel and ranks 1-7 in
`ncclAllReduce`; a raw 8-node NCCL all-reduce sweep up to 128MB ran clean,
which ruled out the network. MoE backends, cudagraph modes, NCCL settings,
and the FlashInfer sampler were all swapped during diagnosis and made no
difference.

## The fix

In this MQA-style kernel every query token attends independently over its
own top-k indices, so a large call can be executed as a sequence of small
calls with identical results. `run.sh` patches `_paged_attention` in
`flashinfer/mla/_sparse_mla_sm120.py`: calls over 64 tokens are sliced into
64-token chunks and each chunk dispatches to the working `decode_dsv3_2`
kernel. The broken fallback is never used for these shapes. No arithmetic
changes; only dispatch.

The patch is Python-only, idempotent, and aborts loudly if FlashInfer's
code layout changes. Requires `block_size 64` (the kernel's page size) and
head/topk shapes the kernel supports (GLM-5.2 at TP=8: 8 heads, topk 2048).

## Results (8x DGX Spark, TP=8, nvidia/GLM-5.2-NVFP4)

Without the mod, `max_num_batched_tokens` had to stay at 64 to avoid the
hang, which held prefill to ~180 tok/s. With the mod and 2048-token chunks:

- prefill: ~180 -> ~880 tok/s (a fresh 61K-token prompt in 69s)
- decode: unchanged, ~14 tok/s single-stream
- correctness: needle retrieval at 3 depths of a 12.5K-token document,
  greedy math/string checks, a 20K-token needle, and a full GSM8K run
  (93.1% strict, 1319 problems) all pass
- stability: a 60-request soak with 4-way concurrency plus the ~6-hour
  GSM8K run, zero hangs

## When to remove

When FlashInfer fixes the fallback kernel on sm_121 (their SM120 sparse-MLA
kernels are from June 2026; see flashinfer PRs #3374/#3395 and the top-k
race issues #3618/#3625 for the state of that area). The patch anchors on
the dsv3_2 dispatch block and will refuse to apply rather than misapply.
