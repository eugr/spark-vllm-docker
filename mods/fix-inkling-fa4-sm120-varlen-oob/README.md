# Inkling SM12 FA4: fix out-of-bounds reads with more than one request per batch

Fixes intermittent engine deaths with `cudaErrorIllegalAddress` / Xid 31
(`FAULT_PDE ACCESS_TYPE_VIRT_READ`) when Inkling-Small serves more than one request at a
time (`--max-num-seqs` > 1, the recipe default is 2). With
`CUDA_LAUNCH_BLOCKING=1` the traceback ends in
`inkling_sm120_fa4/interface.py _flash_attn_fwd`; without it the error usually surfaces
later, e.g. as `CUBLAS_STATUS_INTERNAL_ERROR` in the MLP.

It patches the FA4 bundle vendored by `mods/inkling-sm12-paged-kv`, so it must come after
that mod. It edits one file in the container
(`vllm/third_party/inkling_sm120_fa4/flash_fwd.py`), only if it finds the exact expected
code, and is safe to apply more than once.

## Cause

`FlashAttentionForwardSm80.kernel`, which the SM120 kernel inherits, misses two things the
SM90/SM100 kernels in the same bundle do. Both only matter with 2+ sequences per batch.

1. **Score-mod indices wrapped by the batch length (the crash).** For `aux_tensors` reads,
   `q_idx`/`kv_idx` are wrapped "to not read OOB" with `fastdiv_mods`, which the host builds
   from the whole batch (`mQ.shape[0]`). SM90/SM100 rebuild them per sequence. Inkling's
   relative-bias score-mod reads `rel_logits[offset_q + q_idx]`, so for every sequence after
   the first, the padding rows of its 128-row tile read up to `min(offset_q, padding rows)`
   rows past the end of `rel_logits`. Those values are masked out, so outputs are correct,
   but the read faults whenever `rel_logits` ends at the edge of mapped memory, hence the
   intermittent crashes.
2. **Spare varlen tiles not skipped.** `SingleTileVarlenScheduler` over-provisions the grid;
   spare tiles carry `batch_idx == num_batch` and `is_valid_tile == False`. SM90/SM100 skip
   them; SM80 read `cu_seqlens_q[num_batch + 1]` and `seqused_k[num_batch]`.

## What the patch does

- Rebuilds `fastdiv_mods` from the sequence's own `seqlen_q`/`seqlen_k` after
  `SeqlenInfoQK.create` (the SM90 code, plus a `max(.., 1)` guard).
- Points a spare tile's batch index at a real sequence and gives it an empty K range, so the
  existing `n_block_max > n_block_min` guard skips it.

## Testing (2x GB10 / DGX Spark, SM121, vLLM 0.26.1rc1.dev515)

- Each kernel input placed in its own mapping with unmapped guard pages on both sides, so
  one byte out of bounds faults immediately. Original kernel: every 2-request batch faults,
  at 0 bytes past the end of `rel_logits`; with spare tiles also past `cu_seqlens_q`.
  Patched: 140 kernel calls (70 random batches of 1-8 requests, each for both layer types),
  guards before and after, no faults, all match a float32 reference.
- Outputs bit-identical to the original kernel over 80 kernel calls (25.3 M values).
- Live: before the fix, 2 sequences under mixed load crashed after ~5 minutes. With it:
  30-min soak at 4 sequences (529 requests), 10-min soak at 8 (232 requests), no faults.
  Throughput: 22.8 tok/s for one request, 43 for two, 71 for four.

The same score-mod bug exists in upstream FlashAttention's SM80/SM120 kernel; remove this mod
once a bundle with per-sequence `fastdiv_mods` is vendored (the patcher refuses to patch a
kernel that already has them).
