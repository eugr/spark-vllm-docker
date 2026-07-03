# Bug report: SM120 sparse-MLA paged-attention fallback kernel hangs on sm_121 (DGX Spark) for calls over 64 query tokens

Intended for: https://github.com/flashinfer-ai/flashinfer/issues
Related: #3374 / #3395 (the SM120 sparse-MLA kernels), #3610/#3615/#3618/#3625
(multi-CTA top-k races on SM120/121), #3170 (DGX Spark SM121 support audit).

## Environment

- 8x NVIDIA DGX Spark (GB10, sm_121, aarch64), one GPU per node,
  connected by 100G RoCE through a MikroTik QSFP switch
- FlashInfer 0.6.14 (release wheels: flashinfer_python + flashinfer_cubin +
  flashinfer_jit_cache cu130 aarch64)
- vLLM main (0.23.1rc1.dev730, 2026-07-02), CUDA 13.0, driver 580.159.03
- Model: nvidia/GLM-5.2-NVFP4 (GlmMoeDsaForCausalLM, DeepSeek Sparse
  Attention), TP=8 across the 8 nodes, kv_cache_dtype fp8 (fp8_ds_mla),
  block/page size 64, index_topk 2048, 8 heads per rank

## Symptom

Any request whose attention call carries more than 64 query tokens can hang
the whole cluster: one rank never returns from the attention kernel, the
other 7 ranks wait forever in their next NCCL all-reduce, and every GPU
shows ~96% utilization at low power (SMs parked, no work). No error, no
timeout, no Xid. Requests with fewer query tokens per call always work.

The hang is data-dependent at moderate sizes (roughly 100-250 query tokens
sometimes pass, sometimes hang) and reproduces reliably at 2048-token
prefill chunks. It survives every unrelated knob: cudagraph mode, MoE
backend, NCCL protocol/algorithm settings, and VLLM_USE_FLASHINFER_SAMPLER=0
(so it is not the #3610 sampler top-k race).

## Where it points

`flashinfer/mla/_sparse_mla_sm120.py` dispatches per call:

- `num_tokens <= _DECODE_MAX_TOKENS (64)` with page size 64 and a
  supported (num_heads, topk) pair -> `sparse_mla_sm120_decode_dsv3_2`
  (autotuned). This kernel works on sm_121, always.
- anything else -> `module.sparse_mla_sm120_paged_attention` (the generic
  fallback). This is the kernel that hangs on sm_121.

With `CUDA_LAUNCH_BLOCKING=1`, the stuck rank's Python stack ends inside
the fallback call:

```
_paged_attention (flashinfer/mla/_sparse_mla_sm120.py:392)   <- module.sparse_mla_sm120_paged_attention(...)
_sparse_mla_sm120_paged_attention (flashinfer/mla/_sparse_mla_sm120.py:492)
run (flashinfer/mla/_sparse_mla_sm120.py:708)
_trtllm_batch_decode_sparse_mla_sm120 (flashinfer/mla/_core.py:444)
```

while the other 7 ranks all sit in `ncclAllReduce` (pynccl). A raw 8-node
torch.distributed all-reduce sweep (1KB to 128MB) over the same NCCL build
and fabric is clean at ~89 Gb/s bus bandwidth, so the network is not
involved.

Given the #3615 finding (multi-CTA software-barrier reset race, "SM120/121
hit first because their 99KB smem/block makes the CTA groups widest"), a
plausible suspect is the same class of barrier bug in the fallback kernel,
or the fallback consuming top-k indices corrupted by the (still open)
#3618/#3625 shared-row-state issue. We did not disassemble the kernel; the
dispatch boundary and the blocking-mode stack are the hard evidence.

## Reproduction sketch

Serve GLM-5.2-NVFP4 (or presumably any DSv3.2-style DSA model) on sm_121
with vLLM's FLASHINFER_MLA_SPARSE_SM120 backend and chunked prefill at
2048 tokens; send any prompt of a few hundred tokens. First forward that
routes >64 query tokens into `sparse_mla_sm120_paged_attention` hangs the
stream. With `max_num_batched_tokens=64` (every call stays on the dsv3_2
kernel) the same deployment served two full GSM8K runs and a 60-request
concurrency soak with zero hangs.

## Workaround we ship

Because each query token in this MQA-style kernel attends independently
over its own top-k indices, oversized calls can be executed as exact
<=64-token slices on the dsv3_2 kernel. We patch `_paged_attention` to do
this (see `run.sh` in this directory). Measured on the 8x Spark cluster:
prefill went from ~180 tok/s (64-token scheduler cap) to ~880 tok/s
(2048-token chunks, attention sliced), output exactness verified with
multi-depth needle retrieval up to 319K tokens of context and GSM8K.

A proper fix in the fallback kernel (or extending dsv3_2 dispatch to larger
num_tokens) would make this patch unnecessary.
