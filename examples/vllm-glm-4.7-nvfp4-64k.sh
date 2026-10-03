#!/bin/bash
# PROFILE: Salyut1/GLM-4.7-NVFP4 (full 355B GLM-4.7, NVFP4)
# DESCRIPTION: Full GLM-4.7-NVFP4 serving at 64K context on 2x DGX Spark GB10 (TP=2)
# NOTE: Requires --apply-mod mods/fix-Salyut1-GLM-4.7-NVFP4, and launch with --no-ray
#       (on vLLM 0.23.x the ray executor collides with the launcher's ray cluster:
#        "ActorHandleNotFoundError: ActorHandle objects are not valid across Ray sessions").
#       No --enforce-eager: CUDAGraph fits alongside the 64K KV cache at
#       --gpu-memory-utilization 0.90. Fallback for tighter units: add --enforce-eager
#       (and/or drop to 0.88) for ~12.4 tok/s instead of ~17.5.
# Verified: 2x GB10 (128 GB, driver 580.159.03), TP=2, 64K served, ~17.5 tok/s,
#           tool calls working, 21K-token needle recall passed.

vllm serve Salyut1/GLM-4.7-NVFP4 \
    --attention-config.backend flashinfer \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    -tp 2 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 8 \
    --max-model-len 65536 \
    --distributed-executor-backend ray \
    --host 0.0.0.0 \
    --port 8000
