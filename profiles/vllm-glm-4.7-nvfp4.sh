#!/bin/bash
# Profile: Salyut1/GLM-4.7-NVFP4
# Description: vLLM serving GLM-4.7-NVFP4 with required compatibility patch

PROFILE_NAME="GLM-4.7-NVFP4"
PROFILE_DESCRIPTION="vLLM serving Salyut1/GLM-4.7-NVFP4 with glm4_moe patch for fused k/v scales"

PROFILE_RUNTIME="vllm"
PROFILE_COMMAND="serve"
PROFILE_MODEL="Salyut1/GLM-4.7-NVFP4"

PROFILE_ARGS=(
    --attention-config.backend flashinfer
    --tool-call-parser glm47
    --reasoning-parser glm45
    --enable-auto-tool-choice
    -tp 2
    --gpu-memory-utilization 0.88
    --max-model-len 32000
    --distributed-executor-backend ray
    --host 0.0.0.0
    --port 8000
)

# Required mod to fix k/v scales incompatibility
# See: https://huggingface.co/Salyut1/GLM-4.7-NVFP4/discussions/3#694ab9b6e2efa04b7ecb0c4b
PROFILE_MODS=(
    "mods/fix-Salyut1-GLM-4.7-NVFP4"
)
