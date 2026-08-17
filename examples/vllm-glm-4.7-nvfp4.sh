#!/bin/bash
# PROFILE: Salyut1/GLM-4.7-NVFP4
# DESCRIPTION: vLLM serving GLM-4.7-NVFP4
# NOTE: This profile requires --apply-mod mods/fix-Salyut1-GLM-4.7-NVFP4 to fix k/v scales incompatibility
# See: https://huggingface.co/Salyut1/GLM-4.7-NVFP4/discussions/3#694ab9b6e2efa04b7ecb0c4b

vllm serve Salyut1/GLM-4.7-NVFP4 \
    --attention-config.backend flashinfer \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    -tp 2 \
    --gpu-memory-utilization 0.88 \
    --max-model-len 32000 \
    --host 0.0.0.0 \
    --port 8000
