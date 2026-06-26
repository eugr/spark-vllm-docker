# Expected vLLM serve arguments for each recipe
# This file is used by test_recipes.sh to verify recipes match README documentation
#
# Format: Each recipe has a section with expected arguments
# Tests will verify these arguments appear in the dry-run output
#
# IMPORTANT: Keep this in sync with README.md documentation
# When updating recipes, update both README.md and this file

# ==============================================================================
# glm-4.7-flash-awq
# README Reference: Lines 186-198 (solo) and 203-218 (cluster)
# ==============================================================================
GLM_FLASH_AWQ_MODEL="cyankiwi/GLM-4.7-Flash-AWQ-4bit"
GLM_FLASH_AWQ_CONTAINER="vllm-node-tf5"
GLM_FLASH_AWQ_MOD="mods/fix-glm-4.7-flash-AWQ"
GLM_FLASH_AWQ_ARGS=(
    "--tool-call-parser glm47"
    "--reasoning-parser glm45"
    "--enable-auto-tool-choice"
    "--served-model-name glm-4.7-flash"
    "--max-model-len 202752"
    "--max-num-batched-tokens 4096"
    "--max-num-seqs 64"
    "--gpu-memory-utilization 0.7"
    "--port 8000"
    "--host 0.0.0.0"
)

# ==============================================================================
# openai-gpt-oss-120b
# README Reference: Lines 244-257 (solo) and 264-280 (cluster)
# ==============================================================================
GPT_OSS_MODEL="openai/gpt-oss-120b"
GPT_OSS_CONTAINER="sparkarena/spark-vllm-docker:mxfp4"
GPT_OSS_ARGS=(
    "--port 8000"
    "--host 0.0.0.0"
    "--enable-auto-tool-choice"
    "--tool-call-parser openai"
    "--reasoning-parser openai_gptoss"
    "--gpu-memory-utilization 0.7"
    "--enable-prefix-caching"
    "--load-format fastsafetensors"
    "--quantization mxfp4"
    "--mxfp4-backend CUTLASS"
    "--mxfp4-layers moe,qkv,o,lm_head"
    "--attention-backend FLASHINFER"
    "--kv-cache-dtype fp8"
    "--max-num-batched-tokens 8192"
)

# ==============================================================================
# minimax-m2-awq
# README Reference: Not explicitly documented, but based on model requirements
# ==============================================================================
MINIMAX_MODEL="QuantTrio/MiniMax-M2-AWQ"
MINIMAX_CONTAINER="vllm-node"
MINIMAX_ARGS=(
    "--port 8000"
    "--host 0.0.0.0"
    "--gpu-memory-utilization 0.7"
    "--max-model-len 128000"
    "--load-format fastsafetensors"
    "--enable-auto-tool-choice"
    "--tool-call-parser minimax_m2"
    "--reasoning-parser minimax_m2"
)

# ==============================================================================
# Cluster Mode Expected Arguments
# These are arguments that should appear ONLY in cluster mode
# Note: Tests use 2 nodes, so tensor_parallel = 2 (1 GPU per node)
# ==============================================================================

# glm-4.7-flash-awq cluster mode (no distributed backend - single GPU model)
GLM_FLASH_AWQ_CLUSTER_TP="1"

# openai-gpt-oss-120b cluster mode (2 nodes = tp 2)
GPT_OSS_CLUSTER_TP="2"
GPT_OSS_CLUSTER_ARGS=(
    "--distributed-executor-backend ray"
)

# minimax-m2-awq cluster mode (2 nodes = tp 2)
MINIMAX_CLUSTER_TP="2"
MINIMAX_CLUSTER_ARGS=(
    "--distributed-executor-backend ray"
)
