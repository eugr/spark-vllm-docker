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
GLM_FLASH_AWQ_CONTAINER="vllm-node"
GLM_FLASH_AWQ_MOD="mods/fix-glm-4.7-flash-AWQ"
GLM_FLASH_AWQ_ARGS=(
    "--tool-call-parser glm47"
    "--reasoning-parser glm45"
    "--enable-auto-tool-choice"
    "--served-model-name glm-4.7-flash"
    "--max-model-len 202752"
    "--max-num-batched-tokens 4096"
    "--max-num-seqs 64"
    "--gpu-memory-utilization 0.8"
    "--port 8000"
    "--host 0.0.0.0"
)

# ==============================================================================
# openai-gpt-oss-120b
# README Reference: Lines 244-257 (solo) and 264-280 (cluster)
# ==============================================================================
GPT_OSS_MODEL="openai/gpt-oss-120b"
GPT_OSS_CONTAINER="vllm-node-mxfp4"
GPT_OSS_ARGS=(
    "--port 8000"
    "--host 0.0.0.0"
    "--enable-auto-tool-choice"
    "--tool-call-parser openai"
    "--reasoning-parser openai_gptoss"
    "--gpu-memory-utilization 0.8"
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
    "--gpu-memory-utilization 0.8"
    "--max-model-len 128000"
    "--load-format instanttensor"
    "--enable-auto-tool-choice"
    "--tool-call-parser minimax_m2"
    "--reasoning-parser minimax_m2"
)

# ==============================================================================
# inkling-2-mini
# README Reference: Inkling 2 Mini on a Single DGX Spark
# ==============================================================================
INKLING_2_MINI_MODEL="thinkingmachines/Inkling-2-Mini"
INKLING_2_MINI_CONTAINER="vllm/vllm-openai:inkling-2-mini"
INKLING_2_MINI_ARGS=(
    "--port 8000"
    "--host 0.0.0.0"
    "--tensor-parallel-size 1"
    "--max-model-len 262144"
    "--gpu-memory-utilization 0.8"
    "--reasoning-parser inkling"
    "--enable-auto-tool-choice"
    "--tool-call-parser inkling"
    "--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}'"
)

# Inkling 2 Mini with the Inco AI DFlash2 draft
INKLING_2_MINI_DFLASH2_CONTAINER="vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e"
INKLING_2_MINI_DFLASH2_ARGS=(
    "--revision c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
    "--tokenizer-revision c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
    "--port 8000"
    "--host 0.0.0.0"
    "--tensor-parallel-size 1"
    "--max-model-len 262144"
    "--gpu-memory-utilization 0.8"
    "--reasoning-parser inkling"
    "--enable-auto-tool-choice"
    "--tool-call-parser inkling"
    "--moe-backend humming"
    "--speculative-config '{\"method\":\"dflash\",\"model\":\"incoai/Inkling-2-Mini-DFlash2\",\"revision\":\"4f9088ea66c21f2b99ed657d1941c306b03e84d2\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}'"
)

# ==============================================================================
# qwen3.8-27b-nvfp4-dflash2-fp8-head
# README Reference: DFlash2 with compressed-tensors FP8 lm_head
# ==============================================================================
QWEN38_DFLASH2_MODEL="unsloth/Qwen3.8-27B-NVFP4"
QWEN38_DFLASH2_CONTAINER="vllm-node-dflash2"
QWEN38_DFLASH2_MOD="mods/dflash2-compressed-lm-head"
QWEN38_DFLASH2_BUILD_ARG="--apply-vllm-pr 52816"
QWEN38_DFLASH2_ARGS=(
    "--port 8000"
    "--host 0.0.0.0"
    "--tensor-parallel-size 1"
    "--gpu-memory-utilization 0.8"
    "--max-model-len 262144"
    "--max-num-seqs 4"
    "--max-num-batched-tokens 8192"
    "--kv-cache-dtype fp8"
    "--attention-backend flashinfer"
    "--load-format fastsafetensors"
    "--enable-chunked-prefill"
    "--enable-prefix-caching"
    "--speculative-config '{\"method\":\"dflash\",\"model\":\"incoai/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":7}'"
    "--enable-auto-tool-choice"
    "--tool-call-parser qwen3_xml"
    "--reasoning-parser qwen3"
)

# ==============================================================================
# Cluster Mode Expected Arguments
# These are arguments that should appear ONLY in cluster mode
# Note: Tests use 2 nodes, so tensor_parallel = 2 (1 GPU per node)
# ==============================================================================

# glm-4.7-flash-awq cluster mode (no distributed backend - single GPU model)
GLM_FLASH_AWQ_CLUSTER_TP="1"

# openai-gpt-oss-120b is solo_only; cluster mode is tested as a rejection.

# minimax-m2-awq cluster mode (2 nodes = tp 2)
MINIMAX_CLUSTER_TP="2"
MINIMAX_CLUSTER_ARGS=()

# Qwen3.8 Flash Next cluster: device PLE is part of the generated command.
# recipes/README.md: Environment Variables and Volumes
QWEN38_FLASH_NEXT_CLUSTER_TP="2"
QWEN38_FLASH_NEXT_CLUSTER_ARGS=(
    "--additional-config '{\"ple_table_memory\":\"device\"}'"
)
