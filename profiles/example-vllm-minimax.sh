#!/bin/bash
# Profile: MiniMax-M2-AWQ Example
# Description: vLLM serving MiniMax-M2-AWQ with Ray distributed backend

# Metadata (optional, for documentation)
PROFILE_NAME="MiniMax-M2-AWQ Example"
PROFILE_DESCRIPTION="vLLM serving MiniMax-M2-AWQ with Ray distributed backend"

# Runtime configuration
PROFILE_RUNTIME="vllm"
PROFILE_COMMAND="serve"
PROFILE_MODEL="QuantTrio/MiniMax-M2-AWQ"

# Arguments array - each element is a complete argument (handles = signs, spaces, etc.)
PROFILE_ARGS=(
    --port 8000
    --host 0.0.0.0
    --gpu-memory-utilization 0.7
    -tp 2
    --distributed-executor-backend ray
    --max-model-len 128000
    --load-format fastsafetensors
    --enable-auto-tool-choice
    --tool-call-parser minimax_m2
    --reasoning-parser minimax_m2_append_think
)

# Optional: profile_init() hook for custom setup logic
# This runs on the head node before the main command executes.
# Uncomment and customize as needed:
#
# profile_init() {
#     # Set environment variables
#     export HF_TOKEN="${HF_TOKEN:-}"
#     
#     # Download model ahead of time
#     # huggingface-cli download "$PROFILE_MODEL"
#     
#     # Any other pre-launch setup
#     echo "Profile initialized: $PROFILE_NAME"
# }
