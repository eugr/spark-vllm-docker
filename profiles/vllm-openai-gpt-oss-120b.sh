#!/bin/bash
# Profile: OpenAI GPT-OSS 120B
# Description: vLLM serving openai/gpt-oss-120b with FlashInfer MOE optimization

PROFILE_NAME="OpenAI GPT-OSS 120B"
PROFILE_DESCRIPTION="vLLM serving openai/gpt-oss-120b with FlashInfer MOE MXFP4/MXFP8 optimization"

PROFILE_RUNTIME="vllm"
PROFILE_COMMAND="serve"
PROFILE_MODEL="openai/gpt-oss-120b"

PROFILE_ARGS=(
    --tool-call-parser openai
    --enable-auto-tool-choice
    --tensor-parallel-size 2
    --distributed-executor-backend ray
    --kv-cache-dtype fp8
    --gpu-memory-utilization 0.90
    --max-model-len 128000
    --max-num-batched-tokens 4096
    --max-num-seqs 8
    --enable-prefix-caching
    --host 0.0.0.0
    --port 8000
)

profile_init() {
    # Enable FlashInfer MOE with MXFP4/MXFP8 quantization
    export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
    echo "Enabled VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1"
}
