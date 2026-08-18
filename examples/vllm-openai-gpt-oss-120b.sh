#!/bin/bash
# PROFILE: OpenAI GPT-OSS 120B
# DESCRIPTION: vLLM serving openai/gpt-oss-120b with FlashInfer MOE optimization

# Enable FlashInfer MOE with MXFP4/MXFP8 quantization
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

vllm serve openai/gpt-oss-120b \
    --tool-call-parser openai \
    --enable-auto-tool-choice \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 128000 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000
