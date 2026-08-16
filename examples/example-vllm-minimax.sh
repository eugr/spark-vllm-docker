#!/bin/bash
# PROFILE: MiniMax-M2-AWQ Example
# DESCRIPTION: vLLM serving MiniMax-M2-AWQ through launch-cluster.sh

vllm serve QuantTrio/MiniMax-M2-AWQ \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.8 \
    -tp 2 \
    --max-model-len 128000 \
    --load-format fastsafetensors \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2
