#!/bin/bash
set -euo pipefail

# vllm-spark-min-nodes: 4
# vllm-spark-safety-note: TP=2 on two GB10 128G nodes caused NVIDIA OOM and host reboot.

MODEL_PATH="${MODEL_PATH:-/models/deepseek-v4-flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Flash}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
LINEAR_BACKEND="${LINEAR_BACKEND:-triton}"
EAGER_ARGS=()
if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" ]]; then
    EAGER_ARGS+=(--enforce-eager)
fi
LINEAR_BACKEND_ARGS=()
if [[ -n "$LINEAR_BACKEND" && "$LINEAR_BACKEND" != "auto" ]]; then
    LINEAR_BACKEND_ARGS+=(--linear-backend "$LINEAR_BACKEND")
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "model path not mounted: $MODEL_PATH" >&2
    echo "start with: VLLM_SPARK_EXTRA_DOCKER_ARGS='-v /home/novaadmin/models:/models:ro'" >&2
    exit 66
fi

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4 \
    --host 0.0.0.0 \
    --port 30000 \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    "${EAGER_ARGS[@]}" \
    "${LINEAR_BACKEND_ARGS[@]}" \
    --distributed-executor-backend ray
