#!/bin/bash
set -e
echo "--- Applying Gemma4 NVFP4 MoE scale key patch..."
# Copy patched gemma4.py with NVFP4 MoE scale key mapping fix
# See: https://github.com/vllm-project/vllm/issues/38912
cp gemma4_patched.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py
echo "=== OK"
