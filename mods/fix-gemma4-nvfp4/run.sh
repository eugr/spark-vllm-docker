#!/bin/bash
set -e

# Replace the gemma4.py model executor with the patched version for modelopt
cp gemma4_patched.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py
echo "Successfully applied Gemma 4 NVFP4 patch"
