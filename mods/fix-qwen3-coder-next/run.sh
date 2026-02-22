#!/bin/bash
set -e

# Triton allocator workaround for DGX Spark (GB10/sm121)
# Patches Triton's NullAllocator to use PyTorch's CUDA caching allocator
# Tracking: https://github.com/vllm-project/vllm/issues/33857
echo "Fixing Triton allocator bug"
cp _triton* /usr/local/lib/python3.12/dist-packages/
