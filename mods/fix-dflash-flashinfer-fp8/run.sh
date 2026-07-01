#!/bin/bash
set -e

echo "--- Locating vLLM package directory..."
VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || true)

if [ -z "$VLLM_DIR" ]; then
    echo "=== Error: vLLM package not found in Python environment!" >&2
    exit 1
fi

PARENT_DIR=$(dirname "$VLLM_DIR")
echo "--- Found vLLM parent directory: $PARENT_DIR"

echo "--- Applying DFlash FlashInfer SWA & FP8 KV cache patch (PR #39995)..."
patch -p1 -d "$PARENT_DIR" < dflash_flashinfer.patch || echo "=== Warning: Failed to apply patch, check if already applied."
echo "=== OK"
