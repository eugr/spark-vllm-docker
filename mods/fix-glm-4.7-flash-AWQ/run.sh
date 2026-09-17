#!/usr/bin/env bash
set -e
echo "--- Applying GLM 4.7 AWQ speed patch..."
patch -p1 -d / < glm47_flash.patch
echo "=== OK"
echo "--- Applying vLLM crash patch (34695)..."
# Check if PR 34695 is already applied by looking for the changed file
if [ -f /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py ]; then
    # Check if the specific line from PR 34695 is already present
    if grep -q "and hasattr(self.kv_b_proj, \"weight\")" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py 2>/dev/null; then
        echo "=== PR 34695 is already applied, skipping"
    else
        curl -L https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/34695.diff | patch -p1 -d /usr/local/lib/python3.12/dist-packages || echo "=== Warning: Failed to apply PR 34695, continuing..."
    fi
else
    curl -L https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/34695.diff | patch -p1 -d /usr/local/lib/python3.12/dist-packages || echo "=== Warning: Failed to apply PR 34695, continuing..."
fi
echo "=== OK"
