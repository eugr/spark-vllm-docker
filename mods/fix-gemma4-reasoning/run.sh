#!/bin/bash
set -e

# Gemma 4 requires transformers>=5.0 for model architecture recognition
if python3 -c "import transformers; v=int(transformers.__version__.split('.')[0]); exit(0 if v>=5 else 1)" 2>/dev/null; then
    echo "transformers >= 5.0 already installed, skipping upgrade."
else
    echo "Upgrading transformers to >=5.0 for Gemma 4 support..."
    pip install -q --upgrade "transformers>=5.0"
fi

echo "Fixing Gemma 4 reasoning parser (skip_special_tokens bug, vllm #38855)"

# Try applying PR #38858's diff first (most correct fix)
if curl -sfL https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/38858.diff \
    | patch -p1 --dry-run -d /usr/local/lib/python3.12/dist-packages > /dev/null 2>&1; then
    echo "Applying PR #38858 diff..."
    curl -sfL https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/38858.diff \
        | patch -p1 -d /usr/local/lib/python3.12/dist-packages
    echo "PR #38858 applied successfully."
else
    echo "PR #38858 diff does not apply cleanly, falling back to inline patch..."
    python3 "$(dirname "$0")/patch_reasoning.py"
    echo "Inline patch applied successfully."
fi
