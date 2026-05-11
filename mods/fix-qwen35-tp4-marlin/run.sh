#!/bin/bash
# Fix Marlin TP=4 constraint for Qwen3.5-397B: in_proj_ba output_size=128 / TP=4 = 32 < min_thread_n=64
# Solution: Replace MergedColumnParallelLinear with two ReplicatedLinear for B/A projections
# Delivery: unified diff patches (portable across vLLM versions)

set -e
MOD_DIR="$(dirname "$0")"
MODELS_DIR="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"

echo "[fix-qwen35-tp4-marlin] Applying patches..."

# Apply patches with --forward (skip if already applied)
patch --forward --batch -p0 -d "$MODELS_DIR" < "$MOD_DIR/qwen3_next.patch" || {
    echo "[fix-qwen35-tp4-marlin] qwen3_next.patch already applied or failed"
}
patch --forward --batch -p0 -d "$MODELS_DIR" < "$MOD_DIR/qwen3_5.patch" || {
    echo "[fix-qwen35-tp4-marlin] qwen3_5.patch already applied or failed"
}

# Fix rope validation (idempotent)
python3 "$MOD_DIR/fix_rope.py"

echo "[fix-qwen35-tp4-marlin] Done."
