#!/bin/bash
set -euo pipefail

PREFIX="[fix-dsv4-effort-levels]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON_ROOT="/usr/local/lib/python3.12/dist-packages"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-$DEFAULT_PYTHON_ROOT}}"
TOK_DIR="$PYTHON_ROOT/vllm/tokenizers"
PATCHER="$MOD_DIR/patch_dsv4_effort.py"

echo "=== DeepSeek-V4 reasoning-effort levels mod ==="

if [[ ! -d "$TOK_DIR" ]]; then
    echo "$PREFIX vLLM tokenizers dir not found at $TOK_DIR" >&2
    exit 1
fi
if [[ ! -f "$PATCHER" ]]; then
    echo "$PREFIX patcher not found at $PATCHER" >&2
    exit 1
fi

for f in deepseek_v4_encoding.py deepseek_v4.py; do
    target="$TOK_DIR/$f"
    if [[ ! -f "$target" ]]; then
        echo "$PREFIX target not found: $target" >&2
        exit 1
    fi
    python3 "$PATCHER" --check "$target"
    python3 "$PATCHER" "$target"
    python3 "$PATCHER" --check "$target"
done

find "$TOK_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "$PREFIX low / high / max now render three distinct reasoning-effort prefixes."
echo "=== OK ==="
