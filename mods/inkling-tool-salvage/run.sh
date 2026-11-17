#!/bin/bash
set -euo pipefail

PREFIX="[inkling-tool-salvage]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON_ROOT="/usr/local/lib/python3.12/dist-packages"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-$DEFAULT_PYTHON_ROOT}}"
VLLM_ROOT="$PYTHON_ROOT/vllm"
TARGET="$VLLM_ROOT/parser/abstract_parser.py"
PATCHER="$MOD_DIR/patch_inkling_salvage.py"

echo "=== Inkling streaming tool-call salvage mod ==="

if [[ ! -d "$VLLM_ROOT" ]]; then
    echo "$PREFIX vLLM package not found at $VLLM_ROOT" >&2
    exit 1
fi
if [[ ! -f "$PATCHER" ]]; then
    echo "$PREFIX patcher not found at $PATCHER" >&2
    exit 1
fi
if [[ ! -f "$TARGET" ]]; then
    echo "$PREFIX DelegatingParser module not found at $TARGET" >&2
    echo "$PREFIX (this image's vLLM parser layout is unexpected)" >&2
    exit 1
fi

python3 "$PATCHER" "$TARGET"
python3 "$PATCHER" --check "$TARGET"

find "$(dirname "$TARGET")" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "$PREFIX Streaming tool calls that leak as content are recovered as tool_calls."
echo "=== OK: multi-turn streaming turn-initial tool calls no longer leak ==="
