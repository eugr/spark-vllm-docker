#!/bin/bash
set -euo pipefail

PREFIX="[inkling-tool-unescape]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON_ROOT="/usr/local/lib/python3.12/dist-packages"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-$DEFAULT_PYTHON_ROOT}}"
VLLM_ROOT="$PYTHON_ROOT/vllm"
TARGET="$VLLM_ROOT/parser/inkling.py"
PATCHER="$MOD_DIR/patch_inkling_tools.py"

echo "=== Inkling tool-call HTML-entity unescape mod ==="

if [[ ! -d "$VLLM_ROOT" ]]; then
    echo "$PREFIX vLLM package not found at $VLLM_ROOT" >&2
    exit 1
fi
if [[ ! -f "$PATCHER" ]]; then
    echo "$PREFIX patcher not found at $PATCHER" >&2
    exit 1
fi
if [[ ! -f "$TARGET" ]]; then
    echo "$PREFIX Inkling parser not found at $TARGET" >&2
    echo "$PREFIX (this image has no Inkling tool parser — is it Inkling-capable?)" >&2
    exit 1
fi

python3 "$PATCHER" "$TARGET"
python3 "$PATCHER" --check "$TARGET"

# Drop stale bytecode so vLLM re-imports the patched source.
find "$(dirname "$TARGET")" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "$PREFIX Tool-call args are HTML-unescaped before the JSON scanner."
echo "=== OK: sampled entities (e.g. &quot;) no longer break Inkling tool calls ==="
