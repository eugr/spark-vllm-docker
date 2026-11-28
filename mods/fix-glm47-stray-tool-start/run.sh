#!/bin/bash
set -euo pipefail

PREFIX="[fix-glm47-stray-tool-start]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$MOD_DIR/glm47_stray_tool_start.patch"

echo "=== glm47 stray-tool-start mod ==="

# VLLM_PACKAGE_ROOT is useful for tests and unusual image layouts. Avoid
# importing vLLM during discovery so container preparation cannot initialize
# CUDA before the serving process starts.
if [[ -z "${VLLM_PACKAGE_ROOT:-}" ]]; then
    if [[ -n "${VLLM_SITE_PACKAGES:-}" ]]; then
        VLLM_PACKAGE_ROOT="$VLLM_SITE_PACKAGES/vllm"
    elif [[ -n "${PYTHON_ROOT:-}" ]]; then
        VLLM_PACKAGE_ROOT="$PYTHON_ROOT/vllm"
    else
        VLLM_PACKAGE_ROOT=$(python3 - <<'PY'
import importlib.util

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vLLM package is not installed for the active Python interpreter")
print(next(iter(spec.submodule_search_locations)))
PY
        )
    fi
fi

TARGET="$VLLM_PACKAGE_ROOT/parser/glm47_moe.py"
if [[ ! -f "$TARGET" ]]; then
    echo "$PREFIX vLLM glm47 parser not found: $TARGET" >&2
    exit 1
fi

if grep -Fq '(ParserState.TOOL_NAME, "TOOL_START")' "$TARGET"; then
    echo "$PREFIX already applied, skipping"
    exit 0
fi

patch -p1 -d "$VLLM_PACKAGE_ROOT" < "$PATCH"
echo "$PREFIX applied to $TARGET"
