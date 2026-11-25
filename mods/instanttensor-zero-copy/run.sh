#!/bin/bash
set -euo pipefail

PREFIX="[instanttensor-zero-copy]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHER="$MOD_DIR/patch_weight_utils.py"

echo "=== InstantTensor zero-copy weight loader mod ==="

if ! command -v python3 >/dev/null 2>&1; then
    echo "$PREFIX python3 is required to locate and patch vLLM." >&2
    exit 1
fi

if [[ ! -f "$PATCHER" ]]; then
    echo "$PREFIX patcher not found: $PATCHER" >&2
    exit 1
fi

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

TARGET="$VLLM_PACKAGE_ROOT/model_executor/model_loader/weight_utils.py"
if [[ ! -f "$TARGET" ]]; then
    echo "$PREFIX vLLM weight utilities not found: $TARGET" >&2
    exit 1
fi

python3 "$PATCHER" --check "$TARGET"
python3 "$PATCHER" "$TARGET"
python3 "$PATCHER" --check "$TARGET"

echo "$PREFIX Enabled copy=False for the vLLM InstantTensor iterator."
echo "$PREFIX WARNING: use only with model loaders that consume each weight inline."
echo "=== OK: InstantTensor will avoid its per-tensor ownership clone ==="
