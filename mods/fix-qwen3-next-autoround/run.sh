#!/bin/bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate the active interpreter's package without importing vLLM or starting CUDA.
# The override also permits offline fixture tests.
if [[ -z "${VLLM_PACKAGE_ROOT:-}" ]]; then
    VLLM_PACKAGE_ROOT=$(python3 - <<'PY'
import importlib.util

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vLLM package is not installed for the active Python interpreter")
print(next(iter(spec.submodule_search_locations)))
PY
    )
fi

python3 "$MOD_DIR/patch_qwen3_next.py" \
    "$VLLM_PACKAGE_ROOT/model_executor/models/qwen3_next.py"
