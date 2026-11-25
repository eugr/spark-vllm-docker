#!/bin/bash
set -euo pipefail

PREFIX="[radixark-dspark]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/radixark-dspark.patch"

if ! command -v python3 >/dev/null 2>&1; then
    echo "$PREFIX python3 is required to locate and validate vLLM." >&2
    exit 1
fi

if [ ! -f "$PATCH_FILE" ]; then
    echo "$PREFIX patch file not found: $PATCH_FILE" >&2
    exit 1
fi

# VLLM_PACKAGE_ROOT is useful for tests and unusual image layouts. Normally,
# discover the package without importing vLLM so CUDA initialization cannot
# occur while cluster containers are still being prepared.
if [ -z "${VLLM_PACKAGE_ROOT:-}" ]; then
    VLLM_PACKAGE_ROOT=$(python3 - <<'PY'
import importlib.util

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vLLM package is not installed for the active Python interpreter")
print(next(iter(spec.submodule_search_locations)))
PY
    )
fi

TARGET="$VLLM_PACKAGE_ROOT/config/speculative.py"
if [ ! -f "$TARGET" ]; then
    echo "$PREFIX vLLM speculative config source not found: $TARGET" >&2
    exit 1
fi

has_radixark_support() {
    python3 - "$TARGET" <<'PY'
from pathlib import Path
import sys

source = " ".join(Path(sys.argv[1]).read_text().split())
detection = (
    '"DSparkDraftModel" in self.draft_model_config.architectures '
    'and self.draft_model_config.hf_config.model_type == "qwen3"'
)
conversion = (
    'self.draft_model_config.hf_config.architectures = [ '
    '"Qwen3DSparkModel" ]'
)
raise SystemExit(0 if detection in source and conversion in source else 1)
PY
}

if has_radixark_support; then
    echo "$PREFIX Qwen DSparkDraftModel normalization is already present; skipping."
else
    if ! command -v git >/dev/null 2>&1; then
        echo "$PREFIX git is required to apply this mod." >&2
        echo "$PREFIX Apply mods/use-official-vllm first for images without git." >&2
        exit 1
    fi

    SITE_PACKAGES="$(dirname "$VLLM_PACKAGE_ROOT")"
    if git -C "$SITE_PACKAGES" apply --check "$PATCH_FILE" 2>/dev/null; then
        git -C "$SITE_PACKAGES" apply "$PATCH_FILE"
        echo "$PREFIX Applied RadixArk Qwen DSpark configuration fix."
    else
        echo "$PREFIX Patch could not be applied to $TARGET." >&2
        echo "$PREFIX Expected a vLLM build with Qwen3 DSpark support and the generic DeepSeek fallback." >&2
        echo "$PREFIX Refusing to modify an unknown or incompatible implementation." >&2
        exit 1
    fi
fi

if ! has_radixark_support; then
    echo "$PREFIX Qwen DSpark compatibility postcondition failed." >&2
    exit 1
fi

python3 -m py_compile "$TARGET"
echo "$PREFIX RadixArk Qwen DSpark checkpoints now resolve to Qwen3DSparkModel."
