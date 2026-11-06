#!/bin/bash
set -euo pipefail

PREFIX="[fix-inkling-fa4-sm120-varlen-oob]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
BUNDLE="$PYTHON_ROOT/vllm/third_party/inkling_sm120_fa4"
TARGET="$BUNDLE/flash_fwd.py"

echo "=== Inkling SM12 FA4 multi-request out-of-bounds fix ==="

if [[ ! -f "$TARGET" ]]; then
    echo "$PREFIX target not found: $TARGET" >&2
    echo "$PREFIX apply mods/inkling-sm12-paged-kv first (this mod patches the bundle it installs)." >&2
    exit 1
fi

python3 "$MOD_DIR/patch_fa4_varlen.py" "$TARGET"

COUNT=$(grep -c "oplc-fix-fa4-sm120-varlen-oob" "$TARGET" || true)
if [[ "$COUNT" -ne 3 ]]; then
    echo "$PREFIX verification FAILED (marker count=$COUNT, want 3)" >&2
    exit 1
fi
python3 -m py_compile "$TARGET"
# Compiled kernels are cached per process, but stale bytecode must not shadow the patch.
find "$BUNDLE" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
python3 -c "from vllm.third_party.inkling_sm120_fa4.flash_fwd import FlashAttentionForwardSm80"
echo "$PREFIX applied and verified (marker count=$COUNT)"
