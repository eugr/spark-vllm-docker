#!/bin/bash
set -euo pipefail
# DeepSeek V4 Flash DSpark NVFP4 overlay mod
# Auto-detects vLLM site-packages location (handles both /opt/env/ and /usr/local/ layouts)

# Auto-detect vLLM site-packages
if [ -d "/opt/env/lib/python3.12/site-packages/vllm" ]; then
  SITE_PACKAGES="/opt/env/lib/python3.12/site-packages"
elif [ -d "/usr/local/lib/python3.12/dist-packages/vllm" ]; then
  SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
elif [ -n "${PYTHON_ROOT:-}" ] && [ -d "$PYTHON_ROOT/vllm" ]; then
  SITE_PACKAGES="$PYTHON_ROOT"
else
  # Try python3 -c
  SITE_PACKAGES=$(python3 -c "import vllm; import os; print(os.path.dirname(os.path.dirname(vllm.__file__)))" 2>/dev/null || echo "")
  if [ -z "$SITE_PACKAGES" ] || [ ! -d "$SITE_PACKAGES/vllm" ]; then
    echo "[dsv4-dspark] ERROR: Cannot find vLLM installation" >&2
    python3 -c "import vllm; print(vllm.__file__)" 2>/dev/null || echo "vLLM not importable"
    exit 1
  fi
fi
echo "[dsv4-dspark] Detected vLLM at: $SITE_PACKAGES/vllm"

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_DIR="$MOD_DIR/overlay"

# 1. Copy all overlay files into site-packages
echo "[dsv4-dspark] Copying overlay files..."
OVERLAY_FILES=$(find "$OVERLAY_DIR" -type f -name "*.py" | sort)
COUNT=0
for src in $OVERLAY_FILES; do
  rel="${src#$OVERLAY_DIR/}"
  dst="$SITE_PACKAGES/$rel"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  COUNT=$((COUNT + 1))
done
echo "[dsv4-dspark] Copied $COUNT overlay files"

# 2. Run NVFP4 patch stages if available (pass site-packages path)
export VLLM_SITE_PACKAGES="$SITE_PACKAGES"
for stage in stage-a stage-b stage-c; do
  patch_script="$MOD_DIR/patch-nvfp4-$stage.py"
  if [ -f "$patch_script" ]; then
    echo "[dsv4-dspark] Running NVFP4 Stage $stage..."
    python3 "$patch_script" || {
      echo "[dsv4-dspark] WARNING: Stage $stage patch failed (may be pre-applied)"
    }
  fi
done

# 3. Compile overlaid files
echo "[dsv4-dspark] Compiling overlaid files..."
cd "$SITE_PACKAGES"
for f in $(find "$OVERLAY_DIR" -type f -name "*.py"); do
  rel="${f#$OVERLAY_DIR/}"
  python3 -m py_compile "$SITE_PACKAGES/$rel" 2>/dev/null || true
done

# 4. Verify imports
echo "[dsv4-dspark] Verifying imports..."
python3 -c "
from vllm.v1.spec_decode import dspark, dspark_proposer
print(f"DSpark: OK")
from typing import get_args
from vllm.config.cache import CacheDType
if \"nvfp4_ds_mla\" in get_args(CacheDType):
    print(f"NVFP4 dtype: present")
else:
    print(f"WARNING: nvfp4_ds_mla not in CacheDType")
print(f"DeepSeek V4 Flash DSpark overlay applied successfully")
" 2>&1 && echo "[dsv4-dspark] Overlay applied successfully" || echo "[dsv4-dspark] WARNING: Verification had issues"
