#!/bin/bash
set -euo pipefail

PREFIX="[dflash2-compressed-lm-head]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/dflash2-compressed-lm-head.patch"
DEFAULT_PYTHON_ROOT="/usr/local/lib/python3.12/dist-packages"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-$DEFAULT_PYTHON_ROOT}}"
VLLM_ROOT="$PYTHON_ROOT/vllm"
TARGET="$VLLM_ROOT/model_executor/models/qwen3_dflash2.py"

if ! command -v git >/dev/null 2>&1; then
  echo "$PREFIX git is required to apply this mod." >&2
  exit 1
fi

if [[ ! -f "$TARGET" ]]; then
  echo "$PREFIX DFlash2 source not found at $TARGET" >&2
  echo "$PREFIX Build vLLM with PR #52816 before applying this mod." >&2
  exit 1
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "$PREFIX patch file not found at $PATCH_FILE" >&2
  exit 1
fi

cd "$PYTHON_ROOT"

if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "$PREFIX Patch is already applied; skipping."
elif git apply --check "$PATCH_FILE"; then
  git apply "$PATCH_FILE"
  echo "$PREFIX Enabled compressed-tensors target LM heads for DFlash2."
else
  echo "$PREFIX Patch could not be applied to the installed DFlash2 source." >&2
  echo "$PREFIX Requires the current vLLM PR #52816 implementation." >&2
  exit 1
fi

echo "$PREFIX The target lm_head remains quantized; DFlash2 calls its quant_method.apply()."
