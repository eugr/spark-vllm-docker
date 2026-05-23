#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/uma_fix.patch"

if ! command -v git >/dev/null 2>&1; then
  echo "[uma-fix] git is required to apply this mod." >&2
  echo "[uma-fix] Apply mods/use-official-vllm first if this container does not include git." >&2
  exit 1
fi

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "[uma-fix] vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

cd "$PYTHON_ROOT"

if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "[uma-fix] Patch is already applied; skipping."
elif git apply --check "$PATCH_FILE"; then
  git apply "$PATCH_FILE"
  echo "[uma-fix] Applied UMA memory accounting fix."
else
  echo "[uma-fix] Patch could not be applied to installed vLLM." >&2
  exit 1
fi
