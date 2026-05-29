#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/step-3.7-support.patch"
WORKSPACE="${WORKSPACE:-${WORKSPACE_DIR:-}}"
PATCH_EXCLUDES=(
  --exclude="docs/*"
  --exclude="examples/*"
  --exclude="tests/*"
)

if ! command -v git >/dev/null 2>&1; then
  echo "[step-3.7-flash] git is required to apply this mod." >&2
  echo "[step-3.7-flash] Apply mods/use-official-vllm first if this container does not include git." >&2
  exit 1
fi

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "[step-3.7-flash] vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

cd "$PYTHON_ROOT"

if git apply --reverse --check "${PATCH_EXCLUDES[@]}" "$PATCH_FILE" 2>/dev/null; then
  echo "[step-3.7-flash] Patch is already applied; skipping."
elif git apply --check "${PATCH_EXCLUDES[@]}" "$PATCH_FILE"; then
  git apply "${PATCH_EXCLUDES[@]}" "$PATCH_FILE"
  echo "[step-3.7-flash] Applied step-3.7-flash support patch."
else
  echo "[step-3.7-flash] Patch could not be applied to installed vLLM." >&2
  exit 1
fi
