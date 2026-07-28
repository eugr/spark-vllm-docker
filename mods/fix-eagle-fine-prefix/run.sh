#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/fix-eagle-fine-prefix.patch"
PREFIX="[fix-eagle-fine-prefix]"

if ! command -v git >/dev/null 2>&1; then
  echo "$PREFIX git is required to apply this mod." >&2
  echo "$PREFIX Apply mods/use-official-vllm first if needed." >&2
  exit 1
fi

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "$PREFIX vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

cd "$PYTHON_ROOT"

if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "$PREFIX Patch is already applied; skipping."
elif git apply --check "$PATCH_FILE"; then
  git apply "$PATCH_FILE"
  echo "$PREFIX Applied changed-suffix EAGLE fine-prefix patch."
else
  echo "$PREFIX Patch could not be applied to installed vLLM." >&2
  echo "$PREFIX Requires vLLM with PR #46384, based on 41ea2dd44a3a20c46ebeb985de0022c7673fb953." >&2
  exit 1
fi

echo "=====> Fine-grained changed-suffix MTP hits enabled; set --prefix-match-unit to a divisor of every KV cache group block size."
