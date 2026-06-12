#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/gpu_mem.patch"

if ! command -v git >/dev/null 2>&1; then
  echo "[gpu-mem-util-gb] git is required to apply this mod." >&2
  echo "[gpu-mem-util-gb] Apply mods/use-official-vllm first if this container does not include git." >&2
  exit 1
fi

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "[gpu-mem-util-gb] vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

cd "$PYTHON_ROOT"

if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "[gpu-mem-util-gb] Patch is already applied; skipping."
elif git apply --check "$PATCH_FILE"; then
  git apply "$PATCH_FILE"
  echo "[gpu-mem-util-gb] Applied --gpu-memory-utilization-gb support."
else
  echo "[gpu-mem-util-gb] Patch could not be applied to installed vLLM." >&2
  exit 1
fi

echo "=====> You can now use --gpu-memory-utilization-gb to specify reserved memory in GiB"
