#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="[fix-kv-offload-disk-tier]"

PATCHES=(
  "01-eagle-store-filter.patch"
  "02-multinode-promoted-row-resync.patch"
)

if ! command -v git >/dev/null 2>&1; then
  echo "$PREFIX git is required to apply this mod." >&2
  echo "$PREFIX Apply mods/use-official-vllm first if needed." >&2
  exit 1
fi

if [ ! -d "$PYTHON_ROOT/vllm/v1/kv_offload/tiering" ]; then
  echo "$PREFIX This vLLM has no v1/kv_offload/tiering; it predates" >&2
  echo "$PREFIX TieringOffloadingSpec and does not need this mod." >&2
  exit 1
fi

cd "$PYTHON_ROOT"

# Applied in order: 02 touches scheduler.py after 01 does.
for patch in "${PATCHES[@]}"; do
  file="$MOD_DIR/$patch"
  if git apply --reverse --check "$file" 2>/dev/null; then
    echo "$PREFIX $patch already applied; skipping."
  elif git apply --check "$file" 2>/dev/null; then
    git apply "$file"
    echo "$PREFIX applied $patch"
  else
    echo "$PREFIX $patch could not be applied to installed vLLM." >&2
    echo "$PREFIX Verified against vLLM e2666d9a65f41fc376607531453cbd57c4c71016." >&2
    exit 1
  fi
done

echo "=====> Disk-backed KV offload tier: EAGLE/MTP store filter + multi-node re-sync."
echo "=====> Set PYTHONHASHSEED so block hashes are stable across restarts."
