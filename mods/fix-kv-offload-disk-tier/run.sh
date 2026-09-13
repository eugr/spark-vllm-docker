#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="[fix-kv-offload-disk-tier]"

PATCHES=(
  "01-eagle-store-filter.patch"
  "02-multinode-promoted-row-resync.patch"
  "03-match-without-staging.patch"
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

# Whole-set guard, checked BEFORE the per-patch loop.
#
# The per-patch "already applied" test below reverse-checks each patch in
# isolation, which stopped working once a third overlapping patch existed:
# reversing 01 alone fails while 03's edits to the same regions are present, so
# a second run ERRORED instead of skipping. Since 03 was generated against
# 01+02, its context contains their changes -- so if 03 reverse-applies, the
# whole stack is in place.
LAST="${PATCHES[${#PATCHES[@]}-1]}"
if git apply --reverse --check "$MOD_DIR/$LAST" 2>/dev/null; then
  echo "$PREFIX all ${#PATCHES[@]} patches already applied; skipping."
  exit 0
fi

# Applied in order: 02 touches scheduler.py after 01 does, and 03 after both.
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

echo "=====> Disk-backed KV offload tier: EAGLE/MTP store filter + multi-node re-sync"
echo "=====> + matching decoupled from staging (03), which is what makes the tier"
echo "=====> actually usable on a prefix larger than your primary tier."
echo "=====> Set PYTHONHASHSEED so block hashes are stable across restarts."
echo "=====> Tuning (all optional, sane defaults):"
echo "=====>   VLLM_OFFLOAD_STREAM_WAVE_CHUNKS=64  chunks per wave; 0 = 03 fully inert"
echo "=====>   VLLM_OFFLOAD_PARK=0                 admission gate; off by default"
echo "=====> If you see repeated \"cannot store chunks\": your primary tier is too"
echo "=====> small for the store batch. Raise cpu_bytes_to_use."
