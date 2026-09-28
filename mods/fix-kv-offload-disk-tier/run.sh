#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="[fix-kv-offload-disk-tier]"

PATCHES=(
  "01-eagle-store-filter.patch"
  "02-multinode-promoted-row-resync.patch"
  "03-match-without-staging.patch"
  "04-wave-readiness-at-load.patch"
  "05-fs-tier-complete-empty-jobs.patch"
  "06-bound-stuck-waves.patch"
  "07-swa-store-stride.patch"
  "08-fs-tier-free-space-eviction.patch"
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

# Find how much of the stack is already in place, then apply only the rest.
#
# A patch cannot be tested for "already applied" in isolation: reversing 01
# alone fails while a later patch's edits to the same regions are present. But
# each patch was generated against all the ones before it, so if patch k
# reverse-applies, 01..k are in place and k+1.. are not. Scanning from the top
# finds k; this also upgrades an install that has an older, shorter stack
# (e.g. 01-04 baked into an image) instead of erroring on it.
applied=0
for ((i=${#PATCHES[@]}; i>=1; i--)); do
  if git apply --reverse --check "$MOD_DIR/${PATCHES[i-1]}" 2>/dev/null; then
    applied=$i
    break
  fi
done
if [ "$applied" -eq "${#PATCHES[@]}" ]; then
  echo "$PREFIX all ${#PATCHES[@]} patches already applied; skipping."
  exit 0
fi
if [ "$applied" -gt 0 ]; then
  echo "$PREFIX ${PATCHES[0]} .. ${PATCHES[applied-1]} already applied."
fi

for patch in "${PATCHES[@]:applied}"; do
  file="$MOD_DIR/$patch"
  if git apply --check "$file" 2>/dev/null; then
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
echo "=====> actually usable on a prefix larger than your primary tier,"
echo "=====> + wave readiness evaluated at load (04), which fixes a prepare_load"
echo "=====> crash after long uptimes,"
echo "=====> + fs-tier zero-task jobs completed instead of leaked (05; idle CPU burn),"
echo "=====> + a stuck wave kills the engine after 120 s instead of spinning (06),"
echo "=====> + optional SWA store stride, ~4x less disk on hybrid models (07),"
echo "=====> + optional free-space LRU eviction for the fs tier (08)."
echo "=====> Set PYTHONHASHSEED so block hashes are stable across restarts."
echo "=====> Tuning (all optional, sane defaults):"
echo "=====>   VLLM_OFFLOAD_STREAM_WAVE_CHUNKS=64  chunks per wave; 0 = 03 and 04 fully inert"
echo "=====>   VLLM_OFFLOAD_PARK=0                 admission gate; off by default"
echo "=====>   VLLM_OFFLOAD_STREAM_WAVE_STUCK_S=120  seconds before a stuck wave kills the engine; 0 = never"
echo "=====>   VLLM_OFFLOAD_SWA_STORE_STRIDE=1     8 = ~25% of the disk bytes on DeepSeek V4; 1 = stock"
echo "=====>   VLLM_OFFLOAD_FS_MIN_FREE_GB=0      evict LRU chunk files below this much free disk; 0 = never"
echo "=====>   VLLM_OFFLOAD_FS_TARGET_FREE_GB     ...until this much is free (default 1.5x the minimum)"
echo "=====> If you see repeated \"cannot store chunks\": your primary tier is too"
echo "=====> small for the store batch. Raise cpu_bytes_to_use."
