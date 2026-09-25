#!/usr/bin/env bash
# Applies Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark's four
# upstream sglang/flash-attention fixes required to run Qwen3.8-Flash-Next on
# a single Spark (SM121, single-device fallback paths the reference two-Spark
# TP2 deployment never exercises).
# https://github.com/Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark
#
# Requires the lmsysorg/sglang:qwen38flashnext image (or an image with the
# same sglang/flash-attn install layout). Full-file replacements, not
# anchored edits — each target is backed up to <path>.orig on first run and
# the mod is a no-op on repeat runs against an already-patched container.
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGL="/sgl-workspace/sglang/python/sglang/srt"
FA="/usr/local/lib/python3.12/dist-packages/flash_attn/cute"

declare -A TARGETS=(
  ["qwen4_exp_nvfp4.py"]="$SGL/models/qwen4_exp.py"
  ["qwen_sparse_attn_backend.py"]="$SGL/layers/attention/qwen_sparse_attn_backend.py"
  ["sparse_attn.py"]="$SGL/layers/attention/qsa/sparse_attn.py"
  ["flash_fwd.py"]="$FA/flash_fwd.py"
)

for src in "${!TARGETS[@]}"; do
  dest="${TARGETS[$src]}"
  if [ ! -f "$dest" ]; then
    echo "!! $dest not found — this mod requires the qwen38flashnext sglang image" >&2
    exit 1
  fi
  if ! cmp -s "$MOD_DIR/$src" "$dest"; then
    [ -f "$dest.orig" ] || cp "$dest" "$dest.orig"
    cp "$MOD_DIR/$src" "$dest"
    echo "patched $dest"
  else
    echo "$dest already patched, skipping"
  fi
  python3 -c "import ast; ast.parse(open('$dest').read())"
done

echo "qwen3.8-flash-next-180b-hashk: all 4 patches applied OK"
