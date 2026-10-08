#!/bin/bash
set -euo pipefail

# MiMo-V2.6-Flash-RL enablement mod.
#
# Stages a loadable copy of the checkpoint's DFlash drafter at
# /workspace/MiMo-V2.6-Flash-RL-dflash. vLLM cannot address a Hub subdirectory
# ("org/repo/dflash") as a draft model, the drafter reads mask_embedding.pt from
# its own model root, and the shipped dflash/config.json contains a trailing
# comma that transformers' strict JSON parsing rejects. The staged directory
# symlinks the original weight files and writes a sanitized config.json.
#
# The mod also fails fast when the installed vLLM predates the MiMo-V2 fixes the
# recipe relies on (upstream PRs #57508 and #57784, both in vLLM main since
# 2026-09-20). It does not patch vLLM; use `--apply-vllm-pr` or a newer image.

PREFIX="[mimo-v2.6-flash]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
MODELS_DIR="$PYTHON_ROOT/vllm/model_executor/models"
MODEL_ID="${MIMO_V26_MODEL_ID:-XiaomiMiMo/MiMo-V2.6-Flash-RL}"
LINK_DIR="${MIMO_V26_LINK_DIR:-/workspace}"
DRAFT_DIR="$LINK_DIR/${MIMO_V26_LINK_NAME:-MiMo-V2.6-Flash-RL}-dflash"

echo "=== MiMo-V2.6-Flash mod ==="

if [ ! -f "$MODELS_DIR/mimo_v2.py" ]; then
  echo "$PREFIX Installed vLLM has no mimo_v2 model implementation at $MODELS_DIR; a newer image is required." >&2
  exit 1
fi
if ! grep -qF "def _shard_fp8_qkv_proj(" "$MODELS_DIR/mimo_v2.py"; then
  echo "$PREFIX Installed vLLM predates PR #57508 (fused fp8 qkv_proj TP sharding)." >&2
  echo "$PREFIX Rebuild from a newer vLLM ref or launch with --apply-vllm-pr 57508." >&2
  exit 1
fi
if ! grep -qF "GateLinear" "$MODELS_DIR/mimo_v2.py"; then
  echo "$PREFIX Installed vLLM predates PR #57784 (bf16 MoE router, MXFP4 experts, DFlash value scale)." >&2
  echo "$PREFIX Rebuild from a newer vLLM ref or launch with --apply-vllm-pr 57784." >&2
  exit 1
fi

if [ -n "${MIMO_V26_MODEL_DIR:-}" ]; then
  SNAPSHOT="$MIMO_V26_MODEL_DIR"
elif ! SNAPSHOT="$(python3 -c '
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_files_only=True))
' "$MODEL_ID")"; then
  echo "$PREFIX Could not locate a local snapshot of $MODEL_ID in the Hugging Face cache." >&2
  echo "$PREFIX Download it first (run-recipe.sh --setup / --download-only) or set MIMO_V26_MODEL_DIR." >&2
  exit 1
fi

for required in config.json dflash/config.json dflash/dflash_draft_model.safetensors dflash/mask_embedding.pt; do
  if [ ! -e "$SNAPSHOT/$required" ]; then
    echo "$PREFIX Snapshot $SNAPSHOT is missing $required." >&2
    exit 1
  fi
done

rm -rf "$DRAFT_DIR"
mkdir -p "$DRAFT_DIR"
for f in "$SNAPSHOT"/dflash/*; do
  base="$(basename "$f")"
  [ "$base" = "config.json" ] && continue
  ln -s "$(readlink -f "$f")" "$DRAFT_DIR/$base"
done
python3 - "$SNAPSHOT/dflash/config.json" "$DRAFT_DIR/config.json" <<'PY'
import json
import re
import sys

src, dst = sys.argv[1:3]
text = open(src, encoding="utf-8").read()
try:
    data = json.loads(text)
except json.JSONDecodeError:
    # Remove trailing commas before a closing brace/bracket.
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    data = json.loads(text)
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
PY
echo "$PREFIX DFlash drafter staged at $DRAFT_DIR (from $SNAPSHOT/dflash)"
echo "=== OK ==="
