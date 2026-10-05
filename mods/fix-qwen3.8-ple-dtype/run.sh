#!/bin/bash
set -euo pipefail

# Inject text_config.ple_embedding_dtype into the Qwen3.8-Flash-Next NVFP4
# checkpoint config. The upstream config.json omits the field while the PLE
# n-gram embedding is quantized to NVFP4, so vLLM plans a bf16 PLE table and
# weight loading fails with a shape mismatch. See this mod's README.md.
#
# Runs inside the launched container, where the host HF cache is bind-mounted
# at /root/.cache/huggingface.

PREFIX="[fix-qwen3.8-ple-dtype]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="${FIX_QWEN38_PLE_MODEL_ID:-local-inference-lab/Qwen3.8-Flash-Next-NVFP4}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
REPO_DIR_NAME="models--${MODEL_ID//\//--}"
HUB_DIR="$HF_HOME/hub/$REPO_DIR_NAME"

if ! command -v python3 >/dev/null 2>&1; then
    echo "$PREFIX python3 is required to patch the checkpoint config." >&2
    exit 1
fi

if [ ! -d "$HUB_DIR" ]; then
    echo "$PREFIX Model '$MODEL_ID' not found under $HF_HOME/hub;" >&2
    echo "$PREFIX download it before launching a recipe that uses this mod." >&2
    exit 1
fi

REF="${FIX_QWEN38_PLE_REF:-main}"
REF_FILE="$HUB_DIR/refs/$REF"

CONFIGS=()
if [ -s "$REF_FILE" ]; then
    REVISION=$(tr -d "[:space:]" <"$REF_FILE")
    if [ -z "$REVISION" ]; then
        echo "$PREFIX Ref '$REF' in $HUB_DIR/refs is empty." >&2
        exit 1
    fi
    TARGET_CONFIG="$HUB_DIR/snapshots/$REVISION/config.json"
    if [ ! -f "$TARGET_CONFIG" ]; then
        echo "$PREFIX Snapshot for revision '$REVISION' (refs/$REF) is missing" >&2
        echo "$PREFIX under $HUB_DIR/snapshots; re-download the model." >&2
        exit 1
    fi
    echo "$PREFIX Patching revision $REVISION (refs/$REF)"
    CONFIGS=("$TARGET_CONFIG")
else
    # No resolvable ref (legacy cache layout): patch every snapshot.
    shopt -s nullglob
    CONFIGS=("$HUB_DIR"/snapshots/*/config.json)
    shopt -u nullglob
    if [ "${#CONFIGS[@]}" -eq 0 ]; then
        echo "$PREFIX No config.json snapshot found in $HUB_DIR/snapshots." >&2
        exit 1
    fi
fi

python3 "$MOD_DIR/patch_config.py" "${CONFIGS[@]}"
