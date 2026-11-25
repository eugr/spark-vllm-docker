#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MOD_DIR="$PROJECT_DIR/mods/dflash2-compressed-lm-head"
FIXTURE="$SCRIPT_DIR/fixtures/dflash2/qwen3_dflash2.py"
TMP_BASE="$(mktemp -d)"

cleanup() {
    rm -rf "$TMP_BASE"
}
trap cleanup EXIT

SITE_PACKAGES="$TMP_BASE/site-packages"
TARGET_DIR="$SITE_PACKAGES/vllm/model_executor/models"
TARGET="$TARGET_DIR/qwen3_dflash2.py"
mkdir -p "$TARGET_DIR"
cp "$FIXTURE" "$TARGET"

first_output=$(VLLM_SITE_PACKAGES="$SITE_PACKAGES" "$MOD_DIR/run.sh")

grep -qF "CompressedTensorsLinearMethod" "$TARGET"
grep -qF "self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)" "$TARGET"
echo "$first_output" | grep -qF "Enabled compressed-tensors target LM heads"

checksum_before=$(sha256sum "$TARGET")
second_output=$(VLLM_SITE_PACKAGES="$SITE_PACKAGES" "$MOD_DIR/run.sh")
checksum_after=$(sha256sum "$TARGET")

[[ "$checksum_before" == "$checksum_after" ]]
echo "$second_output" | grep -qF "Patch is already applied; skipping"

echo "[PASS] DFlash2 compressed-tensors LM-head mod applies cleanly and is repeatable"
