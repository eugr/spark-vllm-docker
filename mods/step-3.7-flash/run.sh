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

has_upstream_step37_support() {
  local model_file="$PYTHON_ROOT/vllm/model_executor/models/step3p7.py"
  local model_registry="$PYTHON_ROOT/vllm/model_executor/models/registry.py"
  local tokenizer_registry="$PYTHON_ROOT/vllm/tokenizers/registry.py"
  local speculative_config="$PYTHON_ROOT/vllm/config/speculative.py"

  [ -f "$model_file" ] || return 1
  [ -f "$model_registry" ] || return 1
  [ -f "$tokenizer_registry" ] || return 1
  [ -f "$speculative_config" ] || return 1

  grep -q "class Step3p7ForConditionalGeneration" "$model_file" || return 1
  grep -q "Step3p7ForConditionalGeneration" "$model_registry" || return 1
  grep -q "step3p7" "$tokenizer_registry" || return 1
  grep -q "step3p7" "$speculative_config" || return 1
}

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "[step-3.7-flash] vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

cd "$PYTHON_ROOT"

if has_upstream_step37_support; then
  echo "[step-3.7-flash] Installed vLLM already has Step-3.7-Flash support; skipping patch."
elif ! command -v git >/dev/null 2>&1; then
  echo "[step-3.7-flash] git is required to apply this mod." >&2
  echo "[step-3.7-flash] Apply mods/use-official-vllm first if this container does not include git." >&2
  exit 1
elif git apply --reverse --check "${PATCH_EXCLUDES[@]}" "$PATCH_FILE" 2>/dev/null; then
  echo "[step-3.7-flash] Patch is already applied; skipping."
elif git apply --check "${PATCH_EXCLUDES[@]}" "$PATCH_FILE"; then
  git apply "${PATCH_EXCLUDES[@]}" "$PATCH_FILE"
  echo "[step-3.7-flash] Applied step-3.7-flash support patch."
else
  echo "[step-3.7-flash] Patch could not be applied to installed vLLM." >&2
  exit 1
fi
