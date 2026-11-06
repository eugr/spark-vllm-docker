#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/fix-anthropic-templateless-inline-system.patch"
PREFIX="[fix-anthropic-templateless-inline-system]"

if ! command -v git >/dev/null 2>&1; then
  echo "$PREFIX git is required to apply this mod." >&2
  echo "$PREFIX Apply mods/use-official-vllm first if needed." >&2
  exit 1
fi

if [ ! -f "$PYTHON_ROOT/vllm/entrypoints/anthropic/serving.py" ]; then
  echo "$PREFIX This vLLM has no Anthropic Messages frontend; nothing to do." >&2
  exit 1
fi

cd "$PYTHON_ROOT"

if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "$PREFIX Patch is already applied; skipping."
elif git apply --check "$PATCH_FILE" 2>/dev/null; then
  git apply "$PATCH_FILE"
  echo "$PREFIX Applied: templateless models no longer merge inline system messages."
else
  echo "$PREFIX Patch could not be applied to installed vLLM." >&2
  echo "$PREFIX Requires _detect_merge_inline_system, added by PR #46025." >&2
  exit 1
fi

echo "=====> Inline system messages stay in place for templateless models."
echo "=====> Only affects models with NO jinja chat template (--tokenizer-mode"
echo "=====> renderers such as deepseek_v4). Templated models are unchanged."
echo "=====> Set VLLM_ANTHROPIC_MERGE_INLINE_SYSTEM=1 to restore stock merging."
