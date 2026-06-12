#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
START_DIR="$PWD"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/diffusiongemma-support.patch"
ATTENTION_LEGACY_PATCH_FILE="$MOD_DIR/diffusiongemma-attention-legacy.patch"
ATTENTION_MAIN_PATCH_FILE="$MOD_DIR/diffusiongemma-attention-main.patch"
STREAMING_REASONING_PATCH_FILE="$MOD_DIR/gemma4-streaming-reasoning.patch"
CONTENT_CHANNEL_SANITIZER_PATCH_FILE="$MOD_DIR/gemma4-content-channel-sanitizer.patch"
CHAT_TEMPLATE_FILE="$MOD_DIR/chat_template_no_think.jinja"
PATCH_EXCLUDES=(
  --exclude="benchmarks/*"
  --exclude="cmake/*"
  --exclude="diffusion_gemma_scratch/*"
  --exclude="docs/*"
  --exclude="examples/*"
  --exclude="tests/*"
  --exclude="vllm/v1/attention/backends/flash_attn.py"
  --exclude="vllm/vllm_flash_attn/flash_attn_interface.py"
  --exclude="wip.md"
)

has_upstream_diffusiongemma_support() {
  local model_file="$PYTHON_ROOT/vllm/model_executor/models/diffusion_gemma.py"
  local model_registry="$PYTHON_ROOT/vllm/model_executor/models/registry.py"
  local config_registry="$PYTHON_ROOT/vllm/transformers_utils/config.py"
  local diffusion_config="$PYTHON_ROOT/vllm/config/diffusion.py"

  [ -f "$model_file" ] || return 1
  [ -f "$model_registry" ] || return 1
  [ -f "$config_registry" ] || return 1
  [ -f "$diffusion_config" ] || return 1

  grep -q "class DiffusionGemmaForConditionalGeneration" "$model_file" || return 1
  grep -q "DiffusionGemmaForBlockDiffusion" "$model_registry" || return 1
  grep -q "diffusion_gemma" "$config_registry" || return 1
  grep -q "class DiffusionConfig" "$diffusion_config" || return 1
  grep -q "self_conditioning_embeds" "$model_file" || return 1
}

has_marker() {
  local path="$1"
  local pattern="$2"
  [ -f "$path" ] && grep -Fq "$pattern" "$path"
}

has_gemma4_streaming_reasoning_patch() {
  has_marker "vllm/entrypoints/openai/chat_completion/serving.py" "replacement_delta: dict[str, Any]" &&
    has_marker "vllm/reasoning/gemma4_reasoning_parser.py" "_disabled_raw_text"
}

has_gemma4_content_channel_sanitizer_patch() {
  has_marker "vllm/entrypoints/openai/chat_completion/serving.py" "_strip_gemma4_content_channels" &&
    has_marker "vllm/reasoning/gemma4_reasoning_parser.py" "wait_for_end=True"
}

has_diffusiongemma_attention_patch() {
  has_marker "vllm/v1/attention/backends/flash_attn.py" "dynamic_causal=dynamic_causal" &&
    has_marker "vllm/vllm_flash_attn/flash_attn_interface.py" "dynamic_causal: \"torch.Tensor | None\" = None"
}

select_attention_patch() {
  if has_marker "vllm/v1/attention/backends/flash_attn.py" "mm_prefix_range_tensor"; then
    echo "$ATTENTION_MAIN_PATCH_FILE"
  else
    echo "$ATTENTION_LEGACY_PATCH_FILE"
  fi
}

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "[diffusiongemma] vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

cd "$PYTHON_ROOT"

if has_upstream_diffusiongemma_support; then
  echo "[diffusiongemma] Installed vLLM already has DiffusionGemma support; skipping patch."
elif ! command -v git >/dev/null 2>&1; then
  echo "[diffusiongemma] git is required to apply this mod." >&2
  echo "[diffusiongemma] Apply mods/use-official-vllm first if this container does not include git." >&2
  exit 1
elif git apply --reverse --check "${PATCH_EXCLUDES[@]}" "$PATCH_FILE" 2>/dev/null; then
  echo "[diffusiongemma] Patch is already applied; skipping."
elif git apply --check "${PATCH_EXCLUDES[@]}" "$PATCH_FILE"; then
  git apply "${PATCH_EXCLUDES[@]}" "$PATCH_FILE"
  echo "[diffusiongemma] Applied DiffusionGemma support patch."
else
  echo "[diffusiongemma] Patch could not be applied to installed vLLM." >&2
  exit 1
fi

ATTENTION_PATCH_FILE="$(select_attention_patch)"
if [ ! -f "$ATTENTION_PATCH_FILE" ]; then
  echo "[diffusiongemma] attention compatibility patch not found: $ATTENTION_PATCH_FILE" >&2
  exit 1
elif ! command -v git >/dev/null 2>&1; then
  echo "[diffusiongemma] git is required to apply the attention compatibility patch." >&2
  echo "[diffusiongemma] Apply mods/use-official-vllm first if this container does not include git." >&2
  exit 1
elif has_diffusiongemma_attention_patch; then
  echo "[diffusiongemma] DiffusionGemma attention compatibility patch is already applied; skipping."
elif git apply --reverse --check "$ATTENTION_PATCH_FILE" 2>/dev/null; then
  echo "[diffusiongemma] DiffusionGemma attention compatibility patch is already applied; skipping."
elif git apply --check "$ATTENTION_PATCH_FILE"; then
  git apply "$ATTENTION_PATCH_FILE"
  echo "[diffusiongemma] Applied DiffusionGemma attention compatibility patch."
else
  echo "[diffusiongemma] DiffusionGemma attention compatibility patch could not be applied to installed vLLM." >&2
  exit 1
fi

if [ ! -f "$STREAMING_REASONING_PATCH_FILE" ]; then
  echo "[diffusiongemma] Gemma4 streaming reasoning patch not found: $STREAMING_REASONING_PATCH_FILE" >&2
  exit 1
elif ! command -v git >/dev/null 2>&1; then
  echo "[diffusiongemma] git is required to apply the Gemma4 streaming reasoning patch." >&2
  echo "[diffusiongemma] Apply mods/use-official-vllm first if this container does not include git." >&2
  exit 1
elif has_gemma4_streaming_reasoning_patch; then
  echo "[diffusiongemma] Gemma4 streaming reasoning patch is already applied; skipping."
elif git apply --reverse --check "$STREAMING_REASONING_PATCH_FILE" 2>/dev/null; then
  echo "[diffusiongemma] Gemma4 streaming reasoning patch is already applied; skipping."
elif git apply --check "$STREAMING_REASONING_PATCH_FILE"; then
  git apply "$STREAMING_REASONING_PATCH_FILE"
  echo "[diffusiongemma] Applied Gemma4 streaming reasoning patch."
else
  echo "[diffusiongemma] Gemma4 streaming reasoning patch could not be applied to installed vLLM." >&2
  exit 1
fi

if [ ! -f "$CONTENT_CHANNEL_SANITIZER_PATCH_FILE" ]; then
  echo "[diffusiongemma] Gemma4 content channel sanitizer patch not found: $CONTENT_CHANNEL_SANITIZER_PATCH_FILE" >&2
  exit 1
elif ! command -v git >/dev/null 2>&1; then
  echo "[diffusiongemma] git is required to apply the Gemma4 content channel sanitizer patch." >&2
  echo "[diffusiongemma] Apply mods/use-official-vllm first if this container does not include git." >&2
  exit 1
elif has_gemma4_content_channel_sanitizer_patch; then
  echo "[diffusiongemma] Gemma4 content channel sanitizer patch is already applied; skipping."
elif git apply --reverse --check "$CONTENT_CHANNEL_SANITIZER_PATCH_FILE" 2>/dev/null; then
  echo "[diffusiongemma] Gemma4 content channel sanitizer patch is already applied; skipping."
elif git apply --check "$CONTENT_CHANNEL_SANITIZER_PATCH_FILE"; then
  git apply "$CONTENT_CHANNEL_SANITIZER_PATCH_FILE"
  echo "[diffusiongemma] Applied Gemma4 content channel sanitizer patch."
else
  echo "[diffusiongemma] Gemma4 content channel sanitizer patch could not be applied to installed vLLM." >&2
  exit 1
fi

if [ ! -f "$CHAT_TEMPLATE_FILE" ]; then
  echo "[diffusiongemma] chat template not found: $CHAT_TEMPLATE_FILE" >&2
  exit 1
else
  CHAT_TEMPLATE_TARGET_DIR="${WORKSPACE_DIR:-$START_DIR}"
  cp "$CHAT_TEMPLATE_FILE" "$CHAT_TEMPLATE_TARGET_DIR/fixed_chat_template.jinja"
  echo "[diffusiongemma]=======> to apply chat template, use --chat-template fixed_chat_template.jinja"
fi
