#!/bin/bash
# Fix Marlin TP=4 constraint for Qwen3.5 with int4 (GPTQ-AutoRound) quantization.
#
# Problem: gdn_linear_attn.GatedDeltaNetAttention uses MergedColumnParallelLinear
# for in_proj_ba with output_sizes=[num_v_heads]*2 (Qwen3.5 layout). At TP=4
# with num_v_heads=64 the per-rank output_size_per_partition is 16, far below
# Marlin's GPTQ_MARLIN_MIN_THREAD_N=64. vLLM falls back through every WNA16
# kernel (Cutlass W4A8/Machete need Hopper; AllSpark doesn't support
# device_capability=121; Conch isn't installed; Exllama needs fp16
# activations) and aborts engine init.
#
# Solution (per upstream issue vllm-project/vllm#35924, ported to
# post-refactor gdn_linear_attn.py location in vLLM 0.18+):
#   1. Replace the single MergedColumnParallelLinear in_proj_ba with two
#      ReplicatedLinear modules, prefixed `<...>.in_proj_ba.0` and `.1`.
#      Each rank holds the full b/a weight; we slice the local TP portion
#      in forward (see _project_ba). Marlin then sees the full
#      output_size = num_v_heads = 64 per partition, clearing the floor.
#   2. Update Qwen3_5MoeForCausalLM and Qwen3_5ForConditionalGeneration's
#      packed_modules_mapping so the loader continues to split the
#      checkpoint's packed in_proj_ba.{scales,qweight,qzeros} tensors —
#      now routing the halves to in_proj_ba.0.* and in_proj_ba.1.* (the
#      two ReplicatedLinear prefixes) instead of the previous in_proj_b /
#      in_proj_a names. Without this rename, int4 quantization scales fail
#      to load and engine init hangs in profile_run.
#
# Qwen3-Next (gqa_interleaved_layout=True) keeps the original packed module
# — its interleaved-GQA layout has different constraints and isn't the
# target of this fix. qwen3_next.py needs no changes.
#
# Hard-fails (set -e + reject detection) on any patch rejection. If a
# future vLLM bump moves the patch context, the launch must STOP here
# loudly rather than ship a partially-patched build that crashes later
# with a misleading error.

set -e
MOD_DIR="$(dirname "$0")"
MAMBA_DIR="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba"
MODELS_DIR="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"

apply_patch() {
  local NAME="$1"
  local PATCH_FILE="$2"
  local TARGET_DIR="$3"
  local TARGET_FILE="$4"

  echo "[fix-qwen35-tp4-marlin] Applying $NAME..."

  if [ ! -f "$TARGET_DIR/$TARGET_FILE" ]; then
    echo "[fix-qwen35-tp4-marlin] ERROR: target file missing: $TARGET_DIR/$TARGET_FILE"
    echo "[fix-qwen35-tp4-marlin] vLLM layout may have changed — refresh patch context."
    exit 1
  fi

  rm -f "$TARGET_DIR/$TARGET_FILE.rej"

  set +e
  local PATCH_OUTPUT
  PATCH_OUTPUT=$(patch --forward --batch -p0 -d "$TARGET_DIR" < "$PATCH_FILE" 2>&1)
  local PATCH_EXIT=$?
  set -e

  if [ $PATCH_EXIT -eq 0 ]; then
    if [ -f "$TARGET_DIR/$TARGET_FILE.rej" ]; then
      echo "[fix-qwen35-tp4-marlin] $NAME PARTIALLY REJECTED (despite exit 0):"
      echo "$PATCH_OUTPUT"
      echo "--- rejected hunks ---"
      cat "$TARGET_DIR/$TARGET_FILE.rej"
      echo "[fix-qwen35-tp4-marlin] STOP: refusing to ship a partially-patched build."
      exit 1
    fi
    echo "$PATCH_OUTPUT"
    echo "[fix-qwen35-tp4-marlin] $NAME applied."
  elif echo "$PATCH_OUTPUT" | grep -q "Reversed (or previously applied)"; then
    rm -f "$TARGET_DIR/$TARGET_FILE.rej"
    echo "[fix-qwen35-tp4-marlin] $NAME was already applied."
  else
    echo "[fix-qwen35-tp4-marlin] $NAME FAILED (exit $PATCH_EXIT):"
    echo "$PATCH_OUTPUT"
    if [ -f "$TARGET_DIR/$TARGET_FILE.rej" ]; then
      echo "--- rejected hunks ---"
      cat "$TARGET_DIR/$TARGET_FILE.rej"
    fi
    echo "[fix-qwen35-tp4-marlin] STOP: refusing to ship a partially-patched build."
    exit 1
  fi
}

apply_patch "gdn_linear_attn.patch" "$MOD_DIR/gdn_linear_attn.patch" \
  "$MAMBA_DIR" "gdn_linear_attn.py"

apply_patch "qwen3_5.patch" "$MOD_DIR/qwen3_5.patch" \
  "$MODELS_DIR" "qwen3_5.py"

echo "[fix-qwen35-tp4-marlin] Done."
