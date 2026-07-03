#!/bin/bash
# Mod: guard DSpark speculator against models without draft_id_to_target_id.
# Fixes: AttributeError: 'DSparkDeepseekV4ForCausalLM' object has no attribute
#        'draft_id_to_target_id'  (vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:86)
# Upstream speculator assumes every DSpark model exposes draft_id_to_target_id
# (reduced-vocab drafting). DeepSeek-V4 DSpark uses full-vocab Markov drafting
# and does not define it. interfaces.py already uses getattr(..., None) for the
# same attribute; this applies the same guard to the speculator, so the optional
# reduced-vocab scatter path is simply skipped (the None defaults).
set -e

VLLM_DIR=/usr/local/lib/python3.12/dist-packages
F="$VLLM_DIR/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"

if [ ! -f "$F" ]; then
  echo "--- [fix-dspark-dsv4-d2t] speculator not found; this image has no DSpark speculator, skipping."
  exit 0
fi

if grep -q '_d2t = getattr(model, "draft_id_to_target_id", None)' "$F"; then
  echo "--- [fix-dspark-dsv4-d2t] patch already applied, skipping."
  exit 0
fi

echo "--- [fix-dspark-dsv4-d2t] applying DSpark speculator draft_id_to_target_id guard..."
patch -p1 -d "$VLLM_DIR" < dspark_speculator_d2t.patch
echo "=== OK"
