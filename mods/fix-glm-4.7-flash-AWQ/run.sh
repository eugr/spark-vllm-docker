#!/bin/bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Overridable for tests; the default matches the vLLM image layout.
VLLM_PACKAGE_ROOT="${VLLM_PACKAGE_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SITE_PACKAGES="$(dirname "$VLLM_PACKAGE_ROOT")"

TRITON_MLA="$VLLM_PACKAGE_ROOT/v1/attention/backends/mla/triton_mla.py"
MLA_ATTENTION="$VLLM_PACKAGE_ROOT/model_executor/layers/attention/mla_attention.py"

echo "--- Applying GLM 4.7 AWQ speed patch..."
if [ ! -f "$TRITON_MLA" ]; then
    echo "=== Error: $TRITON_MLA not found" >&2
    exit 1
fi
if grep -q "max(32, min(128, max_seq_len // 1500))" "$TRITON_MLA"; then
    echo "=== Speed patch is already applied, skipping"
elif grep -q "_compute_num_kv_splits" "$TRITON_MLA"; then
    # Upstream vLLM now sizes num_kv_splits from the sequence length itself,
    # which replaces the hardcoded split count this patch was written for.
    echo "=== vLLM already computes num_kv_splits dynamically (_compute_num_kv_splits), skipping"
elif patch -p1 --dry-run -s -f -d "$SITE_PACKAGES" < "$MOD_DIR/glm47_flash.patch" >/dev/null; then
    patch -p1 -s -f -d "$SITE_PACKAGES" < "$MOD_DIR/glm47_flash.patch"
    echo "=== OK"
else
    echo "=== Error: $TRITON_MLA is in an unexpected state; speed patch does not apply" >&2
    exit 1
fi

echo "--- Applying vLLM crash patch (34695)..."
# Vendored copy of vllm-project/vllm#34695 so the mod works offline.
if [ -f "$MLA_ATTENTION" ] && grep -q "and hasattr(self.kv_b_proj, \"weight\")" "$MLA_ATTENTION"; then
    echo "=== PR 34695 is already applied, skipping"
elif patch -p1 --dry-run -s -f -d "$SITE_PACKAGES" < "$MOD_DIR/glm47_vllm_bug.patch" >/dev/null; then
    patch -p1 -s -f -d "$SITE_PACKAGES" < "$MOD_DIR/glm47_vllm_bug.patch"
    echo "=== OK"
else
    # Not fatal: the crash only affects some AWQ/GPTQ paths, and the dry run
    # guarantees the file was left untouched.
    echo "=== Warning: Failed to apply PR 34695, continuing..."
fi
