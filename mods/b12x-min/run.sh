#!/bin/bash
set -euo pipefail

# Minimal b12x MoE enablement for the current (2026-07) vLLM main wheel +
# FlashInfer 0.6.14, which require nvidia-cutlass-dsl 4.5.x APIs. Unlike
# mods/exp-b12x, this does NOT pin cutlass-dsl to 4.4.2 (that pin breaks
# FlashInfer 0.6.14: cutlass.cute.nvgpu.OperandMajorMode is 4.5-only).

SITE_PACKAGES="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"

# 1. Drop stale sm120_moe_dispatch_context import if present (harmless if not)
SM12X_INIT="$SITE_PACKAGES/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/__init__.py"
if [ -f "$SM12X_INIT" ] && grep -q "sm120_moe_dispatch_context" "$SM12X_INIT"; then
  sed -i '/sm120_moe_dispatch_context/d' "$SM12X_INIT"
  echo "[b12x-min] cleaned stale import in blackwell_sm12x/__init__.py"
else
  echo "[b12x-min] blackwell_sm12x/__init__.py OK"
fi

# 2. Enable the b12x NVFP4 linear GEMM path on sm_121 (vLLM PR 40080 gate)
LINEAR="$SITE_PACKAGES/vllm/model_executor/kernels/linear/nvfp4/flashinfer.py"
if [ -f "$LINEAR" ] && grep -q "if current_platform.has_device_capability(120) and has_flashinfer_b12x_gemm():" "$LINEAR"; then
  sed -i "s/if current_platform.has_device_capability(120) and has_flashinfer_b12x_gemm():/if True:/" "$LINEAR"
  echo "[b12x-min] enabled b12x NVFP4 GEMM on sm_121"
else
  echo "[b12x-min] linear gate already patched or moved; skipping"
fi

find "$SITE_PACKAGES/flashinfer" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "=====> b12x-min applied (stock cutlass-dsl retained)"
