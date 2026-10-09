#!/usr/bin/env bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== b12x-native-backend mod ==="

# b12x_moe.py / b12x_ep_moe.py / mxfp4/nvfp4/scaled_mm b12x.py / the attention
# backends all call into the real, public b12x package (b12x.moe.*,
# b12x.attention.*, b12x.gemm.*, b12x._lib.intrinsics) at import/call time.
#
# --no-deps is required: b12x declares torch>=2.12.0, but pristine
# vllm-openai:v0.26.0 ships torch==2.11.0+cu130 built against vLLM's own
# compiled CUDA extensions. An unconstrained `pip install b12x` silently
# pulls torch 2.13.0 (+ new cudnn/nccl/triton/cublas, 2GB+) and breaks that
# ABI. Every other declared b12x dependency (cuda-python, nvidia-cutlass-dsl
# ==4.6.0, transformers, safetensors, rich, apache-tvm-ffi) is already
# present and already at a satisfying version in the pristine image, so
# --no-deps only skips the one problematic pin.
python3 -c "import b12x" 2>/dev/null || pip install --no-cache-dir --no-deps b12x

python3 "${MOD_DIR}/patch_b12x_native.py"

# sm_121 (GB10) o-proj bf16 fallback: bypass DeepGEMM fp8_einsum (no arch-12
# scale-layout; vLLM #41063) for the shared DeepSeek-V4 attention o-proj.
python3 "${MOD_DIR}/patch_sm121_o_proj_bf16.py"

# sm_121 (GB10) b12x MoE prep wiring: delegate process_weights_after_loading to fused_experts.
python3 "${MOD_DIR}/patch_sm121_moe_prep.py"
