#!/usr/bin/env bash
# glm53-nvidia-online-mxfp8 — load-time quantization of NVIDIA's excluded BF16 layers.
# STATUS: experimental; disabled unless VLLM_GLM53_ONLINE_MXFP8 selects layer groups.
# WHY: quantizing KDA/MLA attention and shared experts reduces weight bandwidth.
# ENABLE: VLLM_GLM53_ONLINE_MXFP8=attn,shared; VLLM_GLM53_ONLINE_MXFP8_A16=0.
# NVFP4 OVERRIDE: VLLM_GLM53_ONLINE_NVFP4=shared,oproj,qkv;
#   VLLM_GLM53_ONLINE_NVFP4_A16=1. Overrides only already-selected groups.
# WARNING: do not enable mtp. Simultaneous BF16 and quantized expert weights
#   during loading exhausted memory and took a node offline in a 2x DGX Spark setup.
# WARNING: MXFP8 A16=1 reduced prefill throughput by 14%; use A16=0.
#   NVFP4 defaults to A16=1 (BF16 activations); A16=0 uses FP4 activations.
# MEASURED: see README.md for the TP2+DCP2, MTP k3, 512K benchmark results.
# EXECUTION EVIDENCE: "[GLM53-ONLINE-MXFP8] <layer> -> MXFP8 online" or
#   "[GLM53-ONLINE-NVFP4] <layer> -> NVFP4 online" in the loading log.
# ROLLBACK: unset the four variables and restart the engine to reload original weights;
#   remove the mod from the recipe before recreating the container to remove the patch.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
python3 "$HERE/patch_online_mxfp8.py" --vllm-root "$VLLM"
python3 -m py_compile "$VLLM/model_executor/layers/quantization/modelopt.py"
echo "[glm53-nvidia-online-mxfp8] OK"
