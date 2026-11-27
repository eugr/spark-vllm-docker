#!/usr/bin/env bash
# glm53-nvidia-mtp-unquantized — the NVIDIA checkpoint stores its MTP layer in BF16.
#
# STATUS (2026-09-22): drafter loading verified; performance not measured.
# EXPECTED RESULT: load the MTP drafter instead of failing in _load_w2.
# Predicted accepted length: 2.3–2.5, based on a Spark-checkpoint MTP3 baseline
# of 2.428 at k=3; comparable decode speed expected, not measured here.
# If acceptance is much lower, check drafter loading before tuning k.
# EXECUTION EVIDENCE: the startup log uses [GLM53-NVIDIA-MTP-UNQUANT]
# and reports layer [45]. Check the reported indices and reason, not only the prefix.
# REQUIREMENT: set "moe_backend":"auto" in speculative-config.
# Marlin does not support the unquantized drafter MoE.
# ROLLBACK: remove this mod from mods: and recreate the container.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
python3 "$HERE/patch_mtp_unquant.py" --vllm-root "$VLLM"
python3 -m py_compile "$VLLM/model_executor/layers/quantization/modelopt.py"
echo "[glm53-nvidia-mtp-unquantized] OK"
