#!/usr/bin/env bash
set -e

FILE="/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/model_arch_config_convertor.py"

echo "[mod] Enabling MLA for glm4_moe_lite"

# Insert glm4 entries right after deepseek_mtp if not already present
if ! grep -q '"glm4_moe_lite"' "$FILE"; then
  sed -i '/"deepseek_mtp",/a\
        "glm4_moe_lite",\
        "glm4_moe_lite_mtp",' "$FILE"
fi

echo "[mod] Verifying change"
grep -n "glm4_moe_lite" "$FILE"

echo "[mod] MLA patch applied successfully"
