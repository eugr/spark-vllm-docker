#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/radixark-dspark/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

VLLM_ROOT="$TMP_DIR/site-packages/vllm"
TARGET="$VLLM_ROOT/config/speculative.py"
mkdir -p "$(dirname "$TARGET")"

printf '%s\n' \
    'class SpeculativeConfig:' \
    '    def normalize(self):' \
    '        if True:' \
    '            if True:' \
    '                if self.method in ("eagle", "eagle3", "dflash", "dspark"):' \
    '                    pass' \
    '                elif (' \
    '                    "dspark" in self.draft_model_config.model.lower()' \
    '                    or "Qwen3DSparkModel" in self.draft_model_config.architectures' \
    '                    or "Gemma4DSparkModel" in self.draft_model_config.architectures' \
    '                ):' \
    '                    self.method = "dspark"' \
    '                elif self.draft_model_config.hf_config.model_type == "medusa":' \
    '                    self.method = "medusa"' \
    '' \
    '                if self.method in ("eagle", "eagle3", "dflash"):' \
    '                    if self.draft_model_config.hf_config:' \
    '                        pass' \
    '                    else:' \
    '                        eagle_config = None' \
    '                        self.draft_model_config.hf_config = eagle_config' \
    '                        self.update_arch_()' \
    '' \
    '                if self.method == "dspark" and (' \
    '                    "Qwen3DSparkModel" not in self.draft_model_config.architectures' \
    '                    and "Gemma4DSparkModel" not in self.draft_model_config.architectures' \
    '                    and "K3DSparkModel" not in self.draft_model_config.architectures' \
    '                ):' \
    '                    self.draft_model_config.hf_config.model_type = "deepseek_v4"' \
    '                    self.draft_model_config.hf_config.architectures = [' \
    '                        "DSparkDraftModel"' \
    '                    ]' \
    '                    self.draft_model_config.quantization = (' \
    '                        self.target_model_config.quantization' \
    '                    )' \
    '                    self.update_arch_()' \
    '' \
    '    def update_arch_(self):' \
    '        self.draft_model_config.architectures = list(' \
    '            self.draft_model_config.hf_config.architectures' \
    '        )' \
    > "$TARGET"

first_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
grep -Fq 'Applied RadixArk Qwen DSpark configuration fix.' \
    <<< "$first_output"
grep -Fq 'and self.draft_model_config.hf_config.model_type == "qwen3"' "$TARGET"
grep -Fq '"Qwen3DSparkModel"' "$TARGET"
python3 -m py_compile "$TARGET"

python3 - "$TARGET" <<'PY'
import importlib.util
import sys
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("fixture_speculative", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

draft_hf = SimpleNamespace(
    model_type="qwen3",
    architectures=["DSparkDraftModel"],
)
draft = SimpleNamespace(
    model="RadixArk/Qwen3.8-27B-DSpark",
    architectures=["DSparkDraftModel"],
    hf_config=draft_hf,
    quantization=None,
)
config = module.SpeculativeConfig()
config.method = "dspark"
config.draft_model_config = draft
config.target_model_config = SimpleNamespace(quantization="modelopt")
config.normalize()

assert draft.architectures == ["Qwen3DSparkModel"]
assert draft_hf.model_type == "qwen3"
assert draft.quantization is None
PY

before=$(sha256sum "$TARGET")
second_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
after=$(sha256sum "$TARGET")
test "$before" = "$after"
grep -Fq 'normalization is already present; skipping.' <<< "$second_output"

unsupported_root="$TMP_DIR/unsupported/vllm"
unsupported_target="$unsupported_root/config/speculative.py"
mkdir -p "$(dirname "$unsupported_target")"
printf '%s\n' 'class SpeculativeConfig: pass' > "$unsupported_target"
unsupported_before=$(sha256sum "$unsupported_target")
if VLLM_PACKAGE_ROOT="$unsupported_root" bash "$MOD" >/dev/null 2>&1; then
    echo "[FAIL] mod accepted an unsupported speculative config layout" >&2
    exit 1
fi
unsupported_after=$(sha256sum "$unsupported_target")
test "$unsupported_before" = "$unsupported_after"

echo "[PASS] radixark-dspark mod is targeted, valid, and idempotent"
