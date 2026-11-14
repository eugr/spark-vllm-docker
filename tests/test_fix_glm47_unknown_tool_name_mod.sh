#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/fix-glm47-unknown-tool-name"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

TARGET="$TMP_DIR/vllm/parser/glm47_moe.py"
mkdir -p "$(dirname "$TARGET")"

# Fixture shaped like the shipped parser config block (the flag line is what
# the mod must flip; context around it varies across vLLM builds).
printf '%s\n' \
    '        arg_converter=_glm47_arg_converter,' \
    '        stream_arg_deltas=True,' \
    '        tool_args_json=False,' \
    '        validate_tool_names=True,' \
    '    )' \
    > "$TARGET"

(cd "$MOD" && VLLM_PACKAGE_ROOT="$TMP_DIR/vllm" ./run.sh) >/dev/null

grep -Fq 'validate_tool_names=False' "$TARGET"
if grep -q 'validate_tool_names=True' "$TARGET"; then
    echo "FAIL: validate_tool_names=True still present after patch" >&2
    exit 1
fi

# Idempotent re-run must be a no-op.
(cd "$MOD" && VLLM_PACKAGE_ROOT="$TMP_DIR/vllm" ./run.sh) >/dev/null
grep -c 'validate_tool_names=False' "$TARGET" | grep -qx '1'

echo "test_fix_glm47_unknown_tool_name_mod: OK"
