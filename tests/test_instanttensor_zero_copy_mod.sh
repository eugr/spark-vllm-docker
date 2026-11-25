#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/instanttensor-zero-copy/run.sh"
PATCHER="$PROJECT_DIR/mods/instanttensor-zero-copy/patch_weight_utils.py"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

VLLM_ROOT="$TMP_DIR/site-packages/vllm"
TARGET="$VLLM_ROOT/model_executor/model_loader/weight_utils.py"
mkdir -p "$(dirname "$TARGET")"

printf '%s\n' \
    'def unrelated_upstream_helper():' \
    '    return "source drift outside the iterator"' \
    '' \
    'def instanttensor_weights_iterator(hf_weights_files, use_tqdm_on_load):' \
    '    import instanttensor' \
    '    device = 0' \
    '    process_group = None' \
    '    # Upstream may revise this comment without changing loader semantics.' \
    '    with instanttensor.safe_open(' \
    '        hf_weights_files,' \
    '        device=device,' \
    '        framework="pt",' \
    '        process_group=process_group,' \
    '        copy=True,  # Upstream may also add an inline explanation.' \
    '    ) as f:' \
    '        for name, tensor in f.tensors():' \
    '            yield name, tensor' \
    > "$TARGET"

first_output=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$MOD")
grep -Fq 'Patched' <<< "$first_output"
grep -Fq '# spark-vllm mod: instanttensor-zero-copy v1' "$TARGET"
grep -Fq 'copy=False' "$TARGET"

python3 - "$TARGET" <<'PY'
import ast
import sys
from pathlib import Path

tree = ast.parse(Path(sys.argv[1]).read_text())
functions = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef)
    and node.name == "instanttensor_weights_iterator"
]
assert len(functions) == 1
calls = [
    node
    for node in ast.walk(functions[0])
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "safe_open"
]
assert len(calls) == 1
copy_keywords = [kw for kw in calls[0].keywords if kw.arg == "copy"]
assert len(copy_keywords) == 1
assert isinstance(copy_keywords[0].value, ast.Constant)
assert copy_keywords[0].value.value is False
PY

unsupported_root="$TMP_DIR/unsupported/vllm"
unsupported_target="$unsupported_root/model_executor/model_loader/weight_utils.py"
mkdir -p "$(dirname "$unsupported_target")"
printf '%s\n' \
    'def instanttensor_weights_iterator(files, progress):' \
    '    import instanttensor' \
    '    preserve_ownership = True' \
    '    with instanttensor.safe_open(' \
    '        files,' \
    '        framework="pt",' \
    '        copy=preserve_ownership,' \
    '    ) as f:' \
    '        yield from f.tensors()' \
    > "$unsupported_target"

unsupported_before=$(sha256sum "$unsupported_target")
if python3 "$PATCHER" "$unsupported_target" >/dev/null 2>&1; then
    echo '[FAIL] patcher accepted changed InstantTensor copy semantics' >&2
    exit 1
fi
unsupported_after=$(sha256sum "$unsupported_target")
test "$unsupported_before" = "$unsupported_after"

echo '[PASS] instanttensor-zero-copy mod tolerates source drift and fails closed'
