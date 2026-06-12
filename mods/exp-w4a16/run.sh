#!/bin/bash

set -euo pipefail

PYTHON_ROOT="/usr/local/lib/python3.12/dist-packages"
VLLM_ROOT="$PYTHON_ROOT/vllm"
PRS="42124 42566 42546"

if [ ! -d "$VLLM_ROOT" ]; then
    echo "[w4a16-pr] vLLM package not found at $VLLM_ROOT"
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cd "$PYTHON_ROOT"

apply_pr() {
    local pr="$1"
    local diff="$TMP_DIR/pr-${pr}.diff"
    local check_log="$TMP_DIR/pr-${pr}.check.log"

    echo "[w4a16-pr] Checking PR #${pr}"
    curl -fsSL "https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/${pr}.diff" \
        -o "$diff"

    if git apply --reverse --check --exclude="tests/*" --exclude="examples/*" "$diff" 2>/dev/null; then
        echo "[w4a16-pr] PR #${pr} is already present in installed vLLM; skipping"
        return 0
    fi

    echo "[w4a16-pr] Applying PR #${pr}"
    if git apply --check --exclude="tests/*" --exclude="examples/*" "$diff" 2>"$check_log"; then
        git apply --exclude="tests/*" --exclude="examples/*" "$diff"
        echo "[w4a16-pr] PR #${pr} applied successfully"
        return 0
    fi

    echo "[w4a16-pr] PR #${pr} could not be applied"
    cat "$check_log"
    return 1
}

for pr in $PRS; do
    apply_pr "$pr"
done

echo "[w4a16-pr] All requested PRs handled"
