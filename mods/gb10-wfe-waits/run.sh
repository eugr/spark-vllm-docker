#!/bin/bash
# gb10-wfe-waits: park busy-waiting host threads in WFET instead of spinning on
# aarch64 CPUs with FEAT_WFxT (DGX Spark / GB10). See README.md in this directory.
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "[gb10-wfe-waits] vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[gb10-wfe-waits] python3 is required to apply this mod." >&2
  exit 1
fi

exec python3 "$MOD_DIR/apply.py" "$PYTHON_ROOT" "$MOD_DIR"
