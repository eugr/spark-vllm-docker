#!/usr/bin/env bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$MOD_DIR/patch_vllm_minimax_m27.py"
python3 "$MOD_DIR/patch_nvfp4_scalefix.py"
