#!/usr/bin/env bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$MOD_DIR/patch_vllm_reap_gb10.py"
