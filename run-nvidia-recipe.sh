#!/bin/bash
# Translate a NVIDIA Local AI Inference Recipe and run it with our recipe runner.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/run-nvidia-recipe.py" "$@"
