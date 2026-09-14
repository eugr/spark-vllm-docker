#!/bin/bash
#
# run-recipe.sh - Wrapper for run-recipe.py
#
# Ensures Python dependencies are available and runs the recipe runner.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/ensure-python.sh"

exec $PYTHON "$SCRIPT_DIR/run-recipe.py" "$@"
