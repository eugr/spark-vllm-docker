#!/bin/bash
#
# run-stack.sh - Wrapper for run-stack.py
#
# Serve multiple vLLM recipes at once, each with its own placement. Ensures
# Python dependencies are available and runs the stack runner. See
# stacks/README.md for the manifest format.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/ensure-python.sh"

exec $PYTHON "$SCRIPT_DIR/run-stack.py" "$@"
