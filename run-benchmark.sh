#!/bin/bash
#
# run-benchmark.sh - Wrapper for run_benchmark.py
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="$SCRIPT_DIR/run_benchmark.py"

# Check for Python 3.10+
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: Python 3 not found. Please install Python 3.10 or later."
    exit 1
fi

PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')
if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    echo "Error: Python 3.10+ required"
    exit 1
fi

exec $PYTHON "$BENCHMARK_SCRIPT" "$@"
