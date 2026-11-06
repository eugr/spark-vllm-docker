#!/bin/bash
#
# run-recipe.sh - Wrapper for run-recipe.py
#
# Ensures Python dependencies are available and runs the recipe runner.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_SCRIPT="$SCRIPT_DIR/run-recipe.py"

# Check for Python 3.10+
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: Python 3 not found. Please install Python 3.10 or later."
    exit 1
fi

# Verify version
PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    echo "Error: Python 3.10+ required, found $PY_VERSION"
    exit 1
fi

# Check for PyYAML. Keep wrapper-only dependencies in a virtual environment
# instead of modifying an externally managed Homebrew/system Python (PEP 668).
if ! "$PYTHON" -c "import yaml" 2>/dev/null; then
    VENV_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/spark-vllm-docker/run-recipe-venv-py${PY_MAJOR}.${PY_MINOR}"
    VENV_PYTHON="$VENV_DIR/bin/python"
    PYTHON_ID=$("$PYTHON" -c 'import os, sys; print(os.path.realpath(sys.executable))')
    VENV_PYTHON_ID=""

    if [[ -x "$VENV_PYTHON" ]]; then
        VENV_PYTHON_ID=$("$VENV_PYTHON" -c 'import os, sys; print(os.path.realpath(sys._base_executable))' 2>/dev/null || true)
    fi

    if [[ "$VENV_PYTHON_ID" != "$PYTHON_ID" ]]; then
        echo "Creating an isolated Python environment..."
        if ! "$PYTHON" -m venv --clear "$VENV_DIR"; then
            echo "Error: Failed to create $VENV_DIR." >&2
            exit 1
        fi
    fi

    if ! "$VENV_PYTHON" -c "import yaml" 2>/dev/null; then
        echo "Installing PyYAML in the isolated Python environment..."
        if ! "$VENV_PYTHON" -m pip install --quiet pyyaml; then
            echo "Error: Failed to install PyYAML in $VENV_DIR." >&2
            exit 1
        fi
    fi

    PYTHON="$VENV_PYTHON"
fi

# Run the recipe script
exec "$PYTHON" "$RECIPE_SCRIPT" "$@"
