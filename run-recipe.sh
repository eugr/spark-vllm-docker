#!/usr/bin/env bash
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

# Check for PyYAML and install if missing
if ! $PYTHON -c "import yaml" 2>/dev/null; then
    echo "Installing PyYAML..."
    $PYTHON -m pip install --quiet pyyaml
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to install PyYAML. Try: pip install pyyaml"
        exit 1
    fi
fi

# Run the recipe script
exec $PYTHON "$RECIPE_SCRIPT" "$@"
