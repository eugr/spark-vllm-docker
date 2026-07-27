#!/bin/bash
#
# ensure-python.sh - Shared bootstrap sourced by run-recipe.sh and run-stack.sh.
#
# Locates Python 3.10+ (sets $PYTHON) and installs PyYAML if missing.
# Exits the sourcing script with an error message if either is unavailable.
#

if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: Python 3 not found. Please install Python 3.10 or later."
    exit 1
fi

if ! $PYTHON -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo "Error: Python 3.10+ required, found $($PYTHON -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    exit 1
fi

if ! $PYTHON -c "import yaml" 2>/dev/null; then
    echo "Installing PyYAML..."
    if ! $PYTHON -m pip install --quiet pyyaml; then
        echo "Error: Failed to install PyYAML. Try: pip install pyyaml"
        exit 1
    fi
fi
