#!/bin/bash
# Run on the head Spark as the login user, not through sudo.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/setup-cluster.py" "$@"
