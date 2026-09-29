#!/bin/bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VLLM_PACKAGE_ROOT:-}" ]]; then
    VLLM_PACKAGE_ROOT=$(python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vLLM is not installed")
print(next(iter(spec.submodule_search_locations)))
PY
    )
fi

python3 "$MOD_DIR/patch.py" "$VLLM_PACKAGE_ROOT"
MANIFEST="$VLLM_PACKAGE_ROOT/_spark_memory_profile.json"
PROFILE_HOST_DIR=$(python3 - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["host_directory"])
PY
)
if [[ ! -s "$PROFILE_HOST_DIR/host.jsonl" ]]; then
    python3 "$MOD_DIR/probe.py" --manifest "$MANIFEST" --baseline
fi
nohup python3 "$MOD_DIR/probe.py" --manifest "$MANIFEST" \
    >> "$PROFILE_HOST_DIR/monitor.log" 2>&1 < /dev/null &
echo "[memory-profile] JSONL measurements and profile.yaml: $PROFILE_HOST_DIR"
echo "[memory-profile] Bind-mount the output directory to retain profiles after container removal."
