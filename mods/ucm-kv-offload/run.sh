#!/bin/bash
set -euo pipefail

# UCM (unified-cache-management) KV cache offload mod.
#
# Installs the pinned uc-manager wheel and writes the UCM connector config that
# the recipe's --kv-transfer-config points at (UCM_CONFIG_FILE). The serve
# flag and ENABLE_UCM_PATCH live in the recipe, not here: mods run in a
# separate `docker exec` and cannot set the serve process environment.
#
# The store is an uncompressed Cache|Posix pipeline on local storage. The
# default directory sits inside the host-mounted vLLM cache so offloaded KV
# survives container recreation without extra -v flags (assumptions.md).

PREFIX="[ucm-kv-offload]"
UCM_REQ="uc-manager[cu130]==0.7.0"
CONFIG_FILE="${UCM_CONFIG_FILE:-/workspace/ucm_config.yaml}"
KV_DIR="${UCM_KV_DIR:-/root/.cache/vllm/ucm-kv}"
KV_CAPACITY_GB="${UCM_KV_CAPACITY_GB:-10240}"

echo "=== UCM KV offload mod ==="

if ! [[ "$KV_CAPACITY_GB" =~ ^[0-9]+$ ]]; then
  echo "$PREFIX UCM_KV_CAPACITY_GB must be a positive integer, got '$KV_CAPACITY_GB'." >&2
  exit 1
fi

installed="$(python3 -c 'from importlib.metadata import version; print(version("uc-manager"))' 2>/dev/null || true)"
if [ "$installed" = "0.7.0" ]; then
  echo "$PREFIX uc-manager 0.7.0 already installed; skipping install."
else
  command -v uv >/dev/null 2>&1 || {
    echo "$PREFIX uv is required to install $UCM_REQ." >&2
    exit 1
  }
  echo "$PREFIX Installing $UCM_REQ (found: ${installed:-none})."
  uv pip install "$UCM_REQ"
fi

# Verifies the install and the single-backend guard in ucm/__init__.py.
python3 -c 'import ucm' || {
  echo "$PREFIX UCM import check failed after install." >&2
  exit 1
}

mkdir -p "$KV_DIR" "$(dirname "$CONFIG_FILE")"

cat > "$CONFIG_FILE" <<EOF
# Written by mods/ucm-kv-offload at container launch; do not edit.
# Local NVMe, uncompressed: Cache|Posix pipeline store (see the mod README).
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "$KV_DIR"
      posix_capacity_gb: $KV_CAPACITY_GB
      posix_io_engine: "psync"
EOF

echo "=====> UCM installed; config at $CONFIG_FILE; KV store at $KV_DIR"