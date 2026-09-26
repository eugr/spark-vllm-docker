#!/bin/bash
set -euo pipefail

# Focused check for mods/ucm-kv-offload/run.sh: install-skip logic and the
# generated UCM config, using a stubbed site-packages and a PATH-stubbed uv.

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/ucm-kv-offload/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Fake site-packages: stub ucm package plus pinned uc-manager metadata, so the
# mod's version check and `import ucm` succeed without a real install.
SITE="$TMP_DIR/site"
mkdir -p "$SITE/ucm" "$SITE/uc_manager-0.7.0.dist-info"
: > "$SITE/ucm/__init__.py"
printf 'Metadata-Version: 2.1\nName: uc-manager\nVersion: 0.7.0\n' \
    > "$SITE/uc_manager-0.7.0.dist-info/METADATA"

# uv stub: record install attempts instead of hitting the network.
UV_LOG="$TMP_DIR/uv.log"
mkdir -p "$TMP_DIR/bin"
cat > "$TMP_DIR/bin/uv" <<EOF
#!/bin/bash
echo "\$@" >> "$UV_LOG"
EOF
chmod +x "$TMP_DIR/bin/uv"

CONFIG="$TMP_DIR/ucm_config.yaml"
KV="$TMP_DIR/kv"

run_mod() {
    PYTHONPATH="$SITE" PATH="$TMP_DIR/bin:$PATH" \
        UCM_CONFIG_FILE="$CONFIG" UCM_KV_DIR="$KV" UCM_KV_CAPACITY_GB="$1" \
        bash "$MOD"
}

fail() {
    echo "FAIL: $1" >&2
    exit 1
}

# 1. Pinned version present: no install, config written with defaults.
run_mod 10240
[ ! -e "$UV_LOG" ] || fail "uv invoked despite pinned install being present"
grep -Fq 'store_pipeline: "Cache|Posix"' "$CONFIG" || fail "store_pipeline missing"
grep -Fq "storage_backends: \"$KV\"" "$CONFIG" || fail "storage_backends not wired to UCM_KV_DIR"
grep -Fq 'posix_capacity_gb: 10240' "$CONFIG" || fail "capacity missing"
grep -Fq 'posix_io_engine: "psync"' "$CONFIG" || fail "io engine missing"
[ -d "$KV" ] || fail "store directory not created"

# 2. Missing install: uv called with the pinned requirement; capacity override.
rm -rf "$SITE/uc_manager-0.7.0.dist-info"
run_mod 42
grep -Fq 'uc-manager[cu130]==0.7.0' "$UV_LOG" || fail "pinned requirement not passed to uv"
grep -Fq 'posix_capacity_gb: 42' "$CONFIG" || fail "capacity override ignored"

# 3. Invalid capacity rejected before touching anything.
if run_mod notanumber 2>/dev/null; then
    fail "invalid UCM_KV_CAPACITY_GB accepted"
fi

echo "PASS: ucm-kv-offload mod"