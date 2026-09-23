#!/bin/bash
# Exercise the real copy function and rsync on tiny local cache fixtures.
# Only the SSH destination is redirected; no downloads or remote hosts are used.
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RSYNC_BIN="${RSYNC_BIN:-$(command -v rsync)}"
if [[ "$("$RSYNC_BIN" --help)" != *--mkpath* ]]; then
    echo 'This test requires rsync with --mkpath support (3.2.3+).' >&2
    exit 1
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
SSH_USER=testuser

# Load only the function so configuration, discovery, and downloads cannot run.
sed -n '/^copy_model_to_host() {$/,/^}$/p' "$PROJECT_DIR/hf-download.sh" > "$TMP_DIR/copy-function.sh"
source "$TMP_DIR/copy-function.sh"

rsync() {
    local args=("$@")
    local last=$((${#args[@]} - 1))
    local destination="${args[$last]}"
    [[ "$destination" == "$SSH_USER@peer:"* ]] || return 1
    args[$last]="$PEER_ROOT${destination#*:}"
    "$RSYNC_BIN" "${args[@]}"
}

fail() {
    echo "[FAIL] $*" >&2
    cat "$TMP_DIR/copy.log" >&2
    exit 1
}

assert_copy() {
    local copied_model="$PEER_ROOT$HUB_PATH/$(basename "$model_dir")"
    local blob revision snapshot
    for blob in a b; do
        [[ -f "$copied_model/blobs/blob-$blob" && ! -L "$copied_model/blobs/blob-$blob" ]] \
            || fail "$layout: blob $blob was not materialized"
        for revision in first second; do
            snapshot="$copied_model/snapshots/$revision/model-$blob.safetensors"
            [[ -L "$snapshot" ]] || fail "$layout: snapshot link was dereferenced"
            cmp "$model_dir/blobs/blob-$blob" "$snapshot" \
                || fail "$layout: snapshot has missing or incorrect weights"
        done
    done
    [[ ! -e "$PEER_ROOT$HUB_PATH/blobs" ]] || fail "$layout: copied a shared blob store"
}

for layout in legacy shared mixed; do
    # Include spaces to exercise argument quoting.
    HUB_PATH="$TMP_DIR/$layout/source cache/hub"
    PEER_ROOT="$TMP_DIR/$layout/peer"
    model_dir="$HUB_PATH/models--org--model"
    mkdir -p "$model_dir/blobs" "$model_dir/refs"
    printf 'second\n' > "$model_dir/refs/main"
    for blob in a b; do
        if [[ "$layout" == shared || ( "$layout" == mixed && "$blob" == a ) ]]; then
            hash="${blob}$(printf '%063d' 0)"
            mkdir -p "$HUB_PATH/blobs/${hash:0:2}"
            printf 'weights-%s\n' "$blob" > "$HUB_PATH/blobs/${hash:0:2}/$hash"
            ln -s "../../blobs/${hash:0:2}/$hash" "$model_dir/blobs/blob-$blob"
        else
            printf 'weights-%s\n' "$blob" > "$model_dir/blobs/blob-$blob"
        fi
        for revision in first second; do
            mkdir -p "$model_dir/snapshots/$revision"
            ln -s "../../blobs/blob-$blob" "$model_dir/snapshots/$revision/model-$blob.safetensors"
        done
    done

    copy_model_to_host peer org/model "$model_dir" > "$TMP_DIR/copy.log" 2>&1 \
        || fail "$layout: initial copy failed"
    assert_copy
    copy_model_to_host peer org/model "$model_dir" > "$TMP_DIR/copy.log" 2>&1 \
        || fail "$layout: repeated copy failed"
    assert_copy
    echo "[PASS] $layout cache: repeated copies preserve snapshots and materialize weights once"
done

# A missing external payload must fail instead of reporting a successful copy.
rm "$HUB_PATH/blobs/a0/a$(printf '%063d' 0)"
if copy_model_to_host peer org/model "$model_dir" > "$TMP_DIR/copy.log" 2>&1; then
    fail 'copy succeeded with a missing shared blob'
fi
echo '[PASS] missing shared blob fails the copy'
