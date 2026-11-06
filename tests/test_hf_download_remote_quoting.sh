#!/bin/bash
# Remote cache commands must survive any POSIX login shell on the peer.
# SSH passes the command string to the remote user's login shell, which is
# often dash (/bin/sh) rather than bash. Run the generated commands through
# real local shells; no SSH, Docker, or downloads are used.
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

failures=0
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*" >&2; failures=$((failures + 1)); }

# Load only the functions under test so configuration and discovery cannot run.
extract() {
    sed -n "/^$1() {\$/,/^}\$/p" "$PROJECT_DIR/hf-download.sh"
}
for fn in sh_quote remote_bash_command prepare_remote_cache remote_cache_action; do
    [[ -n "$(extract "$fn")" ]] || { echo "[FAIL] hf-download.sh lacks $fn()" >&2; exit 1; }
    source <(extract "$fn")
done

SHELLS=()
for candidate in dash sh bash; do
    command -v "$candidate" >/dev/null 2>&1 && SHELLS+=("$(command -v "$candidate")")
done
[[ " ${SHELLS[*]} " == *dash* ]] || echo "[WARN] dash not installed; covering sh and bash only" >&2

# Arguments that previously forced bash's $'...' or backslash quoting.
ARGS=(
    "/home/user/.cache/huggingface"
    "/path with spaces/hub"
    "it's quoted"
    $'two\nlines'
    'dollar $HOME and `backticks`'
    'back\slash'
    '*'
    ''
)
expected_args() {
    local arg
    for arg in "${ARGS[@]}"; do
        printf '<%s>\n' "$arg"
    done
}

# A multi-line script, like the declare -f output sent to the peer.
SCRIPT=$'first() { printf "<%s>\\n" "$@"; }\nfirst "$@"'

expected=$(expected_args)
command_string=$(remote_bash_command "$SCRIPT" "${ARGS[@]}")
for shell in "${SHELLS[@]}"; do
    actual=$(cd "$TMP_DIR" && "$shell" -c "$command_string" 2>&1) || true
    if [[ "$actual" == "$expected" ]]; then
        pass "remote_bash_command arguments survive $shell"
    else
        fail "remote_bash_command under $shell: got [$actual]"
    fi
done

# Exercise the real callers with SSH replaced by each login shell.
HF_CACHE_DIR="/tmp/hf home's"
HUB_PATH=$'/tmp/hf\nhub'
SSH_USER=testuser
repair_cache_ownership() {
    printf 'repair:<%s>\n' "$@"
}
ensure_uvx() {
    :
}
CACHE_HELPER="$TMP_DIR/helper.py"
cat > "$CACHE_HELPER" <<'PY'
import sys
for arg in sys.argv[1:]:
    print(f"helper:<{arg}>")
PY

for shell in "${SHELLS[@]}"; do
    LOGIN_SHELL="$shell"
    cluster_copy_ssh() {
        # Mimic sshd: options are consumed by ssh, the last word goes to the login shell.
        "$LOGIN_SHELL" -c "${@: -1}"
    }

    expected=$(printf 'repair:<%s>\n' "$HF_CACHE_DIR" "$HUB_PATH")
    # Skip the "Checking cache ownership" banner; keep repair output (may span lines).
    actual=$(prepare_remote_cache peer </dev/null 2>&1 | sed -n '/^repair:/,$p') || true
    if [[ "$actual" == "$expected" ]]; then
        pass "prepare_remote_cache runs under $shell"
    else
        fail "prepare_remote_cache under $shell: got [$actual]"
    fi

    expected=$(printf 'helper:<%s>\n' inventory --cache-dir "$HUB_PATH" --model "org/it's model")
    actual=$(remote_cache_action peer inventory --cache-dir "$HUB_PATH" --model "org/it's model" 2>&1) || true
    if [[ "$actual" == "$expected" ]]; then
        pass "remote_cache_action runs under $shell"
    else
        fail "remote_cache_action under $shell: got [$actual]"
    fi
done

if [[ "$failures" -ne 0 ]]; then
    echo "$failures check(s) failed." >&2
    exit 1
fi
echo "All remote quoting checks passed."
