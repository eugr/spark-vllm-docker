#!/bin/bash
# Verify that remote command strings built by launch-cluster.sh survive the
# remote login shell intact. SSH hands the command to that shell, which may be
# dash rather than bash, so each command is replayed through dash, sh, and bash
# with fake ssh and docker executables. No real host, SSH, or Docker is used.
#
# Usage: tests/test_launch_cluster_remote_quoting.sh [path/to/launch-cluster.sh]
set -o pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LAUNCH_SCRIPT="${1:-$PROJECT_DIR/launch-cluster.sh}"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
FAKE_BIN="$TMP_DIR/bin"
LOG_ROOT="$TMP_DIR/logs"
mkdir -p "$FAKE_BIN" "$LOG_ROOT"
FAILURES=0

# Load only helper and builder functions so no configuration, discovery, or
# cluster action at the top level of launch-cluster.sh can run.
extract_function() {
    sed -n "/^$1() {\$/,/^}\$/p" "$LAUNCH_SCRIPT"
}
for fn in sh_quote sh_quote_words ensure_container_earlyoom \
    verify_cluster_image_consistency start_cluster dump_cluster_failure_logs \
    exec_no_ray_cluster; do
    source <(extract_function "$fn")
done

# Fake docker: record argv (NUL-separated) per node and invocation.
cat > "$FAKE_BIN/docker" <<'DOCKER'
#!/bin/bash
node="${TEST_NODE:-local}"
dir="$LOG_ROOT/$node"
mkdir -p "$dir"
index=$(find "$dir" -type f | wc -l)
printf '%s\0' "$@" > "$dir/$(printf '%04d' "$index")"
if [[ "$1 $2" == "image inspect" ]]; then
    echo "sha256:test-image-id"
fi
DOCKER

# Fake ssh: run the final argument as a command string in the shell under test,
# as sshd does with the remote user's login shell.
cat > "$FAKE_BIN/ssh" <<'SSH'
#!/bin/bash
TEST_NODE=remote exec "$REMOTE_SHELL" -c "${@: -1}"
SSH
chmod +x "$FAKE_BIN/docker" "$FAKE_BIN/ssh"
export PATH="$FAKE_BIN:$PATH" LOG_ROOT

TRICKY=(
    "two words"
    "it's quoted"
    $'line one\nline two'
    'dollar $HOME and $(id) and `id`'
    'back\slash \n literal'
    '*.glob ? [ab]'
    ''
    $'tab\there'
    '"double" quotes; & | > <'
)

reset_logs() {
    rm -rf "$LOG_ROOT"
    mkdir -p "$LOG_ROOT"
}

# Read the Nth docker invocation recorded on a node into the named array.
read_call() {
    local node="$1" index="$2" target="$3" file
    file="$LOG_ROOT/$node/$(printf '%04d' "$index")"
    if [[ ! -f "$file" ]]; then
        eval "$target=()"
        return 1
    fi
    mapfile -d '' -t "$target" < "$file"
}

# Read the first docker invocation on a node whose first argument matches.
find_call() {
    local node="$1" first="$2" target="$3" file
    local -n found_ref="$target"
    found_ref=()
    for file in "$LOG_ROOT/$node"/*; do
        [[ -f "$file" ]] || continue
        mapfile -d '' -t found_ref < "$file"
        [[ "${found_ref[0]:-}" == "$first" ]] && return 0
    done
    found_ref=()
    return 1
}

check_argv() {
    local label="$1"
    shift
    local -n actual_ref="$1"
    shift
    local expected=("$@")
    if [[ "${#actual_ref[@]}" -eq "${#expected[@]}" ]]; then
        local i same=true
        for i in "${!expected[@]}"; do
            [[ "${actual_ref[$i]}" == "${expected[$i]}" ]] || same=false
        done
        if [[ "$same" == "true" ]]; then
            echo "[PASS] $label"
            return 0
        fi
    fi
    echo "[FAIL] $label" >&2
    echo "  expected: $(declare -p expected)" >&2
    local actual=("${actual_ref[@]}")
    echo "  actual:   $(declare -p actual)" >&2
    FAILURES=$((FAILURES + 1))
}

# --- Test doubles for functions start_cluster and exec_no_ray_cluster call. ---
check_cluster_running() { CLUSTER_WAS_RUNNING=false; }
prepare_vllm_pr_mods() { :; }
get_env_flags() { echo "-e NODE_FLAG=1"; }
cleanup() { :; }
cleanup_generated_mods() { :; }
sleep() { :; }
_exec_on_head() { :; }
exec_head_with_workers() {
    local pid
    for pid in "${CLUSTER_EXEC_PIDS[@]}"; do
        wait "$pid"
    done
}

run_earlyoom_case() {
    local shell="$1" local_call remote_call
    reset_logs
    ENABLE_EARLYOOM=true
    CONTAINER_NAME=vllm_node
    IMAGE_NAME=vllm-node
    ensure_container_earlyoom head true >/dev/null 2>&1
    ensure_container_earlyoom worker false >/dev/null 2>&1
    read_call local 0 local_call
    read_call remote 0 remote_call
    if [[ "${#local_call[@]}" -lt 7 || "${local_call[*]:0:6}" != "exec --user 0 -w / vllm_node" ]]; then
        echo "[FAIL] $shell: earlyoom local reference call was not recorded" >&2
        FAILURES=$((FAILURES + 1))
        return
    fi
    check_argv "$shell: earlyoom setup script reaches worker docker exec intact" \
        remote_call "${local_call[@]}"
}

run_start_cluster_case() {
    local shell="$1" local_call remote_call tricky
    reset_logs
    ENABLE_EARLYOOM=true
    EARLYOOM_ARGS="-M 524288,102400 -s 100 -r 60"
    CONTAINER_NAME=vllm_node
    IMAGE_NAME=vllm-node
    PEER_NODES=(worker)
    HEAD_IP=head
    MOD_PATHS=()
    MOD_TYPES=()
    LAUNCH_SCRIPT_PATH=""
    SOLO_MODE=false
    NO_RAY_MODE=true
    MOUNT_CACHE_DIRS=false
    NON_PRIVILEGED_MODE=false
    KEEP_ENTRYPOINT=false
    PORT_MAPPINGS=()
    DOCKER_ARGS=""
    NOFILE_LIMIT=1024
    CLUSTER_WAS_RUNNING=false
    VLLM_PRS_REQUESTED=false
    CONTAINER_ENV_ARGS=()
    for tricky in "${TRICKY[@]}"; do
        CONTAINER_ENV_ARGS+=(-e "VALUE=$tricky")
    done
    ensure_container_earlyoom() { :; }
    start_cluster >/dev/null 2>&1
    unset -f ensure_container_earlyoom
    source <(extract_function ensure_container_earlyoom)

    # Head and worker get identical flags from the get_env_flags double, so the
    # worker must receive exactly the argv the head docker run received.
    find_call local run local_call
    find_call remote run remote_call
    if [[ "${local_call[0]:-}" != "run" ]]; then
        echo "[FAIL] $shell: head docker run reference call was not recorded" >&2
        FAILURES=$((FAILURES + 1))
        return
    fi
    check_argv "$shell: worker docker run keeps env values and earlyoom keepalive intact" \
        remote_call "${local_call[@]}"
}

run_no_ray_case() {
    local shell="$1" daemon="$2" remote_call payload
    reset_logs
    CONTAINER_NAME=vllm_node
    PEER_NODES=(worker)
    HEAD_IP=head
    MASTER_PORT=29501
    LAUNCH_SCRIPT_MODE=true
    DAEMON_MODE="$daemon"
    CLUSTER_EXEC_PIDS=()
    local base_cmd
    printf -v base_cmd '%s\n' "/workspace/run.sh" "${TRICKY[@]}"
    exec_no_ray_cluster "$base_cmd" >/dev/null 2>&1
    payload="$base_cmd >> /proc/1/fd/1 2>&1"
    read_call remote 0 remote_call
    if [[ "$daemon" == "true" ]]; then
        check_argv "$shell: daemon worker docker exec payload intact" \
            remote_call exec -d vllm_node bash -c "$payload"
    else
        check_argv "$shell: attached worker docker exec payload intact" \
            remote_call exec vllm_node bash -c "$payload"
    fi
}

run_inspect_and_logs_case() {
    local shell="$1" remote_call
    reset_logs
    PEER_NODES=(worker)
    HEAD_IP=head
    IMAGE_NAME="registry.example/vllm-node:it's"
    CONTAINER_NAME="vllm_node \$HOME"
    verify_cluster_image_consistency >/dev/null 2>&1
    read_call remote 0 remote_call
    check_argv "$shell: worker image inspect keeps image name intact" \
        remote_call image inspect --format '{{.Id}}' "$IMAGE_NAME"

    reset_logs
    dump_cluster_failure_logs >/dev/null 2>&1
    read_call remote 0 remote_call
    check_argv "$shell: worker docker logs keeps container name intact" \
        remote_call logs --timestamps --tail 200 "$CONTAINER_NAME"
}

SHELLS=()
for candidate in dash sh bash; do
    if command -v "$candidate" >/dev/null 2>&1; then
        SHELLS+=("$candidate")
    else
        echo "[SKIP] $candidate is not installed"
    fi
done

for REMOTE_SHELL in "${SHELLS[@]}"; do
    export REMOTE_SHELL
    run_earlyoom_case "$REMOTE_SHELL"
    run_start_cluster_case "$REMOTE_SHELL"
    run_no_ray_case "$REMOTE_SHELL" true
    run_no_ray_case "$REMOTE_SHELL" false
    run_inspect_and_logs_case "$REMOTE_SHELL"
done

# The helper itself must round-trip every value through each shell.
if declare -F sh_quote_words >/dev/null; then
    for REMOTE_SHELL in "${SHELLS[@]}"; do
        quoted=$(sh_quote_words "${TRICKY[@]}")
        mapfile -d '' -t round_trip < <("$REMOTE_SHELL" -c "printf '%s\\0' $quoted")
        check_argv "$REMOTE_SHELL: sh_quote_words round-trips tricky values" \
            round_trip "${TRICKY[@]}"
    done
else
    echo "[FAIL] sh_quote_words helper is missing" >&2
    FAILURES=$((FAILURES + 1))
fi

if [[ "$FAILURES" -gt 0 ]]; then
    echo "$FAILURES remote quoting check(s) failed." >&2
    exit 1
fi
echo "All remote quoting checks passed."
