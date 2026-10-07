#!/bin/bash
set -e

# Default values
COPY_HOSTS=()
SSH_USER="$USER"
PARALLEL_COPY=false
CONFIG_FILE=""
CONFIG_FILE_SET=false
ACTION=download
DELETE_REQUESTED=false
FORCE=false
BACKUP_DIR=""
OUTPUT_FORMAT=console
SORT_ORDER=size:desc
LIST_OPTIONS=false
MODEL_NAME=""
REVISION=""
CACHE_HELPER="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/hf-cache.py"

# Help function
usage() {
    echo "Usage: $0 [OPTIONS] [<model-name>]"
    echo "  <model-name>                : HuggingFace model name (e.g., 'QuantTrio/MiniMax-M2-AWQ')"
    echo "  --list                     : List cached model revisions (including remote-only models with -c)."
    echo "  --delete <model-name>      : Delete all cached revisions; also valid after --backup."
    echo "  --backup <model-name>      : Back up all local revisions to --backup-dir."
    echo "  --list-backup [model-name] : List all backups, optionally selecting one model."
    echo "  --restore <model-name>     : Restore from --backup-dir; distribute with -c."
    echo "  --cleanup                  : Remove unreferenced model revisions."
    echo "  --backup-dir <path>        : Existing backup directory on the head node."
    echo "  --force                    : Skip deletion/cleanup confirmation."
    echo "  --revision <revision>      : Back up/delete one cached commit (hash, unique prefix, branch, or tag)."
    echo "  --format console|csv|json|md : Listing output format (default: console)."
    echo "  --sort <order>             : hf cache ls sort order (default: size:desc)."
    echo "  -c, --copy-to [hosts]       : Include peers. Omit hosts to use COPY_HOSTS from .env or autodiscovery."
    echo "      --copy-to-host          : Alias for --copy-to (backwards compatibility)."
    echo "      --copy-parallel         : With -c, copy to all resolved hosts concurrently."
    echo "  -u, --user <user>           : Username for ssh commands (default: \$USER)"
    echo "  --config <file>             : Path to .env configuration file (default: .env in script directory)"
    echo "  -h, --help                  : Show this help message"
    echo "Cache ownership is repaired using an existing vllm-node or vllm-node-b12x image, then sudo if needed."
    echo "Sudo may prompt on each node before downloads or copies."
    exit "${1:-0}"
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || fail "$1 requires a value."
}

set_action() {
    [[ "$ACTION" == download || "$ACTION" == "$1" ]] || fail "Conflicting actions: $ACTION and $1"
    ACTION="$1"
}

ensure_uvx() {
    local bin_dir installer status=0
    command -v uvx >/dev/null 2>&1 && return 0

    # Noninteractive SSH often omits the user-local install from PATH.
    for bin_dir in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        if [[ -x "$bin_dir/uvx" && ! -d "$bin_dir/uvx" ]]; then
            export PATH="$bin_dir:$PATH"
            return 0
        fi
    done

    bin_dir="$HOME/.local/bin"
    echo "uvx is missing on $(hostname); installing uv for the current user in $bin_dir..." >&2
    installer=$(mktemp) || return 1
    # Download fully before executing; do not consume the helper on SSH stdin.
    if command -v curl >/dev/null 2>&1; then
        curl --fail --silent --show-error --location --connect-timeout 15 --max-time 120 \
            https://astral.sh/uv/install.sh --output "$installer" || status=$?
    elif command -v wget >/dev/null 2>&1; then
        wget --quiet --timeout=30 --tries=2 --output-document="$installer" \
            https://astral.sh/uv/install.sh || status=$?
    else
        echo "Error: Installing uv requires curl or wget on this node." >&2
        status=1
    fi
    if [[ "$status" -eq 0 ]]; then
        UV_INSTALL_DIR="$bin_dir" UV_NO_MODIFY_PATH=1 sh "$installer" </dev/null >&2 || status=$?
    fi
    rm -f -- "$installer"
    if [[ "$status" -ne 0 || ! -x "$bin_dir/uvx" || -d "$bin_dir/uvx" ]]; then
        echo "Error: Could not install uvx on $(hostname). Install uv for the SSH user (https://docs.astral.sh/uv/getting-started/installation/) and retry." >&2
        return 1
    fi
    export PATH="$bin_dir:$PATH"
}

resolve_cache_path() {
    # Match huggingface_hub's expansion without evaluating shell code from paths.
    python3 - "$1" <<'PY'
import os
import sys

def expand(path):
    return os.path.expandvars(os.path.expanduser(path))

cache_home = expand(os.environ.get(
    "HF_HOME", os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "huggingface")
))
path = cache_home if sys.argv[1] == "home" else expand(os.environ.get(
    "HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(cache_home, "hub"))
))
if not path:
    sys.exit("Error: Hugging Face cache paths must not be empty.")
print(os.path.abspath(path))
PY
}

repair_cache_ownership() {
    local cache_dir checked_dir mismatch owner owner_uid image mount_source docker_repaired
    local checked_dirs=() sudo_args=()
    # Repair this host's files even if the caller has selected a remote Docker context.
    local docker_cmd=(docker --host unix:///var/run/docker.sock)
    owner=$(id -un) || return 1
    owner_uid=$(id -u) || return 1
    # A noninteractive run may use passwordless sudo, but must never wait for input.
    [[ -t 0 ]] || sudo_args=(-n)

    for cache_dir in "$@"; do
        cache_dir=$(realpath -m -- "$cache_dir") || return 1
        # Never recursively change a whole home directory or one of its parents.
        if [[ "$cache_dir" == / || "$HOME/" == "$cache_dir/"* ]]; then
            echo "Error: Refusing ownership repair for unsafe cache path: $cache_dir" >&2
            return 1
        fi
        # The Hub cache normally lives inside HF_HOME; avoid scanning it twice.
        for checked_dir in "${checked_dirs[@]}"; do
            if [[ "$cache_dir/" == "$checked_dir/"* ]]; then
                continue 2
            fi
        done
        [[ -e "$cache_dir" ]] || continue

        if mismatch=$(find "$cache_dir" ! -uid "$owner_uid" -print -quit); then
            if [[ -z "$mismatch" ]]; then
                checked_dirs+=("$cache_dir")
                continue
            fi
        fi

        echo "Repairing Hugging Face cache ownership for $owner: $cache_dir" >&2
        docker_repaired=false
        if command -v docker >/dev/null 2>&1; then
            # Docker's --mount value is CSV; quote the source field for commas/quotes.
            mount_source=${cache_dir//\"/\"\"}
            for image in vllm-node vllm-node-b12x; do
                "${docker_cmd[@]}" image inspect "$image" >/dev/null 2>&1 || continue
                echo "Trying cache ownership repair with Docker image $image..." >&2
                if "${docker_cmd[@]}" run --rm --pull=never --network none --user 0 \
                        --mount "type=bind,\"src=$mount_source\",dst=/hf-cache" \
                        --entrypoint chown "$image" -R -h -- "$owner_uid" /hf-cache \
                        && mismatch=$(find "$cache_dir" ! -uid "$owner_uid" -print -quit) \
                        && [[ -z "$mismatch" ]]; then
                    docker_repaired=true
                    break
                fi
            done
        fi
        if [[ "$docker_repaired" == true ]]; then
            checked_dirs+=("$cache_dir")
            continue
        fi

        echo "Docker repair unavailable or unsuccessful; trying sudo..." >&2
        # Do not follow cache symlinks to files outside this tree.
        if ! sudo "${sudo_args[@]}" chown -R -h -- "$owner" "$cache_dir"; then
            echo "Error: Could not repair cache ownership: $cache_dir" >&2
            echo "Rerun from an interactive terminal to enter a sudo password, or repair this cache manually." >&2
            return 1
        fi
        if ! mismatch=$(find "$cache_dir" ! -uid "$owner_uid" -print -quit) || [[ -n "$mismatch" ]]; then
            echo "Error: Cache ownership is still incorrect or unreadable: $cache_dir" >&2
            return 1
        fi
        checked_dirs+=("$cache_dir")
    done
}

prepare_remote_cache() {
    local host="$1" remote_script remote_command
    local ssh_args=(-o BatchMode=yes)
    if [[ -t 0 ]]; then
        ssh_args+=(-t)
    else
        ssh_args+=(-nT)
    fi
    # Pass the script as an argument, leaving stdin available for sudo's password.
    printf -v remote_script '%s\nrepair_cache_ownership "$@"' "$(declare -f repair_cache_ownership)"
    printf -v remote_command 'bash -c %q -- %q %q' "$remote_script" "$HF_CACHE_DIR" "$HUB_PATH"
    echo "Checking cache ownership on ${SSH_USER}@${host}..."
    if ! cluster_copy_ssh "$host" "${ssh_args[@]}" "$remote_command"; then
        echo "Error: Cache preparation failed on ${SSH_USER}@${host}." >&2
        return 1
    fi
}

add_copy_hosts() {
    local token part
    for token in "$@"; do
        IFS=',' read -ra PARTS <<< "$token"
        for part in "${PARTS[@]}"; do
            part="${part//[[:space:]]/}"
            if [ -n "$part" ]; then
                COPY_HOSTS+=("$part")
            fi
        done
    done
}

copy_model_to_host() {
    local host="$1"
    local model_name="$2"
    local model_dir="$3"

    echo "Copying model '$model_name' to ${SSH_USER}@${host}..."
    local host_copy_start host_copy_end host_copy_time
    host_copy_start=$(date +%s)

    # The trailing slash makes the model directory the transfer root: preserve
    # snapshot links, but materialize links to hub-level blobs as repo-local files.
    # Downside is duplication of storage if cross-model shared blobs are used, but basically replicates old behavior.
    # -s protects remote paths containing spaces or shell metacharacters.
    if cluster_copy_rsync "$host" "$model_dir/" "$HUB_PATH/$(basename "$model_dir")/"; then
        host_copy_end=$(date +%s)
        host_copy_time=$((host_copy_end - host_copy_start))
        printf "Copy to %s completed in %02d:%02d:%02d\n" "$host" $((host_copy_time/3600)) $((host_copy_time%3600/60)) $((host_copy_time%60))
    else
        echo "Copy to $host failed."
        return 1
    fi
}

remote_cache_action() {
    local host="$1" remote_command remote_script arg
    shift
    # Prepare uvx in the same shell that starts Python so PATH changes survive.
    # The Python helper still arrives on stdin; no repo checkout is needed.
    printf -v remote_script '%s\nensure_uvx || exit 1\nexec python3 - "$@"' "$(declare -f ensure_uvx)"
    printf -v remote_command 'bash -c %q --' "$remote_script"
    for arg in "$@"; do
        printf -v remote_command '%s %q' "$remote_command" "$arg"
    done
    cluster_copy_ssh "$host" -o BatchMode=yes -T "$remote_command" < "$CACHE_HELPER"
}

collect_inventory() {
    local host index=0
    if [[ -z "${INVENTORY_DIR:-}" ]]; then
        INVENTORY_DIR=$(mktemp -d)
    fi
    trap 'rm -rf -- "$INVENTORY_DIR"' EXIT
    INVENTORY_FILES=("$INVENTORY_DIR/0.json")
    python3 "$CACHE_HELPER" inventory --cache-dir "$HUB_PATH" --sort "$SORT_ORDER" \
        --model "$MODEL_NAME" --node "$(hostname) (local)" > "${INVENTORY_FILES[0]}"
    for host in "${COPY_HOSTS[@]}"; do
        index=$((index + 1))
        INVENTORY_FILES+=("$INVENTORY_DIR/$index.json")
        if ! remote_cache_action "$host" inventory --cache-dir "$HUB_PATH" --sort "$SORT_ORDER" \
                --model "$MODEL_NAME" --node "$host" > "$INVENTORY_DIR/$index.json"; then
            fail "Could not inventory ${SSH_USER}@${host}; no models were deleted."
        fi
    done
}

render_inventory() {
    local options=()
    [[ "$COPY_TO_FLAG" == false ]] || options+=(--nodes)
    python3 "$CACHE_HELPER" render --format "$OUTPUT_FORMAT" --sort "$SORT_ORDER" \
        --model "$MODEL_NAME" "${options[@]}" -- "${INVENTORY_FILES[@]}"
}

delete_models() {
    local answer host failed=0
    local options=(--model "$MODEL_NAME")
    if [[ "$ACTION" == cleanup ]]; then
        options=(--cleanup)
        echo "Remove unreferenced model revisions on the local node${COPY_HOSTS[*]:+ and ${COPY_HOSTS[*]}}." >&2
    elif [[ -n "$REVISION" ]]; then
        options+=(--revision "$REVISION")
        python3 "$CACHE_HELPER" check-revision --model "$MODEL_NAME" --revision "$REVISION" -- "${INVENTORY_FILES[@]}"
        echo "Delete revision '$REVISION' of '$MODEL_NAME' on the local node${COPY_HOSTS[*]:+ and ${COPY_HOSTS[*]}}." >&2
    else
        echo "Delete all revisions of '$MODEL_NAME' on the local node${COPY_HOSTS[*]:+ and ${COPY_HOSTS[*]}}." >&2
    fi
    if [[ "$FORCE" != true ]]; then
        printf 'Proceed? [y/N] ' >&2
        if ! read -r answer || [[ "$answer" != [yY] && "$answer" != [yY][eE][sS] ]]; then
            echo "Cancelled; no models were deleted." >&2
            return 0
        fi
    fi
    repair_cache_ownership "$HF_CACHE_DIR" "$HUB_PATH"
    for host in "${COPY_HOSTS[@]}"; do
        prepare_remote_cache "$host" >&2
    done
    if [[ "$ACTION" == backup || -n "$REVISION" ]]; then
        # Ownership repair can make previously unreadable revisions visible.
        # Recheck every node after repair and the user's confirmation delay.
        collect_inventory
    fi
    if [[ -n "$REVISION" ]]; then
        python3 "$CACHE_HELPER" check-revision --model "$MODEL_NAME" --revision "$REVISION" -- "${INVENTORY_FILES[@]}"
    fi
    if [[ "$ACTION" == backup ]]; then
        python3 "$CACHE_HELPER" covered --backup-dir "$BACKUP_DIR" --model "$MODEL_NAME" \
            --revision "$REVISION" -- "${INVENTORY_FILES[@]}"
    fi
    python3 "$CACHE_HELPER" delete --cache-dir "$HUB_PATH" "${options[@]}" || failed=1
    for host in "${COPY_HOSTS[@]}"; do
        if ! remote_cache_action "$host" delete --cache-dir "$HUB_PATH" "${options[@]}"; then
            echo "Error: Cache deletion failed on ${SSH_USER}@${host}." >&2
            failed=1
        fi
    done
    return "$failed"
}

# Argument parsing
COPY_TO_FLAG=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--copy-to|--copy-to-host|--copy-to-hosts)
            COPY_TO_FLAG=true
            shift
            # Consume arguments until the next flag or end of args
            while [[ "$#" -gt 0 && "$1" != -* ]]; do
                # Model IDs containing an organization are unambiguous, including
                # the documented '-c org/model' and '-c host org/model' forms.
                if [[ "$1" == */* ]]; then
                    [[ -z "$MODEL_NAME" ]] || fail "Multiple model names supplied."
                    MODEL_NAME="$1"
                    shift
                    break
                fi
                add_copy_hosts "$1"
                shift
            done
            continue
            ;;
        --copy-parallel) PARALLEL_COPY=true ;;
        --list) set_action list ;;
        --backup) set_action backup ;;
        --restore) set_action restore ;;
        --list-backup) set_action list-backup ;;
        --cleanup) set_action cleanup ;;
        --delete) DELETE_REQUESTED=true ;;
        --force) FORCE=true ;;
        --revision) require_value "$@"; REVISION="$2"; shift ;;
        --backup-dir) require_value "$@"; BACKUP_DIR="$2"; shift ;;
        --format) require_value "$@"; OUTPUT_FORMAT="$2"; LIST_OPTIONS=true; shift ;;
        --sort) require_value "$@"; SORT_ORDER="$2"; LIST_OPTIONS=true; shift ;;
        -u|--user) require_value "$@"; SSH_USER="$2"; shift ;;
        --config) require_value "$@"; CONFIG_FILE="$2"; CONFIG_FILE_SET=true; shift ;;
        -h|--help) usage ;;
        -*) fail "Unknown option: $1" ;;
        *)
            # If positional argument is provided
            if [ -z "${MODEL_NAME:-}" ]; then
                MODEL_NAME="$1"
            else
                fail "Unknown parameter: $1"
            fi
            ;;
    esac
    shift
done

if [[ "$DELETE_REQUESTED" == true ]]; then
    case "$ACTION" in
        download) ACTION=delete ;;
        backup) : ;;
        *) fail "--delete can only be used alone or with --backup." ;;
    esac
fi
case "$ACTION" in
    download|delete|backup|restore) [[ -n "$MODEL_NAME" ]] || fail "Model name is required." ;;
    list|cleanup) [[ -z "$MODEL_NAME" ]] || fail "--$ACTION does not accept a model name." ;;
esac
case "$ACTION" in
    backup|restore|list-backup)
        [[ -n "$BACKUP_DIR" && -d "$BACKUP_DIR" ]] || fail "--$ACTION requires an existing --backup-dir."
        BACKUP_DIR=$(realpath -- "$BACKUP_DIR") ;;
    *) [[ -z "$BACKUP_DIR" ]] || fail "--backup-dir requires --backup, --restore, or --list-backup." ;;
esac
[[ "$LIST_OPTIONS" == false || "$ACTION" == list || "$ACTION" == list-backup ]] \
    || fail "--format and --sort require a listing action."
[[ "$FORCE" == false || "$DELETE_REQUESTED" == true || "$ACTION" == cleanup ]] \
    || fail "--force requires --delete or --cleanup."
[[ "$ACTION" != list-backup || "$COPY_TO_FLAG" == false ]] \
    || fail "Backups are listed on the head node; omit -c."
[[ -z "$REVISION" || "$ACTION" == backup || "$ACTION" == delete ]] \
    || fail "--revision requires --backup or --delete."
[[ "$PARALLEL_COPY" == false || ( "$COPY_TO_FLAG" == true && ( "$ACTION" == download || "$ACTION" == restore ) ) ]] \
    || fail "--copy-parallel requires -c and a download or restore."
python3 "$CACHE_HELPER" validate --model "$MODEL_NAME" --revision "$REVISION" --format "$OUTPUT_FORMAT" --sort "$SORT_ORDER"

# Export config so autodiscover.sh picks it up
export CONFIG_FILE CONFIG_FILE_SET

# Source autodiscover.sh to load .env (for DOTENV_COPY_HOSTS) and make detection functions available
source "$(dirname "$0")/autodiscover.sh" >&2

# Resolve COPY_HOSTS if --copy-to was given without hosts, or use .env
resolve_copy_hosts() {
    # Saved hosts never trigger remote work without an explicit -c.
    [[ "$COPY_TO_FLAG" == true && "${#COPY_HOSTS[@]}" -eq 0 ]] || return 0
    if [[ -n "$DOTENV_COPY_HOSTS" ]]; then
        echo "Using COPY_HOSTS from .env: $DOTENV_COPY_HOSTS"
        add_copy_hosts "$DOTENV_COPY_HOSTS"
        return
    fi
    if [[ -n "${DOTENV_CLUSTER_LINKS:-}" ]]; then
        IFS=',' read -ra COPY_HOSTS <<< "${DOTENV_CLUSTER_NODES#*,}"
        [[ "$DOTENV_CLUSTER_NODES" == *,* ]] || COPY_HOSTS=()
        return
    fi
    echo "No hosts specified. Using autodiscovery..."
    detect_interfaces || fail "Interface detection failed."
    detect_local_ip || fail "Local IP detection failed."
    detect_nodes || fail "Node detection failed."
    detect_copy_hosts || fail "Copy host detection failed."

    if [ "${#COPY_PEER_NODES[@]}" -gt 0 ]; then
        COPY_HOSTS=("${COPY_PEER_NODES[@]}")
    fi
    [[ "${#COPY_HOSTS[@]}" -gt 0 ]] || fail "Autodiscovery found no other nodes."
    echo "Autodiscovered copy hosts: ${COPY_HOSTS[*]}"
}
resolve_copy_hosts >&2
if [[ "$COPY_TO_FLAG" == true && -n "${DOTENV_CLUSTER_LINKS:-}" ]]; then
    python3 "$SCRIPT_DIR/cluster_topology.py" check
fi

# Use the same dependency preparation on the head and on SSH peers.
ensure_uvx

# Start time tracking
START_TIME=$(date +%s)

# Resolve the same paths as hf, including its legacy Hub cache override.
HF_CACHE_DIR=$(resolve_cache_path home)
HUB_PATH=$(resolve_cache_path hub)

case "$ACTION" in
    list)
        collect_inventory
        render_inventory
        exit 0 ;;
    list-backup)
        INVENTORY_DIR=$(mktemp -d)
        trap 'rm -rf -- "$INVENTORY_DIR"' EXIT
        INVENTORY_FILES=("$INVENTORY_DIR/backup.json")
        python3 "$CACHE_HELPER" inventory --cache-dir "$BACKUP_DIR" --sort "$SORT_ORDER" \
            --model "$MODEL_NAME" > "${INVENTORY_FILES[0]}"
        render_inventory
        exit 0 ;;
    delete|cleanup)
        collect_inventory
        if [[ -n "$REVISION" ]]; then
            REVISION=$(python3 "$CACHE_HELPER" resolve-revision --model "$MODEL_NAME" \
                --revision "$REVISION" -- "${INVENTORY_FILES[@]}")
        fi
        delete_models
        exit $? ;;
    backup)
        if [[ -n "$REVISION" ]]; then
            # Pin the head's selected commit for both backup and cluster deletion.
            REVISION=$(python3 "$CACHE_HELPER" resolve-revision --cache-dir "$HUB_PATH" \
                --model "$MODEL_NAME" --revision "$REVISION")
        fi
        python3 "$CACHE_HELPER" backup --cache-dir "$HUB_PATH" --backup-dir "$BACKUP_DIR" \
            --model "$MODEL_NAME" --revision "$REVISION"
        if [[ "$DELETE_REQUESTED" == true ]]; then
            collect_inventory
            python3 "$CACHE_HELPER" covered --backup-dir "$BACKUP_DIR" --model "$MODEL_NAME" \
                --revision "$REVISION" -- "${INVENTORY_FILES[@]}"
            delete_models
        fi
        exit 0 ;;
esac
repair_cache_ownership "$HF_CACHE_DIR" "$HUB_PATH"

# Download or restore locally before using the common distribution path.
DOWNLOAD_START=$(date +%s)
if [[ "$ACTION" == restore ]]; then
    python3 "$CACHE_HELPER" restore --cache-dir "$HUB_PATH" --backup-dir "$BACKUP_DIR" --model "$MODEL_NAME"
    DOWNLOAD_TIME=$(($(date +%s) - DOWNLOAD_START))
else
    echo "Downloading model '$MODEL_NAME' using uvx..."
    if uvx hf download "$MODEL_NAME"; then
        DOWNLOAD_END=$(date +%s)
        DOWNLOAD_TIME=$((DOWNLOAD_END - DOWNLOAD_START))
        printf "Download completed in %02d:%02d:%02d\n" $((DOWNLOAD_TIME/3600)) $((DOWNLOAD_TIME%3600/60)) $((DOWNLOAD_TIME%60))
    else
        fail "Failed to download model '$MODEL_NAME'."
    fi
fi

# Determine model directory path
# uvx hf download stores models in ~/.cache/huggingface/hub with the pattern: models--<org>--<model>-<suffix>
MODEL_DIR=""

# Try to find the model directory
# The pattern for model directories is: ~/.cache/huggingface/hub/models--ORG--MODEL-VARIATION (or similar)
# Model names like "QuantTrio/MiniMax-M2-AWQ" become "models--QuantTrio--MiniMax-M2-AQW" or similar

# Parse org and model name from MODEL_NAME
if [[ "$MODEL_NAME" == */* ]]; then
    ORG="${MODEL_NAME%%/*}"
    MODEL="${MODEL_NAME##*/}"
else
    ORG=""
    MODEL="$MODEL_NAME"
fi

# Convert to the directory pattern used by HuggingFace

if [ -d "$HUB_PATH" ]; then
    if [ -n "$ORG" ]; then
        MODEL_DIR="$HUB_PATH/models--${ORG}--${MODEL}"
    else
        # For models without org, check both patterns
        if [ -d "$HUB_PATH/models--${MODEL}" ]; then
            MODEL_DIR="$HUB_PATH/models--${MODEL}"
        else
            MODEL_DIR="$HUB_PATH/${MODEL}"
        fi
    fi
fi

if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Could not find downloaded model directory in $HUB_PATH"
    echo "Please check the $HUB_PATH directory manually."
    exit 1
fi

echo "Model directory: $MODEL_DIR"

# Copy to host if requested
COPY_TIME=0
if [ "${#COPY_HOSTS[@]}" -gt 0 ]; then
    echo ""
    echo "Copying model to ${#COPY_HOSTS[@]} host(s): ${COPY_HOSTS[*]}"
    if [ "$PARALLEL_COPY" = true ]; then
        echo "Parallel copy enabled."
    fi
    COPY_START=$(date +%s)

    # Finish all sudo prompts in the foreground, even when transfers run in parallel.
    for host in "${COPY_HOSTS[@]}"; do
        prepare_remote_cache "$host"
    done

    if [ "$PARALLEL_COPY" = true ]; then
        PIDS=()
        for host in "${COPY_HOSTS[@]}"; do
            copy_model_to_host "$host" "$MODEL_NAME" "$MODEL_DIR" &
            PIDS+=($!)
        done
        COPY_FAILURE=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                COPY_FAILURE=1
            fi
        done
        if [ "$COPY_FAILURE" -ne 0 ]; then
            echo "One or more copies failed."
            exit 1
        fi
    else
        for host in "${COPY_HOSTS[@]}"; do
            copy_model_to_host "$host" "$MODEL_NAME" "$MODEL_DIR"
        done
    fi

    COPY_END=$(date +%s)
    COPY_TIME=$((COPY_END - COPY_START))
    echo ""
    echo "Copy complete."
else
    echo "No host specified, skipping copy."
fi

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

# Display timing statistics
echo ""
echo "========================================="
echo "         TIMING STATISTICS"
echo "========================================="
echo "${ACTION^}:   $(printf '%02d:%02d:%02d' $((DOWNLOAD_TIME/3600)) $((DOWNLOAD_TIME%3600/60)) $((DOWNLOAD_TIME%60)))"
if [ "$COPY_TIME" -gt 0 ]; then
    echo "Copy:      $(printf '%02d:%02d:%02d' $((COPY_TIME/3600)) $((COPY_TIME%3600/60)) $((COPY_TIME%60)))"
fi
echo "Total:     $(printf '%02d:%02d:%02d' $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60)))"
echo "========================================="
echo "Done: $ACTION $MODEL_NAME."
