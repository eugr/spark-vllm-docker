#!/bin/bash
set -e

# Default values
COPY_HOSTS=()
SSH_USER="$USER"
PARALLEL_COPY=false
CONFIG_FILE=""
CONFIG_FILE_SET=false

# Help function
usage() {
    echo "Usage: $0 [OPTIONS] <model-name>"
    echo "  <model-name>                : HuggingFace model name (e.g., 'QuantTrio/MiniMax-M2-AWQ')"
    echo "  -c, --copy-to [hosts]       : Copy the model. Omit hosts to use COPY_HOSTS from .env or autodiscovery."
    echo "      --copy-to-host          : Alias for --copy-to (backwards compatibility)."
    echo "      --copy-parallel         : With -c, copy to all resolved hosts concurrently."
    echo "  -u, --user <user>           : Username for ssh commands (default: \$USER)"
    echo "  --config <file>             : Path to .env configuration file (default: .env in script directory)"
    echo "  -h, --help                  : Show this help message"
    echo "Cache ownership is repaired using an existing vllm-node or vllm-node-b12x image, then sudo if needed."
    echo "Sudo may prompt on each node before downloads or copies."
    exit 1
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
    if ! ssh "${ssh_args[@]}" "${SSH_USER}@${host}" "$remote_command"; then
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
    if rsync -av -s --mkpath --progress --copy-unsafe-links \
            "$model_dir/" "${SSH_USER}@${host}:$HUB_PATH/$(basename "$model_dir")/"; then
        host_copy_end=$(date +%s)
        host_copy_time=$((host_copy_end - host_copy_start))
        printf "Copy to %s completed in %02d:%02d:%02d\n" "$host" $((host_copy_time/3600)) $((host_copy_time%3600/60)) $((host_copy_time%60))
    else
        echo "Copy to $host failed."
        return 1
    fi
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
                add_copy_hosts "$1"
                shift
            done
            continue
            ;;
        --copy-parallel) PARALLEL_COPY=true ;;
        -u|--user) SSH_USER="$2"; shift ;;
        --config) CONFIG_FILE="$2"; CONFIG_FILE_SET=true; shift ;;
        -h|--help) usage ;;
        *)
            # If positional argument is provided
            if [ -z "${MODEL_NAME:-}" ]; then
                MODEL_NAME="$1"
            else
                echo "Error: Unknown parameter: $1"
                usage
            fi
            ;;
    esac
    shift
done

# Export config so autodiscover.sh picks it up
export CONFIG_FILE CONFIG_FILE_SET

# Source autodiscover.sh to load .env (for DOTENV_COPY_HOSTS) and make detection functions available
source "$(dirname "$0")/autodiscover.sh"

# Validate model name is provided
if [ -z "${MODEL_NAME:-}" ]; then
    echo "Error: Model name is required."
    usage
fi

# Resolve COPY_HOSTS if --copy-to was given without hosts, or use .env
if [ "$COPY_TO_FLAG" = true ] && [ "${#COPY_HOSTS[@]}" -eq 0 ]; then
    # --copy-to was specified but no hosts given: use .env or autodiscover
    if [[ -n "$DOTENV_COPY_HOSTS" ]]; then
        echo "Using COPY_HOSTS from .env: $DOTENV_COPY_HOSTS"
        IFS=',' read -ra HOSTS_FROM_ENV <<< "$DOTENV_COPY_HOSTS"
        COPY_HOSTS=("${HOSTS_FROM_ENV[@]}")
    else
        echo "No hosts specified. Using autodiscovery..."
        detect_interfaces || { echo "Error: Interface detection failed."; exit 1; }
        detect_local_ip || { echo "Error: Local IP detection failed."; exit 1; }
        detect_nodes || { echo "Error: Node detection failed."; exit 1; }
        detect_copy_hosts || { echo "Error: Copy host detection failed."; exit 1; }

        if [ "${#COPY_PEER_NODES[@]}" -gt 0 ]; then
            COPY_HOSTS=("${COPY_PEER_NODES[@]}")
        fi

        if [ "${#COPY_HOSTS[@]}" -eq 0 ]; then
            echo "Error: Autodiscovery found no other nodes."
            exit 1
        fi
        echo "Autodiscovered copy hosts: ${COPY_HOSTS[*]}"
    fi
elif [ "$COPY_TO_FLAG" = false ] && [ "${#COPY_HOSTS[@]}" -eq 0 ] && [[ -n "$DOTENV_COPY_HOSTS" ]]; then
    # No --copy-to flag but .env has COPY_HOSTS — don't auto-copy; user must request it explicitly
    : # intentional no-op; user didn't ask for copy
fi

# Check if uvx is installed
if ! command -v uvx &> /dev/null; then
    echo "Error: 'uvx' command not found."
    echo ""
    echo "Please install uvx first by running:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  # or"
    echo "  pip install uvx"
    echo ""
    exit 1
fi

# Start time tracking
START_TIME=$(date +%s)

# Resolve the same paths as hf, including its legacy Hub cache override.
HF_CACHE_DIR=$(resolve_cache_path home)
HUB_PATH=$(resolve_cache_path hub)
repair_cache_ownership "$HF_CACHE_DIR" "$HUB_PATH"

# Download model
echo "Downloading model '$MODEL_NAME' using uvx..."
DOWNLOAD_START=$(date +%s)
if uvx hf download "$MODEL_NAME"; then
    DOWNLOAD_END=$(date +%s)
    DOWNLOAD_TIME=$((DOWNLOAD_END - DOWNLOAD_START))
    printf "Download completed in %02d:%02d:%02d\n" $((DOWNLOAD_TIME/3600)) $((DOWNLOAD_TIME%3600/60)) $((DOWNLOAD_TIME%60))
else
    echo "Error: Failed to download model '$MODEL_NAME'."
    exit 1
fi

# Determine model directory path
# uvx hf download stores models in the hub cache ($HUB_PATH) with the pattern: models--<org>--<model>-<suffix>
MODEL_DIR=""

# Try to find the model directory
# The pattern for model directories is: $HUB_PATH/models--ORG--MODEL-VARIATION (or similar)
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

if [ -z "$MODEL_DIR" ]; then
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
echo "Download:   $(printf '%02d:%02d:%02d' $((DOWNLOAD_TIME/3600)) $((DOWNLOAD_TIME%3600/60)) $((DOWNLOAD_TIME%60)))"
if [ "$COPY_TIME" -gt 0 ]; then
    echo "Copy:      $(printf '%02d:%02d:%02d' $((COPY_TIME/3600)) $((COPY_TIME%3600/60)) $((COPY_TIME%60)))"
fi
echo "Total:     $(printf '%02d:%02d:%02d' $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60)))"
echo "========================================="
echo "Done downloading $MODEL_NAME."
