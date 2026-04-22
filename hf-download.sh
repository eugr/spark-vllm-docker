#!/bin/bash
set -e

HUB_PATH="${HF_HOME:-$HOME/.cache/huggingface}/hub"
HF_DOWNLOAD_UVX_SPEC="${HF_DOWNLOAD_UVX_SPEC:-huggingface_hub==1.11.0}"
MODEL_REVISION=""

# Default values
COPY_HOSTS=()
SSH_USER="$USER"
PARALLEL_COPY=false

# Help function
usage() {
    echo "Usage: $0 [OPTIONS] <model-name>"
    echo "  <model-name>                : HuggingFace model name (e.g., 'QuantTrio/MiniMax-M2-AWQ')"
    echo "  -c, --copy-to <hosts>       : Host(s) to copy the model to. Accepts comma or space-delimited lists after the flag."
    echo "      --copy-to-host          : Alias for --copy-to (backwards compatibility)."
    echo "      --copy-parallel         : Copy to all hosts in parallel instead of serially."
    echo "      --revision <sha>        : Pin the download to a specific Hub revision."
    echo "  -u, --user <user>           : Username for ssh commands (default: \$USER)"
    echo "  -h, --help                  : Show this help message"
    exit 1
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
    
    if rsync -a --mkpath --partial --partial-dir=.rsync-partial --info=progress2 "$model_dir" "${SSH_USER}@${host}:$HUB_PATH/"; then
        host_copy_end=$(date +%s)
        host_copy_time=$((host_copy_end - host_copy_start))
        printf "Copy to %s completed in %02d:%02d:%02d\n" "$host" $((host_copy_time/3600)) $((host_copy_time%3600/60)) $((host_copy_time%60))
    else
        echo "Copy to $host failed."
        return 1
    fi
}

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--copy-to|--copy-to-host|--copy-to-hosts)
            shift
            # Consume arguments until the next flag or end of args
            while [[ "$#" -gt 0 && "$1" != -* ]]; do
                add_copy_hosts "$1"
                shift
            done

            # If no hosts specified, use autodiscovery
            if [ "${#COPY_HOSTS[@]}" -eq 0 ]; then
                echo "No hosts specified. Using autodiscovery..."
                source "$(dirname "$0")/autodiscover.sh"
                
                detect_nodes
                if [ $? -ne 0 ]; then
                    echo "Error: Autodiscovery failed."
                    exit 1
                fi
                
                # Use PEER_NODES directly
                if [ ${#PEER_NODES[@]} -gt 0 ]; then
                    COPY_HOSTS=("${PEER_NODES[@]}")
                fi
                
                if [ "${#COPY_HOSTS[@]}" -eq 0 ]; then
                     echo "Error: Autodiscovery found no other nodes."
                     exit 1
                fi
                echo "Autodiscovered hosts: ${COPY_HOSTS[*]}"
            fi
            continue
            ;;
        --copy-parallel) PARALLEL_COPY=true ;;
        --revision) MODEL_REVISION="$2"; shift ;;
        -u|--user) SSH_USER="$2"; shift ;;
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

# Validate model name is provided
if [ -z "${MODEL_NAME:-}" ]; then
    echo "Error: Model name is required."
    usage
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

# Download model
echo "Downloading model '$MODEL_NAME' using uvx..."
DOWNLOAD_START=$(date +%s)
download_cmd=(uvx --from "$HF_DOWNLOAD_UVX_SPEC" hf download "$MODEL_NAME")
if [ -n "$MODEL_REVISION" ]; then
    download_cmd+=(--revision "$MODEL_REVISION")
fi
if "${download_cmd[@]}"; then
    DOWNLOAD_END=$(date +%s)
    DOWNLOAD_TIME=$((DOWNLOAD_END - DOWNLOAD_START))
    printf "Download completed in %02d:%02d:%02d\n" $((DOWNLOAD_TIME/3600)) $((DOWNLOAD_TIME%3600/60)) $((DOWNLOAD_TIME%60))
else
    echo "Error: Failed to download model '$MODEL_NAME'."
    exit 1
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

if [ -z "$MODEL_DIR" ]; then
    echo "Error: Could not find downloaded model directory in $HUB_PATH"
    echo "Please check the ~/.cache/huggingface/hub directory manually."
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
