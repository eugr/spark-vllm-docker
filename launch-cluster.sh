#!/bin/bash

# Default Configuration
IMAGE_NAME="vllm-node"
DEFAULT_CONTAINER_NAME="vllm_node"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
# Modify these if you want to pass additional docker args or set VLLM_SPARK_EXTRA_DOCKER_ARGS variable
DOCKER_ARGS="-e NCCL_IGNORE_CPU_AFFINITY=1 -v $HF_CACHE_DIR:/root/.cache/huggingface"

# Append additional arguments from environment variable
if [[ -n "$VLLM_SPARK_EXTRA_DOCKER_ARGS" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS $VLLM_SPARK_EXTRA_DOCKER_ARGS"
fi

# ETH_IF and IB_IF will be auto-detected if not provided
ETH_IF=""
IB_IF=""
NCCL_DEBUG_VAL=""
MASTER_PORT="29501"

# Initialize variables
NODES_ARG=""
CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
COMMAND_TO_RUN=""
DAEMON_MODE="false"
CHECK_CONFIG="false"
ACTION=""
CLUSTER_WAS_RUNNING="false"
MOD_PATHS=()
MOD_TYPES=()
LAUNCH_SCRIPT_PATH=""
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
CONFIG_FILE=""  # Will be set to default after argument parsing

ACTIONS_ARG=""
SOLO_MODE="false"
NO_RAY_MODE="false"
LAUNCH_SCRIPT_MODE="false"
MOUNT_CACHE_DIRS="true"
BUILD_JOBS=""
NON_PRIVILEGED_MODE="false"
MEM_LIMIT_GB="110"
MEM_SWAP_LIMIT_GB=""
PIDS_LIMIT="4096"
SHM_SIZE_GB="64"

# Function to print usage
usage() {
    echo "Usage: $0 [-n <node_ips>] [-t <image_name>] [--name <container_name>] [--eth-if <if_name>] [--ib-if <if_name>] [--nccl-debug <level>] [--check-config] [--solo] [-d] [action] [command]"
    echo "  -n, --nodes     Comma-separated list of node IPs (Optional, auto-detected if omitted)"
    echo "  -t              Docker image name (Optional, default: $IMAGE_NAME)"
    echo "  --name          Container name (Optional, default: $DEFAULT_CONTAINER_NAME)"
    echo "  --eth-if        Ethernet interface (Optional, auto-detected)"
    echo "  --ib-if         InfiniBand interface (Optional, auto-detected)"
    echo "  -e, --env       Environment variable to pass to container (e.g. -e VAR=val)"
    echo "  -j              Number of parallel jobs for build environment variables (optional)"
    echo "  --nccl-debug    NCCL debug level (Optional, one of: VERSION, WARN, INFO, TRACE). If no level is provided, defaults to INFO."
    echo "  --apply-mod     Path to directory or zip file containing run.sh to apply before launch (Can be specified multiple times)"
    echo "  --launch-script Path to bash script to execute in the container (from examples/ directory or absolute path). If launch script is specified, action should be omitted."
    echo "  --check-config  Check configuration and auto-detection without launching"
    echo "  --solo          Solo mode: skip autodetection, launch only on current node, do not launch Ray cluster"
    echo "  --master-port   Port for cluster coordination: Ray head port or PyTorch distributed master port (default: 29501)"
    echo "  --no-ray        No-Ray mode: run multi-node vLLM without Ray (uses PyTorch distributed backend)"
    echo "  --no-cache-dirs Do not mount default cache directories (~/.cache/vllm, ~/.cache/flashinfer, ~/.triton)"
    echo "  -d              Daemon mode (only for 'start' action)"
    echo "  --non-privileged Run in non-privileged mode (removes --privileged and --ipc=host)"
    echo "  --mem-limit-gb  Memory limit in GB (default: 110, only with --non-privileged)"
    echo "  --mem-swap-limit-gb Memory+swap limit in GB (default: mem-limit + 10, only with --non-privileged)"
    echo "  --pids-limit    Process limit (default: 4096, only with --non-privileged)"
    echo "  --shm-size-gb   Shared memory size in GB (default: 64, only with --non-privileged)"
    echo "  --config        Path to .env configuration file (default: .env in script directory)
  --setup/--discover  Force autodiscovery and save configuration (even if .env exists)"
    echo "  action          start | stop | status | exec (Default: start). Not compatible with --launch-script."
    echo "  command         Command to run (only for 'exec' action). Not compatible with --launch-script."
    echo ""
    echo "Supported .env file variables:"
    echo "  CLUSTER_NODES       Comma-separated list of node IPs"
    echo "  ETH_IF              Ethernet interface name"
    echo "  IB_IF               InfiniBand interface name"
    echo "  MASTER_PORT         Port for cluster coordination (default: 29501)"
    echo "  CONTAINER_NAME      Container name (default: vllm_node)"
    echo "  LOCAL_IP            Local IP address (for solo mode or override auto-detection)"
    echo "  CONTAINER_*         Any variable starting with CONTAINER_ (except CONTAINER_NAME)"
    echo "                      becomes -e flag. Example: CONTAINER_NCCL_DEBUG=INFO -> -e NCCL_DEBUG=INFO"
    echo ""
    echo "Example .env file:"
    echo "  CLUSTER_NODES=192.168.1.1,192.168.1.2"
    echo "  ETH_IF=eth0"
    echo "  IB_IF=ib0"
    echo "  MASTER_PORT=29501"
    echo "  CONTAINER_NAME=vllm_node"
    echo "  LOCAL_IP=192.168.1.1"
    echo "  CONTAINER_NCCL_DEBUG=INFO"
    echo "  CONTAINER_HF_TOKEN=abc123"
    echo ""
    echo "Launch Script Usage:"
    echo "  $0 --launch-script examples/my-script.sh   # Script copied to container and executed"
    echo "  $0 --launch-script /path/to/script.sh      # Uses absolute path to script"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--nodes) NODES_ARG="$2"; shift ;;
        -t) IMAGE_NAME="$2"; shift ;;
        --name) CONTAINER_NAME="$2"; shift ;;
        --eth-if) ETH_IF="$2"; shift ;;
        --ib-if) IB_IF="$2"; shift ;;
        -e|--env) DOCKER_ARGS="$DOCKER_ARGS -e $2"; shift ;;
        -j) BUILD_JOBS="$2"; shift ;;
        --apply-mod) MOD_PATHS+=("$2"); shift ;;
        --launch-script) LAUNCH_SCRIPT_PATH="$2"; shift ;;
        --nccl-debug)
            if [[ -n "$2" && "$2" =~ ^(VERSION|WARN|INFO|TRACE)$ ]]; then
                NCCL_DEBUG_VAL="$2"
                shift
            else
                NCCL_DEBUG_VAL="INFO"
            fi
            ;;
        --master-port|--head-port) MASTER_PORT="$2"; shift ;;
        --check-config) CHECK_CONFIG="true" ;;
        --solo) SOLO_MODE="true" ;;
        --no-ray) NO_RAY_MODE="true" ;;
        --no-cache-dirs) MOUNT_CACHE_DIRS="false" ;;
        --non-privileged) NON_PRIVILEGED_MODE="true" ;;
        --mem-limit-gb) MEM_LIMIT_GB="$2"; shift ;;
        --mem-swap-limit-gb) MEM_SWAP_LIMIT_GB="$2"; shift ;;
        --pids-limit) PIDS_LIMIT="$2"; shift ;;
        --shm-size-gb) SHM_SIZE_GB="$2"; shift ;;
        -d) DAEMON_MODE="true" ;;
        -h|--help) usage ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --setup|--discover) FORCE_DISCOVER=true; export FORCE_DISCOVER ;;
        start|stop|status) 
            if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
                echo "Error: Action '$1' is not compatible with --launch-script. Please omit the action or not use --launch-script."
                exit 1
            fi
            ACTION="$1" 
            ;;
        exec)
            if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
                echo "Error: Action 'exec' is not compatible with --launch-script. Please omit the action or not use --launch-script."
                exit 1
            fi
            ACTION="exec"
            shift
            COMMAND_TO_RUN=$(printf "%q " "$@")
            break
            ;;
        *) 
            echo "Error: Unknown argument or action: $1"
            usage
            ;;
    esac
    shift
done

# Set .env file path (use default if not specified)
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$SCRIPT_DIR/.env"
    CONFIG_FILE_SET=false
else
    CONFIG_FILE_SET=true
fi

# Load .env file
if [[ -f "$CONFIG_FILE" ]]; then
    echo "Loading configuration from .env file..."
    
    # Validate .env file syntax
    if ! python3 -c "
import sys
import re

env_file = '$CONFIG_FILE'
seen_keys = set()

with open(env_file, 'r') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Check for key=value format
        if '=' not in line:
            print(f'Error: Invalid syntax at line {line_num}: missing \"=\"')
            sys.exit(1)
        
        key = line.split('=', 1)[0].strip()
        
        # Validate key format (alphanumeric + underscore)
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key):
            print(f'Error: Invalid key format at line {line_num}: {key}')
            sys.exit(1)
        
        # Check for duplicates
        if key in seen_keys:
            print(f'Error: Duplicate key at line {line_num}: {key}')
            sys.exit(1)
        
        seen_keys.add(key)

sys.exit(0)
" 2>/dev/null; then
        echo "Error: Invalid .env file syntax. Aborting."
        exit 1
    fi
    
    # Load .env variables with DOTENV_ prefix
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Skip comments and empty lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        
        # Remove leading/trailing whitespace from key
        key=$(echo "$key" | xargs)
        
        # Skip if key is empty after trimming
        [[ -z "$key" ]] && continue
        
        # Remove quotes and whitespace from value using Python for proper shlex handling
        value=$(python3 -c "
import shlex
import sys
value = '''$value'''
# Strip whitespace
value = value.strip()
# Remove surrounding quotes if present
if (value.startswith('\"') and value.endswith('\"')) or (value.startswith(\"'\" ) and value.endswith(\"'\")):
    value = value[1:-1]
print(value)
")
        
        # Export with DOTENV_ prefix
        export "DOTENV_$key=$value"
    done < "$CONFIG_FILE"
    
    echo "Loaded .env variables: $(compgen -v DOTENV_ | tr '\n' ' ')"
fi

# Apply .env configuration (CLI args take precedence)
if [[ -z "$NODES_ARG" && -n "$DOTENV_CLUSTER_NODES" ]]; then
    NODES_ARG="$DOTENV_CLUSTER_NODES"
fi

if [[ -z "$ETH_IF" && -n "$DOTENV_ETH_IF" ]]; then
    ETH_IF="$DOTENV_ETH_IF"
fi

if [[ -z "$IB_IF" && -n "$DOTENV_IB_IF" ]]; then
    IB_IF="$DOTENV_IB_IF"
fi

if [[ -z "$MASTER_PORT" || "$MASTER_PORT" == "29501" ]] && [[ -n "$DOTENV_MASTER_PORT" ]]; then
    MASTER_PORT="$DOTENV_MASTER_PORT"
fi

if [[ -z "$CONTAINER_NAME" || "$CONTAINER_NAME" == "vllm_node" ]] && [[ -n "$DOTENV_CONTAINER_NAME" ]]; then
    CONTAINER_NAME="$DOTENV_CONTAINER_NAME"
fi

if [[ -n "$DOTENV_LOCAL_IP" ]]; then
    export LOCAL_IP="$DOTENV_LOCAL_IP"
fi

# Validate non-privileged mode flags
if [[ "$NON_PRIVILEGED_MODE" == "true" ]]; then
    # Set default swap limit if not specified
    if [[ -z "$MEM_SWAP_LIMIT_GB" ]]; then
        MEM_SWAP_LIMIT_GB=$((MEM_LIMIT_GB + 10))
    fi
else
    # Check if non-privileged flags were used without --non-privileged
    for flag in "--mem-limit-gb" "--mem-swap-limit-gb" "--pids-limit" "--shm-size-gb"; do
        if [[ "$*" == *"$flag"* ]]; then
            echo "Error: $flag can only be used with --non-privileged"
            exit 1
        fi
    done
fi

# Append NCCL_DEBUG if set, with validation
if [[ -n "$NCCL_DEBUG_VAL" ]]; then
    case "$NCCL_DEBUG_VAL" in
        VERSION|WARN|INFO|TRACE)
            DOCKER_ARGS="$DOCKER_ARGS -e NCCL_DEBUG=$NCCL_DEBUG_VAL"
            ;;
        *)
            echo "Error: Invalid value for --nccl-debug: $NCCL_DEBUG_VAL"
            echo "Allowed values: VERSION, WARN, INFO, TRACE"
            exit 1
            ;;
    esac
fi

# Add container environment variables from .env (CONTAINER_* pattern)
# Excludes CONTAINER_NAME which is a configuration variable, not an env var
for env_var in $(compgen -v DOTENV_CONTAINER_); do
    # Skip CONTAINER_NAME as it's a configuration variable
    [[ "$env_var" == "DOTENV_CONTAINER_NAME" ]] && continue
    
    # Get the value
    value="${!env_var}"
    
    # Extract the actual env var name (remove DOTENV_CONTAINER_ prefix)
    actual_var="${env_var#DOTENV_CONTAINER_}"
    
    # Properly escape the value for shell using Python
    escaped_value=$(python3 -c "import shlex; print(shlex.quote('$value'))")
    
    # Add to docker args
    DOCKER_ARGS="$DOCKER_ARGS -e $actual_var=$escaped_value"
    echo "Adding container env: $actual_var"
done

# Add build job parallelization environment variables if BUILD_JOBS is set
if [[ -n "$BUILD_JOBS" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS -e MAX_JOBS=$BUILD_JOBS"
    DOCKER_ARGS="$DOCKER_ARGS -e CMAKE_BUILD_PARALLEL_LEVEL=$BUILD_JOBS"
    DOCKER_ARGS="$DOCKER_ARGS -e NINJAFLAGS=-j$BUILD_JOBS"
    DOCKER_ARGS="$DOCKER_ARGS -e MAKEFLAGS=-j$BUILD_JOBS"
fi

# Add cache dirs if requested
CACHE_DIRS_TO_CREATE=()
if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
    # vLLM Cache
    DOCKER_ARGS="$DOCKER_ARGS -v $HOME/.cache/vllm:/root/.cache/vllm"
    CACHE_DIRS_TO_CREATE+=("$HOME/.cache/vllm")
    
    # FlashInfer Cache
    DOCKER_ARGS="$DOCKER_ARGS -v $HOME/.cache/flashinfer:/root/.cache/flashinfer"
    CACHE_DIRS_TO_CREATE+=("$HOME/.cache/flashinfer")

    # Triton Cache
    DOCKER_ARGS="$DOCKER_ARGS -v $HOME/.triton:/root/.triton"
    CACHE_DIRS_TO_CREATE+=("$HOME/.triton")
fi

# Resolve launch script path if specified
if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
    # Check if it's an absolute path or relative path that exists
    if [[ -f "$LAUNCH_SCRIPT_PATH" ]]; then
        LAUNCH_SCRIPT_PATH=$(realpath "$LAUNCH_SCRIPT_PATH")
    # Check if it's just a filename, look in examples/ directory
    elif [[ -f "$SCRIPT_DIR/examples/$LAUNCH_SCRIPT_PATH" ]]; then
        LAUNCH_SCRIPT_PATH="$SCRIPT_DIR/examples/$LAUNCH_SCRIPT_PATH"
    # Check if it's a name without .sh extension
    elif [[ -f "$SCRIPT_DIR/examples/${LAUNCH_SCRIPT_PATH}.sh" ]]; then
        LAUNCH_SCRIPT_PATH="$SCRIPT_DIR/examples/${LAUNCH_SCRIPT_PATH}.sh"
    else
        echo "Error: Launch script '$LAUNCH_SCRIPT_PATH' not found."
        echo "Searched in:"
        echo "  - $LAUNCH_SCRIPT_PATH"
        echo "  - $SCRIPT_DIR/examples/$LAUNCH_SCRIPT_PATH"
        echo "  - $SCRIPT_DIR/examples/${LAUNCH_SCRIPT_PATH}.sh"
        exit 1
    fi
    
    echo "Using launch script: $LAUNCH_SCRIPT_PATH"
    
    # Set command to run the copied script (use absolute path since docker exec may not be in /workspace)
    COMMAND_TO_RUN="/workspace/exec-script.sh"
    LAUNCH_SCRIPT_MODE="true"

    # If launch script is specified, default action to exec unless explicitly set to stop/status
    if [[ -z "$ACTION" || "$ACTION" == "start" ]]; then
        ACTION="exec"
    fi
fi

# Validate MOD_PATHS if set
for i in "${!MOD_PATHS[@]}"; do
    mod_path="${MOD_PATHS[$i]}"
    if [[ ! -e "$mod_path" ]]; then
        echo "Error: Mod path '$mod_path' does not exist."
        exit 1
    fi
    
    if [[ -d "$mod_path" ]]; then
        if [[ ! -f "$mod_path/run.sh" ]]; then
             echo "Error: Mod directory '$mod_path' must contain 'run.sh'."
             exit 1
        fi
        MOD_TYPES[$i]="dir"
    elif [[ -f "$mod_path" && "$mod_path" == *.zip ]]; then
        # Check zip content using unzip if available, else python
        if command -v unzip &> /dev/null; then
            if ! unzip -l "$mod_path" | grep -q "run.sh"; then
                 echo "Error: Mod zip file '$mod_path' must contain 'run.sh'."
                 exit 1
            fi
        else
             # Fallback to python for checking zip content
             if ! python3 -c "import zipfile, sys; sys.exit(0 if 'run.sh' in zipfile.ZipFile(sys.argv[1]).namelist() else 1)" "$mod_path"; then
                 echo "Error: Mod zip file '$mod_path' must contain 'run.sh'."
                 exit 1
             fi
        fi
        MOD_TYPES[$i]="zip"
    else
        echo "Error: --apply-mod '$mod_path' must be a directory or a .zip file."
        exit 1
    fi
    MOD_PATHS[$i]=$(realpath "$mod_path")
done

# --- Auto-Detection Logic ---
# Source autodiscover module
source "$(dirname "$0")/autodiscover.sh"

if [[ "${FORCE_DISCOVER:-false}" == "true" ]]; then
    # --setup: force full autodiscovery and save configuration
    echo "Running full autodiscovery (--setup)..."
    # Clear pre-loaded values so detect functions run fresh instead of short-circuiting
    ETH_IF="" IB_IF="" NODES_ARG="" LOCAL_IP=""
    detect_interfaces || exit 1
    detect_local_ip || exit 1
    detect_nodes || exit 1
    detect_copy_hosts || exit 1
    save_config || exit 1
    # Reload .env so DOTENV_* variables reflect saved config
    load_env_if_exists
    [[ -z "$NODES_ARG" && -n "$DOTENV_CLUSTER_NODES" ]] && NODES_ARG="$DOTENV_CLUSTER_NODES"
    [[ -z "$ETH_IF" && -n "$DOTENV_ETH_IF" ]] && ETH_IF="$DOTENV_ETH_IF"
    [[ -z "$IB_IF" && -n "$DOTENV_IB_IF" ]] && IB_IF="$DOTENV_IB_IF"
    # If no action was specified, setup was the only intent — exit cleanly
    if [[ -z "$ACTION" && "$LAUNCH_SCRIPT_MODE" != "true" ]]; then
        exit 0
    fi
fi

if [[ "$SOLO_MODE" == "true" ]]; then
    # Solo mode: skip node detection, just get local IP
    # Use LOCAL_IP from .env if set, otherwise default to 127.0.0.1
    if [[ -z "$LOCAL_IP" ]]; then
        LOCAL_IP="127.0.0.1"
    fi
    NODES_ARG="$LOCAL_IP"
    PEER_NODES=()
    echo "Solo mode enabled. Skipping node detection."
else
    # Perform auto-detection
    detect_interfaces || exit 1
    detect_nodes || exit 1
fi

if [[ -z "$NODES_ARG" ]]; then
    echo "Error: Nodes argument (-n) is mandatory or could not be auto-detected."
    usage
fi

# Split nodes into array
IFS=',' read -r -a ALL_NODES <<< "$NODES_ARG"

if [[ "$SOLO_MODE" != "true" ]]; then
    # Detect Head IP (Local IP)
    detect_local_ip || exit 1
fi

HEAD_IP="$LOCAL_IP"

# Verify HEAD_IP is in ALL_NODES
FOUND_HEAD=false
for ip in "${ALL_NODES[@]}"; do
    ip=$(echo "$ip" | xargs)
    if [[ "$ip" == "$HEAD_IP" ]]; then
        FOUND_HEAD=true
        break
    fi
done

if [ "$FOUND_HEAD" = false ]; then
    echo "Error: Local IP ($HEAD_IP) is not in the list of nodes ($NODES_ARG)."
    exit 1
fi

# Implicit Solo Mode Detection
if [[ "$SOLO_MODE" == "false" && ${#PEER_NODES[@]} -eq 0 ]]; then
    echo "Only local node detected/configured. Activating solo mode (no Ray cluster)."
    SOLO_MODE="true"
fi

if [[ "$NO_RAY_MODE" == "true" && "$SOLO_MODE" == "true" ]]; then
    echo "Warning: Only one node detected; --no-ray has no effect in solo mode. Proceeding normally."
    NO_RAY_MODE="false"
fi

echo "Head Node: $HEAD_IP"
echo "Worker Nodes: ${PEER_NODES[*]}"
echo "Container Name: $CONTAINER_NAME"
echo "Image Name: $IMAGE_NAME"
echo "Action: $ACTION"

# Check SSH connectivity to worker nodes
if [[ "$ACTION" == "start" || "$ACTION" == "exec" || "$CHECK_CONFIG" == "true" ]]; then
    if [ ${#PEER_NODES[@]} -gt 0 ]; then
        echo "Checking SSH connectivity to worker nodes..."
        for worker in "${PEER_NODES[@]}"; do
            if ! ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$worker" true 2>/dev/null; then
                echo "Error: Passwordless SSH to $worker failed."
                echo "  Please ensure SSH keys are configured and the host is reachable."
                exit 1
            fi
            echo "  SSH to $worker: OK"
        done
    fi
fi

if [[ -z "$ACTION" && "$LAUNCH_SCRIPT_MODE" != "true" && "$CHECK_CONFIG" != "true" ]]; then
    echo "Error: No action specified. Use: start | stop | status | exec"
    usage
    exit 1
fi

if [[ "$CHECK_CONFIG" == "true" ]]; then
    echo "Configuration Check Complete."
    echo "  Image Name: $IMAGE_NAME"
    echo "  ETH Interface: $ETH_IF"
    echo "  IB Interface: $IB_IF"
    echo "  Docker Args: $DOCKER_ARGS"
    if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
         echo "  Mounting Cache Dirs: ${CACHE_DIRS_TO_CREATE[*]}"
    else
         echo "  Mounting Cache Dirs: (Disabled)"
    fi
    exit 0
fi

# Cleanup Function
cleanup() {
    # Remove traps to prevent nested cleanup
    trap - EXIT INT TERM HUP

    if [[ "$CLUSTER_WAS_RUNNING" == "true" ]]; then
        echo "Cluster was already running when script started. Skipping cleanup."
        return
    fi

    echo ""
    echo "Stopping cluster..."
    
    # Stop Head
    echo "Stopping head node ($HEAD_IP)..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    
    # Stop Workers
    for worker in "${PEER_NODES[@]}"; do
        echo "Stopping worker node ($worker)..."
        ssh "$worker" "docker stop $CONTAINER_NAME" >/dev/null 2>&1 || true
    done
    
    echo "Cluster stopped."
}

# Handle 'stop' action
if [[ "$ACTION" == "stop" ]]; then
    cleanup
    exit 0
fi

# Handle 'status' action
if [[ "$ACTION" == "status" ]]; then
    echo "Checking status..."
    
    # Check Head
    if docker ps | grep -q "$CONTAINER_NAME"; then
        echo "[HEAD] $HEAD_IP: Container '$CONTAINER_NAME' is RUNNING."
        if [[ "$NO_RAY_MODE" == "false" ]]; then
            echo "--- Ray Status ---"
            docker exec "$CONTAINER_NAME" ray status || echo "Failed to get ray status."
            echo "------------------"
        fi
    else
        echo "[HEAD] $HEAD_IP: Container '$CONTAINER_NAME' is NOT running."
    fi
    
    # Check Workers
    for worker in "${PEER_NODES[@]}"; do
        if ssh "$worker" "docker ps | grep -q '$CONTAINER_NAME'"; then
             echo "[WORKER] $worker: Container '$CONTAINER_NAME' is RUNNING."
        else
             echo "[WORKER] $worker: Container '$CONTAINER_NAME' is NOT running."
        fi
    done
    exit 0
fi

# Trap signals
# Only trap if we are NOT in daemon mode (container should persist in daemon mode)
if [[ "$DAEMON_MODE" == "false" ]]; then
    trap cleanup EXIT INT TERM HUP
fi

# Check if cluster is already running
check_cluster_running() {
    local running=false
    
    # Check Head
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Warning: Container '$CONTAINER_NAME' is already running on head node ($HEAD_IP)."
        running=true
    fi
    
    # Check Workers
    for worker in "${PEER_NODES[@]}"; do
        if ssh "$worker" "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'"; then
             echo "Warning: Container '$CONTAINER_NAME' is already running on worker node ($worker)."
             running=true
        fi
    done
    
    if [[ "$running" == "true" ]]; then
        echo "Cluster containers are already running. Skipping launch."
        CLUSTER_WAS_RUNNING="true"
        return 0
    fi
}

# Apply Mod Function
apply_mod_to_container() {
    local node_ip="$1"
    local container="$2"
    local is_local="$3" # true/false
    local mod_path="$4"
    local mod_type="$5"

    local mod_name=$(basename "$mod_path")
    if [[ "$mod_type" == "zip" ]]; then
        mod_name="${mod_name%.*}"
    fi

    echo "Applying mod '$mod_name' to $node_ip..."

    # 1. Copy mod to node (if remote)
    local target_mod_path=""
    local remote_cleanup_path=""

    if [[ "$is_local" == "true" ]]; then
        target_mod_path="$mod_path"
    else
        # SCP to remote
        local remote_tmp="/tmp/vllm_mod_pkg_$(date +%s)_$RANDOM"
        echo "  Copying mod package to $node_ip:$remote_tmp..."
        
        # Create directory first to ensure consistent path structure
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$node_ip" "mkdir -p $remote_tmp"
        remote_cleanup_path="$remote_tmp"
        
        if [[ "$mod_type" == "zip" ]]; then
             if ! scp -o BatchMode=yes -o StrictHostKeyChecking=no "$mod_path" "$node_ip:$remote_tmp/"; then
                echo "Error: Failed to copy mod to $node_ip"
                exit 1
             fi
             target_mod_path="$remote_tmp/$(basename "$mod_path")"
        else
             # Directory
             # Copy contents using wildcard to avoid creating a subdirectory
             if ! scp -r -o BatchMode=yes -o StrictHostKeyChecking=no "$mod_path"/* "$node_ip:$remote_tmp/"; then
                echo "Error: Failed to copy mod to $node_ip"
                exit 1
             fi
             target_mod_path="$remote_tmp"
        fi
    fi

    # 2. Copy into container
    local container_dest="/workspace/mods/$mod_name"
    
    # Command prefix for remote vs local
    local cmd_prefix=""
    if [[ "$is_local" == "false" ]]; then
        cmd_prefix="ssh -o BatchMode=yes -o StrictHostKeyChecking=no $node_ip"
    fi

    # Create workspace in container
    $cmd_prefix docker exec "$container" mkdir -p "$container_dest"

    if [[ "$mod_type" == "zip" ]]; then
        local zip_name=$(basename "$mod_path")
        echo "  Copying zip to container..."
        $cmd_prefix docker cp "$target_mod_path" "$container:$container_dest/$zip_name"
        
        # Unzip in container using python
        echo "  Extracting zip..."
        local py_unzip="import zipfile, sys; zipfile.ZipFile(sys.argv[1], 'r').extractall(sys.argv[2])"
        if [[ "$is_local" == "true" ]]; then
            docker exec "$container" python3 -c "$py_unzip" "$container_dest/$zip_name" "$container_dest"
        else
            $cmd_prefix docker exec "$container" python3 -c "\"$py_unzip\"" "$container_dest/$zip_name" "$container_dest"
        fi
    else
        # Directory
        echo "  Copying directory content to container..."
        if [[ "$is_local" == "true" ]]; then
             docker cp "$mod_path/." "$container:$container_dest/"
        else
             # For remote, we copied contents to $target_mod_path.
             # We want to copy contents of $target_mod_path to $container_dest.
             $cmd_prefix docker cp "$target_mod_path/." "$container:$container_dest/"
        fi
    fi

    # 3. Run run.sh
    echo "  Running patch script on $node_ip..."

    local local_exec_cmd="export WORKSPACE_DIR=\$PWD && cd $container_dest && chmod +x run.sh && ./run.sh"
    local remote_exec_cmd="export WORKSPACE_DIR=\\\$PWD && cd $container_dest && chmod +x run.sh && ./run.sh"
    local ret_code=0

    if [[ "$is_local" == "true" ]]; then
        docker exec "$container" bash -c "$local_exec_cmd"
        ret_code=$?
    else
        $cmd_prefix docker exec "$container" bash -c "\"$remote_exec_cmd\""
        ret_code=$?
    fi

    if [[ $ret_code -ne 0 ]]; then
        echo "Error: Patch script failed on $node_ip"
        # We should probably stop the cluster here or at least fail hard
        exit 1
    fi

    # 4. Cleanup remote temp
    if [[ "$is_local" == "false" ]]; then
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$node_ip" "rm -rf $remote_cleanup_path"
    fi
}

# Parse -tp/-pp/-dp (and long forms) from a text string (command or script content).
# Sets TP_SIZE, PP_SIZE, DP_SIZE, PARALLELISM_FOUND globals.
# Only acts when at least one parallelism flag is present.
parse_parallelism_from_text() {
    local text="$1"
    TP_SIZE=1; PP_SIZE=1; DP_SIZE=1
    PARALLELISM_FOUND=false

    # Normalize --flag=value to --flag value for uniform word-by-word parsing
    local normalized
    normalized=$(echo "$text" | sed 's/\(--[a-z-]*\)=/\1 /g')

    local prev=""
    for word in $normalized; do
        case "$prev" in
            -tp|--tensor-parallel-size)
                [[ "$word" =~ ^[0-9]+$ ]] && TP_SIZE="$word" && PARALLELISM_FOUND=true ;;
            -pp|--pipeline-parallel-size)
                [[ "$word" =~ ^[0-9]+$ ]] && PP_SIZE="$word" && PARALLELISM_FOUND=true ;;
            -dp|--data-parallel-size)
                [[ "$word" =~ ^[0-9]+$ ]] && DP_SIZE="$word" && PARALLELISM_FOUND=true ;;
        esac
        prev="$word"
    done
}

# Build a patched copy of the launch script on the host for a specific node.
# Strips --distributed-executor-backend and appends multi-node args.
# Prints the path of the temp file (caller must delete it).
make_node_script() {
    local script_path="$1"; local nnodes="$2"; local node_rank="$3"; local master_addr="$4"
    local extra="--nnodes $nnodes --node-rank $node_rank --master-addr $master_addr --master-port $MASTER_PORT"
    [[ "$node_rank" -gt 0 ]] && extra="$extra --headless"

    local tmp; tmp=$(mktemp /tmp/vllm_node_script_XXXXXX.sh)
    # Remove just the flag and its value (not the whole line), then filter empty/backslash-only lines
    sed 's/--distributed-executor-backend[[:space:]]*[^[:space:]]*//' "$script_path" | \
        grep -Ev '^[[:space:]\\]*$' > "$tmp"
    # Strip trailing backslash from last line before appending multi-node args
    sed -i "$ s/[[:space:]]*\\\\[[:space:]]*$//" "$tmp"
    sed -i "$ s/$/ $extra/" "$tmp"
    chmod +x "$tmp"
    echo "$tmp"
}

# Copy a script file into a local container as /workspace/exec-script.sh
copy_script_to_container() {
    local container="$1"; local script_path="$2"; local label="${3:-node}"
    echo "Copying launch script to $label..."
    docker cp "$script_path" "$container:/workspace/exec-script.sh" || { echo "Error: docker cp to $label failed"; exit 1; }
    docker exec "$container" chmod +x /workspace/exec-script.sh
}

# Copy a script file to a remote container via scp + docker cp
copy_script_to_worker() {
    local worker_ip="$1"; local container="$2"; local script_path="$3"
    echo "Copying launch script to worker $worker_ip..."
    local remote_tmp="/tmp/vllm_script_$(date +%s)_$RANDOM.sh"
    scp -o BatchMode=yes -o StrictHostKeyChecking=no "$script_path" "$worker_ip:$remote_tmp" || { echo "Error: scp to $worker_ip failed"; exit 1; }
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker_ip" \
        "docker cp $remote_tmp $container:/workspace/exec-script.sh && \
         docker exec $container chmod +x /workspace/exec-script.sh && \
         rm -f $remote_tmp" || { echo "Error: docker cp to worker $worker_ip failed"; exit 1; }
}

# Build -e KEY=VALUE flags for a given node IP (used in docker run and docker exec)
get_env_flags() {
    local node_ip="$1"
    printf -- '-e %s ' \
        "VLLM_HOST_IP=$node_ip" \
        "RAY_NODE_IP_ADDRESS=$node_ip" \
        "RAY_OVERRIDE_NODE_IP_ADDRESS=$node_ip" \
        "MN_IF_NAME=$ETH_IF" \
        "UCX_NET_DEVICES=$ETH_IF" \
        "NCCL_SOCKET_IFNAME=$ETH_IF" \
        "NCCL_IB_HCA=$IB_IF" \
        "NCCL_IB_DISABLE=0" \
        "OMPI_MCA_btl_tcp_if_include=$ETH_IF" \
        "GLOO_SOCKET_IFNAME=$ETH_IF" \
        "TP_SOCKET_IFNAME=$ETH_IF" \
        "RAY_memory_monitor_refresh_ms=0" \
        "RAY_num_prestart_python_workers=0" \
        "RAY_object_store_memory=1073741824"
}

# Start Ray head node inside the container
start_ray_head() {
    local container="$1"
    echo "Starting Ray HEAD node on $HEAD_IP..."
    docker exec -d "$container" bash -c \
        "ray start --block --head --port $MASTER_PORT --object-store-memory 1073741824 --num-cpus 2 \
         --node-ip-address $HEAD_IP --include-dashboard=false --disable-usage-stats \
         >> /proc/1/fd/1 2>&1"
}

# Start Ray worker node inside the container on a remote host
start_ray_worker() {
    local worker_ip="$1"; local container="$2"
    echo "Starting Ray WORKER node on $worker_ip..."
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker_ip" \
        "docker exec -d $container bash -c \
         'ray start --block --object-store-memory 1073741824 --num-cpus 2 --disable-usage-stats \
          --address=$HEAD_IP:$MASTER_PORT --node-ip-address $worker_ip >> /proc/1/fd/1 2>&1'"
}

# Start Cluster Function
start_cluster() {
    check_cluster_running

    if [[ "$CLUSTER_WAS_RUNNING" == "true" ]]; then
        return
    fi

    # Build docker run arguments based on mode
    local docker_args_common="--gpus all -d --rm --network host --name $CONTAINER_NAME $DOCKER_ARGS $IMAGE_NAME"
    local docker_caps_args=""
    local docker_resource_args=""

    if [[ "$NON_PRIVILEGED_MODE" == "true" ]]; then
        echo "Running in non-privileged mode..."
        docker_caps_args="--cap-add=IPC_LOCK"
        docker_resource_args="--shm-size=${SHM_SIZE_GB}g --device=/dev/infiniband --memory ${MEM_LIMIT_GB}g --memory-swap ${MEM_SWAP_LIMIT_GB}g --pids-limit ${PIDS_LIMIT}"
    else
        docker_caps_args="--privileged"
        docker_resource_args="--ipc=host"
    fi

    # Start Head Node
    echo "Starting Head Node on $HEAD_IP..."
    if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
        for dir in "${CACHE_DIRS_TO_CREATE[@]}"; do
            mkdir -p "$dir"
        done
    fi
    docker run $docker_caps_args $docker_resource_args \
        $(get_env_flags "$HEAD_IP") $docker_args_common sleep infinity

    # Start Worker Nodes
    for worker in "${PEER_NODES[@]}"; do
        echo "Starting Worker Node on $worker..."
        if [[ "$MOUNT_CACHE_DIRS" == "true" ]]; then
            ssh "$worker" "mkdir -p ${CACHE_DIRS_TO_CREATE[*]}"
        fi
        local docker_run_cmd="docker run $docker_caps_args $docker_resource_args $(get_env_flags "$worker") $docker_args_common"
        ssh "$worker" "$docker_run_cmd sleep infinity"
    done

    # Apply mods (containers are idle — no mod_done sync needed)
    if [[ ${#MOD_PATHS[@]} -gt 0 ]]; then
        echo "Applying modifications to cluster nodes..."
        for i in "${!MOD_PATHS[@]}"; do
            apply_mod_to_container "$HEAD_IP" "$CONTAINER_NAME" "true" "${MOD_PATHS[$i]}" "${MOD_TYPES[$i]}"
        done
        for worker in "${PEER_NODES[@]}"; do
            for i in "${!MOD_PATHS[@]}"; do
                apply_mod_to_container "$worker" "$CONTAINER_NAME" "false" "${MOD_PATHS[$i]}" "${MOD_TYPES[$i]}"
            done
        done
    fi

    # Copy (and patch for no-ray) launch script
    if [[ -n "$LAUNCH_SCRIPT_PATH" ]]; then
        local total_nodes=$(( 1 + ${#PEER_NODES[@]} ))
        if [[ "$NO_RAY_MODE" == "true" ]]; then
            # Build per-node patched scripts on the host, then copy
            local head_script; head_script=$(make_node_script "$LAUNCH_SCRIPT_PATH" "$total_nodes" "0" "$HEAD_IP")
            copy_script_to_container "$CONTAINER_NAME" "$head_script" "head node ($HEAD_IP)"
            rm -f "$head_script"

            local rank=1
            for worker in "${PEER_NODES[@]}"; do
                local worker_script; worker_script=$(make_node_script "$LAUNCH_SCRIPT_PATH" "$total_nodes" "$rank" "$HEAD_IP")
                copy_script_to_worker "$worker" "$CONTAINER_NAME" "$worker_script"
                rm -f "$worker_script"
                (( rank++ ))
            done
        else
            copy_script_to_container "$CONTAINER_NAME" "$LAUNCH_SCRIPT_PATH" "head node"
        fi
    fi

    # Start Ray cluster (unless solo or no-ray)
    if [[ "$SOLO_MODE" == "false" && "$NO_RAY_MODE" == "false" ]]; then
        start_ray_head "$CONTAINER_NAME"
        for worker in "${PEER_NODES[@]}"; do
            start_ray_worker "$worker" "$CONTAINER_NAME"
        done
        wait_for_cluster
    else
        sleep 2
    fi
}

# Wait for Cluster Readiness
wait_for_cluster() {
    echo "Waiting for cluster to be ready..."
    local retries=30
    local count=0
    
    while [[ $count -lt $retries ]]; do
        # Check if ray is responsive
        if docker exec "$CONTAINER_NAME" ray status >/dev/null 2>&1; then
             echo "Cluster head is responsive."
             # Give workers a moment to connect
             sleep 5
             return 0
        fi
        
        sleep 2
        ((count++))
    done
    
    echo "Timeout waiting for cluster to start."
    exit 1
}

# Execute command on head node (daemon or interactive)
_exec_on_head() {
    local cmd="$1"
    if [[ "$DAEMON_MODE" == "true" ]]; then
        docker exec -d "$CONTAINER_NAME" bash -c "$cmd >> /proc/1/fd/1 2>&1"
        echo "Command dispatched in background (Daemon mode). Container: $CONTAINER_NAME"
    else
        if [ -t 0 ]; then DOCKER_EXEC_FLAGS="-it"; else DOCKER_EXEC_FLAGS="-i"; fi
        docker exec $DOCKER_EXEC_FLAGS "$CONTAINER_NAME" bash -c "$cmd"
    fi
}

# Execute a no-ray multi-node command: workers (background) then head
exec_no_ray_cluster() {
    local base_cmd="$1"
    local total_nodes=$(( 1 + ${#PEER_NODES[@]} ))

    # Launch workers first (always background)
    local rank=1
    for worker in "${PEER_NODES[@]}"; do
        local worker_cmd
        if [[ "$LAUNCH_SCRIPT_MODE" == "true" ]]; then
            worker_cmd="$base_cmd"  # script already patched per-node in start_cluster()
        else
            local clean
            clean=$(echo "$base_cmd" | sed 's/--distributed-executor-backend[[:space:]]*[^[:space:]]*//')
            worker_cmd="$clean --nnodes $total_nodes --node-rank $rank --master-addr $HEAD_IP --master-port $MASTER_PORT --headless"
        fi
        echo "Launching worker (rank $rank) on $worker..."
        local remote_payload remote_cmd
        remote_payload="$worker_cmd >> /proc/1/fd/1 2>&1"
        printf -v remote_cmd 'docker exec -d %q bash -c %q' "$CONTAINER_NAME" "$remote_payload"
        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker" "$remote_cmd"
        (( rank++ ))
    done

    # Launch head (rank 0) last
    local head_cmd
    if [[ "$LAUNCH_SCRIPT_MODE" == "true" ]]; then
        head_cmd="$base_cmd"
    else
        local clean
        clean=$(echo "$base_cmd" | sed 's/--distributed-executor-backend[[:space:]]*[^[:space:]]*//')
        head_cmd="$clean --nnodes $total_nodes --node-rank 0 --master-addr $HEAD_IP --master-port $MASTER_PORT"
    fi

    echo "Executing command on head node (rank 0): $head_cmd"
    if [[ "$DAEMON_MODE" == "true" ]]; then
        docker exec -d "$CONTAINER_NAME" bash -c "$head_cmd >> /proc/1/fd/1 2>&1"
        echo "Command dispatched in background (Daemon mode). Container: $CONTAINER_NAME"
    else
        if [ -t 0 ]; then DOCKER_EXEC_FLAGS="-it"; else DOCKER_EXEC_FLAGS="-i"; fi
        docker exec $DOCKER_EXEC_FLAGS "$CONTAINER_NAME" bash -c "$head_cmd"
    fi
}

if [[ "$ACTION" == "exec" ]]; then
    # Trim (or error on) PEER_NODES based on declared parallelism, for any multi-node exec
    if [[ "$SOLO_MODE" != "true" && ${#PEER_NODES[@]} -gt 0 ]]; then
        if [[ "$LAUNCH_SCRIPT_MODE" == "true" ]]; then
            cmd_text=$(cat "$LAUNCH_SCRIPT_PATH" 2>/dev/null || true)
        else
            cmd_text="$COMMAND_TO_RUN"
        fi
        parse_parallelism_from_text "$cmd_text"

        if [[ "$PARALLELISM_FOUND" == "true" ]]; then
            required_nodes=$(( TP_SIZE * PP_SIZE * DP_SIZE ))
            total_nodes=$(( 1 + ${#PEER_NODES[@]} ))

            if [[ "$required_nodes" -gt "$total_nodes" ]]; then
                echo "Error: Command requires $required_nodes nodes (tp=$TP_SIZE * pp=$PP_SIZE * dp=$DP_SIZE) but only $total_nodes node(s) are configured."
                exit 1
            elif [[ "$required_nodes" -lt "$total_nodes" ]]; then
                echo "Note: Command requires $required_nodes node(s) (tp=$TP_SIZE * pp=$PP_SIZE * dp=$DP_SIZE); using $required_nodes of $total_nodes configured node(s)."
                PEER_NODES=("${PEER_NODES[@]:0:$(( required_nodes - 1 ))}")
            fi
        fi
    fi

    start_cluster
    echo "Executing command: $COMMAND_TO_RUN"

    if [[ "$NO_RAY_MODE" == "true" && ${#PEER_NODES[@]} -gt 0 ]]; then
        if [[ "$LAUNCH_SCRIPT_MODE" == "true" ]] || echo "$COMMAND_TO_RUN" | grep -q "vllm serve"; then
            exec_no_ray_cluster "$COMMAND_TO_RUN"
        else
            _exec_on_head "$COMMAND_TO_RUN"
        fi
    else
        _exec_on_head "$COMMAND_TO_RUN"
    fi
elif [[ "$ACTION" == "start" ]]; then
    start_cluster
    if [[ "$DAEMON_MODE" == "true" ]]; then
        echo "Cluster started in background (Daemon mode)."
    else
        echo "Cluster started. Tailing logs from head node..."
        echo "Press Ctrl+C to stop the cluster."
        docker logs -f "$CONTAINER_NAME" &
        wait $!
    fi
fi
