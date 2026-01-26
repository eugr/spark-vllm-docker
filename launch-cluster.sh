#!/bin/bash

# Default Configuration
IMAGE_NAME="vllm-node"
DEFAULT_CONTAINER_NAME="vllm_node"
# Modify these if you want to pass additional docker args or set VLLM_SPARK_EXTRA_DOCKER_ARGS variable
DOCKER_ARGS="-e NCCL_IGNORE_CPU_AFFINITY=1 -v $HOME/.cache/huggingface:/root/.cache/huggingface"

# Append additional arguments from environment variable
if [[ -n "$VLLM_SPARK_EXTRA_DOCKER_ARGS" ]]; then
    DOCKER_ARGS="$DOCKER_ARGS $VLLM_SPARK_EXTRA_DOCKER_ARGS"
fi

# ETH_IF and IB_IF will be auto-detected if not provided
ETH_IF=""
IB_IF=""
NCCL_DEBUG_VAL=""

# Initialize variables
NODES_ARG=""
CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
COMMAND_TO_RUN=""
DAEMON_MODE="false"
CHECK_CONFIG="false"
ACTION="start"
CLUSTER_WAS_RUNNING="false"
MOD_PATHS=()
MOD_TYPES=()
PROFILE_PATH=""
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Function to print usage
usage() {
    echo "Usage: $0 [-n <node_ips>] [-t <image_name>] [--name <container_name>] [--eth-if <if_name>] [--ib-if <if_name>] [--nccl-debug <level>] [--check-config] [-d] [action] [command]"
    echo "  -n, --nodes     Comma-separated list of node IPs (Optional, auto-detected if omitted)"
    echo "  -t              Docker image name (Optional, default: $IMAGE_NAME)"
    echo "  --name          Container name (Optional, default: $DEFAULT_CONTAINER_NAME)"
    echo "  --eth-if        Ethernet interface (Optional, auto-detected)"
    echo "  --ib-if         InfiniBand interface (Optional, auto-detected)"
    echo "  --nccl-debug    NCCL debug level (Optional, one of: VERSION, WARN, INFO, TRACE). If no level is provided, defaults to INFO."
    echo "  --apply-mod     Path to directory or zip file containing run.sh to apply before launch (Can be specified multiple times)"
    echo "  --profile       Path to bash profile script or profile name (from profiles/ directory)"
    echo "  --check-config  Check configuration and auto-detection without launching"
    echo "  -d              Daemon mode (only for 'start' action)"
    echo "  action          start | stop | status | exec (Default: start)"
    echo "  command         Command to run (only for 'exec' action, ignored if --profile is used)"
    echo ""
    echo "Profile Usage:"
    echo "  $0 --profile my-profile.sh exec       # Uses profile from profiles/ directory"
    echo "  $0 --profile /path/to/profile.sh exec # Uses absolute path to profile"
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
        --apply-mod) MOD_PATHS+=("$2"); shift ;;
        --profile) PROFILE_PATH="$2"; shift ;;
        --nccl-debug)
            if [[ -n "$2" && "$2" =~ ^(VERSION|WARN|INFO|TRACE)$ ]]; then
                NCCL_DEBUG_VAL="$2"
                shift
            else
                NCCL_DEBUG_VAL="INFO"
            fi
            ;;
        --check-config) CHECK_CONFIG="true" ;;
        -d) DAEMON_MODE="true" ;;
        -h|--help) usage ;;
        start|stop|status) 
            ACTION="$1" 
            ;;
        exec)
            ACTION="exec"
            shift
            COMMAND_TO_RUN="$@"
            break
            ;;
        *) 
            # If it's not a flag and not a known action, treat as exec command for backward compatibility
            # unless it's the default 'start' implied.
            # However, to support "omitted" = start, we need to be careful.
            # If the arg looks like a command, it's exec.
            ACTION="exec"
            COMMAND_TO_RUN="$@"
            break 
            ;;
    esac
    shift
done

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

# Resolve profile path if specified
if [[ -n "$PROFILE_PATH" ]]; then
    # Check if it's an absolute path or relative path that exists
    if [[ -f "$PROFILE_PATH" ]]; then
        PROFILE_PATH=$(realpath "$PROFILE_PATH")
    # Check if it's just a filename, look in profiles/ directory
    elif [[ -f "$SCRIPT_DIR/profiles/$PROFILE_PATH" ]]; then
        PROFILE_PATH="$SCRIPT_DIR/profiles/$PROFILE_PATH"
    # Check if it's a name without .sh extension
    elif [[ -f "$SCRIPT_DIR/profiles/${PROFILE_PATH}.sh" ]]; then
        PROFILE_PATH="$SCRIPT_DIR/profiles/${PROFILE_PATH}.sh"
    else
        echo "Error: Profile '$PROFILE_PATH' not found."
        echo "Searched in:"
        echo "  - $PROFILE_PATH"
        echo "  - $SCRIPT_DIR/profiles/$PROFILE_PATH"
        echo "  - $SCRIPT_DIR/profiles/${PROFILE_PATH}.sh"
        exit 1
    fi
    
    echo "Using profile: $PROFILE_PATH"
    
    # Source the profile script
    # shellcheck disable=SC1090
    source "$PROFILE_PATH" || { echo "Error: Failed to source profile '$PROFILE_PATH'." >&2; exit 1; }
    
    # Validate required variables
    [[ -z "$PROFILE_RUNTIME" ]] && { echo "Error: Profile must define PROFILE_RUNTIME." >&2; exit 1; }
    
    # Call profile_init if defined
    if declare -f profile_init > /dev/null; then
        echo "Running profile initialization..."
        profile_init
    fi
    
    # Build command string
    COMMAND_TO_RUN="$PROFILE_RUNTIME ${PROFILE_COMMAND:-} ${PROFILE_MODEL:-} ${PROFILE_ARGS[*]}"
    echo "Profile command: $COMMAND_TO_RUN"
    
    # Append profile mods to MOD_PATHS
    if [[ ${#PROFILE_MODS[@]} -gt 0 ]]; then
        for mod in "${PROFILE_MODS[@]}"; do
            # Resolve relative paths from SCRIPT_DIR
            if [[ ! -e "$mod" && -e "$SCRIPT_DIR/$mod" ]]; then
                mod="$SCRIPT_DIR/$mod"
            fi
            MOD_PATHS+=("$mod")
        done
        echo "Profile mods: ${PROFILE_MODS[*]}"
    fi
    
    # If profile is specified, default action to exec unless explicitly set to stop/status
    if [[ "$ACTION" == "start" ]]; then
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

# Perform auto-detection
detect_interfaces || exit 1
detect_nodes || exit 1

if [[ -z "$NODES_ARG" ]]; then
    echo "Error: Nodes argument (-n) is mandatory or could not be auto-detected."
    usage
fi

# Split nodes into array
IFS=',' read -r -a ALL_NODES <<< "$NODES_ARG"

# Detect Head IP (Local IP)
detect_local_ip || exit 1
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

if [[ "$CHECK_CONFIG" == "true" ]]; then
    echo "Configuration Check Complete."
    echo "  Image Name: $IMAGE_NAME"
    echo "  ETH Interface: $ETH_IF"
    echo "  IB Interface: $IB_IF"
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
        echo "--- Ray Status ---"
        docker exec "$CONTAINER_NAME" ray status || echo "Failed to get ray status."
        echo "------------------"
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
# Only trap if we are NOT in daemon mode, OR if we are in exec mode (always cleanup after exec)
if [[ "$DAEMON_MODE" == "false" ]] || [[ "$ACTION" == "exec" ]]; then
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

    local exec_cmd="cd $container_dest && chmod +x run.sh && ./run.sh"
    local ret_code=0

    if [[ "$is_local" == "true" ]]; then
        docker exec "$container" bash -c "$exec_cmd"
        ret_code=$?
    else
        $cmd_prefix docker exec "$container" bash -c "\"$exec_cmd\""
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

# Start Cluster Function
start_cluster() {
    check_cluster_running

    if [[ "$CLUSTER_WAS_RUNNING" == "true" ]]; then
        return
    fi

    # Start Head Node
    echo "Starting Head Node on $HEAD_IP..."
    
    local head_cmd_args=()
    if [[ ${#MOD_PATHS[@]} -gt 0 ]]; then
        head_cmd_args=(bash -c "echo Waiting for mod application...; while [ ! -f /tmp/mod_done ]; do sleep 1; done; echo Mod applied, starting node...; exec ./run-cluster-node.sh --role head --host-ip $HEAD_IP --eth-if $ETH_IF --ib-if $IB_IF")
    else
        head_cmd_args=(./run-cluster-node.sh --role head --host-ip "$HEAD_IP" --eth-if "$ETH_IF" --ib-if "$IB_IF")
    fi

    docker run -d --privileged --gpus all --rm \
        --ipc=host --network host \
        --name "$CONTAINER_NAME" \
        $DOCKER_ARGS \
        "$IMAGE_NAME" \
        "${head_cmd_args[@]}"

    # Start Worker Nodes
    for worker in "${PEER_NODES[@]}"; do
        echo "Starting Worker Node on $worker..."
        
        local docker_run_cmd="docker run -d --privileged --gpus all --rm --ipc=host --network host --name $CONTAINER_NAME $DOCKER_ARGS $IMAGE_NAME"
        
        if [[ ${#MOD_PATHS[@]} -gt 0 ]]; then
            local inner_script="echo Waiting for mod application...; while [ ! -f /tmp/mod_done ]; do sleep 1; done; echo Mod applied, starting node...; exec ./run-cluster-node.sh --role node --host-ip $worker --eth-if $ETH_IF --ib-if $IB_IF --head-ip $HEAD_IP"
            ssh "$worker" "$docker_run_cmd bash -c \"$inner_script\""
        else
            ssh "$worker" "$docker_run_cmd ./run-cluster-node.sh --role node --host-ip $worker --eth-if $ETH_IF --ib-if $IB_IF --head-ip $HEAD_IP"
        fi
    done

    # Apply mods if requested
    if [[ ${#MOD_PATHS[@]} -gt 0 ]]; then
        echo "Applying modifications to cluster nodes..."
        
        # Apply to Head
        for i in "${!MOD_PATHS[@]}"; do
            apply_mod_to_container "$HEAD_IP" "$CONTAINER_NAME" "true" "${MOD_PATHS[$i]}" "${MOD_TYPES[$i]}"
        done
        # Signal completion on Head
        docker exec "$CONTAINER_NAME" touch /tmp/mod_done
        
        # Apply to Workers
        for worker in "${PEER_NODES[@]}"; do
            for i in "${!MOD_PATHS[@]}"; do
                apply_mod_to_container "$worker" "$CONTAINER_NAME" "false" "${MOD_PATHS[$i]}" "${MOD_TYPES[$i]}"
            done
            # Signal completion on Worker
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$worker" "docker exec $CONTAINER_NAME touch /tmp/mod_done"
        done
    fi

    wait_for_cluster
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

if [[ "$ACTION" == "exec" ]]; then
    start_cluster
    echo "Executing command on head node: $COMMAND_TO_RUN"
    
    # Check if running in a TTY to avoid "input device is not a TTY" error
    if [ -t 0 ]; then
        DOCKER_EXEC_FLAGS="-it"
    else
        DOCKER_EXEC_FLAGS="-i"
    fi
    
    docker exec $DOCKER_EXEC_FLAGS "$CONTAINER_NAME" bash -i -c "$COMMAND_TO_RUN"
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
