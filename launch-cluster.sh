#!/bin/bash

# Default Configuration
IMAGE_NAME="vllm-node"
DEFAULT_CONTAINER_NAME="vllm_node"
# ETH_IF and IB_IF will be auto-detected if not provided
ETH_IF=""
IB_IF=""

# Initialize variables
NODES_ARG=""
CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
COMMAND_TO_RUN=""
DAEMON_MODE="false"
CHECK_CONFIG="false"
ACTION="start"

# Function to print usage
usage() {
    echo "Usage: $0 [-n <node_ips>] [-t <image_name>] [--name <container_name>] [--eth-if <if_name>] [--ib-if <if_name>] [--check-config] [-d] [action] [command]"
    echo "  -n, --nodes     Comma-separated list of node IPs (Optional, auto-detected if omitted)"
    echo "  -t              Docker image name (Optional, default: $IMAGE_NAME)"
    echo "  --name          Container name (Optional, default: $DEFAULT_CONTAINER_NAME)"
    echo "  --eth-if        Ethernet interface (Optional, auto-detected)"
    echo "  --ib-if         InfiniBand interface (Optional, auto-detected)"
    echo "  --check-config  Check configuration and auto-detection without launching"
    echo "  -d              Daemon mode (only for 'start' action)"
    echo "  action          start | stop | status | exec (Default: start)"
    echo "  command         Command to run (only for 'exec' action)"
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

# --- Auto-Detection Logic ---

# Check for required tools if auto-detection is needed
if [[ -z "$ETH_IF" || -z "$IB_IF" || -z "$NODES_ARG" ]]; then
    if ! command -v ibdev2netdev &> /dev/null; then
        echo "Error: ibdev2netdev not found. Cannot auto-detect interfaces."
        exit 1
    fi
fi

# 1. Detect Interfaces (ETH_IF and IB_IF)
if [[ -z "$ETH_IF" || -z "$IB_IF" ]]; then
    echo "Auto-detecting interfaces..."
    
    # Get all Up interfaces: "rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)"
    # We capture: IB_DEV, NET_DEV
    mapfile -t IB_NET_PAIRS < <(ibdev2netdev | awk '/Up\)/ {print $1 " " $5}')
    
    if [ ${#IB_NET_PAIRS[@]} -eq 0 ]; then
        echo "Error: No active IB interfaces found."
        exit 1
    fi

    DETECTED_IB_IFS=()
    CANDIDATE_ETH_IFS=()

    for pair in "${IB_NET_PAIRS[@]}"; do
        ib_dev=$(echo "$pair" | awk '{print $1}')
        net_dev=$(echo "$pair" | awk '{print $2}')
        
        DETECTED_IB_IFS+=("$ib_dev")
        
        # Check if interface has an IP address
        if ip addr show "$net_dev" | grep -q "inet "; then
            CANDIDATE_ETH_IFS+=("$net_dev")
        fi
    done

    # Set IB_IF if not provided
    if [[ -z "$IB_IF" ]]; then
        IB_IF=$(IFS=,; echo "${DETECTED_IB_IFS[*]}")
        echo "  Detected IB_IF: $IB_IF"
    fi

    # Set ETH_IF if not provided
    if [[ -z "$ETH_IF" ]]; then
        if [ ${#CANDIDATE_ETH_IFS[@]} -eq 0 ]; then
            echo "Error: No active IB-associated interfaces have IP addresses."
            exit 1
        fi
        
        # Selection logic: Prefer interface without capital 'P'
        SELECTED_ETH=""
        for iface in "${CANDIDATE_ETH_IFS[@]}"; do
            if [[ "$iface" != *"P"* ]]; then
                SELECTED_ETH="$iface"
                break
            fi
        done
        
        # Fallback: Use the first one if all have 'P' or none found yet
        if [[ -z "$SELECTED_ETH" ]]; then
            SELECTED_ETH="${CANDIDATE_ETH_IFS[0]}"
        fi
        
        ETH_IF="$SELECTED_ETH"
        echo "  Detected ETH_IF: $ETH_IF"
    fi
fi

# 2. Detect Nodes if not provided
if [[ -z "$NODES_ARG" ]]; then
    echo "Auto-detecting nodes..."
    
    if ! command -v nc &> /dev/null; then
        echo "Error: nc (netcat) not found. Please install netcat."
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 not found. Please install python3."
        exit 1
    fi

    # Get CIDR of the selected ETH_IF
    CIDR=$(ip -o -f inet addr show "$ETH_IF" | awk '{print $4}' | head -n 1)
    
    if [[ -z "$CIDR" ]]; then
        echo "Error: Could not determine IP/CIDR for interface $ETH_IF"
        exit 1
    fi
    
    LOCAL_IP=${CIDR%/*}
    echo "  Detected Local IP: $LOCAL_IP ($CIDR)"

    DETECTED_IPS=("$LOCAL_IP")
    
    echo "  Scanning for SSH peers on $CIDR..."
    
    # Generate list of IPs using python
    ALL_IPS=$(python3 -c "import ipaddress, sys; [print(ip) for ip in ipaddress.ip_network(sys.argv[1], strict=False).hosts()]" "$CIDR")
    
    TEMP_IPS_FILE=$(mktemp)
    
    # Scan in parallel
    for ip in $ALL_IPS; do
        # Skip own IP
        if [[ "$ip" == "$LOCAL_IP" ]]; then continue; fi
        
        (
            # Check port 22 with 1 second timeout
            if nc -z -w 1 "$ip" 22 &>/dev/null; then
                echo "$ip" >> "$TEMP_IPS_FILE"
            fi
        ) &
    done
    
    # Wait for all background scans to complete
    wait
    
    # Read found IPs
    if [[ -f "$TEMP_IPS_FILE" ]]; then
        while read -r ip; do
             DETECTED_IPS+=("$ip")
             echo "  Found peer: $ip"
        done < "$TEMP_IPS_FILE"
        rm -f "$TEMP_IPS_FILE"
    fi
    
    # Sort IPs
    IFS=$'\n' SORTED_IPS=($(sort <<<"${DETECTED_IPS[*]}"))
    unset IFS
    
    NODES_ARG=$(IFS=,; echo "${SORTED_IPS[*]}")
    echo "  Cluster Nodes: $NODES_ARG"
fi

if [[ -z "$NODES_ARG" ]]; then
    echo "Error: Nodes argument (-n) is mandatory or could not be auto-detected."
    usage
fi

# Split nodes into array
IFS=',' read -r -a ALL_NODES <<< "$NODES_ARG"

# Detect Head IP (Local IP)
HEAD_IP=""
LOCAL_IPS=$(hostname -I)
for ip in "${ALL_NODES[@]}"; do
    # Trim whitespace
    ip=$(echo "$ip" | xargs)
    if [[ " $LOCAL_IPS " =~ " $ip " ]]; then
        HEAD_IP="$ip"
        break
    fi
done

if [[ -z "$HEAD_IP" ]]; then
    echo "Error: Could not determine Head IP. This script must be run on one of the nodes specified in -n."
    exit 1
fi

# Identify Worker Nodes
WORKER_NODES=()
for ip in "${ALL_NODES[@]}"; do
    ip=$(echo "$ip" | xargs)
    if [[ "$ip" != "$HEAD_IP" ]]; then
        WORKER_NODES+=("$ip")
    fi
done

echo "Head Node: $HEAD_IP"
echo "Worker Nodes: ${WORKER_NODES[*]}"
echo "Container Name: $CONTAINER_NAME"
echo "Action: $ACTION"

# Check SSH connectivity to worker nodes
if [[ "$ACTION" == "start" || "$ACTION" == "exec" || "$CHECK_CONFIG" == "true" ]]; then
    if [ ${#WORKER_NODES[@]} -gt 0 ]; then
        echo "Checking SSH connectivity to worker nodes..."
        for worker in "${WORKER_NODES[@]}"; do
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

    echo ""
    echo "Stopping cluster..."
    
    # Stop Head
    echo "Stopping head node ($HEAD_IP)..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    
    # Stop Workers
    for worker in "${WORKER_NODES[@]}"; do
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
    for worker in "${WORKER_NODES[@]}"; do
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
    for worker in "${WORKER_NODES[@]}"; do
        if ssh "$worker" "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'"; then
             echo "Warning: Container '$CONTAINER_NAME' is already running on worker node ($worker)."
             running=true
        fi
    done
    
    if [[ "$running" == "true" ]]; then
        echo "Cluster containers are already running. Please stop them first or use a different name."
        exit 1
    fi
}

# Start Cluster Function
start_cluster() {
    check_cluster_running

    # Start Head Node
    echo "Starting Head Node on $HEAD_IP..."
    docker run -d --privileged --gpus all --rm \
        --ipc=host --network host \
        --name "$CONTAINER_NAME" \
        -e NCCL_DEBUG=INFO -e NCCL_IGNORE_CPU_AFFINITY=1 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        "$IMAGE_NAME" \
        ./run-cluster-node.sh \
        --role head \
        --host-ip "$HEAD_IP" \
        --eth-if "$ETH_IF" \
        --ib-if "$IB_IF"

    # Start Worker Nodes
    for worker in "${WORKER_NODES[@]}"; do
        echo "Starting Worker Node on $worker..."
        ssh "$worker" "docker run -d --privileged --gpus all --rm \
            --ipc=host --network host \
            --name $CONTAINER_NAME \
            -e NCCL_DEBUG=INFO -e NCCL_IGNORE_CPU_AFFINITY=1 \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            $IMAGE_NAME \
            ./run-cluster-node.sh \
            --role node \
            --host-ip $worker \
            --eth-if $ETH_IF \
            --ib-if $IB_IF \
            --head-ip $HEAD_IP"
    done

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
    docker exec -it "$CONTAINER_NAME" bash -i -c "$COMMAND_TO_RUN"
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
