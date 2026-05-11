#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# Load .env file if exists (for shared configuration)
# This is called early so that DOTENV_* variables are available to all functions
load_env_if_exists() {
    local env_file="${CONFIG_FILE:-}"
    local config_explicit="${CONFIG_FILE_SET:-false}"

    # If CONFIG_FILE is not set, check default location
    if [[ -z "$env_file" ]]; then
        env_file="$SCRIPT_DIR/.env"
        config_explicit="false"
    fi

    # Validate config file exists if explicitly specified
    # Exception: if --setup is also specified, the file will be created by the setup procedure
    if [[ "$config_explicit" == "true" ]] && [[ ! -f "$env_file" ]] && [[ "${FORCE_DISCOVER:-false}" != "true" ]]; then
        echo "Error: Config file not found: $env_file"
        exit 1
    fi

    if [[ -f "$env_file" ]]; then
        # Load .env variables with DOTENV_ prefix
        while IFS='=' read -r key value || [[ -n "$key" ]]; do
            # Skip comments and empty lines
            [[ "$key" =~ ^[[:space:]]*# ]] && continue
            [[ -z "$key" ]] && continue

            # Remove leading/trailing whitespace from key
            key=$(echo "$key" | xargs)

            # Skip if key is empty after trimming
            [[ -z "$key" ]] && continue

            # Remove quotes from value
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"

            # Export with DOTENV_ prefix
            export "DOTENV_$key=$value"
        done < "$env_file"
    fi
}

# Load .env file
load_env_if_exists

# Mesh mode flag (set by detect_interfaces)
MESH_MODE="false"

# Function to detect IB and Ethernet interfaces
detect_interfaces() {
    # If both interfaces are already set, nothing to do
    if [[ -n "$ETH_IF" && -n "$IB_IF" ]]; then
        return 0
    fi

    # Check for required tools
    if ! command -v ibdev2netdev &> /dev/null; then
        echo "Error: ibdev2netdev not found. Cannot auto-detect interfaces."
        return 1
    fi

    echo "Auto-detecting interfaces..."

    # Get all Up interfaces: "rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)"
    # We capture: IB_DEV, NET_DEV
    mapfile -t IB_NET_PAIRS < <(ibdev2netdev | awk '/Up\)/ {print $1 " " $5}')

    if [ ${#IB_NET_PAIRS[@]} -eq 0 ]; then
        echo "Error: No active IB interfaces found."
        return 1
    fi

    DETECTED_IB_IFS=()
    ALL_NET_IFS=()

    for pair in "${IB_NET_PAIRS[@]}"; do
        ib_dev=$(echo "$pair" | awk '{print $1}')
        net_dev=$(echo "$pair" | awk '{print $2}')
        DETECTED_IB_IFS+=("$ib_dev")
        ALL_NET_IFS+=("$net_dev")
    done

    local num_up="${#IB_NET_PAIRS[@]}"

    # --- Sanity checks ---

    # 1. enp* (no capital P) interfaces MUST have an IP
    for net_dev in "${ALL_NET_IFS[@]}"; do
        if [[ "$net_dev" =~ ^enp[^P] ]] || [[ "$net_dev" == enp* && "$net_dev" != *P* ]]; then
            if ! ip addr show "$net_dev" 2>/dev/null | grep -q "inet "; then
                echo "Error: Interface $net_dev (enp*, no capital P) is Up but has no IP address assigned."
                return 1
            fi
        fi
    done

    # 2. No two interfaces with IPs should share the same subnet
    declare -A SEEN_SUBNETS
    for net_dev in "${ALL_NET_IFS[@]}"; do
        local cidr
        cidr=$(ip -o -f inet addr show "$net_dev" 2>/dev/null | awk '{print $4}' | head -n1)
        [[ -z "$cidr" ]] && continue
        # Compute network address using python3
        local net_addr
        net_addr=$(python3 -c "import ipaddress; print(str(ipaddress.ip_network('$cidr', strict=False)))" 2>/dev/null)
        if [[ -n "${SEEN_SUBNETS[$net_addr]}" ]]; then
            echo "Error: Interfaces $net_dev and ${SEEN_SUBNETS[$net_addr]} share the same subnet ($net_addr). Check network configuration."
            return 1
        fi
        SEEN_SUBNETS["$net_addr"]="$net_dev"
    done

    # --- Mode selection ---

    if [[ "$num_up" -eq 2 ]]; then
        # Non-mesh configuration
        MESH_MODE="false"
        echo "  Non-mesh mode: 2 CX7 interfaces active."

        # Set IB_IF if not provided
        if [[ -z "$IB_IF" ]]; then
            IB_IF=$(IFS=,; echo "${DETECTED_IB_IFS[*]}")
            echo "  Detected IB_IF: $IB_IF"
        fi

        # Set ETH_IF if not provided: prefer interface without capital 'P'
        if [[ -z "$ETH_IF" ]]; then
            local selected_eth=""
            for net_dev in "${ALL_NET_IFS[@]}"; do
                if ip addr show "$net_dev" 2>/dev/null | grep -q "inet "; then
                    if [[ "$net_dev" != *P* ]]; then
                        selected_eth="$net_dev"
                        break
                    fi
                fi
            done
            # Fallback: first interface with an IP
            if [[ -z "$selected_eth" ]]; then
                for net_dev in "${ALL_NET_IFS[@]}"; do
                    if ip addr show "$net_dev" 2>/dev/null | grep -q "inet "; then
                        selected_eth="$net_dev"
                        break
                    fi
                done
            fi
            if [[ -z "$selected_eth" ]]; then
                echo "Error: No active IB-associated interfaces have IP addresses."
                return 1
            fi
            ETH_IF="$selected_eth"
            echo "  Detected ETH_IF: $ETH_IF"
        fi

    elif [[ "$num_up" -eq 4 ]]; then
        # Mesh configuration
        MESH_MODE="true"
        echo "  Mesh mode: all 4 CX7 interfaces active."

        # Set IB_IF to all four RoCE interfaces (hardcoded for mesh)
        if [[ -z "$IB_IF" ]]; then
            IB_IF="rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1"
            echo "  Detected IB_IF: $IB_IF"
        fi

        # Set ETH_IF: check enP7s7 first, then wlP9s9
        if [[ -z "$ETH_IF" ]]; then
            if ip addr show enP7s7 2>/dev/null | grep -q "inet "; then
                ETH_IF="enP7s7"
                echo "  Detected ETH_IF: $ETH_IF"
            elif ip addr show wlP9s9 2>/dev/null | grep -q "inet "; then
                ETH_IF="wlP9s9"
                echo "  Detected ETH_IF: $ETH_IF"
                echo "  Warning: using wireless interface (wlP9s9) for cluster coordination. Performance may be limited."
            else
                echo "Error: Mesh mode requires enP7s7 or wlP9s9 to be up with an IP address for cluster coordination."
                return 1
            fi
        fi

        # Export mesh NCCL settings directly so launch-cluster.sh picks them up
        # even if the user declines to save config to .env
        export DOTENV_CONTAINER_NCCL_NET_PLUGIN=none
        export DOTENV_CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING=1
        export DOTENV_CONTAINER_NCCL_IB_MERGE_NICS=0

    else
        echo "Error: Unexpected number of active CX7 interfaces ($num_up). Expected 2 (non-mesh) or 4 (mesh)."
        return 1
    fi
}

# Function to detect local IP
detect_local_ip() {
    if [[ -n "$LOCAL_IP" ]]; then
        return 0
    fi

    # Ensure interface is detected if not provided
    if [[ -z "$ETH_IF" ]]; then
        detect_interfaces || return 1
    fi

    # Get CIDR of the selected ETH_IF
    CIDR=$(ip -o -f inet addr show "$ETH_IF" | awk '{print $4}' | head -n 1)

    if [[ -z "$CIDR" ]]; then
        echo "Error: Could not determine IP/CIDR for interface $ETH_IF"
        return 1
    fi

    LOCAL_IP=${CIDR%/*}
    echo "  Detected Local IP: $LOCAL_IP ($CIDR)"
}

# Scan a subnet for GB10-capable peers via SSH
# Usage: _scan_subnet_for_gb10 <cidr> <local_ip_to_exclude> <output_file>
_scan_subnet_for_gb10() {
    local cidr="$1"
    local exclude_ip="$2"
    local out_file="$3"

    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 not found."
        return 1
    fi
    if ! command -v nc &> /dev/null; then
        echo "Error: nc (netcat) not found."
        return 1
    fi

    local all_ips
    all_ips=$(python3 -c "import ipaddress, sys; [print(ip) for ip in ipaddress.ip_network(sys.argv[1], strict=False).hosts()]" "$cidr")

    for ip in $all_ips; do
        [[ "$ip" == "$exclude_ip" ]] && continue
        (
            if nc -z -w 1 "$ip" 22 &>/dev/null; then
                # Check if remote is a GB10 system
                if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -o BatchMode=yes "$ip" \
                    "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null" \
                    2>/dev/null | grep -q "NVIDIA GB10"; then
                    echo "$ip" >> "$out_file"
                fi
            fi
        ) &
    done
    wait
}

# Function to detect cluster nodes
detect_nodes() {
    detect_local_ip || return 1

    # If nodes are already set, populate PEER_NODES and return
    if [[ -n "$NODES_ARG" ]]; then
        PEER_NODES=()
        IFS=',' read -ra ALL_NODES <<< "$NODES_ARG"
        for node in "${ALL_NODES[@]}"; do
            node=$(echo "$node" | xargs)
            if [[ "$node" != "$LOCAL_IP" ]]; then
                PEER_NODES+=("$node")
            fi
        done
        return 0
    fi

    # Try to use CLUSTER_NODES from .env
    if [[ -n "$DOTENV_CLUSTER_NODES" ]]; then
        echo "  Using CLUSTER_NODES from .env: $DOTENV_CLUSTER_NODES"
        PEER_NODES=()
        IFS=',' read -ra ALL_NODES <<< "$DOTENV_CLUSTER_NODES"
        for node in "${ALL_NODES[@]}"; do
            node=$(echo "$node" | xargs)
            [[ "$node" != "$LOCAL_IP" ]] && PEER_NODES+=("$node")
        done
        NODES_ARG="$DOTENV_CLUSTER_NODES"
        return 0
    fi

    echo "Auto-detecting nodes on $CIDR (checking for NVIDIA GB10)..."

    local temp_file
    temp_file=$(mktemp)

    _scan_subnet_for_gb10 "$CIDR" "$LOCAL_IP" "$temp_file"

    PEER_NODES=()
    local detected_ips=("$LOCAL_IP")
    if [[ -f "$temp_file" ]]; then
        while read -r ip; do
            PEER_NODES+=("$ip")
            detected_ips+=("$ip")
            echo "  Found GB10 peer: $ip"
        done < <(sort "$temp_file")
        rm -f "$temp_file"
    fi

    # Sort and set NODES_ARG
    IFS=$'\n' SORTED_IPS=($(sort <<<"${detected_ips[*]}"))
    unset IFS
    NODES_ARG=$(IFS=,; echo "${SORTED_IPS[*]}")
    echo "  Cluster Nodes: $NODES_ARG"
}

# Function to detect COPY_HOSTS for build/model distribution
# In non-mesh mode: COPY_PEER_NODES = PEER_NODES (same network)
# In mesh mode: scan enp* interfaces (direct IB-attached) for GB10 peers
detect_copy_hosts() {
    if [[ "$MESH_MODE" == "false" ]]; then
        COPY_PEER_NODES=("${PEER_NODES[@]}")
        return 0
    fi

    # Mesh mode: scan enp1s0f0np0 and enp1s0f1np1 subnets
    echo "Auto-detecting COPY_HOSTS on direct IB interfaces (mesh mode)..."

    local temp_file
    temp_file=$(mktemp)

    for iface in enp1s0f0np0 enp1s0f1np1; do
        local cidr
        cidr=$(ip -o -f inet addr show "$iface" 2>/dev/null | awk '{print $4}' | head -n1)
        [[ -z "$cidr" ]] && continue
        local local_iface_ip="${cidr%/*}"
        echo "  Scanning $iface ($cidr)..."
        _scan_subnet_for_gb10 "$cidr" "$local_iface_ip" "$temp_file"
    done

    # Deduplicate and collect results.
    # On two-cable setups two IB IPs may belong to the same host; deduplicate by
    # querying each host's ETH_IF IP as a canonical identity.
    COPY_PEER_NODES=()
    declare -A _SEEN_COPY   # keyed by IB IP
    declare -A _SEEN_HOST   # keyed by ETH_IF IP → first IB IP seen for that host
    if [[ -f "$temp_file" ]]; then
        while read -r ip; do
            [[ -n "${_SEEN_COPY[$ip]}" ]] && continue
            _SEEN_COPY["$ip"]=1
            # Resolve canonical host identity via ETH_IF IP
            local host_ip
            host_ip=$(ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -o BatchMode=yes "$ip" \
                "ip -o -f inet addr show $ETH_IF 2>/dev/null | awk '{print \$4}' | head -n1 | cut -d/ -f1" \
                </dev/null 2>/dev/null)
            if [[ -n "$host_ip" && -n "${_SEEN_HOST[$host_ip]}" ]]; then
                echo "  Skipping $ip (same host as ${_SEEN_HOST[$host_ip]}, ETH_IF: $host_ip)"
                continue
            fi
            [[ -n "$host_ip" ]] && _SEEN_HOST["$host_ip"]="$ip"
            COPY_PEER_NODES+=("$ip")
            echo "  Found GB10 copy host: $ip"
        done < <(sort "$temp_file")
        rm -f "$temp_file"
    fi
}

# Save discovered configuration to .env
# Skips if .env already exists unless FORCE_DISCOVER=true
save_config() {
    local env_file="${CONFIG_FILE:-$SCRIPT_DIR/.env}"

    # Skip if .env exists and not forced
    if [[ -f "$env_file" && "${FORCE_DISCOVER:-false}" != "true" ]]; then
        return 0
    fi

    echo ""
    local save_prompt="Save discovered configuration to $env_file?"
    if [[ -f "$env_file" ]]; then
        save_prompt="Overwrite existing configuration in $env_file?"
    fi
    read -r -p "$save_prompt [Y/n]: " response
    response="${response,,}"
    if [[ "$response" =~ ^(n|no)$ ]]; then
        return 0
    fi

    # Build list of all cluster nodes (local + peers)
    local all_cluster_nodes=()
    if [[ -n "$LOCAL_IP" ]]; then
        all_cluster_nodes+=("$LOCAL_IP")
    fi
    for node in "${PEER_NODES[@]}"; do
        all_cluster_nodes+=("$node")
    done

    # Per-node confirmation for CLUSTER_NODES
    echo ""
    echo "Select nodes for CLUSTER_NODES:"
    local selected_cluster=()
    for node in "${all_cluster_nodes[@]}"; do
        local label="$node"
        [[ "$node" == "$LOCAL_IP" ]] && label="$node (this machine)"
        read -r -p "  Include $label? [Y/n]: " r
        r="${r,,}"
        if [[ ! "$r" =~ ^(n|no)$ ]]; then
            selected_cluster+=("$node")
        fi
    done

    if [[ "${#selected_cluster[@]}" -eq 0 ]]; then
        echo "No nodes selected. Aborting save."
        return 1
    fi

    # Per-node confirmation for COPY_HOSTS
    echo ""
    echo "Select nodes for COPY_HOSTS (build/model distribution):"
    local selected_copy=()
    for node in "${COPY_PEER_NODES[@]}"; do
        read -r -p "  Include $node in COPY_HOSTS? [Y/n]: " r
        r="${r,,}"
        if [[ ! "$r" =~ ^(n|no)$ ]]; then
            selected_copy+=("$node")
        fi
    done

    # Write .env
    {
        echo "# Auto-generated by autodiscover.sh"
        echo "CLUSTER_NODES=$(IFS=,; echo "${selected_cluster[*]}")"
        if [[ "${#selected_copy[@]}" -gt 0 ]]; then
            echo "COPY_HOSTS=$(IFS=,; echo "${selected_copy[*]}")"
        fi
        echo "LOCAL_IP=$LOCAL_IP"
        echo "ETH_IF=$ETH_IF"
        echo "IB_IF=$IB_IF"
        if [[ "$MESH_MODE" == "true" ]]; then
            echo "# Mesh mode NCCL settings"
            echo "CONTAINER_NCCL_NET_PLUGIN=none"
            echo "CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING=1"
            echo "CONTAINER_NCCL_IB_MERGE_NICS=0"
        fi
    } > "$env_file"
    echo ""
    echo "Saved to $env_file"
}

# Convenience function: run full autodiscovery pipeline
run_autodiscover() {
    detect_interfaces || return 1
    detect_local_ip || return 1
    detect_nodes || return 1
    detect_copy_hosts || return 1
    save_config
}
