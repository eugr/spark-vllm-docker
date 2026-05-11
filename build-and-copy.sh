#!/bin/bash
set -e

# Start total time tracking
START_TIME=$(date +%s)

# Default values
IMAGE_TAG="vllm-node"
IMAGE_TAG_SET=false
REBUILD_FLASHINFER=false
REBUILD_VLLM=false
COPY_HOSTS=()
COPY_TO_FLAG=false
SSH_USER="$USER"
NO_BUILD=false
VLLM_REF="main"
VLLM_REF_SET=false
FLASHINFER_REF="main"
FLASHINFER_REF_SET=false
TMP_IMAGE=""
PARALLEL_COPY=false
EXP_MXFP4=false
VLLM_PRS=""
FLASHINFER_PRS=""
PRE_TRANSFORMERS=false
FULL_LOG=false
BUILD_JOBS="16"
GPU_ARCH_LIST="12.1a"
NETWORK_ARG=""
WHEELS_REPO="eugr/spark-vllm-docker"
FLASHINFER_RELEASE_TAG="prebuilt-flashinfer-current"
VLLM_RELEASE_TAG="prebuilt-vllm-current"
# Space-separated list of GPU architectures for which prebuilt wheels are available
PREBUILT_WHEELS_SUPPORTED_ARCHS="12.1a"
CLEANUP_MODE="false"
CONFIG_FILE=""

cleanup() {
    if [ -n "$TMP_IMAGE" ] && [ -f "$TMP_IMAGE" ]; then
        echo "Cleaning up temporary image $TMP_IMAGE"
        rm -f "$TMP_IMAGE"
    fi
    rm -f ./build-metadata.yaml
}

trap cleanup EXIT

generate_build_metadata() {
    local dockerfile="$1"
    local vllm_version="$2"
    local vllm_commit="$3"
    local flashinfer_commit="$4"
    local vllm_ref="$5"
    local pre_transformers="$6"
    local exp_mxfp4="$7"
    local vllm_prs="$8"

    local base_image
    base_image=$(grep -m1 '^FROM .* AS runner' "$dockerfile" | awk '{print $2}')

    cat > ./build-metadata.yaml <<EOF
build_date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
build_script_commit: $(git rev-parse HEAD 2>/dev/null || echo "unknown")
vllm_version: ${vllm_version:-unknown}
vllm_commit: ${vllm_commit:-unknown}
flashinfer_commit: ${flashinfer_commit:-unknown}
gpu_arch: ${GPU_ARCH_LIST}
base_image: ${base_image:-unknown}
build_args:
  vllm_ref: ${vllm_ref}
  transformers_5: ${pre_transformers}
  exp_mxfp4: ${exp_mxfp4}
  vllm_prs: "${vllm_prs}"
  build_jobs: ${BUILD_JOBS}
EOF
    echo "Generated build-metadata.yaml"
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

copy_to_host() {
    local host="$1"
    echo "Loading image into ${SSH_USER}@${host}..."
    local host_copy_start host_copy_end host_copy_time
    host_copy_start=$(date +%s)
    if cat "$TMP_IMAGE" | ssh "${SSH_USER}@${host}" "docker load"; then
        host_copy_end=$(date +%s)
        host_copy_time=$((host_copy_end - host_copy_start))
        printf "Copy to %s completed in %02d:%02d:%02d\n" "$host" $((host_copy_time/3600)) $((host_copy_time%3600/60)) $((host_copy_time%60))
    else
        echo "Copy to $host failed."
        return 1
    fi
}

# try_download_wheels TAG PREFIX
# Downloads wheels matching PREFIX*.whl from a GitHub release.
# Skip conditions (either is sufficient):
#   1. Commit hash in release name matches .wheels/.{PREFIX}_commit (primary check).
#   2. All local wheels are newer than the latest GitHub asset (freshly built).
# Only downloads a file when the remote asset is newer than the local copy AND
# the above skip conditions are not met.
# On success, persists the release commit hash to .wheels/.{PREFIX}_commit.
# Returns 0 if all matching wheels are now available, 1 on any error.
try_download_wheels() {
    local TAG="$1"
    local PREFIX="$2"
    local WHEELS_DIR="./wheels"

    local arch
    for arch in $PREBUILT_WHEELS_SUPPORTED_ARCHS; do
        [ "$arch" = "$GPU_ARCH_LIST" ] && break
        arch=""
    done
    if [ -z "$arch" ]; then
        echo "GPU arch '$GPU_ARCH_LIST' not supported by prebuilt wheels (supported: $PREBUILT_WHEELS_SUPPORTED_ARCHS) — skipping download."
        return 1
    fi

    local RELEASE_JSON
    RELEASE_JSON=$(curl -sf --connect-timeout 10 \
        "https://api.github.com/repos/$WHEELS_REPO/releases/tags/$TAG") || {
        echo "Could not fetch release metadata for '$TAG' — skipping download."
        return 1
    }

    local DOWNLOAD_LIST
    DOWNLOAD_LIST=$(echo "$RELEASE_JSON" | python3 -c '
import json, sys, os, re
from datetime import datetime, timezone

wheels_dir, prefix = sys.argv[1], sys.argv[2]
data = json.load(sys.stdin)
assets = [a for a in data.get("assets", [])
          if a["name"].startswith(prefix) and a["name"].endswith(".whl")]

if not assets:
    print("No assets found matching prefix: " + prefix, file=sys.stderr)
    sys.exit(1)

# Extract commit hash from the release name:
#   FlashInfer: "Prebuilt FlashInfer Wheels (0.6.5-124a2d32-d20260305) - DGX Spark Only"
#   vLLM:       "Prebuilt vLLM Wheels (0.16.1rc1.dev296+ga73af584f.d20260305.cu131) - DGX Spark only"
release_name = data.get("name", "")
commit_hash = None
if prefix.startswith("flashinfer"):
    m = re.search(r"\([\d.]+\w*-([0-9a-f]{6,})-d\d{8}\)", release_name, re.IGNORECASE)
    if m:
        commit_hash = m.group(1)
else:
    m = re.search(r"\+g([0-9a-f]{6,})\.", release_name, re.IGNORECASE)
    if m:
        commit_hash = m.group(1)

# Compare against the locally stored commit hash
commit_file = os.path.join(wheels_dir, "." + prefix + "-commit")
local_commit = None
if os.path.exists(commit_file):
    with open(commit_file) as f:
        local_commit = f.read().strip()

if commit_hash and local_commit and local_commit[:len(commit_hash)] == commit_hash:
    print("Commit hash matches (" + commit_hash + ") — wheels are up to date.", file=sys.stderr)
    sys.exit(0)

newest_remote_ts = max(
    datetime.strptime(a["updated_at"], "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc).timestamp()
    for a in assets
)

# If local wheels (any version matching prefix) are all newer than the
# latest GitHub asset, they were freshly built and should not be replaced.
local_wheels = [
    os.path.join(wheels_dir, f) for f in os.listdir(wheels_dir)
    if f.startswith(prefix) and f.endswith(".whl")
]
if local_wheels and all(os.path.getmtime(p) >= newest_remote_ts for p in local_wheels):
    sys.exit(0)

downloads = []
for a in assets:
    local_path = os.path.join(wheels_dir, a["name"])
    remote_ts = datetime.strptime(a["updated_at"], "%Y-%m-%dT%H:%M:%SZ") \
                    .replace(tzinfo=timezone.utc).timestamp()
    if not os.path.exists(local_path) or remote_ts > os.path.getmtime(local_path):
        downloads.append(a["browser_download_url"] + " " + a["name"])

if downloads:
    if commit_hash:
        print("#commit:" + commit_hash)
    for d in downloads:
        print(d)
' "$WHEELS_DIR" "$PREFIX") || return 1

    if [ -z "$DOWNLOAD_LIST" ]; then
        echo "All $PREFIX wheels are up to date — skipping download."
        return 0
    fi

    # Parse the optional '#commit:HASH' sentinel emitted by the Python script
    local REMOTE_COMMIT=""
    local DOWNLOAD_ENTRIES=""
    while IFS= read -r LINE; do
        if [[ "$LINE" == "#commit:"* ]]; then
            REMOTE_COMMIT="${LINE#"#commit:"}"
        elif [[ -n "$LINE" ]]; then
            DOWNLOAD_ENTRIES+="$LINE"$'\n'
        fi
    done <<< "$DOWNLOAD_LIST"

    if [ -z "$DOWNLOAD_ENTRIES" ]; then
        echo "All $PREFIX wheels are up to date — skipping download."
        return 0
    fi

    # Back up existing wheels so we never leave a mix of old and new on failure
    local DL_BACKUP="$WHEELS_DIR/.backup-download-${PREFIX}"
    rm -rf "$DL_BACKUP" && mkdir -p "$DL_BACKUP"
    for f in "$WHEELS_DIR/${PREFIX}"*.whl; do
        [ -f "$f" ] && mv "$f" "$DL_BACKUP/"
    done
    for f in "$WHEELS_DIR/.${PREFIX}"*; do
        [ -f "$f" ] && mv "$f" "$DL_BACKUP/"
    done

    local URL NAME TMP_WHL
    local DOWNLOADED=()
    while IFS=' ' read -r URL NAME; do
        [ -z "$URL" ] && continue
        echo "Downloading $NAME..."
        TMP_WHL=$(mktemp "$WHEELS_DIR/${NAME}.XXXXXX")
        if curl -L --progress-bar --connect-timeout 30 "$URL" -o "$TMP_WHL"; then
            mv "$TMP_WHL" "$WHEELS_DIR/$NAME"
            DOWNLOADED+=("$WHEELS_DIR/$NAME")
        else
            rm -f "$TMP_WHL"
            echo "Failed to download $NAME — removing other downloaded files."
            for f in "${DOWNLOADED[@]}"; do rm -f "$f"; done
            if compgen -G "$DL_BACKUP/${PREFIX}*.whl" > /dev/null 2>&1; then
                echo "Restoring previous $PREFIX wheels..."
                mv "$DL_BACKUP/${PREFIX}"*.whl "$WHEELS_DIR/"
                mv "$DL_BACKUP/.${PREFIX}"* "$WHEELS_DIR/"
            fi
            rm -rf "$DL_BACKUP"
            return 1
        fi
    done <<< "$DOWNLOAD_ENTRIES"

    rm -rf "$DL_BACKUP"
    if [ -n "$REMOTE_COMMIT" ]; then
        echo "$REMOTE_COMMIT" > "$WHEELS_DIR/.${PREFIX}-commit"
        echo "Recorded $PREFIX commit hash: $REMOTE_COMMIT"
    fi
    return 0
}

# Help function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -t, --tag <tag>               : Image tag (default: 'vllm-node', 'vllm-node-tf5' with --tf5, 'vllm-node-mxfp4' with --exp-mxfp4)"
    echo "  --gpu-arch <arch>             : GPU architecture (default: '12.1a')"
    echo "  --rebuild-flashinfer          : Force rebuild of FlashInfer wheels (ignore cached wheels)"
    echo "  --rebuild-vllm                : Force rebuild of vLLM wheels (ignore cached wheels)"
    echo "  --vllm-ref <ref>              : vLLM commit SHA, branch or tag (default: 'main')"
    echo "  --flashinfer-ref <ref>        : FlashInfer commit SHA, branch or tag (default: 'main')"
    echo "  -c, --copy-to <hosts>         : Host(s) to copy the image to. Accepts comma or space-delimited lists."
    echo "      --copy-to-host            : Alias for --copy-to (backwards compatibility)."
    echo "      --copy-parallel           : Copy to all hosts in parallel instead of serially."
    echo "  -j, --build-jobs <jobs>       : Number of concurrent build jobs (default: ${BUILD_JOBS})"
    echo "  -u, --user <user>             : Username for ssh command (default: \$USER)"
    echo "  --tf5                         : Install transformers>=5 (aliases: --pre-tf, --pre-transformers)"
    echo "  --exp-mxfp4, --experimental-mxfp4 : Build with experimental native MXFP4 support"
    echo "  --apply-vllm-pr <pr-num>      : Apply a specific PR patch to vLLM source. Can be specified multiple times."
    echo "  --apply-flashinfer-pr <pr-num>: Apply a specific PR patch to FlashInfer source. Can be specified multiple times."
    echo "  --full-log                    : Enable full build logging (--progress=plain)"
    echo "  --no-build                    : Skip building, only copy image (requires --copy-to)"
    echo "  --network <network>           : Docker network to use during build"
    echo "  --cleanup                     : Remove all *.whl and *.-commit files in wheels directory"
    echo "  --config                      : Path to .env configuration file (default: .env in script directory)"
    echo "  --setup                       : Force autodiscovery and save configuration (even if .env exists)"
    echo "  -h, --help                    : Show this help message"
    exit 1
}

# Parse all arguments
CONFIG_FILE_SET=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--tag) IMAGE_TAG="$2"; IMAGE_TAG_SET=true; shift ;;
        --gpu-arch) GPU_ARCH_LIST="$2"; shift ;;
        --rebuild-flashinfer) REBUILD_FLASHINFER=true ;;
        --rebuild-vllm) REBUILD_VLLM=true ;;
        --vllm-ref) VLLM_REF="$2"; VLLM_REF_SET=true; shift ;;
        --flashinfer-ref) FLASHINFER_REF="$2"; FLASHINFER_REF_SET=true; shift ;;
        -c|--copy-to|--copy-to-host|--copy-to-hosts)
            COPY_TO_FLAG=true
            shift
            while [[ "$#" -gt 0 && "$1" != -* ]]; do
                add_copy_hosts "$1"
                shift
            done
            continue
            ;;
        -j|--build-jobs) BUILD_JOBS="$2"; shift ;;
        -u|--user) SSH_USER="$2"; shift ;;
        --copy-parallel) PARALLEL_COPY=true ;;
        --tf5|--pre-tf|--pre-transformers) PRE_TRANSFORMERS=true ;;
        --exp-mxfp4|--experimental-mxfp4) EXP_MXFP4=true ;;
        --apply-vllm-pr)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
               if [ -n "$VLLM_PRS" ]; then
                   VLLM_PRS="$VLLM_PRS $2"
               else
                   VLLM_PRS="$2"
               fi
               shift
            else
               echo "Error: --apply-vllm-pr requires a PR number."
               exit 1
            fi
            ;;
        --apply-flashinfer-pr)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
               if [ -n "$FLASHINFER_PRS" ]; then
                   FLASHINFER_PRS="$FLASHINFER_PRS $2"
               else
                   FLASHINFER_PRS="$2"
               fi
               shift
            else
               echo "Error: --apply-flashinfer-pr requires a PR number."
               exit 1
            fi
            ;;
        --full-log) FULL_LOG=true ;;
        --no-build) NO_BUILD=true ;;
        --cleanup) CLEANUP_MODE=true ;;
        --network)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
                NETWORK_ARG="$2"
                shift
            else
                echo "Error: --network requires a network name."
                exit 1
            fi
            ;;
        --config) CONFIG_FILE="$2"; CONFIG_FILE_SET=true; shift ;;
        --setup) FORCE_DISCOVER=true; export FORCE_DISCOVER ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Apply default IMAGE_TAG based on flags if -t was not specified
if [ "$IMAGE_TAG_SET" = false ]; then
    if [ "$PRE_TRANSFORMERS" = true ]; then
        IMAGE_TAG="vllm-node-tf5"
    elif [ "$EXP_MXFP4" = true ]; then
        IMAGE_TAG="vllm-node-mxfp4"
    fi
fi

# Source autodiscover.sh to load .env file
source "$(dirname "$0")/autodiscover.sh"

# If --setup: force full autodiscovery and save configuration
if [[ "${FORCE_DISCOVER:-false}" == "true" ]]; then
    echo "Running full autodiscovery (--setup)..."
    detect_interfaces || exit 1
    detect_local_ip || exit 1
    detect_nodes || exit 1
    detect_copy_hosts || exit 1
    save_config || exit 1
    # Reload .env so DOTENV_* variables reflect saved config
    load_env_if_exists
fi

# Handle COPY_HOSTS from .env or autodiscovery only if -c was explicitly specified
if [ "$COPY_TO_FLAG" = true ] && [ "${#COPY_HOSTS[@]}" -eq 0 ]; then
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
        echo "Autodiscovered hosts: ${COPY_HOSTS[*]}"
    fi
fi

# Validate flag combinations
if [ -n "$VLLM_PRS" ]; then
    if [ "$EXP_MXFP4" = true ]; then echo "Error: --apply-vllm-pr is incompatible with --exp-mxfp4"; exit 1; fi
fi

if [ -n "$FLASHINFER_PRS" ]; then
    if [ "$EXP_MXFP4" = true ]; then echo "Error: --apply-flashinfer-pr is incompatible with --exp-mxfp4"; exit 1; fi
fi

if [ "$EXP_MXFP4" = true ]; then
    if [ "$VLLM_REF_SET" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --vllm-ref"; exit 1; fi
    if [ "$FLASHINFER_REF_SET" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --flashinfer-ref"; exit 1; fi
    if [ "$PRE_TRANSFORMERS" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --tf5"; exit 1; fi
    if [ "$REBUILD_FLASHINFER" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --rebuild-flashinfer"; exit 1; fi
    if [ "$REBUILD_VLLM" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --rebuild-vllm"; exit 1; fi
fi

# Validate --no-build usage
if [ "$NO_BUILD" = true ] && [ "${#COPY_HOSTS[@]}" -eq 0 ]; then
    echo "Error: --no-build requires --copy-to to be specified"
    exit 1
fi

# Handle cleanup mode
if [[ "$CLEANUP_MODE" == "true" ]]; then
    WHEELS_DIR="./wheels"
    echo "Cleaning up wheels directory..."
    
    # Remove all .whl files
    if compgen -G "$WHEELS_DIR/*.whl" > /dev/null 2>&1; then
        rm -f "$WHEELS_DIR"/*.whl
        echo "Removed *.whl files from $WHEELS_DIR"
    else
        echo "No *.whl files found in $WHEELS_DIR"
    fi
    
    # Remove all .-commit files
    if compgen -G "$WHEELS_DIR/.*-commit" > /dev/null 2>&1; then
        rm -f "$WHEELS_DIR"/.*-commit
        echo "Removed .*-commit files from $WHEELS_DIR"
    else
        echo "No .*-commit files found in $WHEELS_DIR"
    fi
    
    echo "Cleanup complete."
fi

# Ensure wheels directory exists
mkdir -p ./wheels

# Common build flags used across all non-mxfp4 sub-builds
COMMON_BUILD_FLAGS=()
if [ "$FULL_LOG" = true ]; then
    COMMON_BUILD_FLAGS+=("--progress=plain")
fi
COMMON_BUILD_FLAGS+=("--build-arg" "BUILD_JOBS=$BUILD_JOBS")
COMMON_BUILD_FLAGS+=("--build-arg" "TORCH_CUDA_ARCH_LIST=$GPU_ARCH_LIST")
COMMON_BUILD_FLAGS+=("--build-arg" "FLASHINFER_CUDA_ARCH_LIST=$GPU_ARCH_LIST")
if [ -n "$NETWORK_ARG" ]; then
    COMMON_BUILD_FLAGS+=("--network" "$NETWORK_ARG")
fi

# =====================================================
# Build image (unless --no-build or --exp-mxfp4)
# =====================================================
FLASHINFER_BUILD_TIME=0
VLLM_BUILD_TIME=0
RUNNER_BUILD_TIME=0

if [ "$NO_BUILD" = false ]; then
    if [ "$EXP_MXFP4" = true ]; then
        echo "Building with experimental MXFP4 support..."

        # Generate build metadata YAML for mxfp4 build
        MXFP4_VLLM_SHA=$(grep -m1 '^ARG VLLM_SHA=' Dockerfile.mxfp4 | cut -d= -f2)
        MXFP4_FLASHINFER_SHA=$(grep -m1 '^ARG FLASHINFER_SHA=' Dockerfile.mxfp4 | cut -d= -f2)
        generate_build_metadata Dockerfile.mxfp4 "unknown" "$MXFP4_VLLM_SHA" "$MXFP4_FLASHINFER_SHA" \
            "mxfp4-pinned" "false" "true" ""

        CMD=("docker" "build" "-t" "$IMAGE_TAG" "${COMMON_BUILD_FLAGS[@]}" "-f" "Dockerfile.mxfp4" ".")
        echo "Building image with command: ${CMD[*]}"
        BUILD_START=$(date +%s)
        "${CMD[@]}"
        BUILD_END=$(date +%s)
        RUNNER_BUILD_TIME=$((BUILD_END - BUILD_START))
    else
        # ----------------------------------------------------------
        # Phase 1: FlashInfer wheels
        # ----------------------------------------------------------
        if [ "$FLASHINFER_REF_SET" = true ] || [ -n "$FLASHINFER_PRS" ]; then
            REBUILD_FLASHINFER=true
        fi

        BUILD_FLASHINFER=false
        if [ "$REBUILD_FLASHINFER" = true ]; then
            if [ "$FLASHINFER_REF_SET" = true ] && [ -n "$FLASHINFER_PRS" ]; then
                echo "Rebuilding FlashInfer wheels (--flashinfer-ref and --apply-flashinfer-pr specified)..."
            elif [ "$FLASHINFER_REF_SET" = true ]; then
                echo "Rebuilding FlashInfer wheels (--flashinfer-ref specified)..."
            elif [ -n "$FLASHINFER_PRS" ]; then
                echo "Rebuilding FlashInfer wheels (--apply-flashinfer-pr specified)..."
            else
                echo "Rebuilding FlashInfer wheels (--rebuild-flashinfer specified)..."
            fi
            BUILD_FLASHINFER=true
        elif try_download_wheels "$FLASHINFER_RELEASE_TAG" "flashinfer"; then
            echo "FlashInfer wheels ready."
        elif compgen -G "./wheels/flashinfer*.whl" > /dev/null 2>&1; then
            echo "Download failed — using existing local FlashInfer wheels."
        else
            echo "No FlashInfer wheels available (download failed) — building..."
            BUILD_FLASHINFER=true
        fi

        if [ "$BUILD_FLASHINFER" = true ]; then
            # Back up existing flashinfer wheels; restore them if the build fails
            FI_BACKUP="./wheels/.backup-flashinfer"
            rm -rf "$FI_BACKUP" && mkdir -p "$FI_BACKUP"
            for f in ./wheels/flashinfer*.whl; do
                [ -f "$f" ] && mv "$f" "$FI_BACKUP/"
            done

            FI_CMD=("docker" "build"
                "--target" "flashinfer-export"
                "--output" "type=local,dest=./wheels"
                "${COMMON_BUILD_FLAGS[@]}"
                "--build-arg" "FLASHINFER_REF=$FLASHINFER_REF")

            if [ "$REBUILD_FLASHINFER" = true ]; then
                FI_CMD+=("--build-arg" "CACHEBUST_FLASHINFER=$(date +%s)")
            fi

            if [ -n "$FLASHINFER_PRS" ]; then
                echo "Applying FlashInfer PRs: $FLASHINFER_PRS"
                FI_CMD+=("--build-arg" "FLASHINFER_PRS=$FLASHINFER_PRS")
            fi

            FI_CMD+=(".")

            echo "FlashInfer build command: ${FI_CMD[*]}"
            FI_START=$(date +%s)
            if "${FI_CMD[@]}"; then
                FI_END=$(date +%s)
                FLASHINFER_BUILD_TIME=$((FI_END - FI_START))
                rm -rf "$FI_BACKUP"
            else
                echo "FlashInfer build failed — restoring previous wheels..."
                mv "$FI_BACKUP"/flashinfer*.whl ./wheels/ 2>/dev/null || true
                rm -rf "$FI_BACKUP"
                exit 1
            fi
        fi

        # ----------------------------------------------------------
        # Phase 2: vLLM wheels
        # ----------------------------------------------------------
        if [ "$VLLM_REF_SET" = true ] || [ -n "$VLLM_PRS" ]; then
            REBUILD_VLLM=true
        fi

        BUILD_VLLM=false
        if [ "$REBUILD_VLLM" = true ]; then
            if [ "$VLLM_REF_SET" = true ] && [ -n "$VLLM_PRS" ]; then
                echo "Rebuilding vLLM wheels (--vllm-ref and --apply-vllm-pr specified)..."
            elif [ "$VLLM_REF_SET" = true ]; then
                echo "Rebuilding vLLM wheels (--vllm-ref specified)..."
            elif [ -n "$VLLM_PRS" ]; then
                echo "Rebuilding vLLM wheels (--apply-vllm-pr specified)..."
            else
                echo "Rebuilding vLLM wheels (--rebuild-vllm specified)..."
            fi
            BUILD_VLLM=true
        elif try_download_wheels "$VLLM_RELEASE_TAG" "vllm"; then
            echo "vLLM wheels ready."
        elif compgen -G "./wheels/vllm*.whl" > /dev/null 2>&1; then
            echo "Download failed — using existing local vLLM wheels."
        else
            echo "No vLLM wheels available (download failed) — building..."
            BUILD_VLLM=true
        fi

        if [ "$BUILD_VLLM" = true ]; then
            # Back up existing vllm wheels; restore them if the build fails
            VLLM_BACKUP="./wheels/.backup-vllm"
            rm -rf "$VLLM_BACKUP" && mkdir -p "$VLLM_BACKUP"
            for f in ./wheels/vllm*.whl; do
                [ -f "$f" ] && mv "$f" "$VLLM_BACKUP/"
            done

            VLLM_CMD=("docker" "build"
                "--target" "vllm-export"
                "--output" "type=local,dest=./wheels"
                "${COMMON_BUILD_FLAGS[@]}"
                "--build-arg" "VLLM_REF=$VLLM_REF")

            if [ "$REBUILD_VLLM" = true ]; then
                VLLM_CMD+=("--build-arg" "CACHEBUST_VLLM=$(date +%s)")
            fi

            if [ -n "$VLLM_PRS" ]; then
                echo "Applying vLLM PRs: $VLLM_PRS"
                VLLM_CMD+=("--build-arg" "VLLM_PRS=$VLLM_PRS")
            fi

            VLLM_CMD+=(".")

            echo "vLLM build command: ${VLLM_CMD[*]}"
            VLLM_START=$(date +%s)
            if "${VLLM_CMD[@]}"; then
                VLLM_END=$(date +%s)
                VLLM_BUILD_TIME=$((VLLM_END - VLLM_START))
                rm -rf "$VLLM_BACKUP"
            else
                echo "vLLM build failed — restoring previous wheels..."
                mv "$VLLM_BACKUP"/vllm*.whl ./wheels/ 2>/dev/null || true
                rm -rf "$VLLM_BACKUP"
                exit 1
            fi
        fi

        # ----------------------------------------------------------
        # Phase 3: Runner image
        # ----------------------------------------------------------
        if ! compgen -G "./wheels/*.whl" > /dev/null 2>&1; then
            echo "Error: No wheel files found in ./wheels/ — cannot build runner image."
            exit 1
        fi

        # Generate build metadata YAML
        VLLM_VERSION=$(ls ./wheels/vllm-*.whl 2>/dev/null | head -1 | sed 's|.*/vllm-||;s|-.*||')
        VLLM_COMMIT=""
        [ -f "./wheels/.vllm-commit" ] && VLLM_COMMIT=$(cat ./wheels/.vllm-commit)
        FLASHINFER_COMMIT=""
        [ -f "./wheels/.flashinfer-commit" ] && FLASHINFER_COMMIT=$(cat ./wheels/.flashinfer-commit)
        generate_build_metadata Dockerfile "$VLLM_VERSION" "$VLLM_COMMIT" "$FLASHINFER_COMMIT" \
            "$VLLM_REF" "$PRE_TRANSFORMERS" "false" "$VLLM_PRS"

        RUNNER_CMD=("docker" "build"
            "-t" "$IMAGE_TAG"
            "${COMMON_BUILD_FLAGS[@]}")

        if [ "$PRE_TRANSFORMERS" = true ]; then
            echo "Using transformers>=5.0.0..."
            RUNNER_CMD+=("--build-arg" "PRE_TRANSFORMERS=1")
        fi

        RUNNER_CMD+=(".")

        echo "Building runner image with command: ${RUNNER_CMD[*]}"
        RUNNER_START=$(date +%s)
        "${RUNNER_CMD[@]}"
        RUNNER_END=$(date +%s)
        RUNNER_BUILD_TIME=$((RUNNER_END - RUNNER_START))
    fi
else
    echo "Skipping build (--no-build specified)"
fi

# =====================================================
# Copy to host(s) if requested
# =====================================================
COPY_TIME=0
if [ "${#COPY_HOSTS[@]}" -gt 0 ]; then
    echo "Copying image '$IMAGE_TAG' to ${#COPY_HOSTS[@]} host(s): ${COPY_HOSTS[*]}"
    if [ "$PARALLEL_COPY" = true ]; then
        echo "Parallel copy enabled."
    fi
    COPY_START=$(date +%s)

    TMP_IMAGE=$(mktemp -t vllm_image.XXXXXX)
    echo "Saving image locally to $TMP_IMAGE..."
    docker save -o "$TMP_IMAGE" "$IMAGE_TAG"

    if [ "$PARALLEL_COPY" = true ]; then
        PIDS=()
        for host in "${COPY_HOSTS[@]}"; do
            copy_to_host "$host" &
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
            copy_to_host "$host"
        done
    fi

    COPY_END=$(date +%s)
    COPY_TIME=$((COPY_END - COPY_START))
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
if [ "$FLASHINFER_BUILD_TIME" -gt 0 ]; then
    echo "FlashInfer Build: $(printf '%02d:%02d:%02d' $((FLASHINFER_BUILD_TIME/3600)) $((FLASHINFER_BUILD_TIME%3600/60)) $((FLASHINFER_BUILD_TIME%60)))"
fi
if [ "$VLLM_BUILD_TIME" -gt 0 ]; then
    echo "vLLM Build:       $(printf '%02d:%02d:%02d' $((VLLM_BUILD_TIME/3600)) $((VLLM_BUILD_TIME%3600/60)) $((VLLM_BUILD_TIME%60)))"
fi
if [ "$RUNNER_BUILD_TIME" -gt 0 ]; then
    echo "Runner Build:     $(printf '%02d:%02d:%02d' $((RUNNER_BUILD_TIME/3600)) $((RUNNER_BUILD_TIME%3600/60)) $((RUNNER_BUILD_TIME%60)))"
fi
if [ "$COPY_TIME" -gt 0 ]; then
    echo "Image Copy:       $(printf '%02d:%02d:%02d' $((COPY_TIME/3600)) $((COPY_TIME%3600/60)) $((COPY_TIME%60)))"
fi
echo "Total Time:       $(printf '%02d:%02d:%02d' $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60)))"
echo "========================================="
echo "Done building $IMAGE_TAG."
