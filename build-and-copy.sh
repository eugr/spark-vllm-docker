#!/bin/bash
set -e

# Start total time tracking
START_TIME=$(date +%s)

# Default values
IMAGE_TAG="vllm-node"
IMAGE_TAG_SET=false
PREBUILT_RUNNER_IMAGE="eugr/spark-vllm:latest"
PREBUILT_B12X_RUNNER_IMAGE="eugr/spark-vllm-b12x:latest"
USE_WHEELS=false
REBUILD_FLASHINFER=false
REBUILD_VLLM=false
FLASHINFER_ARCH_REBUILD=false
FLASHINFER_ARCH_MISMATCH=false
VLLM_ARCH_REBUILD=false
VLLM_ARCH_MISMATCH=false
FORCE_FLASHINFER_DOWNLOAD=false
FORCE_VLLM_DOWNLOAD=false
COPY_HOSTS=()
COPY_TO_FLAG=false
SSH_USER="$USER"
NO_BUILD=false
DEFAULT_VLLM_REPO="https://github.com/vllm-project/vllm.git"
VLLM_REPO="$DEFAULT_VLLM_REPO"
VLLM_REPO_SET=false
VLLM_REF="main"
VLLM_REF_SET=false
EXP_B12X=false
EXP_B12X_VLLM_REPO="https://github.com/local-inference-lab/vllm"
EXP_B12X_VLLM_REF="dev/gilded-gnosis"
EXP_B12X_PACKAGE_REPO="https://github.com/lukealonso/b12x.git"
EXP_B12X_PACKAGE_REF="master"
EXP_B12X_TORCH_VERSION="2.12.0"
EXP_B12X_TORCHVISION_VERSION="0.27.0"
EXP_B12X_TORCHAUDIO_VERSION="none"
SPARKINFER_REPO=""
SPARKINFER_REF=""
SPARKINFER_CACHEBUST=""
FLASHINFER_REF="main"
FLASHINFER_REF_SET=false
TMP_IMAGE=""
PARALLEL_COPY=false
EXP_MXFP4=false
VLLM_PRS=""
APPLY_PRESET_VLLM_PRS=false
FLASHINFER_PRS=""
# Deprecated --tf5 aliases are kept for tag compatibility only; they no longer alter dependency resolution.
PRE_TRANSFORMERS=false
FULL_LOG=false
BUILD_JOBS="16"
BUILD_JOBS_SET=false
DEFAULT_GPU_ARCH_LIST="12.1a"
GPU_ARCH_LIST="$DEFAULT_GPU_ARCH_LIST"
GPU_ARCH_SET=false
DEFAULT_TORCH_VERSION="2.11.0"
TORCH_VERSION="$DEFAULT_TORCH_VERSION"
TORCH_VERSION_SET=false
TORCHVISION_VERSION=""
TORCHVISION_VERSION_SET=false
TORCHAUDIO_VERSION=""
TORCHAUDIO_VERSION_SET=false
NETWORK_ARG=""
WHEELS_REPO="eugr/spark-vllm-docker"
FLASHINFER_RELEASE_TAG="prebuilt-flashinfer-current"
VLLM_RELEASE_TAG="prebuilt-vllm-current"
# Space-separated list of GPU architectures for which prebuilt wheels are available
PREBUILT_WHEELS_SUPPORTED_ARCHS="12.1a"
CLEANUP_MODE="false"
CONFIG_FILE=""
WHEEL_CACHE_ROOT="./.wheel-cache"
FLASHINFER_PROFILE="regular"
VLLM_PROFILE="regular"
FLASHINFER_WHEELS_DIR=""
VLLM_WHEELS_DIR=""
FLASHINFER_STAGING_DIR=""
VLLM_STAGING_DIR=""

cleanup() {
    if [ -n "$TMP_IMAGE" ] && [ -f "$TMP_IMAGE" ]; then
        echo "Cleaning up temporary image $TMP_IMAGE"
        rm -f "$TMP_IMAGE"
    fi
    if [ -n "$FLASHINFER_STAGING_DIR" ] && [ -d "$FLASHINFER_STAGING_DIR" ]; then
        rm -rf "$FLASHINFER_STAGING_DIR"
    fi
    if [ -n "$VLLM_STAGING_DIR" ] && [ -d "$VLLM_STAGING_DIR" ]; then
        rm -rf "$VLLM_STAGING_DIR"
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
    local transformers_5="$6"
    local exp_mxfp4="$7"
    local vllm_prs="$8"
    local vllm_repo="$9"
    local torch_version="${10}"
    local torchvision_version="${11}"
    local torchaudio_version="${12}"
    local sparkinfer_repo="${13}"
    local sparkinfer_ref="${14}"

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
  vllm_repo: "${vllm_repo}"
  vllm_ref: ${vllm_ref}
  torch_version: "${torch_version}"
  torchvision_version: "${torchvision_version}"
  torchaudio_version: "${torchaudio_version}"
  sparkinfer_repo: "${sparkinfer_repo}"
  sparkinfer_ref: "${sparkinfer_ref}"
  transformers_5: ${transformers_5}
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

# Convert --gpu-arch value (e.g. 12.0, 12.0f, 12.1a) to NCCL NVCC_GENCODE format.
gpu_arch_to_nccl_gencode() {
    local arch="$1"
    # Strip optional feature suffix (12.1a -> 12.1, 12.0f -> 12.0).
    arch="${arch%[a-z]}"
    local sm="${arch//./}"
    echo "-gencode=arch=compute_${sm},code=sm_${sm}"
}

get_remote_image_id() {
    local host="$1"
    local image="$2"
    ssh "${SSH_USER}@${host}" "docker image inspect --format '{{.Id}}' ${image}" 2>/dev/null
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

get_local_mtime() {
    local path="$1"
    stat -c %Y "$path" 2>/dev/null || stat -f %m "$path"
}

get_remote_asset_mtime() {
    local url="$1"
    curl -fsIL --connect-timeout 10 "$url" | python3 -c '
import email.utils
import sys

last_modified = None
for line in sys.stdin:
    if line.lower().startswith("last-modified:"):
        last_modified = line.split(":", 1)[1].strip()

if not last_modified:
    sys.exit(1)

try:
    print(int(email.utils.parsedate_to_datetime(last_modified).timestamp()))
except Exception:
    sys.exit(1)
'
}

local_wheels_are_newer_than_release() {
    local wheels_dir="$1"
    local prefix="$2"
    local release_entries="$3"

    local local_oldest_ts=""
    local f local_ts
    for f in "$wheels_dir/${prefix}"*.whl; do
        [ -f "$f" ] || continue
        local_ts=$(get_local_mtime "$f" 2>/dev/null || echo 0)
        if [ -z "$local_oldest_ts" ] || [ "$local_ts" -lt "$local_oldest_ts" ]; then
            local_oldest_ts="$local_ts"
        fi
    done

    if [ -z "$local_oldest_ts" ] || [ "$local_oldest_ts" -eq 0 ]; then
        return 1
    fi

    local remote_newest_ts=0
    local url name remote_ts
    while IFS=' ' read -r url name; do
        [ -z "$url" ] && continue
        remote_ts=$(get_remote_asset_mtime "$url" 2>/dev/null || true)
        if [ -z "$remote_ts" ]; then
            return 1
        fi
        if [ "$remote_ts" -gt "$remote_newest_ts" ]; then
            remote_newest_ts="$remote_ts"
        fi
    done <<< "$release_entries"

    if [ "$remote_newest_ts" -eq 0 ]; then
        return 1
    fi

    [ "$local_oldest_ts" -ge "$remote_newest_ts" ]
}

# try_download_wheels TAG PREFIX FORCE_DOWNLOAD WHEELS_DIR
# Downloads wheels matching PREFIX*.whl from a GitHub release.
# Uses GitHub release pages and HTTP Last-Modified headers instead of GitHub API metadata.
# Skips download when exact release assets are current, or when a newer locally
# built wheel set is present even if its filenames differ from the release.
# When FORCE_DOWNLOAD is true, downloads every matching release asset.
# On success, persists the release commit hash to WHEELS_DIR/.{PREFIX}-commit.
# Returns 0 if all matching wheels are now available, 1 on any error.
try_download_wheels() {
    local TAG="$1"
    local PREFIX="$2"
    local FORCE_DOWNLOAD="${3:-false}"
    local WHEELS_DIR="$4"

    mkdir -p "$WHEELS_DIR"

    local arch
    for arch in $PREBUILT_WHEELS_SUPPORTED_ARCHS; do
        [ "$arch" = "$GPU_ARCH_LIST" ] && break
        arch=""
    done
    if [ -z "$arch" ]; then
        echo "GPU arch '$GPU_ARCH_LIST' not supported by prebuilt wheels (supported: $PREBUILT_WHEELS_SUPPORTED_ARCHS) — skipping download."
        return 1
    fi

    local RELEASE_ASSETS_HTML
    RELEASE_ASSETS_HTML=$(curl -sfL --connect-timeout 10 \
        "https://github.com/$WHEELS_REPO/releases/expanded_assets/$TAG") || {
        echo "Could not fetch release assets for '$TAG' — skipping download."
        return 1
    }

    local RELEASE_ENTRIES
    RELEASE_ENTRIES=$(printf '%s' "$RELEASE_ASSETS_HTML" | python3 -c '
import html.parser, os, sys
from urllib.parse import unquote, urlparse

repo, tag, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
asset_path_prefix = "/" + repo + "/releases/download/" + tag + "/"

class ReleaseAssetParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attrs = dict(attrs)
        href = attrs.get("href")
        if href:
            self.hrefs.append(href)

parser = ReleaseAssetParser()
parser.feed(sys.stdin.read())

seen = set()
for href in parser.hrefs:
    if not href.startswith(asset_path_prefix):
        continue
    name = unquote(os.path.basename(urlparse(href).path))
    if not name.startswith(prefix) or not name.endswith(".whl"):
        continue
    if name in seen:
        continue
    seen.add(name)
    print("https://github.com" + href + " " + name)

if not seen:
    print("No assets found matching prefix: " + prefix, file=sys.stderr)
    sys.exit(1)
' "$WHEELS_REPO" "$TAG" "$PREFIX") || return 1

    local RELEASE_PAGE_HTML REMOTE_COMMIT
    REMOTE_COMMIT=""
    if RELEASE_PAGE_HTML=$(curl -sfL --connect-timeout 10 \
        "https://github.com/$WHEELS_REPO/releases/tag/$TAG"); then
        REMOTE_COMMIT=$(printf '%s' "$RELEASE_PAGE_HTML" | python3 -c '
import re, sys

prefix = sys.argv[1]
html = sys.stdin.read()
match = None
if prefix.startswith("flashinfer"):
    match = re.search(r"\([\d.]+\w*-([0-9a-f]{6,})-d\d{8}\)", html, re.IGNORECASE)
else:
    match = re.search(r"\+g([0-9a-f]{6,})\.", html, re.IGNORECASE)
if match:
    print(match.group(1))
' "$PREFIX")
    fi

    local DOWNLOAD_ENTRIES=""
    if [ "$FORCE_DOWNLOAD" = true ]; then
        echo "Force downloading $PREFIX wheels from release '$TAG'..."
        DOWNLOAD_ENTRIES="$RELEASE_ENTRIES"
    else
        local LOCAL_COMMIT=""
        if [ -f "$WHEELS_DIR/.${PREFIX}-commit" ]; then
            LOCAL_COMMIT=$(cat "$WHEELS_DIR/.${PREFIX}-commit")
        fi

        local NEED_DOWNLOAD=false
        local RELEASE_ASSETS_PRESENT=true
        local URL NAME
        while IFS=' ' read -r URL NAME; do
            [ -z "$URL" ] && continue
            if [ ! -f "$WHEELS_DIR/$NAME" ]; then
                RELEASE_ASSETS_PRESENT=false
                break
            fi
        done <<< "$RELEASE_ENTRIES"

        if [ "$RELEASE_ASSETS_PRESENT" = false ]; then
            if local_wheels_are_newer_than_release "$WHEELS_DIR" "$PREFIX" "$RELEASE_ENTRIES"; then
                echo "Local $PREFIX wheels are newer than release '$TAG' — skipping download."
                return 0
            fi
            NEED_DOWNLOAD=true
        fi

        if [ "$NEED_DOWNLOAD" = false ]; then
            if [ -n "$REMOTE_COMMIT" ] && [ -n "$LOCAL_COMMIT" ] && [[ "$LOCAL_COMMIT" == "$REMOTE_COMMIT"* ]]; then
                echo "Commit hash matches ($REMOTE_COMMIT) — wheels are up to date."
                return 0
            fi
        fi

        while [ "$NEED_DOWNLOAD" = false ] && IFS=' ' read -r URL NAME; do
            [ -z "$URL" ] && continue
            local LOCAL_TS REMOTE_TS
            LOCAL_TS=$(get_local_mtime "$WHEELS_DIR/$NAME" 2>/dev/null || echo 0)
            REMOTE_TS=$(get_remote_asset_mtime "$URL" 2>/dev/null || true)
            if [ -z "$REMOTE_TS" ] || [ "$REMOTE_TS" -gt "$LOCAL_TS" ]; then
                NEED_DOWNLOAD=true
                break
            fi
        done <<< "$RELEASE_ENTRIES"

        if [ "$NEED_DOWNLOAD" = false ]; then
            echo "All $PREFIX wheels are up to date — skipping download."
            return 0
        fi

        DOWNLOAD_ENTRIES="$RELEASE_ENTRIES"
    fi

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
            fi
            if compgen -G "$DL_BACKUP/.${PREFIX}*" > /dev/null 2>&1; then
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

validate_exported_wheel_set() {
    local component="$1"
    local wheels_dir="$2"

    if [ "$component" = "flashinfer" ]; then
        local cubin=("$wheels_dir"/flashinfer_cubin-*.whl)
        local jit=("$wheels_dir"/flashinfer_jit_cache-*.whl)
        local python=("$wheels_dir"/flashinfer_python-*.whl)
        if [ "${#cubin[@]}" -ne 1 ] || [ ! -f "${cubin[0]}" ] || \
           [ "${#jit[@]}" -ne 1 ] || [ ! -f "${jit[0]}" ] || \
           [ "${#python[@]}" -ne 1 ] || [ ! -f "${python[0]}" ] || \
           [ ! -s "$wheels_dir/.flashinfer-commit" ] || \
           [ ! -s "$wheels_dir/.flashinfer-arch" ]; then
            echo "Error: FlashInfer export did not produce one complete wheel set with provenance markers."
            return 1
        fi
    else
        local vllm=("$wheels_dir"/vllm-*.whl)
        if [ "${#vllm[@]}" -ne 1 ] || [ ! -f "${vllm[0]}" ] || \
           [ ! -s "$wheels_dir/.vllm-commit" ] || \
           [ ! -s "$wheels_dir/.deepgemm-commit" ] || \
           [ ! -s "$wheels_dir/.vllm-arch" ]; then
            echo "Error: vLLM export did not produce exactly one wheel with provenance markers."
            return 1
        fi
    fi
}

validate_runner_wheel_inputs() {
    local flashinfer_dir="$1"
    local vllm_dir="$2"
    local cubin=("$flashinfer_dir"/flashinfer_cubin-*.whl)
    local jit=("$flashinfer_dir"/flashinfer_jit_cache-*.whl)
    local python=("$flashinfer_dir"/flashinfer_python-*.whl)
    local vllm=("$vllm_dir"/vllm-*.whl)

    if [ "${#cubin[@]}" -ne 1 ] || [ ! -f "${cubin[0]}" ] || \
       [ "${#jit[@]}" -ne 1 ] || [ ! -f "${jit[0]}" ] || \
       [ "${#python[@]}" -ne 1 ] || [ ! -f "${python[0]}" ]; then
        echo "Error: FlashInfer profile $flashinfer_dir does not contain exactly one complete wheel set."
        return 1
    fi
    if [ "${#vllm[@]}" -ne 1 ] || [ ! -f "${vllm[0]}" ]; then
        echo "Error: vLLM profile $vllm_dir does not contain exactly one wheel."
        return 1
    fi
}

promote_wheel_set() {
    local staging_dir="$1"
    local target_dir="$2"
    local backup_dir="${target_dir}.backup.$$"

    rm -rf "$backup_dir"
    if [ -d "$target_dir" ]; then
        mv "$target_dir" "$backup_dir"
    fi

    if mv "$staging_dir" "$target_dir"; then
        rm -rf "$backup_dir"
        return 0
    fi

    echo "Error: Could not promote wheel set into $target_dir."
    if [ -d "$backup_dir" ]; then
        mv "$backup_dir" "$target_dir"
    fi
    return 1
}

# Help function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -t, --tag <tag>               : Local image tag (default: 'vllm-node'; preset tags: 'vllm-node-tf5', 'vllm-node-mxfp4', or 'vllm-node-b12x')"
    echo "  --use-wheels                  : Build only the runner from precompiled wheels; never implicitly build source."
    echo "  --gpu-arch <arch>             : GPU architecture for NCCL, wheel, and source builds (default: '${DEFAULT_GPU_ARCH_LIST}')"
    echo "  --rebuild-flashinfer          : Force rebuild of FlashInfer wheels (ignore cached wheels)"
    echo "  --rebuild-vllm                : Force rebuild of vLLM wheels (ignore cached wheels)"
    echo "  --force-flashinfer-download   : Force download of FlashInfer wheels (skip cached wheel checks)"
    echo "  --force-vllm-download         : Force download of vLLM wheels (skip cached wheel checks)"
    echo "  --force-download              : Force download of all prebuilt wheels (skip cached wheel checks)"
    echo "  --vllm-repo <url>             : vLLM Git repository (default: '${DEFAULT_VLLM_REPO}'); custom repositories bypass the shared checkout cache"
    echo "  --vllm-ref <ref>              : vLLM commit SHA, branch or tag (default: 'main')"
    echo "  --torch-version <version>     : PyTorch version for build and runner images (default: '${DEFAULT_TORCH_VERSION}')"
    echo "  --torchvision-version <ver>   : Optional torchvision version (default: resolver-selected for the requested PyTorch version)"
    echo "  --torchaudio-version <ver>    : Optional torchaudio version; use 'none' to omit it (default: resolver-selected)"
    echo "  --flashinfer-ref <ref>        : FlashInfer commit SHA, branch or tag (default: 'main')"
    echo "  -c, --copy-to <hosts>         : Host(s) to copy image to. Accepts comma or space-delimited lists; matching remote image IDs are skipped."
    echo "      --copy-to-host            : Alias for --copy-to (backwards compatibility)."
    echo "      --copy-parallel           : Copy to all hosts in parallel instead of serially."
    echo "  -j, --build-jobs <jobs>       : Number of concurrent build jobs (default: ${BUILD_JOBS})"
    echo "  -u, --user <user>             : Username for ssh command (default: \$USER)"
    echo "  --tf5                         : Deprecated compatibility flag; tag defaults to 'vllm-node-tf5' (aliases: --pre-tf, --pre-transformers)"
    echo "  --exp-mxfp4, --experimental-mxfp4 : Build with experimental native MXFP4 support"
    echo "  --exp-b12x, --experimental-b12x   : Select B12X; pulls its prebuilt image unless a local wheel/image build is requested"
    echo "  --apply-vllm-pr <pr-num>      : Apply a specific PR patch to vLLM source. Can be specified multiple times."
    echo "  --apply-preset-vllm-prs       : Apply preset vLLM PRs even with --vllm-repo, --vllm-ref, or --apply-vllm-pr."
    echo "  --apply-flashinfer-pr <pr-num>: Apply a specific PR patch to FlashInfer source. Can be specified multiple times."
    echo "  --full-log                    : Enable full build logging (--progress=plain)"
    echo "  --no-build                    : Skip building, only copy image (requires --copy-to)"
    echo "  --network <network>           : Docker network to use during build"
    echo "  --cleanup                     : Remove wheel files and provenance markers from all cache profiles"
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
        --use-wheels) USE_WHEELS=true ;;
        --gpu-arch) GPU_ARCH_LIST="$2"; GPU_ARCH_SET=true; shift ;;
        --rebuild-flashinfer) REBUILD_FLASHINFER=true ;;
        --rebuild-vllm) REBUILD_VLLM=true ;;
        --force-flashinfer-download) FORCE_FLASHINFER_DOWNLOAD=true ;;
        --force-vllm-download) FORCE_VLLM_DOWNLOAD=true ;;
        --force-download)
            FORCE_FLASHINFER_DOWNLOAD=true
            FORCE_VLLM_DOWNLOAD=true
            ;;
        --vllm-repo)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
                VLLM_REPO="$2"
                VLLM_REPO_SET=true
                shift
            else
                echo "Error: --vllm-repo requires a repository URL."
                exit 1
            fi
            ;;
        --vllm-ref) VLLM_REF="$2"; VLLM_REF_SET=true; shift ;;
        --torch-version)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
                TORCH_VERSION="$2"
                TORCH_VERSION_SET=true
                shift
            else
                echo "Error: --torch-version requires a version."
                exit 1
            fi
            ;;
        --torchvision-version)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
                TORCHVISION_VERSION="$2"
                TORCHVISION_VERSION_SET=true
                shift
            else
                echo "Error: --torchvision-version requires a version."
                exit 1
            fi
            ;;
        --torchaudio-version)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
                TORCHAUDIO_VERSION="$2"
                TORCHAUDIO_VERSION_SET=true
                shift
            else
                echo "Error: --torchaudio-version requires a version."
                exit 1
            fi
            ;;
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
        -j|--build-jobs) BUILD_JOBS="$2"; BUILD_JOBS_SET=true; shift ;;
        -u|--user) SSH_USER="$2"; shift ;;
        --copy-parallel) PARALLEL_COPY=true ;;
        --tf5|--pre-tf|--pre-transformers) PRE_TRANSFORMERS=true ;;
        --exp-mxfp4|--experimental-mxfp4) EXP_MXFP4=true ;;
        --exp-b12x|--experimental-b12x) EXP_B12X=true ;;
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
        --apply-preset-vllm-prs) APPLY_PRESET_VLLM_PRS=true ;;
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

# The B12X preset uses the standard Dockerfile and source-build path, but owns
# the fork/ref and Torch-family versions needed by that integration.
if [ "$EXP_B12X" = true ]; then
    if [ "$EXP_MXFP4" = true ]; then echo "Error: --exp-b12x is incompatible with --exp-mxfp4"; exit 1; fi
    if [ "$USE_WHEELS" = true ]; then echo "Error: --exp-b12x is incompatible with --use-wheels because B12X vLLM wheels are not published"; exit 1; fi
    if [ "$VLLM_REPO_SET" = true ]; then echo "Error: --exp-b12x is incompatible with --vllm-repo"; exit 1; fi
    if [ "$VLLM_REF_SET" = true ]; then echo "Error: --exp-b12x is incompatible with --vllm-ref"; exit 1; fi
    if [ "$TORCH_VERSION_SET" = true ]; then echo "Error: --exp-b12x is incompatible with --torch-version"; exit 1; fi
    if [ "$TORCHVISION_VERSION_SET" = true ]; then echo "Error: --exp-b12x is incompatible with --torchvision-version"; exit 1; fi
    if [ "$TORCHAUDIO_VERSION_SET" = true ]; then echo "Error: --exp-b12x is incompatible with --torchaudio-version"; exit 1; fi
    if [ "$PRE_TRANSFORMERS" = true ]; then echo "Error: --exp-b12x is incompatible with --tf5"; exit 1; fi
    if [ "$APPLY_PRESET_VLLM_PRS" = true ]; then echo "Error: --exp-b12x is incompatible with --apply-preset-vllm-prs"; exit 1; fi

    VLLM_REPO="$EXP_B12X_VLLM_REPO"
    VLLM_REF="$EXP_B12X_VLLM_REF"
    TORCH_VERSION="$EXP_B12X_TORCH_VERSION"
    TORCHVISION_VERSION="$EXP_B12X_TORCHVISION_VERSION"
    TORCHAUDIO_VERSION="$EXP_B12X_TORCHAUDIO_VERSION"
    PREBUILT_RUNNER_IMAGE="$PREBUILT_B12X_RUNNER_IMAGE"
fi

# Apply default IMAGE_TAG based on flags if -t was not specified
if [ "$IMAGE_TAG_SET" = false ]; then
    if [ "$PRE_TRANSFORMERS" = true ]; then
        IMAGE_TAG="vllm-node-tf5"
    elif [ "$EXP_MXFP4" = true ]; then
        IMAGE_TAG="vllm-node-mxfp4"
    elif [ "$EXP_B12X" = true ]; then
        IMAGE_TAG="vllm-node-b12x"
    fi
fi

if [ "$PRE_TRANSFORMERS" = true ]; then
    echo "Warning: --tf5/--pre-tf/--pre-transformers is deprecated; vLLM now uses Transformers v5 by default."
    echo "         No Transformers override will be applied; image tag remains $IMAGE_TAG."
fi

CUSTOM_VLLM_REPO=false
if [ "$VLLM_REPO" != "$DEFAULT_VLLM_REPO" ]; then
    CUSTOM_VLLM_REPO=true
fi

NORMALIZED_VLLM_REPO="${VLLM_REPO%/}"
NORMALIZED_VLLM_REPO="${NORMALIZED_VLLM_REPO%.git}"
if [ "$NORMALIZED_VLLM_REPO" = "$EXP_B12X_VLLM_REPO" ]; then
    SPARKINFER_REPO="$EXP_B12X_PACKAGE_REPO"
    SPARKINFER_REF="$EXP_B12X_PACKAGE_REF"
    SPARKINFER_CACHEBUST="$(date +%s)"
    TORCH_BASE_VERSION="${TORCH_VERSION%%+*}"
    if [ "$(printf '%s\n' "2.12.0" "$TORCH_BASE_VERSION" | sort -V | head -n1)" != "2.12.0" ]; then
        echo "Error: ${EXP_B12X_VLLM_REPO} requires --torch-version 2.12.0 or newer for SparkInfer (got ${TORCH_VERSION})."
        exit 1
    fi
    echo "Building SparkInfer from ${SPARKINFER_REPO} ref ${SPARKINFER_REF} for ${NORMALIZED_VLLM_REPO} ref ${VLLM_REF}."
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
    if [ "$VLLM_REPO_SET" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --vllm-repo"; exit 1; fi
    if [ "$VLLM_REF_SET" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --vllm-ref"; exit 1; fi
    if [ "$TORCH_VERSION_SET" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --torch-version"; exit 1; fi
    if [ "$TORCHVISION_VERSION_SET" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --torchvision-version"; exit 1; fi
    if [ "$TORCHAUDIO_VERSION_SET" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --torchaudio-version"; exit 1; fi
    if [ "$FLASHINFER_REF_SET" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --flashinfer-ref"; exit 1; fi
    if [ "$PRE_TRANSFORMERS" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --tf5"; exit 1; fi
    if [ "$REBUILD_FLASHINFER" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --rebuild-flashinfer"; exit 1; fi
    if [ "$REBUILD_VLLM" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --rebuild-vllm"; exit 1; fi
fi

if [ "$EXP_B12X" = true ] && [ "$FORCE_VLLM_DOWNLOAD" = true ]; then
    echo "Error: B12X vLLM wheels are not published; --force-vllm-download cannot be used with --exp-b12x."
    echo "       Use --rebuild-vllm or provide a cached B12X wheel."
    exit 1
fi

# Resolve wheel profiles independently from whether this invocation pulls a
# prebuilt image or performs a local build. Regular FlashInfer wheels are shared
# by the regular and B12X runners. Custom source/ref/architecture builds are
# isolated so they cannot replace the tested shared wheel set.
if [ "$FLASHINFER_REF_SET" = true ] || [ -n "$FLASHINFER_PRS" ] || \
   { [ "$GPU_ARCH_SET" = true ] && [ "$GPU_ARCH_LIST" != "$DEFAULT_GPU_ARCH_LIST" ]; }; then
    FLASHINFER_PROFILE="custom"
fi

if [ "$EXP_B12X" = true ]; then
    VLLM_PROFILE="b12x"
elif [ "$CUSTOM_VLLM_REPO" = true ] || [ "$VLLM_REF_SET" = true ] || \
     [ -n "$VLLM_PRS" ] || \
     [ "$TORCH_VERSION" != "$DEFAULT_TORCH_VERSION" ] || \
     [ "$TORCHVISION_VERSION_SET" = true ] || \
     [ "$TORCHAUDIO_VERSION_SET" = true ] || \
     { [ "$GPU_ARCH_SET" = true ] && [ "$GPU_ARCH_LIST" != "$DEFAULT_GPU_ARCH_LIST" ]; }; then
    VLLM_PROFILE="custom"
fi

FLASHINFER_WHEELS_DIR="$WHEEL_CACHE_ROOT/flashinfer/$FLASHINFER_PROFILE"
VLLM_WHEELS_DIR="$WHEEL_CACHE_ROOT/vllm/$VLLM_PROFILE"

# FlashInfer wheels are architecture-specific for every build flavor. Trust a
# cache only when its marker matches the selected target. An absent marker
# remains compatible with the historical default-architecture cache, but is
# unsafe for alternate targets.
CACHED_FLASHINFER_ARCH=""
if [ -f "$FLASHINFER_WHEELS_DIR/.flashinfer-arch" ]; then
    CACHED_FLASHINFER_ARCH=$(cat "$FLASHINFER_WHEELS_DIR/.flashinfer-arch")
fi
if { [ -n "$CACHED_FLASHINFER_ARCH" ] && [ "$CACHED_FLASHINFER_ARCH" != "$GPU_ARCH_LIST" ]; } || \
   { [ -z "$CACHED_FLASHINFER_ARCH" ] && [ "$GPU_ARCH_LIST" != "$DEFAULT_GPU_ARCH_LIST" ]; }; then
    FLASHINFER_ARCH_MISMATCH=true
fi

CACHED_VLLM_ARCH=""
if [ -f "$VLLM_WHEELS_DIR/.vllm-arch" ]; then
    CACHED_VLLM_ARCH=$(cat "$VLLM_WHEELS_DIR/.vllm-arch")
fi
if { [ -n "$CACHED_VLLM_ARCH" ] && [ "$CACHED_VLLM_ARCH" != "$GPU_ARCH_LIST" ]; } || \
   { [ -z "$CACHED_VLLM_ARCH" ] && [ "$GPU_ARCH_LIST" != "$DEFAULT_GPU_ARCH_LIST" ]; }; then
    VLLM_ARCH_MISMATCH=true
fi

# Validate --no-build usage
if [ "$NO_BUILD" = true ] && [ "${#COPY_HOSTS[@]}" -eq 0 ]; then
    echo "Error: --no-build requires --copy-to to be specified"
    exit 1
fi

# Select image preparation path. By default, use the tested nightly runner image.
# Flags that materially change image contents keep the existing local build path.
VLLM_PR_APPLICATION_REQUESTED=false
if [ -n "$VLLM_PRS" ] || [ "$APPLY_PRESET_VLLM_PRS" = true ]; then
    VLLM_PR_APPLICATION_REQUESTED=true
    REBUILD_VLLM=true
fi

CUSTOM_BUILD_REQUESTED=false
if [ "$EXP_MXFP4" = true ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$GPU_ARCH_SET" = true ] && [ "$GPU_ARCH_LIST" != "$DEFAULT_GPU_ARCH_LIST" ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$VLLM_PROFILE" = "custom" ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$VLLM_REF_SET" = true ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$TORCH_VERSION" != "$DEFAULT_TORCH_VERSION" ] && [ "$EXP_B12X" != true ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$TORCHVISION_VERSION_SET" = true ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$TORCHAUDIO_VERSION_SET" = true ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$FLASHINFER_REF_SET" = true ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$REBUILD_FLASHINFER" = true ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$REBUILD_VLLM" = true ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$FORCE_FLASHINFER_DOWNLOAD" = true ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$FORCE_VLLM_DOWNLOAD" = true ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ -n "$VLLM_PRS" ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ "$APPLY_PRESET_VLLM_PRS" = true ]; then CUSTOM_BUILD_REQUESTED=true; fi
if [ -n "$FLASHINFER_PRS" ]; then CUSTOM_BUILD_REQUESTED=true; fi

# Only local wheel/image builds consume the wheel cache. A normal default invocation
# still pulls the prebuilt runner even if the local wheel cache targets another
# architecture. --use-wheels never compiles implicitly, so reject an unsafe
# cache unless the caller also explicitly requested a FlashInfer rebuild.
if [ "$FLASHINFER_ARCH_MISMATCH" = true ] && \
   { [ "$CUSTOM_BUILD_REQUESTED" = true ] || [ "$USE_WHEELS" = true ]; }; then
    if [ "$USE_WHEELS" = true ] && [ "$REBUILD_FLASHINFER" != true ]; then
        echo "Error: Cached FlashInfer wheels do not match GPU architecture $GPU_ARCH_LIST."
        echo "       Re-run with --rebuild-flashinfer or provide matching wheels with a .flashinfer-arch marker."
        exit 1
    fi
    REBUILD_FLASHINFER=true
    FLASHINFER_ARCH_REBUILD=true
    CUSTOM_BUILD_REQUESTED=true
fi

if [ "$VLLM_ARCH_MISMATCH" = true ] && \
   { [ "$CUSTOM_BUILD_REQUESTED" = true ] || [ "$USE_WHEELS" = true ]; }; then
    if [ "$USE_WHEELS" = true ] && [ "$REBUILD_VLLM" != true ]; then
        echo "Error: Cached $VLLM_PROFILE vLLM wheel does not match GPU architecture $GPU_ARCH_LIST."
        echo "       Re-run with --rebuild-vllm or provide a matching wheel with a .vllm-arch marker."
        exit 1
    fi
    REBUILD_VLLM=true
    VLLM_ARCH_REBUILD=true
    CUSTOM_BUILD_REQUESTED=true
fi

USE_PREBUILT_IMAGE=false
if [ "$NO_BUILD" = false ] && [ "$USE_WHEELS" = false ] && [ "$CUSTOM_BUILD_REQUESTED" = false ]; then
    USE_PREBUILT_IMAGE=true
fi

# Handle cleanup mode
if [[ "$CLEANUP_MODE" == "true" ]]; then
    echo "Cleaning up wheel cache profiles..."
    CACHE_DIRS=(
        "$WHEEL_CACHE_ROOT/flashinfer/regular"
        "$WHEEL_CACHE_ROOT/flashinfer/custom"
        "$WHEEL_CACHE_ROOT/vllm/regular"
        "$WHEEL_CACHE_ROOT/vllm/b12x"
        "$WHEEL_CACHE_ROOT/vllm/custom"
    )
    for cache_dir in "${CACHE_DIRS[@]}"; do
        [ -d "$cache_dir" ] || continue
        rm -f "$cache_dir"/*.whl \
            "$cache_dir"/.*-commit \
            "$cache_dir"/.*-arch
        echo "Cleaned $cache_dir"
    done
    echo "Cleanup complete."
fi

# Ensure the selected named-context directories exist for local builds.
if [ "$NO_BUILD" = false ] && [ "$USE_PREBUILT_IMAGE" != true ]; then
    mkdir -p "$FLASHINFER_WHEELS_DIR" "$VLLM_WHEELS_DIR"
    echo "Using FlashInfer wheel profile: $FLASHINFER_PROFILE ($FLASHINFER_WHEELS_DIR)"
    echo "Using vLLM wheel profile: $VLLM_PROFILE ($VLLM_WHEELS_DIR)"
fi

# Common build flags used across all non-mxfp4 sub-builds
COMMON_BUILD_FLAGS=()
if [ "$FULL_LOG" = true ]; then
    COMMON_BUILD_FLAGS+=("--progress=plain")
fi
COMMON_BUILD_FLAGS+=("--build-arg" "BUILD_JOBS=$BUILD_JOBS")
COMMON_BUILD_FLAGS+=("--build-arg" "TORCH_CUDA_ARCH_LIST=$GPU_ARCH_LIST")
COMMON_BUILD_FLAGS+=("--build-arg" "FLASHINFER_CUDA_ARCH_LIST=$GPU_ARCH_LIST")
if [ "$EXP_MXFP4" = false ]; then
    COMMON_BUILD_FLAGS+=("--build-arg" "TORCH_VERSION=$TORCH_VERSION")
    COMMON_BUILD_FLAGS+=("--build-arg" "TORCHVISION_VERSION=$TORCHVISION_VERSION")
    COMMON_BUILD_FLAGS+=("--build-arg" "TORCHAUDIO_VERSION=$TORCHAUDIO_VERSION")
fi
NCCL_NVCC_GENCODE="$(gpu_arch_to_nccl_gencode "$GPU_ARCH_LIST")"
COMMON_BUILD_FLAGS+=("--build-arg" "NCCL_NVCC_GENCODE=$NCCL_NVCC_GENCODE")
if [ -n "$NETWORK_ARG" ]; then
    COMMON_BUILD_FLAGS+=("--network" "$NETWORK_ARG")
fi

# =====================================================
# Prepare image (unless --no-build)
# =====================================================
FLASHINFER_BUILD_TIME=0
VLLM_BUILD_TIME=0
RUNNER_BUILD_TIME=0
PREBUILT_PULL_TIME=0

if [ "$NO_BUILD" = false ]; then
    if [ "$USE_PREBUILT_IMAGE" = true ]; then
        echo "Using prebuilt runner image ${PREBUILT_RUNNER_IMAGE}..."
        if [ -n "$NETWORK_ARG" ]; then
            echo "Warning: --network is only used for Docker builds; ignoring it while pulling ${PREBUILT_RUNNER_IMAGE}."
        fi
        if [ "$FULL_LOG" = true ]; then
            echo "Warning: --full-log is only used for Docker builds; ignoring it while pulling ${PREBUILT_RUNNER_IMAGE}."
        fi
        if [ "$BUILD_JOBS_SET" = true ]; then
            echo "Warning: --build-jobs is only used for Docker builds; ignoring it while pulling ${PREBUILT_RUNNER_IMAGE}."
        fi

        PULL_START=$(date +%s)
        docker pull "$PREBUILT_RUNNER_IMAGE"
        if [ "$IMAGE_TAG" != "$PREBUILT_RUNNER_IMAGE" ]; then
            docker tag "$PREBUILT_RUNNER_IMAGE" "$IMAGE_TAG"
        fi
        PULL_END=$(date +%s)
        PREBUILT_PULL_TIME=$((PULL_END - PULL_START))
    elif [ "$EXP_MXFP4" = true ]; then
        echo "Building with experimental MXFP4 support..."

        # Generate build metadata YAML for mxfp4 build
        MXFP4_VLLM_SHA=$(grep -m1 '^ARG VLLM_SHA=' Dockerfile.mxfp4 | cut -d= -f2)
        MXFP4_VLLM_REPO=$(grep -m1 '^ARG VLLM_REPO=' Dockerfile.mxfp4 | cut -d= -f2-)
        MXFP4_FLASHINFER_SHA=$(grep -m1 '^ARG FLASHINFER_SHA=' Dockerfile.mxfp4 | cut -d= -f2)
        generate_build_metadata Dockerfile.mxfp4 "unknown" "$MXFP4_VLLM_SHA" "$MXFP4_FLASHINFER_SHA" \
            "mxfp4-pinned" "false" "true" "" "$MXFP4_VLLM_REPO" "base-image" \
            "base-image" "base-image" "disabled"

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
            elif [ "$FLASHINFER_ARCH_REBUILD" = true ]; then
                echo "Rebuilding FlashInfer wheels for GPU architecture $GPU_ARCH_LIST..."
            else
                echo "Rebuilding FlashInfer wheels (--rebuild-flashinfer specified)..."
            fi
            BUILD_FLASHINFER=true
        elif try_download_wheels "$FLASHINFER_RELEASE_TAG" "flashinfer" "$FORCE_FLASHINFER_DOWNLOAD" "$FLASHINFER_WHEELS_DIR"; then
            printf '%s\n' "$GPU_ARCH_LIST" > "$FLASHINFER_WHEELS_DIR/.flashinfer-arch"
            echo "FlashInfer wheels ready."
        elif compgen -G "$FLASHINFER_WHEELS_DIR/flashinfer*.whl" > /dev/null 2>&1; then
            echo "Download failed — using existing local FlashInfer wheels."
        else
            echo "Error: No precompiled FlashInfer wheels are available and the download failed."
            echo "       Re-run with --rebuild-flashinfer to explicitly build FlashInfer from source."
            exit 1
        fi

        if [ "$BUILD_FLASHINFER" = true ]; then
            FLASHINFER_STAGING_DIR=$(mktemp -d "$WHEEL_CACHE_ROOT/.flashinfer-${FLASHINFER_PROFILE}.XXXXXX")

            FI_CMD=("docker" "build"
                "--target" "flashinfer-export"
                "--output" "type=local,dest=$FLASHINFER_STAGING_DIR"
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
            if "${FI_CMD[@]}" && \
               validate_exported_wheel_set "flashinfer" "$FLASHINFER_STAGING_DIR" && \
               promote_wheel_set "$FLASHINFER_STAGING_DIR" "$FLASHINFER_WHEELS_DIR"; then
                FLASHINFER_STAGING_DIR=""
                FI_END=$(date +%s)
                FLASHINFER_BUILD_TIME=$((FI_END - FI_START))
            else
                echo "FlashInfer build failed — keeping the previous wheel profile unchanged."
                exit 1
            fi
        fi

        # ----------------------------------------------------------
        # Phase 2: vLLM wheels
        # ----------------------------------------------------------
        if { [ "$VLLM_PROFILE" = "custom" ] && [ "$USE_WHEELS" != true ]; } || \
           [ "$VLLM_REF_SET" = true ] || [ "$VLLM_PR_APPLICATION_REQUESTED" = true ]; then
            REBUILD_VLLM=true
        fi

        BUILD_VLLM=false
        if [ "$REBUILD_VLLM" = true ]; then
            if [ "$EXP_B12X" = true ] && [ -n "$VLLM_PRS" ]; then
                echo "Rebuilding vLLM wheels (--exp-b12x preset with requested vLLM PRs)..."
            elif [ "$EXP_B12X" = true ]; then
                echo "Rebuilding vLLM wheels (--exp-b12x preset)..."
            elif [ "$VLLM_REF_SET" = true ] && [ "$VLLM_PR_APPLICATION_REQUESTED" = true ]; then
                echo "Rebuilding vLLM wheels (applying vLLM PRs to --vllm-ref $VLLM_REF)..."
            elif [ "$VLLM_REF_SET" = true ]; then
                echo "Rebuilding vLLM wheels (--vllm-ref specified)..."
            elif [ "$CUSTOM_VLLM_REPO" = true ]; then
                echo "Rebuilding vLLM wheels (--vllm-repo specified)..."
            elif [ -n "$VLLM_PRS" ]; then
                echo "Rebuilding vLLM wheels (--apply-vllm-pr specified)..."
            elif [ "$APPLY_PRESET_VLLM_PRS" = true ]; then
                echo "Rebuilding vLLM wheels (--apply-preset-vllm-prs specified)..."
            elif [ "$VLLM_ARCH_REBUILD" = true ]; then
                echo "Rebuilding vLLM wheels for GPU architecture $GPU_ARCH_LIST..."
            else
                echo "Rebuilding vLLM wheels (--rebuild-vllm specified)..."
            fi
            BUILD_VLLM=true
        elif [ "$VLLM_PROFILE" != "regular" ]; then
            if compgen -G "$VLLM_WHEELS_DIR/vllm*.whl" > /dev/null 2>&1; then
                echo "Using cached $VLLM_PROFILE vLLM wheel."
            else
                echo "Error: No cached $VLLM_PROFILE vLLM wheel is available."
                echo "       vLLM wheels for this profile are not downloaded from the regular release."
                echo "       Re-run with --rebuild-vllm to build it from source."
                exit 1
            fi
        elif try_download_wheels "$VLLM_RELEASE_TAG" "vllm" "$FORCE_VLLM_DOWNLOAD" "$VLLM_WHEELS_DIR"; then
            printf '%s\n' "$GPU_ARCH_LIST" > "$VLLM_WHEELS_DIR/.vllm-arch"
            echo "vLLM wheels ready."
        elif compgen -G "$VLLM_WHEELS_DIR/vllm*.whl" > /dev/null 2>&1; then
            echo "Download failed — using existing local vLLM wheels."
        else
            echo "Error: No precompiled vLLM wheels are available and the download failed."
            echo "       Re-run with --rebuild-vllm to explicitly build vLLM from source."
            exit 1
        fi

        if [ "$BUILD_VLLM" = true ]; then
            VLLM_STAGING_DIR=$(mktemp -d "$WHEEL_CACHE_ROOT/.vllm-${VLLM_PROFILE}.XXXXXX")

            VLLM_CMD=("docker" "build"
                "--target" "vllm-export"
                "--output" "type=local,dest=$VLLM_STAGING_DIR"
                "${COMMON_BUILD_FLAGS[@]}"
                "--build-arg" "VLLM_REF=$VLLM_REF"
                "--build-arg" "VLLM_REPO=$VLLM_REPO")

            if [ "$APPLY_PRESET_VLLM_PRS" = true ]; then
                echo "Applying preset vLLM PRs from the Dockerfile (explicitly requested)."
                VLLM_CMD+=("--build-arg" "VLLM_APPLY_PRESET_PRS=1")
            elif [ "$CUSTOM_VLLM_REPO" = true ] || [ "$VLLM_REF_SET" = true ] || [ -n "$VLLM_PRS" ]; then
                echo "Skipping preset vLLM PRs because --vllm-repo, --vllm-ref, or --apply-vllm-pr was specified."
                VLLM_CMD+=("--build-arg" "VLLM_APPLY_PRESET_PRS=0")
            else
                echo "Applying preset vLLM PRs from the Dockerfile by default."
                VLLM_CMD+=("--build-arg" "VLLM_APPLY_PRESET_PRS=1")
            fi

            if [ "$REBUILD_VLLM" = true ]; then
                VLLM_CMD+=("--build-arg" "CACHEBUST_VLLM=$(date +%s)")
            fi

            if [ -n "$VLLM_PRS" ]; then
                echo "Applying vLLM PRs: $VLLM_PRS"
                VLLM_CMD+=("--build-arg" "VLLM_PRS=$VLLM_PRS")
            fi

            if [ "$EXP_B12X" = true ]; then
                # Preserve the selected SM12x target. This prevents 12.1a from
                # being reduced to plain sm_120 without overriding explicit
                # --gpu-arch 12.0a or 12.0f selections.
                VLLM_CMD+=("--build-arg" "VLLM_PRESERVE_SM12X_TARGET=1")
            fi

            VLLM_CMD+=(".")

            echo "vLLM build command: ${VLLM_CMD[*]}"
            VLLM_START=$(date +%s)
            if "${VLLM_CMD[@]}" && \
               validate_exported_wheel_set "vllm" "$VLLM_STAGING_DIR" && \
               promote_wheel_set "$VLLM_STAGING_DIR" "$VLLM_WHEELS_DIR"; then
                VLLM_STAGING_DIR=""
                VLLM_END=$(date +%s)
                VLLM_BUILD_TIME=$((VLLM_END - VLLM_START))
            else
                echo "vLLM build failed — keeping the previous wheel profile unchanged."
                exit 1
            fi
        fi

        # ----------------------------------------------------------
        # Phase 3: Runner image
        # ----------------------------------------------------------
        if ! validate_runner_wheel_inputs "$FLASHINFER_WHEELS_DIR" "$VLLM_WHEELS_DIR"; then
            echo "Error: Selected wheel profiles are incomplete — cannot build runner image."
            exit 1
        fi

        # Generate build metadata YAML
        VLLM_VERSION=$(ls "$VLLM_WHEELS_DIR"/vllm-*.whl 2>/dev/null | head -1 | sed 's|.*/vllm-||;s|-.*||')
        VLLM_COMMIT=""
        [ -f "$VLLM_WHEELS_DIR/.vllm-commit" ] && VLLM_COMMIT=$(cat "$VLLM_WHEELS_DIR/.vllm-commit")
        FLASHINFER_COMMIT=""
        [ -f "$FLASHINFER_WHEELS_DIR/.flashinfer-commit" ] && FLASHINFER_COMMIT=$(cat "$FLASHINFER_WHEELS_DIR/.flashinfer-commit")
        generate_build_metadata Dockerfile "$VLLM_VERSION" "$VLLM_COMMIT" "$FLASHINFER_COMMIT" \
            "$VLLM_REF" "true" "false" "$VLLM_PRS" "$VLLM_REPO" "$TORCH_VERSION" \
            "${TORCHVISION_VERSION:-resolver-selected}" "${TORCHAUDIO_VERSION:-resolver-selected}" \
            "${SPARKINFER_REPO:-disabled}" "${SPARKINFER_REF:-disabled}"

        RUNNER_CMD=("docker" "build"
            "-t" "$IMAGE_TAG"
            "${COMMON_BUILD_FLAGS[@]}"
            "--build-context" "flashinfer_wheels=$FLASHINFER_WHEELS_DIR"
            "--build-context" "vllm_wheels=$VLLM_WHEELS_DIR")

        if [ -n "$SPARKINFER_REPO" ]; then
            RUNNER_CMD+=("--build-arg" "SPARKINFER_REPO=$SPARKINFER_REPO")
            RUNNER_CMD+=("--build-arg" "SPARKINFER_REF=$SPARKINFER_REF")
            RUNNER_CMD+=("--build-arg" "SPARKINFER_CACHEBUST=$SPARKINFER_CACHEBUST")
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
    echo "Checking image '$IMAGE_TAG' on ${#COPY_HOSTS[@]} host(s): ${COPY_HOSTS[*]}"
    COPY_START=$(date +%s)

    if ! LOCAL_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG"); then
        echo "Error: Local image '$IMAGE_TAG' not found."
        exit 1
    fi

    COPY_TARGETS=()
    for host in "${COPY_HOSTS[@]}"; do
        REMOTE_IMAGE_ID=$(get_remote_image_id "$host" "$IMAGE_TAG" || true)
        if [ -n "$REMOTE_IMAGE_ID" ] && [ "$REMOTE_IMAGE_ID" = "$LOCAL_IMAGE_ID" ]; then
            echo "Image '$IMAGE_TAG' is already up to date on ${SSH_USER}@${host}; skipping."
        else
            if [ -n "$REMOTE_IMAGE_ID" ]; then
                echo "Image '$IMAGE_TAG' differs on ${SSH_USER}@${host}; will copy."
            else
                echo "Image '$IMAGE_TAG' not found on ${SSH_USER}@${host}; will copy."
            fi
            COPY_TARGETS+=("$host")
        fi
    done

    if [ "${#COPY_TARGETS[@]}" -eq 0 ]; then
        COPY_END=$(date +%s)
        COPY_TIME=$((COPY_END - COPY_START))
        echo "All remote images are up to date; skipping save/copy."
    else
        echo "Copying image '$IMAGE_TAG' to ${#COPY_TARGETS[@]} host(s): ${COPY_TARGETS[*]}"
        if [ "$PARALLEL_COPY" = true ]; then
            echo "Parallel copy enabled."
        fi

        TMP_IMAGE=$(mktemp -t vllm_image.XXXXXX)
        echo "Saving image locally to $TMP_IMAGE..."
        docker save -o "$TMP_IMAGE" "$IMAGE_TAG"

        if [ "$PARALLEL_COPY" = true ]; then
            PIDS=()
            for host in "${COPY_TARGETS[@]}"; do
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
            for host in "${COPY_TARGETS[@]}"; do
                copy_to_host "$host"
            done
        fi

        COPY_END=$(date +%s)
        COPY_TIME=$((COPY_END - COPY_START))
        echo "Copy complete."
    fi
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
if [ "$PREBUILT_PULL_TIME" -gt 0 ]; then
    echo "Prebuilt Pull:    $(printf '%02d:%02d:%02d' $((PREBUILT_PULL_TIME/3600)) $((PREBUILT_PULL_TIME%3600/60)) $((PREBUILT_PULL_TIME%60)))"
fi
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
echo "Done preparing $IMAGE_TAG."
