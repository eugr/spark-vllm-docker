# syntax=docker/dockerfile:1.6

# Limit build parallelism to reduce OOM situations
ARG BUILD_JOBS=16
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG NCCL_NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG CUTLASS_DSL_VERSION=4.7.0
ARG B12X_REPO=""
ARG B12X_REF=""
ARG B12X_CACHEBUST=""

# =========================================================
# STAGE 1: Base Build Image
# =========================================================
FROM ${CUDA_IMAGE} AS base

ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG CUTLASS_DSL_VERSION

# Build parallemism
ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"
# disable for conflicts with DeepGEMM
ENV DG_JIT_USE_NVRTC=0
ENV USE_CUDNN=1

# Set non-interactive frontend to prevent apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Allow pip to install globally on Ubuntu 24.04 without a venv
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
# Set timeouts
ENV UV_HTTP_TIMEOUT=600
ENV UV_HTTP_RETRIES=10

# Set the base directory environment variable
ENV VLLM_BASE_DIR=/workspace/vllm

# 1. Install Build Dependencies & Ccache
# Added ccache to enable incremental compilation caching
RUN apt update && \
    apt install -y --no-install-recommends \
    curl vim cmake build-essential ninja-build \
    libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
    python3-dev python3-pip git wget \
    libibverbs1 libibverbs-dev rdma-core \
    ccache devscripts debhelper fakeroot \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Additional deps
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     TORCHVISION_SPEC="torchvision" && \
     TORCHAUDIO_SPEC="torchaudio" && \
     if [ -n "$TORCHVISION_VERSION" ]; then TORCHVISION_SPEC="torchvision==$TORCHVISION_VERSION"; fi && \
     if [ "$TORCHAUDIO_VERSION" = "none" ]; then \
         TORCHAUDIO_SPEC=""; \
     elif [ -n "$TORCHAUDIO_VERSION" ]; then \
         TORCHAUDIO_SPEC="torchaudio==$TORCHAUDIO_VERSION"; \
     fi && \
     set -- "torch==$TORCH_VERSION" "$TORCHVISION_SPEC" && \
     if [ -n "$TORCHAUDIO_SPEC" ]; then set -- "$@" "$TORCHAUDIO_SPEC"; fi && \
     uv pip install "$@" triton --index-url https://download.pytorch.org/whl/cu130 && \
     uv pip install nvidia-nvshmem-cu13 \
        "nvidia-cutlass-dsl[cu13]==$CUTLASS_DSL_VERSION" \
        "apache-tvm-ffi==0.1.12" filelock pynvml requests tqdm

# Configure Ccache for CUDA/C++
ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
# Limit ccache size to prevent unbounded growth (e.g. 50G)
ENV CCACHE_MAXSIZE=50G
# Enable compression to save space
ENV CCACHE_COMPRESS=1
# Tell CMake to use ccache for compilation
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# 2. Set Environment Variables
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG NCCL_NVCC_GENCODE
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Setup Workspace
WORKDIR $VLLM_BASE_DIR

# Build NCCL with mesh support (TODO: only do it if arch is 12.1) - artifacts will be in /workspace/nccl/build/pkg/deb
# RUN git clone -b dgxspark-3node-ring https://github.com/zyang-dev/nccl.git && \
#     cd nccl && make -j ${BUILD_JOBS} src.build NVCC_GENCODE="${NCCL_NVCC_GENCODE}" && \
#     make pkg.debian.build && apt install -y --no-install-recommends --allow-downgrades ./build/pkg/deb/*.deb

RUN git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && make -j ${BUILD_JOBS} src.build NVCC_GENCODE="${NCCL_NVCC_GENCODE}" && \
    make pkg.debian.build && apt install -y --no-install-recommends --allow-downgrades --allow-change-held-packages ./build/pkg/deb/*.deb

# =========================================================
# STAGE 2: FlashInfer Builder
# =========================================================
FROM base AS flashinfer-builder

ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR
ARG FLASHINFER_REF=main
ARG FLASHINFER_BUILD_PYTHON=/usr/bin/python3

# FlashInfer's source checkout may carry a .python-version. Keep no-isolation
# builds on the prepared system interpreter instead of letting upstream source
# select a separate uv-managed Python without the installed build dependencies.
ENV UV_PYTHON_DOWNLOADS=never

# --- CACHE BUSTER ---
# Change this argument to force a re-download of FlashInfer
ARG CACHEBUST_FLASHINFER=1

# Additional deps
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install --python "$FLASHINFER_BUILD_PYTHON" packaging filelock

# Smart Git Clone (Fetch changes instead of full re-clone)
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    echo "CACHEBUST_FLASHINFER=${CACHEBUST_FLASHINFER}" && \
    cd /repo-cache && \
    if [ ! -d "flashinfer" ]; then \
        echo "Cache miss: Cloning FlashInfer from scratch..." && \
        git clone --recursive https://github.com/flashinfer-ai/flashinfer.git; \
        if [ "$FLASHINFER_REF" != "main" ]; then \
            cd flashinfer && \
            git checkout ${FLASHINFER_REF}; \
        fi; \
    else \
        echo "Cache hit: Fetching flashinfer updates..." && \
        cd flashinfer && \
        git fetch origin && \
        git fetch origin --tags --force && \
        (git checkout --detach origin/${FLASHINFER_REF} 2>/dev/null || git checkout ${FLASHINFER_REF}) && \
        git reset --hard HEAD && \
        git submodule update --init --recursive && \
        git clean -fdx && \
        git gc --auto; \
    fi && \
    cp -a /repo-cache/flashinfer /workspace/flashinfer

WORKDIR /workspace/flashinfer

ARG FLASHINFER_PRS=""

# PR refs include the branch history they were developed on. Use upstream main
# only to identify each PR's patch range, then apply that patch to FLASHINFER_REF.
RUN set -eux; \
    FLASHINFER_REQUESTED_HEAD="$(git rev-parse HEAD)"; \
    if [ -n "$FLASHINFER_PRS" ]; then \
        # cp -a preserves the source repository's index stat data, but the copied
        # files have different filesystem identities. Refresh before --index apply.
        git update-index --refresh; \
        git config --global user.email "builder@example.com"; \
        git config --global user.name "Docker Builder"; \
        \
        echo "Applying PR patches to FlashInfer ref $FLASHINFER_REF ($FLASHINFER_REQUESTED_HEAD): $FLASHINFER_PRS"; \
        echo "Fetching origin/main only to calculate PR patch ranges; current checkout remains $FLASHINFER_REF."; \
        git fetch origin +refs/heads/main:refs/remotes/origin/main; \
        for pr in $FLASHINFER_PRS; do \
            echo "Fetching PR #$pr and applying its patch onto current HEAD..."; \
            git fetch origin +pull/${pr}/head:pr-${pr}; \
            pr_base="$(git merge-base origin/main pr-${pr} || true)"; \
            if [ -z "$pr_base" ]; then \
                echo "Unable to find an origin/main merge-base for FlashInfer PR #$pr."; \
                exit 1; \
            fi; \
            patch_file="/tmp/flashinfer-pr-${pr}.patch"; \
            echo "FlashInfer PR #$pr patch range: $pr_base..pr-${pr}; apply target: $(git rev-parse HEAD)."; \
            git diff --binary "$pr_base" "pr-${pr}" > "$patch_file"; \
            if [ ! -s "$patch_file" ]; then \
                echo "FlashInfer PR #$pr has no patch relative to origin/main; skipping."; \
                rm -f "$patch_file"; \
                continue; \
            fi; \
            if git apply --reverse --check --binary "$patch_file" >/dev/null 2>&1; then \
                echo "FlashInfer PR #$pr patch is already applied to HEAD; skipping."; \
                rm -f "$patch_file"; \
                continue; \
            fi; \
            if git apply --3way --index --binary "$patch_file"; then \
                if git diff --cached --quiet; then \
                    echo "FlashInfer PR #$pr patch produced no staged changes; skipping."; \
                else \
                    git commit -m "Apply FlashInfer PR #${pr}"; \
                fi; \
                rm -f "$patch_file"; \
            else \
                conflict_files="$(git diff --name-only --diff-filter=U)"; \
                if [ -n "$conflict_files" ]; then \
                    echo "FlashInfer PR #$pr has patch conflicts: $conflict_files"; \
                else \
                    echo "FlashInfer PR #$pr patch failed without unmerged files."; \
                fi; \
                rm -f "$patch_file"; \
                git reset --hard HEAD; \
                exit 1; \
            fi; \
        done; \
        if ! git merge-base --is-ancestor "$FLASHINFER_REQUESTED_HEAD" HEAD; then \
            echo "Requested FlashInfer ref $FLASHINFER_REF ($FLASHINFER_REQUESTED_HEAD) is not an ancestor of final HEAD $(git rev-parse HEAD) after PR application."; \
            exit 1; \
        fi; \
        echo "Final FlashInfer source after PR application: requested $FLASHINFER_REF ($FLASHINFER_REQUESTED_HEAD), final $(git describe --tags --always --dirty)."; \
    fi



# Apply patch to avoid re-downloading existing cubins
COPY flashinfer_cache.patch .
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=cubins-cache,target=/workspace/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins \
    patch -p1 < flashinfer_cache.patch && \
    # flashinfer-python
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    "$FLASHINFER_BUILD_PYTHON" -c 'import filelock, packaging, requests, torch, tqdm' && \
    uv build --python "$FLASHINFER_BUILD_PYTHON" --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    # flashinfer-cubin
    cd flashinfer-cubin && uv build --python "$FLASHINFER_BUILD_PYTHON" --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    # flashinfer-jit-cache
    cd ../flashinfer-jit-cache && \
    uv build --python "$FLASHINFER_BUILD_PYTHON" --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    # dump git ref and target architecture in the wheels dir
    cd .. && \
    git rev-parse HEAD > /workspace/wheels/.flashinfer-commit && \
    printf '%s\n' "$FLASHINFER_CUDA_ARCH_LIST" > /workspace/wheels/.flashinfer-arch

# =========================================================
# STAGE 3: FlashInfer Wheel Export
# =========================================================
FROM scratch AS flashinfer-export
COPY --from=flashinfer-builder /workspace/wheels /

# =========================================================
# STAGE 4: vLLM Builder
# =========================================================
FROM base AS vllm-builder
ARG RUSTUP_TOOLCHAIN=stable
ARG CUTLASS_DSL_VERSION
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
ENV PATH=/opt/cargo/bin:$PATH
ENV PROTOC_INCLUDE=/usr/include

RUN apt update && \
    apt install -y --no-install-recommends ca-certificates pkg-config protobuf-compiler libprotobuf-dev && \
    rm -rf /var/lib/apt/lists/* && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --profile minimal --default-toolchain ${RUSTUP_TOOLCHAIN} --no-modify-path && \
    rustc --version && \
    cargo --version

ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR

# --- VLLM SOURCE CACHE BUSTER ---
ARG CACHEBUST_VLLM=1

# Git reference (branch, tag, or SHA) to checkout
ARG VLLM_UPSTREAM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=main

# Pinned while investigating an SM121 DeepSeek-V4 MXFP4 grouped scale-factor
# regression first observed at nv_dev f8e8fb5 (PR #384); last known good.
ARG DEEPGEMM_REPO=https://github.com/deepseek-ai/DeepGEMM.git
ARG DEEPGEMM_REF=a6b593d2826719dcf4892609af7b84ee23aaf32a
ENV DEEPGEMM_SRC_DIR=/workspace/DeepGEMM

# The upstream repository uses the shared checkout cache. Custom repositories
# are cloned outside it so a fork can never reuse or mutate the upstream clone.
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    set -eux; \
    echo "CACHEBUST_VLLM=${CACHEBUST_VLLM}"; \
    if [ "$VLLM_REPO" != "$VLLM_UPSTREAM_REPO" ]; then \
        echo "Custom vLLM repository selected; bypassing shared checkout cache."; \
        git clone --recursive "$VLLM_REPO" /tmp/vllm-custom; \
        cd /tmp/vllm-custom; \
        if git rev-parse --verify --quiet "refs/remotes/origin/$VLLM_REF"; then \
            git checkout --detach "origin/$VLLM_REF"; \
        else \
            git checkout --detach "$VLLM_REF"; \
        fi; \
        git submodule update --init --recursive; \
        cp -a /tmp/vllm-custom "$VLLM_BASE_DIR/vllm"; \
    else \
        cd /repo-cache; \
        if [ ! -d "vllm" ]; then \
            echo "Cache miss: Cloning vLLM from scratch..."; \
            git clone --recursive "$VLLM_REPO" vllm; \
        else \
            echo "Cache hit: Fetching updates..."; \
            cd vllm; \
            git fetch origin; \
            git fetch origin --tags --force; \
            cd ..; \
        fi; \
        cd vllm; \
        if git rev-parse --verify --quiet "refs/remotes/origin/$VLLM_REF"; then \
            git checkout --detach "origin/$VLLM_REF"; \
        else \
            git checkout --detach "$VLLM_REF"; \
        fi; \
        git reset --hard HEAD; \
        git submodule update --init --recursive; \
        git clean -fdx; \
        git gc --auto; \
        cp -a /repo-cache/vllm "$VLLM_BASE_DIR/"; \
    fi

RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    set -eux; \
    cd /repo-cache; \
    if [ ! -d "deepgemm" ]; then \
        echo "Cache miss: Cloning DeepGEMM from scratch..."; \
        git clone --recursive "$DEEPGEMM_REPO" deepgemm; \
    else \
        echo "Cache hit: Fetching DeepGEMM updates..."; \
        cd deepgemm; \
        git fetch origin; \
        git fetch origin --tags --force; \
        cd ..; \
    fi; \
    cd deepgemm; \
    git checkout --detach "$DEEPGEMM_REF" 2>/dev/null || git checkout --detach "origin/$DEEPGEMM_REF"; \
    git reset --hard; \
    git submodule update --init --recursive; \
    git clean -fdx; \
    rm -rf "$DEEPGEMM_SRC_DIR"; \
    cp -a /repo-cache/deepgemm "$DEEPGEMM_SRC_DIR"

WORKDIR $VLLM_BASE_DIR/vllm

# Optional upstream PR patches requested by the build wrapper. PR #47392 is
# carried as a source-aware runtime patch below because its full diff now
# conflicts with current upstream main.
ARG VLLM_PRESET_PRS=""
ARG VLLM_APPLY_PRESET_PRS=""
ARG VLLM_PRS=""
ARG VLLM_PRESERVE_SM12X_TARGET=0
ARG VLLM_PATCH_B12X_C128A_ALIGNMENT=0

# PR refs include the branch history they were developed on. Use upstream main
# only to identify each PR's patch range, then apply that patch to VLLM_REF.
RUN set -eux; \
    VLLM_ALL_PRS=""; \
    VLLM_SELECTED_PRESET_PRS=""; \
    VLLM_REQUESTED_HEAD="$(git rev-parse HEAD)"; \
    case "$VLLM_APPLY_PRESET_PRS" in \
        1|true|TRUE|yes|YES) VLLM_SELECTED_PRESET_PRS="$VLLM_PRESET_PRS";; \
        0|false|FALSE|no|NO) VLLM_SELECTED_PRESET_PRS="";; \
        ""|auto|AUTO) \
            if [ -z "$VLLM_PRS" ]; then \
                if [ "$VLLM_REF" = "main" ]; then \
                    VLLM_SELECTED_PRESET_PRS="$VLLM_PRESET_PRS"; \
                else \
                    echo "Skipping preset vLLM PRs in auto mode because VLLM_REF=$VLLM_REF is not main."; \
                fi; \
            fi;; \
        *) echo "Invalid VLLM_APPLY_PRESET_PRS value: $VLLM_APPLY_PRESET_PRS"; exit 1;; \
    esac; \
    for pr in $VLLM_SELECTED_PRESET_PRS $VLLM_PRS; do \
        case " $VLLM_ALL_PRS " in \
            *" $pr "*) ;; \
            *) VLLM_ALL_PRS="${VLLM_ALL_PRS:+$VLLM_ALL_PRS }$pr";; \
        esac; \
    done; \
    if [ -n "$VLLM_ALL_PRS" ]; then \
        # cp -a preserves the source repository's index stat data, but the copied
        # files have different filesystem identities. Refresh before --index apply.
        git update-index --refresh; \
        git config --global user.email "builder@example.com"; \
        git config --global user.name "Docker Builder"; \
        \
        echo "Applying PR patches to vLLM ref $VLLM_REF ($VLLM_REQUESTED_HEAD): $VLLM_ALL_PRS"; \
        echo "Fetching upstream main only to calculate PR patch ranges; current checkout remains $VLLM_REF."; \
        git remote remove vllm-upstream >/dev/null 2>&1 || true; \
        git remote add vllm-upstream "$VLLM_UPSTREAM_REPO"; \
        git fetch vllm-upstream +refs/heads/main:refs/remotes/vllm-upstream/main; \
        for pr in $VLLM_ALL_PRS; do \
            echo "Fetching PR #$pr and applying its patch onto current HEAD..."; \
            git fetch vllm-upstream +pull/${pr}/head:pr-${pr}; \
            pr_base="$(git merge-base vllm-upstream/main pr-${pr} || true)"; \
            if [ -z "$pr_base" ]; then \
                echo "Unable to find an upstream main merge-base for PR #$pr."; \
                exit 1; \
            fi; \
            patch_file="/tmp/pr-${pr}.patch"; \
            echo "PR #$pr patch range: $pr_base..pr-${pr}; apply target: $(git rev-parse HEAD)."; \
            git diff --binary "$pr_base" "pr-${pr}" > "$patch_file"; \
            if [ ! -s "$patch_file" ]; then \
                echo "PR #$pr has no patch relative to upstream main; skipping."; \
                rm -f "$patch_file"; \
                continue; \
            fi; \
            if git apply --reverse --check --binary "$patch_file" >/dev/null 2>&1; then \
                echo "PR #$pr patch is already applied to HEAD; skipping."; \
                rm -f "$patch_file"; \
                continue; \
            fi; \
            if git apply --3way --index --binary "$patch_file"; then \
                if git diff --cached --quiet; then \
                    echo "PR #$pr patch produced no staged changes; skipping."; \
                else \
                    git commit -m "Apply vLLM PR #${pr}"; \
                fi; \
                rm -f "$patch_file"; \
            else \
                conflict_files="$(git diff --name-only --diff-filter=U)"; \
                code_conflicts=""; \
                for conflict_file in $conflict_files; do \
                    case "$conflict_file" in \
                        tests/*|docs/*|*.md|*.rst) ;; \
                        *) code_conflicts="${code_conflicts:+$code_conflicts }$conflict_file";; \
                    esac; \
                done; \
                if [ -z "$conflict_files" ]; then \
                    echo "PR #$pr patch failed without unmerged files."; \
                    rm -f "$patch_file"; \
                    git reset --hard HEAD; \
                    exit 1; \
                fi; \
                if [ -n "$code_conflicts" ]; then \
                    echo "PR #$pr has code patch conflicts: $code_conflicts"; \
                    rm -f "$patch_file"; \
                    git reset --hard HEAD; \
                    exit 1; \
                fi; \
                echo "Skipping tests/docs conflicts for PR #$pr: $conflict_files"; \
                for conflict_file in $conflict_files; do \
                    git checkout --ours -- "$conflict_file"; \
                    git add "$conflict_file"; \
                done; \
                if git diff --cached --quiet; then \
                    echo "PR #$pr only changed conflicting tests/docs files; skipping."; \
                    git reset --hard HEAD; \
                else \
                    git commit -m "Apply vLLM PR #${pr}"; \
                fi; \
                rm -f "$patch_file"; \
            fi; \
        done; \
        if ! git merge-base --is-ancestor "$VLLM_REQUESTED_HEAD" HEAD; then \
            echo "Requested vLLM ref $VLLM_REF ($VLLM_REQUESTED_HEAD) is not an ancestor of final HEAD $(git rev-parse HEAD) after PR application."; \
            exit 1; \
        fi; \
        echo "Final vLLM source after PR application: requested $VLLM_REF ($VLLM_REQUESTED_HEAD), final $(git describe --tags --always --dirty)."; \
    fi

# Targeted production subset of vLLM PR #47392. This patches the upstream
# FlashInfer B12x backend without carrying the PR's conflicting tests/docs.
# It is also safe for older refs (backend absent) and refs that already contain
# the fix (idempotent); unknown partial source shapes fail the build.
COPY docker/patch_vllm_*.py docker/pin_cutlass_dsl.py /tmp/vllm-patches/

# TEMPORARY PATCH: local-inference-lab/vllm commit ad848fc41 added a dynamic
# DeepSeek V4 C128A top-k width but omitted the alignment constant import.
# Keep this B12X-only and source-aware so it skips refs where the bug is absent
# or already fixed. Remove after the oldest supported B12X ref contains a fix.
RUN VLLM_PATCH_B12X_C128A_ALIGNMENT="${VLLM_PATCH_B12X_C128A_ALIGNMENT}" \
    python3 /tmp/vllm-patches/patch_vllm_b12x_c128a_topk_alignment.py .

RUN python3 /tmp/vllm-patches/patch_vllm_flashinfer_b12x_swigluoai.py .

# TEMPORARY PATCH: vLLM PR #49408 / commit d6dbdb9 misplaced the XPU-only
# return in topk_hash_softplus_sqrt, making the CUDA/ROCm kernel call dead code.
# Remove after upstream fix PR #49452 is merged and present in the oldest
# supported VLLM_REF. Inspect source shape rather than commit ancestry so this
# also handles rebases, cherry-picks, and builds that already include the fix.
RUN python3 /tmp/vllm-patches/patch_vllm_topk_softplus_sqrt_control_flow.py .

# CUDA 13 vLLM builds normally collapse requested subarchitectures to generic
# family entries: 10.3a becomes 10.0 and 12.1a becomes 12.0. Add 10.3 and 12.1
# to the supported set so CMake preserves the selected target and its a/f
# suffix. Keep this opt-in so the standard upstream build retains its own
# architecture policy.
RUN VLLM_PRESERVE_SM12X_TARGET="${VLLM_PRESERVE_SM12X_TARGET}" \
    python3 /tmp/vllm-patches/patch_vllm_preserve_sm12x_target.py .

# TEMPORARY PATCH: vLLM PR #47914 added per-KV-group causal metadata by
# treating non-bool causal as Mapping[int, bool]. DiffusionGemma passes a
# per-request torch.Tensor causal mask and crashes on causal.get(...). Keep this
# until upstream build_attn_metadata accepts Tensor causal again.
RUN python3 /tmp/vllm-patches/patch_vllm_diffusion_tensor_causal.py .

# TEMPORARY PATCH: vLLM PR #43957 added a generic embedding-width guard for
# EAGLE3, but Gemma4 MTP intentionally replaces its draft embedding with the
# target backbone embedding before pre_projection. Without sharing, Gemma4 MTP
# concatenates 1024-wide draft embeddings with 2816-wide backbone hidden states
# and crashes in a 5632-wide pre_projection. Keep the guard scoped to EAGLE-style
# draft models until upstream fixes https://github.com/vllm-project/vllm/issues/47794.
RUN python3 /tmp/vllm-patches/patch_vllm_gemma4_mtp_embedding_share.py .

# TEMPORARY PATCH (source build only): vLLM PR #43008 selects cooperative_topk
# for all SM90+ devices. On DGX Spark / SM12.x this fails at launch with
# "cooperative_topk launch failed: invalid argument". Keep the cooperative
# path on SM90 and let newer architectures use the existing persistent_topk fallback.
RUN python3 /tmp/vllm-patches/patch_vllm_sm120_cooperative_topk.py .

# TEMPORARY PATCH: vLLM PR #43409 started passing AutoGPTQ MoE qzeros
# through even for symmetric GPTQ. On CUDA Marlin MoE this can select the
# wrong zero-point kernel path and crash Qwen3-Coder-Next AutoRound during
# startup. Apply only when the vulnerable upstream pattern is present.
RUN python3 /tmp/vllm-patches/patch_vllm_autogptq_symmetric_moe_qzeros.py .

# # TEMPORARY PATCH for broken FP8 kernels - https://github.com/vllm-project/vllm/pull/35568
# RUN curl -fsL https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/35568.diff -o pr35568.diff \
#     && if git apply --reverse --check pr35568.diff 2>/dev/null; then \
#          echo "PR 35568 already applied, skipping."; \
#        else \
#          echo "Applying PR 35568..."; \
#          git apply -v --exclude="tests/*" pr35568.diff; \
#        fi \
#     && rm pr35568.diff

# TEMPORARY PATCH: revert vLLM PR #41524 / commit c51df430,
# which disables FlashInfer autotune and regresses DGX Spark throughput.
# RUN set -eux; \
#     patch_commit="c51df43005726a09c6eb7348e8c1b00501c70a8e"; \
#     target="vllm/config/vllm.py"; \
#     marker="https://github.com/flashinfer-ai/flashinfer/issues/3197"; \
#     if grep -q "$marker" "$target"; then \
#         echo "PR #41524 regression found; reverting ${patch_commit}"; \
#         if ! git revert --no-commit "$patch_commit"; then \
#             git revert --abort 2>/dev/null || true; \
#             echo "ERROR: PR #41524 appears present but could not be reverted"; \
#             exit 1; \
#         fi; \
#         if grep -q "$marker" "$target"; then \
#             echo "ERROR: revert completed but PR #41524 marker is still present"; \
#             exit 1; \
#         fi; \
#     else \
#         echo "PR #41524 regression marker not present; skipping revert"; \
#     fi

# TEMPORARY PATCH: disable the MiniMax QK RMSNorm CUDA IPC fusion from vLLM
# PR #43410. A full git revert now conflicts with current upstream, and the
# runtime failure happens while allocating the Lamport workspace.
RUN python3 /tmp/vllm-patches/patch_vllm_disable_minimax_qk_rmsnorm_ipc.py .

# TEMPORARY PATCH: vLLM PR #43362 made RoutedExperts scalarize all
# _load_single_value() inputs. That is correct for scalar input scales, but
# compressed-tensors MoE checkpoints also load 2-element weight_shape metadata
# through this path. Preserve vector metadata when the destination slot matches.
RUN python3 /tmp/vllm-patches/patch_vllm_routed_experts_weight_shape.py .

# DGX Spark UMA cleanup: profile warmup can leave temporary CUDA allocator
# reservations behind just before vLLM sizes and allocates KV cache blocks.
RUN python3 /tmp/vllm-patches/patch_vllm_spark_kv_cache_cleanup.py .


# Prepare build requirements
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 /tmp/vllm-patches/pin_cutlass_dsl.py \
        "$CUTLASS_DSL_VERSION" --expected-count 1 requirements/cuda.txt && \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test/cuda.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test/cuda.txt && \
    uv pip install -r requirements/build/cuda.txt "setuptools-rust>=1.9.0"

# Apply Patches
# TEMPORARY PATCH for fastsafetensors loading in cluster setup - tracking https://github.com/vllm-project/vllm/issues/34180
# COPY fastsafetensors.patch .
# RUN if patch -p1 --dry-run --reverse < fastsafetensors.patch &>/dev/null; then \
#         echo "PR #34180 is already applied"; \
#     else \
#         patch -p1 < fastsafetensors.patch; \
#     fi
# TEMPORARY PATCH for broken vLLM build (unguarded Hopper code) - reverting PR #34758 and #34302
# RUN curl -L https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/34758.diff | patch -p1 -R || echo "Cannot revert PR #34758, skipping"
# RUN curl -L https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/34302.diff | patch -p1 -R || echo "Cannot revert PR #34302, skipping"

# Final Compilation
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=cargo-registry,target=/opt/cargo/registry \
    --mount=type=cache,id=cargo-git,target=/opt/cargo/git \
    --mount=type=cache,id=vllm-rust-target,target=/workspace/vllm/vllm/target \
    VLLM_REQUIRE_RUST_FRONTEND=1 CARGO_BUILD_JOBS=${MAX_JOBS} \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

# Dump git refs in the wheels dir.
RUN \
    git rev-parse HEAD > /workspace/wheels/.vllm-commit && \
    git -C "$DEEPGEMM_SRC_DIR" rev-parse HEAD > /workspace/wheels/.deepgemm-commit && \
    printf '%s\n' "$TORCH_CUDA_ARCH_LIST" > /workspace/wheels/.vllm-arch

# =========================================================
# STAGE 5: vLLM Wheel Export
# =========================================================
FROM scratch AS vllm-export
COPY --from=vllm-builder /workspace/wheels /

# =========================================================
# STAGE 6: Runner (Installs wheels from selected named contexts)
# =========================================================
FROM ${CUDA_IMAGE} AS runner

ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG CUTLASS_DSL_VERSION
ARG B12X_REPO
ARG B12X_REF
ARG B12X_CACHEBUST

# Transferring build settings from build image because of ptxas/jit compilation during vLLM startup
# Build parallemism
ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"
# For compatibility with DeepGEMM changes
ENV DG_JIT_USE_NVRTC=0
ENV USE_CUDNN=1

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV VLLM_BASE_DIR=/workspace/vllm

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# Mount additional packages from base builder image
# Install runtime dependencies
RUN --mount=type=bind,from=base,source=/workspace/vllm/nccl/build/pkg/deb,target=/workspace/nccl-pkg \
    apt update && \
    apt install -y --no-install-recommends \
    python3 python3-pip python3-dev vim curl git wget \
    libcudnn9-cuda-13 \
    libibverbs1 libibverbs-dev rdma-core \
    libxcb1 earlyoom \
    && cd /workspace/nccl-pkg && apt install -y --no-install-recommends --allow-downgrades --allow-change-held-packages ./*.deb \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Set final working directory
WORKDIR $VLLM_BASE_DIR

# Download Tiktoken files
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

ARG PRE_TRANSFORMERS=0

# Install deps
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     TORCHVISION_SPEC="torchvision" && \
     TORCHAUDIO_SPEC="torchaudio" && \
     if [ -n "$TORCHVISION_VERSION" ]; then TORCHVISION_SPEC="torchvision==$TORCHVISION_VERSION"; fi && \
     if [ "$TORCHAUDIO_VERSION" = "none" ]; then \
         TORCHAUDIO_SPEC=""; \
     elif [ -n "$TORCHAUDIO_VERSION" ]; then \
         TORCHAUDIO_SPEC="torchaudio==$TORCHAUDIO_VERSION"; \
     fi && \
     set -- "torch==$TORCH_VERSION" "$TORCHVISION_SPEC" && \
     if [ -n "$TORCHAUDIO_SPEC" ]; then set -- "$@" "$TORCHAUDIO_SPEC"; fi && \
     uv pip install "$@" triton --index-url https://download.pytorch.org/whl/cu130 && \
     uv pip install nvidia-nvshmem-cu13 \
        "nvidia-cutlass-dsl[cu13]==$CUTLASS_DSL_VERSION" \
        "apache-tvm-ffi==0.1.12"

# Install the shared/selected FlashInfer and vLLM profiles from independent
# named contexts (bind-mounted without adding the wheel files to an image layer).
# PRE_TRANSFORMERS=1 is retained for manual legacy builds; build-and-copy.sh no longer sets it for --tf5.
# FastAPI 0.137.0 adds _IncludedRouter entries that currently break
# prometheus-fastapi-instrumentator route name lookup.
# quack-kernels 0.6.4 still declares CUTLASS DSL 4.6.2, so this solve
# deliberately overrides that transitive constraint with the image-wide pin.
RUN --mount=type=bind,from=flashinfer_wheels,target=/workspace/flashinfer-wheels \
    --mount=type=bind,from=vllm_wheels,target=/workspace/vllm-wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    PINNED_TORCH=$(python3 -c "import torch; print(torch.__version__)") && \
    PINNED_TORCHVISION=$(python3 -c "import importlib.metadata as m; print(m.version('torchvision'))") && \
    PINNED_TORCHAUDIO=$(python3 -c "import importlib.metadata as m; print(m.version('torchaudio'))" 2>/dev/null || true) && \
    echo "torch==${PINNED_TORCH}" > /tmp/wheel-override.txt && \
    echo "torchvision==${PINNED_TORCHVISION}" >> /tmp/wheel-override.txt && \
    if [ -n "$PINNED_TORCHAUDIO" ]; then echo "torchaudio==${PINNED_TORCHAUDIO}" >> /tmp/wheel-override.txt; fi && \
    echo "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" >> /tmp/wheel-override.txt && \
    echo "fastapi[standard]>=0.115.0,<0.137.0" >> /tmp/wheel-override.txt && \
    if [ "$PRE_TRANSFORMERS" = "1" ]; then \
        echo "transformers>=5.0.0" >> /tmp/wheel-override.txt; \
    fi && \
    uv pip install /workspace/flashinfer-wheels/*.whl /workspace/vllm-wheels/*.whl \
        --override /tmp/wheel-override.txt

# Setup environment for runtime
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH


# Final extra deps
# Pin torch and CUTLASS DSL via --override so transitive dependencies cannot
# trigger an upgrade/downgrade or swap CUDA-built torch for PyPI's CPU wheel.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    PINNED_TORCH=$(python3 -c "import torch; print(torch.__version__)") && \
    PINNED_TORCHVISION=$(python3 -c "import importlib.metadata as m; print(m.version('torchvision'))") && \
    PINNED_TORCHAUDIO=$(python3 -c "import importlib.metadata as m; print(m.version('torchaudio'))" 2>/dev/null || true) && \
    echo "torch==${PINNED_TORCH}" > /tmp/torch-override.txt && \
    echo "torchvision==${PINNED_TORCHVISION}" >> /tmp/torch-override.txt && \
    if [ -n "$PINNED_TORCHAUDIO" ]; then echo "torchaudio==${PINNED_TORCHAUDIO}" >> /tmp/torch-override.txt; fi && \
    echo "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" >> /tmp/torch-override.txt && \
    echo "fastapi[standard]>=0.115.0,<0.137.0" >> /tmp/torch-override.txt && \
    uv pip install ray[default] fastsafetensors instanttensor \
        --override /tmp/torch-override.txt

# The local-inference-lab vLLM fork consumes the external B12X kernel package
# at runtime. Keep this opt-in so ordinary vLLM images do not pull a
# package that requires Torch 2.12+. Build B12X from its source repository but
# install it without dependencies: vLLM already provides the runtime packages
# and this image deliberately advances nvidia-cutlass-dsl to 4.7.0 for both
# regular and B12X builds. B12X kernels remain JIT-compiled on first use;
# building its Python wheel here does not compile the CUDA kernels.
COPY docker/pin_cutlass_dsl.py /tmp/pin_cutlass_dsl.py
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    if [ -n "$B12X_REPO" ]; then \
        echo "Refreshing B12X source (cache key: $B12X_CACHEBUST)" && \
        git clone --depth 1 --branch "$B12X_REF" "$B12X_REPO" /tmp/b12x-source && \
        B12X_COMMIT=$(git -C /tmp/b12x-source rev-parse HEAD) && \
        python3 /tmp/pin_cutlass_dsl.py "$CUTLASS_DSL_VERSION" \
            --expected-count 5 /tmp/b12x-source/pyproject.toml && \
        uv pip install --reinstall --no-deps /tmp/b12x-source && \
        printf '%s\n' "$B12X_COMMIT" > /workspace/b12x-source-commit && \
        python3 -c "import importlib.metadata as m, sys; import b12x; print('Verified B12X', m.version('b12x'), 'from source commit', sys.argv[1], 'with CUTLASS DSL', m.version('nvidia-cutlass-dsl'))" "$B12X_COMMIT" && \
        rm -rf /tmp/b12x-source; \
    else \
        echo "B12X source build not requested; skipping."; \
    fi

# Fix NCCL
RUN rm /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnccl.so.2 /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
    
# Build metadata (generated by build-and-copy.sh)
COPY build-metadata.yaml /workspace/build-metadata.yaml
