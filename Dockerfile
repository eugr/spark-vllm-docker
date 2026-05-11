# syntax=docker/dockerfile:1.6

# Limit build parallelism to reduce OOM situations
ARG BUILD_JOBS=16

# =========================================================
# STAGE 1: Base Build Image
# =========================================================
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04 AS base

# Build parallemism
ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"
ENV DG_JIT_USE_NVRTC=1
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
     uv pip install torch==2.11.0 torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu130 && \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2" filelock pynvml requests tqdm

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
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Setup Workspace
WORKDIR $VLLM_BASE_DIR

# Build NCCL with mesh support (TODO: only do it if arch is 12.1) - artifacts will be in /workspace/nccl/build/pkg/deb
RUN git clone -b dgxspark-3node-ring https://github.com/zyang-dev/nccl.git && \
    cd nccl && make -j ${BUILD_JOBS} src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" && \
    make pkg.debian.build && apt install -y --no-install-recommends --allow-downgrades ./build/pkg/deb/*.deb

# =========================================================
# STAGE 2: FlashInfer Builder
# =========================================================
FROM base AS flashinfer-builder

ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR
ARG FLASHINFER_REF=main

# --- CACHE BUSTER ---
# Change this argument to force a re-download of FlashInfer
ARG CACHEBUST_FLASHINFER=1

# Additional deps
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install packaging

# Smart Git Clone (Fetch changes instead of full re-clone)
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
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
        git submodule update --init --recursive && \
        git clean -fdx && \
        git gc --auto; \
    fi && \
    cp -a /repo-cache/flashinfer /workspace/flashinfer

WORKDIR /workspace/flashinfer

ARG FLASHINFER_PRS=""

RUN if [ -n "$FLASHINFER_PRS" ]; then \
        # Git requires a user identity to create merge commits
        git config --global user.email "builder@example.com"; \
        git config --global user.name "Docker Builder"; \
        \
        echo "Applying PRs: $FLASHINFER_PRS"; \
        for pr in $FLASHINFER_PRS; do \
            echo "Fetching and merging PR #$pr..."; \
            git fetch origin pull/${pr}/head:pr-${pr}; \
            git merge pr-${pr} --no-edit; \
        done; \
    fi

# TEMPORARY patch for flashinfer autotune and other improvements (PR 2927) - MERGED 4/3
# RUN curl -fsL https://github.com/flashinfer-ai/flashinfer/pull/2927.diff -o pr2927.diff \
#     && if git apply --reverse --check pr2927.diff 2>/dev/null; then \
#          echo "PR #2927 already applied, skipping."; \
#        else \
#          echo "Applying FI PR #2927..."; \
#          git apply -v pr2927.diff; \
#        fi \
#     && rm pr2927.diff

# Apply patch to avoid re-downloading existing cubins
COPY flashinfer_cache.patch .
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=cubins-cache,target=/workspace/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins \
    patch -p1 < flashinfer_cache.patch && \
    # flashinfer-python
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    # flashinfer-cubin
    cd flashinfer-cubin && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    # flashinfer-jit-cache
    cd ../flashinfer-jit-cache && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    # dump git ref in the wheels dir
    cd .. && git rev-parse HEAD > /workspace/wheels/.flashinfer-commit

# =========================================================
# STAGE 3: FlashInfer Wheel Export
# =========================================================
FROM scratch AS flashinfer-export
COPY --from=flashinfer-builder /workspace/wheels /

# =========================================================
# STAGE 4: vLLM Builder
# =========================================================
FROM base AS vllm-builder

ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR

# --- VLLM SOURCE CACHE BUSTER ---
ARG CACHEBUST_VLLM=1

# Git reference (branch, tag, or SHA) to checkout
ARG VLLM_REF=main

# Smart Git Clone (Fetch changes instead of full re-clone)
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "vllm" ]; then \
        echo "Cache miss: Cloning vLLM from scratch..." && \
        git clone --recursive https://github.com/vllm-project/vllm.git; \
        if [ "$VLLM_REF" != "main" ]; then \
            cd vllm && \
            git checkout ${VLLM_REF}; \
        fi; \
    else \
        echo "Cache hit: Fetching updates..." && \
        cd vllm && \
        git fetch origin && \
        git fetch origin --tags --force && \
        (git checkout --detach origin/${VLLM_REF} 2>/dev/null || git checkout ${VLLM_REF}) && \
        git submodule update --init --recursive && \
        git clean -fdx && \
        git gc --auto; \
    fi && \
    cp -a /repo-cache/vllm $VLLM_BASE_DIR/

WORKDIR $VLLM_BASE_DIR/vllm

ARG VLLM_PRS=""

RUN if [ -n "$VLLM_PRS" ]; then \
        # Git requires a user identity to create merge commits
        git config --global user.email "builder@example.com"; \
        git config --global user.name "Docker Builder"; \
        \
        echo "Applying PRs: $VLLM_PRS"; \
        for pr in $VLLM_PRS; do \
            echo "Fetching and merging PR #$pr..."; \
            git fetch origin pull/${pr}/head:pr-${pr}; \
            git merge pr-${pr} --no-edit; \
        done; \
    fi

# TEMPORARY PATCH for broken FP8 kernels - https://github.com/vllm-project/vllm/pull/35568
RUN curl -fsL https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/35568.diff -o pr35568.diff \
    && if git apply --reverse --check pr35568.diff 2>/dev/null; then \
         echo "PR 35568 already applied, skipping."; \
       else \
         echo "Applying PR 35568..."; \
         git apply -v --exclude="tests/*" pr35568.diff; \
       fi \
    && rm pr35568.diff

# TEMPORARY PATCH: revert vLLM PR #41524 / commit c51df430,
# which disables FlashInfer autotune and regresses DGX Spark throughput.
RUN set -eux; \
    patch_commit="c51df43005726a09c6eb7348e8c1b00501c70a8e"; \
    target="vllm/config/vllm.py"; \
    marker="https://github.com/flashinfer-ai/flashinfer/issues/3197"; \
    if grep -q "$marker" "$target"; then \
        echo "PR #41524 regression found; reverting ${patch_commit}"; \
        if ! git revert --no-commit "$patch_commit"; then \
            git revert --abort 2>/dev/null || true; \
            echo "ERROR: PR #41524 appears present but could not be reverted"; \
            exit 1; \
        fi; \
        if grep -q "$marker" "$target"; then \
            echo "ERROR: revert completed but PR #41524 marker is still present"; \
            exit 1; \
        fi; \
    else \
        echo "PR #41524 regression marker not present; skipping revert"; \
    fi

# Prepare build requirements
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test/cuda.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test/cuda.txt && \
    uv pip install -r requirements/build/cuda.txt

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
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    # dump git ref in the wheels dir
    git rev-parse HEAD > /workspace/wheels/.vllm-commit

# =========================================================
# STAGE 5: vLLM Wheel Export
# =========================================================
FROM scratch AS vllm-export
COPY --from=vllm-builder /workspace/wheels /

# =========================================================
# STAGE 6: Runner (Installs wheels from host ./wheels/)
# =========================================================
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04 AS runner

# Transferring build settings from build image because of ptxas/jit compilation during vLLM startup
# Build parallemism
ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"
ENV DG_JIT_USE_NVRTC=1
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
    libxcb1 \
    && cd /workspace/nccl-pkg && apt install -y --no-install-recommends --allow-downgrades ./*.deb \
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
     uv pip install torch==2.11.0 torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu130 && \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# Install wheels from host ./wheels/ (bind-mounted from build context — no layer bloat)
# With --tf5: override vLLM's transformers<5 constraint to get transformers>=5
RUN --mount=type=bind,source=wheels,target=/workspace/wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    if [ "$PRE_TRANSFORMERS" = "1" ]; then \
        echo "transformers>=5.0.0" > /tmp/tf-override.txt && \
        uv pip install /workspace/wheels/*.whl --override /tmp/tf-override.txt; \
    else \
        uv pip install /workspace/wheels/*.whl; \
    fi

# Setup environment for runtime
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH


# Final extra deps
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install ray[default] fastsafetensors instanttensor

# Fix NCCL
RUN rm /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnccl.so.2 /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
    
# Build metadata (generated by build-and-copy.sh)
COPY build-metadata.yaml /workspace/build-metadata.yaml
