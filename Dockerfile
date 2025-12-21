# syntax=docker/dockerfile:1.6

# Limit build parallelism to reduce OOM situations
ARG BUILD_JOBS=16

# =========================================================
# STAGE 1: Base Image (Installs Dependencies)
# =========================================================
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS base

# Build parallemism
ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

# Set non-interactive frontend to prevent apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Allow pip to install globally on Ubuntu 24.04 without a venv
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set the base directory environment variable
ENV VLLM_BASE_DIR=/workspace/vllm

# 1. Install Build Dependencies & Ccache
# Added ccache to enable incremental compilation caching
RUN apt update && apt upgrade -y \
    && apt install -y --allow-change-held-packages --no-install-recommends \
    curl vim cmake build-essential ninja-build \
    libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
    python3-dev python3-pip git wget \
    libnccl-dev libnccl2 libibverbs1 libibverbs-dev rdma-core \
    ccache \
    && rm -rf /var/lib/apt/lists/*

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

# Setup Workspace
WORKDIR $VLLM_BASE_DIR

# 2. Set Environment Variables
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# --- CACHE BUSTER ---
# Change this argument to force a re-download of PyTorch/FlashInfer
ARG CACHEBUST_DEPS=1

# 3. Install Python Dependencies with Cache Mounts
# Using --mount=type=cache ensures that even if this layer invalidates, 
# pip reuses previously downloaded wheels.

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install additional dependencies
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install xgrammar fastsafetensors

ARG FLASHINFER_PRE=""

# Install FlashInfer packages
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install ${FLASHINFER_PRE} flashinfer-python --no-deps --index-url https://flashinfer.ai/whl && \
    pip install ${FLASHINFER_PRE} flashinfer-cubin --index-url https://flashinfer.ai/whl && \
    pip install ${FLASHINFER_PRE} flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130 && \
    pip install apache-tvm-ffi nvidia-cudnn-frontend nvidia-cutlass-dsl nvidia-ml-py tabulate

# =========================================================
# STAGE 2: Triton Builder (Compiles Triton independently)
# =========================================================
FROM base AS triton-builder

WORKDIR $VLLM_BASE_DIR

# Initial Triton repo clone (cached forever)
RUN git clone https://github.com/triton-lang/triton.git

# We expect TRITON_REF to be passed from the command line to break the cache
# Set to v3.5.1 tag by default
ARG TRITON_REF=v3.5.1

WORKDIR $VLLM_BASE_DIR/triton

# This only runs if TRITON_REF differs from the last build
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    git fetch origin && \
    git checkout ${TRITON_REF} && \
    git submodule sync && \
    git submodule update --init --recursive && \
    pip install -r python/requirements.txt && \
    mkdir -p /workspace/wheels && \
    pip wheel --no-build-isolation . --wheel-dir=/workspace/wheels -v && \
    pip wheel --no-build-isolation  python/triton_kernels --no-deps --wheel-dir=/workspace/wheels

# =========================================================
# STAGE 3: vLLM Builder (Builds vLLM from Source)
# =========================================================
FROM base AS builder

# --- VLLM SOURCE CACHE BUSTER ---
# Change THIS argument to force a fresh git clone and rebuild of vLLM
# without re-installing the dependencies above.
ARG CACHEBUST_VLLM=1

# Git reference (branch, tag, or SHA) to checkout
ARG VLLM_REF=main

# 4. Smart Git Clone (Fetch changes instead of full re-clone)
# We mount a cache at /repo-cache. This directory persists on your host machine.
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    # 1. Go into the persistent cache directory
    cd /repo-cache && \
    # 2. Logic: Clone if missing, otherwise Fetch & Reset
    if [ ! -d "vllm" ]; then \
        echo "Cache miss: Cloning vLLM from scratch..." && \
        git clone --recursive https://github.com/vllm-project/vllm.git; \
    else \
        echo "Cache hit: Fetching updates..." && \
        cd vllm && \
        git fetch --all && \
        git checkout ${VLLM_REF} && \
        if [ "${VLLM_REF}" = "main" ]; then \
            git reset --hard origin/main; \
        fi && \
        git submodule update --init --recursive && \
        # Optimize git repo size
        git gc --auto; \
    fi && \
    # 3. Copy the updated code from the cache to the actual container workspace
    # We use 'cp -a' to preserve permissions
    cp -a /repo-cache/vllm $VLLM_BASE_DIR/

WORKDIR $VLLM_BASE_DIR/vllm

# Prepare build requirements
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test.txt && \
    pip install -r requirements/build.txt

# Apply Patches
# TEMPORARY PATCH for fastsafetensors loading in cluster setup - tracking https://github.com/foundation-model-stack/fastsafetensors/issues/36
COPY fastsafetensors.patch .
RUN patch -p1 < fastsafetensors.patch

# Final Compilation
# We mount the ccache directory here. Ideally, map this to a host volume for persistence 
# across totally separate `docker build` invocations.
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --no-build-isolation . -v

# Install custom Triton from triton-builder
COPY --from=triton-builder /workspace/wheels /workspace/wheels
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install /workspace/wheels/*.whl

# =========================================================
# STAGE 4: Runner (Transfers only necessary artifacts)
# =========================================================
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV VLLM_BASE_DIR=/workspace/vllm

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip

# Install minimal runtime dependencies (NCCL, Python)
# Note: "devel" tools like cmake/gcc are NOT installed here to save space
RUN apt update && apt upgrade -y \
    && apt install -y --allow-change-held-packages --no-install-recommends \
    python3 python3-pip python3-dev vim curl git wget \
    libcudnn9-cuda-13 \
    libnccl-dev libnccl2 libibverbs1 libibverbs-dev rdma-core \
    # vllm bench plotting
    gnuplot-nox \
    && rm -rf /var/lib/apt/lists/*

# Set final working directory
WORKDIR $VLLM_BASE_DIR

# Download and cache Tiktoken encodings
RUN --mount=type=cache,id=tiktoken-encodings,target=/root/.cache/tiktoken_encodings \
    set -eux; \
    mkdir -p /root/.cache/tiktoken_encodings; \
    for f in o200k_base.tiktoken cl100k_base.tiktoken; do \
      if [ ! -s "/root/.cache/tiktoken_encodings/$f" ]; then \
        echo "Downloading $f..."; \
        wget -q -O "/root/.cache/tiktoken_encodings/$f" \
          "https://openaipublic.blob.core.windows.net/encodings/$f"; \
      else \
        echo "Cache hit: $f"; \
      fi; \
    done; \
    mkdir -p "$VLLM_BASE_DIR/tiktoken_encodings"; \
    cp -a /root/.cache/tiktoken_encodings/. "$VLLM_BASE_DIR/tiktoken_encodings/"

# Copy artifacts from Builder Stage
# We copy the python packages and executables
# No need to copy source code, as it's already in the site-packages
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Setup Env for Runtime
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH

# Copy scripts
COPY run-cluster-node.sh $VLLM_BASE_DIR/
RUN chmod +x $VLLM_BASE_DIR/run-cluster-node.sh

# Final extra deps
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install ray[default] \
    # vllm bench plotting
    termplotlib

