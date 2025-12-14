# =========================================================
# STAGE 1: Builder (Heavy image with compiler toolchain)
# =========================================================
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS builder

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
    python3-dev python3-pip git wget \
    libnccl-dev libnccl2 libibverbs1 libibverbs-dev rdma-core \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Configure Ccache for CUDA/C++
ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
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

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# You can add termplotlib to the list below if you want to visualize text graphs in vllm bench serve
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install xgrammar triton fastsafetensors 

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flashinfer-python --no-deps --index-url https://flashinfer.ai/whl && \
    pip install flashinfer-cubin --index-url https://flashinfer.ai/whl && \
    pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130 && \
    pip install apache-tvm-ffi nvidia-cudnn-frontend nvidia-cutlass-dsl nvidia-ml-py tabulate

# --- VLLM SOURCE CACHE BUSTER ---
# Change THIS argument to force a fresh git clone and rebuild of vLLM
# without re-installing the dependencies above.
ARG CACHEBUST_VLLM=1

# 4. Smart Git Clone (Fetch changes instead of full re-clone)
# We mount a cache at /repo-cache. This directory persists on your host machine.
RUN --mount=type=cache,target=/repo-cache \
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
        git reset --hard origin/main && \
        git submodule update --init --recursive; \
    fi && \
    # 3. Copy the updated code from the cache to the actual container workspace
    # We use 'cp -a' to preserve permissions
    cp -a /repo-cache/vllm $VLLM_BASE_DIR/

WORKDIR $VLLM_BASE_DIR/vllm

# Prepare build requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    pip install -r requirements/build.txt

# Apply Patches
# TEMPORARY PATCH for fastsafetensors loading in cluster setup - tracking https://github.com/foundation-model-stack/fastsafetensors/issues/36
COPY fastsafetensors.patch .
RUN patch -p1 < fastsafetensors.patch

# Final Compilation
# We mount the ccache directory here. Ideally, map this to a host volume for persistence 
# across totally separate `docker build` invocations.
RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation . -v


# =========================================================
# STAGE 2: Runner (Lightweight Runtime Image)
# =========================================================
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV VLLM_BASE_DIR=/workspace/vllm

# Install minimal runtime dependencies (NCCL, Python)
# Note: "devel" tools like cmake/gcc are NOT installed here to save space
RUN apt update && apt upgrade -y \
    && apt install -y --allow-change-held-packages --no-install-recommends \
    python3 python3-pip python3-dev vim curl git wget \
    libnccl-dev libnccl2 libibverbs1 rdma-core \
    && rm -rf /var/lib/apt/lists/*

# Set final working directory
WORKDIR $VLLM_BASE_DIR

# Copy artifacts from Builder Stage
# We copy the python packages and executables
# No need to copy source code, as it's already in the site-packages
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Download Tiktoken files
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# Setup Env for Runtime
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH

# Copy scripts
COPY run-cluster-node.sh $VLLM_BASE_DIR/
RUN chmod +x $VLLM_BASE_DIR/run-cluster-node.sh

# Final extra deps
RUN pip install ray[default]
