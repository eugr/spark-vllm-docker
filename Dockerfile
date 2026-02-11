# syntax=docker/dockerfile:1.6

# Limit build parallelism to reduce OOM situations
ARG BUILD_JOBS=16

# =========================================================
# STAGE 1: Base Image (Installs Dependencies)
# =========================================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS base

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

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# Set the base directory environment variable
ENV VLLM_BASE_DIR=/workspace/vllm

# 1. Install Build Dependencies & Ccache
# Added ccache to enable incremental compilation caching
RUN apt update && \
    apt install -y --no-install-recommends \
    curl vim ninja-build git \
    ccache \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv && pip uninstall -y flash-attn

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
ARG TORCH_CUDA_ARCH_LIST=12.1a
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# =========================================================
# STAGE 2: Builder (Builds Triton, Flashinfer and vLLM from Source)
# =========================================================
FROM base AS builder


# # ======= Triton Build ========== 

# # Initial Triton repo clone (cached forever)
# RUN git clone https://github.com/triton-lang/triton.git

# # We expect TRITON_REF to be passed from the command line to break the cache
# # Set to v3.6.0 by default
# ARG TRITON_REF=v3.6.0

# WORKDIR $VLLM_BASE_DIR/triton

# # This only runs if TRITON_REF differs from the last build
# RUN --mount=type=cache,id=ccache,target=/root/.ccache \
#     --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
#     git fetch origin && \
#     git checkout ${TRITON_REF} && \
#     git submodule sync && \
#     git submodule update --init --recursive && \
#     uv pip install -r python/requirements.txt && \
#     mkdir -p /workspace/wheels && \
#     rm -rf .git && \
#     uv build --no-build-isolation --wheel --out-dir=/workspace/wheels -v .  && \
#     uv build --no-build-isolation --wheel --no-index --out-dir=/workspace/wheels python/triton_kernels 

# ======= FlashInfer Build ==========

ARG FLASHINFER_CUDA_ARCH_LIST='12.1a'
ENV FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST
WORKDIR $VLLM_BASE_DIR
ARG FLASHINFER_REF=main

# --- CACHE BUSTER ---
# Change this argument to force a re-download of FlashInfer
ARG CACHEBUST_DEPS=1

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# 4. Smart Git Clone (Fetch changes instead of full re-clone)
# We mount a cache at /repo-cache. This directory persists on your host machine.
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    # 1. Go into the persistent cache directory
    cd /repo-cache && \
    # 2. Logic: Clone if missing, otherwise Fetch & Reset
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
        git fetch --all && \
        git checkout ${FLASHINFER_REF} && \
        if [ "${FLASHINFER_REF}" = "main" ]; then \
            git reset --hard origin/main; \
        fi && \
        git submodule update --init --recursive && \
        # Optimize git repo size
        git gc --auto; \
    fi && \
    # 3. Copy the updated code from the cache to the actual container workspace
    # We use 'cp -a' to preserve permissions
    cp -a /repo-cache/flashinfer /workspace/flashinfer

# Build FlashInfer wheels

WORKDIR /workspace/flashinfer

# flashinfer-python
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

# flashinfer-cubin
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    cd flashinfer-cubin && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

# flashinfer-jit-cache
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    cd flashinfer-jit-cache && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

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
        if [ "$VLLM_REF" != "main" ]; then \
            cd vllm && \
            git checkout ${VLLM_REF}; \
        fi; \
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

ARG VLLM_PRS=""

RUN if [ -n "$VLLM_PRS" ]; then \
        echo "Applying PRs: $VLLM_PRS"; \
        for pr in $VLLM_PRS; do \
            echo "Fetching and applying PR #$pr..."; \
            curl -fL "https://github.com/vllm-project/vllm/pull/${pr}.diff" | git apply -v; \
        done; \
    fi

ARG PRE_TRANSFORMERS=0

# Prepare build requirements
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test.txt && \
    if [ "$PRE_TRANSFORMERS" = "1" ]; then \
        sed -i '/^transformers\b/d' requirements/common.txt; \
        sed -i '/^transformers\b/d' requirements/test.txt; \
    fi && \
    uv pip install -r requirements/build.txt

# Apply Patches
# TEMPORARY PATCH for fastsafetensors loading in cluster setup - tracking https://github.com/vllm-project/vllm/issues/34180
# COPY fastsafetensors.patch .
# RUN if patch -p1 --dry-run --reverse < fastsafetensors.patch &>/dev/null; then \
#         echo "PR #34180 is already applied"; \
#     else \
#         patch -p1 < fastsafetensors.patch; \
#     fi

# Final Compilation
# We mount the ccache directory here. Ideally, map this to a host volume for persistence 
# across totally separate `docker build` invocations.
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

# # Install custom Triton from triton-builder
# COPY --from=triton-builder /workspace/wheels /workspace/wheels
# RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
#     uv pip install /workspace/wheels/*.whl

# =========================================================
# STAGE 4: Runner (Transfers only necessary artifacts)
# =========================================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS runner

# Transferring build settings from build image because of ptxas/jit compilation during vLLM startup
# Build parallemism
ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV VLLM_BASE_DIR=/workspace/vllm

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# Install runtime dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
    curl vim git \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv && pip uninstall -y flash-attn # triton-kernels pytorch-triton

# Set final working directory
WORKDIR $VLLM_BASE_DIR

# Download Tiktoken files
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# Copy artifacts from Builder Stage
RUN --mount=type=bind,from=builder,source=/workspace/wheels,target=/mount/wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install /mount/wheels/*.whl

ARG PRE_TRANSFORMERS=0
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    if [ "$PRE_TRANSFORMERS" = "1" ]; then \
        uv pip install -U transformers --pre; \
    fi

# Setup Env for Runtime
ARG TORCH_CUDA_ARCH_LIST=12.1a
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
ARG FLASHINFER_CUDA_ARCH_LIST='12.1a'
ENV FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH

# Copy scripts
COPY run-cluster-node.sh $VLLM_BASE_DIR/
RUN chmod +x $VLLM_BASE_DIR/run-cluster-node.sh

# Final extra deps
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install ray[default] fastsafetensors

# Cleanup

# Keeping it here for reference - this won't work as is without squashing layers
# RUN uv pip uninstall absl-py apex argon2-cffi \
#     argon2-cffi-bindings arrow asttokens astunparse async-lru audioread babel beautifulsoup4 \
#     black bleach comm contourpy cycler datasets debugpy decorator defusedxml dllist dm-tree \
#     execnet executing expecttest fastjsonschema fonttools fqdn gast hypothesis \
#     ipykernel ipython ipython_pygments_lexers isoduration isort jedi joblib jupyter-events \
#     jupyter-lsp jupyter_client jupyter_core jupyter_server jupyter_server_terminals jupyterlab \
#     jupyterlab_code_formatter jupyterlab_code_formatter jupyterlab_pygments jupyterlab_server \
#     jupyterlab_tensorboard_pro jupytext kiwisolver matplotlib matplotlib-inline matplotlib-inline \
#     mistune ml_dtypes mock nbclient nbconvert nbformat nest-asyncio notebook notebook_shim \
#     opt_einsum optree outlines_core overrides pandas pandocfilters parso pexpect polygraphy pooch \
#     pyarrow pycocotools pytest-flakefinder pytest-rerunfailures pytest-shard pytest-xdist \
#     scikit-learn scipy Send2Trash soundfile soupsieve soxr spin stack-data \
#     wcwidth webcolors xdoctest Werkzeug
