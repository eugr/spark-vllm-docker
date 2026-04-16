# syntax=docker/dockerfile:1.6
#
# vLLM Docker image — optimized for NVIDIA Jetson AGX Thor (T5000)
#
#   - GPU arch:  sm_110 / sm_110a  (Blackwell-Jetson)
#   - SoC:       Jetson Thor, 14-core Arm Neoverse-V3AE
#   - CPU ABI:   aarch64 / arm64-sbsa  (JetPack 7.x uses SBSA, not legacy tegra-aarch64)
#   - JetPack:   7.x  (Jetson Linux / L4T r38.x, Ubuntu 24.04)
#   - CUDA:      13.0  (cuDNN 9.12, TensorRT 10.13)
#   - Memory:    128 GB LPDDR5X unified (coherent with GPU)
#
# This image is purpose-built for Thor and is NOT compatible with DGX Spark
# (sm_121) or data-centre GPUs (sm_100 / sm_90).  Default builds target the
# T5000; T4000 (64 GB) uses the same sm_110 so the image works there too.
#
# Build parallelism
ARG BUILD_JOBS=8

# =========================================================
# STAGE 1: Base Build Image (arm64-sbsa, CUDA 13.0)
# =========================================================
# Manifest resolves to the linux/arm64 (arm64-sbsa) variant automatically
# when building on a Thor host.  NOTE: do NOT use the legacy
# nvcr.io/nvidia/l4t-cuda / l4t-jetpack line — those are Orin-era and
# ship the old tegra-aarch64 ABI which Thor no longer uses.
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS base

# Build parallelism — Thor's 14 V3AE cores can compile, but memory-heavy
# CUDA TUs easily OOM the 128 GB LPDDR5X (which is shared with the GPU).
# Default to 8 — bump via `--build-jobs` only if you have headroom.
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

# Set pip / uv cache directories
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=600
ENV UV_HTTP_RETRIES=10

# Workspace
ENV VLLM_BASE_DIR=/workspace/vllm

# Thor PyTorch wheel index (arm64-sbsa + cu130).  The NVIDIA-hosted
# Jetson AI Lab index carries Thor-validated torch 2.9 builds; upstream
# `whl/cu130` may still fall back to SBSA generic wheels that lack
# sm_110 kernels.  We let `--torch-index-url` override at build time.
ARG TORCH_INDEX_URL="https://pypi.jetson-ai-lab.io/sbsa/cu130"
ARG TORCH_VERSION="2.9.0"

# 1. Install Build Dependencies & Ccache
RUN apt update && \
    apt install -y --no-install-recommends \
        curl vim cmake build-essential ninja-build \
        libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
        libnccl2 libnccl-dev \
        python3-dev python3-pip git wget patch \
        libibverbs1 libibverbs-dev rdma-core \
        ccache devscripts debhelper fakeroot \
        pkg-config libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# PyTorch + Triton for Thor (sm_110, cu130)
# Fallback order: jetson-ai-lab SBSA index → upstream cu130 → source hint.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    (uv pip install "torch==${TORCH_VERSION}" torchvision torchaudio triton \
         --index-url "${TORCH_INDEX_URL}" \
     || uv pip install "torch==${TORCH_VERSION}" torchvision torchaudio triton \
         --index-url "https://download.pytorch.org/whl/cu130") && \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2" \
        filelock pynvml requests tqdm packaging

# Configure Ccache for CUDA/C++ — shared across FlashInfer + vLLM stages.
ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=50G
ENV CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# 2. GPU architecture — Thor is sm_110a (Blackwell-Jetson).
# `11.0a` keeps architecture-specific features (tcgen05 / tmem) enabled.
ARG TORCH_CUDA_ARCH_LIST="11.0a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Thor only has one on-die GPU per board, so custom ring NCCL builds are
# unnecessary.  Stock libnccl2 (installed above) handles multi-Thor runs
# over 25 GbE when you wire several boards together.

WORKDIR $VLLM_BASE_DIR

# =========================================================
# STAGE 2: FlashInfer Builder
# =========================================================
FROM base AS flashinfer-builder

ARG FLASHINFER_CUDA_ARCH_LIST="11.0a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR
ARG FLASHINFER_REF=main

# --- CACHE BUSTER ---
ARG CACHEBUST_FLASHINFER=1

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install packaging

# Smart Git Clone (Fetch changes instead of full re-clone)
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "flashinfer" ]; then \
        echo "Cache miss: Cloning FlashInfer from scratch..." && \
        git clone --recursive https://github.com/flashinfer-ai/flashinfer.git; \
        if [ "$FLASHINFER_REF" != "main" ]; then \
            cd flashinfer && git checkout ${FLASHINFER_REF}; \
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
        echo "Applying PRs: $FLASHINFER_PRS"; \
        for pr in $FLASHINFER_PRS; do \
            echo "Fetching and applying PR #$pr..."; \
            curl -fL "https://github.com/flashinfer-ai/flashinfer/pull/${pr}.diff" | git apply -v; \
        done; \
    fi

# trtllm-gen FMHA cubins upstream are not compiled for sm_110 yet
# (tracking: https://github.com/NVIDIA/TensorRT-LLM/issues/11799 and
# https://github.com/flashinfer-ai/flashinfer/pull/2913).  Pass
# `--apply-flashinfer-pr 2913` at build time when the upstream patch
# is not yet merged, or rely on the PTX-JIT fallback path for now.

# Apply patch to avoid re-downloading existing cubins
COPY flashinfer_cache.patch .
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=cubins-cache,target=/workspace/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins \
    patch -p1 < flashinfer_cache.patch && \
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    cd flashinfer-cubin && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    cd ../flashinfer-jit-cache && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
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

ARG TORCH_CUDA_ARCH_LIST="11.0a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR

ARG CACHEBUST_VLLM=1
ARG VLLM_REF=main

# Smart Git Clone (Fetch changes instead of full re-clone)
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "vllm" ]; then \
        echo "Cache miss: Cloning vLLM from scratch..." && \
        git clone --recursive https://github.com/vllm-project/vllm.git; \
        if [ "$VLLM_REF" != "main" ]; then \
            cd vllm && git checkout ${VLLM_REF}; \
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
        echo "Applying PRs: $VLLM_PRS"; \
        for pr in $VLLM_PRS; do \
            echo "Fetching and applying PR #$pr..."; \
            curl -fL "https://github.com/vllm-project/vllm/pull/${pr}.diff" | git apply -v; \
        done; \
    fi

# Thor-specific notes on upstream vLLM issues:
#   * #26791 "sm_110 not compatible"  — fixed once torch lists sm_110 in the
#     CUDAExtension arch list.  We rebuild from source so the arch we
#     pass via TORCH_CUDA_ARCH_LIST is respected.
#   * #38411 `_vllm_fa2_C` built SM80-only — harmless when FlashInfer is
#     the attention backend; to force it set `VLLM_ATTENTION_BACKEND=FLASHINFER`.
#   * #27364 Qwen3-VL FP8 emits `!!!!` — use NVFP4 quant on Thor instead.
#   * #32093 Nemotron Nano V3 FP16 mis-generates — see mods/nemotron-nano.

# Prepare build requirements
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test/cuda.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test/cuda.txt && \
    uv pip install -r requirements/build/cuda.txt

# Final Compilation
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    git rev-parse HEAD > /workspace/wheels/.vllm-commit

# =========================================================
# STAGE 5: vLLM Wheel Export
# =========================================================
FROM scratch AS vllm-export
COPY --from=vllm-builder /workspace/wheels /

# =========================================================
# STAGE 6: Runner (Installs wheels from host ./wheels/)
# =========================================================
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS runner

# Transferring build settings because of ptxas JIT at vLLM startup
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

ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# Runtime dependencies.  We use stock libnccl2 — no custom NCCL fork,
# no InfiniBand/RoCE plumbing (Thor does not ship a ConnectX NIC).
RUN apt update && \
    apt install -y --no-install-recommends \
        python3 python3-pip python3-dev vim curl git wget \
        libcudnn9-cuda-13 \
        libnccl2 \
        libibverbs1 libibverbs-dev rdma-core \
        libxcb1 iproute2 iputils-ping \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

WORKDIR $VLLM_BASE_DIR

# Download Tiktoken files (needed by gpt-oss and a few other models)
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

ARG PRE_TRANSFORMERS=0
ARG TORCH_INDEX_URL="https://pypi.jetson-ai-lab.io/sbsa/cu130"
ARG TORCH_VERSION="2.9.0"

# Install torch + triton matching the builder image.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    (uv pip install "torch==${TORCH_VERSION}" torchvision torchaudio triton \
         --index-url "${TORCH_INDEX_URL}" \
     || uv pip install "torch==${TORCH_VERSION}" torchvision torchaudio triton \
         --index-url "https://download.pytorch.org/whl/cu130") && \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# Install wheels from host ./wheels/ (bind-mount — no layer bloat)
RUN --mount=type=bind,source=wheels,target=/workspace/wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    if [ "$PRE_TRANSFORMERS" = "1" ]; then \
        echo "transformers>=5.0.0" > /tmp/tf-override.txt && \
        uv pip install /workspace/wheels/*.whl --override /tmp/tf-override.txt; \
    else \
        uv pip install /workspace/wheels/*.whl; \
    fi

# Runtime environment
ARG TORCH_CUDA_ARCH_LIST="11.0a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG FLASHINFER_CUDA_ARCH_LIST="11.0a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH

# Final extra deps
#   - ray[default]       : single-node scheduler (works without Ray too)
#   - fastsafetensors    : faster weight loading on Thor's LPDDR5X
#   - instanttensor      : experimental, even faster than fastsafetensors
#   - lmcache            : optional KV-cache reuse layer; wire via
#                          `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1",
#                          "kv_role":"kv_both"}'`.  Builds from source against the
#                          torch version installed above.
ARG INSTALL_LMCACHE=1
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install ray[default] fastsafetensors instanttensor && \
    if [ "$INSTALL_LMCACHE" = "1" ]; then \
        uv pip install lmcache || echo "WARN: lmcache install failed — skipping (source build may be required on Thor)"; \
    fi

# NCCL: keep the shipped python-nvidia-nccl shared lib pointing at the
# apt-installed libnccl2 on aarch64 so we use the stock build rather
# than the datacentre wheel.  The original DGX Spark image replaced a
# custom-compiled NCCL here; on Thor we use the Ubuntu package.
RUN NCCL_SYS=/usr/lib/aarch64-linux-gnu/libnccl.so.2 && \
    if [ -f "$NCCL_SYS" ]; then \
        for f in $(find /usr/local/lib/python3*/dist-packages/nvidia/nccl/lib -name 'libnccl.so.2' 2>/dev/null); do \
            rm -f "$f" && ln -s "$NCCL_SYS" "$f"; \
        done; \
    fi

# Build metadata (generated by build-and-copy.sh)
COPY build-metadata.yaml /workspace/build-metadata.yaml
