# syntax=docker/dockerfile:1.6

# Limit build parallelism to reduce OOM situations
ARG BUILD_JOBS=16
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04

# =========================================================
# STAGE 1: Base Build Image
# =========================================================
FROM ${CUDA_IMAGE} AS base

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
# RUN git clone -b dgxspark-3node-ring https://github.com/zyang-dev/nccl.git && \
#     cd nccl && make -j ${BUILD_JOBS} src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" && \
#     make pkg.debian.build && apt install -y --no-install-recommends --allow-downgrades ./build/pkg/deb/*.deb

RUN git clone -b v2.30u1 https://github.com/NVIDIA/nccl.git && \
    cd nccl && make -j ${BUILD_JOBS} src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" && \
    make pkg.debian.build && apt install -y --no-install-recommends --allow-downgrades --allow-change-held-packages ./build/pkg/deb/*.deb

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
        git submodule update --init --recursive && \
        git clean -fdx && \
        git gc --auto; \
    fi && \
    cp -a /repo-cache/flashinfer /workspace/flashinfer

WORKDIR /workspace/flashinfer

ARG FLASHINFER_PRS=""

RUN set -eux; \
    if [ -n "$FLASHINFER_PRS" ]; then \
        # Git requires a user identity to create merge commits
        git config --global user.email "builder@example.com"; \
        git config --global user.name "Docker Builder"; \
        \
        echo "Applying PRs: $FLASHINFER_PRS"; \
        for pr in $FLASHINFER_PRS; do \
            echo "Fetching and merging PR #$pr..."; \
            git fetch origin +pull/${pr}/head:pr-${pr}; \
            if git merge-base --is-ancestor pr-${pr} HEAD; then \
                echo "PR #$pr is already contained in HEAD; skipping."; \
            else \
                cherry_file="/tmp/pr-${pr}.cherry"; \
                git cherry HEAD pr-${pr} > "$cherry_file"; \
                if ! grep -q '^+' "$cherry_file"; then \
                    echo "PR #$pr is already patch-equivalent to HEAD; skipping."; \
                    rm -f "$cherry_file"; \
                    continue; \
                fi; \
                rm -f "$cherry_file"; \
                git merge pr-${pr} --no-edit; \
            fi; \
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

# DeepGEMM PR #324 is required by vLLM PR #43477 for SM120/SM121 MXFP4 paths.
ARG DEEPGEMM_REPO=https://github.com/deepseek-ai/DeepGEMM.git
ARG DEEPGEMM_PR=324
ARG DEEPGEMM_REF=9ca30487a6d1a484757f2d87f532c5f6707b9f25
ENV DEEPGEMM_SRC_DIR=/workspace/DeepGEMM

# Smart Git Clone (Fetch changes instead of full re-clone)
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    echo "CACHEBUST_VLLM=${CACHEBUST_VLLM}" && \
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
    if [ -n "$DEEPGEMM_PR" ]; then \
        git fetch origin +pull/${DEEPGEMM_PR}/head:pr-${DEEPGEMM_PR}; \
    fi; \
    git checkout --detach "$DEEPGEMM_REF"; \
    git reset --hard; \
    git submodule update --init --recursive; \
    git clean -fdx; \
    rm -rf "$DEEPGEMM_SRC_DIR"; \
    cp -a /repo-cache/deepgemm "$DEEPGEMM_SRC_DIR"

WORKDIR $VLLM_BASE_DIR/vllm

ARG VLLM_PRESET_PRS=""
ARG VLLM_APPLY_PRESET_PRS=""
ARG VLLM_PRS=""

RUN set -eux; \
    VLLM_ALL_PRS=""; \
    VLLM_SELECTED_PRESET_PRS=""; \
    case "$VLLM_APPLY_PRESET_PRS" in \
        1|true|TRUE|yes|YES) VLLM_SELECTED_PRESET_PRS="$VLLM_PRESET_PRS";; \
        0|false|FALSE|no|NO) VLLM_SELECTED_PRESET_PRS="";; \
        ""|auto|AUTO) \
            if [ -z "$VLLM_PRS" ]; then \
                VLLM_SELECTED_PRESET_PRS="$VLLM_PRESET_PRS"; \
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
        # Git requires a user identity to create merge commits
        git config --global user.email "builder@example.com"; \
        git config --global user.name "Docker Builder"; \
        \
        echo "Applying PRs: $VLLM_ALL_PRS"; \
        for pr in $VLLM_ALL_PRS; do \
            echo "Fetching and merging PR #$pr..."; \
            git fetch origin +pull/${pr}/head:pr-${pr}; \
            if git merge-base --is-ancestor pr-${pr} HEAD; then \
                echo "PR #$pr is already contained in HEAD; skipping."; \
            else \
                cherry_file="/tmp/pr-${pr}.cherry"; \
                git cherry HEAD pr-${pr} > "$cherry_file"; \
                if ! grep -q '^+' "$cherry_file"; then \
                    echo "PR #$pr is already patch-equivalent to HEAD; skipping."; \
                    rm -f "$cherry_file"; \
                    continue; \
                fi; \
                rm -f "$cherry_file"; \
                git merge pr-${pr} --no-edit; \
            fi; \
        done; \
    fi

# TEMPORARY PATCH: vLLM PR #43409 started passing AutoGPTQ MoE qzeros
# through even for symmetric GPTQ. On CUDA Marlin MoE this can select the
# wrong zero-point kernel path and crash Qwen3-Coder-Next AutoRound during
# startup. Apply only when the vulnerable upstream pattern is present.
RUN python3 - <<PY
from pathlib import Path

target = Path("vllm/model_executor/layers/quantization/auto_gptq.py")
bad = '''            w1_zp=getattr(layer, "w13_qzeros", None),
            w2_zp=getattr(layer, "w2_qzeros", None),'''
fixed = '''            w1_zp=getattr(layer, "w13_qzeros", None)
            if not self.quant_config.is_sym
            else None,
            w2_zp=getattr(layer, "w2_qzeros", None)
            if not self.quant_config.is_sym
            else None,'''

if not target.exists():
    print(f"{target} not found; skipping AutoGPTQ MoE qzeros workaround")
else:
    text = target.read_text()
    if fixed in text:
        print("AutoGPTQ MoE qzeros workaround already present; skipping")
    elif bad in text:
        target.write_text(text.replace(bad, fixed, 1))
        print("Applied AutoGPTQ symmetric MoE qzeros workaround")
    else:
        print("Known vulnerable AutoGPTQ MoE qzeros pattern not found; skipping")
PY

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
RUN set -eux; \
    target="vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py"; \
    marker='_MINIMAX_FUSED_AR_RMS_QK = getattr(torch.ops._C, "minimax_allreduce_rms_qk", None)'; \
    replacement='_MINIMAX_FUSED_AR_RMS_QK = None  # Disabled for DGX Spark multi-node TP'; \
    if [ -f "$target" ] && grep -Fq "$marker" "$target"; then \
        echo "MiniMax QK norm fusion found; disabling CUDA IPC fused path"; \
        sed -i "s|$marker|$replacement|" "$target"; \
    elif [ -f "$target" ] && grep -Fq "$replacement" "$target"; then \
        echo "MiniMax QK norm fusion already disabled"; \
    else \
        echo "MiniMax QK norm fusion marker not present; skipping patch"; \
    fi; \
    if [ -f "$target" ] && grep -Fq "$marker" "$target"; then \
        echo "ERROR: MiniMax QK norm fusion marker is still present"; \
        exit 1; \
    fi

# TEMPORARY PATCH: vLLM PR #43362 made RoutedExperts scalarize all
# _load_single_value() inputs. That is correct for scalar input scales, but
# compressed-tensors MoE checkpoints also load 2-element weight_shape metadata
# through this path. Preserve vector metadata when the destination slot matches.
RUN python3 - <<'PY'
from pathlib import Path

target = Path("vllm/model_executor/layers/fused_moe/routed_experts.py")
old = '''    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = self._to_scalar(loaded_weight)
'''
new = '''    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data
        target = param_data[expert_id]

        if target.ndim > 0 and target.numel() == loaded_weight.numel():
            target.copy_(loaded_weight.reshape_as(target).to(
                device=target.device, dtype=target.dtype))
            return

        # Scalar input scales can be loaded directly and should be equal.
        param_data[expert_id] = self._to_scalar(loaded_weight)
'''

if not target.exists():
    print(f"{target} not found; skipping RoutedExperts weight_shape workaround")
else:
    text = target.read_text()
    if "target = param_data[expert_id]" in text:
        print("RoutedExperts weight_shape workaround already present; skipping")
    elif old in text:
        target.write_text(text.replace(old, new, 1))
        print("Applied RoutedExperts weight_shape workaround")
    else:
        print("Known vulnerable RoutedExperts _load_single_value pattern not found; skipping")
PY

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
    git rev-parse HEAD > /workspace/wheels/.vllm-commit && \
    git -C "$DEEPGEMM_SRC_DIR" rev-parse HEAD > /workspace/wheels/.deepgemm-commit

# =========================================================
# STAGE 5: vLLM Wheel Export
# =========================================================
FROM scratch AS vllm-export
COPY --from=vllm-builder /workspace/wheels /

# =========================================================
# STAGE 6: Runner (Installs wheels from host ./wheels/)
# =========================================================
FROM ${CUDA_IMAGE} AS runner

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
    libxcb1 \
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
     uv pip install torch==2.11.0 torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu130 && \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# Install wheels from host ./wheels/ (bind-mounted from build context — no layer bloat)
# With --tf5: override vLLM's transformers<5 constraint to get transformers>=5
# FastAPI 0.137.0 adds _IncludedRouter entries that currently break
# prometheus-fastapi-instrumentator route name lookup.
RUN --mount=type=bind,source=wheels,target=/workspace/wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    PINNED_TORCH=$(python3 -c "import torch; print(torch.__version__)") && \
    echo "torch==${PINNED_TORCH}" > /tmp/wheel-override.txt && \
    echo "fastapi[standard]>=0.115.0,<0.137.0" >> /tmp/wheel-override.txt && \
    if [ "$PRE_TRANSFORMERS" = "1" ]; then \
        echo "transformers>=5.0.0" >> /tmp/wheel-override.txt; \
    fi && \
    uv pip install /workspace/wheels/*.whl --override /tmp/wheel-override.txt

# Setup environment for runtime
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH


# Final extra deps
# Pin torch via --override so transitive deps (e.g. instanttensor) can't trigger
# a re-resolve that swaps the CUDA-built torch for PyPI's CPU wheel.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    PINNED_TORCH=$(python3 -c "import torch; print(torch.__version__)") && \
    echo "torch==${PINNED_TORCH}" > /tmp/torch-override.txt && \
    echo "fastapi[standard]>=0.115.0,<0.137.0" >> /tmp/torch-override.txt && \
    uv pip install ray[default] fastsafetensors instanttensor \
        --override /tmp/torch-override.txt

# Fix NCCL
RUN rm /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnccl.so.2 /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
    
# Build metadata (generated by build-and-copy.sh)
COPY build-metadata.yaml /workspace/build-metadata.yaml
