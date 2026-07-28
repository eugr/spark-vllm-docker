# syntax=docker/dockerfile:1.6

# Limit build parallelism to reduce OOM situations
ARG BUILD_JOBS=16
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG NCCL_NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

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
ARG RUSTUP_TOOLCHAIN=stable
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
ARG VLLM_REF=main

# Pinned while investigating an SM121 DeepSeek-V4 MXFP4 grouped scale-factor
# regression first observed at nv_dev f8e8fb5 (PR #384); last known good.
ARG DEEPGEMM_REPO=https://github.com/deepseek-ai/DeepGEMM.git
ARG DEEPGEMM_REF=a6b593d2826719dcf4892609af7b84ee23aaf32a
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
        git reset --hard HEAD && \
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
    git checkout --detach "$DEEPGEMM_REF" 2>/dev/null || git checkout --detach "origin/$DEEPGEMM_REF"; \
    git reset --hard; \
    git submodule update --init --recursive; \
    git clean -fdx; \
    rm -rf "$DEEPGEMM_SRC_DIR"; \
    cp -a /repo-cache/deepgemm "$DEEPGEMM_SRC_DIR"

WORKDIR $VLLM_BASE_DIR/vllm

# Temporary upstream fixes carried until they are present in the pinned vLLM ref.
# See https://github.com/vllm-project/vllm/pull/47392
ARG VLLM_PRESET_PRS="47392"
ARG VLLM_APPLY_PRESET_PRS=""
ARG VLLM_PRS=""

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
        echo "Fetching origin/main only to calculate PR patch ranges; current checkout remains $VLLM_REF."; \
        git fetch origin +refs/heads/main:refs/remotes/origin/main; \
        for pr in $VLLM_ALL_PRS; do \
            echo "Fetching PR #$pr and applying its patch onto current HEAD..."; \
            git fetch origin +pull/${pr}/head:pr-${pr}; \
            pr_base="$(git merge-base origin/main pr-${pr} || true)"; \
            if [ -z "$pr_base" ]; then \
                echo "Unable to find an origin/main merge-base for PR #$pr."; \
                exit 1; \
            fi; \
            patch_file="/tmp/pr-${pr}.patch"; \
            echo "PR #$pr patch range: $pr_base..pr-${pr}; apply target: $(git rev-parse HEAD)."; \
            git diff --binary "$pr_base" "pr-${pr}" > "$patch_file"; \
            if [ ! -s "$patch_file" ]; then \
                echo "PR #$pr has no patch relative to origin/main; skipping."; \
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

# TEMPORARY PATCH: vLLM PR #49408 / commit d6dbdb9 misplaced the XPU-only
# return in topk_hash_softplus_sqrt, making the CUDA/ROCm kernel call dead code.
# Remove after upstream fix PR #49452 is merged and present in the oldest
# supported VLLM_REF. Inspect source shape rather than commit ancestry so this
# also handles rebases, cherry-picks, and builds that already include the fix.
RUN python3 - <<'PY'
from pathlib import Path

target = Path("vllm/_custom_ops.py")
function_marker = "def topk_hash_softplus_sqrt("
direct_call = "\n    torch.ops._moe_C.topk_softplus_sqrt("
misplaced_return = "\n\n    return" + direct_call
fixed_return = "\n        return\n" + direct_call

if not target.exists():
    raise SystemExit(f"{target} not found; cannot inspect vLLM PR #49408 regression")

text = target.read_text()
start = text.find(function_marker)
if start == -1:
    print("topk_hash_softplus_sqrt is absent; vLLM PR #49408 is not applicable")
else:
    end = text.find("\ndef ", start + len(function_marker))
    end = len(text) if end == -1 else end
    function = text[start:end]

    if "is_padding" not in function:
        print("topk_hash_softplus_sqrt predates vLLM PR #49408; skipping workaround")
    elif fixed_return in function:
        print("vLLM topk_softplus_sqrt non-XPU path already fixed; skipping")
    elif (
        direct_call in function
        and "current_platform.is_xpu()" not in function
        and "\n    return" not in function
    ):
        print("topk_hash_softplus_sqrt has no XPU workaround; patch is not applicable")
    elif function.count(misplaced_return) == 1:
        function = function.replace(misplaced_return, fixed_return, 1)
        updated = text[:start] + function + text[end:]
        compile(updated, str(target), "exec")
        target.write_text(updated)
        print("Applied vLLM PR #49452 topk_softplus_sqrt control-flow fix")
    else:
        raise SystemExit(
            "Unknown is_padding-aware topk_hash_softplus_sqrt layout; "
            "refusing to guess whether vLLM PR #49408 is fixed"
        )
PY

# TEMPORARY PATCH: vLLM PR #47914 added per-KV-group causal metadata by
# treating non-bool causal as Mapping[int, bool]. DiffusionGemma passes a
# per-request torch.Tensor causal mask and crashes on causal.get(...). Keep this
# until upstream build_attn_metadata accepts Tensor causal again.
RUN python3 - <<'PY'
from pathlib import Path

target = Path("vllm/v1/worker/gpu/attn_utils.py")
bad_signature = "causal: bool | Mapping[int, bool] = True,"
fixed_signature = "causal: bool | Mapping[int, bool] | torch.Tensor = True,"
bad_group_causal = (
    "        group_causal = causal if isinstance(causal, bool) else "
    "causal.get(i, True)"
)
fixed_group_causal = """        if isinstance(causal, (bool, torch.Tensor)):
            group_causal = causal
        else:
            group_causal = causal.get(i, True)"""

if not target.exists():
    raise SystemExit(f"{target} not found; cannot apply DiffusionGemma causal patch")

text = target.read_text()
if fixed_signature in text and fixed_group_causal in text:
    print("DiffusionGemma Tensor causal workaround already present; skipping")
elif bad_signature in text and bad_group_causal in text:
    text = text.replace(bad_signature, fixed_signature, 1)
    text = text.replace(bad_group_causal, fixed_group_causal, 1)
    target.write_text(text)
    print("Applied DiffusionGemma Tensor causal workaround for vLLM PR #47914")
else:
    print("Known vLLM PR #47914 causal regression pattern not found; skipping")
PY

# TEMPORARY PATCH: vLLM PR #43957 added a generic embedding-width guard for
# EAGLE3, but Gemma4 MTP intentionally replaces its draft embedding with the
# target backbone embedding before pre_projection. Without sharing, Gemma4 MTP
# concatenates 1024-wide draft embeddings with 2816-wide backbone hidden states
# and crashes in a 5632-wide pre_projection. Keep the guard scoped to EAGLE-style
# draft models until upstream fixes https://github.com/vllm-project/vllm/issues/47794.
RUN python3 - <<'PY'
from pathlib import Path

target = Path("vllm/v1/spec_decode/llm_base_proposer.py")
old = """            if share_embeddings:
                draft_embed = self.model.model.embed_tokens
                # Only share when both models use the same embedding width.
                # Guard with isinstance so non-Tensor weights (e.g. in tests)
"""
new = """            if share_embeddings and hasattr(self.model, "has_own_embed_tokens"):
                draft_embed = self.model.model.embed_tokens
                # Only share when both models use the same embedding width.
                # Guard with isinstance so non-Tensor weights (e.g. in tests)
"""

if not target.exists():
    print(f"{target} not found; skipping Gemma4 MTP embedding-share workaround")
else:
    text = target.read_text()
    if 'if share_embeddings and hasattr(self.model, "has_own_embed_tokens"):' in text:
        print("Gemma4 MTP embedding-share workaround already present; skipping")
    elif old in text:
        target.write_text(text.replace(old, new, 1))
        print("Applied Gemma4 MTP embedding-share workaround")
    else:
        print("Known Gemma4 MTP embedding-share pattern not found; skipping")
PY

# TEMPORARY PATCH (source build only): vLLM PR #43008 selects cooperative_topk
# for all SM90+ devices. On DGX Spark / SM12.x this fails at launch with
# "cooperative_topk launch failed: invalid argument". Keep the cooperative
# path on SM90 and let newer architectures use the existing persistent_topk fallback.
RUN python3 - <<'PY'
from pathlib import Path

target = Path("vllm/model_executor/layers/sparse_attn_indexer.py")
old = '''        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
        )'''
new = '''        device_capability = current_platform.get_device_capability()
        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and device_capability is not None
            and device_capability.to_int() == 90
        )'''

if not target.exists():
    print(f"{target} not found; skipping SM120 cooperative_topk workaround")
else:
    text = target.read_text()
    if "device_capability.to_int() == 90" in text:
        print("SM120 cooperative_topk workaround already present; skipping")
    elif old in text:
        target.write_text(text.replace(old, new, 1))
        print("Applied SM120 cooperative_topk workaround")
    else:
        print("Known cooperative_topk selector pattern not found; skipping")
PY

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

# DGX Spark UMA cleanup: profile warmup can leave temporary CUDA allocator
# reservations behind just before vLLM sizes and allocates KV cache blocks.
RUN python3 - <<'PY'
from pathlib import Path
import re

target = Path("vllm/v1/worker/gpu_worker.py")

if not target.exists():
    raise SystemExit(f"{target} not found; cannot apply KV cache cleanup patch")

text = target.read_text()
lines = text.splitlines(keepends=True)
changed = False

profile_cleanup_present = (
    "profile_result.after_profile.measure()" in text
    and "diff_from_create.non_torch_memory" in text
)
prealloc_cleanup_present = (
    "memory_reserved(self.device)" in text
    and "memory_allocated(self.device)" in text
    and "empty_cache()" in text
)
needs_cleanup = not (profile_cleanup_present and prealloc_cleanup_present)

if needs_cleanup and not re.search(r"(?m)^import gc$", text):
    insert_at = None
    last_future_import = None
    for i, line in enumerate(lines):
        if line.startswith("from __future__ import "):
            last_future_import = i
        elif insert_at is None and (
            line.startswith("import ") or line.startswith("from ")
        ):
            insert_at = i
    if last_future_import is not None:
        lines.insert(last_future_import + 1, "import gc\n")
    elif insert_at is not None:
        lines.insert(insert_at, "import gc\n")
    else:
        lines.insert(0, "import gc\n")
    changed = True


def find_line(pattern: str) -> tuple[int, re.Match[str]]:
    regex = re.compile(pattern)
    for index, line in enumerate(lines):
        match = regex.match(line)
        if match:
            return index, match
    raise SystemExit(f"Could not find expected vLLM pattern: {pattern}")


def insert_after_docstring(func_index: int, func_indent: str, block: list[str]) -> None:
    insert_at = func_index + 1
    if insert_at < len(lines):
        stripped = lines[insert_at].lstrip()
        quote = None
        if stripped.startswith(chr(34) * 3):
            quote = chr(34) * 3
        elif stripped.startswith(chr(39) * 3):
            quote = chr(39) * 3

        if quote is not None:
            if stripped.count(quote) >= 2 and not stripped.startswith(quote * 2):
                insert_at += 1
            else:
                insert_at += 1
                while insert_at < len(lines):
                    if quote in lines[insert_at]:
                        insert_at += 1
                        break
                    insert_at += 1

    lines[insert_at:insert_at] = block


if not profile_cleanup_present:
    snapshot_line = (
        r"^(?P<indent>[ \t]+)free_gpu_memory = "
        r"profile_result\.after_profile\.free_memory\n$"
    )
    index, match = find_line(snapshot_line)
    indent = match.group("indent")
    lines[index:index] = [
        f"{indent}# spark-vllm-docker: post-profile cleanup before KV sizing\n",
        f'{indent}if self.device_config.device_type == "cuda":\n',
        f"{indent}    before_cleanup = profile_result.after_profile.free_memory\n",
        f'{indent}    if hasattr(self.model_runner, "_cleanup_profiling_kv_cache"):\n',
        f"{indent}        self.model_runner._cleanup_profiling_kv_cache()\n",
        f"{indent}    gc.collect()\n",
        f"{indent}    torch.cuda.synchronize(self.device)\n",
        f"{indent}    torch.cuda.empty_cache()\n",
        f"{indent}    profile_result.after_profile.measure()\n",
        f"{indent}    diff_from_create = (\n",
        f"{indent}        profile_result.after_profile - profile_result.before_create\n",
        f"{indent}    )\n",
        f"{indent}    profile_result.non_torch_increase = (\n",
        f"{indent}        diff_from_create.non_torch_memory\n",
        f"{indent}    )\n",
        f"{indent}    profile_result.non_kv_cache_memory = (\n",
        f"{indent}        profile_result.non_torch_increase\n",
        f"{indent}        + profile_result.torch_peak_increase\n",
        f"{indent}        + profile_result.weights_memory\n",
        f"{indent}    )\n",
        f"{indent}    cleanup_freed = (\n",
        f"{indent}        profile_result.after_profile.free_memory - before_cleanup\n",
        f"{indent}    )\n",
        f"{indent}    if cleanup_freed > 0:\n",
        f"{indent}        logger.info_once(\n",
        f'{indent}            "Freed %.2f GiB before KV cache sizing; "\n',
        f'{indent}            "non-torch profile increase is %.2f GiB.",\n',
        f"{indent}            cleanup_freed / (1024**3),\n",
        f"{indent}            profile_result.non_torch_increase / (1024**3),\n",
        f"{indent}        )\n",
        "\n",
    ]
    changed = True

if not prealloc_cleanup_present:
    func_index = None
    func_indent = None
    for i, line in enumerate(lines):
        match = re.match(
            r"^(?P<indent>[ \t]+)def initialize_from_config"
            r"\(self,\s*kv_cache_config\b",
            line,
        )
        if match:
            func_index = i
            func_indent = match.group("indent")
            break

    if func_index is None or func_indent is None:
        raise SystemExit("Could not find initialize_from_config in vLLM gpu_worker.py")

    body_indent = func_indent + "    "
    block = [
        f"{body_indent}# spark-vllm-docker: pre-KV cache allocator cleanup\n",
        f'{body_indent}if self.device_config.device_type == "cuda":\n',
        f"{body_indent}    gc.collect()\n",
        f"{body_indent}    torch.cuda.synchronize(self.device)\n",
        f"{body_indent}    cached_memory = max(\n",
        f"{body_indent}        torch.cuda.memory_reserved(self.device)\n",
        f"{body_indent}        - torch.cuda.memory_allocated(self.device),\n",
        f"{body_indent}        0,\n",
        f"{body_indent}    )\n",
        f"{body_indent}    torch.cuda.empty_cache()\n",
        f"{body_indent}    if cached_memory > 0:\n",
        f"{body_indent}        logger.info_once(\n",
        f'{body_indent}            "Cleared %.2f GiB of cached CUDA allocator memory before "\n',
        f'{body_indent}            "KV cache allocation.",\n',
        f"{body_indent}            cached_memory / (1024**3),\n",
        f"{body_indent}        )\n",
        "\n",
    ]
    insert_after_docstring(func_index, func_indent, block)
    changed = True

if changed:
    target.write_text("".join(lines))
    print("Applied Spark KV cache cleanup patch")
else:
    print("Equivalent Spark KV cache cleanup already present; skipping")
PY


# Prepare build requirements
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
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
     uv pip install torch==2.11.0 torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu130 && \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# Install wheels from host ./wheels/ (bind-mounted from build context — no layer bloat)
# PRE_TRANSFORMERS=1 is retained for manual legacy builds; build-and-copy.sh no longer sets it for --tf5.
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
