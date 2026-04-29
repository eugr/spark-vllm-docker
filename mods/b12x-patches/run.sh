#!/bin/bash
set -e

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"

echo "=== b12x-patches mod ==="

# ---------------------------------------------------------------
# 1. Pin nvidia-cutlass-dsl + companion libs to 4.4.2
#    (4.5.x generates bad PTX on SM121 — `_mma` rejected by ptxas).
#    All THREE packages must match: the python frontend, the base libs,
#    and the CUDA 13 libs (which contain the MLIR compiler).
# ---------------------------------------------------------------
DSL_VER=$(pip show nvidia-cutlass-dsl 2>/dev/null | grep '^Version:' | awk '{print $2}' || true)
LIBS_BASE_VER=$(pip show nvidia-cutlass-dsl-libs-base 2>/dev/null | grep '^Version:' | awk '{print $2}' || true)
LIBS_CU13_VER=$(pip show nvidia-cutlass-dsl-libs-cu13 2>/dev/null | grep '^Version:' | awk '{print $2}' || true)
if [ "$DSL_VER" != "4.4.2" ] || [ "$LIBS_BASE_VER" != "4.4.2" ] || [ "$LIBS_CU13_VER" != "4.4.2" ]; then
    echo "[b12x] Pinning nvidia-cutlass-dsl{,-libs-base,-libs-cu13} to 4.4.2"
    echo "[b12x]   current: dsl=${DSL_VER:-none} libs-base=${LIBS_BASE_VER:-none} libs-cu13=${LIBS_CU13_VER:-none}"
    pip install \
        nvidia-cutlass-dsl==4.4.2 \
        nvidia-cutlass-dsl-libs-base==4.4.2 \
        nvidia-cutlass-dsl-libs-cu13==4.4.2 \
        -q 2>/dev/null || echo "[b12x] WARNING: cutlass-dsl pin install returned non-zero"
else
    echo "[b12x] nvidia-cutlass-dsl + libs already at 4.4.2"
fi

# ---------------------------------------------------------------
# 2. Apply cutlass-dsl SM121 patches
#    FlashInfer/vLLM install wipes vendored cutlass, so re-apply every time
# ---------------------------------------------------------------
echo "[b12x] Applying cutlass-dsl SM121 patches..."

# 2a. warp/mma.py: allow sm_121a alongside sm_120a in both the runtime
#     arch check and the `admissible_archs` string list (used in error msgs)
for f in $(find "$SITE_PACKAGES" -name "mma.py" -path "*/warp/*" 2>/dev/null); do
    if grep -q "if not arch == Arch.sm_120a:" "$f" 2>/dev/null; then
        sed -i "s/if not arch == Arch.sm_120a:/if arch not in (Arch.sm_120a, Arch.sm_121a):/" "$f"
        echo "  patched $f (warp sm_121a runtime check)"
    fi
    # Add sm_121a to the admissible_archs list if missing
    if grep -q '"sm_120a",' "$f" 2>/dev/null && ! grep -q '"sm_121a"' "$f" 2>/dev/null; then
        sed -i 's/^\(\s*\)"sm_120a",$/\1"sm_120a",\n\1"sm_121a",/' "$f"
        echo "  patched $f (warp sm_121a admissible_archs)"
    fi
done

# 2b. tcgen05/mma.py: add sm_120a and sm_121a to supported arch list
for f in $(find "$SITE_PACKAGES" -name "mma.py" -path "*/tcgen05/*" 2>/dev/null); do
    if ! grep -q "Arch.sm_121a" "$f" 2>/dev/null; then
        sed -i "/Arch.sm_103a,/a\\        Arch.sm_120a,\n        Arch.sm_121a," "$f"
        echo "  patched $f (tcgen05 mma sm_121a)"
    fi
done

# 2c. tcgen05/copy.py: allow sm_120f family
for f in $(find "$SITE_PACKAGES" -name "copy.py" -path "*/tcgen05/*" 2>/dev/null); do
    if ! grep -q "sm_120f" "$f" 2>/dev/null; then
        sed -i "s/arch.is_family_of(Arch.sm_110f)/arch.is_family_of(Arch.sm_110f) or arch.is_family_of(Arch.sm_120f)/" "$f"
        echo "  patched $f (tcgen05 copy sm_120f)"
    fi
done

# Clear pycache so patched code takes effect
find "$SITE_PACKAGES" -name "__pycache__" -path "*/cutlass*" -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -name "__pycache__" -path "*/flashinfer*" -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------
# 3 Patch FlashInfer's blackwell_sm12x __init__.py to drop the
#      broken `sm120_moe_dispatch_context` import (FlashInfer main
#      has a stale __init__ that references a function that no
#      longer exists in moe_dispatch.py — but the symbol isn't
#      actually used by anything, so we just remove it from the
#      import + __all__ list).
# ---------------------------------------------------------------
SM12X_INIT="$SITE_PACKAGES/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/__init__.py"
if [ -f "$SM12X_INIT" ]; then
    if grep -q "sm120_moe_dispatch_context" "$SM12X_INIT"; then
        # Drop the line that imports/exports the missing symbol
        sed -i '/sm120_moe_dispatch_context/d' "$SM12X_INIT"
        find "$SITE_PACKAGES/flashinfer" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        echo "[b12x] patched $SM12X_INIT (dropped stale sm120_moe_dispatch_context references)"
    else
        echo "[b12x] $SM12X_INIT already cleaned"
    fi
else
    echo "[b12x] $SM12X_INIT not found (older FlashInfer?), skipping"
fi

# ---------------------------------------------------------------
# 6. Export required environment variables
# ---------------------------------------------------------------
echo "[b12x] Setting environment variables..."

export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_NVFP4_GEMM_BACKEND=flashinfer-b12x
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export VLLM_USE_FLASHINFER_MOE_FP4=1
