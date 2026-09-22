#!/bin/bash
# Run from the FlashInfer checkout before building its JIT-cache shim.
set -euo pipefail

# Older refs still build a monolithic JIT-cache wheel.
if [ ! -d flashinfer-jit-cache-provider ]; then
    exit 0
fi

: "${FLASHINFER_JIT_CACHE_PROVIDER_ARCHS:?Set the target CUDA architectures}"
build_python="${1:?Pass the prepared build Python}"
wheel_dir="${2:?Pass the absolute wheel output directory}"

cd flashinfer-jit-cache-provider
# Match the shim's whitespace-separated arch list. Disable filename expansion.
set -f
for arch in $FLASHINFER_JIT_CACHE_PROVIDER_ARCHS; do
    # Setuptools and the provider package reuse these directories. Keep each
    # wheel free of packages and compiled modules from the preceding target.
    rm -rf build flashinfer_jit_cache_provider/jit_cache
    FLASHINFER_JIT_CACHE_PROVIDER_ARCH="$arch" \
        uv build --python "$build_python" --no-build-isolation --wheel . \
        --out-dir="$wheel_dir" -v
done
