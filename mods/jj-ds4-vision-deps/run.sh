#!/bin/bash
# run.sh for mods/jj-ds4-vision-deps
#
# vLLM #634 (DeepSeek-V4-Flash-Vision-Exp on b12x) runtime dependency stack.
# Applied at container start (see launch-cluster.sh apply_mod_to_container):
#   - b12x  : overlay #301 (prefill.py/prefill_mg.py) + #246 (comm/pcie/*.py)
#             + #306 (mhc/_impl.py, rms_eps allowance). b12x is a pure
#             Python/JIT package: overlaying the .py files is sufficient,
#             no wheel rebuild needed. PRs:
#               https://github.com/local-inference-lab/b12x/pull/301
#               https://github.com/local-inference-lab/b12x/pull/246
#               https://github.com/local-inference-lab/b12x/pull/306
#   - LMCache : build+install dev+5 source (engine-driven hybrid KV) from the
#             vendored tree (lmcache-dev5-src.tar.gz): LMCache `dev`
#             @7ed46754 + follow-up PRs #49/#50/#51/#55/#56, which replaced
#             PR #44 in vLLM PR #634. No network needed at runtime. PRs:
#               https://github.com/local-inference-lab/LMCache/pull/49
#               https://github.com/local-inference-lab/LMCache/pull/50
#               https://github.com/local-inference-lab/LMCache/pull/51
#               https://github.com/local-inference-lab/LMCache/pull/55
#               https://github.com/local-inference-lab/LMCache/pull/56
#   - flashinfer : nothing — the baked 0.6.18 already contains 803c logic.
#
# Idempotent: safe to re-run.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
B12X_OVERLAY="$ROOT_DIR/b12x-overlay/b12x"
LMCACHE_TARGZ="$ROOT_DIR/lmcache-dev5-src.tar.gz"
LMCACHE_HEAD="14f4b01306af6c81705128c1ee87c1ee78ca9f81"  # dev @7ed46754 + PRs #49/#50/#51/#55/#56 (see PR_SOURCE in tarball)
LMCACHE_DEV="7ed4675404a31f4ffafd98975899dc83832ba965"  # dev branch base
LOG_PREFIX="[jj-ds4-vision-deps]"

log() { echo "$LOG_PREFIX $*"; }
die() { echo "$LOG_PREFIX ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1) b12x overlay (#301 prefill, #246 pcie, #306 mhc)
# ---------------------------------------------------------------------------
SITE="$(python3 -c 'import b12x, os; print(os.path.dirname(os.path.dirname(os.path.abspath(b12x.__file__))))' 2>/dev/null \
          || python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
B12X_PKG="$SITE/b12x"
log "b12x installed at: $B12X_PKG"
[ -d "$B12X_PKG" ] || die "b12x package dir not found under $SITE"

overlay_file() {
    local rel="$1"
    local src="$B12X_OVERLAY/$rel"
    local dst="$B12X_PKG/$rel"
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
    log "  overlaid $rel ($(wc -l < "$src") lines)"
}

overlay_file "attention/_shared/mla/prefill.py"
overlay_file "attention/_shared/mla/prefill_mg.py"
overlay_file "comm/pcie/_cute_intrinsics.py"
overlay_file "comm/pcie/_oneshot_cute.py"
overlay_file "comm/pcie/pcie_allreduce.py"
overlay_file "comm/pcie/pcie_oneshot.py"
overlay_file "norm/mhc/_impl.py"

# Verify the mhc rms_eps allowance actually contains 1e-20 (the #306 effect).
RUST_OES="$(python3 -c 'from b12x.norm.mhc import _impl as i; print(i.MHC_SUPPORTED_RMS_EPS)' 2>&1 || true)"
log "post-overlay MHC_SUPPORTED_RMS_EPS = $RUST_OES"
case "$RUST_OES" in
    *1e-20*|*1e-20,*) log "b12x rms_eps=1e-20 support OK" ;;
    *) die "b12x overlay did not take effect (MHC_SUPPORTED_RMS_EPS=$RUST_OES)" ;;
esac

# Sanity: verify the overlay files import cleanly (no missing symbols).
python3 - <<'PY' || die "b12x overlay import check failed"
import b12x.attention._shared.mla.prefill
import b12x.attention._shared.mla.prefill_mg
import b12x.comm.pcie._cute_intrinsics
import b12x.comm.pcie._oneshot_cute
import b12x.comm.pcie.pcie_allreduce
import b12x.comm.pcie.pcie_oneshot
import b12x.norm.mhc._impl
print("[jj-ds4-vision-deps] b12x overlay imports clean")
PY

# ---------------------------------------------------------------------------
# 2) LMCache — build & install vendored dev+5 source (engine-driven hybrid KV)
#    dev @7ed46754 + follow-up PRs #49/#50/#51/#55/#56 (replaces PR #44,
#    which vLLM PR #634 dropped in favor of these modern dev-lineage PRs)
# ---------------------------------------------------------------------------
if python3 -c 'import lmcache' 2>/dev/null; then
    v="$(python3 -c 'import lmcache; print(getattr(lmcache,"__version__","?"))' 2>/dev/null || echo '?')"
    # detect whether existing install already carries the dev+5 markers
    # (PagedKVTransferWorkspace is #50; engine_driven_shm_pool is #56)
    if python3 - <<'PY' 2>/dev/null
import lmcache, pathlib
p = pathlib.Path(lmcache.__file__).parent
assert (p/"v1"/"multiprocess"/"transfer_context"/"base.py").exists()
from lmcache.v1.multiprocess.transfer_context.base import PagedKVTransferWorkspace
from lmcache.v1.multiprocess.modules.engine_driven_transfer import EngineDrivenTransferModule
PY
    then
        log "LMCache already installed (version=$v) with dev+5 markers — skipping build."
        exit_or_skip=0
    else
        log "LMCache present but missing dev+5 markers — will rebuild from vendored source."
        exit_or_skip=1
    fi
else
    log "LMCache not installed — building from vendored dev+5 source."
    exit_or_skip=1
fi

if [ "${exit_or_skip:-0}" = "1" ]; then
    [ -f "$LMCACHE_TARGZ" ] || die "vendored LMCache tarball missing ($LMCACHE_TARGZ)"

    log "  installing setuptools_scm (needed by pyproject build-system when --no-build-isolation)"
    uv pip install --python /usr/bin/python3 setuptools_scm 2>&1 | tail -2 || die "could not install setuptools_scm"

    # The mod tree itself may be read-only (docker cp from a :ro context), and
    # setuptools writes a build/ dir inside the source root — so extract the
    # tarball to a writable dir in /tmp and build from there. This also keeps
    # the installed mod dir pristine.
    log "  extracting vendored source to a writable build dir"
    LMCACHE_BUILD_DIR="/tmp/jj-ds4-vision-deps-lmcache-build"
    rm -rf "$LMCACHE_BUILD_DIR"
    mkdir -p "$LMCACHE_BUILD_DIR"
    tar xzf "$LMCACHE_TARGZ" -C "$LMCACHE_BUILD_DIR"

    # Ensure the source HEAD matches the tarball (traceability guard).
    # PR_SOURCE records dev base + the 5 follow-up PR heads.
    if [ -f "$LMCACHE_BUILD_DIR/PR_SOURCE" ] && ! grep -q "$LMCACHE_HEAD" "$LMCACHE_BUILD_DIR/PR_SOURCE"; then
        log "Warning: vendored LMCache HEAD $(grep -oE '[0-9a-f]{40}' "$LMCACHE_BUILD_DIR/PR_SOURCE" | tail -1) != expected $LMCACHE_HEAD; continuing with vendored content."
    fi

    # setuptools_scm reads version from git; vendored tree is not a git repo,
    # so pretend a version matching the dev lineage.
    export SETUPTOOLS_SCM_PRETEND_VERSION="0.5.2.post0+jjdev5.$LMCACHE_DEV"

    log "  building with --no-build-isolation (uses image torch 2.13 / toolchain; pyproject's pinned torch==2.13.0 would re-resolve under build isolation)"
    # Build the wheel once, install it. --no-deps: every runtime dep the image
    # needs is already present (torch, cuda-python, ...). redis/cufile are
    # optional storage backends, intentionally not pulled.
    uv pip install --python /usr/bin/python3 --no-build-isolation --no-deps "$LMCACHE_BUILD_DIR" \
        || die "LMCache dev+5 build/install failed"
    rm -rf "$LMCACHE_BUILD_DIR"

    # Install the light-weight runtime dependencies LMCache imports at startup
    # (see requirements/common.txt) that are missing from this image. Intentionally
    # NOT installed (would collide / be heavy / downgrade): numpy (must keep 2.3.5),
    # numba, cupy, ray, xformers, awscrt, cufile-python, google-cloud-*, pytest.
    log "  installing missing light LMCache runtime deps"
    uv pip install --python /usr/bin/python3 \
        sortedcontainers aiofiles aiofile pyzmq pyyaml 2>&1 | tail -3 \
        || die "could not install LMCache light runtime deps"
fi

# Verify installed LMCache is the dev+5 source and importable.
python3 - <<'PY' || die "LMCache post-install verification failed"
import lmcache, pathlib
print("[jj-ds4-vision-deps] LMCache installed:", getattr(lmcache, "__version__", "?"))
p = pathlib.Path(lmcache.__file__).parent
for rel in (
    "v1/multiprocess/transfer_context/async_engine_driven.py",
    "v1/multiprocess/transfer_context/base.py",
    "v1/multiprocess/modules/engine_driven_transfer.py",
):
    if not (p / rel).exists():
        raise SystemExit(f"LMCache module missing: {rel} (not dev+5 source?)")
from lmcache.v1.multiprocess.transfer_context.base import (
    PagedKVTransferWorkspace,          # PR #50
)

# Verify the follow-up markers are present (#55 lands in the H2D scatter path).
import inspect as _inspect
from lmcache.v1.multiprocess.transfer_context.base import scatter_cpu_to_paged_kv
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
if "dynamically_pinned" not in _inspect.getsource(scatter_cpu_to_paged_kv):
    raise SystemExit("PR #55 async-copy lifetime marker missing")
if "engine_driven_shm_pool" not in _inspect.getsource(EngineDrivenTransferModule):
    raise SystemExit("PR #56 SHM transport report marker missing")
print("[jj-ds4-vision-deps] LMCache dev+5 markers present (#50/#55/#56)")
PY

# ---------------------------------------------------------------------------
# 3) flashinfer — no-op (baked 0.6.18 already has the 803c dispatch)
# ---------------------------------------------------------------------------
log "flashinfer: skipping (image 0.6.18 already contains SM120 topk-512 fallback)"

log "jj-ds4-vision-deps applied successfully"
