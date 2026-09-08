#!/bin/bash
# verify.sh — standalone verification for mods/jj-ds4-vision-deps
#
# Checks, against the *installed* environment, that the mod's three claims hold:
#   1. b12x overlay applied (MHC_SUPPORTED_RMS_EPS contains 1e-20)
#   2. b12x overlay modules import clean
#   3. LMCache is the dev+5 source (engine-driven path + follow-up markers)
#
# Usage:
#   apt/container:  bash /path/to/mods/jj-ds4-vision-deps/verify.sh
# Exits 0 on all-pass, 1 otherwise.
set -u

fail=0
ok() { echo "  PASS: $*"; }
bad() { echo "  FAIL: $*"; fail=1; }

echo "== verify 1: b12x rms_eps (PR #306) =="
RUST_OES="$(python3 -c 'from b12x.norm.mhc import _impl as i; print(i.MHC_SUPPORTED_RMS_EPS)' 2>&1 || true)"
case "$RUST_OES" in
    *1e-20*) ok "MHC_SUPPORTED_RMS_EPS=$RUST_OES contains 1e-20" ;;
    *) bad "MHC_SUPPORTED_RMS_EPS=$RUST_OES missing 1e-20 (overlay not applied?)" ;;
esac

echo "== verify 2: b12x overlay modules import =="
if python3 - <<'PY' 2>&1
import b12x.attention._shared.mla.prefill
import b12x.attention._shared.mla.prefill_mg
import b12x.comm.pcie._cute_intrinsics
import b12x.comm.pcie._oneshot_cute
import b12x.comm.pcie.pcie_allreduce
import b12x.comm.pcie.pcie_oneshot
import b12x.norm.mhc._impl
PY
then
    ok "all 7 overlay modules import clean"
else
    bad "overlay module import failed"
fi

echo "== verify 3: LMCache is dev+5 source (replaces PR #44) =="
if python3 - <<'PY' 2>&1
import lmcache, pathlib, inspect
p = pathlib.Path(lmcache.__file__).parent
for rel in (
    "v1/multiprocess/transfer_context/async_engine_driven.py",
    "v1/multiprocess/transfer_context/base.py",
    "v1/multiprocess/modules/engine_driven_transfer.py",
):
    assert (p / rel).exists(), f"module missing: {rel}"
from lmcache.v1.multiprocess.transfer_context.base import PagedKVTransferWorkspace  # PR #50
from lmcache.v1.multiprocess.transfer_context.base import scatter_cpu_to_paged_kv
from lmcache.v1.multiprocess.modules.engine_driven_transfer import EngineDrivenTransferModule
assert "dynamically_pinned" in inspect.getsource(scatter_cpu_to_paged_kv), "PR #55 marker missing"
assert "engine_driven_shm_pool" in inspect.getsource(EngineDrivenTransferModule), "PR #56 marker missing"
print("lmcache", getattr(lmcache, "__version__", "?"), "@", lmcache.__file__)
PY
then
    ok "LMCache dev+5 source present (PagedKVTransferWorkspace + #55/#56 markers)"
else
    bad "LMCache dev+5 markers missing — not the dev+5 source?"
fi

echo
if [ "$fail" = "0" ]; then
    echo "verify: ALL PASS"
    exit 0
else
    echo "verify: FAILURES DETECTED"
    exit 1
fi
