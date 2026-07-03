#!/bin/bash
# VALIDATION mod: add the (num_heads=32, topk=256) sparse-MLA decode-dsv4
# instantiation that the DeepSeek-V4 DSpark draft decode needs.
#
# The DSpark draft decodes with topk=256, which is absent from both the Python
# dispatch table (_DECODE_DSV4_DISPATCH: TOPK in {128,512,1024}) and the CUDA
# instantiation switch (DSV4_DISPATCH macros). The shape therefore falls through
# to the paged kernel, which hard-asserts num_tokens>64 and aborts.
#
# This patches BOTH sides so the JIT module rebuilds with the (32,256) kernel:
#   - Python: add (32,256) to _DECODE_DSV4_DISPATCH so the dispatcher routes to
#     the standalone decode kernel instead of the paged orchestrator.
#   - CUDA:   add DSV4_DISPATCH(32,256) so launch_decode_dsv4_impl<DSV4,32,256,64>
#     is instantiated; ninja recompiles the object on next launch (mtime bump).
#
# Validation only. The upstream PR adds the full TOPK=256 column. REMOVE after.
set -e
D=/usr/local/lib/python3.12/dist-packages/flashinfer
PY=$D/mla/_sparse_mla_sm120.py
CU=$D/data/csrc/sparse_mla_sm120_decode_dsv4.cu
for f in "$PY" "$CU"; do
  [ -f "$f" ] || { echo "--- [dsv4-topk256] missing $f; skipping."; exit 0; }
done
python3 - "$PY" "$CU" <<'PY'
import sys
py, cu = sys.argv[1], sys.argv[2]

# 1) Python dispatch table: add (32, 256) to _DECODE_DSV4_DISPATCH (DSV4 block
#    comes before DSV3_2 in the file, so replace(count=1) targets the DSV4 set).
s = open(py).read()
anchor = "_DECODE_DSV4_DISPATCH = frozenset(\n    {\n"
if anchor not in s:
    print("--- [dsv4-topk256] python anchor not found; ABORT (layout changed).")
    sys.exit(1)
dsv4_region = s.split("_DECODE_DSV3_2_DISPATCH")[0]
if "(32, 256)" in dsv4_region:
    print("--- [dsv4-topk256] python already has (32,256); skipping python.")
else:
    s = s.replace(anchor, anchor + "        (32, 256),\n", 1)
    open(py, "w").write(s)
    print("--- [dsv4-topk256] python patched: (32,256) added to dispatch table.")

# 2) CUDA instantiation: add DSV4_DISPATCH(32, 256) before #undef DSV4_DISPATCH.
c = open(cu).read()
if "DSV4_DISPATCH(32, 256)" in c:
    print("--- [dsv4-topk256] cuda already has DSV4_DISPATCH(32,256); skipping cuda.")
else:
    marker = "#undef DSV4_DISPATCH"
    if marker not in c:
        print("--- [dsv4-topk256] cuda marker not found; ABORT (layout changed).")
        sys.exit(1)
    c = c.replace(marker, "  DSV4_DISPATCH(32, 256)\n" + marker, 1)
    open(cu, "w").write(c)
    print("--- [dsv4-topk256] cuda patched: DSV4_DISPATCH(32,256) added.")
PY

# 3) Force JIT rebuild: the image ships an AOT-prebuilt sparse_mla_sm120.so.
#    JitSpec.build_and_load() short-circuits to that .so whenever it exists
#    (is_aot == aot_path.exists()), so our data/csrc patch is otherwise ignored.
#    Rename it so is_aot becomes False and the module JIT-compiles from source
#    (picking up DSV4_DISPATCH(32,256)). Container-layer only; gone on recreate.
AOT=/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so
if [ -f "$AOT" ]; then
  mv "$AOT" "$AOT.disabled-for-topk256"
  echo "--- [dsv4-topk256] AOT prebuilt .so disabled; module will JIT-rebuild from patched source."
elif [ -f "$AOT.disabled-for-topk256" ]; then
  echo "--- [dsv4-topk256] AOT prebuilt .so already disabled; skipping."
else
  echo "--- [dsv4-topk256] AOT prebuilt .so not found (already JIT?); continuing."
fi
echo "=== OK"
