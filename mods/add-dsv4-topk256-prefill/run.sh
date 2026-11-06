#!/bin/bash
# Carry FlashInfer #3834: add the DSV4 TOPK=256 prefill instantiation for SM12x.
# Keep paired with mods/add-dsv4-topk256 (FlashInfer #3817).
set -euo pipefail

PREFILL_CU=/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/sparse_mla_sm120_prefill.cu
[ -f "$PREFILL_CU" ] || { echo "--- [dsv4-topk256-prefill] missing $PREFILL_CU; ABORT."; exit 1; }

python3 - "$PREFILL_CU" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
branch = "  else if (topk == 256)\n    DISPATCH_BY_NH_CM(BF16, 256);"
wrong_branch = "  else if (topk == 256)\n    DISPATCH_BY_NH_CM(FP8, 256);"
anchor = "  else if (topk == 512)\n    DISPATCH_BY_NH_CM(FP8, 512);"

if wrong_branch in text:
    print("--- [dsv4-topk256-prefill] found obsolete FP8 TOPK=256 carry; ABORT.")
    raise SystemExit(1)
if branch in text:
    print("--- [dsv4-topk256-prefill] BF16 TOPK=256 dispatch already present.")
elif anchor not in text:
    print("--- [dsv4-topk256-prefill] dispatch anchor not found; FlashInfer layout changed; ABORT.")
    raise SystemExit(1)
else:
    path.write_text(text.replace(anchor, branch + "\n" + anchor, 1))
    print("--- [dsv4-topk256-prefill] added BF16 TOPK=256 dispatch.")
PY

# The image's AOT module shadows patched package sources; force one JIT rebuild.
AOT=/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so
DISABLED=$AOT.disabled-for-topk256
if [ -f "$AOT" ]; then
  mv "$AOT" "$DISABLED"
  echo "--- [dsv4-topk256-prefill] disabled stale AOT module."
elif [ -f "$DISABLED" ]; then
  echo "--- [dsv4-topk256-prefill] stale AOT module already disabled."
else
  echo "--- [dsv4-topk256-prefill] no AOT module found; JIT will use patched sources."
fi

echo "=== OK"
