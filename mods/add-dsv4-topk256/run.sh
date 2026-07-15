#!/bin/bash
# Carry FlashInfer #3817: add DSV4 TOPK=256 decode instantiations for SM12x.
# Keep paired with mods/add-dsv4-topk256-prefill (FlashInfer #3834).
set -euo pipefail

D=/usr/local/lib/python3.12/dist-packages/flashinfer
PY_FILE=$D/mla/_sparse_mla_sm120.py
DECODE_CU=$D/data/csrc/sparse_mla_sm120_decode_dsv4.cu

for f in "$PY_FILE" "$DECODE_CU"; do
  [ -f "$f" ] || { echo "--- [dsv4-topk256-decode] missing $f; ABORT."; exit 1; }
done

python3 - "$PY_FILE" "$DECODE_CU" <<'PY'
import sys
from pathlib import Path

py_path, decode_path = map(Path, sys.argv[1:])
head_sizes = (8, 16, 32, 64, 128)


def fail(message: str) -> None:
    print(f"--- [dsv4-topk256-decode] {message}")
    raise SystemExit(1)


py_text = py_path.read_text()
dispatch_anchor = "_DECODE_DSV4_DISPATCH = frozenset(\n    {\n"
if dispatch_anchor not in py_text:
    fail("Python dispatch anchor not found; FlashInfer layout changed.")

dsv4_region = py_text.split("_DECODE_DSV3_2_DISPATCH", 1)[0]
missing_python = [h for h in head_sizes if f"({h}, 256)" not in dsv4_region]
if missing_python:
    entries = "".join(f"        ({h}, 256),\n" for h in missing_python)
    py_text = py_text.replace(dispatch_anchor, dispatch_anchor + entries, 1)
    print("--- [dsv4-topk256-decode] added Python dispatch entries: " + ", ".join(map(str, missing_python)))
else:
    print("--- [dsv4-topk256-decode] Python dispatch already patched.")

guard_marker = "SM120 sparse-MLA has no decode kernel for this shape:"
if guard_marker not in py_text:
    paged_call = "        module.sparse_mla_sm120_paged_attention(\n"
    if paged_call not in py_text:
        fail("Paged-attention call anchor not found; FlashInfer layout changed.")
    guard = '''        # Decode inputs must use a standalone decode instantiation. The paged
        # orchestrator only supports prefill and otherwise aborts in C++.
        if num_tokens <= _DECODE_MAX_TOKENS:
            raise ValueError(
                "SM120 sparse-MLA has no decode kernel for this shape: "
                f"num_tokens={num_tokens}, num_heads={num_heads}, topk={topk}, "
                f"d_qk={d_qk}, page_block_size={kv_pbs}, model_type={model_type}, "
                f"extra_topk={extra_topk}. Supported decode shapes are enumerated in "
                "_DECODE_DSV4_DISPATCH / _DECODE_DSV3_2_DISPATCH; add the matching "
                "(num_heads, topk) instantiation to support it."
            )

'''
    py_text = py_text.replace(paged_call, guard + paged_call, 1)
    print("--- [dsv4-topk256-decode] added unsupported-shape guard.")
else:
    print("--- [dsv4-topk256-decode] unsupported-shape guard already present.")
py_path.write_text(py_text)

decode_text = decode_path.read_text()
decode_marker = "#undef DSV4_DISPATCH"
if decode_marker not in decode_text:
    fail("CUDA decode marker not found; FlashInfer layout changed.")
missing_cuda = [h for h in head_sizes if f"DSV4_DISPATCH({h}, 256)" not in decode_text]
if missing_cuda:
    entries = "".join(f"  DSV4_DISPATCH({h}, 256)\n" for h in missing_cuda)
    decode_text = decode_text.replace(decode_marker, entries + decode_marker, 1)
    decode_path.write_text(decode_text)
    print("--- [dsv4-topk256-decode] added CUDA instantiations: " + ", ".join(map(str, missing_cuda)))
else:
    print("--- [dsv4-topk256-decode] CUDA decode already patched.")
PY

# The image's AOT module shadows patched package sources; force one JIT rebuild.
AOT=/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so
DISABLED=$AOT.disabled-for-topk256
if [ -f "$AOT" ]; then
  mv "$AOT" "$DISABLED"
  echo "--- [dsv4-topk256-decode] disabled stale AOT module."
elif [ -f "$DISABLED" ]; then
  echo "--- [dsv4-topk256-decode] stale AOT module already disabled."
else
  echo "--- [dsv4-topk256-decode] no AOT module found; JIT will use patched sources."
fi

echo "=== OK"
