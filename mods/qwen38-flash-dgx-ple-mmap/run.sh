#!/usr/bin/env bash
# Applies blazux/qwen3.8-Flash-DGX's PLE-mmap patch at container start.
# https://github.com/blazux/qwen3.8-Flash-DGX
#
# Serves the ~51B-param n-gram (PLE) table from disk via mmap
# (VLLM_PLE_MMAP=1) instead of keeping it resident in the unified pool.
# No-op unless VLLM_PLE_MMAP=1 is set at runtime.
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP="/usr/local/lib/python3.12/dist-packages"
PLE="$SP/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"

if [ ! -f "$PLE" ]; then
  echo "!! $PLE not found — this mod requires the qwen38-flash-next image" >&2
  exit 1
fi

cp "$MOD_DIR/vllm_ple_mmap.py" "$SP/vllm_ple_mmap.py"

if ! grep -q "qwen38-flash-dgx: serve the PLE n-gram table from disk" "$PLE"; then
  cp "$PLE" "$PLE.orig"
  printf '\n\n# --- qwen38-flash-dgx: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)\n' >> "$PLE"
fi

python3 -c "import ast; ast.parse(open('$PLE').read()); print('ple_layer.py patched OK')"
