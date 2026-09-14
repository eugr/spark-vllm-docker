#!/usr/bin/env bash
# DSpark sparse-MLA sm120 decode fix: overlay the 4 interdependent files that
# route small spec-decode batches through the sm120 decode kernel instead of
# the >64 paged path. Standalone; compose after a router-patch mod.
set -euo pipefail
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
for rel in \
  models/deepseek_v4/nvidia/flashinfer_sparse.py \
  models/deepseek_v4/attention.py \
  models/deepseek_v4/sparse_mla.py \
  v1/attention/backends/mla/sparse_swa.py ; do
  cp "$MOD_DIR/dspark_overlay/vllm/$rel" "$VLLM/$rel"
  python3 -m py_compile "$VLLM/$rel"
  echo "OK overlay:$rel"
done
echo "SPARSE_MLA_SM120_OVERLAY_OK"
